#!/usr/bin/env python3
"""
Diagnostica: Quanto vale global_coherence nei vari benchmark?
Se global_coherence < 0.5, il PF è completamente disattivato!
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# YAHPO WRAPPER (semplificato)
# ============================================================================

class YAHPOWrapper:
    def __init__(self, scenario: str, task_id=None):
        from yahpo_gym import benchmark_set
        self.bench = benchmark_set.BenchmarkSet(scenario)
        if task_id is None:
            task_id = self.bench.instances[0]
        self.task_id = task_id
        self.bench.set_instance(str(task_id))
        self._build_param_space()
    
    def _build_param_space(self):
        cs = self.bench.get_opt_space()
        self.param_space = {}
        for hp in cs.get_hyperparameters():
            name = hp.name
            hp_type = type(hp).__name__
            if name == 'task_id':
                continue
            if hp_type == 'UniformFloatHyperparameter':
                if hp.log:
                    self.param_space[name] = (hp.lower, hp.upper, 'log')
                else:
                    self.param_space[name] = (hp.lower, hp.upper)
            elif hp_type == 'UniformIntegerHyperparameter':
                self.param_space[name] = (hp.lower, hp.upper, 'int')
            elif hp_type == 'CategoricalHyperparameter':
                self.param_space[name] = list(hp.choices)
            elif hp_type == 'Constant':
                continue
    
    def evaluate(self, config: Dict) -> float:
        eval_cfg = dict(config)
        eval_cfg['task_id'] = str(self.task_id)
        if 'trainsize' not in eval_cfg:
            eval_cfg['trainsize'] = 1.0
        result = self.bench.objective_function(eval_cfg)
        return float(result[0].get('mmce', 0.5))


# ============================================================================
# DIAGNOSTIC RUN
# ============================================================================

def run_diagnostic(wrapper, n_trials: int, seed: int) -> Dict:
    """Run e raccogli global_coherence nel tempo."""
    
    opt = ALBA(
        param_space=wrapper.param_space,
        seed=seed,
        maximize=False,
        total_budget=n_trials,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    coherence_history = []
    n_leaves_history = []
    
    for i in range(n_trials):
        cfg = opt.ask()
        try:
            y = wrapper.evaluate(cfg)
        except:
            y = 1.0
        opt.tell(cfg, y)
        
        # Raccogli coherence ogni 20 iterazioni
        if i % 20 == 0 and opt._coherence_tracker is not None:
            leaves = opt._root.leaves() if hasattr(opt, '_root') else []
            if len(leaves) >= 5:
                opt._coherence_tracker.update(leaves, i, force=True)
                coh = opt._coherence_tracker.global_coherence
                coherence_history.append((i, coh, len(leaves)))
                n_leaves_history.append(len(leaves))
    
    return {
        'coherence_history': coherence_history,
        'final_coherence': coherence_history[-1][1] if coherence_history else 0.5,
    }


def main():
    print("=" * 75)
    print("  DIAGNOSTICA: Valori di global_coherence nei vari benchmark")
    print("  Threshold per PF: coherence > 0.5 (poi scala fino a 0.8)")
    print("=" * 75)
    
    BUDGET = 200
    
    scenarios = [
        ('iaml_glmnet', '40981'),
        ('iaml_rpart', '40981'),
        ('iaml_xgboost', '40981'),
        ('iaml_ranger', '40981'),
    ]
    
    for scenario, task_id in scenarios:
        print(f"\n{'='*60}")
        print(f"  {scenario} (task={task_id})")
        print(f"{'='*60}")
        
        try:
            wrapper = YAHPOWrapper(scenario, task_id)
            result = run_diagnostic(wrapper, BUDGET, seed=42)
            
            print(f"\n  Evoluzione global_coherence:")
            print(f"  {'Iter':<8} {'Coherence':<12} {'Leaves':<8} {'PF Scale':<10}")
            print("  " + "-" * 45)
            
            for iter_i, coh, n_leaves in result['coherence_history']:
                pf_scale = max(0.0, min(1.0, (coh - 0.5) * 3.33))
                status = "🔥 ON" if pf_scale > 0.5 else ("⚡ partial" if pf_scale > 0 else "❌ OFF")
                print(f"  {iter_i:<8} {coh:<12.4f} {n_leaves:<8} {pf_scale:<6.2f} {status}")
            
            final_coh = result['final_coherence']
            final_scale = max(0.0, min(1.0, (final_coh - 0.5) * 3.33))
            print(f"\n  FINALE: coherence={final_coh:.4f}, PF scale={final_scale:.2f}")
            
            if final_scale < 0.1:
                print("  ⚠️ PF è quasi completamente DISATTIVATO!")
            elif final_scale < 0.5:
                print("  ⚡ PF è parzialmente attivo")
            else:
                print("  ✅ PF è attivo")
                
        except Exception as e:
            print(f"  Errore: {e}")
    
    # Test anche su funzioni sintetiche
    print(f"\n{'='*75}")
    print("  TEST SU FUNZIONI SINTETICHE (per confronto)")
    print(f"{'='*75}")
    
    # Sphere smooth
    print("\n  SPHERE (smooth, 8D):")
    bounds = [(-5.0, 5.0)] * 8
    
    opt = ALBA(
        bounds=bounds,
        seed=42,
        maximize=False,
        total_budget=200,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    coherence_history = []
    for i in range(200):
        x = opt.ask()
        if isinstance(x, dict):
            x_arr = np.array(list(x.values()))
        else:
            x_arr = np.array(x)
        y = float(np.sum(x_arr**2))
        opt.tell(x, y)
        
        if i % 40 == 0 and opt._coherence_tracker is not None:
            leaves = opt._root.leaves() if hasattr(opt, '_root') else []
            if len(leaves) >= 5:
                opt._coherence_tracker.update(leaves, i, force=True)
                coh = opt._coherence_tracker.global_coherence
                pf_scale = max(0.0, min(1.0, (coh - 0.5) * 3.33))
                print(f"    Iter {i}: coherence={coh:.4f}, PF scale={pf_scale:.2f}")
    
    print("\n" + "=" * 75)
    print("  CONCLUSIONE: Se coherence < 0.5, il PF è automaticamente OFF")
    print("=" * 75)


if __name__ == "__main__":
    main()
