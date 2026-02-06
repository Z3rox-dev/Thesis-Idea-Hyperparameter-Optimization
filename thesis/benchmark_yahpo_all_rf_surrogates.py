#!/usr/bin/env python3
"""
Benchmark: ALBA COV-only vs ALBA COV+PotentialField su TUTTI i surrogati YAHPO RF

YAHPO usa Random Forest per TUTTI i suoi surrogati.
Questo benchmark testa se PF aiuta su diversi tipi di spazi:
- Puro continuo: iaml_glmnet (2 cont), iaml_rpart (3 cont)
- Quasi continuo: iaml_xgboost (12 cont + 1 cat), iaml_ranger (7 cont + 1 cat)
- Misto: rbv2_ranger (5 cont + 3 cat), rbv2_svm (4 cont + 2 cat)
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# YAHPO WRAPPER GENERICO
# ============================================================================

class YAHPOWrapper:
    """Wrapper generico per qualsiasi scenario YAHPO."""
    
    def __init__(self, scenario: str, task_id: Optional[int] = None):
        from yahpo_gym import benchmark_set
        
        self.scenario = scenario
        self.bench = benchmark_set.BenchmarkSet(scenario)
        
        # Usa primo task se non specificato
        if task_id is None:
            task_id = self.bench.instances[0]
        
        self.task_id = task_id
        self.bench.set_instance(str(task_id))
        
        # Analizza config space
        self._analyze_config_space()
    
    def _analyze_config_space(self):
        """Estrae bounds e categoriche dal config space."""
        cs = self.bench.get_opt_space()
        
        self.param_names = []
        self.bounds = []
        self.categorical_dims = []
        self.param_space = {}
        
        for i, hp in enumerate(cs.get_hyperparameters()):
            name = hp.name
            hp_type = type(hp).__name__
            
            # Skip task_id - lo gestiamo separatamente
            if name == 'task_id':
                continue
            
            self.param_names.append(name)
            
            if hp_type == 'UniformFloatHyperparameter':
                if hp.log:
                    self.param_space[name] = (hp.lower, hp.upper, 'log')
                else:
                    self.param_space[name] = (hp.lower, hp.upper)
                self.bounds.append((hp.lower, hp.upper))
                
            elif hp_type == 'UniformIntegerHyperparameter':
                self.param_space[name] = (hp.lower, hp.upper, 'int')
                self.bounds.append((float(hp.lower), float(hp.upper)))
                
            elif hp_type == 'Constant':
                # Costante - non ottimizzare
                continue
                
            elif hp_type in ('CategoricalHyperparameter', 'OrdinalHyperparameter'):
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                self.param_space[name] = choices
                idx = len(self.param_names) - 1  # Current index
                self.categorical_dims.append((idx, len(choices)))
                self.bounds.append((0.0, float(len(choices) - 1)))
            else:
                # Fallback
                self.param_space[name] = (0.0, 1.0)
                self.bounds.append((0.0, 1.0))
        
        self.dim = len(self.param_names)
        self.n_continuous = self.dim - len(self.categorical_dims)
        self.n_categorical = len(self.categorical_dims)
    
    def evaluate(self, config: Dict[str, Any]) -> float:
        """Valuta config e ritorna errore (lower is better)."""
        # Aggiungi task_id se mancante
        eval_config = dict(config)
        if 'task_id' not in eval_config:
            eval_config['task_id'] = str(self.task_id)
        # Aggiungi trainsize se mancante (usa default 1.0 = full training set)
        if 'trainsize' not in eval_config and 'trainsize' not in self.param_names:
            eval_config['trainsize'] = 1.0
            
        result = self.bench.objective_function(eval_config)
        
        # Trova la metrica principale (mmce = misclassification error)
        if 'mmce' in result[0]:
            return float(result[0]['mmce'])
        elif 'acc' in result[0]:
            return 1.0 - float(result[0]['acc'])
        elif 'auc' in result[0]:
            return 1.0 - float(result[0]['auc'])
        else:
            key = list(result[0].keys())[0]
            return float(result[0][key])
    
    def get_info(self) -> str:
        """Info string."""
        return f"{self.scenario} (task={self.task_id}): {self.n_continuous} cont + {self.n_categorical} cat"


# ============================================================================
# BENCHMARK
# ============================================================================

def run_alba(
    wrapper: YAHPOWrapper,
    n_trials: int,
    seed: int,
    use_potential_field: bool,
) -> float:
    """Run ALBA on YAHPO benchmark, return best error."""
    
    opt = ALBA(
        param_space=wrapper.param_space,
        seed=seed,
        maximize=False,
        total_budget=n_trials,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    best_y = np.inf
    for _ in range(n_trials):
        cfg = opt.ask()
        try:
            y = wrapper.evaluate(cfg)
        except Exception:
            y = 1.0  # Worst error
        opt.tell(cfg, y)
        
        if y < best_y:
            best_y = y
    
    return best_y


def test_scenario(scenario: str, budget: int, n_seeds: int, n_tasks: int = 3) -> Dict:
    """Test un scenario YAHPO."""
    
    from yahpo_gym import benchmark_set
    
    try:
        bench = benchmark_set.BenchmarkSet(scenario)
        all_tasks = bench.instances[:n_tasks]  # Primi n task
    except Exception as e:
        print(f"  ⚠️ Scenario {scenario} non disponibile: {e}")
        return None
    
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario}")
    print(f"{'='*70}")
    
    # Crea wrapper per info
    try:
        info_wrapper = YAHPOWrapper(scenario, all_tasks[0])
        print(f"  Config: {info_wrapper.n_continuous} continuous + {info_wrapper.n_categorical} categorical")
    except Exception as e:
        print(f"  ⚠️ Errore wrapper: {e}")
        return None
    
    results_cov = []
    results_pf = []
    
    for task_id in all_tasks:
        try:
            wrapper = YAHPOWrapper(scenario, task_id)
        except Exception as e:
            print(f"  ⚠️ Task {task_id} fallito: {e}")
            continue
        
        print(f"\n  Task {task_id}:")
        
        for seed in range(n_seeds):
            print(f"    Seed {seed}: ", end="", flush=True)
            
            # COV-only
            val_cov = run_alba(wrapper, budget, 100 + seed, use_potential_field=False)
            results_cov.append(val_cov)
            print(f"COV={val_cov:.4f}", end=" ", flush=True)
            
            # COV+PF
            val_pf = run_alba(wrapper, budget, 100 + seed, use_potential_field=True)
            results_pf.append(val_pf)
            print(f"PF={val_pf:.4f}", end="", flush=True)
            
            if val_cov < val_pf:
                print(" → COV")
            elif val_pf < val_cov:
                print(" → PF")
            else:
                print(" → TIE")
    
    if not results_cov:
        return None
    
    # Stats
    mean_cov = np.mean(results_cov)
    mean_pf = np.mean(results_pf)
    
    wins_cov = sum(1 for a, b in zip(results_cov, results_pf) if a < b)
    wins_pf = sum(1 for a, b in zip(results_cov, results_pf) if b < a)
    ties = sum(1 for a, b in zip(results_cov, results_pf) if a == b)
    
    delta_pct = (mean_cov - mean_pf) / abs(mean_cov) * 100 if mean_cov != 0 else 0
    
    return {
        'scenario': scenario,
        'n_continuous': info_wrapper.n_continuous,
        'n_categorical': info_wrapper.n_categorical,
        'mean_cov': mean_cov,
        'mean_pf': mean_pf,
        'wins_cov': wins_cov,
        'wins_pf': wins_pf,
        'ties': ties,
        'total': len(results_cov),
        'delta_pct': delta_pct,
    }


def main():
    print("=" * 75)
    print("  BENCHMARK: ALBA COV-only vs COV+PF su TUTTI i surrogati YAHPO (RF)")
    print("=" * 75)
    
    try:
        from yahpo_gym import benchmark_set
        print("✓ yahpo_gym importato")
    except ImportError:
        print("ERROR: yahpo_gym non trovato")
        return
    
    BUDGET = 200
    N_SEEDS = 5
    N_TASKS = 3
    
    # Tutti gli scenari interessanti
    scenarios = [
        # PURO CONTINUO (ideale per PF)
        'iaml_glmnet',   # 2 cont + 0 cat
        'iaml_rpart',    # 3 cont + 0 cat
        
        # QUASI CONTINUO
        'iaml_xgboost',  # 12 cont + 1 cat
        'iaml_ranger',   # 7 cont + 1 cat
        
        # MISTO
        'rbv2_ranger',   # 5 cont + 3 cat
        'rbv2_svm',      # 4 cont + 2 cat
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        result = test_scenario(scenario, BUDGET, N_SEEDS, N_TASKS)
        if result is not None:
            all_results[scenario] = result
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 85)
    print("  FINAL SUMMARY - YAHPO RF Surrogates")
    print("=" * 85)
    
    print(f"\n{'Scenario':<15} {'Dims':<12} {'COV err':<10} {'PF err':<10} {'Δ%':<8} {'COV wins':<10} {'PF wins':<10}")
    print("-" * 85)
    
    total_wins_cov = 0
    total_wins_pf = 0
    total_runs = 0
    
    # Ordina per rapporto continuo/categoriale
    sorted_results = sorted(all_results.values(), 
                           key=lambda x: x['n_categorical'] / max(1, x['n_continuous']))
    
    for r in sorted_results:
        dims = f"{r['n_continuous']}c+{r['n_categorical']}cat"
        status = "✅" if r['wins_pf'] > r['wins_cov'] else ("❌" if r['wins_cov'] > r['wins_pf'] else "➖")
        
        print(f"{r['scenario']:<15} {dims:<12} {r['mean_cov']:<10.4f} {r['mean_pf']:<10.4f} "
              f"{r['delta_pct']:>+7.1f}% {r['wins_cov']:>5}/{r['total']} {r['wins_pf']:>5}/{r['total']} {status}")
        
        total_wins_cov += r['wins_cov']
        total_wins_pf += r['wins_pf']
        total_runs += r['total']
    
    print("-" * 85)
    avg_delta = np.mean([r['delta_pct'] for r in all_results.values()])
    print(f"{'TOTALE':<15} {'':<12} {'':<10} {'':<10} {avg_delta:>+7.1f}% "
          f"{total_wins_cov:>5}/{total_runs} {total_wins_pf:>5}/{total_runs}")
    
    # Analisi per tipo di spazio
    print("\n" + "=" * 85)
    print("  ANALISI PER TIPO DI SPAZIO")
    print("=" * 85)
    
    # Puro continuo
    pure_cont = [r for r in all_results.values() if r['n_categorical'] == 0]
    if pure_cont:
        wins_pf = sum(r['wins_pf'] for r in pure_cont)
        wins_cov = sum(r['wins_cov'] for r in pure_cont)
        total = sum(r['total'] for r in pure_cont)
        print(f"\n  PURO CONTINUO (0 cat):  PF wins {wins_pf}/{total}, COV wins {wins_cov}/{total}")
    
    # Quasi continuo (1 cat)
    quasi_cont = [r for r in all_results.values() if r['n_categorical'] == 1]
    if quasi_cont:
        wins_pf = sum(r['wins_pf'] for r in quasi_cont)
        wins_cov = sum(r['wins_cov'] for r in quasi_cont)
        total = sum(r['total'] for r in quasi_cont)
        print(f"  QUASI CONTINUO (1 cat): PF wins {wins_pf}/{total}, COV wins {wins_cov}/{total}")
    
    # Misto (2+ cat)
    mixed = [r for r in all_results.values() if r['n_categorical'] >= 2]
    if mixed:
        wins_pf = sum(r['wins_pf'] for r in mixed)
        wins_cov = sum(r['wins_cov'] for r in mixed)
        total = sum(r['total'] for r in mixed)
        print(f"  MISTO (2+ cat):         PF wins {wins_pf}/{total}, COV wins {wins_cov}/{total}")
    
    print("\n" + "=" * 85)
    if total_wins_pf > total_wins_cov:
        print(f"🔥 VERDETTO: PF è MIGLIORE su RF surrogates (vince {total_wins_pf}/{total_runs})")
    elif total_wins_cov > total_wins_pf:
        print(f"❌ VERDETTO: COV-only è MIGLIORE (vince {total_wins_cov}/{total_runs})")
    else:
        print("➖ VERDETTO: Sostanzialmente equivalenti")
    print("=" * 85)


if __name__ == "__main__":
    main()
