#!/usr/bin/env python3
"""
Benchmark: ALBA COV-only vs ALBA COV+PotentialField su YAHPO rbv2_xgboost

Esegue su diversi task_id con budget 400.
"""

import sys
import os

sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS
from yahpo_gym import benchmark_set, local_config

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# YAHPO WRAPPER for rbv2_xgboost
# ============================================================================

class RBV2XGBoostWrapper:
    """Wrapper per YAHPO rbv2_xgboost che gestisce config space."""
    
    def __init__(self, task_id: str = "3"):
        local_config.init_config()
        local_config.set_data_path("/mnt/workspace/data/")
        
        self.bench = benchmark_set.BenchmarkSet("rbv2_xgboost")
        self.bench.set_instance(task_id)
        self.task_id = task_id
        
        self.cs = self.bench.get_opt_space()
        self._build_simple_bounds()
    
    def _build_simple_bounds(self):
        """Build simple [0,1] bounds."""
        self.hp_names = []
        self.hp_info = []
        self.categorical_dims = []
        
        idx = 0
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CS.Constant):
                continue
            
            name = hp.name
            self.hp_names.append(name)
            
            if isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.log:
                    self.hp_info.append(("float_log", hp.lower, hp.upper))
                else:
                    self.hp_info.append(("float", hp.lower, hp.upper))
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                if hp.log:
                    self.hp_info.append(("int_log", hp.lower, hp.upper))
                else:
                    self.hp_info.append(("int", hp.lower, hp.upper))
            elif isinstance(hp, CS.CategoricalHyperparameter):
                self.hp_info.append(("cat", hp.choices))
                self.categorical_dims.append((idx, len(hp.choices)))
            elif isinstance(hp, CS.OrdinalHyperparameter):
                self.hp_info.append(("ord", hp.sequence))
                self.categorical_dims.append((idx, len(hp.sequence)))
            else:
                self.hp_names.pop()
                continue
            
            idx += 1
        
        self.bounds = [(0.0, 1.0)] * len(self.hp_names)
        self.dim = len(self.bounds)
    
    def x_to_config(self, x_norm: np.ndarray) -> dict:
        """Map [0,1]^d to config dict."""
        config = {"task_id": self.task_id}
        
        for i, (name, info) in enumerate(zip(self.hp_names, self.hp_info)):
            val = float(np.clip(x_norm[i], 0.0, 1.0))
            
            if info[0] == "float":
                lo, hi = info[1], info[2]
                config[name] = lo + val * (hi - lo)
            elif info[0] == "float_log":
                lo, hi = np.log(info[1]), np.log(info[2])
                config[name] = np.exp(lo + val * (hi - lo))
            elif info[0] == "int":
                lo, hi = info[1], info[2]
                config[name] = int(round(lo + val * (hi - lo)))
            elif info[0] == "int_log":
                lo, hi = np.log(info[1]), np.log(info[2])
                config[name] = int(round(np.exp(lo + val * (hi - lo))))
            elif info[0] == "cat":
                choices = info[1]
                idx = min(int(val * len(choices)), len(choices) - 1)
                config[name] = choices[idx]
            elif info[0] == "ord":
                sequence = info[1]
                idx = min(int(val * len(sequence)), len(sequence) - 1)
                config[name] = sequence[idx]
        
        return config
    
    def __call__(self, x_norm: np.ndarray) -> float:
        """Evaluate and return AUC (to maximize)."""
        config = self.x_to_config(x_norm)
        try:
            result = self.bench.objective_function(config)[0]
            return float(result.get("auc", 0.5))
        except Exception:
            return 0.5


# ============================================================================
# RUN
# ============================================================================

def run_alba(
    wrapper: RBV2XGBoostWrapper, 
    n_trials: int, 
    seed: int, 
    use_potential_field: bool
) -> float:
    """Run ALBA su YAHPO, ritorna best AUC."""
    
    opt = ALBA(
        bounds=wrapper.bounds,
        seed=seed,
        total_budget=n_trials,
        maximize=True,  # Massimizziamo AUC
        categorical_dims=wrapper.categorical_dims,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        try:
            y = wrapper(x)
        except Exception:
            y = 0.5
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("=" * 75)
    print("  BENCHMARK: ALBA COV-only vs ALBA COV+PotentialField")
    print("  YAHPO rbv2_xgboost - Budget 400")
    print("=" * 75)
    
    BUDGET = 400
    N_SEEDS = 10
    
    # Task IDs per rbv2_xgboost (alcuni dataset OpenML)
    task_ids = ["3", "31", "37", "44", "334", "1036"]
    
    all_results = {}
    
    for task_id in task_ids:
        print(f"\n{'='*65}")
        print(f"  rbv2_xgboost task_id={task_id}")
        print(f"{'='*65}")
        
        try:
            wrapper = RBV2XGBoostWrapper(task_id)
            print(f"  Dim: {wrapper.dim}, Categorical: {len(wrapper.categorical_dims)}")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        
        results_cov = []
        results_pf = []
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}: ", end="", flush=True)
            
            # COV-only (PF=False)
            val_cov = run_alba(wrapper, BUDGET, 100 + seed, use_potential_field=False)
            results_cov.append(val_cov)
            print(f"COV={val_cov:.4f}", end=" ", flush=True)
            
            # COV+PF (PF=True)
            val_pf = run_alba(wrapper, BUDGET, 100 + seed, use_potential_field=True)
            results_pf.append(val_pf)
            print(f"PF={val_pf:.4f}", end="", flush=True)
            
            # Winner (higher AUC is better)
            if val_cov > val_pf:
                print(" → COV wins")
            elif val_pf > val_cov:
                print(" → PF wins")
            else:
                print(" → TIE")
        
        mean_cov = np.mean(results_cov)
        std_cov = np.std(results_cov)
        mean_pf = np.mean(results_pf)
        std_pf = np.std(results_pf)
        
        # Higher is better for AUC
        wins_cov = sum(1 for a, b in zip(results_cov, results_pf) if a > b)
        wins_pf = sum(1 for a, b in zip(results_cov, results_pf) if b > a)
        
        # Delta % (positive = PF better since we maximize)
        delta_pct = (mean_pf - mean_cov) / abs(mean_cov) * 100 if mean_cov != 0 else 0
        
        all_results[task_id] = {
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'mean_pf': mean_pf,
            'std_pf': std_pf,
            'wins_cov': wins_cov,
            'wins_pf': wins_pf,
            'delta_pct': delta_pct,
        }
        
        print(f"\n  Summary task {task_id}:")
        print(f"    COV-only: {mean_cov:.4f} ± {std_cov:.4f}")
        print(f"    COV+PF:   {mean_pf:.4f} ± {std_pf:.4f}")
        print(f"    Delta:    {delta_pct:+.2f}% ({'PF better' if delta_pct > 0 else 'PF worse'})")
        print(f"    Wins:     COV={wins_cov}, PF={wins_pf}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY - YAHPO rbv2_xgboost")
    print("=" * 75)
    
    print(f"\n{'Task ID':<12} {'COV-only':>12} {'COV+PF':>12} {'Δ%':>8} {'Wins COV':>10} {'Wins PF':>10}")
    print("-" * 68)
    
    total_wins_cov = 0
    total_wins_pf = 0
    
    for task_id, r in all_results.items():
        status = "✅" if r['wins_cov'] > r['wins_pf'] else ("❌" if r['wins_pf'] > r['wins_cov'] else "➖")
        print(f"{task_id:<12} {r['mean_cov']:>12.4f} {r['mean_pf']:>12.4f} {r['delta_pct']:>+7.2f}% {r['wins_cov']:>7}/{N_SEEDS} {r['wins_pf']:>7}/{N_SEEDS} {status}")
        total_wins_cov += r['wins_cov']
        total_wins_pf += r['wins_pf']
    
    total_runs = len(all_results) * N_SEEDS
    avg_delta = np.mean([r['delta_pct'] for r in all_results.values()])
    
    print("-" * 68)
    print(f"{'TOTALE':<12} {'':<12} {'':<12} {avg_delta:>+7.2f}% {total_wins_cov:>7}/{total_runs} {total_wins_pf:>7}/{total_runs}")
    
    print("\n" + "=" * 75)
    if total_wins_pf > total_wins_cov:
        print(f"🔥 VERDETTO: COV+PF è MIGLIORE (vince {total_wins_pf}/{total_runs})")
        print("   → Potential Field aiuta su YAHPO XGBoost")
    elif total_wins_cov > total_wins_pf:
        print(f"🎉 VERDETTO: COV-only è MIGLIORE (vince {total_wins_cov}/{total_runs})")
        print("   → Conferma: Potential Field non serve")
    else:
        print("➖ VERDETTO: Sostanzialmente equivalenti")
    print("=" * 75)


if __name__ == "__main__":
    main()
