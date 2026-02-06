#!/usr/bin/env python3
"""
Benchmark: ALBA COV-only vs ALBA COV+PotentialField su ParamNet

Esegue su tutti i 6 dataset ParamNet con budget 400.
Usa il venv paramnet (py39).
"""

import sys
import os

# Path setup
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS

from alba_framework_potential.optimizer import ALBA

# Import ParamNet benchmarks
from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)


# ============================================================================
# PARAMNET WRAPPER
# ============================================================================

PARAMNET_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


class ParamNetWrapper:
    """Wrapper per ParamNet che mappa [0,1]^d a ConfigSpace."""
    
    def __init__(self, dataset: str):
        bench_cls = PARAMNET_MAP[dataset.lower()]
        self.bench = bench_cls()
        self.cs = self.bench.get_configuration_space()
        self.hps = self.cs.get_hyperparameters()
        
        self.bounds = []
        self.types = []
        
        for hp in self.hps:
            if isinstance(hp, CS.UniformFloatHyperparameter):
                self.bounds.append((float(hp.lower), float(hp.upper)))
                self.types.append("float")
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                self.bounds.append((float(hp.lower), float(hp.upper)))
                self.types.append("int")
            else:
                raise ValueError(f"Unsupported: {type(hp)}")
        
        self.dim = len(self.bounds)
        self.dataset = dataset
    
    def x_to_config(self, x_norm: np.ndarray) -> CS.Configuration:
        """Mappa x_norm in [0,1]^d a Configuration."""
        values = {}
        for i, (hp, (lo, hi), t) in enumerate(zip(self.hps, self.bounds, self.types)):
            v = lo + float(x_norm[i]) * (hi - lo)
            if t == "int":
                v = int(round(v))
                v = max(int(hp.lower), min(int(hp.upper), int(v)))
            values[hp.name] = v
        return CS.Configuration(self.cs, values=values)
    
    def __call__(self, x_norm: np.ndarray) -> float:
        """Valuta e ritorna validation loss (da minimizzare)."""
        config = self.x_to_config(x_norm)
        result = self.bench.objective_function(config)
        return float(result["function_value"])


# ============================================================================
# RUN
# ============================================================================

def run_alba(
    wrapper: ParamNetWrapper, 
    n_trials: int, 
    seed: int, 
    use_potential_field: bool
) -> float:
    """Run ALBA su ParamNet, ritorna best validation loss."""
    
    # ParamNet usa bounds [0,1]^d
    unit_bounds = [(0.0, 1.0)] * wrapper.dim
    
    opt = ALBA(
        bounds=unit_bounds,
        seed=seed,
        total_budget=n_trials,
        maximize=False,  # Minimizziamo validation loss
        use_potential_field=use_potential_field,
        use_coherence_gating=True,  # Sempre attivo
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        try:
            y = wrapper(x)
        except Exception:
            y = 1.0  # Bad loss
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("=" * 75)
    print("  BENCHMARK: ALBA COV-only vs ALBA COV+PotentialField")
    print("  ParamNet - 6 datasets - Budget 400")
    print("=" * 75)
    
    BUDGET = 400
    N_SEEDS = 10
    
    datasets = ["adult", "letter", "optdigits", "mnist", "higgs", "poker"]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*65}")
        print(f"  ParamNet: {dataset.upper()}")
        print(f"{'='*65}")
        
        try:
            wrapper = ParamNetWrapper(dataset)
            print(f"  Dim: {wrapper.dim}")
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
            
            # Winner
            if val_cov < val_pf:
                print(" → COV wins")
            elif val_pf < val_cov:
                print(" → PF wins")
            else:
                print(" → TIE")
        
        # Statistics
        mean_cov = np.mean(results_cov)
        std_cov = np.std(results_cov)
        mean_pf = np.mean(results_pf)
        std_pf = np.std(results_pf)
        
        wins_cov = sum(1 for a, b in zip(results_cov, results_pf) if a < b)
        wins_pf = sum(1 for a, b in zip(results_cov, results_pf) if b < a)
        
        # Delta % (lower is better, so positive delta = PF worse)
        delta_pct = (mean_pf - mean_cov) / abs(mean_cov) * 100 if mean_cov != 0 else 0
        
        all_results[dataset] = {
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'mean_pf': mean_pf,
            'std_pf': std_pf,
            'wins_cov': wins_cov,
            'wins_pf': wins_pf,
            'delta_pct': delta_pct,
        }
        
        print(f"\n  Summary {dataset}:")
        print(f"    COV-only: {mean_cov:.4f} ± {std_cov:.4f}")
        print(f"    COV+PF:   {mean_pf:.4f} ± {std_pf:.4f}")
        print(f"    Delta:    {delta_pct:+.1f}% ({'PF worse' if delta_pct > 0 else 'PF better'})")
        print(f"    Wins:     COV={wins_cov}, PF={wins_pf}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY")
    print("=" * 75)
    
    print(f"\n{'Dataset':<12} {'COV-only':>12} {'COV+PF':>12} {'Δ%':>8} {'Wins COV':>10} {'Wins PF':>10}")
    print("-" * 68)
    
    total_wins_cov = 0
    total_wins_pf = 0
    
    for dataset, r in all_results.items():
        status = "✅" if r['wins_cov'] > r['wins_pf'] else ("❌" if r['wins_pf'] > r['wins_cov'] else "➖")
        print(f"{dataset:<12} {r['mean_cov']:>12.4f} {r['mean_pf']:>12.4f} {r['delta_pct']:>+7.1f}% {r['wins_cov']:>7}/{N_SEEDS} {r['wins_pf']:>7}/{N_SEEDS} {status}")
        total_wins_cov += r['wins_cov']
        total_wins_pf += r['wins_pf']
    
    total_runs = len(all_results) * N_SEEDS
    avg_delta = np.mean([r['delta_pct'] for r in all_results.values()])
    
    print("-" * 68)
    print(f"{'TOTALE':<12} {'':<12} {'':<12} {avg_delta:>+7.1f}% {total_wins_cov:>7}/{total_runs} {total_wins_pf:>7}/{total_runs}")
    
    print("\n" + "=" * 75)
    if total_wins_cov > total_wins_pf:
        print(f"🎉 VERDETTO: COV-only è MIGLIORE (vince {total_wins_cov}/{total_runs})")
        print("   → Conferma: Potential Field disabilitato di default è corretto")
    elif total_wins_pf > total_wins_cov:
        print(f"🔥 VERDETTO: COV+PF è MIGLIORE (vince {total_wins_pf}/{total_runs})")
        print("   → Considerare: Riabilitare Potential Field")
    else:
        print("➖ VERDETTO: Sostanzialmente equivalenti")
    print("=" * 75)


if __name__ == "__main__":
    main()
