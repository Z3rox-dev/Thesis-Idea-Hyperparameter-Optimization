#!/usr/bin/env python3
"""
Benchmark: ALBA COV-only vs ALBA COV+PotentialField su JAHS-Bench-201

Esegue su tutti e 3 i task JAHS con budget 400.
Usa l'ambiente conda py39.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# JAHS PARAM SPACE
# ============================================================================

def build_jahs_param_space(nepochs: int = 200) -> Dict[str, Any]:
    """Build JAHS param space with correct types for ALBA."""
    return {
        "Optimizer": ["SGD"],  # Fixed
        "epoch": [nepochs],    # Fixed
        "LearningRate": (1e-3, 1.0, "log"),      # Continuous log-scale
        "WeightDecay": (1e-5, 1e-2, "log"),      # Continuous log-scale
        "N": [1, 3, 5],                          # Categorical (3 choices)
        "W": [4, 8, 16],                         # Categorical (3 choices)
        "Resolution": [0.25, 0.5, 1.0],          # Categorical (3 choices)
        "Activation": ["ReLU", "Hardswish", "Mish"],  # Categorical
        "TrivialAugment": [True, False],         # Categorical (bool)
        "Op1": [0, 1, 2, 3, 4],                  # Categorical (5 choices)
        "Op2": [0, 1, 2, 3, 4],                  # Categorical
        "Op3": [0, 1, 2, 3, 4],                  # Categorical
        "Op4": [0, 1, 2, 3, 4],                  # Categorical
        "Op5": [0, 1, 2, 3, 4],                  # Categorical
        "Op6": [0, 1, 2, 3, 4],                  # Categorical
    }


# ============================================================================
# BENCHMARK CACHE
# ============================================================================

_BENCH_CACHE = {}

def get_jahs_benchmark(task: str, save_dir: str = "/mnt/workspace/jahs_bench_data"):
    """Get or create JAHS benchmark (cached)."""
    if task not in _BENCH_CACHE:
        import jahs_bench
        print(f"    Loading JAHS benchmark for {task}...", flush=True)
        _BENCH_CACHE[task] = jahs_bench.Benchmark(
            task=task,
            kind="surrogate",
            download=False,
            save_dir=save_dir,
        )
        print(f"    Loaded!", flush=True)
    return _BENCH_CACHE[task]


# ============================================================================
# RUN
# ============================================================================

def run_alba(
    bench,
    task: str,
    n_trials: int,
    seed: int,
    use_potential_field: bool,
    nepochs: int = 200,
) -> float:
    """Run ALBA on JAHS benchmark, return best valid-acc."""
    
    param_space = build_jahs_param_space(nepochs)
    
    opt = ALBA(
        param_space=param_space,
        seed=seed,
        maximize=True,  # Maximize valid-acc
        total_budget=n_trials,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    best_y = -np.inf
    for _ in range(n_trials):
        cfg = opt.ask()
        try:
            result = bench(cfg, nepochs=nepochs)
            y = float(result[nepochs]["valid-acc"])
        except Exception:
            y = 0.0
        opt.tell(cfg, y)
        
        if y > best_y:
            best_y = y
    
    return best_y


def main():
    print("=" * 75)
    print("  BENCHMARK: ALBA COV-only vs ALBA COV+PotentialField")
    print("  JAHS-Bench-201 - 3 tasks - Budget 400")
    print("=" * 75)
    
    # Check jahs_bench
    try:
        import jahs_bench
        print("✓ jahs_bench imported successfully")
    except ImportError:
        print("ERROR: jahs_bench not found. Use conda py39:")
        print("  source /mnt/workspace/miniconda3/bin/activate py39")
        return
    
    BUDGET = 400
    N_SEEDS = 10
    NEPOCHS = 200
    
    # Solo 2 task (cifar10 causa OOM)
    tasks = ["fashion_mnist", "colorectal_histology"]
    
    all_results = {}
    
    for task in tasks:
        print(f"\n{'='*65}")
        print(f"  JAHS Task: {task}")
        print(f"{'='*65}")
        
        try:
            bench = get_jahs_benchmark(task)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        
        results_cov = []
        results_pf = []
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}: ", end="", flush=True)
            
            # COV-only (PF=False)
            val_cov = run_alba(bench, task, BUDGET, 100 + seed, use_potential_field=False, nepochs=NEPOCHS)
            results_cov.append(val_cov)
            print(f"COV={val_cov:.2f}%", end=" ", flush=True)
            
            # COV+PF (PF=True)
            val_pf = run_alba(bench, task, BUDGET, 100 + seed, use_potential_field=True, nepochs=NEPOCHS)
            results_pf.append(val_pf)
            print(f"PF={val_pf:.2f}%", end="", flush=True)
            
            # Winner (higher valid-acc is better)
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
        
        # Higher is better for accuracy
        wins_cov = sum(1 for a, b in zip(results_cov, results_pf) if a > b)
        wins_pf = sum(1 for a, b in zip(results_cov, results_pf) if b > a)
        
        # Delta % (positive = PF better)
        delta_pct = (mean_pf - mean_cov) / abs(mean_cov) * 100 if mean_cov != 0 else 0
        
        all_results[task] = {
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'mean_pf': mean_pf,
            'std_pf': std_pf,
            'wins_cov': wins_cov,
            'wins_pf': wins_pf,
            'delta_pct': delta_pct,
        }
        
        print(f"\n  Summary {task}:")
        print(f"    COV-only: {mean_cov:.2f}% ± {std_cov:.2f}")
        print(f"    COV+PF:   {mean_pf:.2f}% ± {std_pf:.2f}")
        print(f"    Delta:    {delta_pct:+.2f}% ({'PF better' if delta_pct > 0 else 'PF worse'})")
        print(f"    Wins:     COV={wins_cov}, PF={wins_pf}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY - JAHS-Bench-201")
    print("=" * 75)
    
    print(f"\n{'Task':<25} {'COV-only':>12} {'COV+PF':>12} {'Δ%':>8} {'Wins COV':>10} {'Wins PF':>10}")
    print("-" * 80)
    
    total_wins_cov = 0
    total_wins_pf = 0
    
    for task, r in all_results.items():
        status = "✅" if r['wins_cov'] > r['wins_pf'] else ("❌" if r['wins_pf'] > r['wins_cov'] else "➖")
        print(f"{task:<25} {r['mean_cov']:>11.2f}% {r['mean_pf']:>11.2f}% {r['delta_pct']:>+7.2f}% {r['wins_cov']:>7}/{N_SEEDS} {r['wins_pf']:>7}/{N_SEEDS} {status}")
        total_wins_cov += r['wins_cov']
        total_wins_pf += r['wins_pf']
    
    total_runs = len(all_results) * N_SEEDS
    avg_delta = np.mean([r['delta_pct'] for r in all_results.values()])
    
    print("-" * 80)
    print(f"{'TOTALE':<25} {'':<12} {'':<12} {avg_delta:>+7.2f}% {total_wins_cov:>7}/{total_runs} {total_wins_pf:>7}/{total_runs}")
    
    print("\n" + "=" * 75)
    if total_wins_pf > total_wins_cov:
        print(f"🔥 VERDETTO: COV+PF è MIGLIORE (vince {total_wins_pf}/{total_runs})")
    elif total_wins_cov > total_wins_pf:
        print(f"🎉 VERDETTO: COV-only è MIGLIORE (vince {total_wins_cov}/{total_runs})")
    else:
        print("➖ VERDETTO: Sostanzialmente equivalenti")
    print("=" * 75)


if __name__ == "__main__":
    main()
