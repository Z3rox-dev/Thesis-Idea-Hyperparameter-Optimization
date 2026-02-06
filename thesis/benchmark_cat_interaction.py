"""
Benchmark: ALBA_V1 (categorical interaction improvements) vs ALBA_V1_old (TPE baseline)
Testing on JAHS-Bench-201
"""
import numpy as np
import sys
import time
from typing import Callable, Tuple

# Import optimizers
from ALBA_V1 import ALBA as ALBA_NEW
from ALBA_V1_old import ALBA as ALBA_OLD

# Setup JAHS-bench
sys.path.append('/mnt/workspace')
from jahs_bench import Benchmark

# Define parameter space
JAHS_BOUNDS = [
    (0.0, 1.0),   # 0: N (categorical 1,3,5 -> 3 choices)
    (0.0, 1.0),   # 1: W (categorical 4,8,16 -> 3 choices)
    (0.0, 1.0),   # 2: Resolution (0.25,0.5,1.0 -> 3 choices)
    (0.0, 1.0),   # 3: TrivialAugment (0,1 -> 2 choices)
    (0.0, 1.0),   # 4: Activation (ReLU, Hardswish, Mish -> 3 choices)
    (0.0, 1.0),   # 5-10: Op1-Op6 (5 choices each)
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),   # 11: LearningRate (continuous log-scale)
    (0.0, 1.0),   # 12: WeightDecay (continuous log-scale)
]

# Categorical dimensions: (dim_idx, n_choices)
CATEGORICAL_DIMS = [
    (0, 3),    # N
    (1, 3),    # W
    (2, 3),    # Resolution
    (3, 2),    # TrivialAugment
    (4, 3),    # Activation
    (5, 5),    # Op1
    (6, 5),    # Op2
    (7, 5),    # Op3
    (8, 5),    # Op4
    (9, 5),    # Op5
    (10, 5),   # Op6
]

def decode_config(x: np.ndarray) -> dict:
    """Convert continuous vector to JAHS config."""
    def discretize(val, choices):
        idx = min(int(round(val * (len(choices) - 1))), len(choices) - 1)
        return choices[idx]
    
    N_choices = [1, 3, 5]
    W_choices = [4, 8, 16]
    Res_choices = [0.25, 0.5, 1.0]
    Act_choices = ["ReLU", "Hardswish", "Mish"]
    Op_choices = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
    
    config = {
        "N": discretize(x[0], N_choices),
        "W": discretize(x[1], W_choices),
        "Resolution": discretize(x[2], Res_choices),
        "TrivialAugment": bool(round(x[3])),
        "Activation": discretize(x[4], Act_choices),
        "Op1": discretize(x[5], Op_choices),
        "Op2": discretize(x[6], Op_choices),
        "Op3": discretize(x[7], Op_choices),
        "Op4": discretize(x[8], Op_choices),
        "Op5": discretize(x[9], Op_choices),
        "Op6": discretize(x[10], Op_choices),
        "LearningRate": 10 ** (-3 + 2 * x[11]),    # [0.001, 0.1]
        "WeightDecay": 10 ** (-5 + 2 * x[12]),     # [0.00001, 0.001]
    }
    return config

def run_benchmark(n_runs: int = 10, budget: int = 100, dataset: str = "cifar10"):
    """Run benchmark comparing NEW vs OLD."""
    print(f"{'='*60}")
    print(f"Benchmark: ALBA_V1 (cat-interaction) vs ALBA_V1_old (TPE baseline)")
    print(f"Dataset: {dataset}, Budget: {budget}, Runs: {n_runs}")
    print(f"{'='*60}\n")
    
    # Load benchmark
    bench = Benchmark(
        task=dataset,
        save_dir="/mnt/workspace/jahs_bench_data",
        kind="surrogate",
        download=False
    )
    
    def objective(x: np.ndarray) -> float:
        config = decode_config(x)
        try:
            result = bench(config, nepochs=200)
            acc = result[200]['valid-acc']
            return -acc  # Minimize negative accuracy
        except Exception as e:
            return 0.0  # Worst case
    
    results_new = []
    results_old = []
    
    for run in range(n_runs):
        seed = 42 + run
        print(f"\n--- Run {run+1}/{n_runs} (seed={seed}) ---")
        
        # Run NEW (categorical interaction improvements)
        opt_new = ALBA_NEW(
            bounds=JAHS_BOUNDS,
            categorical_dims=CATEGORICAL_DIMS,
            seed=seed
        )
        t0 = time.time()
        best_x_new, best_y_new = opt_new.optimize(objective, budget=budget)
        time_new = time.time() - t0
        acc_new = -best_y_new
        results_new.append(acc_new)
        print(f"  NEW: {acc_new:.4f} ({time_new:.1f}s)")
        
        # Run OLD (TPE baseline)
        opt_old = ALBA_OLD(
            bounds=JAHS_BOUNDS,
            categorical_dims=CATEGORICAL_DIMS,
            seed=seed
        )
        t0 = time.time()
        best_x_old, best_y_old = opt_old.optimize(objective, budget=budget)
        time_old = time.time() - t0
        acc_old = -best_y_old
        results_old.append(acc_old)
        print(f"  OLD: {acc_old:.4f} ({time_old:.1f}s)")
        
        diff = acc_new - acc_old
        winner = "NEW" if diff > 0.001 else ("OLD" if diff < -0.001 else "TIE")
        print(f"  Diff: {diff:+.4f} ({winner})")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    results_new = np.array(results_new)
    results_old = np.array(results_old)
    
    print(f"\nNEW (cat-interaction):")
    print(f"  Mean: {results_new.mean():.4f} ± {results_new.std():.4f}")
    print(f"  Min:  {results_new.min():.4f}")
    print(f"  Max:  {results_new.max():.4f}")
    
    print(f"\nOLD (TPE baseline):")
    print(f"  Mean: {results_old.mean():.4f} ± {results_old.std():.4f}")
    print(f"  Min:  {results_old.min():.4f}")
    print(f"  Max:  {results_old.max():.4f}")
    
    # Win/loss/tie count
    wins = sum(1 for n, o in zip(results_new, results_old) if n - o > 0.001)
    losses = sum(1 for n, o in zip(results_new, results_old) if o - n > 0.001)
    ties = n_runs - wins - losses
    
    print(f"\nHead-to-head (>0.1% threshold):")
    print(f"  NEW wins: {wins}")
    print(f"  OLD wins: {losses}")
    print(f"  Ties:     {ties}")
    
    # Paired differences
    diffs = results_new - results_old
    print(f"\nPaired difference (NEW - OLD):")
    print(f"  Mean: {diffs.mean():+.4f}")
    print(f"  Std:  {diffs.std():.4f}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results_new, results_old)
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    
    if p_value < 0.05:
        if diffs.mean() > 0:
            print("  => NEW is significantly BETTER (p<0.05)")
        else:
            print("  => OLD is significantly BETTER (p<0.05)")
    else:
        print("  => No significant difference (p>=0.05)")
    
    return results_new, results_old

if __name__ == "__main__":
    run_benchmark(n_runs=10, budget=100, dataset="cifar10")
