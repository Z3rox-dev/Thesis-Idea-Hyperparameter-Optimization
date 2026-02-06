
import numpy as np
import sys
import os
import time
from typing import List, Tuple

# Add current directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HPOBench'))

# --- PATCHES FOR PARAMNET ---
# Mock oslo_concurrency.lockutils
import types
class MockLockUtils:
    def synchronized(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator
sys.modules['oslo_concurrency'] = types.ModuleType('oslo_concurrency')
sys.modules['oslo_concurrency'].lockutils = MockLockUtils()

# Patch sklearn for old pickles
import sklearn.ensemble
import sklearn.tree
if not hasattr(sklearn.ensemble, 'forest'):
    try:
        from sklearn.ensemble import _forest
        sys.modules['sklearn.ensemble.forest'] = _forest
    except ImportError:
        pass
if not hasattr(sklearn.tree, 'tree'):
    try:
        # Map sklearn.tree.tree to sklearn.tree (where DecisionTreeRegressor is)
        sys.modules['sklearn.tree.tree'] = sklearn.tree
    except ImportError:
        pass
# ---------------------------

# Import optimizers
from thesis.hpo_lgs_v3 import HPOptimizer as HPO_v3
from thesis.hpo_lgs_v5 import HPOptimizer as HPO_v5

# Import ParamNet benchmarks
try:
    from hpobench.benchmarks.surrogates.paramnet_benchmark import (
        ParamNetAdultOnStepsBenchmark,
        ParamNetHiggsOnStepsBenchmark,
        ParamNetLetterOnStepsBenchmark,
        ParamNetMnistOnStepsBenchmark,
        ParamNetOptdigitsOnStepsBenchmark,
        ParamNetPokerOnStepsBenchmark
    )
except ImportError:
    print("HPOBench not found. Please install it.")
    sys.exit(1)

def run_comparison(benchmark_cls, n_seeds=5, start_seed=0, budget=50, log_file="benchmark_v5_vs_v3_paramnet_results.txt"):
    print(f"\n=== Benchmarking {benchmark_cls.__name__} ===")
    
    wins_v5 = 0
    wins_v3 = 0
    ties = 0
    
    results_v5 = []
    results_v3 = []
    
    for i in range(n_seeds):
        seed = start_seed + i
        
        # --- Run v5 (Robust & Adaptive) ---
        b_v5 = benchmark_cls(rng=seed)
        cs = b_v5.get_configuration_space()
        
        bounds = []
        for hp in cs.get_hyperparameters():
            if hasattr(hp, 'lower'):
                bounds.append((hp.lower, hp.upper))
            else:
                # Categorical/Ordinal handled as float for simplicity in this test
                bounds.append((0.0, 1.0))
                
        opt_v5 = HPO_v5(bounds=bounds, maximize=False, seed=seed, n_candidates=30, total_budget=budget)
        
        # Run optimization loop
        for _ in range(budget):
            x = opt_v5.ask()
            # Convert to dictionary for benchmark
            config = {}
            for k, hp in enumerate(cs.get_hyperparameters()):
                val = x[k]
                if hasattr(hp, 'lower') and isinstance(hp.lower, int): # Integer parameter
                    val = int(round(val))
                config[hp.name] = val
            
            res = b_v5.objective_function(config)
            y = res['function_value']
            opt_v5.tell(x, y)
            
        best_v5 = opt_v5.best_y
        results_v5.append(best_v5)
        
        # --- Run v3 (Baseline) ---
        b_v3 = benchmark_cls(rng=seed)
        opt_v3 = HPO_v3(bounds=bounds, maximize=False, seed=seed, n_candidates=30)
        
        for _ in range(budget):
            x = opt_v3.ask()
            config = {}
            for k, hp in enumerate(cs.get_hyperparameters()):
                val = x[k]
                if hasattr(hp, 'lower') and isinstance(hp.lower, int): # Integer parameter
                    val = int(round(val))
                config[hp.name] = val
            
            res = b_v3.objective_function(config)
            y = res['function_value']
            opt_v3.tell(x, y)
            
        best_v3 = opt_v3.best_y
        results_v3.append(best_v3)
        
        # Compare
        winner = "Tie"
        if best_v5 < best_v3 - 1e-6:
            winner = "v5"
            wins_v5 += 1
        elif best_v3 < best_v5 - 1e-6:
            winner = "v3"
            wins_v3 += 1
        else:
            ties += 1
            
        print(f"Dataset: {benchmark_cls.__name__}, Seed: {seed}, v5: {best_v5:.6f}, v3: {best_v3:.6f}, Winner: {winner}")
        
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"Dataset: {benchmark_cls.__name__}, Seed: {seed}, v5: {best_v5:.6f}, v3: {best_v3:.6f}, Winner: {winner}\n")

    print(f"  Result: v5 Wins: {wins_v5}, v3 Wins: {wins_v3}, Ties: {ties}")
    avg_v5 = np.mean(results_v5) if results_v5 else float('nan')
    avg_v3 = np.mean(results_v3) if results_v3 else float('nan')
    if results_v5 and results_v3:
        print(f"  Avg v5: {avg_v5:.4f}, Avg v3: {avg_v3:.4f}")
    
    return {
        "dataset": benchmark_cls.__name__.replace("ParamNet", "").replace("OnStepsBenchmark", ""),
        "wins_v5": wins_v5,
        "wins_v3": wins_v3,
        "ties": ties,
        "avg_v5": avg_v5,
        "avg_v3": avg_v3,
        "n_seeds": n_seeds
    }

if __name__ == "__main__":
    benchmarks = [
        ParamNetAdultOnStepsBenchmark,
        ParamNetHiggsOnStepsBenchmark,
        ParamNetLetterOnStepsBenchmark,
        ParamNetMnistOnStepsBenchmark,
        ParamNetOptdigitsOnStepsBenchmark,
        ParamNetPokerOnStepsBenchmark
    ]
    
    total_v5 = 0
    total_v3 = 0
    total_ties = 0
    
    budget = 200
    n_seeds = 5 # Reduced to 5 for quick feedback on the new version
    start_seed = 200 # New seed range
    log_file = "benchmark_v5_vs_v3_paramnet_results.txt"
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write(f"Benchmark Results: v5 (Robust/Adaptive) vs v3 (Baseline) on ParamNet\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {start_seed} to {start_seed + n_seeds - 1}\n")
        f.write("-" * 50 + "\n")
    
    print("Starting Benchmark: v5 (Robust/Adaptive) vs v3 (Baseline)")
    print(f"Budget: {budget} evaluations per run")
    print(f"Seeds: {start_seed} to {start_seed + n_seeds - 1}")
    print(f"Logging to: {log_file}")
    
    summary_stats = []
    
    for b in benchmarks:
        stats = run_comparison(b, n_seeds=n_seeds, start_seed=start_seed, budget=budget, log_file=log_file)
        summary_stats.append(stats)
        total_v5 += stats["wins_v5"]
        total_v3 += stats["wins_v3"]
        total_ties += stats["ties"];

    # Print Final Summary Table
    print("\n" + "="*85)
    print(f"{'DATASET':<15} | {'V5 WINS':<8} | {'V3 WINS':<8} | {'TIES':<5} | {'WIN RATIO (v5)':<15} | {'AVG V5':<10} | {'AVG V3':<10}")
    print("-" * 85)
    
    for s in summary_stats:
        win_ratio = (s['wins_v5'] / s['n_seeds']) * 100
        print(f"{s['dataset']:<15} | {s['wins_v5']:<8} | {s['wins_v3']:<8} | {s['ties']:<5} | {win_ratio:5.1f}%          | {s['avg_v5']:.6f}   | {s['avg_v3']:.6f}")
    
    print("-" * 85)
    total_runs = total_v5 + total_v3 + total_ties
    global_win_ratio = (total_v5 / total_runs) * 100
    print(f"{'TOTAL':<15} | {total_v5:<8} | {total_v3:<8} | {total_ties:<5} | {global_win_ratio:5.1f}%          | {'-':<10} | {'-':<10}")
    print("="*85)
    
    print("\n" + "="*40)
    print(f"FINAL SCORE: v5: {total_v5} | v3: {total_v3} | Ties: {total_ties}")
    print("="*40)
