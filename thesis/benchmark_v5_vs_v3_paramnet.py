
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
except ImportError as e:
    print(f"Error importing HPOBench: {e}")
    print("Make sure HPOBench is in the python path.")
    sys.exit(1)

def run_comparison(benchmark_cls, n_seeds=5, start_seed=0, budget=50, log_file="benchmark_results_log.txt"):
    print(f"\n=== Benchmarking {benchmark_cls.__name__} ===")
    with open(log_file, "a") as f:
        f.write(f"\n=== Benchmarking {benchmark_cls.__name__} ===\n")
    
    wins_v3 = 0
    wins_v5 = 0
    ties = 0
    
    results_v3 = []
    results_v5 = []
    
    for i in range(n_seeds):
        seed = start_seed + i
        print(f"  Seed {seed} ({i+1}/{n_seeds})...", end="", flush=True)
        
        try:
            # Setup benchmark
            bench = benchmark_cls(rng=seed)
            
            # Extract bounds and names from ConfigurationSpace
            cs = bench.get_configuration_space()
            bounds = []
            hp_names = []
            for hp in cs.get_hyperparameters():
                hp_names.append(hp.name)
                if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                    bounds.append((float(hp.lower), float(hp.upper)))
                else:
                    # print(f"Warning: Unsupported hyperparameter type: {type(hp)}")
                    bounds.append((0.0, 1.0))
            
            def objective(x):
                # Convert np.ndarray to dictionary configuration
                config = {}
                for name, val in zip(hp_names, x):
                    hp = cs.get_hyperparameter(name)
                    # Check if hyperparameter is an integer type
                    if "Integer" in str(type(hp)):
                        config[name] = int(round(val))
                    else:
                        config[name] = val
                
                # ParamNet minimizes, but HPOptimizer expects maximize=False (minimization)
                res = bench.objective_function(config)
                return res['function_value']
            
            # Run v3
            opt_v3 = HPO_v3(bounds, maximize=False, seed=seed)
            _, best_v3 = opt_v3.optimize(objective, budget=budget)
            results_v3.append(best_v3)
            
            # Run v5
            opt_v5 = HPO_v5(bounds, maximize=False, seed=seed)
            _, best_v5 = opt_v5.optimize(objective, budget=budget)
            results_v5.append(best_v5)
            
            # Compare
            winner = "Tie"
            if best_v5 < best_v3 - 1e-6:
                wins_v5 += 1
                winner = "v5"
                print(f" v5 Wins ({best_v5:.4f} vs {best_v3:.4f})")
            elif best_v3 < best_v5 - 1e-6:
                wins_v3 += 1
                winner = "v3"
                print(f" v3 Wins ({best_v5:.4f} vs {best_v3:.4f})")
            else:
                ties += 1
                print(f" Tie ({best_v5:.4f})")
            
            # Log to file
            with open(log_file, "a") as f:
                f.write(f"Dataset: {benchmark_cls.__name__}, Seed: {seed}, v5: {best_v5:.6f}, v3: {best_v3:.6f}, Winner: {winner}\n")
                
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
            
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
    n_seeds = 30
    start_seed = 300
    log_file = "benchmark_v5_vs_v3_paramnet_full_results.txt"
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write(f"Benchmark Results: v5 vs v3 on ParamNet\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {start_seed} to {start_seed + n_seeds - 1}\n")
        f.write("-" * 50 + "\n")
    
    print("Starting Benchmark: v5 (Analysis-Derived) vs v3 (Baseline)")
    print(f"Budget: {budget} evaluations per run")
    print(f"Seeds: {start_seed} to {start_seed + n_seeds - 1}")
    print(f"Logging to: {log_file}")
    
    summary_stats = []
    
    for b in benchmarks:
        stats = run_comparison(b, n_seeds=n_seeds, start_seed=start_seed, budget=budget, log_file=log_file)
        summary_stats.append(stats)
        total_v5 += stats["wins_v5"]
        total_v3 += stats["wins_v3"]
        total_ties += stats["ties"]

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
    
    with open(log_file, "a") as f:
        f.write("\n" + "="*85 + "\n")
        f.write(f"{'DATASET':<15} | {'V5 WINS':<8} | {'V3 WINS':<8} | {'TIES':<5} | {'WIN RATIO (v5)':<15} | {'AVG V5':<10} | {'AVG V3':<10}\n")
        f.write("-" * 85 + "\n")
        for s in summary_stats:
            win_ratio = (s['wins_v5'] / s['n_seeds']) * 100
            f.write(f"{s['dataset']:<15} | {s['wins_v5']:<8} | {s['wins_v3']:<8} | {s['ties']:<5} | {win_ratio:5.1f}%          | {s['avg_v5']:.6f}   | {s['avg_v3']:.6f}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'TOTAL':<15} | {total_v5:<8} | {total_v3:<8} | {total_ties:<5} | {global_win_ratio:5.1f}%          | {'-':<10} | {'-':<10}\n")
        f.write("="*85 + "\n")
        
    print("\n" + "="*40)
    print(f"FINAL SCORE: v5: {total_v5} | v3: {total_v3} | Ties: {total_ties}")
    print("="*40)
