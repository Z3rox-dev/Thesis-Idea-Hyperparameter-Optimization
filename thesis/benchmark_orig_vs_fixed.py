#!/usr/bin/env python3
"""
Benchmark completo: ORIGINAL vs FIXED su tutti i task_id
- 8 task_id disponibili
- 100 seeds
- budget 600
- Log risultati in file
"""

from __future__ import annotations
import numpy as np
import sys
import types
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from ConfigSpace import Configuration


# =============================================================================
# FAST LOOKUP
# =============================================================================
def _enable_fast_lookup(benchmark):
    """Abilita lookup veloce con cache"""
    import pandas as pd
    
    df = benchmark.table
    if not hasattr(df, "columns") or "result" not in df.columns:
        return
    
    idx_cols = [c for c in df.columns if c != "result"]
    indexed = df.set_index(idx_cols, drop=False)
    try:
        indexed = indexed.sort_index()
    except Exception:
        pass
    
    cache = {}
    max_cache = 200_000
    
    def _search_dataframe_fast(self, row_dict, _df_unused):
        key = tuple(row_dict[c] for c in idx_cols)
        hit = cache.get(key)
        if hit is not None:
            return hit
        
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise AssertionError(f"Multiple matches for {row_dict}")
            row = row.iloc[0]
        res = row["result"]
        
        if len(cache) < max_cache:
            cache[key] = res
        return res
    
    benchmark._search_dataframe = types.MethodType(_search_dataframe_fast, benchmark)


# =============================================================================
# TASK INFO
# =============================================================================
TASK_IDS = [31, 53, 3917, 9952, 10101, 146818, 146821, 146822]
TASK_NAMES = {
    31: "credit-g",
    53: "vehicle", 
    3917: "kc1",
    9952: "phoneme",
    10101: "blood-transfusion",
    146818: "Australian",
    146821: "car",
    146822: "segment",
}


# =============================================================================
# RUN SINGLE OPTIMIZER
# =============================================================================
def run_optimizer(optimizer_class, bounds, seed, budget, objective, **kwargs):
    """Esegue un singolo optimizer e ritorna il best"""
    opt = optimizer_class(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        split_depth_max=8,
        **kwargs
    )
    
    for _ in range(budget):
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
    
    return opt.best_y


# =============================================================================
# BENCHMARK SINGLE TASK
# =============================================================================
def benchmark_task(task_id, seeds, budget, log_file):
    """Esegue benchmark su un singolo task"""
    
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    print(f"\n{'='*80}")
    print(f"TASK {task_id} ({task_name})")
    print(f"{'='*80}")
    
    # Load benchmark
    try:
        bench = TabularBenchmark(model='nn', task_id=task_id)
        _enable_fast_lookup(bench)
    except Exception as e:
        print(f"  ERROR loading task {task_id}: {e}")
        return None
    
    cs = bench.get_configuration_space()
    hp_names = list(cs.get_hyperparameter_names())
    
    # Get bounds
    bounds = []
    for hp_name in hp_names:
        hp = cs.get_hyperparameter(hp_name)
        if hasattr(hp, 'sequence'):
            bounds.append((0.0, float(len(hp.sequence) - 1)))
        else:
            bounds.append((float(hp.lower), float(hp.upper)))
    
    def objective(x):
        config_dict = {}
        for i, hp_name in enumerate(hp_names):
            hp = cs.get_hyperparameter(hp_name)
            if hasattr(hp, 'sequence'):
                idx = int(round(np.clip(x[i], 0, len(hp.sequence) - 1)))
                config_dict[hp_name] = hp.sequence[idx]
            else:
                config_dict[hp_name] = float(np.clip(x[i], hp.lower, hp.upper))
        config = Configuration(cs, values=config_dict)
        return float(bench.objective_function(config)['function_value'])
    
    # Import optimizers
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s
    from debug_v5s_fixed import HPOptimizerV5sFixed
    
    results_orig = []
    results_fixed = []
    
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(seeds) - i - 1)
            print(f"  Progress: {i+1}/{len(seeds)} seeds | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        
        # Run ORIGINAL
        try:
            best_orig = run_optimizer(HPOptimizerV5s, bounds, seed, budget, objective)
            results_orig.append(best_orig)
        except Exception as e:
            print(f"    ERROR seed {seed} ORIGINAL: {e}")
            results_orig.append(np.nan)
        
        # Run FIXED
        try:
            best_fixed = run_optimizer(HPOptimizerV5sFixed, bounds, seed, budget, objective,
                                       global_random_prob=0.05, stagnation_threshold=50)
            results_fixed.append(best_fixed)
        except Exception as e:
            print(f"    ERROR seed {seed} FIXED: {e}")
            results_fixed.append(np.nan)
    
    total_time = time.time() - start_time
    
    # Compute statistics
    orig_arr = np.array([r for r in results_orig if not np.isnan(r)])
    fixed_arr = np.array([r for r in results_fixed if not np.isnan(r)])
    
    if len(orig_arr) == 0 or len(fixed_arr) == 0:
        print(f"  ERROR: No valid results")
        return None
    
    # Stats
    orig_mean = orig_arr.mean()
    orig_std = orig_arr.std()
    orig_min = orig_arr.min()
    orig_max = orig_arr.max()
    
    fixed_mean = fixed_arr.mean()
    fixed_std = fixed_arr.std()
    fixed_min = fixed_arr.min()
    fixed_max = fixed_arr.max()
    
    # Head-to-head comparison
    wins_fixed = 0
    wins_orig = 0
    ties = 0
    for o, f in zip(results_orig, results_fixed):
        if np.isnan(o) or np.isnan(f):
            continue
        if f < o - 1e-9:
            wins_fixed += 1
        elif o < f - 1e-9:
            wins_orig += 1
        else:
            ties += 1
    
    # Results summary
    result = {
        'task_id': task_id,
        'task_name': task_name,
        'n_seeds': len(seeds),
        'budget': budget,
        'time_seconds': total_time,
        'original': {
            'mean': orig_mean,
            'std': orig_std,
            'min': orig_min,
            'max': orig_max,
            'results': results_orig,
        },
        'fixed': {
            'mean': fixed_mean,
            'std': fixed_std,
            'min': fixed_min,
            'max': fixed_max,
            'results': results_fixed,
        },
        'comparison': {
            'wins_fixed': wins_fixed,
            'wins_orig': wins_orig,
            'ties': ties,
            'better': 'FIXED' if fixed_mean < orig_mean else 'ORIGINAL' if orig_mean < fixed_mean else 'TIE',
            'improvement_pct': (orig_mean - fixed_mean) / orig_mean * 100 if fixed_mean < orig_mean else 0,
        }
    }
    
    # Print summary
    print(f"\n  RESULTS for {task_name}:")
    print(f"    ORIGINAL: {orig_mean:.6f} ± {orig_std:.6f} (min={orig_min:.6f}, max={orig_max:.6f})")
    print(f"    FIXED:    {fixed_mean:.6f} ± {fixed_std:.6f} (min={fixed_min:.6f}, max={fixed_max:.6f})")
    print(f"    HEAD-TO-HEAD: FIXED wins {wins_fixed}, ORIGINAL wins {wins_orig}, ties {ties}")
    print(f"    WINNER: {result['comparison']['better']}")
    if result['comparison']['improvement_pct'] > 0:
        print(f"    Improvement: {result['comparison']['improvement_pct']:.2f}%")
    print(f"    Time: {total_time:.1f}s")
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TASK {task_id} ({task_name})\n")
        f.write(f"{'='*80}\n")
        f.write(f"Seeds: {len(seeds)}, Budget: {budget}\n")
        f.write(f"ORIGINAL: {orig_mean:.6f} ± {orig_std:.6f} (min={orig_min:.6f}, max={orig_max:.6f})\n")
        f.write(f"FIXED:    {fixed_mean:.6f} ± {fixed_std:.6f} (min={fixed_min:.6f}, max={fixed_max:.6f})\n")
        f.write(f"HEAD-TO-HEAD: FIXED wins {wins_fixed}, ORIGINAL wins {wins_orig}, ties {ties}\n")
        f.write(f"WINNER: {result['comparison']['better']}\n")
        if result['comparison']['improvement_pct'] > 0:
            f.write(f"Improvement: {result['comparison']['improvement_pct']:.2f}%\n")
        f.write(f"Time: {total_time:.1f}s\n")
    
    return result


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=600)
    parser.add_argument('--n_seeds', type=int, default=100)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--task_ids', type=str, default=None, 
                        help='Comma-separated task IDs (default: all)')
    parser.add_argument('--output_dir', type=str, default='/mnt/workspace/thesis/results')
    args = parser.parse_args()
    
    # Setup
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    
    if args.task_ids:
        task_ids = [int(t) for t in args.task_ids.split(',')]
    else:
        task_ids = TASK_IDS
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'benchmark_orig_vs_fixed_{timestamp}.txt'
    json_file = output_dir / f'benchmark_orig_vs_fixed_{timestamp}.json'
    
    print("=" * 80)
    print("BENCHMARK: ORIGINAL vs FIXED")
    print("=" * 80)
    print(f"Task IDs: {task_ids}")
    print(f"Seeds: {args.n_seeds} (from {args.seed_start})")
    print(f"Budget: {args.budget}")
    print(f"Log file: {log_file}")
    print(f"JSON file: {json_file}")
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK: ORIGINAL vs FIXED\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Task IDs: {task_ids}\n")
        f.write(f"Seeds: {args.n_seeds} (from {args.seed_start})\n")
        f.write(f"Budget: {args.budget}\n")
    
    # Run benchmarks
    all_results = []
    total_start = time.time()
    
    for task_id in task_ids:
        result = benchmark_task(task_id, seeds, args.budget, log_file)
        if result is not None:
            all_results.append(result)
        
        # Save intermediate results
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    summary_lines = []
    
    total_wins_fixed = 0
    total_wins_orig = 0
    total_ties = 0
    
    for r in all_results:
        line = (f"{r['task_name']:20s}: ORIG={r['original']['mean']:.6f} vs "
                f"FIXED={r['fixed']['mean']:.6f} | "
                f"Winner: {r['comparison']['better']:8s} | "
                f"H2H: F{r['comparison']['wins_fixed']}-O{r['comparison']['wins_orig']}-T{r['comparison']['ties']}")
        print(line)
        summary_lines.append(line)
        
        total_wins_fixed += r['comparison']['wins_fixed']
        total_wins_orig += r['comparison']['wins_orig']
        total_ties += r['comparison']['ties']
    
    print(f"\nTOTAL HEAD-TO-HEAD across all tasks:")
    print(f"  FIXED wins: {total_wins_fixed}")
    print(f"  ORIGINAL wins: {total_wins_orig}")
    print(f"  Ties: {total_ties}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # Append final summary to log
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTOTAL HEAD-TO-HEAD across all tasks:\n")
        f.write(f"  FIXED wins: {total_wins_fixed}\n")
        f.write(f"  ORIGINAL wins: {total_wins_orig}\n")
        f.write(f"  Ties: {total_ties}\n")
        f.write(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
    
    print(f"\nResults saved to:")
    print(f"  {log_file}")
    print(f"  {json_file}")


if __name__ == "__main__":
    main()
