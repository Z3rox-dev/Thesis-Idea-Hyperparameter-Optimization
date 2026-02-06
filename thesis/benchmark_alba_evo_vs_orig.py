#!/usr/bin/env python3
"""
Benchmark: ALBA Original vs ALBA Evolutionary
==============================================

Compares the original ALBA algorithm with the evolutionary version featuring:
- Thompson sampling with Beta posterior for leaf selection
- Active/Suspended leaf pools (seed bank for soft pruning)
- Tournament selection (size=2)
- Resurrection trials for suspended cubes

Runs on all 3 JAHS-Bench-201 tasks with checkpoints up to 2000 evaluations.

Usage:
    python benchmark_alba_evo_vs_orig.py
    python benchmark_alba_evo_vs_orig.py --task cifar10
    python benchmark_alba_evo_vs_orig.py --smoke_test
"""

import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add thesis path
sys.path.insert(0, '/mnt/workspace/thesis')

# Import JAHS wrapper from existing benchmark
from benchmark_jahs import JAHSBenchWrapper

# Import both ALBA versions with aliases
from ALBA_V1 import ALBA as ALBA_Original
from ALBA_V1_evo import ALBA as ALBA_Evo

# Checkpoint budgets
CHECKPOINTS = [100, 200, 400, 600, 800, 1000, 1500, 2000]
SMOKE_CHECKPOINTS = [20, 50, 100]  # For quick testing

# All JAHS tasks
ALL_TASKS = ['cifar10', 'colorectal_histology', 'fashion_mnist']


def get_categorical_dims():
    """
    Return categorical dimensions for JAHS-Bench-201 space.
    HP_ORDER = ['LearningRate', 'WeightDecay', 'N', 'W', 'Resolution', 
                'Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']
    """
    return [
        (2, 3),   # N: 3 choices [1, 3, 5]
        (3, 3),   # W: 3 choices [4, 8, 16]
        (4, 3),   # Resolution: 3 choices [0.25, 0.5, 1.0]
        (5, 3),   # Activation: 3 choices ['ReLU', 'Hardswish', 'Mish']
        (6, 2),   # TrivialAugment: 2 choices [True, False]
        (7, 5),   # Op1: 5 choices [0, 1, 2, 3, 4]
        (8, 5),   # Op2: 5 choices
        (9, 5),   # Op3: 5 choices
        (10, 5),  # Op4: 5 choices
        (11, 5),  # Op5: 5 choices
        (12, 5),  # Op6: 5 choices
    ]


def run_alba_original(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """Run original ALBA algorithm."""
    wrapper.reset()
    dim = wrapper.dim
    categorical_dims = get_categorical_dims()
    
    opt = ALBA_Original(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    
    best_error = float('inf')
    history = []
    
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_error = min(best_error, y)
        history.append(best_error)
    
    return history, best_error


def run_alba_evo(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """Run evolutionary ALBA algorithm with Thompson sampling and seed bank."""
    wrapper.reset()
    dim = wrapper.dim
    categorical_dims = get_categorical_dims()
    
    opt = ALBA_Evo(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    
    best_error = float('inf')
    history = []
    
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_error = min(best_error, y)
        history.append(best_error)
    
    return history, best_error


# Optimizer registry
OPTIMIZERS = {
    'ALBA_Original': run_alba_original,
    'ALBA_Evo': run_alba_evo,
}


def run_single_optimizer_with_history(
    wrapper: JAHSBenchWrapper,
    opt_name: str,
    max_budget: int,
    seed: int
) -> Tuple[List[float], float, float]:
    """
    Run a single optimizer and return full history.
    Returns: (history, final_error, elapsed_time)
    """
    opt_func = OPTIMIZERS[opt_name]
    start_time = time.time()
    
    try:
        history, final_error = opt_func(wrapper, max_budget, seed)
        elapsed = time.time() - start_time
        return history, final_error, elapsed
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return [], float('nan'), 0.0


def extract_checkpoint_results(history: List[float], checkpoints: List[int]) -> Dict[int, float]:
    """Extract results at each checkpoint from history."""
    results = {}
    for cp in checkpoints:
        if cp <= len(history):
            results[cp] = history[cp - 1]  # history is 0-indexed
        else:
            results[cp] = history[-1] if history else float('nan')
    return results


def plot_convergence_curves(
    all_results: Dict[str, Dict[int, Tuple[List[float], int]]],
    checkpoints: List[int],
    output_path: str,
    task: str
):
    """Generate convergence curve plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'ALBA_Original': 'blue',
        'ALBA_Evo': 'red',
    }
    
    max_budget = max(checkpoints)
    
    # Collect all values for y-axis limits
    all_values = []
    for opt_name, seed_data in all_results.items():
        for seed, (history, _) in seed_data.items():
            if history:
                all_values.extend(history)
    
    if all_values:
        y_min = min(all_values) * 0.95
        y_max = max(all_values[min(50, len(all_values)):]) * 1.02
    else:
        y_min, y_max = 0, 1
    
    # Plot 1: All individual curves
    ax1 = axes[0]
    for opt_name, seed_data in all_results.items():
        color = colors.get(opt_name, 'black')
        for seed, (history, _) in seed_data.items():
            if history:
                x = np.arange(1, len(history) + 1)
                ax1.plot(x, history, color=color, alpha=0.3, linewidth=0.5)
    
    # Add mean lines
    for opt_name, seed_data in all_results.items():
        histories = [h for h, _ in seed_data.values() if h]
        if histories:
            min_len = min(len(h) for h in histories)
            histories_arr = np.array([h[:min_len] for h in histories])
            mean_curve = np.mean(histories_arr, axis=0)
            x = np.arange(1, min_len + 1)
            ax1.plot(x, mean_curve, color=colors.get(opt_name, 'black'), 
                    linewidth=2, label=opt_name)
    
    ax1.set_xlabel('Evaluations', fontsize=12)
    ax1.set_ylabel('Validation Error', fontsize=12)
    ax1.set_title(f'{task} - Convergence Curves (all seeds)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_budget)
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Mean ± std
    ax2 = axes[1]
    for opt_name, seed_data in all_results.items():
        histories = [h for h, _ in seed_data.values() if h]
        if histories:
            min_len = min(len(h) for h in histories)
            histories_arr = np.array([h[:min_len] for h in histories])
            mean_curve = np.mean(histories_arr, axis=0)
            std_curve = np.std(histories_arr, axis=0)
            x = np.arange(1, min_len + 1)
            
            color = colors.get(opt_name, 'black')
            ax2.plot(x, mean_curve, color=color, linewidth=2, label=opt_name)
            ax2.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.2)
    
    ax2.set_xlabel('Evaluations', fontsize=12)
    ax2.set_ylabel('Validation Error', fontsize=12)
    ax2.set_title(f'{task} - Mean ± Std', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_budget)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def run_benchmark_single_task(
    task: str,
    n_seeds: int,
    seed_start: int,
    output_dir: str,
    checkpoints: List[int],
    optimizers: List[str],
    log_func
) -> Tuple[Dict, Dict]:
    """Run benchmark for a single task."""
    log_func(f"\n{'=' * 60}")
    log_func(f"TASK: {task}")
    log_func('=' * 60)
    
    max_budget = max(checkpoints)
    
    # Initialize wrapper
    log_func(f"  Initializing JAHS-Bench-201 for {task}...")
    wrapper = JAHSBenchWrapper(task=task)
    log_func(f"  Config space dimension: {wrapper.dim}")
    
    # Storage
    all_results: Dict[str, Dict[int, Tuple[List[float], int]]] = {
        opt: {} for opt in optimizers
    }
    
    checkpoint_results: Dict[int, Dict[str, List[float]]] = {
        cp: {opt: [] for opt in optimizers} for cp in checkpoints
    }
    
    # Run each optimizer
    total_runs = len(optimizers) * n_seeds
    run_count = 0
    
    for opt_name in optimizers:
        log_func(f"\n  Running {opt_name}...")
        
        for seed in range(seed_start, seed_start + n_seeds):
            run_count += 1
            log_func(f"    [{run_count}/{total_runs}] {opt_name} seed={seed}")
            
            history, final_error, elapsed = run_single_optimizer_with_history(
                wrapper, opt_name, max_budget, seed
            )
            
            if history:
                all_results[opt_name][seed] = (history, seed)
                
                cp_results = extract_checkpoint_results(history, checkpoints)
                for cp, error in cp_results.items():
                    checkpoint_results[cp][opt_name].append(error)
                
                log_func(f"      Final: {final_error:.6f} | Time: {elapsed:.1f}s")
            else:
                log_func(f"      FAILED")
    
    # Generate plot for this task
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'convergence_{task}_{timestamp}.png')
    plot_convergence_curves(all_results, checkpoints, plot_path, task)
    
    return checkpoint_results, all_results


def run_full_benchmark(
    tasks: List[str] = None,
    n_seeds: int = 5,
    seed_start: int = 0,
    output_dir: str = '/mnt/workspace/thesis/results/alba_evo_vs_orig',
    smoke_test: bool = False,
    optimizers: Optional[List[str]] = None
):
    """Run full benchmark across all tasks."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if tasks is None:
        tasks = ALL_TASKS
    
    checkpoints = SMOKE_CHECKPOINTS if smoke_test else CHECKPOINTS
    max_budget = max(checkpoints)
    
    if optimizers is None:
        optimizers = list(OPTIMIZERS.keys())
    
    # Log file
    log_file = os.path.join(output_dir, f'benchmark_evo_vs_orig_{timestamp}.txt')
    
    def log(msg: str):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log("=" * 80)
    log("ALBA EVOLUTIONARY vs ORIGINAL BENCHMARK")
    log("=" * 80)
    log(f"Tasks: {tasks}")
    log(f"Max Budget: {max_budget}")
    log(f"Checkpoints: {checkpoints}")
    log(f"Seeds: {n_seeds} (starting from {seed_start})")
    log(f"Optimizers: {optimizers}")
    log(f"Smoke Test: {smoke_test}")
    log(f"Output: {log_file}")
    log("=" * 80)
    
    # Storage for all tasks
    all_task_results = {}
    all_task_histories = {}
    
    for task in tasks:
        checkpoint_results, all_results = run_benchmark_single_task(
            task=task,
            n_seeds=n_seeds,
            seed_start=seed_start,
            output_dir=output_dir,
            checkpoints=checkpoints,
            optimizers=optimizers,
            log_func=log
        )
        all_task_results[task] = checkpoint_results
        all_task_histories[task] = all_results
        
        # Print task summary
        log(f"\n  {task} Summary:")
        for cp in checkpoints:
            log(f"    Checkpoint @{cp}:")
            for opt_name in optimizers:
                errors = checkpoint_results[cp][opt_name]
                if errors:
                    valid_errors = [e for e in errors if not np.isnan(e)]
                    if valid_errors:
                        mean_e = np.mean(valid_errors)
                        std_e = np.std(valid_errors)
                        log(f"      {opt_name}: {mean_e:.6f} ± {std_e:.6f}")
    
    # Final cross-task comparison
    log("\n" + "=" * 80)
    log("FINAL RESULTS - CROSS-TASK COMPARISON")
    log("=" * 80)
    
    for task in tasks:
        log(f"\n--- {task} ---")
        header = f"{'Optimizer':<15} | " + " | ".join([f"@{cp:>4}" for cp in checkpoints])
        log(header)
        log("-" * len(header))
        
        checkpoint_results = all_task_results[task]
        for opt_name in optimizers:
            row = f"{opt_name:<15} | "
            for cp in checkpoints:
                errors = checkpoint_results[cp][opt_name]
                valid_errors = [e for e in errors if not np.isnan(e)]
                if valid_errors:
                    mean_e = np.mean(valid_errors)
                    row += f"{mean_e:.4f} | "
                else:
                    row += f"{'N/A':>6} | "
            log(row)
    
    # Win/Loss summary at final checkpoint
    final_cp = max(checkpoints)
    log("\n" + "-" * 60)
    log(f"WIN/LOSS SUMMARY @ {final_cp} evaluations")
    log("-" * 60)
    
    wins = {opt: 0 for opt in optimizers}
    for task in tasks:
        checkpoint_results = all_task_results[task]
        task_means = {}
        for opt_name in optimizers:
            errors = checkpoint_results[final_cp][opt_name]
            valid_errors = [e for e in errors if not np.isnan(e)]
            if valid_errors:
                task_means[opt_name] = np.mean(valid_errors)
        
        if len(task_means) == len(optimizers):
            winner = min(task_means, key=task_means.get)
            wins[winner] += 1
            log(f"  {task}: {winner} wins ({task_means[winner]:.6f})")
    
    log(f"\nTotal wins: {wins}")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f'results_evo_vs_orig_{timestamp}.json')
    json_data = {
        'tasks': tasks,
        'checkpoints': checkpoints,
        'n_seeds': n_seeds,
        'seed_start': seed_start,
        'timestamp': timestamp,
        'results_by_task': {
            task: {
                str(cp): {
                    opt: {
                        'errors': all_task_results[task][cp][opt],
                        'mean': float(np.nanmean(all_task_results[task][cp][opt])) 
                                if all_task_results[task][cp][opt] else None,
                        'std': float(np.nanstd(all_task_results[task][cp][opt])) 
                               if all_task_results[task][cp][opt] else None,
                    }
                    for opt in optimizers
                }
                for cp in checkpoints
            }
            for task in tasks
        },
        'histories_by_task': {
            task: {
                opt: {
                    str(seed): history 
                    for seed, (history, _) in seed_data.items()
                }
                for opt, seed_data in all_task_histories[task].items()
            }
            for task in tasks
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    log(f"\nJSON results saved: {json_path}")
    
    log("\n" + "=" * 80)
    log("BENCHMARK COMPLETED")
    log("=" * 80)
    
    return all_task_results, all_task_histories


def main():
    parser = argparse.ArgumentParser(description='ALBA Evolutionary vs Original Benchmark')
    parser.add_argument('--task', type=str, default=None,
                       choices=['cifar10', 'colorectal_histology', 'fashion_mnist', 'all'],
                       help='Task to benchmark (default: all)')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of seeds to run')
    parser.add_argument('--seed_start', type=int, default=0,
                       help='Starting seed')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/workspace/thesis/results/alba_evo_vs_orig',
                       help='Output directory for results')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run quick smoke test with small budgets')
    
    args = parser.parse_args()
    
    if args.task is None or args.task == 'all':
        tasks = ALL_TASKS
    else:
        tasks = [args.task]
    
    run_full_benchmark(
        tasks=tasks,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )


if __name__ == '__main__':
    main()
