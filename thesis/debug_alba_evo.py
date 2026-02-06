#!/usr/bin/env python3
"""
Debug Benchmark: ALBA Original vs ALBA Evo (with debug logging)
================================================================

Deep debugging benchmark to understand why ALBA Evo may not improve over original.
Runs on JAHS-Bench-201 with detailed logging and statistics.

Usage:
    python debug_alba_evo.py --task cifar10 --n_evals 500 --seed 0
    python debug_alba_evo.py --smoke_test
"""

import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '/mnt/workspace/thesis')

from benchmark_jahs import JAHSBenchWrapper
from ALBA_V1 import ALBA as ALBA_Original
from ALBA_V1_evo import ALBA as ALBA_Evo


def get_categorical_dims():
    """Return categorical dimensions for JAHS-Bench-201 space."""
    return [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]


def run_with_detailed_tracking(
    wrapper: JAHSBenchWrapper,
    opt,
    n_evals: int,
    opt_name: str,
    verbose: bool = True
) -> Dict:
    """Run optimizer with detailed tracking of internal state."""
    wrapper.reset()
    
    history = []
    best_error = float('inf')
    
    # Track statistics
    stats = {
        'history': [],
        'active_leaves_count': [],
        'suspended_leaves_count': [],
        'gamma_values': [],
        'selected_pools': [],
        'n_trials_selected': [],
        'n_good_selected': [],
        'good_ratio_selected': [],
        'split_events': [],
        'resurrection_attempts': [],
        'resurrection_successes': [],
        'success_rate_20': [],
    }
    
    is_evo = hasattr(opt, 'active_leaves')
    
    start_time = time.time()
    
    for i in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        
        # Track which cube was selected (before tell)
        if is_evo and opt.last_cube is not None:
            stats['n_trials_selected'].append(opt.last_cube.n_trials)
            stats['n_good_selected'].append(opt.last_cube.n_good)
            stats['good_ratio_selected'].append(opt.last_cube.good_ratio())
            stats['selected_pools'].append(opt._last_selected_pool)
        
        opt.tell(x, y)
        
        best_error = min(best_error, y)
        history.append(best_error)
        stats['history'].append(best_error)
        
        if is_evo:
            stats['active_leaves_count'].append(len(opt.active_leaves))
            stats['suspended_leaves_count'].append(len(opt.suspended_leaves))
            stats['gamma_values'].append(opt.gamma)
            stats['resurrection_attempts'].append(opt._resurrection_attempts)
            stats['resurrection_successes'].append(opt._resurrection_successes)
            if len(opt._recent_successes) > 0:
                stats['success_rate_20'].append(sum(opt._recent_successes) / len(opt._recent_successes))
            else:
                stats['success_rate_20'].append(0.0)
        else:
            # Original ALBA
            stats['active_leaves_count'].append(len(opt.leaves))
            stats['suspended_leaves_count'].append(0)
            stats['gamma_values'].append(opt.gamma)
        
        # Verbose logging every 100 iterations
        if verbose and (i + 1) % 100 == 0:
            if is_evo:
                print(f"  [{opt_name}] iter={i+1}: best={best_error:.6f}, "
                      f"active={len(opt.active_leaves)}, susp={len(opt.suspended_leaves)}, "
                      f"res_att={opt._resurrection_attempts}, res_succ={opt._resurrection_successes}, "
                      f"splits={opt._splits_triggered}")
            else:
                print(f"  [{opt_name}] iter={i+1}: best={best_error:.6f}, "
                      f"leaves={len(opt.leaves)}")
    
    elapsed = time.time() - start_time
    
    # Final summary for evo
    if is_evo:
        stats['final_active'] = len(opt.active_leaves)
        stats['final_suspended'] = len(opt.suspended_leaves)
        stats['total_resurrection_attempts'] = opt._resurrection_attempts
        stats['total_resurrection_successes'] = opt._resurrection_successes
        stats['total_splits'] = opt._splits_triggered
        
        # Analyze resurrection effectiveness
        if opt._resurrection_attempts > 0:
            stats['resurrection_rate'] = opt._resurrection_successes / opt._resurrection_attempts
        else:
            stats['resurrection_rate'] = 0.0
    
    stats['elapsed'] = elapsed
    stats['final_error'] = best_error
    
    return stats


def analyze_and_print_comparison(stats_orig: Dict, stats_evo: Dict, output_file=None):
    """Analyze and print detailed comparison."""
    
    def log(msg):
        print(msg)
        if output_file:
            output_file.write(msg + '\n')
    
    log("\n" + "=" * 80)
    log("DETAILED COMPARISON ANALYSIS")
    log("=" * 80)
    
    # Final errors
    log(f"\nFinal Errors:")
    log(f"  Original: {stats_orig['final_error']:.6f}")
    log(f"  Evo:      {stats_evo['final_error']:.6f}")
    diff = stats_evo['final_error'] - stats_orig['final_error']
    winner = "Original" if diff > 0 else "Evo" if diff < 0 else "Tie"
    log(f"  Difference: {diff:+.6f} ({winner} wins)")
    
    # Leaf management
    log(f"\nLeaf Management:")
    log(f"  Original final leaves: {stats_orig['active_leaves_count'][-1]}")
    log(f"  Evo final active:      {stats_evo['final_active']}")
    log(f"  Evo final suspended:   {stats_evo['final_suspended']}")
    log(f"  Evo total leaves:      {stats_evo['final_active'] + stats_evo['final_suspended']}")
    
    # Resurrection analysis
    log(f"\nResurrection Analysis (Evo only):")
    log(f"  Total attempts:   {stats_evo['total_resurrection_attempts']}")
    log(f"  Total successes:  {stats_evo['total_resurrection_successes']}")
    log(f"  Success rate:     {stats_evo['resurrection_rate']:.2%}")
    
    if stats_evo['resurrection_rate'] < 0.1:
        log(f"  WARNING: Low resurrection success rate - success criteria may be too strict!")
    
    # Split analysis
    log(f"\nSplit Analysis:")
    log(f"  Evo total splits: {stats_evo['total_splits']}")
    
    # Convergence analysis
    log(f"\nConvergence Analysis:")
    checkpoints = [50, 100, 200, 500]
    for cp in checkpoints:
        if cp <= len(stats_orig['history']):
            orig_val = stats_orig['history'][cp-1]
            evo_val = stats_evo['history'][cp-1]
            log(f"  @{cp}: Orig={orig_val:.6f}, Evo={evo_val:.6f}, diff={evo_val-orig_val:+.6f}")
    
    # Success rate evolution (Evo)
    if stats_evo['success_rate_20']:
        log(f"\nSuccess Rate Evolution (Evo, last-20 window):")
        sr = stats_evo['success_rate_20']
        log(f"  Early (iter 50):  {sr[min(49, len(sr)-1)]:.2%}")
        log(f"  Mid (iter 250):   {sr[min(249, len(sr)-1)]:.2%}" if len(sr) > 249 else "  Mid: N/A")
        log(f"  Late (last):      {sr[-1]:.2%}")
        
        avg_success = np.mean(sr)
        log(f"  Average:          {avg_success:.2%}")
        
        if avg_success < 0.15:
            log(f"  WARNING: Very low average success rate - gamma may be too strict!")
    
    # Gamma evolution
    log(f"\nGamma Evolution:")
    gamma_orig = stats_orig['gamma_values']
    gamma_evo = stats_evo['gamma_values']
    log(f"  Orig: start={gamma_orig[10] if len(gamma_orig)>10 else 0:.4f}, "
        f"end={gamma_orig[-1]:.4f}")
    log(f"  Evo:  start={gamma_evo[10] if len(gamma_evo)>10 else 0:.4f}, "
        f"end={gamma_evo[-1]:.4f}")
    
    # Pool selection analysis (Evo)
    if stats_evo['selected_pools']:
        pools = stats_evo['selected_pools']
        from collections import Counter
        pool_counts = Counter(pools)
        log(f"\nPool Selection Distribution (Evo):")
        for pool, count in pool_counts.items():
            log(f"  {pool}: {count} ({count/len(pools)*100:.1f}%)")
    
    return stats_orig, stats_evo


def plot_debug_comparison(stats_orig: Dict, stats_evo: Dict, output_path: str, title: str):
    """Create detailed debug plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Convergence curves
    ax = axes[0, 0]
    ax.plot(stats_orig['history'], label='Original', color='blue', alpha=0.8)
    ax.plot(stats_evo['history'], label='Evo', color='red', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Error')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Leaf counts
    ax = axes[0, 1]
    ax.plot(stats_orig['active_leaves_count'], label='Orig Leaves', color='blue', alpha=0.8)
    ax.plot(stats_evo['active_leaves_count'], label='Evo Active', color='red', alpha=0.8)
    ax.plot(stats_evo['suspended_leaves_count'], label='Evo Suspended', color='orange', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Count')
    ax.set_title('Leaf Counts Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Resurrection stats
    ax = axes[0, 2]
    if stats_evo['resurrection_attempts']:
        ax.plot(stats_evo['resurrection_attempts'], label='Attempts', color='blue')
        ax.plot(stats_evo['resurrection_successes'], label='Successes', color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cumulative Count')
    ax.set_title('Resurrection Statistics (Evo)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gamma evolution
    ax = axes[1, 0]
    ax.plot(stats_orig['gamma_values'], label='Original', color='blue', alpha=0.8)
    ax.plot(stats_evo['gamma_values'], label='Evo', color='red', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma Threshold Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Success rate (Evo)
    ax = axes[1, 1]
    if stats_evo['success_rate_20']:
        ax.plot(stats_evo['success_rate_20'], color='red', alpha=0.8)
    ax.axhline(y=0.2, color='gray', linestyle='--', label='20% threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Success Rate (last 20)')
    ax.set_title('Success Rate Evolution (Evo)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 6: Good ratio of selected cubes (Evo)
    ax = axes[1, 2]
    if stats_evo['good_ratio_selected']:
        ax.scatter(range(len(stats_evo['good_ratio_selected'])), 
                   stats_evo['good_ratio_selected'], alpha=0.3, s=5, color='red')
        # Moving average
        window = 50
        if len(stats_evo['good_ratio_selected']) > window:
            ma = np.convolve(stats_evo['good_ratio_selected'], 
                            np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(stats_evo['good_ratio_selected'])), ma, 
                   color='darkred', linewidth=2, label=f'MA({window})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Good Ratio')
    ax.set_title('Good Ratio of Selected Cubes (Evo)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Debug plot saved: {output_path}")


def run_debug_benchmark(
    task: str = 'cifar10',
    n_evals: int = 500,
    seed: int = 0,
    output_dir: str = '/mnt/workspace/thesis/results/alba_debug',
    enable_evo_debug: bool = True,
    verbose: bool = True
):
    """Run detailed debug benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Output file
    log_path = os.path.join(output_dir, f'debug_{task}_seed{seed}_{timestamp}.txt')
    log_file = open(log_path, 'w')
    
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
    
    log("=" * 80)
    log(f"ALBA DEBUG BENCHMARK")
    log("=" * 80)
    log(f"Task: {task}")
    log(f"Evaluations: {n_evals}")
    log(f"Seed: {seed}")
    log(f"Evo Debug Mode: {enable_evo_debug}")
    log(f"Output: {log_path}")
    log("=" * 80)
    
    # Initialize wrapper
    log("\nInitializing JAHS-Bench-201...")
    wrapper = JAHSBenchWrapper(task=task)
    dim = wrapper.dim
    log(f"Dimension: {dim}")
    
    categorical_dims = get_categorical_dims()
    
    # Run Original
    log(f"\n{'='*60}")
    log("Running ALBA Original...")
    log('='*60)
    
    opt_orig = ALBA_Original(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    
    stats_orig = run_with_detailed_tracking(wrapper, opt_orig, n_evals, "Original", verbose)
    
    # Run Evo
    log(f"\n{'='*60}")
    log("Running ALBA Evo...")
    log('='*60)
    
    opt_evo = ALBA_Evo(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    opt_evo._debug = enable_evo_debug  # Enable debug logging
    
    stats_evo = run_with_detailed_tracking(wrapper, opt_evo, n_evals, "Evo", verbose)
    
    # Analyze
    analyze_and_print_comparison(stats_orig, stats_evo, log_file)
    
    # Plot
    plot_path = os.path.join(output_dir, f'debug_plot_{task}_seed{seed}_{timestamp}.png')
    plot_debug_comparison(stats_orig, stats_evo, plot_path, 
                         f'{task} - ALBA Original vs Evo (seed={seed}, n={n_evals})')
    
    # Save JSON
    json_path = os.path.join(output_dir, f'debug_stats_{task}_seed{seed}_{timestamp}.json')
    json_data = {
        'task': task,
        'n_evals': n_evals,
        'seed': seed,
        'timestamp': timestamp,
        'original': {
            'final_error': stats_orig['final_error'],
            'elapsed': stats_orig['elapsed'],
            'final_leaves': stats_orig['active_leaves_count'][-1],
        },
        'evo': {
            'final_error': stats_evo['final_error'],
            'elapsed': stats_evo['elapsed'],
            'final_active': stats_evo['final_active'],
            'final_suspended': stats_evo['final_suspended'],
            'total_resurrection_attempts': stats_evo['total_resurrection_attempts'],
            'total_resurrection_successes': stats_evo['total_resurrection_successes'],
            'resurrection_rate': stats_evo['resurrection_rate'],
            'total_splits': stats_evo['total_splits'],
        },
        'history_orig': stats_orig['history'],
        'history_evo': stats_evo['history'],
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    log(f"\nJSON saved: {json_path}")
    
    log(f"\nDebug benchmark complete. Results in: {output_dir}")
    
    log_file.close()
    
    return stats_orig, stats_evo


def main():
    parser = argparse.ArgumentParser(description='ALBA Debug Benchmark')
    parser.add_argument('--task', type=str, default='cifar10',
                       choices=['cifar10', 'colorectal_histology', 'fashion_mnist'])
    parser.add_argument('--n_evals', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/workspace/thesis/results/alba_debug')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Quick test with 100 evals')
    parser.add_argument('--no_evo_debug', action='store_true',
                       help='Disable Evo internal debug logging')
    
    args = parser.parse_args()
    
    n_evals = 100 if args.smoke_test else args.n_evals
    
    run_debug_benchmark(
        task=args.task,
        n_evals=n_evals,
        seed=args.seed,
        output_dir=args.output_dir,
        enable_evo_debug=not args.no_evo_debug,
    )


if __name__ == '__main__':
    main()
