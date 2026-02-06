#!/usr/bin/env python3
"""
Benchmark JAHS-Bench-201 con Checkpoint e Curve di Convergenza
==============================================================

Esegue benchmark con checkpoint a budget: 100, 200, 400, 600, 800, 1000
Salva risultati incrementalmente e genera curve di convergenza.

Usage:
    python benchmark_jahs_checkpoints.py --task cifar10 --n_seeds 10
    python benchmark_jahs_checkpoints.py --smoke_test  # Test veloce
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

# Import from main benchmark
sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import (
    JAHSBenchWrapper, 
    run_random_search, 
    run_optuna_tpe, 
    run_turbo_m, 
    run_alba,
    OPTIMIZERS
)

# Checkpoint budgets
CHECKPOINTS = [100, 200, 400, 600, 800, 1000, 1500, 2000]
SMOKE_CHECKPOINTS = [10, 20]  # Per test veloce


def run_single_optimizer_with_history(
    wrapper: JAHSBenchWrapper,
    opt_name: str,
    max_budget: int,
    seed: int
) -> Tuple[List[float], float, float]:
    """
    Esegue un singolo optimizer e ritorna la history completa.
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
        return [], float('nan'), 0.0


def extract_checkpoint_results(history: List[float], checkpoints: List[int]) -> Dict[int, float]:
    """Estrae i risultati ai vari checkpoint dalla history."""
    results = {}
    for cp in checkpoints:
        if cp <= len(history):
            results[cp] = history[cp - 1]  # history è 0-indexed
        else:
            results[cp] = history[-1] if history else float('nan')
    return results


def plot_convergence_curves(
    all_results: Dict[str, Dict[int, List[Tuple[List[float], int]]]],
    checkpoints: List[int],
    output_path: str,
    task: str
):
    """
    Genera plot delle curve di convergenza.
    all_results: {opt_name: {seed: (history, seed)}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'Random': 'gray',
        'Optuna_TPE': 'blue',
        'ALBA': 'red'
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
        y_max = max(all_values[min(50, len(all_values)):]) * 1.02  # Skip first 50 for max
    else:
        y_min, y_max = 0, 1
    
    # Plot 1: Tutte le curve individuali
    ax1 = axes[0]
    for opt_name, seed_data in all_results.items():
        color = colors.get(opt_name, 'black')
        for seed, (history, _) in seed_data.items():
            if history:
                x = np.arange(1, len(history) + 1)
                ax1.plot(x, history, color=color, alpha=0.3, linewidth=0.5)
    
    # Aggiungi linee medie
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


def run_benchmark_with_checkpoints(
    task: str = 'cifar10',
    n_seeds: int = 10,
    seed_start: int = 0,
    output_dir: str = '/mnt/workspace/thesis/results/jahs_checkpoints',
    smoke_test: bool = False,
    optimizers: Optional[List[str]] = None
):
    """
    Esegue il benchmark con checkpoint e salvataggio incrementale.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    checkpoints = SMOKE_CHECKPOINTS if smoke_test else CHECKPOINTS
    max_budget = max(checkpoints)
    
    if optimizers is None:
        optimizers = list(OPTIMIZERS.keys())
    
    # File per output incrementale
    log_file = os.path.join(output_dir, f'benchmark_{task}_{timestamp}.txt')
    
    def log(msg: str):
        """Stampa e salva su file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log("=" * 80)
    log(f"JAHS-BENCH-201 BENCHMARK WITH CHECKPOINTS")
    log("=" * 80)
    log(f"Task: {task}")
    log(f"Max Budget: {max_budget}")
    log(f"Checkpoints: {checkpoints}")
    log(f"Seeds: {n_seeds} (starting from {seed_start})")
    log(f"Optimizers: {optimizers}")
    log(f"Smoke Test: {smoke_test}")
    log(f"Output: {log_file}")
    log("=" * 80)
    
    # Initialize wrapper
    log("\nInitializing JAHS-Bench-201...")
    wrapper = JAHSBenchWrapper(task=task)
    log(f"Config space dimension: {wrapper.dim}")
    
    # Storage per tutti i risultati
    # {opt_name: {seed: (history, seed)}}
    all_results: Dict[str, Dict[int, Tuple[List[float], int]]] = {
        opt: {} for opt in optimizers
    }
    
    # Storage per checkpoint results
    # {checkpoint: {opt_name: [errors per seed]}}
    checkpoint_results: Dict[int, Dict[str, List[float]]] = {
        cp: {opt: [] for opt in optimizers} for cp in checkpoints
    }
    
    # Run each optimizer
    total_runs = len(optimizers) * n_seeds
    run_count = 0
    
    for opt_name in optimizers:
        log(f"\n{'=' * 60}")
        log(f"Running {opt_name}...")
        log('=' * 60)
        
        for seed in range(seed_start, seed_start + n_seeds):
            run_count += 1
            log(f"\n  [{run_count}/{total_runs}] {opt_name} seed={seed}")
            
            history, final_error, elapsed = run_single_optimizer_with_history(
                wrapper, opt_name, max_budget, seed
            )
            
            if history:
                all_results[opt_name][seed] = (history, seed)
                
                # Extract checkpoint results
                cp_results = extract_checkpoint_results(history, checkpoints)
                for cp, error in cp_results.items():
                    checkpoint_results[cp][opt_name].append(error)
                
                # Log risultati per questo seed
                cp_str = " | ".join([f"@{cp}:{cp_results[cp]:.4f}" for cp in checkpoints])
                log(f"    Final: {final_error:.6f} | Time: {elapsed:.1f}s")
                log(f"    Checkpoints: {cp_str}")
            else:
                log(f"    FAILED")
        
        # Summary per optimizer dopo tutti i seed
        log(f"\n  {opt_name} Summary:")
        for cp in checkpoints:
            errors = checkpoint_results[cp][opt_name]
            if errors:
                valid_errors = [e for e in errors if not np.isnan(e)]
                if valid_errors:
                    mean_e = np.mean(valid_errors)
                    std_e = np.std(valid_errors)
                    log(f"    @{cp}: {mean_e:.6f} ± {std_e:.6f}")
    
    # Final comparison table
    log("\n" + "=" * 80)
    log("FINAL RESULTS - CHECKPOINT COMPARISON")
    log("=" * 80)
    
    # Header
    header = f"{'Optimizer':<15} | " + " | ".join([f"@{cp:>4}" for cp in checkpoints])
    log(header)
    log("-" * len(header))
    
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
    
    # Ranking per ogni checkpoint
    log("\n" + "-" * 60)
    log("RANKING PER CHECKPOINT")
    log("-" * 60)
    
    for cp in checkpoints:
        log(f"\n@{cp} evaluations:")
        ranking = []
        for opt_name in optimizers:
            errors = checkpoint_results[cp][opt_name]
            valid_errors = [e for e in errors if not np.isnan(e)]
            if valid_errors:
                ranking.append((opt_name, np.mean(valid_errors), np.std(valid_errors)))
        
        ranking.sort(key=lambda x: x[1])
        for rank, (name, mean, std) in enumerate(ranking, 1):
            log(f"  {rank}. {name:<15}: {mean:.6f} ± {std:.6f}")
    
    # Generate convergence plot
    plot_path = os.path.join(output_dir, f'convergence_{task}_{timestamp}.png')
    plot_convergence_curves(all_results, checkpoints, plot_path, task)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f'results_{task}_{timestamp}.json')
    json_data = {
        'task': task,
        'checkpoints': checkpoints,
        'n_seeds': n_seeds,
        'seed_start': seed_start,
        'timestamp': timestamp,
        'checkpoint_results': {
            str(cp): {
                opt: {
                    'errors': checkpoint_results[cp][opt],
                    'mean': float(np.nanmean(checkpoint_results[cp][opt])) if checkpoint_results[cp][opt] else None,
                    'std': float(np.nanstd(checkpoint_results[cp][opt])) if checkpoint_results[cp][opt] else None,
                }
                for opt in optimizers
            }
            for cp in checkpoints
        },
        'histories': {
            opt: {
                str(seed): history 
                for seed, (history, _) in seed_data.items()
            }
            for opt, seed_data in all_results.items()
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    log(f"\nJSON results saved: {json_path}")
    
    log("\n" + "=" * 80)
    log("BENCHMARK COMPLETED")
    log("=" * 80)
    
    return checkpoint_results, all_results


def main():
    parser = argparse.ArgumentParser(description='JAHS-Bench-201 Benchmark with Checkpoints')
    parser.add_argument('--task', type=str, default='cifar10',
                       choices=['cifar10', 'colorectal_histology', 'fashion_mnist'],
                       help='Task to benchmark on')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of seeds to run')
    parser.add_argument('--seed_start', type=int, default=0,
                       help='Starting seed')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/workspace/thesis/results/jahs_checkpoints',
                       help='Output directory for results')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run quick smoke test with small budgets')
    parser.add_argument('--optimizers', type=str, nargs='+', default=None,
                       choices=list(OPTIMIZERS.keys()),
                       help='Optimizers to run (default: all)')
    
    args = parser.parse_args()
    
    run_benchmark_with_checkpoints(
        task=args.task,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        optimizers=args.optimizers
    )


if __name__ == '__main__':
    main()
