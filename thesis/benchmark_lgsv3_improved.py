"""
Benchmark: Improved LGS v3 vs Original on Complex ML-like Functions

Test su:
1. Funzioni classiche difficili (Rosenbrock, Rastrigin, Ackley, ecc.)
2. Nuove funzioni ML-inspired (ml_loss_landscape, hyperparameter_surface, ecc.)

Confronta:
- LGS v3 originale
- LGS v3 migliorato (con le nuove strategie)
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

# Get the repository root directory
try:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback when __file__ is not defined (e.g., interactive mode)
    repo_root = os.getcwd()

# Import ParamSpace functions
sys.path.insert(0, repo_root)
from ParamSpace import FUNS, map_to_domain

# Import HPOptimizer
thesis_dir = os.path.join(repo_root, 'thesis')
sys.path.insert(0, thesis_dir)
from hpo_lgs_v3 import HPOptimizer


def run_optimizer(fun_name: str, seed: int, budget: int) -> Tuple[float, float]:
    """
    Run LGS v3 optimizer on a function.
    Returns: (best_value, time_taken)
    """
    func, bounds = FUNS[fun_name]
    d = len(bounds)
    
    # Optimizer works in normalized space [0,1]^d
    hpo = HPOptimizer(bounds=[(0.0, 1.0)] * d, maximize=False, seed=seed)
    
    def objective(x_norm: np.ndarray) -> float:
        x = map_to_domain(x_norm, bounds)
        return float(func(x))
    
    t0 = time.time()
    best_x, best_y = hpo.optimize(objective, budget=budget)
    elapsed = time.time() - t0
    
    return float(best_y), elapsed


def main():
    print("=" * 80)
    print("  BENCHMARK: LGS v3 IMPROVED - Complex ML-like Functions")
    print("=" * 80)
    
    # Test configuration
    budget = 200
    seeds = [42, 123, 456, 789, 1024]
    
    # Select challenging functions
    test_functions = [
        # Classic difficult functions
        'rosenbrock',
        'rastrigin', 
        'ackley',
        'griewank',
        'levy',
        'schwefel',
        # New ML-inspired functions
        'ml_loss_landscape',
        'hyperparameter_surface',
        'neural_network_loss',
        'ensemble_hyperopt',
        'adversarial_landscape',
        'multiscale_landscape',
    ]
    
    print(f"\nConfiguration:")
    print(f"  Budget: {budget} evaluations per run")
    print(f"  Seeds: {seeds}")
    print(f"  Functions: {len(test_functions)}")
    print()
    
    results: Dict[str, Dict[str, List[float]]] = {}
    
    # Table header
    print(f"{'Function':<25} | {'Best Mean':<12} | {'Best Std':<10} | {'Avg Time':<10}")
    print("-" * 80)
    
    for fun_name in test_functions:
        if fun_name not in FUNS:
            print(f"Skipping {fun_name} (not found)")
            continue
        
        results[fun_name] = {'values': [], 'times': []}
        
        for seed in seeds:
            try:
                best_val, elapsed = run_optimizer(fun_name, seed, budget)
                results[fun_name]['values'].append(best_val)
                results[fun_name]['times'].append(elapsed)
            except Exception as e:
                print(f"Error on {fun_name} seed {seed}: {e}")
                continue
        
        # Compute statistics
        values = np.array(results[fun_name]['values'])
        times = np.array(results[fun_name]['times'])
        
        if len(values) > 0:
            mean_val = float(values.mean())
            std_val = float(values.std())
            mean_time = float(times.mean())
            
            print(f"{fun_name:<25} | {mean_val:<12.6g} | {std_val:<10.6g} | {mean_time:<10.2f}s")
        else:
            print(f"{fun_name:<25} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")
    
    print("-" * 80)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)
    
    for fun_name in test_functions:
        if fun_name not in results or len(results[fun_name]['values']) == 0:
            continue
        
        values = np.array(results[fun_name]['values'])
        print(f"\n{fun_name}:")
        print(f"  Mean: {values.mean():.8g}")
        print(f"  Std:  {values.std():.8g}")
        print(f"  Min:  {values.min():.8g}")
        print(f"  Max:  {values.max():.8g}")
        print(f"  Median: {np.median(values):.8g}")
        
        # Show per-seed results
        print(f"  Per-seed: ", end="")
        for i, (val, seed) in enumerate(zip(values, seeds[:len(values)])):
            print(f"[{seed}: {val:.6g}]", end=" ")
            if (i + 1) % 3 == 0:
                print()
                print(f"            ", end="")
        print()
    
    print("\n" + "=" * 80)
    print("  Benchmark completato!")
    print("=" * 80)


if __name__ == "__main__":
    main()
