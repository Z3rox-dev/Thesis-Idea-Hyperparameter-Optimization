#!/usr/bin/env python3
"""Validate High-Density Hypothesis on multiple functions."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
import time
from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_test(func_name: str, dim: int, factor: float, budget: int, seed: int):
    func_obj, default_bounds = FUNS[func_name]
    bounds = [default_bounds[0]] * dim
    
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        split_trials_factor=factor,
    )
    
    def obj(x):
        return func_obj(x)
    
    best_x, best_y = opt.optimize(obj, budget=budget)
    return best_y

def main():
    print("=" * 80)
    print("ALBA-Potential: High-Density Validation (Rosenbrock & Rastrigin 20D)")
    print("Configuration: Budget 1600 | Factor 10.0 (High Density)")
    print("=" * 80)
    
    dim = 20
    functions = ["rosenbrock", "rastrigin"]
    seeds = [42, 43, 44]
    
    # Baseline: what "High Budget Only" would do (Factor 3.0)
    # Target: "High Density" (Factor 10.0)
    
    for fname in functions:
        print(f"\nFunction: {fname.upper()} 20D")
        print(f"{'Config':<20} {'Factor':<6} {'Mean Y':<10} {'Std Y':<8}")
        print("-" * 50)
        
        # Scenario 1: Base High Budget (Factor 3.0, Budget 1600)
        results_base = []
        for seed in seeds:
            y = run_test(fname, dim, 3.0, 1600, seed)
            results_base.append(y)
            
        print(f"{'Base (Factor 3)':<20} {3.0:<6.1f} {np.mean(results_base):<10.4f} {np.std(results_base):<8.4f}")
        
        # Scenario 2: High Density (Factor 10.0, Budget 1600)
        results_hd = []
        for seed in seeds:
            y = run_test(fname, dim, 10.0, 1600, seed)
            results_hd.append(y)
            
        print(f"{'High Density':<20} {10.0:<6.1f} {np.mean(results_hd):<10.4f} {np.std(results_hd):<8.4f}")
        
        improvement = (np.mean(results_base) - np.mean(results_hd)) / np.mean(results_base) * 100
        print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main()
