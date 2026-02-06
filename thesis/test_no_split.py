#!/usr/bin/env python3
"""Test No-Split Hypothesis (Depth 0)."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_test(func_name: str, dim: int, max_depth: int, budget: int, seed: int):
    func_obj, default_bounds = FUNS[func_name]
    bounds = [default_bounds[0]] * dim
    
    # split_depth_max=0 disable splits entirely
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        split_depth_max=max_depth,
    )
    
    def obj(x):
        return func_obj(x)
    
    best_x, best_y = opt.optimize(obj, budget=budget)
    return best_y

def main():
    print("=" * 80)
    print("ALBA-Potential: No-Split vs High-Density (20D, Budget 1600)")
    print("Hypothesis: Is a single global model (No-Split) better than partitioned?")
    print("=" * 80)
    
    dim = 20
    budget = 1600
    seeds = [42, 43, 44]
    
    configs = [
        # (Label, Depth)
        ("Base (Split)", 16),
        ("No-Split (Global)", 0),
    ]
    
    functions = ["sphere", "rosenbrock"]
    
    for fname in functions:
        print(f"\nFunction: {fname.upper()} 20D")
        print(f"{'Config':<20} {'Mean Y':<10} {'Std Y':<8}")
        print("-" * 40)
        
        for label, depth in configs:
            results = []
            for seed in seeds:
                y = run_test(fname, dim, depth, budget, seed)
                results.append(y)
            print(f"{label:<20} {np.mean(results):<10.4f} {np.std(results):<8.4f}")

if __name__ == "__main__":
    main()
