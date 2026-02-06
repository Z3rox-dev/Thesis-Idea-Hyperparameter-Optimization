#!/usr/bin/env python3
"""Test impact of max tree depth on ALBA-Potential performance."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_test(dim: int, depth: int, budget: int = 400, seeds: list = [42, 43, 44]):
    """Run benchmark with specific max depth."""
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    results = []
    for seed in seeds:
        opt = ALBA(
            bounds=bounds,
            maximize=False,
            seed=seed,
            total_budget=budget,
            use_potential_field=True,
            use_coherence_gating=True,
            split_depth_max=depth,
        )
        
        def obj(x):
            return func_obj(x)
        
        best_x, best_y = opt.optimize(obj, budget=budget)
        results.append(best_y)
    
    return np.mean(results), np.std(results)

def main():
    print("=" * 60)
    print("ALBA-Potential: Max Depth Impact Test")
    print("=" * 60)
    
    depths = [4, 8, 12, 16, 20]
    dims = [10, 20]
    
    for dim in dims:
        print(f"\n=== {dim}D Sphere ===")
        print(f"{'Depth':<10} {'Mean':<15} {'Std':<10}")
        print("-" * 35)
        
        for depth in depths:
            mean, std = run_test(dim=dim, depth=depth)
            print(f"{depth:<10} {mean:<15.4f} {std:<10.4f}")

if __name__ == "__main__":
    main()
