#!/usr/bin/env python3
"""Test impact of split density (split_trials_factor) on 20D performance."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_test(dim: int, factor: float, budget: int = 400, seeds: list = [42, 43, 44]):
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    results = []
    points_per_leaf = []
    
    for seed in seeds:
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
        results.append(best_y)
        points_per_leaf.append(np.mean([l.n_trials for l in opt.leaves]))
    
    return np.mean(results), np.std(results), np.mean(points_per_leaf)

def main():
    print("=" * 60)
    print("ALBA-Potential: Split Density Impact (Sphere 20D)")
    print("=" * 60)
    
    dim = 20
    factors = [3.0, 6.0, 10.0, 15.0]
    
    print(f"{'Factor':<8} {'Req.Pts':<8} {'Mean Y':<12} {'Std Y':<8} {'Pts/Leaf':<8}")
    print("-" * 50)
    
    for factor in factors:
        req_pts = factor * dim + 6
        mean, std, pts = run_test(dim=dim, factor=factor)
        print(f"{factor:<8.1f} {req_pts:<8.1f} {mean:<12.4f} {std:<8.4f} {pts:<8.1f}")

if __name__ == "__main__":
    main()
