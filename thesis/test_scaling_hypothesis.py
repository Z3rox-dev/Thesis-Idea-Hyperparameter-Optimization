#!/usr/bin/env python3
"""Test Density-Budget Scaling Hypothesis."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
import time
from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_test(dim: int, factor: float, budget: int, seed: int):
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    start_t = time.time()
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
    duration = time.time() - start_t
    
    # Collect stats
    leaves = opt.leaves
    n_leaves = len(leaves)
    pts_per_leaf = np.mean([l.n_trials for l in leaves])
    
    return best_y, n_leaves, pts_per_leaf

def main():
    print("=" * 80)
    print("ALBA-Potential: Density-Budget Scaling Validation (Sphere 20D)")
    print("Hypothesis: High Budget needs High Factor to unlock performance.")
    print("=" * 80)
    
    dim = 20
    scenarios = [
        # (Factor, Budget, Label)
        (3.0, 400, "Base (Def)"),
        (3.0, 1600, "High Budget Only"), # Proved ineffective previously
        (10.0, 400, "High Factor Only"),
        (10.0, 1600, "High Budget + High Factor"), # The candidate winner
    ]
    
    print(f"{'Scenario':<25} {'Factor':<6} {'Budget':<6} {'Mean Y':<10} {'Std Y':<8} {'Leaves':<6} {'Pts/Leaf':<8}")
    print("-" * 80)
    
    seeds = [42, 43, 44]
    
    for factor, budget, label in scenarios:
        ys = []
        leaf_counts = []
        pts_counts = []
        
        for seed in seeds:
            y, n_l, pts = run_test(dim, factor, budget, seed)
            ys.append(y)
            leaf_counts.append(n_l)
            pts_counts.append(pts)
            
        print(f"{label:<25} {factor:<6.1f} {budget:<6} {np.mean(ys):<10.4f} {np.std(ys):<8.4f} {np.mean(leaf_counts):<6.1f} {np.mean(pts_counts):<8.1f}")

if __name__ == "__main__":
    main()
