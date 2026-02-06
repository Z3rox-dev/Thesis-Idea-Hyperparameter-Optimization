#!/usr/bin/env python3
"""Diagnostic: Measure LGS gradient distortion and test Budget hypothesis."""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from ParamSpace import FUNS

def true_gradient_sphere(x):
    """True gradient for Sphere function: f(x) = sum(x^2) -> grad = 2x"""
    return 2 * np.array(x)

def measure_gradient_quality(opt, func_obj):
    """
    Measure how well LGS models approximate the true gradient.
    Returns: avg_cosine_similarity, avg_norm_error
    """
    cosine_sims = []
    norm_errors = []
    
    for leaf in opt.leaves:
        if leaf.lgs_model is not None and leaf.lgs_model.get("grad") is not None:
            # LGS gradient (in normalized coordinates, scaled by y_std)
            # stored in lgs_model["grad"]
            
            widths = leaf.widths()
            # Avoid div by zero
            widths = np.where(widths < 1e-9, 1.0, widths)
            
            # g_real = g_norm / width
            est_grad = leaf.lgs_model["grad"] / widths
            
            # True gradient at center
            center = leaf.center()
            real_grad = true_gradient_sphere(center)
            
            # Cosine similarity
            norm_est = np.linalg.norm(est_grad)
            norm_real = np.linalg.norm(real_grad)
            
            if norm_est > 1e-9 and norm_real > 1e-9:
                cos_sim = np.dot(est_grad, real_grad) / (norm_est * norm_real)
                cosine_sims.append(cos_sim)
                
            # Norm error (relative)
            if norm_real > 1e-9:
                err = np.linalg.norm(est_grad - real_grad) / norm_real
                norm_errors.append(err)
                
    return (np.mean(cosine_sims) if cosine_sims else np.nan, 
            np.mean(norm_errors) if norm_errors else np.nan)

def run_test(dim: int, depth: int, budget: int, seed: int = 42):
    """Run optimization and track gradient quality."""
    
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        split_depth_max=depth,
    )
    
    # Run optimization
    def obj(x):
        return func_obj(x)
        
    opt.optimize(obj, budget=budget)
            
    # Final check
    final_cos, final_err = measure_gradient_quality(opt, func_obj)
    
    return {
        "best_y": opt.best_y,
        "final_cos_sim": final_cos,
        "n_leaves": len(opt.leaves),
        "mean_points_per_leaf": np.mean([l.n_trials for l in opt.leaves])
    }

def main():
    print("ALBA-Potential: Distortion Field Analysis")
    print("Hypothesis: Higher budget fixes high-depth distortion.")
    print("=" * 60)
    
    dim = 20
    # True gradient is only easy to compute for Sphere
    
    scenarios = [
        # (Depth, Budget)
        (4, 400),
        (4, 1600),
        (16, 400),
        (16, 1600),
        (16, 3200) # Extreme budget
    ]
    
    print(f"Function: Sphere {dim}D")
    print(f"{'Depth':<6} {'Budget':<8} {'Best Y':<12} {'CosSim':<8} {'Leaves':<6} {'Pts/Leaf':<8}")
    print("-" * 60)
    
    for depth, budget in scenarios:
        # Run average of seeds
        ys = []
        coss = []
        pts = []
        
        for seed in [42]: # Single seed for speed, usually consistent on Sphere
            res = run_test(dim, depth, budget, seed)
            ys.append(res['best_y'])
            coss.append(res['final_cos_sim'])
            pts.append(res['mean_points_per_leaf'])
            
        print(f"{depth:<6} {budget:<8} {np.mean(ys):<12.4f} {np.mean(coss):<8.4f} {res['n_leaves']:<6} {np.mean(pts):<8.1f}")

if __name__ == "__main__":
    main()
