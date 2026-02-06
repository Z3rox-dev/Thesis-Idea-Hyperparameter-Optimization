#!/usr/bin/env python3
"""
Diagnostic script to analyze ALBA-Potential behavior at different dimensions.

Investigates:
1. Number of valid edges in kNN graph
2. Gradient reliability (norm distribution)
3. Potential field variance
4. Edge weight distribution
"""

import sys
import numpy as np
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_diagnostic(dim: int, budget: int = 200, seed: int = 42):
    """Run ALBA-Potential and collect diagnostic info."""
    
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    # Patch to collect diagnostics
    diagnostics = {
        "dim": dim,
        "iterations": [],
        "n_leaves": [],
        "n_valid_edges": [],
        "n_total_possible_edges": [],
        "gradient_norms": [],
        "potential_variance": [],
        "coherence_scores": [],
    }
    
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    # Store original update method to wrap it
    original_update = opt._coherence_tracker.update
    
    def patched_update(leaves, iteration, force=False):
        result = original_update(leaves, iteration, force=force)
        
        # Collect diagnostics from cache
        cache = opt._coherence_tracker._cache
        
        diagnostics["iterations"].append(iteration)
        diagnostics["n_leaves"].append(len(leaves))
        
        # Count valid edges
        n_with_model = sum(1 for l in leaves if l.lgs_model is not None and l.lgs_model.get("grad") is not None)
        max_edges = n_with_model * (n_with_model - 1) if n_with_model > 1 else 0
        diagnostics["n_total_possible_edges"].append(max_edges)
        
        # Count actual edges from potentials (if any)
        n_valid = len([v for v in cache.potentials.values() if v != 0.5])
        diagnostics["n_valid_edges"].append(n_valid)
        
        # Gradient norms
        grads = [l.lgs_model.get("grad") for l in leaves if l.lgs_model is not None and l.lgs_model.get("grad") is not None]
        if grads:
            norms = [np.linalg.norm(g) for g in grads]
            diagnostics["gradient_norms"].append(np.mean(norms))
        else:
            diagnostics["gradient_norms"].append(0.0)
        
        # Potential variance
        if cache.potentials:
            pot_vals = list(cache.potentials.values())
            diagnostics["potential_variance"].append(np.var(pot_vals))
        else:
            diagnostics["potential_variance"].append(0.0)
        
        # Coherence scores
        if cache.scores:
            score_vals = list(cache.scores.values())
            diagnostics["coherence_scores"].append(np.mean(score_vals))
        else:
            diagnostics["coherence_scores"].append(0.5)
        
        return result
    
    opt._coherence_tracker.update = patched_update
    
    # Run optimization
    evals = 0
    def obj(x):
        nonlocal evals
        evals += 1
        return func_obj(x)
    
    best_x, best_y = opt.optimize(obj, budget=budget)
    
    return diagnostics, best_y

def main():
    print("=" * 60)
    print("ALBA-Potential High-Dimensional Diagnostic")
    print("=" * 60)
    
    for dim in [10, 20, 30]:
        print(f"\n--- Dimension: {dim} ---")
        diag, best_y = run_diagnostic(dim, budget=300)
        
        print(f"Best score: {best_y:.4f}")
        print(f"Final #leaves: {diag['n_leaves'][-1] if diag['n_leaves'] else 0}")
        print(f"Avg gradient norm: {np.mean(diag['gradient_norms']):.4f}")
        print(f"Avg potential variance: {np.mean(diag['potential_variance']):.6f}")
        print(f"Avg coherence: {np.mean(diag['coherence_scores']):.4f}")
        print(f"Valid edge ratio: {np.mean(diag['n_valid_edges']) / (np.mean(diag['n_leaves']) + 0.01):.2f}")

if __name__ == "__main__":
    main()
