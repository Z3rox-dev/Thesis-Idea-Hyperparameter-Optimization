#!/usr/bin/env python3
"""Diagnostic: measure potential field noise/stability over iterations."""

import sys
import numpy as np
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_potential.optimizer import ALBA
from ParamSpace import FUNS

def run_diagnostic(dim: int = 10, budget: int = 200, seed: int = 42):
    """Track potential changes over iterations."""
    
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    # Track: for each leaf, potential values over time
    potential_history = {}  # leaf_id -> list of (iter, potential)
    
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        coherence_update_interval=1,  # Force update every iteration
    )
    
    original_update = opt._coherence_tracker.update
    
    def patched_update(leaves, iteration, force=False):
        result = original_update(leaves, iteration, force=force)
        
        # Record potentials for all leaves
        for i, leaf in enumerate(leaves):
            leaf_id = id(leaf)
            pot = opt._coherence_tracker._cache.potentials.get(i, 0.5)
            
            if leaf_id not in potential_history:
                potential_history[leaf_id] = []
            potential_history[leaf_id].append((iteration, pot))
        
        return result
    
    opt._coherence_tracker.update = patched_update
    
    # Run
    def obj(x):
        return func_obj(x)
    
    best_x, best_y = opt.optimize(obj, budget=budget)
    
    # Analyze noise: how much does potential change between consecutive updates?
    deltas = []
    for leaf_id, history in potential_history.items():
        if len(history) > 1:
            for i in range(1, len(history)):
                delta = abs(history[i][1] - history[i-1][1])
                deltas.append(delta)
    
    return {
        "n_leaves_tracked": len(potential_history),
        "total_updates": len(deltas),
        "mean_delta": np.mean(deltas) if deltas else 0,
        "std_delta": np.std(deltas) if deltas else 0,
        "max_delta": np.max(deltas) if deltas else 0,
        "best_y": best_y,
    }

if __name__ == "__main__":
    print("Potential Field Stability Analysis")
    print("=" * 50)
    
    for dim in [10, 20]:
        print(f"\n--- {dim}D ---")
        result = run_diagnostic(dim=dim)
        print(f"Leaves tracked: {result['n_leaves_tracked']}")
        print(f"Mean potential delta: {result['mean_delta']:.4f}")
        print(f"Std potential delta:  {result['std_delta']:.4f}")
        print(f"Max potential delta:  {result['max_delta']:.4f}")
        print(f"Best score: {result['best_y']:.4f}")
