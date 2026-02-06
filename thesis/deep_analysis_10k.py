#!/usr/bin/env python3
"""Deep Analysis: ALBA-Potential (10k Budget) vs Random Search."""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
import time
import matplotlib.pyplot as plt # Though we won't plot, we might save data
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.coherence import CoherenceTracker, _build_knn_graph
from ParamSpace import FUNS

# Random Search Implementation
class RandomSearch:
    def __init__(self, bounds, seed):
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.dim = len(bounds)
        self.best_y = np.inf
        
    def optimize(self, func, budget):
        history = []
        for _ in range(budget):
            x = self.rng.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            y = func(x)
            if y < self.best_y:
                self.best_y = y
            history.append(self.best_y)
        return self.best_y, history

def analyze_lgs_quality(opt, func_obj):
    """Compute cosine similarity of LGS gradients vs True Gradient (Sphere only)."""
    cosine_sims = []
    
    # Needs true gradient function (Sphere: grad = 2x)
    def true_grad(x): return 2 * x
    
    for leaf in opt.leaves:
        if leaf.lgs_model is not None and leaf.lgs_model.get("grad") is not None:
            # Reconstruct real gradient from normalized one
            widths = leaf.widths()
            widths = np.where(widths < 1e-9, 1.0, widths)
            grad_est = leaf.lgs_model["grad"] / widths
            
            center = leaf.center()
            grad_true = true_grad(center)
            
            # Cosine sim
            norm_est = np.linalg.norm(grad_est)
            norm_true = np.linalg.norm(grad_true)
            if norm_est > 1e-9 and norm_true > 1e-9:
                sim = np.dot(grad_est, grad_true) / (norm_est * norm_true)
                cosine_sims.append(sim)
                
    return np.mean(cosine_sims) if cosine_sims else np.nan

def run_deep_analysis():
    print("=" * 80)
    print("ALBA-Potential: Deep Analysis @ 2,000 Budget")
    print("Target: Sphere 5D (Low Dim Noise Check)")
    print("Config: High Density (Factor=6.0)")
    print("=" * 80)
    
    dim = 5
    budget = 2000
    seed = 42
    
    func_obj, default_bounds = FUNS["sphere"]
    bounds = [default_bounds[0]] * dim
    
    # 1. Run Random Search Baseline
    print("Running Random Search (Baseline)...")
    rs = RandomSearch(bounds, seed)
    start_t = time.time()
    rs_best, rs_hist = rs.optimize(lambda x: func_obj(x), budget)
    print(f"Random Search Best: {rs_best:.4f} (Time: {time.time()-start_t:.2f}s)")
    
    # 2. Run ALBA-Potential
    print("\nRunning ALBA-Potential...")
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True,
        split_trials_factor=6.0, # High Density
        split_depth_max=16 # Let it grow deep if it wants, limited by factor
    )
    
    # Instrumentation
    stats_log = []
    
    def tracked_obj(x):
        return func_obj(x)
        
    start_t = time.time()
    
    # We run in chunks to log internal state
    chunk_size = 500
    n_chunks = budget // chunk_size
    
    # opt._initialize_root_cube() removed (handled in __init__) 
    # Actually opt.optimize handles init. We need to hook or use manual loop.
    # Manual loop is safer for introspection.
    
    # Manual init
    # opt.total_budget is set in init
    
    # Use standard loop structure
    for i in range(budget):
        if i < 10:
            x = opt.rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        else:
            x = opt.ask()
            
        y = tracked_obj(x)
        opt.tell(x, y)
        
        if (i + 1) % chunk_size == 0:
            # Collect stats
            n_leaves = len(opt.leaves)
            grad_quality = analyze_lgs_quality(opt, func_obj)
            
            # Graph stats
            coherence = opt._coherence_tracker
            edges = _build_knn_graph(opt.leaves, k=6)
            n_edges = len(edges)
            avg_degree = n_edges / max(1, n_leaves)
            
            # Potential stats
            pot_variance = 0.0
            if coherence._cache.potentials:
                pots = list(coherence._cache.potentials.values())
                pot_variance = np.var(pots)
                
            stats_log.append({
                "iter": i+1,
                "best_y": opt.best_y,
                "n_leaves": n_leaves,
                "grad_quality": grad_quality,
                "avg_degree": avg_degree,
                "pot_variance": pot_variance
            })
            print(f"Iter {i+1:5d} | Best: {opt.best_y:.4f} | Leaves: {n_leaves:3d} | GradCos: {grad_quality:.2f} | PotVar: {pot_variance:.4f}")

    print(f"\nALBA Best: {opt.best_y:.4f} (Time: {time.time()-start_t:.2f}s)")
    
    # Final Report
    print("\n" + "="*80)
    print("FINAL ANALYSIS REPORT")
    print("="*80)
    print(f"Random Search: {rs_best:.4f}")
    print(f"ALBA-Potential: {opt.best_y:.4f}")
    
    print("\nEvolution of Internal State:")
    print(f"{'Iter':<8} {'Leaves':<8} {'GradCos':<10} {'Degree':<8} {'PotVar':<10}")
    print("-" * 50)
    for s in stats_log[::2]: # Print every 1000 iters
        print(f"{s['iter']:<8} {s['n_leaves']:<8} {s['grad_quality']:<10.2f} {s['avg_degree']:<8.1f} {s['pot_variance']:<10.4f}")

    # Conclusion check
    if opt.best_y < rs_best:
        print("\n✅ ALBA Wins.")
        if stats_log[-1]['grad_quality'] > 0.5:
            print("✅ Gradients are reliable (>0.5).")
        else:
            print("⚠️ Gradients are noisy but ALBA managed.")
    else:
        print("\n❌ Random Search Wins.")

if __name__ == "__main__":
    run_deep_analysis()
