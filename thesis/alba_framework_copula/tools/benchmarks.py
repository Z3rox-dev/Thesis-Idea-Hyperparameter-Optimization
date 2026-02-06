
import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple

# Add workspace root to path to find ParamSpace
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

try:
    from ParamSpace import FUNS
except ImportError:
    # Fallback if ParamSpace not found
    print("Warning: ParamSpace not found. Using local function definitions.")
    FUNS = {}

from alba_framework_potential.optimizer import ALBA

def get_function(name: str, dim: int):
    """Get function object and bounds."""
    if name in FUNS:
        func, bounds_1d = FUNS[name]
        bounds = [bounds_1d[0]] * dim
        return func, bounds
    
    # Fallback definitions
    if name == "sphere":
        def sphere(x): return np.sum(np.array(x)**2)
        return sphere, [(-5.0, 5.0)] * dim
    elif name == "rosenbrock":
        def rosenbrock(x):
            x = np.array(x)
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        return rosenbrock, [(-5.0, 10.0)] * dim
    else:
        raise ValueError(f"Unknown function: {name}")

class RandomSearch:
    def __init__(self, bounds, seed):
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        
    def optimize(self, func, budget):
        best_y = np.inf
        for _ in range(budget):
            x = self.rng.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            y = func(x)
            if y < best_y:
                best_y = y
        return best_y

def run_benchmark_suite():
    print("=" * 80)
    print("ALBA-Potential: Validation Benchmark Suite")
    print("Verifying Auto-Scaling Laws implementation")
    print("=" * 80)
    
    tasks = [
        ("sphere", 10, 500),
        ("sphere", 20, 1000), # Higher budget for 20D to verify scaling
        ("rosenbrock", 10, 1000),
        ("rosenbrock", 20, 2000),
    ]
    
    results = []
    
    for name, dim, budget in tasks:
        print(f"\nTask: {name} {dim}D (Budget: {budget})")
        
        func, bounds = get_function(name, dim)
        
        # 1. Random Search (Baseline)
        rs = RandomSearch(bounds, seed=42)
        start_t = time.time()
        rs_score = rs.optimize(func, budget)
        rs_time = time.time() - start_t
        print(f"  > Random Search: {rs_score:.4f} ({rs_time:.2f}s)")
        
        # 2. ALBA (Auto-Configured)
        alba = ALBA(
            bounds=bounds,
            maximize=False,
            seed=42,
            total_budget=budget,
            use_potential_field=True
            # split inputs are None, so they will be auto-configured
        )
        
        # Verify Auto-Configuration
        print(f"    [Auto-Config] Factor: {alba._split_trials_factor:.2f} | MaxDepth: {alba._split_depth_max}")
        
        start_t = time.time()
        alba.optimize(func, budget)
        alba_score = alba.best_y
        alba_time = time.time() - start_t
        print(f"  > ALBA-Potential: {alba_score:.4f} ({alba_time:.2f}s)")
        
        improvement = 0.0
        if rs_score > 1e-9:
             improvement = (rs_score - alba_score) / rs_score * 100
        
        results.append({
            "task": f"{name}-{dim}D",
            "rs": rs_score,
            "alba": alba_score,
            "imp": improvement,
            "factor": alba._split_trials_factor,
            "depth": alba._split_depth_max
        })
        
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"{'Task':<15} {'RS Score':<12} {'ALBA Score':<12} {'Improv %':<10} {'Factor':<8} {'Depth':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['task']:<15} {r['rs']:<12.4f} {r['alba']:<12.4f} {r['imp']:<10.1f} {r['factor']:<8.2f} {r['depth']:<8d}")
    print("-" * 75)

if __name__ == "__main__":
    run_benchmark_suite()
