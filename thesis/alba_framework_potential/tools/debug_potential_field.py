#!/usr/bin/env python3
"""
Deep debug del Potential Field che peggiora le performance.
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler

def make_ellipsoid(dim):
    weights = np.array([10**(6 * i/(dim-1)) for i in range(dim)])
    return lambda x: float(np.sum(weights * np.array(x)**2))

def make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))

def make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))


def test_potential_field_effect():
    """Test PF effect on multiple functions"""
    print("=" * 70)
    print("TEST: Potential Field Effect Across Functions")
    print("=" * 70)
    
    functions = {
        "Sphere": make_sphere,
        "Ellipsoid": make_ellipsoid,
        "Rosenbrock": make_rosenbrock,
    }
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    
    for fname, fmaker in functions.items():
        func = fmaker(dim)
        
        results_no_pf = []
        results_pf = []
        
        for seed in range(5):
            # Without PF
            opt_no_pf = ALBA(
                bounds=bounds,
                total_budget=budget,
                use_potential_field=False,
                local_search_ratio=0.3,
                seed=seed
            )
            _, val = opt_no_pf.optimize(func, budget)
            results_no_pf.append(val)
            
            # With PF
            opt_pf = ALBA(
                bounds=bounds,
                total_budget=budget,
                use_potential_field=True,
                local_search_ratio=0.3,
                seed=seed
            )
            _, val = opt_pf.optimize(func, budget)
            results_pf.append(val)
        
        mean_no_pf = np.mean(results_no_pf)
        mean_pf = np.mean(results_pf)
        
        diff = (mean_pf - mean_no_pf) / mean_no_pf * 100
        
        status = "✅ PF helps" if mean_pf < mean_no_pf else "⚠️  PF hurts"
        print(f"{fname:12s}: no_PF={mean_no_pf:10.2f}, PF={mean_pf:10.2f} | {diff:+.1f}% | {status}")


def test_pf_with_covariance():
    """Test if PF + Covariance creates issues"""
    print("\n" + "=" * 70)
    print("TEST: Potential Field + Covariance Interaction")
    print("=" * 70)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_ellipsoid(dim)
    
    configs = [
        ("no_PF + Gaussian", False, None),
        ("PF + Gaussian", True, None),
        ("no_PF + Cov", False, CovarianceLocalSearchSampler()),
        ("PF + Cov", True, CovarianceLocalSearchSampler()),
    ]
    
    for name, use_pf, sampler in configs:
        results = []
        for seed in range(5):
            # Recreate sampler for fresh state
            if sampler is not None:
                sampler = CovarianceLocalSearchSampler()
            
            opt = ALBA(
                bounds=bounds,
                total_budget=budget,
                use_potential_field=use_pf,
                local_search_ratio=0.3,
                local_search_sampler=sampler,
                seed=seed
            )
            _, val = opt.optimize(func, budget)
            results.append(val)
        
        print(f"{name:20s}: mean={np.mean(results):10.2f}, std={np.std(results):10.2f}")


def analyze_pf_mechanism():
    """Look at what PF actually does during optimization"""
    print("\n" + "=" * 70)
    print("TEST: Potential Field Mechanism Analysis")
    print("=" * 70)
    
    dim = 10
    budget = 100  # Shorter for detailed analysis
    bounds = [(-5.0, 5.0)] * dim
    func = make_ellipsoid(dim)
    
    # Run with PF and track what happens
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        seed=42
    )
    
    exploration_samples = 0
    local_samples = 0
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        
        # Try to track phase
        # Note: This depends on internal implementation
    
    # Check final state
    print(f"Best y found: {opt.best_y:.2f}")
    print(f"Best x norm: {np.linalg.norm(opt.best_x):.4f}")
    print(f"Iterations: {opt.t}")
    
    # Check gamma (threshold)
    print(f"Gamma (threshold): {opt.gamma:.2f}")
    
    # Check if PF pushed away from optimum
    # The optimum is at x=0
    final_dist = np.linalg.norm(opt.best_x)
    print(f"Distance from optimum: {final_dist:.4f}")


if __name__ == "__main__":
    test_potential_field_effect()
    test_pf_with_covariance()
    analyze_pf_mechanism()
