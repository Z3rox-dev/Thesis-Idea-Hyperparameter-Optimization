#!/usr/bin/env python3
"""
Final bug hunting: cerca corner cases e edge cases.

Cose da verificare:
1. Bounds handling (clipping corretto?)
2. NaN/Inf handling
3. Empty history handling
4. Categorical dimensions (se presenti)
5. Maximization vs Minimization consistency
6. Seed reproducibility
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler

def make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))


def test_seed_reproducibility():
    """Same seed should give IDENTICAL results"""
    print("=" * 60)
    print("TEST: Seed Reproducibility")
    print("=" * 60)
    
    dim = 5
    budget = 50
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    results = []
    for trial in range(3):
        sampler = CovarianceLocalSearchSampler()
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42  # SAME seed
        )
        _, val = opt.optimize(func, budget)
        results.append(val)
        print(f"  Trial {trial}: {val}")
    
    if len(set(results)) == 1:
        print("✅ PASS: All trials identical")
    else:
        print(f"⚠️  FAIL: Results differ! {results}")


def test_bounds_clipping():
    """Samples should never exceed bounds"""
    print("\n" + "=" * 60)
    print("TEST: Bounds Clipping")
    print("=" * 60)
    
    dim = 5
    budget = 100
    bounds = [(-2.0, 2.0)] * dim  # Narrow bounds
    func = make_sphere(dim)
    
    sampler = CovarianceLocalSearchSampler()
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        seed=42
    )
    
    violations = 0
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        
        # Check bounds
        for j, (lo, hi) in enumerate(bounds):
            if x[j] < lo - 1e-10 or x[j] > hi + 1e-10:
                violations += 1
                print(f"  Violation at iter {i}, dim {j}: x={x[j]}, bounds=[{lo}, {hi}]")
    
    if violations == 0:
        print("✅ PASS: All samples within bounds")
    else:
        print(f"⚠️  FAIL: {violations} violations")


def test_nan_function():
    """Function that returns NaN sometimes - optimizer should handle it"""
    print("\n" + "=" * 60)
    print("TEST: NaN Function Handling")
    print("=" * 60)
    
    dim = 5
    budget = 50
    bounds = [(-5.0, 5.0)] * dim
    
    nan_count = 0
    def sphere_with_nan(x):
        nonlocal nan_count
        if np.random.random() < 0.1:  # 10% chance of NaN
            nan_count += 1
            return float('nan')
        return float(np.sum(np.array(x)**2))
    
    sampler = CovarianceLocalSearchSampler()
    
    try:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42
        )
        _, val = opt.optimize(sphere_with_nan, budget)
        
        print(f"  NaN returns: {nan_count}")
        print(f"  Final value: {val}")
        
        if np.isfinite(val):
            print("✅ PASS: Optimizer handled NaN gracefully")
        else:
            print("⚠️  FAIL: Final value is NaN/Inf")
    except Exception as e:
        print(f"⚠️  FAIL: Exception raised: {e}")


def test_inf_function():
    """Function that returns Inf sometimes"""
    print("\n" + "=" * 60)
    print("TEST: Inf Function Handling")
    print("=" * 60)
    
    dim = 5
    budget = 50
    bounds = [(-5.0, 5.0)] * dim
    
    inf_count = 0
    def sphere_with_inf(x):
        nonlocal inf_count
        if np.random.random() < 0.1:  # 10% chance of Inf
            inf_count += 1
            return float('inf')
        return float(np.sum(np.array(x)**2))
    
    sampler = CovarianceLocalSearchSampler()
    
    try:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42
        )
        _, val = opt.optimize(sphere_with_inf, budget)
        
        print(f"  Inf returns: {inf_count}")
        print(f"  Final value: {val}")
        
        if np.isfinite(val):
            print("✅ PASS: Optimizer handled Inf gracefully")
        else:
            print("⚠️  FAIL: Final value is NaN/Inf")
    except Exception as e:
        print(f"⚠️  FAIL: Exception raised: {e}")


def test_maximization_mode():
    """Test that maximize=True works correctly"""
    print("\n" + "=" * 60)
    print("TEST: Maximization Mode")
    print("=" * 60)
    
    dim = 5
    budget = 50
    bounds = [(-5.0, 5.0)] * dim
    
    # Negative sphere - maximum is at origin
    def neg_sphere(x):
        return -float(np.sum(np.array(x)**2))
    
    sampler = CovarianceLocalSearchSampler()
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        maximize=True,  # MAXIMIZE
        seed=42
    )
    _, val = opt.optimize(neg_sphere, budget)
    
    print(f"  Best value (maximizing): {val}")
    print(f"  Best x norm: {np.linalg.norm(opt.best_x):.4f}")
    
    # Should be close to 0 (since max of -x^2 is at x=0)
    if val > -1.0:  # Close to 0
        print("✅ PASS: Maximization found near-optimal")
    else:
        print(f"⚠️  FAIL: Expected ~0, got {val}")


def test_very_narrow_bounds():
    """Bounds that are almost a point"""
    print("\n" + "=" * 60)
    print("TEST: Very Narrow Bounds")
    print("=" * 60)
    
    dim = 5
    budget = 50
    # Very narrow bounds - almost a point at 1.0
    bounds = [(0.999, 1.001)] * dim
    func = make_sphere(dim)
    
    sampler = CovarianceLocalSearchSampler()
    
    try:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42
        )
        _, val = opt.optimize(func, budget)
        
        print(f"  Final value: {val}")
        print(f"  Best x: {opt.best_x}")
        
        # All x should be close to 1.0
        expected = dim * 1.0  # sphere at x=1
        if abs(val - expected) < 0.1:
            print("✅ PASS: Narrow bounds handled correctly")
        else:
            print(f"⚠️  FAIL: Expected ~{expected}, got {val}")
    except Exception as e:
        print(f"⚠️  FAIL: Exception: {e}")


def test_asymmetric_bounds():
    """Bounds that are not centered at 0"""
    print("\n" + "=" * 60)
    print("TEST: Asymmetric Bounds")
    print("=" * 60)
    
    dim = 5
    budget = 100
    # Optimum is at 0, but bounds are [-10, -1] - optimum is outside!
    bounds = [(-10.0, -1.0)] * dim
    func = make_sphere(dim)
    
    sampler = CovarianceLocalSearchSampler()
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        seed=42
    )
    _, val = opt.optimize(func, budget)
    
    print(f"  Final value: {val}")
    print(f"  Best x: {opt.best_x}")
    
    # Best should be at the boundary closest to 0, i.e., x = [-1, -1, -1, -1, -1]
    expected = dim * 1.0  # 5 * 1^2 = 5
    if val < expected * 1.5:  # Allow some tolerance
        print("✅ PASS: Found constrained optimum at boundary")
    else:
        print(f"⚠️  FAIL: Expected ~{expected}, got {val}")


def test_single_dimension():
    """1D optimization"""
    print("\n" + "=" * 60)
    print("TEST: Single Dimension (1D)")
    print("=" * 60)
    
    dim = 1
    budget = 50
    bounds = [(-5.0, 5.0)]
    func = make_sphere(dim)
    
    sampler = CovarianceLocalSearchSampler()
    
    try:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42
        )
        _, val = opt.optimize(func, budget)
        
        print(f"  Final value: {val}")
        print(f"  Best x: {opt.best_x}")
        
        if val < 1.0:
            print("✅ PASS: 1D optimization works")
        else:
            print(f"⚠️  FAIL: Expected <1.0, got {val}")
    except Exception as e:
        print(f"⚠️  FAIL: Exception: {e}")


def test_high_dimension():
    """50D optimization"""
    print("\n" + "=" * 60)
    print("TEST: High Dimension (50D)")
    print("=" * 60)
    
    dim = 50
    budget = 500
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    sampler = CovarianceLocalSearchSampler()
    
    try:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=42
        )
        _, val = opt.optimize(func, budget)
        
        print(f"  Final value: {val}")
        print(f"  Best x norm: {np.linalg.norm(opt.best_x):.4f}")
        
        # Very rough check - just ensure it doesn't crash
        if np.isfinite(val):
            print("✅ PASS: 50D optimization completed")
        else:
            print(f"⚠️  FAIL: Non-finite result")
    except Exception as e:
        print(f"⚠️  FAIL: Exception: {e}")


if __name__ == "__main__":
    test_seed_reproducibility()
    test_bounds_clipping()
    test_nan_function()
    test_inf_function()
    test_maximization_mode()
    test_very_narrow_bounds()
    test_asymmetric_bounds()
    test_single_dimension()
    test_high_dimension()
    
    print("\n" + "=" * 60)
    print("ALL EDGE CASE TESTS COMPLETE")
    print("=" * 60)
