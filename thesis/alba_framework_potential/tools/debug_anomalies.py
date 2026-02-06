#!/usr/bin/env python3
"""
Debug script per investigare anomalie nei benchmark.

Anomalie sospette:
1. ALBA_Drill peggiore di ALBA base su Ellipsoid
2. Varianza alta di ALBA_Cov su Rosenbrock
3. Drilling sembra disattivato o non funziona correttamente?
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler

# Test functions
def make_ellipsoid(dim):
    weights = np.array([10**(6 * i/(dim-1)) for i in range(dim)])
    return lambda x: float(np.sum(weights * np.array(x)**2))

def make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))

def debug_drilling_behavior():
    """
    Bug Hypothesis #1: Drilling non si attiva o usa troppo budget in modo inutile
    """
    print("=" * 60)
    print("DEBUG #1: Drilling Behavior")
    print("=" * 60)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_ellipsoid(dim)
    
    np.random.seed(42)
    
    # Test ALBA senza drilling
    opt_no_drill = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_drilling=False,
        local_search_ratio=0.3,
        seed=42
    )
    
    _, val_no_drill = opt_no_drill.optimize(func, budget)
    print(f"ALBA (no drill): {val_no_drill:.2f}")
    
    # Test ALBA con drilling - aggiungo logging
    opt_with_drill = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_drilling=True,
        local_search_ratio=0.3,
        seed=42
    )
    
    # Track drilling
    drilling_triggered = 0
    drilling_budget_used = 0
    
    for i in range(budget):
        x = opt_with_drill.ask()
        y = func(x)
        opt_with_drill.tell(x, y)
        
        # Check if drilling is active
        if hasattr(opt_with_drill, 'driller') and opt_with_drill.driller is not None:
            drilling_triggered += 1
    
    print(f"ALBA_Drill result: {opt_with_drill.best_y:.2f}")
    print(f"Drilling iterations: {drilling_triggered}/{budget}")
    print(f"Drilling budget used: {getattr(opt_with_drill, 'drilling_budget_used', 'N/A')}")
    print(f"Drilling budget max: {getattr(opt_with_drill, 'drilling_budget_max', 'N/A')}")
    
    # Il drill peggiora? Perché?
    if val_no_drill < opt_with_drill.best_y:
        print(f"⚠️  BUG DETECTED: Drilling WORSE by {(opt_with_drill.best_y - val_no_drill)/val_no_drill*100:.1f}%")
    else:
        print(f"✅ Drilling improved by {(val_no_drill - opt_with_drill.best_y)/val_no_drill*100:.1f}%")


def debug_covariance_variance():
    """
    Bug Hypothesis #2: La covarianza ha varianza alta perché a volte fallisce
    """
    print("\n" + "=" * 60)
    print("DEBUG #2: Covariance Variance Analysis")
    print("=" * 60)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_rosenbrock(dim)
    
    results = []
    
    for seed in range(10):  # 10 runs
        cov_sampler = CovarianceLocalSearchSampler()
        
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            local_search_ratio=0.3,
            local_search_sampler=cov_sampler,
            use_drilling=False,
            seed=seed
        )
        
        _, val = opt.optimize(func, budget)
        results.append(val)
        print(f"  Seed {seed}: {val:.2f}")
    
    results = np.array(results)
    print(f"\nMean: {np.mean(results):.2f}, Std: {np.std(results):.2f}")
    print(f"Min: {np.min(results):.2f}, Max: {np.max(results):.2f}")
    print(f"CV (Coefficient of Variation): {np.std(results)/np.mean(results)*100:.1f}%")
    
    if np.std(results) / np.mean(results) > 0.5:
        print("⚠️  BUG SUSPECTED: CV > 50% indicates unstable behavior")


def debug_local_search_activation():
    """
    Bug Hypothesis #3: Local search non si attiva correttamente
    """
    print("\n" + "=" * 60)
    print("DEBUG #3: Local Search Activation Check")
    print("=" * 60)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_ellipsoid(dim)
    
    cov_sampler = CovarianceLocalSearchSampler()
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=cov_sampler,
        use_drilling=False,
        seed=42
    )
    
    global_samples = 0
    local_samples = 0
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        
        # Check what phase we're in
        if hasattr(opt, '_last_sample_type'):
            if opt._last_sample_type == 'local':
                local_samples += 1
            else:
                global_samples += 1
    
    # Check sampler state
    print(f"Total iterations: {budget}")
    print(f"Local search sampler call count: {getattr(cov_sampler, '_sample_count', 'not tracked')}")
    print(f"Expected local samples: ~{int(budget * 0.3)} (30% ratio)")
    
    # Check if covariance was ever computed
    if hasattr(cov_sampler, '_last_cov'):
        cov = cov_sampler._last_cov
        if cov is not None:
            cond = np.linalg.cond(cov)
            print(f"Last covariance condition number: {cond:.2f}")
        else:
            print("⚠️  Covariance never computed!")


def debug_drilling_implementation():
    """
    Bug Hypothesis #4: Drilling implementation issue
    """
    print("\n" + "=" * 60)
    print("DEBUG #4: Drilling Implementation Deep Dive")
    print("=" * 60)
    
    from alba_framework_potential.drilling import DrillingOptimizer
    
    dim = 10
    bounds = [(-5.0, 5.0)] * dim  # Must be list of tuples, not np.array
    
    # Simple center point
    center = np.zeros(dim)
    scale = 0.1
    
    driller = DrillingOptimizer(
        start_x=center,
        start_y=100.0,  # Some initial cost
        initial_sigma=scale,
        bounds=bounds
    )
    
    # Generate a few samples
    rng = np.random.default_rng(42)
    samples = [driller.ask(rng) for _ in range(10)]
    
    print(f"Driller samples (first 3):")
    for i, s in enumerate(samples[:3]):
        dist = np.linalg.norm(s - center)
        print(f"  Sample {i}: dist from center = {dist:.4f}")
    
    # Check if samples are within scale
    dists = [np.linalg.norm(s - center) for s in samples]
    print(f"Mean distance: {np.mean(dists):.4f} (expected ~{scale})")
    
    if np.mean(dists) > scale * 3:
        print("⚠️  BUG: Driller samples too far from center!")


def debug_potential_field_interference():
    """
    Bug Hypothesis #5: Potential field might interfere with local search
    """
    print("\n" + "=" * 60)
    print("DEBUG #5: Potential Field Effect on Local Search")
    print("=" * 60)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_ellipsoid(dim)
    
    # Test senza potential field
    opt_no_pf = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=False,
        local_search_ratio=0.3,
        seed=42
    )
    _, val_no_pf = opt_no_pf.optimize(func, budget)
    
    # Test con potential field
    opt_pf = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        seed=42
    )
    _, val_pf = opt_pf.optimize(func, budget)
    
    print(f"Without potential field: {val_no_pf:.2f}")
    print(f"With potential field: {val_pf:.2f}")
    
    if val_no_pf < val_pf:
        print(f"⚠️  Potential field HURTS by {(val_pf - val_no_pf)/val_no_pf*100:.1f}%")
    else:
        print(f"✅ Potential field helps by {(val_no_pf - val_pf)/val_no_pf*100:.1f}%")


def debug_all():
    debug_drilling_behavior()
    debug_covariance_variance()
    debug_local_search_activation()
    debug_drilling_implementation()
    debug_potential_field_interference()


if __name__ == "__main__":
    debug_all()
