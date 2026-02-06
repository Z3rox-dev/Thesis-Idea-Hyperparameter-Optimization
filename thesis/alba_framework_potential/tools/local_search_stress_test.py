#!/usr/bin/env python3
"""
LOCAL_SEARCH.PY STRESS TEST

Test aggressivi per trovare bug nel modulo local_search.
Focus: NaN in history, covariance degenerata, alta dimensionalità.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
import traceback

from alba_framework_potential.local_search import (
    GaussianLocalSearchSampler,
    CovarianceLocalSearchSampler
)

bugs_found = []

def run_test(name, test_fn):
    print(f"\n{name}")
    print("-" * 50)
    try:
        return test_fn()
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# TEST 1: best_x = None
# ============================================================================
def test_best_x_none():
    """sample() con best_x = None."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1), (0, 1)]
    widths = np.array([1.0, 1.0, 1.0])
    
    for sampler in [GaussianLocalSearchSampler(), CovarianceLocalSearchSampler()]:
        name = sampler.__class__.__name__
        x = sampler.sample(None, bounds, widths, 0.5, rng)
        if np.any(np.isnan(x)):
            bugs_found.append(f"1: {name} produce NaN con best_x=None")
            print(f"  ❌ {name}: NaN")
            return False
        print(f"  ✓ {name} gestisce best_x=None")
    
    return True

# ============================================================================
# TEST 2: best_x con NaN
# ============================================================================
def test_best_x_nan():
    """sample() con best_x contenente NaN."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1), (0, 1)]
    widths = np.array([1.0, 1.0, 1.0])
    best_x_nan = np.array([0.5, np.nan, 0.5])
    
    for sampler in [GaussianLocalSearchSampler(), CovarianceLocalSearchSampler()]:
        name = sampler.__class__.__name__
        x = sampler.sample(best_x_nan, bounds, widths, 0.5, rng)
        if np.any(np.isnan(x)):
            bugs_found.append(f"2: {name} propaga NaN da best_x")
            print(f"  ❌ {name}: NaN propagato: {x}")
            return False
        print(f"  ✓ {name} con best_x NaN → {x}")
    
    return True

# ============================================================================
# TEST 3: y_history con NaN
# ============================================================================
def test_y_history_nan():
    """CovarianceLocalSearchSampler con NaN in y_history."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([1.0, 1.0])
    best_x = np.array([0.5, 0.5])
    
    # History con NaN
    X_history = [np.random.rand(2) for _ in range(20)]
    y_history = [float(i) for i in range(20)]
    y_history[5] = np.nan  # Un NaN
    y_history[10] = np.inf  # Un Inf
    
    sampler = CovarianceLocalSearchSampler()
    
    for i in range(10):
        x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_history, y_history)
        if np.any(np.isnan(x)):
            bugs_found.append(f"3: NaN con y_history contenente NaN")
            print(f"  ❌ NaN a iterazione {i}: {x}")
            return False
    
    print("  ✓ CovarianceLocalSearchSampler gestisce NaN in y_history")
    return True

# ============================================================================
# TEST 4: X_history con NaN
# ============================================================================
def test_x_history_nan():
    """CovarianceLocalSearchSampler con NaN in X_history."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([1.0, 1.0])
    best_x = np.array([0.5, 0.5])
    
    # History con NaN in X
    X_history = [np.random.rand(2) for _ in range(20)]
    X_history[3] = np.array([np.nan, 0.5])
    X_history[7] = np.array([0.5, np.inf])
    y_history = [float(i) for i in range(20)]
    
    sampler = CovarianceLocalSearchSampler()
    
    for i in range(10):
        x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_history, y_history)
        if np.any(np.isnan(x)):
            bugs_found.append(f"4: NaN con X_history contenente NaN")
            print(f"  ❌ NaN a iterazione {i}: {x}")
            return False
    
    print("  ✓ CovarianceLocalSearchSampler gestisce NaN in X_history")
    return True

# ============================================================================
# TEST 5: Punti tutti uguali (covariance singolare)
# ============================================================================
def test_singular_covariance():
    """Covariance singolare con punti identici."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([1.0, 1.0])
    best_x = np.array([0.5, 0.5])
    
    # Tutti punti identici!
    X_history = [np.array([0.5, 0.5]) for _ in range(20)]
    y_history = [float(i) for i in range(20)]
    
    sampler = CovarianceLocalSearchSampler()
    
    for i in range(10):
        x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_history, y_history)
        if np.any(np.isnan(x)):
            bugs_found.append(f"5: NaN con covariance singolare")
            print(f"  ❌ NaN a iterazione {i}: {x}")
            return False
    
    print("  ✓ Gestisce covariance singolare (punti identici)")
    return True

# ============================================================================
# TEST 6: Punti collineari
# ============================================================================
def test_collinear_points():
    """Covariance rank-deficient con punti collineari."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1), (0, 1)]
    widths = np.array([1.0, 1.0, 1.0])
    best_x = np.array([0.5, 0.5, 0.5])
    
    # Punti collineari: x2 = x1 sempre
    X_history = []
    for i in range(20):
        t = i / 20
        X_history.append(np.array([t, t, 0.5]))  # x0 = x1, x2 = costante
    y_history = [float(i) for i in range(20)]
    
    sampler = CovarianceLocalSearchSampler()
    
    for i in range(10):
        x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_history, y_history)
        if np.any(np.isnan(x)):
            bugs_found.append(f"6: NaN con punti collineari")
            print(f"  ❌ NaN a iterazione {i}: {x}")
            return False
    
    print("  ✓ Gestisce punti collineari (rank-deficient)")
    return True

# ============================================================================
# TEST 7: Alta dimensionalità
# ============================================================================
def test_high_dim():
    """Test in alta dimensionalità."""
    rng = np.random.default_rng(42)
    
    for dim in [20, 50]:
        bounds = [(0, 1)] * dim
        widths = np.ones(dim)
        best_x = np.full(dim, 0.5)
        
        X_history = [np.random.rand(dim) for _ in range(100)]
        y_history = [np.random.randn() for _ in range(100)]
        
        sampler = CovarianceLocalSearchSampler()
        
        for i in range(10):
            x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_history, y_history)
            if np.any(np.isnan(x)):
                bugs_found.append(f"7: NaN in {dim}D")
                print(f"  ❌ NaN in {dim}D a iterazione {i}")
                return False
        
        print(f"  ✓ {dim}D OK")
    
    return True

# ============================================================================
# TEST 8: Progress estremi
# ============================================================================
def test_extreme_progress():
    """progress fuori [0,1]."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([1.0, 1.0])
    best_x = np.array([0.5, 0.5])
    
    for progress in [-1.0, 2.0, np.nan, np.inf]:
        sampler = GaussianLocalSearchSampler()
        try:
            x = sampler.sample(best_x, bounds, widths, progress, rng)
            if np.any(np.isnan(x)):
                bugs_found.append(f"8: NaN con progress={progress}")
                print(f"  ❌ progress={progress} → NaN")
                return False
            print(f"  progress={progress} → OK")
        except:
            print(f"  progress={progress} → Exception (accettabile)")
    
    return True

# ============================================================================
# TEST 9: widths = 0
# ============================================================================
def test_zero_widths():
    """global_widths con zeri."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([0.0, 1.0])  # Una dimensione con width=0
    best_x = np.array([0.5, 0.5])
    
    sampler = GaussianLocalSearchSampler()
    
    for i in range(10):
        x = sampler.sample(best_x, bounds, widths, 0.5, rng)
        if np.any(np.isnan(x)):
            bugs_found.append(f"9: NaN con width=0")
            print(f"  ❌ NaN")
            return False
    
    print("  ✓ Gestisce widths con zeri")
    return True

# ============================================================================
# TEST 10: Empty history
# ============================================================================
def test_empty_history():
    """History vuota o troppo corta."""
    rng = np.random.default_rng(42)
    bounds = [(0, 1), (0, 1)]
    widths = np.array([1.0, 1.0])
    best_x = np.array([0.5, 0.5])
    
    sampler = CovarianceLocalSearchSampler()
    
    # Empty
    x = sampler.sample(best_x, bounds, widths, 0.5, rng, [], [])
    if np.any(np.isnan(x)):
        bugs_found.append("10: NaN con history vuota")
        print("  ❌ NaN con history vuota")
        return False
    
    # Troppo corta
    X_hist = [np.random.rand(2) for _ in range(3)]
    y_hist = [1.0, 2.0, 3.0]
    x = sampler.sample(best_x, bounds, widths, 0.5, rng, X_hist, y_hist)
    if np.any(np.isnan(x)):
        bugs_found.append("10: NaN con history corta")
        print("  ❌ NaN con history corta")
        return False
    
    print("  ✓ Gestisce history vuota/corta")
    return True

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LOCAL_SEARCH.PY STRESS TESTS")
    print("=" * 70)
    
    tests = [
        ("TEST 1: best_x = None", test_best_x_none),
        ("TEST 2: best_x con NaN", test_best_x_nan),
        ("TEST 3: y_history con NaN", test_y_history_nan),
        ("TEST 4: X_history con NaN", test_x_history_nan),
        ("TEST 5: Covariance singolare", test_singular_covariance),
        ("TEST 6: Punti collineari", test_collinear_points),
        ("TEST 7: Alta dimensionalità", test_high_dim),
        ("TEST 8: Progress estremi", test_extreme_progress),
        ("TEST 9: widths = 0", test_zero_widths),
        ("TEST 10: History vuota", test_empty_history),
    ]
    
    results = []
    for name, test_fn in tests:
        result = run_test(name, test_fn)
        results.append((name, result))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✓" if result else "❌"
        print(f"  {status} {name.split(':')[0]}")
    
    if bugs_found:
        print(f"\n❌ BUGS TROVATI: {len(bugs_found)}")
        for bug in bugs_found:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in local_search.py!")
