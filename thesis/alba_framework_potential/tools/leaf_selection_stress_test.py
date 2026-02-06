#!/usr/bin/env python3
"""
LEAF_SELECTION.PY STRESS TEST

Test per trovare bug nel modulo leaf_selection.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
import traceback

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector, PotentialAwareLeafSelector
from alba_framework_potential.cube import Cube

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
# TEST 1: Empty leaves
# ============================================================================
def test_empty_leaves():
    """select() con lista vuota."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    try:
        selector.select([], 3, False, rng)
        print("  ❌ Dovrebbe sollevare ValueError")
        return False
    except ValueError:
        print("  ✓ ValueError con lista vuota")
        return True

# ============================================================================
# TEST 2: Cube con good_ratio = NaN
# ============================================================================
def test_nan_good_ratio():
    """select() con cubo che ha good_ratio NaN."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    # Crea cubi con stato patologico
    cubes = []
    for i in range(3):
        c = Cube([(0, 1), (0, 1), (0, 1)])
        # Simula stato interno
        c.n_trials = 10
        c.n_good = 5 if i != 1 else float('nan')  # Un cubo con n_good = NaN
        cubes.append(c)
    
    try:
        for _ in range(10):
            selected = selector.select(cubes, 3, False, rng)
        print("  ✓ select() gestisce good_ratio edge cases")
        return True
    except Exception as e:
        bugs_found.append(f"2: Exception con good_ratio patologico: {e}")
        print(f"  ❌ Exception: {e}")
        return False

# ============================================================================
# TEST 3: Tutti scores identici
# ============================================================================
def test_identical_scores():
    """select() con tutti scores uguali."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    cubes = []
    for i in range(5):
        c = Cube([(0, 1), (0, 1)])
        c.n_trials = 10
        c.n_good = 5  # Tutti uguali
        cubes.append(c)
    
    selections = []
    for _ in range(100):
        selected = selector.select(cubes, 2, False, rng)
        selections.append(cubes.index(selected))
    
    # Dovrebbe distribuire uniformemente
    counts = [selections.count(i) for i in range(5)]
    print(f"  Counts: {counts}")
    
    if max(counts) > 50:  # Troppo sbilanciato
        print("  ⚠ Distribuzione sbilanciata")
    else:
        print("  ✓ Distribuzione ragionevole")
    
    return True

# ============================================================================
# TEST 4: Score molto diversi (overflow in exp)
# ============================================================================
def test_extreme_scores():
    """select() con scores molto diversi."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    cubes = []
    # Un cubo con molti trials (bassa exploration), altri con pochi
    for i in range(3):
        c = Cube([(0, 1), (0, 1)])
        c.n_trials = 1 if i == 0 else 1000000  # Huge difference
        c.n_good = 1 if i == 0 else 500000
        cubes.append(c)
    
    for _ in range(10):
        selected = selector.select(cubes, 2, False, rng)
    
    print("  ✓ select() gestisce scores estremi")
    return True

# ============================================================================
# TEST 5: Un solo cubo
# ============================================================================
def test_single_cube():
    """select() con un solo cubo."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    c = Cube([(0, 1), (0, 1)])
    c.n_trials = 5
    c.n_good = 2
    
    selected = selector.select([c], 2, False, rng)
    if selected is not c:
        bugs_found.append("5: Con un solo cubo dovrebbe restituire quello")
        print("  ❌ Non ha restituito l'unico cubo")
        return False
    
    print("  ✓ select() con un solo cubo")
    return True

# ============================================================================
# TEST 6: PotentialAwareLeafSelector senza tracker
# ============================================================================
def test_potential_no_tracker():
    """PotentialAwareLeafSelector senza tracker."""
    rng = np.random.default_rng(42)
    selector = PotentialAwareLeafSelector()
    
    cubes = []
    for i in range(3):
        c = Cube([(0, 1), (0, 1)])
        c.n_trials = 5
        c.n_good = 2
        cubes.append(c)
    
    for _ in range(10):
        selected = selector.select(cubes, 2, False, rng)
    
    print("  ✓ PotentialAwareLeafSelector funziona senza tracker")
    return True

# ============================================================================
# TEST 7: Cubo con n_trials = 0
# ============================================================================
def test_zero_trials():
    """select() con cubo che ha n_trials = 0."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    cubes = []
    for i in range(3):
        c = Cube([(0, 1), (0, 1)])
        c.n_trials = 0  # Nessun trial!
        c.n_good = 0
        cubes.append(c)
    
    for _ in range(10):
        selected = selector.select(cubes, 2, False, rng)
    
    print("  ✓ select() gestisce n_trials = 0")
    return True

# ============================================================================
# TEST 8: Molti cubi
# ============================================================================
def test_many_cubes():
    """select() con molti cubi."""
    rng = np.random.default_rng(42)
    selector = UCBSoftmaxLeafSelector()
    
    cubes = []
    for i in range(100):
        c = Cube([(0, 1), (0, 1)])
        c.n_trials = rng.integers(1, 50)
        c.n_good = rng.integers(0, c.n_trials + 1)
        cubes.append(c)
    
    for _ in range(50):
        selected = selector.select(cubes, 2, False, rng)
    
    print("  ✓ select() funziona con 100 cubi")
    return True

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LEAF_SELECTION.PY STRESS TESTS")
    print("=" * 70)
    
    tests = [
        ("TEST 1: Empty leaves", test_empty_leaves),
        ("TEST 2: NaN good_ratio", test_nan_good_ratio),
        ("TEST 3: Identical scores", test_identical_scores),
        ("TEST 4: Extreme scores", test_extreme_scores),
        ("TEST 5: Single cube", test_single_cube),
        ("TEST 6: PotentialAware no tracker", test_potential_no_tracker),
        ("TEST 7: Zero trials", test_zero_trials),
        ("TEST 8: Many cubes", test_many_cubes),
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
        print("\n✓ NESSUN BUG TROVATO in leaf_selection.py!")
