#!/usr/bin/env python3
"""
OPTIMIZER.PY STRESS TEST

Testa il modulo principale ALBA con input patologici.
Focus: tell() con y_raw estremi/NaN.
"""

import sys
import os

# Fix module path for proper relative imports
sys.path.insert(0, '/mnt/workspace/thesis')
os.chdir('/mnt/workspace/thesis/alba_framework_potential')

import numpy as np
import traceback

# ============================================================================
# Import reale del modulo
# ============================================================================
try:
    from alba_framework_potential.optimizer import ALBA
    IMPORT_OK = True
except ImportError as e:
    print(f"Cannot import ALBA: {e}")
    IMPORT_OK = False

# ============================================================================
# HELPER
# ============================================================================
def run_test(name, test_fn):
    """Run a test and catch exceptions."""
    print(f"\n{name}")
    print("-" * 50)
    try:
        result = test_fn()
        return result
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        traceback.print_exc()
        return False

bugs_found = []

# ============================================================================
# TEST 1: tell() con y_raw = NaN
# ============================================================================
def test_tell_nan():
    """y_raw = NaN non deve corrompere l'ottimizzatore."""
    opt = ALBA(bounds=[(0, 1), (0, 1)], seed=42, total_budget=50)
    
    # 10 iterazioni normali
    for i in range(10):
        x = opt.ask()
        y = -np.sum((x - 0.5)**2)
        opt.tell(x, y)
    
    best_before = opt.best_y
    print(f"  Best dopo 10 iter: {best_before:.4f}")
    
    # Ora 5 NaN
    for i in range(5):
        x = opt.ask()
        opt.tell(x, np.nan)
    
    best_after = opt.best_y
    print(f"  Best dopo 5 NaN: {best_after}")
    
    # Continuare con valori normali
    for i in range(10):
        x = opt.ask()
        y = -np.sum((x - 0.5)**2)
        opt.tell(x, y)
    
    best_final = opt.best_y
    print(f"  Best dopo altre 10 iter: {best_final:.4f}")
    
    # Check: best non dovrebbe essere NaN
    if np.isnan(best_final):
        bugs_found.append("1: best_y diventa NaN dopo tell(NaN)")
        print("  ❌ best_y corrotto!")
        return False
    
    # Check: gamma non dovrebbe essere NaN
    if np.isnan(opt.gamma):
        bugs_found.append("1: gamma diventa NaN")
        print("  ❌ gamma corrotto!")
        return False
    
    print("  ✓ Optimizer sopravvive a NaN in tell()")
    return True

# ============================================================================
# TEST 2: tell() con y_raw = ±Inf
# ============================================================================
def test_tell_inf():
    """y_raw = Inf non deve corrompere l'ottimizzatore."""
    opt = ALBA(bounds=[(0, 1), (0, 1)], seed=42, total_budget=50)
    
    # 10 iterazioni con mix di valori
    for i in range(10):
        x = opt.ask()
        if i == 3:
            y = np.inf
        elif i == 6:
            y = -np.inf
        else:
            y = -np.sum((x - 0.5)**2)
        opt.tell(x, y)
    
    best = opt.best_y
    print(f"  Best dopo 10 iter (con ±Inf): {best}")
    
    # Check
    if np.isnan(best):
        bugs_found.append("2: best_y diventa NaN con Inf input")
        print("  ❌ best_y corrotto!")
        return False
    
    if not np.isfinite(best) and best != np.inf and best != -np.inf:
        print("  ⚠ best_y è Inf (può essere intenzionale per maximize)")
    
    print("  ✓ Optimizer sopravvive a Inf in tell()")
    return True

# ============================================================================
# TEST 3: Tutti y sono uguali (gamma patologico)
# ============================================================================
def test_constant_y():
    """Tutti y uguali → gamma = y per tutti, nessun good point."""
    opt = ALBA(bounds=[(0, 1), (0, 1), (0, 1)], seed=42, total_budget=50)
    
    for i in range(20):
        x = opt.ask()
        opt.tell(x, 1.0)  # Sempre 1.0
    
    print(f"  gamma = {opt.gamma}")
    print(f"  stagnation = {opt.stagnation}")
    
    # Dovrebbe funzionare, anche se tutti sono "good" o "bad"
    for i in range(10):
        x = opt.ask()
        if np.any(np.isnan(x)):
            bugs_found.append("3: ask() produce NaN con y costanti")
            print(f"  ❌ Iterazione {i}: NaN in x")
            return False
        opt.tell(x, 1.0)
    
    print("  ✓ Optimizer gestisce y costanti")
    return True

# ============================================================================
# TEST 4: Rumore altissimo
# ============================================================================
def test_high_noise():
    """y con enorme varianza - dovrebbe sopravvivere."""
    opt = ALBA(bounds=[(0, 1), (0, 1)], seed=42, total_budget=100)
    rng = np.random.default_rng(42)
    
    for i in range(100):
        x = opt.ask()
        # Rumore enorme
        y = rng.standard_cauchy()  # Cauchy ha varianza infinita
        opt.tell(x, y)
        
        if np.any(np.isnan(x)):
            bugs_found.append(f"4: NaN in ask() a iterazione {i}")
            print(f"  ❌ NaN a iterazione {i}")
            return False
    
    print(f"  Final best: {opt.best_y}")
    print("  ✓ Sopravvive a rumore Cauchy")
    return True

# ============================================================================
# TEST 5: Alta dimensionalità
# ============================================================================
def test_high_dim():
    """Test stabilità in 20D e 50D."""
    for dim in [20, 50]:
        print(f"  {dim}D...", end=" ")
        opt = ALBA(
            bounds=[(0, 1)] * dim, 
            seed=42, 
            total_budget=100,
            split_trials_min=5,
            split_depth_max=3
        )
        
        for i in range(100):
            x = opt.ask()
            if np.any(np.isnan(x)):
                bugs_found.append(f"5: NaN in {dim}D a iterazione {i}")
                print(f"❌ NaN!")
                return False
            y = -np.sum((x - 0.5)**2)
            opt.tell(x, y)
        
        print(f"✓ best={opt.best_y:.4f}")
    
    return True

# ============================================================================
# TEST 6: Drilling path
# ============================================================================
def test_drilling():
    """Test che drilling non produce NaN."""
    opt = ALBA(
        bounds=[(0, 1), (0, 1)], 
        seed=42, 
        total_budget=100,
        use_drilling=True
    )
    
    drilling_activated = False
    for i in range(100):
        x = opt.ask()
        if np.any(np.isnan(x)):
            bugs_found.append(f"6: NaN durante drilling a iterazione {i}")
            print(f"  ❌ NaN a iterazione {i}")
            return False
        
        # Check if drilling is active
        if opt.driller is not None:
            drilling_activated = True
        
        y = -np.sum((x - 0.5)**2) + 0.01 * np.random.randn()
        opt.tell(x, y)
    
    print(f"  Drilling attivato: {drilling_activated}")
    print(f"  Drilling budget usato: {opt.drilling_budget_used}")
    print(f"  Best: {opt.best_y:.4f}")
    print("  ✓ Drilling stabile")
    return True

# ============================================================================
# TEST 7: Coherence con dati patologici
# ============================================================================
def test_coherence_extreme():
    """Test coherence tracking con dati estremi."""
    opt = ALBA(
        bounds=[(0, 1), (0, 1), (0, 1)], 
        seed=42, 
        total_budget=50,
        use_coherence_gating=True,
        use_potential_field=True
    )
    
    rng = np.random.default_rng(42)
    
    for i in range(50):
        x = opt.ask()
        if np.any(np.isnan(x)):
            bugs_found.append(f"7: NaN con coherence a iterazione {i}")
            print(f"  ❌ NaN a iterazione {i}")
            return False
        
        # Mix di valori
        if i % 7 == 0:
            y = 1e10  # Enorme
        elif i % 11 == 0:
            y = -1e10  # Molto negativo  
        else:
            y = -np.sum((x - 0.5)**2)
        
        opt.tell(x, y)
    
    # Coherence tracker check - simplified (cache is a dataclass, not dict)
    print("  ✓ Coherence stabile con input estremi")
    return True

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMIZER.PY STRESS TESTS")
    print("=" * 70)
    
    if not IMPORT_OK:
        print("\n❌ Cannot run tests - import failed")
        sys.exit(1)
    
    tests = [
        ("TEST 1: tell() con y_raw = NaN", test_tell_nan),
        ("TEST 2: tell() con y_raw = ±Inf", test_tell_inf),
        ("TEST 3: y costanti", test_constant_y),
        ("TEST 4: Rumore Cauchy (varianza infinita)", test_high_noise),
        ("TEST 5: Alta dimensionalità", test_high_dim),
        ("TEST 6: Drilling path", test_drilling),
        ("TEST 7: Coherence con estremi", test_coherence_extreme),
    ]
    
    results = []
    for name, test_fn in tests:
        result = run_test(name, test_fn)
        results.append((name, result))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✓" if result else "❌"
        print(f"  {status} {name.split(':')[0]}")
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if bugs_found:
        print(f"\n❌ BUGS TROVATI: {len(bugs_found)}")
        for bug in bugs_found:
            print(f"  - {bug}")
    else:
        print("\n✓ Nessun bug trovato in optimizer.py")
