#!/usr/bin/env python3
"""
CATEGORICAL.PY STRESS TEST

Test aggressivi per trovare bug nel modulo categorical sampling.
Focus: divisioni per zero, NaN in score, edge cases discretizzazione.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
import traceback

from alba_framework_potential.categorical import CategoricalSampler

# Mock Cube for testing
class MockCube:
    def __init__(self):
        self.cat_stats = {}
        self.tested_pairs = []

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
# TEST 1: Discretizzazione con NaN
# ============================================================================
def test_discretize_nan():
    """discretize() con NaN input."""
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    
    # NaN
    try:
        result = sampler.discretize(np.nan, 3)
        print(f"  discretize(NaN, 3) = {result}")
        if np.isnan(result) or result < 0 or result >= 3:
            bugs_found.append("1: discretize(NaN) produce valore invalido")
            print("  ❌ Risultato invalido!")
            return False
    except:
        print("  ✓ discretize(NaN) solleva eccezione (OK)")
    
    # Inf
    try:
        result = sampler.discretize(np.inf, 3)
        print(f"  discretize(Inf, 3) = {result}")
        if result < 0 or result >= 3:
            bugs_found.append("1: discretize(Inf) fuori range")
            print("  ❌ Fuori range!")
            return False
    except:
        print("  ✓ discretize(Inf) solleva eccezione (OK)")
    
    # Valore negativo
    result = sampler.discretize(-0.5, 3)
    print(f"  discretize(-0.5, 3) = {result}")
    if result < 0:
        bugs_found.append("1: discretize(negativo) produce indice negativo")
        print("  ❌ Indice negativo!")
        return False
    
    print("  ✓ discretize gestisce edge cases")
    return True

# ============================================================================
# TEST 2: to_continuous edge cases
# ============================================================================
def test_to_continuous_edge():
    """to_continuous() con edge cases."""
    sampler = CategoricalSampler([(0, 3)])
    
    # n_choices = 1 (divisione per zero!)
    result = sampler.to_continuous(0, 1)
    print(f"  to_continuous(0, 1) = {result}")
    if np.isnan(result):
        bugs_found.append("2: to_continuous con n_choices=1 produce NaN")
        print("  ❌ NaN!")
        return False
    
    # indice negativo
    result = sampler.to_continuous(-1, 3)
    print(f"  to_continuous(-1, 3) = {result}")
    if result < 0 or result > 1:
        print(f"  ⚠ Fuori range [0,1]: {result}")
    
    print("  ✓ to_continuous gestisce edge cases")
    return True

# ============================================================================
# TEST 3: record_observation con NaN score
# ============================================================================
def test_record_nan_score():
    """record_observation() con score = NaN."""
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    
    x = np.array([0.5, 0.5])
    
    # Record normale
    sampler.record_observation(x, 1.0)
    print(f"  Elite dopo record(1.0): {len(sampler._elite_configs)}")
    
    # Record con NaN
    sampler.record_observation(x, np.nan)
    print(f"  Elite dopo record(NaN): {len(sampler._elite_configs)}")
    
    # Record con Inf
    sampler.record_observation(x, np.inf)
    print(f"  Elite dopo record(Inf): {len(sampler._elite_configs)}")
    
    # Verifica che elite_configs sia ordinabile
    try:
        sampler._elite_configs.sort(key=lambda p: p[1], reverse=True)
        print(f"  ✓ Elite configs ordinabili")
    except Exception as e:
        bugs_found.append(f"3: Elite configs non ordinabili con NaN: {e}")
        print(f"  ❌ Sort fallito: {e}")
        return False
    
    # Verifica ordine (NaN rompe il sorting)
    scores = [s for _, s in sampler._elite_configs]
    print(f"  Scores in elite: {scores}")
    
    return True

# ============================================================================
# TEST 4: elite_crossover con elite vuoto/singolo
# ============================================================================
def test_elite_crossover_edge():
    """elite_crossover() con casi limite."""
    rng = np.random.default_rng(42)
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    
    # Nessun elite
    result = sampler.elite_crossover(rng)
    print(f"  Crossover con 0 elite: {result}")
    if result is not None:
        bugs_found.append("4: Crossover con 0 elite dovrebbe essere None")
        return False
    
    # Un solo elite
    sampler.record_observation(np.array([0.5, 0.5]), 1.0)
    result = sampler.elite_crossover(rng)
    print(f"  Crossover con 1 elite: {result}")
    if result is not None:
        bugs_found.append("4: Crossover con 1 elite dovrebbe essere None")
        return False
    
    # Due elite
    sampler.record_observation(np.array([0.2, 0.8]), 2.0)
    result = sampler.elite_crossover(rng)
    print(f"  Crossover con 2 elite: {result}")
    if result is None:
        bugs_found.append("4: Crossover con 2 elite dovrebbe funzionare")
        return False
    
    print("  ✓ elite_crossover gestisce edge cases")
    return True

# ============================================================================
# TEST 5: sample() con cube vuoto
# ============================================================================
def test_sample_empty_cube():
    """sample() con cube senza statistiche."""
    rng = np.random.default_rng(42)
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    cube = MockCube()
    
    x = np.array([0.5, 0.5])
    
    for i in range(20):
        result = sampler.sample(x, cube, rng)
        if np.any(np.isnan(result)):
            bugs_found.append(f"5: sample() produce NaN a iterazione {i}")
            print(f"  ❌ NaN a iterazione {i}: {result}")
            return False
    
    print("  ✓ sample() funziona con cube vuoto")
    return True

# ============================================================================
# TEST 6: Thompson Sampling con statistiche estreme
# ============================================================================
def test_thompson_extreme_stats():
    """Thompson sampling con alpha/beta estremi."""
    rng = np.random.default_rng(42)
    sampler = CategoricalSampler([(0, 3)])
    cube = MockCube()
    
    # Statistiche estreme: una categoria con migliaia di successi
    cube.cat_stats = {
        0: {
            0: (10000, 10000),  # 100% success rate
            1: (0, 10000),      # 0% success rate
            2: (5000, 10000),   # 50% success rate
        }
    }
    
    x = np.array([0.5])
    
    for i in range(50):
        result = sampler.sample(x, cube, rng)
        if np.any(np.isnan(result)):
            bugs_found.append(f"6: NaN con statistiche estreme")
            print(f"  ❌ NaN: {result}")
            return False
    
    print("  ✓ Thompson sampling stabile con statistiche estreme")
    return True

# ============================================================================
# TEST 7: Molte dimensioni categoriche
# ============================================================================
def test_many_categorical_dims():
    """Test con molte dimensioni categoriche."""
    rng = np.random.default_rng(42)
    
    # 10 dimensioni categoriche, ciascuna con diversi n_choices
    cat_dims = [(i, 2 + i % 5) for i in range(10)]
    sampler = CategoricalSampler(cat_dims)
    cube = MockCube()
    
    x = np.random.rand(10)
    
    for i in range(100):
        # Popola elite
        sampler.record_observation(x, np.random.randn())
        
        result = sampler.sample(x, cube, rng)
        if np.any(np.isnan(result)):
            bugs_found.append(f"7: NaN con 10 dim categoriche")
            print(f"  ❌ NaN a iterazione {i}")
            return False
    
    print(f"  ✓ 100 sample con 10 dim categoriche")
    return True

# ============================================================================
# TEST 8: recompute_cube_cat_stats con NaN in tested_pairs
# ============================================================================
def test_recompute_with_nan():
    """recompute_cube_cat_stats() con NaN nei tested_pairs."""
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    cube = MockCube()
    
    # Aggiungi tested_pairs con NaN
    cube.tested_pairs = [
        (np.array([0.5, 0.5]), 1.0),
        (np.array([0.2, 0.8]), np.nan),  # NaN score
        (np.array([np.nan, 0.5]), 2.0),  # NaN in point
        (np.array([0.8, 0.2]), np.inf),  # Inf score
    ]
    
    try:
        sampler.recompute_cube_cat_stats(cube, gamma=0.5)
        print(f"  cat_stats dopo recompute: {cube.cat_stats}")
        
        # Verifica che non ci siano NaN nelle statistiche
        for dim_idx, stats in cube.cat_stats.items():
            for val_idx, (n_g, n_t) in stats.items():
                if np.isnan(n_g) or np.isnan(n_t):
                    bugs_found.append("8: NaN in cat_stats dopo recompute")
                    print(f"  ❌ NaN in stats[{dim_idx}][{val_idx}]")
                    return False
        
        print("  ✓ recompute gestisce NaN nei tested_pairs")
        return True
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

# ============================================================================
# TEST 9: get_cat_key con x contenente NaN
# ============================================================================
def test_get_cat_key_nan():
    """get_cat_key() con NaN in x."""
    sampler = CategoricalSampler([(0, 3), (1, 5)])
    
    x_nan = np.array([np.nan, 0.5])
    
    try:
        result = sampler.get_cat_key(x_nan)
        print(f"  get_cat_key([NaN, 0.5]) = {result}")
        
        # Verifica che il risultato sia usabile come chiave dict
        d = {}
        d[result] = 1
        print(f"  ✓ Risultato usabile come chiave dict")
        
        # Ma NaN in tuple causa problemi di lookup!
        result2 = sampler.get_cat_key(x_nan)
        if result != result2:
            print(f"  ⚠ NaN causa chiavi diverse: {result} != {result2}")
        
        return True
    except Exception as e:
        print(f"  Exception: {e}")
        return True  # L'eccezione può essere OK

# ============================================================================
# TEST 10: Softmax overflow in sample()
# ============================================================================
def test_softmax_overflow():
    """Test che softmax non va in overflow."""
    rng = np.random.default_rng(42)
    sampler = CategoricalSampler([(0, 100)])  # 100 categorie!
    cube = MockCube()
    
    # Crea visit_counts molto sbilanciati
    for i in range(100):
        key = (i,)
        sampler._visit_counts[key] = 1 if i < 50 else 1000000
    
    x = np.array([0.5])
    
    for i in range(50):
        result = sampler.sample(x, cube, rng)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            bugs_found.append("10: Overflow in softmax")
            print(f"  ❌ NaN/Inf: {result}")
            return False
    
    print("  ✓ Softmax stabile con 100 categorie")
    return True

# ============================================================================
# TEST 11: Integration con ALBA optimizer
# ============================================================================
def test_integration_alba():
    """Test end-to-end con ALBA su problema misto."""
    try:
        from alba_framework_potential.optimizer import ALBA
    except ImportError as e:
        print(f"  Skip: {e}")
        return True
    
    # Problema misto: 2 continuous + 2 categorical
    opt = ALBA(
        bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
        categorical_dims=[(2, 3), (3, 5)],  # dim 2 ha 3 scelte, dim 3 ha 5 scelte
        seed=42,
        total_budget=100
    )
    
    rng = np.random.default_rng(42)
    
    for i in range(100):
        x = opt.ask()
        if np.any(np.isnan(x)):
            bugs_found.append(f"11: NaN in ask() con categoricals a iter {i}")
            print(f"  ❌ NaN a iterazione {i}: {x}")
            return False
        
        # Objective misto
        cont_part = -((x[0] - 0.3)**2 + (x[1] - 0.7)**2)
        cat_bonus = 0.1 if int(x[2] * 2) == 1 else 0  # Categoria 1 è la migliore
        y = cont_part + cat_bonus + rng.normal(0, 0.01)
        
        opt.tell(x, y)
    
    print(f"  Best: {opt.best_y:.4f}")
    print(f"  Best x: {opt.best_x}")
    print("  ✓ ALBA funziona con categoricals")
    return True

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CATEGORICAL.PY STRESS TESTS")
    print("=" * 70)
    
    tests = [
        ("TEST 1: discretize con NaN", test_discretize_nan),
        ("TEST 2: to_continuous edge cases", test_to_continuous_edge),
        ("TEST 3: record_observation con NaN", test_record_nan_score),
        ("TEST 4: elite_crossover edge cases", test_elite_crossover_edge),
        ("TEST 5: sample con cube vuoto", test_sample_empty_cube),
        ("TEST 6: Thompson sampling estremo", test_thompson_extreme_stats),
        ("TEST 7: Molte dimensioni categoriche", test_many_categorical_dims),
        ("TEST 8: recompute con NaN", test_recompute_with_nan),
        ("TEST 9: get_cat_key con NaN", test_get_cat_key_nan),
        ("TEST 10: Softmax overflow", test_softmax_overflow),
        ("TEST 11: Integration ALBA", test_integration_alba),
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
        print("\n✓ NESSUN BUG TROVATO in categorical.py!")
