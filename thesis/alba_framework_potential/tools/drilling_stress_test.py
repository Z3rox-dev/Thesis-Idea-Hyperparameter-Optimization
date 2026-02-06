#!/usr/bin/env python3
"""
DRILLING.PY STRESS TEST
========================

Test aggressivi per trovare bug nel modulo drilling.py
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Use proper module import
sys.path.insert(0, '/mnt/workspace/thesis')
import os
os.chdir('/mnt/workspace/thesis/alba_framework_potential')

from alba_framework_potential.drilling import DrillingOptimizer

# ============================================================
# TEST FUNCTIONS
# ============================================================

# ============================================================
# STRESS TESTS
# ============================================================

def test_nan_start_point():
    """Test con punto di partenza NaN."""
    print("=" * 70)
    print("TEST 1: Punto di partenza con NaN")
    print("=" * 70)
    
    bugs = []
    rng = np.random.default_rng(42)
    
    # Test 1.1: start_x con NaN
    print("\n1.1 start_x con NaN:")
    start_x_nan = np.array([0.5, float('nan'), 0.5])
    start_y = 1.0
    
    try:
        driller = DrillingOptimizer(start_x_nan, start_y, bounds=[(0,1), (0,1), (0,1)])
        x = driller.ask(rng)
        
        if np.any(np.isnan(x)):
            bugs.append("1.1: NaN propagato in ask()")
            print(f"  ❌ x contiene NaN: {x}")
        else:
            print(f"  Candidato generato: {x}")
    except Exception as e:
        print(f"  Exception: {e}")
        # Non necessariamente un bug - potrebbe essere comportamento valido
    
    # Test 1.2: start_y con NaN
    print("\n1.2 start_y = NaN:")
    start_x_ok = np.array([0.5, 0.5, 0.5])
    start_y_nan = float('nan')
    
    try:
        driller = DrillingOptimizer(start_x_ok, start_y_nan)
        x = driller.ask(rng)
        y_new = 0.5  # Miglioramento
        
        # tell() confronta y < best_y, ma NaN < 0.5 = False
        cont = driller.tell(x, y_new)
        
        print(f"  Dopo tell(0.5): best_y = {driller.best_y}, continue = {cont}")
        
        if np.isnan(driller.best_y):
            print("  ⚠ best_y resta NaN - potrebbe non migliorare mai")
        else:
            print("  ✓ best_y aggiornato correttamente")
    except Exception as e:
        bugs.append(f"1.2: Exception: {e}")
    
    return bugs


def test_covariance_degeneracy():
    """Test quando la matrice di covarianza degenera."""
    print("\n" + "=" * 70)
    print("TEST 2: Degenerazione della covarianza")
    print("=" * 70)
    
    bugs = []
    rng = np.random.default_rng(42)
    
    # Test 2.1: Molti fallimenti consecutivi
    print("\n2.1 Molti fallimenti → sigma shrinks:")
    driller = DrillingOptimizer(
        np.array([0.5, 0.5]),
        start_y=0.0,  # Già ottimo → ogni nuovo y sarà peggiore
        initial_sigma=0.1
    )
    
    for i in range(50):
        x = driller.ask(rng)
        y = 1.0 + i  # Sempre peggiore di best_y=0
        cont = driller.tell(x, y)
        
        if not cont:
            print(f"  Fermato dopo {i+1} iterazioni, sigma = {driller.sigma:.2e}")
            break
    
    if driller.sigma < 1e-7:
        print("  ✓ Sigma underflow gestito (stop condition)")
    
    # Test 2.2: Covariance explosion (molti successi)
    print("\n2.2 Molti successi → covariance growth:")
    driller2 = DrillingOptimizer(
        np.array([0.5, 0.5]),
        start_y=1000.0,  # Alto → ogni nuovo y sarà migliore
        initial_sigma=0.1
    )
    
    for i in range(100):
        x = driller2.ask(rng)
        y = 999.0 - i  # Sempre migliore
        cont = driller2.tell(x, y)
        
        # Check C matrix
        c_max = np.max(np.abs(driller2.C))
        c_cond = np.linalg.cond(driller2.C)
        
        if np.isnan(c_max) or np.isinf(c_max):
            bugs.append(f"2.2: C matrix NaN/Inf dopo {i+1} successi")
            print(f"  ❌ C matrix degenera dopo {i+1} successi")
            break
            
        if not cont:
            print(f"  Fermato dopo {i+1} iterazioni")
            print(f"  sigma = {driller2.sigma:.2e}, C_max = {c_max:.2e}, cond(C) = {c_cond:.2e}")
            break
    else:
        print(f"  100 iterazioni completate")
        print(f"  sigma = {driller2.sigma:.2e}, C_max = {c_max:.2e}")
    
    return bugs


def test_high_dimensionality():
    """Test con molte dimensioni."""
    print("\n" + "=" * 70)
    print("TEST 3: Alta dimensionalità")
    print("=" * 70)
    
    bugs = []
    rng = np.random.default_rng(42)
    
    for dim in [10, 50, 100]:
        print(f"\n3.{dim//10} {dim}D:")
        
        start_x = np.random.rand(dim)
        driller = DrillingOptimizer(start_x, start_y=10.0, initial_sigma=0.1)
        
        import time
        start = time.time()
        
        try:
            for i in range(20):
                x = driller.ask(rng)
                y = 10.0 - i * 0.1  # Graduale miglioramento
                cont = driller.tell(x, y)
                
                if np.any(np.isnan(x)):
                    bugs.append(f"3.{dim}: NaN in x dopo {i+1} iterazioni")
                    break
                    
                if not cont:
                    break
            
            elapsed = time.time() - start
            print(f"  ✓ {driller.current_step} iterazioni in {elapsed:.3f}s")
            
        except Exception as e:
            bugs.append(f"3.{dim}: Exception: {e}")
    
    return bugs


def test_bounds_handling():
    """Test gestione dei bounds."""
    print("\n" + "=" * 70)
    print("TEST 4: Bounds handling")
    print("=" * 70)
    
    bugs = []
    rng = np.random.default_rng(42)
    
    # Test 4.1: Bounds stretti
    print("\n4.1 Bounds stretti [0.4, 0.6]:")
    bounds = [(0.4, 0.6), (0.4, 0.6)]
    driller = DrillingOptimizer(
        np.array([0.5, 0.5]),
        start_y=1.0,
        initial_sigma=0.5,  # Sigma > range
        bounds=bounds
    )
    
    for _ in range(20):
        x = driller.ask(rng)
        
        # Verifica bounds
        for d, (lo, hi) in enumerate(bounds):
            if x[d] < lo or x[d] > hi:
                bugs.append(f"4.1: x[{d}] = {x[d]} fuori bounds [{lo}, {hi}]")
                break
        
        y = np.random.rand()
        cont = driller.tell(x, y)
        if not cont:
            break
    
    if not bugs or not any("4.1" in b for b in bugs):
        print("  ✓ Tutti i candidati rispettano i bounds")
    
    # Test 4.2: Bounds con Inf
    print("\n4.2 Bounds con Inf:")
    bounds_inf = [(0, float('inf')), (-float('inf'), 1)]
    
    try:
        driller2 = DrillingOptimizer(
            np.array([0.5, 0.5]),
            start_y=1.0,
            bounds=bounds_inf
        )
        
        for _ in range(10):
            x = driller2.ask(rng)
            
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                bugs.append(f"4.2: x contiene NaN/Inf: {x}")
                break
                
            y = np.random.rand()
            driller2.tell(x, y)
        else:
            print("  ✓ Gestisce bounds con Inf")
    except Exception as e:
        bugs.append(f"4.2: Exception: {e}")
    
    return bugs


def test_extreme_initial_values():
    """Test con valori iniziali estremi."""
    print("\n" + "=" * 70)
    print("TEST 5: Valori iniziali estremi")
    print("=" * 70)
    
    bugs = []
    rng = np.random.default_rng(42)
    
    # Test 5.1: sigma enorme
    print("\n5.1 sigma iniziale = 1e10:")
    try:
        driller = DrillingOptimizer(
            np.array([0.5, 0.5]),
            start_y=1.0,
            initial_sigma=1e10
        )
        
        x = driller.ask(rng)
        print(f"  x = {x}")
        
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            bugs.append("5.1: x NaN/Inf con sigma enorme")
        else:
            print("  ✓ Genera candidato (molto lontano)")
    except Exception as e:
        bugs.append(f"5.1: Exception: {e}")
    
    # Test 5.2: sigma minuscolo
    print("\n5.2 sigma iniziale = 1e-20:")
    try:
        driller = DrillingOptimizer(
            np.array([0.5, 0.5]),
            start_y=1.0,
            initial_sigma=1e-20
        )
        
        # sigma viene clippato a 1e-4
        print(f"  sigma effettivo: {driller.sigma}")
        
        x = driller.ask(rng)
        print(f"  x ≈ mu: {np.allclose(x, [0.5, 0.5], atol=0.01)}")
        print("  ✓ Gestisce sigma minuscolo")
    except Exception as e:
        bugs.append(f"5.2: Exception: {e}")
    
    # Test 5.3: start_y = Inf
    print("\n5.3 start_y = Inf:")
    try:
        driller = DrillingOptimizer(
            np.array([0.5, 0.5]),
            start_y=float('inf'),
            initial_sigma=0.1
        )
        
        x = driller.ask(rng)
        y = 100.0
        cont = driller.tell(x, y)
        
        print(f"  Dopo tell(100): best_y = {driller.best_y}")
        
        if driller.best_y == 100.0:
            print("  ✓ Migliora da Inf a 100")
        else:
            bugs.append(f"5.3: best_y non aggiornato: {driller.best_y}")
    except Exception as e:
        bugs.append(f"5.3: Exception: {e}")
    
    return bugs


def test_best_y_comparison_with_nan():
    """Test specifico: confronto y < best_y quando best_y è NaN."""
    print("\n" + "=" * 70)
    print("TEST 6: Confronto con best_y = NaN")
    print("=" * 70)
    
    bugs = []
    
    # In Python: x < NaN = False per qualsiasi x
    # Quindi se best_y = NaN, nessun improvement sarà mai riconosciuto
    
    print("\n6.1 Verifica comportamento Python:")
    print(f"  1.0 < float('nan') = {1.0 < float('nan')}")
    print(f"  float('nan') < 1.0 = {float('nan') < 1.0}")
    print(f"  → Se best_y = NaN, success = (y < NaN) = False sempre!")
    
    print("\n6.2 Simulazione drilling con best_y = NaN:")
    rng = np.random.default_rng(42)
    
    driller = DrillingOptimizer(
        np.array([0.5, 0.5]),
        start_y=float('nan'),
        initial_sigma=0.1
    )
    
    improvements = 0
    for i in range(10):
        x = driller.ask(rng)
        y = 0.5 - i * 0.1  # Sempre migliore (più basso)
        
        old_best = driller.best_y
        cont = driller.tell(x, y)
        new_best = driller.best_y
        
        if new_best != old_best:
            improvements += 1
    
    print(f"  Improvements riconosciuti: {improvements}/10")
    print(f"  best_y finale: {driller.best_y}")
    
    if improvements == 0 and np.isnan(driller.best_y):
        bugs.append("6: best_y = NaN blocca tutti i miglioramenti")
        print("  ❌ BUG: best_y = NaN blocca drilling!")
    elif improvements > 0:
        print("  ✓ Miglioramenti riconosciuti")
    
    return bugs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DRILLING.PY STRESS TESTS")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_nan_start_point())
    all_bugs.extend(test_covariance_degeneracy())
    all_bugs.extend(test_high_dimensionality())
    all_bugs.extend(test_bounds_handling())
    all_bugs.extend(test_extreme_initial_values())
    all_bugs.extend(test_best_y_comparison_with_nan())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in drilling.py!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
