#!/usr/bin/env python3
"""
COHERENCE STRESS TEST - Edge Cases Estremi
============================================

Test aggressivi per trovare bug nascosti nella coherence.
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
import os
os.chdir(parent_dir)


# ============================================================
# MOCK CLASSES
# ============================================================

class MockCube:
    def __init__(self, bounds: List[tuple], lgs_model: Optional[Dict] = None, good_ratio_val: float = 0.5):
        self.bounds = bounds
        self.lgs_model = lgs_model
        self._good_ratio = good_ratio_val
    
    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])
    
    def widths(self) -> np.ndarray:
        return np.array([hi - lo for lo, hi in self.bounds])
    
    def good_ratio(self) -> float:
        return self._good_ratio


from coherence import (
    _build_knn_graph,
    _compute_predicted_drops,
    _solve_potential_least_squares,
    compute_coherence_scores,
    CoherenceTracker,
)


# ============================================================
# STRESS TESTS
# ============================================================

def test_numerical_extremes():
    """Test con valori numerici estremi."""
    
    print("=" * 70)
    print("STRESS TEST 1: Valori Numerici Estremi")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 1.1: Gradienti molto grandi
    print("\n1.1 Gradienti enormi (1e10):")
    
    grad_huge = np.array([1e10, 1e10])
    leaves = []
    for i in range(5):
        leaves.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": grad_huge.copy()},
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves)
        print(f"  global_coherence: {coh}")
        
        # Verifica NaN/Inf
        if np.isnan(coh) or np.isinf(coh):
            bugs_found.append("1.1: NaN/Inf con gradienti enormi")
        else:
            print("  ✓ Gestisce gradienti enormi")
    except Exception as e:
        bugs_found.append(f"1.1: Exception con gradienti enormi: {e}")
    
    # Test 1.2: Gradienti molto piccoli
    print("\n1.2 Gradienti minuscoli (1e-15):")
    
    grad_tiny = np.array([1e-15, 1e-15])
    leaves_tiny = []
    for i in range(5):
        leaves_tiny.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": grad_tiny.copy()},
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves_tiny)
        print(f"  global_coherence: {coh}")
        
        if np.isnan(coh) or np.isinf(coh):
            bugs_found.append("1.2: NaN/Inf con gradienti minuscoli")
        else:
            print("  ✓ Gestisce gradienti minuscoli")
    except Exception as e:
        bugs_found.append(f"1.2: Exception con gradienti minuscoli: {e}")
    
    # Test 1.3: Cubo con larghezza zero
    print("\n1.3 Cubo degenere (larghezza 0):")
    
    grad_normal = np.array([1.0, 1.0])
    leaves_degen = []
    for i in range(5):
        # Prima dimensione ha larghezza 0
        leaves_degen.append(MockCube(
            [(0.5, 0.5), (i*0.2, (i+1)*0.2)],  # Prima dim: punto
            lgs_model={"grad": grad_normal.copy()},
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves_degen)
        print(f"  global_coherence: {coh}")
        
        if np.isnan(coh) or np.isinf(coh):
            bugs_found.append("1.3: NaN/Inf con cubo degenere")
        else:
            print("  ✓ Gestisce cubo degenere")
    except Exception as e:
        bugs_found.append(f"1.3: Exception con cubo degenere: {e}")
    
    # Test 1.4: good_ratio estremi
    print("\n1.4 good_ratio estremi (0, 1, negativi):")
    
    leaves_ratio = []
    ratios = [0.0, 1.0, -0.5, 1.5, float('nan')]
    for i, ratio in enumerate(ratios):
        leaves_ratio.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": np.array([1.0, 0.0])},
            good_ratio_val=ratio
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves_ratio)
        print(f"  global_coherence: {coh}")
        print(f"  potentials: {potentials}")
        
        # Verifica che potentials siano in [0, 1]
        for idx, pot in potentials.items():
            if np.isnan(pot) or np.isinf(pot):
                bugs_found.append(f"1.4: NaN/Inf in potentials[{idx}]")
                break
            if pot < 0 or pot > 1:
                bugs_found.append(f"1.4: Potential out of bounds: {pot}")
                break
        else:
            print("  ✓ Gestisce good_ratio estremi")
    except Exception as e:
        bugs_found.append(f"1.4: Exception con good_ratio estremi: {e}")
    
    return bugs_found


def test_high_dimensionality():
    """Test con molte dimensioni."""
    
    print("\n" + "=" * 70)
    print("STRESS TEST 2: Alta Dimensionalità")
    print("=" * 70)
    
    bugs_found = []
    
    for dim in [10, 50, 100]:
        print(f"\n2.{dim//10} Dimensionalità {dim}D:")
        
        np.random.seed(42)
        leaves = []
        for i in range(10):
            bounds = [(i*0.1, (i+1)*0.1)] * dim
            grad = np.random.randn(dim)
            leaves.append(MockCube(
                bounds,
                lgs_model={"grad": grad},
                good_ratio_val=np.random.rand()
            ))
        
        try:
            scores, potentials, coh, _, _ = compute_coherence_scores(leaves)
            print(f"  global_coherence: {coh:.4f}")
            
            if np.isnan(coh) or np.isinf(coh):
                bugs_found.append(f"2.{dim}: NaN/Inf in {dim}D")
            elif coh < 0 or coh > 1:
                bugs_found.append(f"2.{dim}: Coherence out of bounds in {dim}D: {coh}")
            else:
                print(f"  ✓ Funziona in {dim}D")
        except Exception as e:
            bugs_found.append(f"2.{dim}: Exception in {dim}D: {e}")
    
    return bugs_found


def test_many_leaves():
    """Test con molte foglie."""
    
    print("\n" + "=" * 70)
    print("STRESS TEST 3: Molte Foglie")
    print("=" * 70)
    
    bugs_found = []
    
    for n_leaves in [50, 100, 200]:
        print(f"\n3.{n_leaves//50} {n_leaves} foglie:")
        
        np.random.seed(42)
        leaves = []
        
        # Disponi in griglia
        side = int(np.ceil(np.sqrt(n_leaves)))
        for i in range(n_leaves):
            row = i // side
            col = i % side
            bounds = [
                (col/side, (col+1)/side),
                (row/side, (row+1)/side)
            ]
            grad = np.random.randn(2)
            leaves.append(MockCube(
                bounds,
                lgs_model={"grad": grad},
                good_ratio_val=np.random.rand()
            ))
        
        import time
        start = time.time()
        
        try:
            scores, potentials, coh, _, _ = compute_coherence_scores(leaves)
            elapsed = time.time() - start
            
            print(f"  global_coherence: {coh:.4f}")
            print(f"  tempo: {elapsed:.3f}s")
            
            if elapsed > 5.0:
                bugs_found.append(f"3.{n_leaves}: Troppo lento ({elapsed:.1f}s)")
            elif np.isnan(coh) or np.isinf(coh):
                bugs_found.append(f"3.{n_leaves}: NaN/Inf")
            else:
                print(f"  ✓ Performance OK")
        except Exception as e:
            bugs_found.append(f"3.{n_leaves}: Exception: {e}")
    
    return bugs_found


def test_sparse_lgs_models():
    """Test con LGS model sparsi (alcuni mancanti)."""
    
    print("\n" + "=" * 70)
    print("STRESS TEST 4: LGS Models Sparsi")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 4.1: Metà foglie senza LGS
    print("\n4.1 50% foglie senza LGS:")
    
    np.random.seed(42)
    leaves = []
    for i in range(10):
        has_lgs = (i % 2 == 0)
        leaves.append(MockCube(
            [(i*0.1, (i+1)*0.1), (0, 1)],
            lgs_model={"grad": np.array([1.0, 0.0])} if has_lgs else None,
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves)
        print(f"  global_coherence: {coh}")
        
        if np.isnan(coh):
            bugs_found.append("4.1: NaN con LGS sparsi")
        else:
            print("  ✓ Gestisce LGS sparsi")
    except Exception as e:
        bugs_found.append(f"4.1: Exception: {e}")
    
    # Test 4.2: Tutte foglie senza LGS
    print("\n4.2 Tutte foglie senza LGS:")
    
    leaves_no_lgs = []
    for i in range(10):
        leaves_no_lgs.append(MockCube(
            [(i*0.1, (i+1)*0.1), (0, 1)],
            lgs_model=None,
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves_no_lgs)
        print(f"  global_coherence: {coh}")
        
        # Senza LGS validi, dovrebbe restituire 0.5
        if coh != 0.5:
            bugs_found.append(f"4.2: Expected 0.5 without LGS, got {coh}")
        else:
            print("  ✓ Corretto: 0.5 senza LGS")
    except Exception as e:
        bugs_found.append(f"4.2: Exception: {e}")
    
    # Test 4.3: Grad mancante in lgs_model
    print("\n4.3 LGS model senza 'grad' key:")
    
    leaves_no_grad = []
    for i in range(5):
        leaves_no_grad.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"other_key": "value"},  # No 'grad'
            good_ratio_val=0.5
        ))
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves_no_grad)
        print(f"  global_coherence: {coh}")
        print("  ✓ Gestisce LGS senza grad")
    except Exception as e:
        bugs_found.append(f"4.3: Exception con LGS senza grad: {e}")
    
    return bugs_found


def test_categorical_edge_cases():
    """Test con dimensioni categoriche edge cases."""
    
    print("\n" + "=" * 70)
    print("STRESS TEST 5: Dimensioni Categoriche")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 5.1: Tutte le dimensioni categoriche
    print("\n5.1 Tutte dimensioni categoriche:")
    
    leaves = []
    for i in range(5):
        leaves.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": np.array([1.0, 1.0])},
            good_ratio_val=0.5
        ))
    
    # Entrambe le dimensioni sono categoriche
    categorical_dims = [(0, 5), (1, 5)]
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(
            leaves, categorical_dims=categorical_dims
        )
        print(f"  global_coherence: {coh}")
        
        # Con tutte dim categoriche, gradienti effettivi sono tutti zero
        # Dovrebbe restituire 0.5
        if coh != 0.5:
            print(f"  ⚠ Expected 0.5 con tutte cat dims, got {coh}")
        else:
            print("  ✓ Corretto: 0.5 con tutte cat dims")
    except Exception as e:
        bugs_found.append(f"5.1: Exception: {e}")
    
    # Test 5.2: Indice categorico fuori bounds
    print("\n5.2 Indice categorico > dim gradiente:")
    
    # Gradiente 2D, ma categorical_dims dice dim 5
    categorical_dims_oob = [(5, 3)]  # Indice 5 non esiste
    
    try:
        scores, potentials, coh, _, _ = compute_coherence_scores(
            leaves, categorical_dims=categorical_dims_oob
        )
        print(f"  global_coherence: {coh}")
        print("  ✓ Gestisce indice out of bounds")
    except Exception as e:
        bugs_found.append(f"5.2: Exception con indice OOB: {e}")
    
    return bugs_found


def test_coherence_tracker_edge_cases():
    """Test CoherenceTracker edge cases."""
    
    print("\n" + "=" * 70)
    print("STRESS TEST 6: CoherenceTracker Edge Cases")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 6.1: Update con foglie che cambiano
    print("\n6.1 Foglie che cambiano tra update:")
    
    tracker = CoherenceTracker(update_interval=1)
    
    grad = np.array([1.0, 0.0])
    leaves_v1 = [
        MockCube([(0, 0.5), (0, 1)], lgs_model={"grad": grad}),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad}),
    ]
    
    tracker.update(leaves_v1, iteration=0)
    
    # Ora le foglie sono diverse (nuovi oggetti)
    leaves_v2 = [
        MockCube([(0, 0.33), (0, 1)], lgs_model={"grad": grad}),
        MockCube([(0.33, 0.66), (0, 1)], lgs_model={"grad": grad}),
        MockCube([(0.66, 1), (0, 1)], lgs_model={"grad": grad}),
    ]
    
    tracker.update(leaves_v2, iteration=1)
    
    # Prova a ottenere coherence di foglia vecchia (non più in cache)
    try:
        old_coh = tracker.get_coherence(leaves_v1[0], leaves_v2)
        print(f"  Coherence foglia vecchia: {old_coh}")
        
        if old_coh != 0.5:
            bugs_found.append(f"6.1: Expected 0.5 for old leaf, got {old_coh}")
        else:
            print("  ✓ Ritorna 0.5 per foglia non in cache")
    except Exception as e:
        bugs_found.append(f"6.1: Exception: {e}")
    
    # Test 6.2: min_leaves_for_coherence
    print("\n6.2 Sotto min_leaves_for_coherence:")
    
    tracker_min = CoherenceTracker(min_leaves_for_coherence=10)
    
    leaves_few = [MockCube([(0, 1), (0, 1)], lgs_model={"grad": grad}) for _ in range(5)]
    tracker_min.update(leaves_few, iteration=0)
    
    print(f"  global_coherence: {tracker_min.global_coherence}")
    
    if tracker_min.global_coherence != 0.5:
        bugs_found.append(f"6.2: Expected 0.5 under min_leaves, got {tracker_min.global_coherence}")
    else:
        print("  ✓ Ritorna 0.5 sotto min_leaves")
    
    return bugs_found


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COHERENCE STRESS TESTS - Edge Cases Estremi")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_numerical_extremes())
    all_bugs.extend(test_high_dimensionality())
    all_bugs.extend(test_many_leaves())
    all_bugs.extend(test_sparse_lgs_models())
    all_bugs.extend(test_categorical_edge_cases())
    all_bugs.extend(test_coherence_tracker_edge_cases())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO negli stress test!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
