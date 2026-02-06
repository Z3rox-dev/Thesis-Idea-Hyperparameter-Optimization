#!/usr/bin/env python3
"""
DEEP COHERENCE TESTING
=======================

Verifica approfondita di ogni componente del modulo coherence.py

Componenti da testare:
1. _build_knn_graph - costruzione grafo k-NN
2. _compute_predicted_drops - calcolo drop predetti e alignment
3. _solve_potential_least_squares - risoluzione potenziale globale
4. compute_coherence_scores - calcolo score finali
5. CoherenceTracker - caching e gating

Per ogni componente: test unitari + edge cases + verifica numerica
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
# MOCK CLASSES per testing isolato
# ============================================================

@dataclass
class MockLGSModel:
    """Mock LGS model per testing."""
    grad: np.ndarray
    gradient_dir: np.ndarray
    top_k_pts: np.ndarray


class MockCube:
    """Mock Cube per testing coherence senza dipendenze."""
    
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


# ============================================================
# IMPORT FUNZIONI DA TESTARE
# ============================================================

from coherence import (
    _build_knn_graph,
    _compute_predicted_drops,
    _solve_potential_least_squares,
    compute_coherence_scores,
    CoherenceTracker,
)


# ============================================================
# TEST 1: _build_knn_graph
# ============================================================

def test_build_knn_graph():
    """Test costruzione grafo k-NN."""
    
    print("=" * 70)
    print("TEST 1: _build_knn_graph")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 1.1: Caso base - 4 foglie in quadrato
    print("\n1.1 Quattro foglie in quadrato 2x2:")
    leaves = [
        MockCube([(0, 0.5), (0, 0.5)]),      # bottom-left
        MockCube([(0.5, 1), (0, 0.5)]),      # bottom-right
        MockCube([(0, 0.5), (0.5, 1)]),      # top-left
        MockCube([(0.5, 1), (0.5, 1)]),      # top-right
    ]
    
    edges = _build_knn_graph(leaves, k=2)
    print(f"  Edges (k=2): {edges}")
    print(f"  N edges: {len(edges)}")
    
    # Ogni foglia dovrebbe avere 2 vicini → 4*2 = 8 edges
    if len(edges) != 8:
        bugs_found.append(f"1.1: Expected 8 edges, got {len(edges)}")
    else:
        print("  ✓ Corretto: 8 edges")
    
    # Test 1.2: Edge case - meno di 2 foglie
    print("\n1.2 Una sola foglia:")
    leaves_1 = [MockCube([(0, 1), (0, 1)])]
    edges_1 = _build_knn_graph(leaves_1, k=2)
    print(f"  Edges: {edges_1}")
    
    if edges_1 != []:
        bugs_found.append(f"1.2: Expected empty list, got {edges_1}")
    else:
        print("  ✓ Corretto: lista vuota")
    
    # Test 1.3: k maggiore di n-1
    print("\n1.3 k > n-1 (k=10, n=3):")
    leaves_3 = [
        MockCube([(0, 0.33), (0, 1)]),
        MockCube([(0.33, 0.66), (0, 1)]),
        MockCube([(0.66, 1), (0, 1)]),
    ]
    edges_3 = _build_knn_graph(leaves_3, k=10)
    print(f"  Edges: {edges_3}")
    print(f"  N edges: {len(edges_3)}")
    
    # k viene ridotto a n-1=2, quindi 3*2 = 6 edges
    if len(edges_3) != 6:
        bugs_found.append(f"1.3: Expected 6 edges, got {len(edges_3)}")
    else:
        print("  ✓ Corretto: k troncato a n-1")
    
    # Test 1.4: Foglie con stessa posizione (degenere)
    print("\n1.4 Due foglie nella stessa posizione:")
    leaves_same = [
        MockCube([(0.5, 0.5), (0.5, 0.5)]),  # Punto
        MockCube([(0.5, 0.5), (0.5, 0.5)]),  # Stesso punto
    ]
    edges_same = _build_knn_graph(leaves_same, k=1)
    print(f"  Edges: {edges_same}")
    # Dovrebbe gestire distanza 0
    
    return bugs_found


# ============================================================
# TEST 2: _compute_predicted_drops
# ============================================================

def test_compute_predicted_drops():
    """Test calcolo drop predetti e alignment."""
    
    print("\n" + "=" * 70)
    print("TEST 2: _compute_predicted_drops")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 2.1: Due foglie con gradienti paralleli
    print("\n2.1 Gradienti paralleli (→ →):")
    
    grad_right = np.array([1.0, 0.0])
    leaves = [
        MockCube([(0, 0.5), (0, 1)], lgs_model={"grad": grad_right}),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad_right}),
    ]
    edges = [(0, 1)]
    
    d_lm, alignments, valid_edges = _compute_predicted_drops(leaves, edges)
    
    print(f"  d_lm: {d_lm}")
    print(f"  alignments: {alignments}")
    
    # Alignment tra gradienti paralleli dovrebbe essere 1.0
    if len(alignments) > 0 and abs(alignments[0] - 1.0) > 0.01:
        bugs_found.append(f"2.1: Expected alignment ~1.0, got {alignments[0]}")
    else:
        print("  ✓ Alignment corretto: ~1.0")
    
    # Test 2.2: Gradienti opposti (→ ←) - VALLE!
    print("\n2.2 Gradienti opposti (→ ←) = VALLE:")
    
    grad_left = np.array([-1.0, 0.0])
    leaves_opp = [
        MockCube([(0, 0.5), (0, 1)], lgs_model={"grad": grad_right}),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad_left}),
    ]
    
    d_lm_opp, align_opp, _ = _compute_predicted_drops(leaves_opp, edges)
    
    print(f"  alignments: {align_opp}")
    
    if len(align_opp) > 0 and abs(align_opp[0] - (-1.0)) > 0.01:
        bugs_found.append(f"2.2: Expected alignment ~-1.0, got {align_opp[0]}")
    else:
        print("  ✓ Alignment corretto: ~-1.0 (valle)")
    
    # Test 2.3: Gradiente nullo
    print("\n2.3 Gradiente nullo (una foglia):")
    
    grad_zero = np.array([0.0, 0.0])
    leaves_zero = [
        MockCube([(0, 0.5), (0, 1)], lgs_model={"grad": grad_zero}),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad_right}),
    ]
    
    d_lm_zero, align_zero, valid_zero = _compute_predicted_drops(leaves_zero, edges)
    
    print(f"  valid_edges: {valid_zero}")
    
    if len(valid_zero) != 0:
        bugs_found.append(f"2.3: Expected 0 valid edges (grad nullo), got {len(valid_zero)}")
    else:
        print("  ✓ Corretto: edge scartato per gradiente nullo")
    
    # Test 2.4: LGS model mancante
    print("\n2.4 LGS model mancante:")
    
    leaves_no_lgs = [
        MockCube([(0, 0.5), (0, 1)], lgs_model=None),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad_right}),
    ]
    
    _, _, valid_no_lgs = _compute_predicted_drops(leaves_no_lgs, edges)
    
    print(f"  valid_edges: {valid_no_lgs}")
    
    if len(valid_no_lgs) != 0:
        bugs_found.append(f"2.4: Expected 0 valid edges (no LGS), got {len(valid_no_lgs)}")
    else:
        print("  ✓ Corretto: edge scartato per LGS mancante")
    
    # Test 2.5: Dimensioni categoriche mascherate
    print("\n2.5 Dimensioni categoriche mascherate:")
    
    grad_3d = np.array([1.0, 0.5, 2.0])  # dim 2 è categorica
    leaves_cat = [
        MockCube([(0, 0.5), (0, 1), (0, 1)], lgs_model={"grad": grad_3d.copy()}),
        MockCube([(0.5, 1), (0, 1), (0, 1)], lgs_model={"grad": grad_3d.copy()}),
    ]
    
    categorical_dims = [(2, 3)]  # dim 2 con 3 categorie
    
    _, align_cat, _ = _compute_predicted_drops(leaves_cat, [(0, 1)], categorical_dims)
    
    print(f"  alignments (dim 2 mascherata): {align_cat}")
    
    # Con dim 2 mascherata, i gradienti effettivi sono [1.0, 0.5, 0.0]
    # Alignment dovrebbe essere 1.0 (paralleli)
    if len(align_cat) > 0 and abs(align_cat[0] - 1.0) > 0.01:
        bugs_found.append(f"2.5: Expected alignment ~1.0 after masking, got {align_cat[0]}")
    else:
        print("  ✓ Corretto: dimensione categorica mascherata")
    
    return bugs_found


# ============================================================
# TEST 3: _solve_potential_least_squares
# ============================================================

def test_solve_potential():
    """Test risoluzione sistema potenziale."""
    
    print("\n" + "=" * 70)
    print("TEST 3: _solve_potential_least_squares")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 3.1: Catena lineare semplice
    print("\n3.1 Catena lineare (3 foglie, d_lm = 1 costante):")
    
    # 3 foglie: 0 → 1 → 2
    # d_01 = 1, d_12 = 1
    # Soluzione: u = [0, 1, 2]
    
    n_leaves = 3
    edges = [(0, 1), (1, 2)]
    d_lm = np.array([1.0, 1.0])
    
    u = _solve_potential_least_squares(n_leaves, edges, d_lm)
    
    print(f"  u = {u}")
    print(f"  Expected: ~[0, 1, 2]")
    
    # u[0] = 0 (fisso), u[1] ≈ 1, u[2] ≈ 2
    if abs(u[0]) > 0.01:
        bugs_found.append(f"3.1: u[0] should be 0, got {u[0]}")
    if abs(u[1] - 1.0) > 0.1:
        bugs_found.append(f"3.1: u[1] should be ~1, got {u[1]}")
    if abs(u[2] - 2.0) > 0.1:
        bugs_found.append(f"3.1: u[2] should be ~2, got {u[2]}")
    
    if not bugs_found or not any("3.1" in b for b in bugs_found):
        print("  ✓ Soluzione corretta")
    
    # Test 3.2: Ciclo (over-determined)
    print("\n3.2 Ciclo (3 foglie, grafo triangolare):")
    
    # Ciclo: 0 → 1 → 2 → 0
    edges_cycle = [(0, 1), (1, 2), (2, 0)]
    d_lm_cycle = np.array([1.0, 1.0, -2.0])  # Consistente: u1-u0=1, u2-u1=1, u0-u2=-2
    
    u_cycle = _solve_potential_least_squares(3, edges_cycle, d_lm_cycle)
    
    print(f"  u = {u_cycle}")
    
    # Verifica consistenza
    residuals = []
    for e, (i, j) in enumerate(edges_cycle):
        residual = u_cycle[j] - u_cycle[i] - d_lm_cycle[e]
        residuals.append(residual)
    
    print(f"  Residui: {residuals}")
    
    # Test 3.3: Nessun edge
    print("\n3.3 Nessun edge:")
    
    u_empty = _solve_potential_least_squares(3, [], np.array([]))
    
    print(f"  u = {u_empty}")
    
    if not np.allclose(u_empty, np.zeros(3)):
        bugs_found.append(f"3.3: Expected zeros, got {u_empty}")
    else:
        print("  ✓ Corretto: tutti zeri")
    
    return bugs_found


# ============================================================
# TEST 4: compute_coherence_scores (integrazione)
# ============================================================

def test_compute_coherence_scores():
    """Test calcolo score coherence integrato."""
    
    print("\n" + "=" * 70)
    print("TEST 4: compute_coherence_scores")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 4.1: Gradienti tutti allineati → alta coherence
    print("\n4.1 Gradienti tutti allineati (Sphere-like):")
    
    grad_right = np.array([1.0, 0.0])
    leaves = []
    for i in range(5):
        leaves.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": grad_right.copy()},
            good_ratio_val=0.5
        ))
    
    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves)
    
    print(f"  scores: {scores}")
    print(f"  global_coherence: {global_coh}")
    
    # Tutti allineati → coherence alta (vicino a 1)
    if global_coh < 0.8:
        bugs_found.append(f"4.1: Expected high coherence (>0.8), got {global_coh}")
    else:
        print("  ✓ Alta coherence per gradienti allineati")
    
    # Test 4.2: Gradienti casuali → coherence neutrale
    print("\n4.2 Gradienti casuali:")
    
    np.random.seed(42)
    leaves_rand = []
    for i in range(5):
        rand_grad = np.random.randn(2)
        rand_grad = rand_grad / np.linalg.norm(rand_grad)
        leaves_rand.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": rand_grad},
            good_ratio_val=np.random.rand()
        ))
    
    _, _, global_coh_rand, _, _ = compute_coherence_scores(leaves_rand)
    
    print(f"  global_coherence: {global_coh_rand}")
    
    # Casuali → coherence media (intorno a 0.5)
    if global_coh_rand < 0.2 or global_coh_rand > 0.8:
        print(f"  ⚠ Coherence inattesa per gradienti casuali: {global_coh_rand}")
    else:
        print("  ✓ Coherence media per gradienti casuali")
    
    # Test 4.3: Poche foglie (< 3)
    print("\n4.3 Poche foglie (n < 3):")
    
    leaves_2 = [
        MockCube([(0, 0.5), (0, 1)], lgs_model={"grad": grad_right}),
        MockCube([(0.5, 1), (0, 1)], lgs_model={"grad": grad_right}),
    ]
    
    scores_2, _, global_2, _, _ = compute_coherence_scores(leaves_2)
    
    print(f"  scores: {scores_2}")
    print(f"  global_coherence: {global_2}")
    
    # Con < 3 foglie dovrebbe restituire 0.5 (neutro)
    if global_2 != 0.5:
        bugs_found.append(f"4.3: Expected 0.5 for n<3, got {global_2}")
    else:
        print("  ✓ Corretto: 0.5 per poche foglie")
    
    # Test 4.4: Verifica bounds coherence [0, 1]
    print("\n4.4 Bounds coherence [0, 1]:")
    
    np.random.seed(123)
    for trial in range(5):
        leaves_test = []
        for i in range(10):
            rand_grad = np.random.randn(2) * 10  # Gradienti grandi
            leaves_test.append(MockCube(
                [(i*0.1, (i+1)*0.1), (0, 1)],
                lgs_model={"grad": rand_grad},
                good_ratio_val=np.random.rand()
            ))
        
        scores_test, potentials_test, coh_test, _, _ = compute_coherence_scores(leaves_test)
        
        for idx, score in scores_test.items():
            if score < 0 or score > 1:
                bugs_found.append(f"4.4: Score out of bounds: {score}")
                break
        
        for idx, pot in potentials_test.items():
            if pot < 0 or pot > 1:
                bugs_found.append(f"4.4: Potential out of bounds: {pot}")
                break
        
        if coh_test < 0 or coh_test > 1:
            bugs_found.append(f"4.4: Global coherence out of bounds: {coh_test}")
    
    if not any("4.4" in b for b in bugs_found):
        print("  ✓ Tutti i valori in [0, 1]")
    
    return bugs_found


# ============================================================
# TEST 5: CoherenceTracker
# ============================================================

def test_coherence_tracker():
    """Test CoherenceTracker e caching."""
    
    print("\n" + "=" * 70)
    print("TEST 5: CoherenceTracker")
    print("=" * 70)
    
    bugs_found = []
    
    # Test 5.1: Caching funziona
    print("\n5.1 Caching:")
    
    tracker = CoherenceTracker(update_interval=5)
    
    grad_right = np.array([1.0, 0.0])
    leaves = []
    for i in range(5):
        leaves.append(MockCube(
            [(i*0.2, (i+1)*0.2), (0, 1)],
            lgs_model={"grad": grad_right.copy()},
            good_ratio_val=0.5
        ))
    
    # Prima update
    tracker.update(leaves, iteration=0)
    coh_0 = tracker.global_coherence
    print(f"  Iter 0: global_coherence = {coh_0}")
    
    # Stesso iteration, non dovrebbe aggiornare
    tracker.update(leaves, iteration=1)
    coh_1 = tracker.global_coherence
    print(f"  Iter 1 (cached): global_coherence = {coh_1}")
    
    # Dopo update_interval
    tracker.update(leaves, iteration=5)
    coh_5 = tracker.global_coherence
    print(f"  Iter 5: global_coherence = {coh_5}")
    
    # Coh dovrebbe essere uguale (stesse foglie)
    if abs(coh_0 - coh_5) > 0.01:
        print(f"  ⚠ Coherence cambiata inaspettatamente: {coh_0} → {coh_5}")
    else:
        print("  ✓ Caching funziona correttamente")
    
    # Test 5.2: should_exploit
    print("\n5.2 should_exploit:")
    
    exploit = tracker.should_exploit(leaves[0], leaves)
    print(f"  should_exploit: {exploit}")
    print(f"  q60_threshold: {tracker.q60_threshold}")
    print(f"  leaf coherence: {tracker.get_coherence(leaves[0], leaves)}")
    
    # Test 5.3: get_statistics
    print("\n5.3 get_statistics:")
    
    stats = tracker.get_statistics()
    print(f"  Stats: {stats}")
    
    required_keys = ["global_coherence", "q60_threshold", "q80_threshold", 
                     "n_leaves_cached", "min_coherence", "max_coherence", "mean_coherence"]
    
    for key in required_keys:
        if key not in stats:
            bugs_found.append(f"5.3: Missing key in stats: {key}")
    
    if not any("5.3" in b for b in bugs_found):
        print("  ✓ Tutte le statistiche presenti")
    
    return bugs_found


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DEEP COHERENCE TESTING")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_build_knn_graph())
    all_bugs.extend(test_compute_predicted_drops())
    all_bugs.extend(test_solve_potential())
    all_bugs.extend(test_compute_coherence_scores())
    all_bugs.extend(test_coherence_tracker())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO nella coherence!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
