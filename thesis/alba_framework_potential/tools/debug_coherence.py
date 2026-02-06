#!/usr/bin/env python3
"""
Deep debug del modulo Coherence.

Verifiche da fare:
1. La costruzione del grafo kNN funziona correttamente?
2. I gradienti vengono calcolati e normalizzati correttamente?
3. Il sistema least-squares produce potenziali sensati?
4. Le threshold Q60/Q80 sono stabili o esplodono?
5. Il mapping coherence -> exploit/explore funziona?
6. Edge cases: poche foglie, dimensioni alte, gradienti degeneri
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Import del modulo coherence
from alba_framework_potential.coherence import (
    CoherenceTracker,
    compute_coherence_scores,
    _build_knn_graph,
    _compute_predicted_drops,
    _solve_potential_least_squares,
)
from alba_framework_potential.cube import Cube
from alba_framework_potential.optimizer import ALBA


def make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))

def make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))


def test_knn_graph_construction():
    """Test che il grafo kNN sia costruito correttamente"""
    print("=" * 60)
    print("TEST 1: kNN Graph Construction")
    print("=" * 60)
    
    # Crea alcune foglie mock
    class MockCube:
        def __init__(self, center_val):
            self._center = np.array(center_val)
            self.lgs_model = None
        def center(self):
            return self._center
    
    # 4 foglie in 2D
    leaves = [
        MockCube([0.0, 0.0]),
        MockCube([1.0, 0.0]),
        MockCube([0.0, 1.0]),
        MockCube([1.0, 1.0]),
    ]
    
    edges = _build_knn_graph(leaves, k=2)
    
    print(f"Number of edges: {len(edges)}")
    print(f"Edges: {edges}")
    
    # Ogni nodo dovrebbe avere almeno 2 neighbors
    from collections import Counter
    source_counts = Counter(e[0] for e in edges)
    print(f"Edges per source: {dict(source_counts)}")
    
    if all(c >= 2 for c in source_counts.values()):
        print("✅ PASS: Each leaf has at least k neighbors")
    else:
        print("⚠️  FAIL: Some leaves have fewer than k neighbors")


def test_gradient_alignment():
    """Test che l'allineamento dei gradienti funzioni"""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Alignment Computation")
    print("=" * 60)
    
    class MockCube:
        def __init__(self, center_val, grad_val):
            self._center = np.array(center_val)
            self.lgs_model = {"grad": np.array(grad_val)} if grad_val else None
        def center(self):
            return self._center
    
    # Caso 1: Gradienti allineati (stessa direzione)
    leaves_aligned = [
        MockCube([0.0, 0.0], [1.0, 0.0]),
        MockCube([1.0, 0.0], [1.0, 0.0]),
    ]
    edges = [(0, 1)]
    d_lm, alignments, valid = _compute_predicted_drops(leaves_aligned, edges)
    
    print(f"Aligned gradients:")
    print(f"  d_lm: {d_lm}")
    print(f"  alignments: {alignments}")
    
    if len(alignments) > 0 and alignments[0] > 0.99:
        print("✅ PASS: Aligned gradients have alignment ~1.0")
    else:
        print("⚠️  FAIL: Expected alignment ~1.0")
    
    # Caso 2: Gradienti opposti
    leaves_opposite = [
        MockCube([0.0, 0.0], [1.0, 0.0]),
        MockCube([1.0, 0.0], [-1.0, 0.0]),
    ]
    d_lm, alignments, valid = _compute_predicted_drops(leaves_opposite, edges)
    
    print(f"\nOpposite gradients:")
    print(f"  alignments: {alignments}")
    
    if len(alignments) > 0 and alignments[0] < -0.99:
        print("✅ PASS: Opposite gradients have alignment ~-1.0")
    else:
        print("⚠️  FAIL: Expected alignment ~-1.0")
    
    # Caso 3: Gradienti ortogonali
    leaves_ortho = [
        MockCube([0.0, 0.0], [1.0, 0.0]),
        MockCube([1.0, 0.0], [0.0, 1.0]),
    ]
    d_lm, alignments, valid = _compute_predicted_drops(leaves_ortho, edges)
    
    print(f"\nOrthogonal gradients:")
    print(f"  alignments: {alignments}")
    
    if len(alignments) > 0 and abs(alignments[0]) < 0.1:
        print("✅ PASS: Orthogonal gradients have alignment ~0.0")
    else:
        print("⚠️  FAIL: Expected alignment ~0.0")


def test_least_squares_solver():
    """Test che il solver least-squares funzioni"""
    print("\n" + "=" * 60)
    print("TEST 3: Least Squares Potential Solver")
    print("=" * 60)
    
    # Caso semplice: 3 nodi in linea con drop costante
    n_leaves = 3
    edges = [(0, 1), (1, 2)]
    d_lm = np.array([1.0, 1.0])  # Drop di 1 per ogni edge
    
    u = _solve_potential_least_squares(n_leaves, edges, d_lm)
    
    print(f"Potentials: {u}")
    print(f"Expected: ~[0, 1, 2] (monotonic increase)")
    
    # Verifica che u sia monotonicamente crescente
    if u[0] < u[1] < u[2]:
        print("✅ PASS: Potentials are monotonically increasing")
    else:
        print("⚠️  FAIL: Potentials should be monotonically increasing")
    
    # Caso con ciclo (dovrebbe bilanciare)
    edges_cycle = [(0, 1), (1, 2), (2, 0)]
    d_lm_cycle = np.array([1.0, 1.0, 1.0])  # Inconsistent! Sum != 0
    
    u_cycle = _solve_potential_least_squares(3, edges_cycle, d_lm_cycle)
    
    print(f"\nCycle potentials: {u_cycle}")
    print("(Least squares should find best compromise)")
    
    if np.isfinite(u_cycle).all():
        print("✅ PASS: Solver handles cycles without NaN/Inf")
    else:
        print("⚠️  FAIL: Solver produced NaN/Inf on cycle")


def test_coherence_with_real_optimizer():
    """Test coherence con un optimizer reale"""
    print("\n" + "=" * 60)
    print("TEST 4: Coherence with Real ALBA Optimizer")
    print("=" * 60)
    
    dim = 5
    budget = 100
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        seed=42
    )
    
    # Run e traccia coherence
    coherence_values = []
    potential_values = []
    n_leaves_history = []
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        
        # Traccia ogni 10 iterazioni
        if i % 10 == 0 and opt._coherence_tracker is not None:
            tracker = opt._coherence_tracker
            stats = tracker.get_statistics()
            coherence_values.append(stats["global_coherence"])
            n_leaves_history.append(stats["n_leaves_cached"])
            
            # Traccia potenziali
            if tracker._cache.potentials:
                pot_vals = list(tracker._cache.potentials.values())
                potential_values.append({
                    "min": min(pot_vals),
                    "max": max(pot_vals),
                    "mean": np.mean(pot_vals),
                })
    
    print(f"Coherence evolution:")
    for i, (c, n) in enumerate(zip(coherence_values, n_leaves_history)):
        print(f"  iter {i*10:3d}: coherence={c:.3f}, n_leaves={n}")
    
    print(f"\nPotential evolution:")
    for i, p in enumerate(potential_values):
        print(f"  iter {i*10:3d}: min={p['min']:.3f}, max={p['max']:.3f}, mean={p['mean']:.3f}")
    
    # Verifica stabilità
    if all(0 <= c <= 1 for c in coherence_values):
        print("✅ PASS: Coherence values are in [0, 1]")
    else:
        print("⚠️  FAIL: Coherence values out of bounds!")
    
    if all(np.isfinite(c) for c in coherence_values):
        print("✅ PASS: No NaN/Inf in coherence")
    else:
        print("⚠️  FAIL: NaN/Inf in coherence values!")


def test_high_dimensional_stability():
    """Test stabilità in alta dimensionalità"""
    print("\n" + "=" * 60)
    print("TEST 5: High-Dimensional Stability")
    print("=" * 60)
    
    for dim in [10, 20, 50]:
        budget = max(100, dim * 5)
        bounds = [(-5.0, 5.0)] * dim
        func = make_sphere(dim)
        
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            use_coherence_gating=True,
            seed=42
        )
        
        nan_count = 0
        inf_count = 0
        
        for i in range(budget):
            x = opt.ask()
            y = func(x)
            opt.tell(x, y)
            
            if opt._coherence_tracker is not None:
                stats = opt._coherence_tracker.get_statistics()
                if not np.isfinite(stats["global_coherence"]):
                    nan_count += 1
        
        result = "✅ PASS" if nan_count == 0 else f"⚠️  FAIL ({nan_count} NaN)"
        print(f"  {dim}D: {result}")


def test_few_leaves_edge_case():
    """Test con pochissime foglie"""
    print("\n" + "=" * 60)
    print("TEST 6: Few Leaves Edge Case")
    print("=" * 60)
    
    dim = 5
    budget = 20  # Molto basso, poche foglie
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        seed=42
    )
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
    
    if opt._coherence_tracker is not None:
        stats = opt._coherence_tracker.get_statistics()
        print(f"  n_leaves: {stats['n_leaves_cached']}")
        print(f"  global_coherence: {stats['global_coherence']:.3f}")
        
        if stats['n_leaves_cached'] < 5 and stats['global_coherence'] == 0.5:
            print("✅ PASS: Default neutral coherence for few leaves")
        elif np.isfinite(stats['global_coherence']):
            print("✅ PASS: Valid coherence computed")
        else:
            print("⚠️  FAIL: Invalid coherence value")


def test_potential_field_sanity():
    """Test che il potential field abbia senso fisico"""
    print("\n" + "=" * 60)
    print("TEST 7: Potential Field Sanity Check")
    print("=" * 60)
    
    dim = 5
    budget = 200
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)  # Minimum at origin
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        seed=42
    )
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
    
    # Analizza i potenziali delle foglie
    if opt._coherence_tracker is not None and len(opt.leaves) > 0:
        tracker = opt._coherence_tracker
        
        # Trova la foglia più vicina all'origine
        distances = []
        potentials = []
        
        for i, leaf in enumerate(opt.leaves):
            center = leaf.center()
            dist = np.linalg.norm(center)
            pot = tracker._cache.potentials.get(i, 0.5)
            distances.append(dist)
            potentials.append(pot)
        
        # Correlazione: foglie vicine all'origine dovrebbero avere potenziale BASSO
        # (perché sono vicine al minimo)
        from scipy.stats import spearmanr
        corr, pval = spearmanr(distances, potentials)
        
        print(f"  N leaves: {len(opt.leaves)}")
        print(f"  Distance-Potential correlation: {corr:.3f} (p={pval:.4f})")
        
        if corr > 0.3:
            print("✅ PASS: Positive correlation (closer to origin → lower potential)")
        elif corr > 0:
            print("⚠️  WEAK: Weak positive correlation")
        else:
            print("⚠️  CHECK: Negative correlation - potential field may be inverted")


def test_multimodal_function():
    """Test su funzione multimodale"""
    print("\n" + "=" * 60)
    print("TEST 8: Multimodal Function (Rastrigin)")
    print("=" * 60)
    
    dim = 5
    budget = 150
    bounds = [(-5.0, 5.0)] * dim
    
    def rastrigin(x):
        return float(10 * dim + np.sum(np.array(x)**2 - 10 * np.cos(2 * np.pi * np.array(x))))
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        seed=42
    )
    
    coherences = []
    
    for i in range(budget):
        x = opt.ask()
        y = rastrigin(x)
        opt.tell(x, y)
        
        if i % 30 == 0 and opt._coherence_tracker is not None:
            coherences.append(opt._coherence_tracker.global_coherence)
    
    print(f"  Coherence evolution: {[f'{c:.3f}' for c in coherences]}")
    
    # Su Rastrigin la coherence dovrebbe essere più bassa (gradienti caotici)
    mean_coh = np.mean(coherences) if coherences else 0.5
    print(f"  Mean coherence: {mean_coh:.3f}")
    
    if mean_coh < 0.7:
        print("✅ PASS: Lower coherence on multimodal (expected)")
    else:
        print("⚠️  CHECK: High coherence on multimodal - might be oversmoothing")


if __name__ == "__main__":
    test_knn_graph_construction()
    test_gradient_alignment()
    test_least_squares_solver()
    test_coherence_with_real_optimizer()
    test_high_dimensional_stability()
    test_few_leaves_edge_case()
    test_potential_field_sanity()
    test_multimodal_function()
    
    print("\n" + "=" * 60)
    print("ALL COHERENCE TESTS COMPLETE")
    print("=" * 60)
