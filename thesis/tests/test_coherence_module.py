#!/usr/bin/env python3
"""
Test Suite for ALBA Coherence Module

Comprehensive tests to verify the geometric coherence implementation:
1. Unit tests for kNN graph construction
2. Gradient alignment computation
3. Coherence score range validation
4. Cache update logic
5. Exploit/explore gating
6. Categorical dimension handling
7. Edge cases
8. Numerical accuracy
9. Performance checks

Run with: python3 tests/test_coherence_module.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_coherence import ALBA, CoherenceTracker, compute_coherence_scores
from alba_framework_coherence.coherence import _build_knn_graph, _compute_predicted_drops


class MockCube:
    """Mock cube for testing without full ALBA infrastructure."""
    
    def __init__(self, center_val, grad=None):
        self._center = np.array(center_val)
        self.n_trials = 10
        self.lgs_model = {'grad': grad} if grad is not None else None
    
    def center(self):
        return self._center


def test_knn_graph_construction():
    """Test kNN graph construction."""
    print("[TEST 1] kNN Graph Construction")
    
    cubes = [
        MockCube([0.0, 0.0]),
        MockCube([1.0, 0.0]),
        MockCube([0.0, 1.0]),
        MockCube([1.0, 1.0]),
    ]
    
    edges = _build_knn_graph(cubes, k=2)
    assert len(edges) == 8, f'Expected 8 edges, got {len(edges)}'
    
    for i in range(4):
        outgoing = [e for e in edges if e[0] == i]
        assert len(outgoing) == 2, f'Cube {i} has {len(outgoing)} outgoing edges'
    
    print("  ✓ PASSED")


def test_gradient_alignment():
    """Test gradient alignment computation."""
    print("[TEST 2] Gradient Alignment Computation")
    
    test_cases = [
        ('Parallel', np.array([1.0, 0.0]), np.array([1.0, 0.0]), 1.0),
        ('Orthogonal', np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0),
        ('Opposite', np.array([1.0, 0.0]), np.array([-1.0, 0.0]), -1.0),
        ('45°', np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.cos(np.pi/4)),
    ]
    
    for name, g1, g2, expected in test_cases:
        cubes = [MockCube([0.0, 0.0], grad=g1), MockCube([1.0, 0.0], grad=g2)]
        edges = _build_knn_graph(cubes, k=1)
        _, _, alignments, _ = _compute_predicted_drops(cubes, edges, categorical_dims=[])
        
        assert len(alignments) > 0, f'{name}: No alignments computed'
        error = abs(alignments[0] - expected)
        assert error < 0.01, f'{name}: expected {expected}, got {alignments[0]}'
    
    print("  ✓ PASSED")


def test_coherence_score_range():
    """Test coherence scores are in [0, 1]."""
    print("[TEST 3] Coherence Score Range [0, 1]")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    opt = ALBA(bounds=[(-5, 5)] * 6, maximize=False, seed=42, total_budget=100,
               use_coherence_gating=True)
    for _ in range(100):
        x = opt.ask()
        opt.tell(x, sphere(x))
    
    scores, global_coh, q60, q80 = compute_coherence_scores(opt.leaves, categorical_dims=[])
    
    assert all(0.0 <= s <= 1.0 for s in scores.values()), 'Scores outside [0, 1]'
    assert 0.0 <= global_coh <= 1.0, 'Global coherence outside [0, 1]'
    assert q60 <= q80, 'Q60 should be <= Q80'
    
    print("  ✓ PASSED")


def test_cache_updates():
    """Test coherence cache updates."""
    print("[TEST 4] Coherence Cache Updates")
    
    tracker = CoherenceTracker(categorical_dims=[], k_neighbors=4,
                               update_interval=5, min_leaves_for_coherence=3)
    
    stats = tracker.get_statistics()
    assert stats['n_leaves_cached'] == 0, 'Cache should be empty initially'
    
    mock_leaves = [MockCube([i*0.5, 0.0], grad=np.array([1.0, 0.0])) for i in range(5)]
    tracker.update(mock_leaves, iteration=0, force=True)
    
    stats = tracker.get_statistics()
    assert stats['n_leaves_cached'] == 5, 'Cache should have 5 leaves'
    
    print("  ✓ PASSED")


def test_exploit_explore_gating():
    """Test exploit/explore gating logic."""
    print("[TEST 5] Exploit/Explore Gating Logic")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    opt = ALBA(bounds=[(-5, 5)] * 8, maximize=False, seed=123, total_budget=150,
               use_coherence_gating=True)
    
    exploit_decisions = 0
    explore_decisions = 0
    
    for i in range(150):
        x = opt.ask()
        opt.tell(x, sphere(x))
        
        if i >= 50 and opt._coherence_tracker is not None and len(opt.leaves) >= 5:
            for leaf in opt.leaves[:3]:
                if opt._coherence_tracker.should_exploit(leaf, opt.leaves):
                    exploit_decisions += 1
                else:
                    explore_decisions += 1
    
    total = exploit_decisions + explore_decisions
    if total > 0:
        exploit_ratio = exploit_decisions / total
        assert exploit_ratio > 0.3, f'Expected exploit ratio > 0.3, got {exploit_ratio}'
    
    print("  ✓ PASSED")


def test_categorical_handling():
    """Test categorical dimension masking."""
    print("[TEST 6] Categorical Dimension Handling")
    
    cubes = [
        MockCube([0.0, 0.0, 0.5], grad=np.array([1.0, 0.0, 999.0])),
        MockCube([1.0, 0.0, 0.5], grad=np.array([1.0, 0.0, -999.0])),
    ]
    
    edges = _build_knn_graph(cubes, k=1)
    _, _, alignments, _ = _compute_predicted_drops(cubes, edges, categorical_dims=[(2, 2)])
    
    assert alignments[0] > 0.9, f'Expected high alignment after masking, got {alignments[0]}'
    
    print("  ✓ PASSED")


def test_few_leaves_edge_case():
    """Test handling of few leaves."""
    print("[TEST 7] Edge Case - Few Leaves")
    
    few_leaves = [MockCube([0.0, 0.0], grad=np.array([1.0, 0.0]))]
    scores, global_coh, q60, q80 = compute_coherence_scores(few_leaves, categorical_dims=[])
    
    assert global_coh == 0.5, 'Expected neutral coherence for 1 leaf'
    assert scores[0] == 0.5, 'Expected neutral score for single leaf'
    
    print("  ✓ PASSED")


def test_consistency_same_seed():
    """Test consistency with same seed."""
    print("[TEST 8] Consistency Between Runs (Same Seed)")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    results = []
    for _ in range(3):
        opt = ALBA(bounds=[(-5, 5)] * 5, maximize=False, seed=42, total_budget=80,
                   use_coherence_gating=True)
        for _ in range(80):
            x = opt.ask()
            opt.tell(x, sphere(x))
        results.append(opt.best_y)
    
    assert all(abs(r - results[0]) < 1e-9 for r in results), 'Results should be identical'
    
    print("  ✓ PASSED")


def test_statistics_integration():
    """Test statistics integration."""
    print("[TEST 9] Statistics Integration")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    opt = ALBA(bounds=[(-5, 5)] * 6, maximize=False, seed=42, total_budget=100,
               use_coherence_gating=True)
    for _ in range(100):
        x = opt.ask()
        opt.tell(x, sphere(x))
    
    stats = opt.get_statistics()
    assert 'coherence' in stats, 'coherence should be in statistics'
    
    required_keys = ['global_coherence', 'q60_threshold', 'q80_threshold', 'n_leaves_cached']
    for key in required_keys:
        assert key in stats['coherence'], f'{key} missing from coherence stats'
    
    print("  ✓ PASSED")


def test_performance():
    """Test performance is acceptable."""
    print("[TEST 10] Performance Check")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    opt = ALBA(bounds=[(-5, 5)] * 10, maximize=False, seed=42, total_budget=200,
               use_coherence_gating=True)
    
    start = time.time()
    for _ in range(200):
        x = opt.ask()
        opt.tell(x, sphere(x))
    elapsed = time.time() - start
    
    assert elapsed < 30, f'Performance too slow: {elapsed:.3f}s'
    print(f"  200 iterations in {elapsed:.3f}s ({elapsed/200*1000:.2f}ms per iter)")
    print("  ✓ PASSED")


def test_numerical_accuracy():
    """Test numerical accuracy of cosine similarity."""
    print("[TEST 11] Numerical Accuracy (High-D)")
    
    dim = 20
    np.random.seed(42)
    v1 = np.random.randn(dim)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.random.randn(dim)
    v2 = v2 / np.linalg.norm(v2)
    
    expected = np.dot(v1, v2)
    
    cubes = [MockCube(np.zeros(dim), grad=v1), MockCube(np.ones(dim), grad=v2)]
    edges = _build_knn_graph(cubes, k=1)
    _, _, alignments, _ = _compute_predicted_drops(cubes, edges, categorical_dims=[])
    
    error = abs(alignments[0] - expected)
    assert error < 1e-6, f'High-D alignment error too large: {error}'
    
    print("  ✓ PASSED")


def test_no_nan_inf():
    """Test no NaN/Inf values across configurations."""
    print("[TEST 12] No NaN/Inf Values")
    
    def sphere(x):
        return float(np.sum(x**2))
    
    configs = [
        {'bounds': [(-5, 5)] * 3, 'budget': 50},
        {'bounds': [(-100, 100)] * 5, 'budget': 80},
        {'bounds': [(-0.01, 0.01)] * 4, 'budget': 60},
        {'bounds': [(-5, 5)] * 15, 'budget': 100},
    ]
    
    for i, cfg in enumerate(configs):
        opt = ALBA(bounds=cfg['bounds'], maximize=False, seed=42+i,
                   total_budget=cfg['budget'], use_coherence_gating=True)
        for _ in range(cfg['budget']):
            x = opt.ask()
            opt.tell(x, sphere(x))
        
        scores, global_coh, q60, q80 = compute_coherence_scores(opt.leaves, categorical_dims=[])
        
        all_finite = all(np.isfinite(s) for s in scores.values())
        all_finite = all_finite and np.isfinite(global_coh) and np.isfinite(q60) and np.isfinite(q80)
        
        assert all_finite, f'Config {i}: Found NaN/Inf values'
    
    print("  ✓ PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("COHERENCE MODULE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_knn_graph_construction,
        test_gradient_alignment,
        test_coherence_score_range,
        test_cache_updates,
        test_exploit_explore_gating,
        test_categorical_handling,
        test_few_leaves_edge_case,
        test_consistency_same_seed,
        test_statistics_integration,
        test_performance,
        test_numerical_accuracy,
        test_no_nan_inf,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
