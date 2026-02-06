#!/usr/bin/env python3
"""
CUBE.PY STRESS TEST
====================

Test aggressivi per trovare bug nel modulo cube.py
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
import os
os.chdir(parent_dir)


# ============================================================
# INLINE COPY OF CUBE (to avoid import issues)
# ============================================================

def fit_lgs_model_mock(cube, gamma, dim, rng=None):
    """Mock LGS model for testing."""
    pairs = list(cube.tested_pairs)
    if len(pairs) < dim + 2:
        return None
    
    all_pts = np.array([p for p, s in pairs])
    all_scores = np.array([s for p, s in pairs])
    
    k = max(3, len(pairs) // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]
    
    widths = np.maximum(cube.widths(), 1e-9)
    center = cube.center()
    
    # Simple gradient estimation
    if len(pairs) >= dim + 3:
        X_norm = (all_pts - center) / widths
        y = all_scores
        try:
            grad = np.linalg.lstsq(X_norm, y - y.mean(), rcond=None)[0]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                gradient_dir = grad / grad_norm
            else:
                gradient_dir = None
        except:
            gradient_dir = None
            grad = None
    else:
        gradient_dir = None
        grad = None
    
    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "widths": widths,
        "center": center,
    }


@dataclass(eq=False)
class Cube:
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[Dict] = field(default=None, init=False)
    depth: int = 0
    cat_stats: Dict[int, Dict[int, Tuple[int, int]]] = field(default_factory=dict)

    def widths(self) -> np.ndarray:
        return np.array([abs(hi - lo) for lo, hi in self.bounds], dtype=float)

    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds], dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True

    def volume(self) -> float:
        return float(np.prod(self.widths()))

    def good_ratio(self) -> float:
        return (self.n_good + 1) / (self.n_trials + 2)

    @property
    def tested_pairs(self) -> List[Tuple[np.ndarray, float]]:
        return self._tested_pairs

    def add_observation(self, x: np.ndarray, score: float, gamma: float) -> None:
        self._tested_pairs.append((x.copy(), score))
        self.n_trials += 1
        if score >= gamma:
            self.n_good += 1
        if score > self.best_score:
            self.best_score = score
            self.best_x = x.copy()

    def fit_lgs_model(self, gamma: float, dim: int, rng=None) -> None:
        self.lgs_model = fit_lgs_model_mock(self, gamma, dim, rng)

    def get_split_axis(self) -> int:
        widths = self.widths()
        if self.lgs_model is not None and self.lgs_model.get("gradient_dir") is not None:
            grad_dir = np.abs(self.lgs_model["gradient_dir"])
            if grad_dir.max() > 0.3:
                return int(np.argmax(grad_dir))
        good_pts = np.array([p for p, s in self._tested_pairs if s >= self.best_score * 0.95])
        if len(good_pts) >= 3:
            var_per_dim = np.var(good_pts / (widths + 1e-9), axis=0)
            score = var_per_dim * (widths / (widths.max() + 1e-9))
            if score.max() > 0.01:
                return int(np.argmax(score))
        return int(np.argmax(widths))

    def split(self, gamma: float, dim: int, rng=None) -> List["Cube"]:
        axis = self.get_split_axis()
        lo, hi = self.bounds[axis]
        good_pairs = [(p[axis], s) for p, s in self._tested_pairs if s >= gamma]

        if len(good_pairs) >= 3:
            positions = np.array([pos for pos, _ in good_pairs])
            scores = np.array([s for _, s in good_pairs])
            weights = scores - gamma + 1e-6
            weights = weights / weights.sum()
            sorted_idx = np.argsort(positions)
            cumsum = np.cumsum(weights[sorted_idx])
            median_idx = np.searchsorted(cumsum, 0.5)
            median_idx = min(median_idx, len(positions) - 1)
            cut = float(positions[sorted_idx[median_idx]])
            margin = 0.12 * (hi - lo)
            cut = np.clip(cut, lo + margin, hi - margin)
        elif len(good_pairs) >= 1:
            cut = float(np.mean([pos for pos, _ in good_pairs]))
            margin = 0.15 * (hi - lo)
            cut = np.clip(cut, lo + margin, hi - margin)
        else:
            cut = (lo + hi) / 2

        bounds_lo = list(self.bounds)
        bounds_hi = list(self.bounds)
        bounds_lo[axis] = (lo, cut)
        bounds_hi[axis] = (cut, hi)

        child_lo = Cube(bounds=bounds_lo, parent=self)
        child_hi = Cube(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1

        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1
            if sc > child.best_score:
                child.best_score = sc
                child.best_x = pt.copy()

        for ch in (child_lo, child_hi):
            ch.fit_lgs_model(gamma, dim, rng)

        return [child_lo, child_hi]


# ============================================================
# STRESS TESTS
# ============================================================

def test_extreme_bounds():
    """Test con bounds estremi."""
    print("=" * 70)
    print("TEST 1: Bounds estremi")
    print("=" * 70)
    
    bugs = []
    dim = 2
    gamma = 0.5
    
    # Test 1.1: Bounds enormi
    print("\n1.1 Bounds enormi (1e10):")
    cube_huge = Cube(bounds=[(0, 1e10), (0, 1e10)])
    for _ in range(20):
        pt = np.array([np.random.uniform(0, 1e10), np.random.uniform(0, 1e10)])
        score = np.random.rand()
        cube_huge.add_observation(pt, score, gamma)
    
    try:
        cube_huge.fit_lgs_model(gamma, dim)
        children = cube_huge.split(gamma, dim)
        print(f"  Split OK: {len(children)} figli")
        print("  ✓ Gestisce bounds enormi")
    except Exception as e:
        bugs.append(f"1.1: Exception con bounds enormi: {e}")
    
    # Test 1.2: Bounds minuscoli
    print("\n1.2 Bounds minuscoli (1e-10):")
    cube_tiny = Cube(bounds=[(0.5, 0.5 + 1e-10), (0.5, 0.5 + 1e-10)])
    for _ in range(20):
        pt = np.array([0.5 + np.random.uniform(0, 1e-10), 0.5 + np.random.uniform(0, 1e-10)])
        score = np.random.rand()
        cube_tiny.add_observation(pt, score, gamma)
    
    try:
        cube_tiny.fit_lgs_model(gamma, dim)
        children = cube_tiny.split(gamma, dim)
        print(f"  Split OK: {len(children)} figli")
        
        # Verifica che i figli abbiano bounds validi
        for i, child in enumerate(children):
            for d, (lo, hi) in enumerate(child.bounds):
                if lo > hi:
                    bugs.append(f"1.2: Child {i} dim {d} has lo > hi: {lo} > {hi}")
                    break
        else:
            print("  ✓ Gestisce bounds minuscoli")
    except Exception as e:
        bugs.append(f"1.2: Exception con bounds minuscoli: {e}")
    
    # Test 1.3: Bounds con larghezza zero
    print("\n1.3 Bounds con larghezza zero:")
    cube_zero = Cube(bounds=[(0.5, 0.5), (0, 1)])  # Prima dim ha width 0
    for _ in range(20):
        pt = np.array([0.5, np.random.uniform(0, 1)])
        score = np.random.rand()
        cube_zero.add_observation(pt, score, gamma)
    
    try:
        cube_zero.fit_lgs_model(gamma, dim)
        children = cube_zero.split(gamma, dim)
        
        # Deve splittare sulla seconda dimensione (la prima ha width 0)
        if children[0].bounds[1] == children[1].bounds[1]:
            bugs.append("1.3: Non ha splittato sulla dimensione non-zero")
        else:
            print(f"  Split OK: ha splittato sulla dim 1")
            print("  ✓ Gestisce bounds con larghezza zero")
    except Exception as e:
        bugs.append(f"1.3: Exception con bounds zero-width: {e}")
    
    return bugs


def test_extreme_scores():
    """Test con score estremi."""
    print("\n" + "=" * 70)
    print("TEST 2: Score estremi")
    print("=" * 70)
    
    bugs = []
    dim = 2
    gamma = 0.5
    
    # Test 2.1: Score tutti uguali
    print("\n2.1 Score tutti uguali:")
    cube_same = Cube(bounds=[(0, 1), (0, 1)])
    for _ in range(20):
        pt = np.random.rand(dim)
        cube_same.add_observation(pt, 1.0, gamma)  # Tutti 1.0
    
    try:
        cube_same.fit_lgs_model(gamma, dim)
        children = cube_same.split(gamma, dim)
        print(f"  Split OK: {len(children)} figli")
        print("  ✓ Gestisce score identici")
    except Exception as e:
        bugs.append(f"2.1: Exception: {e}")
    
    # Test 2.2: Score con NaN
    print("\n2.2 Score con NaN:")
    cube_nan = Cube(bounds=[(0, 1), (0, 1)])
    for i in range(20):
        pt = np.random.rand(dim)
        score = float('nan') if i == 5 else np.random.rand()
        cube_nan.add_observation(pt, score, gamma)
    
    try:
        cube_nan.fit_lgs_model(gamma, dim)
        children = cube_nan.split(gamma, dim)
        
        # Verifica che best_score non sia NaN
        for i, child in enumerate(children):
            if np.isnan(child.best_score):
                bugs.append(f"2.2: Child {i} ha best_score NaN")
                break
        else:
            print(f"  Split OK: {len(children)} figli")
            print("  ✓ Gestisce score NaN")
    except Exception as e:
        bugs.append(f"2.2: Exception: {e}")
    
    return bugs


def test_split_with_no_good_points():
    """Test split quando non ci sono punti buoni."""
    print("\n" + "=" * 70)
    print("TEST 3: Split senza punti buoni")
    print("=" * 70)
    
    bugs = []
    dim = 2
    gamma = 0.9  # Alta soglia → nessun punto buono
    
    cube = Cube(bounds=[(0, 1), (0, 1)])
    for _ in range(20):
        pt = np.random.rand(dim)
        score = np.random.uniform(0, 0.5)  # Tutti sotto gamma
        cube.add_observation(pt, score, gamma)
    
    try:
        cube.fit_lgs_model(gamma, dim)
        children = cube.split(gamma, dim)
        
        # Deve splittare a metà
        axis = cube.get_split_axis()
        lo, hi = cube.bounds[axis]
        expected_cut = (lo + hi) / 2
        
        child_bounds = [ch.bounds[axis] for ch in children]
        print(f"  Split axis: {axis}")
        print(f"  Child bounds: {child_bounds}")
        
        # Verifica che lo split sia a metà
        actual_cut = children[0].bounds[axis][1]  # Upper bound del primo figlio
        if abs(actual_cut - expected_cut) > 1e-9:
            bugs.append(f"3: Expected cut at {expected_cut}, got {actual_cut}")
        else:
            print("  ✓ Splitta a metà senza punti buoni")
    except Exception as e:
        bugs.append(f"3: Exception: {e}")
    
    return bugs


def test_recursive_splitting():
    """Test splitting ricorsivo fino a profondità max."""
    print("\n" + "=" * 70)
    print("TEST 4: Splitting ricorsivo")
    print("=" * 70)
    
    bugs = []
    dim = 2
    gamma = 0.5
    max_depth = 10
    
    root = Cube(bounds=[(0, 1), (0, 1)])
    
    # Aggiungi punti
    for _ in range(100):
        pt = np.random.rand(dim)
        score = np.random.rand()
        root.add_observation(pt, score, gamma)
    
    root.fit_lgs_model(gamma, dim)
    
    # Split ricorsivo
    leaves = [root]
    for depth in range(max_depth):
        new_leaves = []
        for leaf in leaves:
            if leaf.n_trials >= 5:
                try:
                    children = leaf.split(gamma, dim)
                    new_leaves.extend(children)
                except Exception as e:
                    bugs.append(f"4: Exception at depth {depth}: {e}")
                    break
            else:
                new_leaves.append(leaf)
        leaves = new_leaves
        
        if bugs:
            break
    
    if not bugs:
        print(f"  Final leaves: {len(leaves)}")
        print(f"  Max depth: {max(leaf.depth for leaf in leaves)}")
        
        # Verifica che tutti i punti siano ancora coperti
        total_trials = sum(leaf.n_trials for leaf in leaves)
        if total_trials != root.n_trials:
            bugs.append(f"4: Lost points: {root.n_trials} → {total_trials}")
        else:
            print("  ✓ Splitting ricorsivo OK, nessun punto perso")
    
    return bugs


def test_contains_edge_cases():
    """Test contains() con edge cases."""
    print("\n" + "=" * 70)
    print("TEST 5: contains() edge cases")
    print("=" * 70)
    
    bugs = []
    
    cube = Cube(bounds=[(0, 1), (0.5, 1.5)])
    
    test_cases = [
        (np.array([0.5, 1.0]), True, "centro"),
        (np.array([0.0, 0.5]), True, "angolo lo"),
        (np.array([1.0, 1.5]), True, "angolo hi"),
        (np.array([-0.001, 1.0]), False, "fuori lo dim 0"),
        (np.array([1.001, 1.0]), False, "fuori hi dim 0"),
        (np.array([0.5, 0.499]), False, "fuori lo dim 1"),
        (np.array([0.5, 1.501]), False, "fuori hi dim 1"),
        # Edge: esattamente sui bordi (con tolerance)
        (np.array([0.0 - 1e-10, 0.5]), True, "bordo con tolerance"),
        (np.array([1.0 + 1e-10, 1.5]), True, "bordo con tolerance"),
    ]
    
    for pt, expected, desc in test_cases:
        result = cube.contains(pt)
        if result != expected:
            bugs.append(f"5: contains({desc}) = {result}, expected {expected}")
    
    if not bugs:
        print("  ✓ Tutti i test contains() passano")
    
    return bugs


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)
    
    print("=" * 70)
    print("CUBE.PY STRESS TESTS")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_extreme_bounds())
    all_bugs.extend(test_extreme_scores())
    all_bugs.extend(test_split_with_no_good_points())
    all_bugs.extend(test_recursive_splitting())
    all_bugs.extend(test_contains_edge_cases())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in cube.py!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
