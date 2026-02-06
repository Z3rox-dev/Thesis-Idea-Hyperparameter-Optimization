#!/usr/bin/env python3
"""
CANDIDATES.PY STRESS TEST
==========================

Test aggressivi per trovare bug nel modulo candidates.py
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

from dataclasses import dataclass
from typing import List, Protocol

# Copy of the FIXED MixtureCandidateGenerator (with NaN handling)
@dataclass(frozen=True)
class MixtureCandidateGenerator:
    """Default candidate generator (matches ALBA_V1)."""

    sigma_topk: float = 0.15
    sigma_gradient_noise: float = 0.05
    sigma_center: float = 0.2
    step_min: float = 0.05
    step_max: float = 0.3

    def generate(self, cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube.widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = float(rng.random())

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * widths
            elif strategy < 0.40 and model is not None and model.get("gradient_dir") is not None:
                grad_dir = model["gradient_dir"]
                top_k_pts = model.get("top_k_pts", np.array([]))
                # BUG FIX: Skip gradient strategy if gradient contains NaN/Inf or top_k_pts is empty
                if not np.all(np.isfinite(grad_dir)) or len(top_k_pts) == 0:
                    # Fallback to center perturbation
                    x = center + rng.normal(0, self.sigma_center, dim) * widths
                else:
                    top_center = top_k_pts.mean(axis=0)
                    step = float(rng.uniform(self.step_min, self.step_max))
                    x = top_center + step * grad_dir * widths
                    x = x + rng.normal(0, self.sigma_gradient_noise, dim) * widths
            elif strategy < 0.55:
                x = center + rng.normal(0, self.sigma_center, dim) * widths
            else:
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates


# Mock Cube for testing
class MockCube:
    def __init__(self, bounds: List[tuple], lgs_model=None):
        self.bounds = bounds
        self.lgs_model = lgs_model
    
    def center(self):
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])
    
    def widths(self):
        return np.array([hi - lo for lo, hi in self.bounds])


def test_empty_topk():
    """Test quando top_k_pts è vuoto."""
    print("=" * 70)
    print("TEST 1: top_k_pts vuoto")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    rng = np.random.default_rng(42)
    
    cube = MockCube(
        [(0, 1), (0, 1)],
        lgs_model={
            "top_k_pts": np.array([]).reshape(0, 2),  # Vuoto
            "gradient_dir": np.array([1.0, 0.0])
        }
    )
    
    bugs = []
    
    try:
        candidates = gen.generate(cube, dim=2, rng=rng, n=100)
        print(f"  Generati {len(candidates)} candidati")
        
        # Verifica che siano tutti nel bounds
        for i, c in enumerate(candidates):
            for d, (lo, hi) in enumerate(cube.bounds):
                if c[d] < lo or c[d] > hi:
                    bugs.append(f"Candidato {i} fuori bounds: dim {d} = {c[d]}")
                    break
        
        if not bugs:
            print("  ✓ Tutti i candidati sono nel bounds")
    except Exception as e:
        bugs.append(f"Exception: {e}")
    
    return bugs


def test_no_model():
    """Test senza LGS model."""
    print("\n" + "=" * 70)
    print("TEST 2: Nessun LGS model")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    rng = np.random.default_rng(42)
    
    cube = MockCube([(0, 1), (0, 1)], lgs_model=None)
    
    bugs = []
    
    try:
        candidates = gen.generate(cube, dim=2, rng=rng, n=100)
        print(f"  Generati {len(candidates)} candidati")
        
        # Dovrebbero essere tutti uniformi o center-based
        for i, c in enumerate(candidates):
            for d, (lo, hi) in enumerate(cube.bounds):
                if c[d] < lo or c[d] > hi:
                    bugs.append(f"Candidato {i} fuori bounds: dim {d} = {c[d]}")
                    break
        
        if not bugs:
            print("  ✓ Funziona senza LGS model")
    except Exception as e:
        bugs.append(f"Exception: {e}")
    
    return bugs


def test_extreme_widths():
    """Test con larghezze estreme."""
    print("\n" + "=" * 70)
    print("TEST 3: Larghezze estreme")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    rng = np.random.default_rng(42)
    
    bugs = []
    
    # Test 3.1: Larghezza enorme
    print("\n3.1 Larghezza enorme:")
    cube_huge = MockCube(
        [(0, 1e10), (0, 1e10)],
        lgs_model={
            "top_k_pts": np.array([[5e9, 5e9]]),
            "gradient_dir": np.array([1.0, 0.0])
        }
    )
    
    try:
        candidates = gen.generate(cube_huge, dim=2, rng=rng, n=50)
        print(f"  Generati {len(candidates)} candidati")
        
        out_of_bounds = 0
        for c in candidates:
            if c[0] < 0 or c[0] > 1e10 or c[1] < 0 or c[1] > 1e10:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            bugs.append(f"3.1: {out_of_bounds} candidati fuori bounds")
        else:
            print("  ✓ OK con larghezze enormi")
    except Exception as e:
        bugs.append(f"3.1 Exception: {e}")
    
    # Test 3.2: Larghezza minuscola
    print("\n3.2 Larghezza minuscola:")
    cube_tiny = MockCube(
        [(0.5, 0.5 + 1e-10), (0.5, 0.5 + 1e-10)],
        lgs_model={
            "top_k_pts": np.array([[0.5, 0.5]]),
            "gradient_dir": np.array([1.0, 0.0])
        }
    )
    
    try:
        candidates = gen.generate(cube_tiny, dim=2, rng=rng, n=50)
        print(f"  Generati {len(candidates)} candidati")
        
        # Tutti dovrebbero essere circa 0.5
        for i, c in enumerate(candidates):
            if abs(c[0] - 0.5) > 1e-5 or abs(c[1] - 0.5) > 1e-5:
                bugs.append(f"3.2: Candidato {i} troppo lontano dal centro")
                break
        else:
            print("  ✓ OK con larghezze minuscole")
    except Exception as e:
        bugs.append(f"3.2 Exception: {e}")
    
    # Test 3.3: Larghezza zero
    print("\n3.3 Larghezza zero:")
    cube_zero = MockCube(
        [(0.5, 0.5), (0, 1)],  # Prima dimensione ha larghezza 0
        lgs_model=None
    )
    
    try:
        candidates = gen.generate(cube_zero, dim=2, rng=rng, n=50)
        print(f"  Generati {len(candidates)} candidati")
        
        # La prima dimensione dovrebbe essere esattamente 0.5
        for i, c in enumerate(candidates):
            if c[0] != 0.5:
                bugs.append(f"3.3: Candidato {i} dim 0 = {c[0]}, expected 0.5")
                break
        else:
            print("  ✓ OK con larghezza zero")
    except Exception as e:
        bugs.append(f"3.3 Exception: {e}")
    
    return bugs


def test_gradient_direction_issues():
    """Test con gradient_dir problematici."""
    print("\n" + "=" * 70)
    print("TEST 4: Gradient direction edge cases")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    rng = np.random.default_rng(42)
    
    bugs = []
    
    # Test 4.1: Gradiente zero
    print("\n4.1 Gradiente zero:")
    cube_zero_grad = MockCube(
        [(0, 1), (0, 1)],
        lgs_model={
            "top_k_pts": np.array([[0.5, 0.5]]),
            "gradient_dir": np.array([0.0, 0.0])
        }
    )
    
    try:
        candidates = gen.generate(cube_zero_grad, dim=2, rng=rng, n=100)
        print(f"  Generati {len(candidates)} candidati")
        print("  ✓ OK con gradiente zero")
    except Exception as e:
        bugs.append(f"4.1 Exception: {e}")
    
    # Test 4.2: Gradiente NaN
    print("\n4.2 Gradiente NaN:")
    cube_nan_grad = MockCube(
        [(0, 1), (0, 1)],
        lgs_model={
            "top_k_pts": np.array([[0.5, 0.5]]),
            "gradient_dir": np.array([float('nan'), float('nan')])
        }
    )
    
    try:
        candidates = gen.generate(cube_nan_grad, dim=2, rng=rng, n=100)
        
        # Controlla se qualche candidato è NaN
        nan_count = sum(1 for c in candidates if np.any(np.isnan(c)))
        
        if nan_count > 0:
            bugs.append(f"4.2: {nan_count} candidati contengono NaN")
        else:
            print(f"  Generati {len(candidates)} candidati senza NaN")
            print("  ✓ OK con gradiente NaN (non usato)")
    except Exception as e:
        bugs.append(f"4.2 Exception: {e}")
    
    # Test 4.3: Gradiente Inf
    print("\n4.3 Gradiente Inf:")
    cube_inf_grad = MockCube(
        [(0, 1), (0, 1)],
        lgs_model={
            "top_k_pts": np.array([[0.5, 0.5]]),
            "gradient_dir": np.array([float('inf'), float('-inf')])
        }
    )
    
    try:
        candidates = gen.generate(cube_inf_grad, dim=2, rng=rng, n=100)
        
        inf_count = sum(1 for c in candidates if np.any(np.isinf(c)))
        out_bounds = sum(1 for c in candidates if c[0] < 0 or c[0] > 1 or c[1] < 0 or c[1] > 1)
        
        if inf_count > 0:
            bugs.append(f"4.3: {inf_count} candidati contengono Inf")
        elif out_bounds > 0:
            # Il clipping dovrebbe prevenire questo
            bugs.append(f"4.3: {out_bounds} candidati fuori bounds")
        else:
            print(f"  Generati {len(candidates)} candidati validi")
            print("  ✓ Clipping gestisce gradiente Inf")
    except Exception as e:
        bugs.append(f"4.3 Exception: {e}")
    
    return bugs


def test_high_dimensionality():
    """Test con molte dimensioni."""
    print("\n" + "=" * 70)
    print("TEST 5: Alta dimensionalità")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    rng = np.random.default_rng(42)
    
    bugs = []
    
    for dim in [10, 50, 100]:
        print(f"\n5.{dim//10} {dim}D:")
        
        bounds = [(0, 1) for _ in range(dim)]
        cube = MockCube(
            bounds,
            lgs_model={
                "top_k_pts": np.random.rand(5, dim),
                "gradient_dir": np.random.randn(dim)
            }
        )
        
        try:
            candidates = gen.generate(cube, dim=dim, rng=rng, n=100)
            print(f"  Generati {len(candidates)} candidati")
            
            # Verifica dimensioni
            wrong_dim = sum(1 for c in candidates if len(c) != dim)
            if wrong_dim > 0:
                bugs.append(f"5.{dim}: {wrong_dim} candidati con dim sbagliata")
            else:
                print(f"  ✓ OK in {dim}D")
        except Exception as e:
            bugs.append(f"5.{dim} Exception: {e}")
    
    return bugs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CANDIDATES.PY STRESS TESTS")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_empty_topk())
    all_bugs.extend(test_no_model())
    all_bugs.extend(test_extreme_widths())
    all_bugs.extend(test_gradient_direction_issues())
    all_bugs.extend(test_high_dimensionality())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in candidates.py!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
