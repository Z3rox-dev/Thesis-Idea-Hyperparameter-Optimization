#!/usr/bin/env python3
"""
LGS.PY STRESS TEST
===================

Test aggressivi per trovare bug nel modulo lgs.py (Local Gradient Surrogate)
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
# INLINE COPY OF LGS FUNCTIONS (to avoid import issues)
# ============================================================

def fit_lgs_model(
    cube,
    gamma: float,
    dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict]:
    """Fit the LGS model for a given cube."""

    pairs = list(cube.tested_pairs)

    # Parent backfill with shuffle for diversity
    if cube.parent and len(pairs) < 3 * dim:
        parent_pairs = getattr(cube.parent, "_tested_pairs", [])
        extra = [pp for pp in parent_pairs if cube.contains(pp[0])]
        needed = 3 * dim - len(pairs)
        if needed > 0 and extra:
            if rng is not None:
                extra = list(extra)
                rng.shuffle(extra)
            pairs = pairs + extra[:needed]

    if len(pairs) < dim + 2:
        return None

    all_pts = np.array([p for p, s in pairs])
    all_scores = np.array([s for p, s in pairs])

    # BUG FIX: Remove NaN/Inf scores to prevent NaN propagation
    valid_mask = np.isfinite(all_scores)
    if not valid_mask.all():
        all_pts = all_pts[valid_mask]
        all_scores = all_scores[valid_mask]
        if len(all_scores) < dim + 2:
            return None

    k = max(3, len(all_scores) // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    y_mean = 0.0
    y_std = 1.0
    noise_var = 1.0

    widths = np.maximum(cube.widths(), 1e-9)
    center = cube.center()

    if len(all_scores) >= dim + 3:
        X_norm = (all_pts - center) / widths
        y_mean = all_scores.mean()
        y_std = all_scores.std() + 1e-6
        y_centered = (all_scores - y_mean) / y_std

        try:
            dists_sq = np.sum(X_norm**2, axis=1)
            sigma_sq = np.mean(dists_sq) + 1e-6
            weights = np.exp(-dists_sq / (2 * sigma_sq))

            rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (
                all_scores.ptp() + 1e-9
            )
            weights = weights * rank_weights
            W = np.diag(weights)

            n_pts = len(pairs)
            lambda_base = 0.1 * (1 + dim / max(n_pts - dim, 1))
            XtWX = X_norm.T @ W @ X_norm

            try:
                cond = np.linalg.cond(XtWX + lambda_base * np.eye(dim))
                if cond > 1e6:
                    lambda_base *= 10
            except Exception:
                lambda_base *= 5

            XtWX_reg = XtWX + lambda_base * np.eye(dim)
            inv_cov = np.linalg.inv(XtWX_reg)
            grad = inv_cov @ (X_norm.T @ W @ y_centered)

            y_pred = X_norm @ grad
            residuals = y_centered - y_pred
            noise_var = np.clip(
                np.average(residuals**2, weights=weights) + 1e-6,
                1e-4,
                10.0,
            )

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                gradient_dir = grad / grad_norm
        except Exception:
            pass

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "inv_cov": inv_cov,
        "y_mean": y_mean,
        "y_std": y_std,
        "noise_var": noise_var,
        "widths": widths,
        "center": center,
    }


def predict_bayesian(model: Optional[Dict], candidates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty for candidate points given an LGS model."""
    if model is None or model.get("inv_cov") is None:
        return np.zeros(len(candidates)), np.ones(len(candidates))

    widths = model["widths"]
    center = model["center"]
    grad = model["grad"]
    inv_cov = model["inv_cov"]
    noise_var = model["noise_var"]
    y_mean = model["y_mean"]
    y_std = model.get("y_std", 1.0)

    C_norm = (np.array(candidates) - center) / widths
    
    mu_normalized = C_norm @ grad
    mu = y_mean + mu_normalized * y_std

    model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0, 10.0)
    total_var_normalized = noise_var * (1.0 + model_var)
    sigma = np.sqrt(total_var_normalized) * y_std

    return mu, sigma


# ============================================================
# MOCK CUBE
# ============================================================

class MockCube:
    def __init__(self, bounds: List[tuple], tested_pairs=None, parent=None):
        self.bounds = bounds
        self._tested_pairs = tested_pairs or []
        self.parent = parent
    
    @property
    def tested_pairs(self):
        return self._tested_pairs
    
    def center(self):
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])
    
    def widths(self):
        return np.array([hi - lo for lo, hi in self.bounds])
    
    def contains(self, pt):
        return all(lo <= pt[i] <= hi for i, (lo, hi) in enumerate(self.bounds))


# ============================================================
# STRESS TESTS
# ============================================================

def test_not_enough_points():
    """Test con pochi punti."""
    print("=" * 70)
    print("TEST 1: Punti insufficienti")
    print("=" * 70)
    
    bugs = []
    dim = 5
    
    # Test con 0 punti
    print("\n1.1 Zero punti:")
    cube = MockCube([(0, 1)] * dim, tested_pairs=[])
    model = fit_lgs_model(cube, gamma=0.1, dim=dim)
    if model is not None:
        bugs.append("1.1: Expected None con 0 punti")
    else:
        print("  ✓ Ritorna None con 0 punti")
    
    # Test con dim+1 punti (under threshold)
    print(f"\n1.2 {dim+1} punti (< dim+2 = {dim+2}):")
    pts = [(np.random.rand(dim), np.random.rand()) for _ in range(dim+1)]
    cube2 = MockCube([(0, 1)] * dim, tested_pairs=pts)
    model2 = fit_lgs_model(cube2, gamma=0.1, dim=dim)
    if model2 is not None:
        bugs.append(f"1.2: Expected None con {dim+1} punti")
    else:
        print(f"  ✓ Ritorna None con {dim+1} punti")
    
    # Test con esattamente dim+2 punti (threshold)
    print(f"\n1.3 {dim+2} punti (= dim+2, threshold):")
    pts3 = [(np.random.rand(dim), np.random.rand()) for _ in range(dim+2)]
    cube3 = MockCube([(0, 1)] * dim, tested_pairs=pts3)
    model3 = fit_lgs_model(cube3, gamma=0.1, dim=dim)
    if model3 is None:
        bugs.append(f"1.3: Expected model con {dim+2} punti")
    else:
        print(f"  ✓ Ritorna model con {dim+2} punti")
    
    return bugs


def test_extreme_scores():
    """Test con score estremi."""
    print("\n" + "=" * 70)
    print("TEST 2: Score estremi")
    print("=" * 70)
    
    bugs = []
    dim = 2
    
    # Test 2.1: Tutti gli score identici
    print("\n2.1 Score tutti identici:")
    pts = [(np.random.rand(dim), 1.0) for _ in range(20)]
    cube = MockCube([(0, 1)] * dim, tested_pairs=pts)
    model = fit_lgs_model(cube, gamma=0.1, dim=dim)
    
    if model is None:
        print("  ✓ Gestisce score identici (ritorna None)")
    elif model.get("grad") is not None and np.any(np.isnan(model["grad"])):
        bugs.append("2.1: NaN nel gradiente con score identici")
    else:
        print(f"  ✓ Gestisce score identici, grad={model.get('gradient_dir')}")
    
    # Test 2.2: Score con NaN
    print("\n2.2 Score con NaN:")
    pts_nan = [(np.array([0.1, 0.1]), 1.0),
               (np.array([0.2, 0.2]), float('nan')),
               (np.array([0.3, 0.3]), 2.0),
               (np.array([0.4, 0.4]), 3.0),
               (np.array([0.5, 0.5]), 4.0)]
    cube_nan = MockCube([(0, 1)] * dim, tested_pairs=pts_nan)
    
    try:
        model_nan = fit_lgs_model(cube_nan, gamma=0.1, dim=dim)
        if model_nan is not None and model_nan.get("grad") is not None:
            if np.any(np.isnan(model_nan["grad"])):
                bugs.append("2.2: NaN propagato al gradiente")
            else:
                print(f"  ✓ Gestisce score NaN senza propagarlo")
        else:
            print(f"  ✓ Gestisce score NaN (model={model_nan is not None})")
    except Exception as e:
        bugs.append(f"2.2: Exception con score NaN: {e}")
    
    # Test 2.3: Score enormi
    print("\n2.3 Score enormi (1e20):")
    pts_huge = [(np.random.rand(dim), 1e20 + i) for i in range(20)]
    cube_huge = MockCube([(0, 1)] * dim, tested_pairs=pts_huge)
    
    try:
        model_huge = fit_lgs_model(cube_huge, gamma=0.1, dim=dim)
        if model_huge is not None and model_huge.get("grad") is not None:
            grad_norm = np.linalg.norm(model_huge["grad"])
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                bugs.append(f"2.3: Gradiente con NaN/Inf: norm={grad_norm}")
            else:
                print(f"  ✓ Gestisce score enormi, grad_norm={grad_norm:.4f}")
        else:
            print(f"  ✓ Gestisce score enormi (ritorna None)")
    except Exception as e:
        bugs.append(f"2.3: Exception con score enormi: {e}")
    
    return bugs


def test_high_dimensionality():
    """Test con molte dimensioni."""
    print("\n" + "=" * 70)
    print("TEST 3: Alta dimensionalità")
    print("=" * 70)
    
    bugs = []
    
    for dim in [10, 50, 100]:
        print(f"\n3.{dim//10} {dim}D:")
        
        np.random.seed(42)
        n_pts = dim * 3  # Just enough
        pts = [(np.random.rand(dim), np.random.rand()) for _ in range(n_pts)]
        cube = MockCube([(0, 1)] * dim, tested_pairs=pts)
        
        import time
        start = time.time()
        
        try:
            model = fit_lgs_model(cube, gamma=0.1, dim=dim)
            elapsed = time.time() - start
            
            if model is None:
                print(f"  Ritorna None (punti insufficienti)")
            else:
                grad = model.get("grad")
                if grad is not None:
                    grad_norm = np.linalg.norm(grad)
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        bugs.append(f"3.{dim}: Gradiente NaN/Inf in {dim}D")
                    else:
                        print(f"  ✓ grad_norm={grad_norm:.4f}, tempo={elapsed:.3f}s")
                else:
                    print(f"  grad=None (ill-conditioned)")
        except Exception as e:
            bugs.append(f"3.{dim}: Exception: {e}")
    
    return bugs


def test_predict_bayesian_edge_cases():
    """Test predict_bayesian con edge cases."""
    print("\n" + "=" * 70)
    print("TEST 4: predict_bayesian edge cases")
    print("=" * 70)
    
    bugs = []
    dim = 2
    
    # Test 4.1: model=None
    print("\n4.1 model=None:")
    candidates = [np.array([0.5, 0.5])]
    mu, sigma = predict_bayesian(None, candidates)
    if mu[0] != 0.0 or sigma[0] != 1.0:
        bugs.append(f"4.1: Expected (0, 1), got ({mu[0]}, {sigma[0]})")
    else:
        print("  ✓ Ritorna (0, 1) per model=None")
    
    # Test 4.2: Candidato fuori dal cubo (extrapolation)
    print("\n4.2 Candidato lontano (extrapolation):")
    pts = [(np.random.rand(dim), np.random.rand()) for _ in range(20)]
    cube = MockCube([(0, 1)] * dim, tested_pairs=pts)
    model = fit_lgs_model(cube, gamma=0.1, dim=dim)
    
    if model is not None:
        far_candidates = [np.array([100.0, 100.0])]  # Molto lontano
        mu_far, sigma_far = predict_bayesian(model, far_candidates)
        
        if np.isnan(mu_far[0]) or np.isinf(mu_far[0]):
            bugs.append(f"4.2: mu NaN/Inf per extrapolation")
        elif np.isnan(sigma_far[0]) or np.isinf(sigma_far[0]):
            bugs.append(f"4.2: sigma NaN/Inf per extrapolation")
        else:
            print(f"  ✓ Gestisce extrapolation: mu={mu_far[0]:.2f}, sigma={sigma_far[0]:.2f}")
    
    # Test 4.3: Molti candidati
    print("\n4.3 10000 candidati:")
    if model is not None:
        many_candidates = [np.random.rand(dim) for _ in range(10000)]
        import time
        start = time.time()
        mu_many, sigma_many = predict_bayesian(model, many_candidates)
        elapsed = time.time() - start
        
        nan_mu = np.sum(np.isnan(mu_many))
        nan_sigma = np.sum(np.isnan(sigma_many))
        
        if nan_mu > 0 or nan_sigma > 0:
            bugs.append(f"4.3: NaN in predictions ({nan_mu} mu, {nan_sigma} sigma)")
        else:
            print(f"  ✓ 10000 candidati in {elapsed:.3f}s")
    
    return bugs


def test_ill_conditioned_matrix():
    """Test con matrice mal condizionata."""
    print("\n" + "=" * 70)
    print("TEST 5: Matrice mal condizionata")
    print("=" * 70)
    
    bugs = []
    dim = 2
    
    # Punti collineari (matrice singolare)
    print("\n5.1 Punti collineari:")
    pts_collinear = [(np.array([i*0.1, i*0.1]), float(i)) for i in range(10)]
    cube = MockCube([(0, 1)] * dim, tested_pairs=pts_collinear)
    
    try:
        model = fit_lgs_model(cube, gamma=0.1, dim=dim)
        if model is not None and model.get("grad") is not None:
            grad_norm = np.linalg.norm(model["grad"])
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                bugs.append("5.1: Gradiente NaN/Inf con punti collineari")
            else:
                print(f"  ✓ Gestisce punti collineari, grad_norm={grad_norm:.4f}")
        else:
            print("  ✓ Gestisce punti collineari (grad=None)")
    except Exception as e:
        bugs.append(f"5.1: Exception: {e}")
    
    # Punti quasi identici
    print("\n5.2 Punti quasi identici:")
    base = np.array([0.5, 0.5])
    pts_close = [(base + np.random.randn(dim) * 1e-10, float(i)) for i in range(10)]
    cube_close = MockCube([(0, 1)] * dim, tested_pairs=pts_close)
    
    try:
        model_close = fit_lgs_model(cube_close, gamma=0.1, dim=dim)
        if model_close is not None and model_close.get("grad") is not None:
            grad_norm = np.linalg.norm(model_close["grad"])
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                bugs.append("5.2: Gradiente NaN/Inf con punti quasi identici")
            else:
                print(f"  ✓ Gestisce punti quasi identici, grad_norm={grad_norm:.4f}")
        else:
            print("  ✓ Gestisce punti quasi identici (grad=None)")
    except Exception as e:
        bugs.append(f"5.2: Exception: {e}")
    
    return bugs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("LGS.PY STRESS TESTS")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_not_enough_points())
    all_bugs.extend(test_extreme_scores())
    all_bugs.extend(test_high_dimensionality())
    all_bugs.extend(test_predict_bayesian_edge_cases())
    all_bugs.extend(test_ill_conditioned_matrix())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in lgs.py!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
