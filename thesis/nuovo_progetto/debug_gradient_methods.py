#!/usr/bin/env python3
"""
DEBUG: Confronto metodi per calcolare il "gradiente" su superfici a gradini

Metodi:
1. LGS originale (regressione lineare pesata)
2. Elite center direction (elite - center)
3. Elite - non-elite direction
4. Smoothed gradient (media locale)
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if not hasattr(np, "bool"):
    np.bool = bool

print("="*80)
print("CONFRONTO METODI GRADIENTE SU GRADINI")
print("="*80)

# Staircase function
def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2)

dim = 5
rng = np.random.default_rng(42)

# Generate points
n_pts = 30
pts = rng.random((n_pts, dim))
scores = np.array([staircase(p) for p in pts])

# Optimum is at 0.5
optimum = np.full(dim, 0.5)
center = np.mean(pts, axis=0)

print(f"\nData: {n_pts} points in {dim}D")
print(f"Center: {center}")
print(f"Optimum: {optimum}")

# Top-k elite
k = 5
elite_idx = np.argsort(scores)[:k]  # Best = lowest score
elite_pts = pts[elite_idx]
elite_scores = scores[elite_idx]
non_elite_pts = pts[~np.isin(np.arange(n_pts), elite_idx)]

print(f"\nElite (top {k}):")
for i, (pt, sc) in enumerate(zip(elite_pts, elite_scores)):
    dist = np.linalg.norm(pt - optimum)
    print(f"  {i}: score={sc:.4f}, dist_to_opt={dist:.4f}")

# Method 1: LGS linear regression gradient
print(f"\n{'='*60}")
print("METHOD 1: LGS Linear Regression")
print("="*60)

widths = np.ones(dim)  # [0,1] cube
X_norm = pts - center
y_mean = scores.mean()
y_std = scores.std() + 1e-6
y_centered = (scores - y_mean) / y_std

dists_sq = np.sum(X_norm**2, axis=1)
sigma_sq = np.mean(dists_sq) + 1e-6
weights = np.exp(-dists_sq / (2 * sigma_sq))
rank_weights = 1.0 + 0.5 * (scores - scores.min()) / (scores.ptp() + 1e-9)
weights = weights * rank_weights
W = np.diag(weights)

lambda_base = 0.1
XtWX = X_norm.T @ W @ X_norm
XtWX_reg = XtWX + lambda_base * np.eye(dim)
inv_cov = np.linalg.inv(XtWX_reg)
grad_lgs = inv_cov @ (X_norm.T @ W @ y_centered)
grad_lgs = grad_lgs * y_std

grad_lgs_norm = grad_lgs / (np.linalg.norm(grad_lgs) + 1e-9)
# For minimization, we want to go OPPOSITE to gradient
direction_lgs = -grad_lgs_norm

print(f"  Raw gradient: {grad_lgs}")
print(f"  Direction (for min): {direction_lgs}")
cos_sim = np.dot(direction_lgs, (optimum - center) / np.linalg.norm(optimum - center))
print(f"  Cos similarity to optimum direction: {cos_sim:.4f}")

# Method 2: Elite center direction
print(f"\n{'='*60}")
print("METHOD 2: Elite Center Direction")
print("="*60)

elite_center = np.mean(elite_pts, axis=0)
direction_elite = elite_center - center
direction_elite = direction_elite / (np.linalg.norm(direction_elite) + 1e-9)

print(f"  Elite center: {elite_center}")
print(f"  Direction: {direction_elite}")
cos_sim = np.dot(direction_elite, (optimum - center) / np.linalg.norm(optimum - center))
print(f"  Cos similarity to optimum direction: {cos_sim:.4f}")

# Method 3: Elite - non-elite direction
print(f"\n{'='*60}")
print("METHOD 3: Elite vs Non-Elite Direction")
print("="*60)

non_elite_center = np.mean(non_elite_pts, axis=0)
direction_vs = elite_center - non_elite_center
direction_vs = direction_vs / (np.linalg.norm(direction_vs) + 1e-9)

print(f"  Non-elite center: {non_elite_center}")
print(f"  Direction: {direction_vs}")
cos_sim = np.dot(direction_vs, (optimum - center) / np.linalg.norm(optimum - center))
print(f"  Cos similarity to optimum direction: {cos_sim:.4f}")

# Method 4: Smoothed gradient (local averaging)
print(f"\n{'='*60}")
print("METHOD 4: Smoothed Gradient (finite differences)")
print("="*60)

# For each elite point, estimate local gradient via finite differences
# Using nearby points
def local_gradient(pt, all_pts, all_scores, k_neighbors=5):
    """Estimate gradient at pt using k nearest neighbors."""
    dists = np.linalg.norm(all_pts - pt, axis=1)
    nearest_idx = np.argsort(dists)[1:k_neighbors+1]  # Exclude self
    
    # Fit local linear model
    neighbors = all_pts[nearest_idx]
    neighbor_scores = all_scores[nearest_idx]
    
    # Simple finite difference approximation
    delta_x = neighbors - pt
    delta_y = neighbor_scores - staircase(pt)
    
    # Least squares: grad ≈ (X^T X)^-1 X^T y
    try:
        grad = np.linalg.lstsq(delta_x, delta_y, rcond=None)[0]
    except:
        grad = np.zeros(dim)
    
    return grad

# Average gradient over elite points
grads = [local_gradient(pt, pts, scores, k_neighbors=5) for pt in elite_pts]
avg_grad = np.mean(grads, axis=0)
direction_smooth = -avg_grad / (np.linalg.norm(avg_grad) + 1e-9)

print(f"  Averaged gradient: {avg_grad}")
print(f"  Direction (for min): {direction_smooth}")
cos_sim = np.dot(direction_smooth, (optimum - center) / np.linalg.norm(optimum - center))
print(f"  Cos similarity to optimum direction: {cos_sim:.4f}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY: Cosine Similarity to True Optimum Direction")
print("="*80)

methods = [
    ("LGS (linear reg)", direction_lgs),
    ("Elite center", direction_elite),
    ("Elite vs Non-elite", direction_vs),
    ("Smoothed gradient", direction_smooth),
]

opt_dir = (optimum - center) / np.linalg.norm(optimum - center)
for name, direction in methods:
    cos_sim = np.dot(direction, opt_dir)
    print(f"  {name:25s}: {cos_sim:+.4f}")

print("\n  (Closer to +1.0 = better direction towards optimum)")
