#!/usr/bin/env python3
"""
TEST: Ensemble di metodi per il gradiente
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
print("TEST: Ensemble Gradient Methods")
print("="*80)

def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2)

dim = 5
optimum = np.full(dim, 0.5)

def compute_lgs_gradient(pts, scores, center):
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
    try:
        inv_cov = np.linalg.inv(XtWX_reg)
        grad = inv_cov @ (X_norm.T @ W @ y_centered) * y_std
        return -grad / (np.linalg.norm(grad) + 1e-9)
    except:
        return np.zeros(dim)

def compute_elite_direction(pts, scores, center, k=5):
    elite_idx = np.argsort(scores)[:k]
    elite_center = np.mean(pts[elite_idx], axis=0)
    direction = elite_center - center
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        return direction / norm
    return np.zeros(dim)

def compute_ensemble(pts, scores, center, alpha=0.5):
    """
    Ensemble: alpha * LGS + (1-alpha) * Elite
    """
    dir_lgs = compute_lgs_gradient(pts, scores, center)
    dir_elite = compute_elite_direction(pts, scores, center, k=5)
    
    combined = alpha * dir_lgs + (1 - alpha) * dir_elite
    norm = np.linalg.norm(combined)
    if norm > 1e-9:
        return combined / norm
    return dir_elite

# Test on multiple seeds
n_seeds = 100
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

print(f"\nTesting different ensemble weights (α * LGS + (1-α) * Elite):")
print(f"{'α':<10} {'Mean Cos':<10} {'Std':<10}")
print("-"*35)

best_alpha = None
best_mean = -float('inf')

for alpha in alphas:
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        
        n_pts = 50
        pts = rng.random((n_pts, dim))
        scores = np.array([staircase(p) for p in pts])
        
        center = np.mean(pts, axis=0)
        opt_dir = (optimum - center) / np.linalg.norm(optimum - center)
        
        direction = compute_ensemble(pts, scores, center, alpha=alpha)
        results.append(np.dot(direction, opt_dir))
    
    mean_val = np.mean(results)
    std_val = np.std(results)
    if mean_val > best_mean:
        best_mean = mean_val
        best_alpha = alpha
    
    label = f"{alpha:.1f}"
    if alpha == 0.0:
        label += " (Elite only)"
    elif alpha == 1.0:
        label += " (LGS only)"
    print(f"{label:<20} {mean_val:>8.4f} {std_val:>8.4f}")

print(f"\n✅ Best α = {best_alpha} (mean cos = {best_mean:.4f})")
print(f"\nInsight: α=0 (Elite only) o α piccolo tende a funzionare meglio su gradini")
