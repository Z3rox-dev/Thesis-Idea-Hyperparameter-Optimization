#!/usr/bin/env python3
"""
TEST: Gradient smoothing - media pesata su più punti

Idea: invece di un singolo gradiente dalla regressione,
facciamo una media pesata dei "gradienti locali" stimati
su gruppi di punti vicini.
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
print("TEST: Gradient Smoothing Methods")
print("="*80)

def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2)

dim = 5
optimum = np.full(dim, 0.5)

def compute_lgs_gradient(pts, scores, center):
    """Standard LGS gradient."""
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
    """Elite center direction."""
    elite_idx = np.argsort(scores)[:k]
    elite_center = np.mean(pts[elite_idx], axis=0)
    direction = elite_center - center
    return direction / (np.linalg.norm(direction) + 1e-9)

def compute_smoothed_gradient(pts, scores, center, n_clusters=5):
    """
    Smoothed gradient: 
    1. Divide points into clusters based on score
    2. Compute direction from worst cluster to best cluster
    """
    n_pts = len(pts)
    sorted_idx = np.argsort(scores)  # Best (lowest) first
    cluster_size = n_pts // n_clusters
    
    # Best cluster (lowest scores) and worst cluster (highest scores)
    best_idx = sorted_idx[:cluster_size]
    worst_idx = sorted_idx[-cluster_size:]
    
    best_center = np.mean(pts[best_idx], axis=0)
    worst_center = np.mean(pts[worst_idx], axis=0)
    
    direction = best_center - worst_center
    return direction / (np.linalg.norm(direction) + 1e-9)

def compute_weighted_centroid_direction(pts, scores, center):
    """
    Weighted centroid direction:
    Weight each point inversely by its score (lower score = higher weight)
    """
    # Transform scores to weights (lower = better = higher weight)
    min_score = scores.min()
    max_score = scores.max()
    weights = (max_score - scores) / (max_score - min_score + 1e-9)
    weights = weights / weights.sum()
    
    weighted_center = np.average(pts, axis=0, weights=weights)
    direction = weighted_center - center
    return direction / (np.linalg.norm(direction) + 1e-9)

# Test on multiple seeds
n_seeds = 50
results = {
    'LGS': [],
    'Elite center': [],
    'Smoothed (clusters)': [],
    'Weighted centroid': [],
}

for seed in range(n_seeds):
    rng = np.random.default_rng(seed)
    
    n_pts = 50  # More points
    pts = rng.random((n_pts, dim))
    scores = np.array([staircase(p) for p in pts])
    
    center = np.mean(pts, axis=0)
    opt_dir = (optimum - center) / np.linalg.norm(optimum - center)
    
    # Compute all directions
    dir_lgs = compute_lgs_gradient(pts, scores, center)
    dir_elite = compute_elite_direction(pts, scores, center, k=5)
    dir_smooth = compute_smoothed_gradient(pts, scores, center, n_clusters=5)
    dir_weighted = compute_weighted_centroid_direction(pts, scores, center)
    
    results['LGS'].append(np.dot(dir_lgs, opt_dir))
    results['Elite center'].append(np.dot(dir_elite, opt_dir))
    results['Smoothed (clusters)'].append(np.dot(dir_smooth, opt_dir))
    results['Weighted centroid'].append(np.dot(dir_weighted, opt_dir))

print(f"\nResults over {n_seeds} seeds ({dim}D staircase, 50 pts):")
print(f"{'Method':<25} {'Mean':>8} {'Std':>8} {'Best':>8}")
print("-"*55)

best_method = None
best_mean = -float('inf')
for name, values in results.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    if mean_val > best_mean:
        best_mean = mean_val
        best_method = name
    print(f"{name:<25} {mean_val:>8.4f} {std_val:>8.4f}")

print(f"\n✅ Best method: {best_method} (mean={best_mean:.4f})")
