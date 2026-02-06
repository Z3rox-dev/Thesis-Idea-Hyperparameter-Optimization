#!/usr/bin/env python3
"""
TEST: Elite center direction vs LGS su multiple seeds
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
print("TEST: Elite Center vs LGS Direction (20 seeds)")
print("="*80)

def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2)

dim = 5
optimum = np.full(dim, 0.5)

lgs_scores = []
elite_scores = []

for seed in range(20):
    rng = np.random.default_rng(seed)
    
    # Generate points
    n_pts = 30
    pts = rng.random((n_pts, dim))
    scores = np.array([staircase(p) for p in pts])
    
    center = np.mean(pts, axis=0)
    opt_dir = (optimum - center) / np.linalg.norm(optimum - center)
    
    # Elite
    k = 5
    elite_idx = np.argsort(scores)[:k]
    elite_pts = pts[elite_idx]
    
    # LGS gradient
    widths = np.ones(dim)
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
        grad_lgs = inv_cov @ (X_norm.T @ W @ y_centered) * y_std
        direction_lgs = -grad_lgs / (np.linalg.norm(grad_lgs) + 1e-9)
    except:
        direction_lgs = np.zeros(dim)
    
    # Elite center direction
    elite_center = np.mean(elite_pts, axis=0)
    direction_elite = elite_center - center
    direction_elite = direction_elite / (np.linalg.norm(direction_elite) + 1e-9)
    
    lgs_cos = np.dot(direction_lgs, opt_dir)
    elite_cos = np.dot(direction_elite, opt_dir)
    
    lgs_scores.append(lgs_cos)
    elite_scores.append(elite_cos)

print(f"\nResults over 20 seeds:")
print(f"  LGS:          mean={np.mean(lgs_scores):.4f}, std={np.std(lgs_scores):.4f}")
print(f"  Elite center: mean={np.mean(elite_scores):.4f}, std={np.std(elite_scores):.4f}")

elite_wins = sum(1 for e, l in zip(elite_scores, lgs_scores) if e > l)
print(f"\n  Elite center wins: {elite_wins}/20 seeds")

if np.mean(elite_scores) > np.mean(lgs_scores):
    print(f"\n  ✅ Elite center is better by {(np.mean(elite_scores) - np.mean(lgs_scores)):.4f}")
else:
    print(f"\n  ❌ LGS is better")
