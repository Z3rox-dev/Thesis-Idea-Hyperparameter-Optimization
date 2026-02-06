#!/usr/bin/env python3
"""
DEBUG: Analisi del REALE sampling della copula
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
from scipy import stats

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

print("="*80)
print("ANALISI SAMPLING REALE COPULA")
print("="*80)

# Simula dati elite (già vicini all'ottimo)
dim = 5
elite_pts = np.array([
    [0.5, 0.6, 0.5, 0.4, 0.5],
    [0.4, 0.5, 0.6, 0.5, 0.4],
    [0.6, 0.4, 0.5, 0.5, 0.6],
])

print(f"\nElite points (3 points):")
for i, pt in enumerate(elite_pts):
    y = np.sum((pt - 0.5)**2)
    dist = np.linalg.norm(pt - 0.5)
    print(f"  {i}: {pt} -> y={y:.4f}, dist={dist:.4f}")

# Compute marginal parameters (mu, sigma per dimension)
marginal_params = []
for d in range(dim):
    col = elite_pts[:, d]
    mu = np.mean(col)
    sigma = np.std(col)
    min_sigma = 0.1  # Minimum width
    sigma = max(sigma, min_sigma)
    marginal_params.append((mu, sigma))

print(f"\nMarginal parameters:")
for d, (mu, sigma) in enumerate(marginal_params):
    print(f"  Dim {d}: mu={mu:.4f}, sigma={sigma:.4f}")

# Compute correlation matrix from elite
print(f"\nComputing copula correlation matrix...")

# Transform to uniform via rank
n_elite = len(elite_pts)
U = np.zeros((n_elite, dim))
for d in range(dim):
    mu, sigma = marginal_params[d]
    U[:, d] = stats.norm.cdf(elite_pts[:, d], loc=mu, scale=sigma)
    U[:, d] = np.clip(U[:, d], 1e-6, 1 - 1e-6)

# Gaussianize
from scipy.stats import norm
Z = norm.ppf(U)
Z = Z - Z.mean(axis=0, keepdims=True)

# Estimate correlation
cov = np.cov(Z.T)
std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
corr = cov / (std[:, None] * std[None, :] + 1e-12)
corr = 0.5 * (corr + corr.T)
np.fill_diagonal(corr, 1.0)

# Shrinkage
alpha = 0.5
corr = (1 - alpha) * corr + alpha * np.eye(dim)

print(f"\nCorrelation matrix (after shrinkage):")
print(corr)

# Cholesky
L = np.linalg.cholesky(corr)

# Sample from copula
print(f"\n" + "="*80)
print("SAMPLING FROM COPULA")
print("="*80)

rng = np.random.default_rng(42)
n_samples = 100

samples = []
for _ in range(n_samples):
    # Sample from correlated Gaussian
    z = rng.normal(0, 1, dim) @ L.T
    u = norm.cdf(z)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    
    # Transform back via marginal inverse CDF
    x = np.zeros(dim)
    for d in range(dim):
        mu, sigma = marginal_params[d]
        x[d] = norm.ppf(u[d], loc=mu, scale=sigma)
        # Clip to [0, 1]
        x[d] = np.clip(x[d], 0, 1)
    
    samples.append(x)

samples = np.array(samples)

print(f"\nCopula samples (100):")
print(f"  Mean: {samples.mean(axis=0)}")
print(f"  Std:  {samples.std(axis=0)}")

# Quality analysis
scores = np.sum((samples - 0.5)**2, axis=1)
dists = np.linalg.norm(samples - 0.5, axis=1)

print(f"\n  Score: mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
print(f"  Dist:  mean={dists.mean():.4f}, min={dists.min():.4f}, max={dists.max():.4f}")
print(f"  <0.2 dist: {np.sum(dists < 0.2)}/{n_samples}")
print(f"  <0.15 dist: {np.sum(dists < 0.15)}/{n_samples}")

# Compare with simple perturbation
print(f"\n" + "="*80)
print("COMPARISON WITH TOP-K PERTURBATION (sigma=0.15)")
print("="*80)

topk_samples = []
for _ in range(n_samples):
    base = elite_pts[rng.integers(len(elite_pts))]
    noise = rng.normal(0, 0.15, dim)
    x = np.clip(base + noise, 0, 1)
    topk_samples.append(x)

topk_samples = np.array(topk_samples)
topk_scores = np.sum((topk_samples - 0.5)**2, axis=1)
topk_dists = np.linalg.norm(topk_samples - 0.5, axis=1)

print(f"\nTop-k samples:")
print(f"  Mean: {topk_samples.mean(axis=0)}")
print(f"  Std:  {topk_samples.std(axis=0)}")
print(f"  Score: mean={topk_scores.mean():.4f}, min={topk_scores.min():.4f}")
print(f"  Dist:  mean={topk_dists.mean():.4f}, min={topk_dists.min():.4f}")
print(f"  <0.2 dist: {np.sum(topk_dists < 0.2)}/{n_samples}")

# Tight perturbation (what copula effectively does)
print(f"\n" + "="*80)
print("COMPARISON WITH TIGHTER PERTURBATION (sigma=0.08)")
print("="*80)

tight_samples = []
for _ in range(n_samples):
    base = elite_pts[rng.integers(len(elite_pts))]
    noise = rng.normal(0, 0.08, dim)
    x = np.clip(base + noise, 0, 1)
    tight_samples.append(x)

tight_samples = np.array(tight_samples)
tight_scores = np.sum((tight_samples - 0.5)**2, axis=1)
tight_dists = np.linalg.norm(tight_samples - 0.5, axis=1)

print(f"\nTight perturbation samples:")
print(f"  Mean: {tight_samples.mean(axis=0)}")
print(f"  Std:  {tight_samples.std(axis=0)}")
print(f"  Score: mean={tight_scores.mean():.4f}, min={tight_scores.min():.4f}")
print(f"  Dist:  mean={tight_dists.mean():.4f}, min={tight_dists.min():.4f}")
print(f"  <0.2 dist: {np.sum(tight_dists < 0.2)}/{n_samples}")

# Summary comparison
print(f"\n" + "="*80)
print("SUMMARY: Best sample found")
print("="*80)

print(f"\n  Copula:           best_score={scores.min():.4f}, dist={dists.min():.4f}")
print(f"  Top-k (σ=0.15):   best_score={topk_scores.min():.4f}, dist={topk_dists.min():.4f}")
print(f"  Tight (σ=0.08):   best_score={tight_scores.min():.4f}, dist={tight_dists.min():.4f}")

# What's the effective sigma of copula?
print(f"\n  Copula effective σ per dim: {samples.std(axis=0)}")
print(f"  (compare with marginal σ:    {[s for m, s in marginal_params]})")
