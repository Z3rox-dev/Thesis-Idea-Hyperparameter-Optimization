#!/usr/bin/env python3
"""
DEBUG: Copula su superfici a scalini (RF-like)
Il sampling copula funziona ancora quando la superficie non è smooth?
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
from scipy import stats
from scipy.stats import norm

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

print("="*80)
print("COPULA SU SUPERFICI A SCALINI (RF-LIKE)")
print("="*80)

# Funzione a scalini
def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2)

dim = 5
rng = np.random.default_rng(42)

# Genera punti iniziali
n_init = 20
X_init = rng.random((n_init, dim))
y_init = np.array([staircase(x) for x in X_init])

# Seleziona elite (top 5)
n_elite = 5
elite_idx = np.argsort(y_init)[:n_elite]
elite_pts = X_init[elite_idx]
elite_scores = y_init[elite_idx]

print(f"\nElite points (top {n_elite}):")
for i, (pt, score) in enumerate(zip(elite_pts, elite_scores)):
    dist = np.linalg.norm(pt - 0.5)
    print(f"  {i}: score={score:.4f}, dist={dist:.4f}")

# Fit copula
print(f"\nFitting copula...")

marginal_params = []
for d in range(dim):
    col = elite_pts[:, d]
    mu = np.mean(col)
    sigma = max(np.std(col), 0.1)
    marginal_params.append((mu, sigma))

# Transform to uniform
U = np.zeros((n_elite, dim))
for d in range(dim):
    mu, sigma = marginal_params[d]
    U[:, d] = norm.cdf(elite_pts[:, d], loc=mu, scale=sigma)
    U[:, d] = np.clip(U[:, d], 1e-6, 1 - 1e-6)

# Gaussianize and get correlation
Z = norm.ppf(U)
Z = Z - Z.mean(axis=0, keepdims=True)

cov = np.cov(Z.T)
std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
corr = cov / (std[:, None] * std[None, :] + 1e-12)
corr = 0.5 * (corr + corr.T)
np.fill_diagonal(corr, 1.0)

# Shrinkage
alpha = 0.5
corr = (1 - alpha) * corr + alpha * np.eye(dim)

# Ensure positive definite
min_eig = np.min(np.linalg.eigvalsh(corr))
if min_eig < 0:
    corr = corr + (-min_eig + 0.01) * np.eye(dim)

L = np.linalg.cholesky(corr)

# Sample from copula
print(f"\nSampling from copula...")
n_samples = 100

copula_samples = []
for _ in range(n_samples):
    z = rng.normal(0, 1, dim) @ L.T
    u = norm.cdf(z)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    
    x = np.zeros(dim)
    for d in range(dim):
        mu, sigma = marginal_params[d]
        x[d] = norm.ppf(u[d], loc=mu, scale=sigma)
        x[d] = np.clip(x[d], 0, 1)
    
    copula_samples.append(x)

copula_samples = np.array(copula_samples)
copula_scores = np.array([staircase(x) for x in copula_samples])
copula_dists = np.linalg.norm(copula_samples - 0.5, axis=1)

print(f"\nCopula samples ({n_samples}):")
print(f"  Score: mean={copula_scores.mean():.4f}, min={copula_scores.min():.4f}")
print(f"  Dist:  mean={copula_dists.mean():.4f}, min={copula_dists.min():.4f}")
print(f"  <0.3 dist: {np.sum(copula_dists < 0.3)}/{n_samples}")

# Compare with top-k perturbation
print(f"\n" + "="*80)
print("COMPARISON WITH TOP-K PERTURBATION")
print("="*80)

topk_samples = []
for _ in range(n_samples):
    base = elite_pts[rng.integers(n_elite)]
    noise = rng.normal(0, 0.15, dim)
    x = np.clip(base + noise, 0, 1)
    topk_samples.append(x)

topk_samples = np.array(topk_samples)
topk_scores = np.array([staircase(x) for x in topk_samples])
topk_dists = np.linalg.norm(topk_samples - 0.5, axis=1)

print(f"\nTop-k samples:")
print(f"  Score: mean={topk_scores.mean():.4f}, min={topk_scores.min():.4f}")
print(f"  Dist:  mean={topk_dists.mean():.4f}, min={topk_dists.min():.4f}")
print(f"  <0.3 dist: {np.sum(topk_dists < 0.3)}/{n_samples}")

# Uniform baseline
uniform_samples = rng.random((n_samples, dim))
uniform_scores = np.array([staircase(x) for x in uniform_samples])
uniform_dists = np.linalg.norm(uniform_samples - 0.5, axis=1)

print(f"\nUniform samples:")
print(f"  Score: mean={uniform_scores.mean():.4f}, min={uniform_scores.min():.4f}")
print(f"  Dist:  mean={uniform_dists.mean():.4f}, min={uniform_dists.min():.4f}")

# Summary
print(f"\n" + "="*80)
print("SUMMARY FOR STAIRCASE FUNCTION")
print("="*80)

print(f"\n  Copula:    best={copula_scores.min():.4f}, mean={copula_scores.mean():.4f}")
print(f"  Top-k:     best={topk_scores.min():.4f}, mean={topk_scores.mean():.4f}")
print(f"  Uniform:   best={uniform_scores.min():.4f}, mean={uniform_scores.mean():.4f}")

# Check: does copula find the quantized optimum?
print(f"\n" + "="*80)
print("ANALYSIS: Finding the quantized optimum")
print("="*80)

# Quantized optimum is at x=0.5 (which falls in bin [0.5, 0.6) after floor)
# Actually for staircase, optimum is at any x in [0.45, 0.55) per dimension
optimal_bin = (np.floor(0.5 * 10) / 10)  # = 0.5
print(f"\nOptimal bin center: {optimal_bin}")
print(f"Points need to be in [{optimal_bin}, {optimal_bin + 0.1}) to hit optimal bin")

# Count how many samples hit the optimal bin
def in_optimal_bin(x):
    x_q = np.floor(x * 10) / 10
    return np.allclose(x_q, 0.5)

copula_optimal = sum(1 for x in copula_samples if in_optimal_bin(x))
topk_optimal = sum(1 for x in topk_samples if in_optimal_bin(x))
uniform_optimal = sum(1 for x in uniform_samples if in_optimal_bin(x))

print(f"\nPoints in optimal bin:")
print(f"  Copula:  {copula_optimal}/{n_samples} ({100*copula_optimal/n_samples:.1f}%)")
print(f"  Top-k:   {topk_optimal}/{n_samples} ({100*topk_optimal/n_samples:.1f}%)")
print(f"  Uniform: {uniform_optimal}/{n_samples} ({100*uniform_optimal/n_samples:.1f}%)")

# Per-dimension analysis
print(f"\n" + "="*80)
print("PER-DIMENSION: How many samples hit optimal bin per dim?")
print("="*80)

for name, samples in [("Copula", copula_samples), ("Top-k", topk_samples)]:
    print(f"\n{name}:")
    for d in range(dim):
        in_bin = sum(1 for x in samples if 0.5 <= x[d] < 0.6)
        print(f"  Dim {d}: {in_bin}/{n_samples} ({100*in_bin/n_samples:.1f}%) in [0.5, 0.6)")
