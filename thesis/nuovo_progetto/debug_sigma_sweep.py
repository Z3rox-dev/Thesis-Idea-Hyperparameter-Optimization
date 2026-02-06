#!/usr/bin/env python3
"""
TEST: Sweep di σ per trovare il valore ottimale su diverse funzioni
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

print("="*80)
print("TEST: σ sweep per copula sampling")
print("="*80)

# Test functions
def sphere(x):
    return np.sum((x - 0.5)**2)

def noisy_staircase(x, rng, n_steps=10, noise_level=0.02):
    x_q = np.floor(x * n_steps) / n_steps
    base = np.sum((x_q - 0.5)**2)
    return base + rng.normal(0, noise_level)

# Simulate copula-like sampling with different σ
dim = 5
n_iter = 50
n_seeds = 3

# Simulate elite points near 0.5
def simulate_optimizer(func, sigma, seed, needs_rng=False):
    """Simple optimization with top-k sampling at given σ"""
    rng = np.random.default_rng(seed)
    
    # Initialize with random points
    points = [rng.random(dim) for _ in range(10)]
    if needs_rng:
        query_rng = np.random.default_rng(seed * 1000)
        scores = [func(p, query_rng) for p in points]
    else:
        scores = [func(p) for p in points]
    
    best = min(scores)
    
    for i in range(n_iter - 10):
        # Top-3 elite
        elite_idx = np.argsort(scores)[:3]
        elite = [points[j] for j in elite_idx]
        
        # Sample around elite with given σ
        base = elite[rng.integers(len(elite))]
        noise = rng.normal(0, sigma, dim)
        candidate = np.clip(base + noise, 0, 1)
        
        if needs_rng:
            query_rng = np.random.default_rng(seed * 1000 + i + 10)
            y = func(candidate, query_rng)
        else:
            y = func(candidate)
        
        points.append(candidate)
        scores.append(y)
        if y < best:
            best = y
    
    return best

sigmas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

print("\nSPHERE (smooth):")
for sigma in sigmas:
    results = [simulate_optimizer(sphere, sigma, s, False) for s in range(n_seeds)]
    print(f"  σ={sigma:.2f}: mean={np.mean(results):.4f}")

print("\nNOISY STAIRCASE:")
for sigma in sigmas:
    results = [simulate_optimizer(noisy_staircase, sigma, s, True) for s in range(n_seeds)]
    print(f"  σ={sigma:.2f}: mean={np.mean(results):.4f}")

print("\nINSIGHT:")
print("- Su smooth: σ piccolo vince (exploitation)")
print("- Su noisy: σ più grande vince (exploration)")
print("- Non esiste σ universale ottimale!")
print("\nCONCLUSIONE:")
print("La copula dovrebbe ADATTARE σ basandosi sul noise osservato:")
print("- Basso noise -> σ tight (exploit)")
print("- Alto noise -> σ ampio (explore)")
