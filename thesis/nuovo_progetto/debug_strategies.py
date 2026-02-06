#!/usr/bin/env python3
"""
DEBUG: Analisi delle STRATEGIE di campionamento usate
Vogliamo vedere: top-k vs gradient vs center vs uniform vs copula
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
np.random.seed(42)

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

# Simuliamo le strategie di campionamento
print("="*80)
print("ANALISI STRATEGIE DI CAMPIONAMENTO")
print("="*80)

# Simula 10000 random draws per vedere la distribuzione delle strategie
rng = np.random.default_rng(42)
n_samples = 10000

# LGS distribution: 25% top-k, 15% gradient, 15% center, 45% uniform
lgs_strats = []
for _ in range(n_samples):
    s = rng.random()
    if s < 0.25:
        lgs_strats.append('top-k')
    elif s < 0.40:
        lgs_strats.append('gradient')
    elif s < 0.55:
        lgs_strats.append('center')
    else:
        lgs_strats.append('uniform')

print("\nLGS Strategy Distribution:")
for strat in ['top-k', 'gradient', 'center', 'uniform']:
    count = lgs_strats.count(strat)
    print(f"  {strat:10s}: {count:5d} ({100*count/n_samples:.1f}%)")

# Copula distribution: 25% top-k, 15% gradient, 15% center, 10% copula, 35% uniform
cop_strats = []
for _ in range(n_samples):
    s = rng.random()
    if s < 0.25:
        cop_strats.append('top-k')
    elif s < 0.40:
        cop_strats.append('gradient')
    elif s < 0.55:
        cop_strats.append('center')
    elif s < 0.65:
        cop_strats.append('copula')
    else:
        cop_strats.append('uniform')

print("\nCopula Strategy Distribution:")
for strat in ['top-k', 'gradient', 'center', 'copula', 'uniform']:
    count = cop_strats.count(strat)
    print(f"  {strat:10s}: {count:5d} ({100*count/n_samples:.1f}%)")

# Analisi della qualità di ogni strategia
print("\n" + "="*80)
print("QUALITÀ DI OGNI STRATEGIA (su Sphere function)")
print("="*80)

dim = 5
bounds = np.array([[0, 1]] * dim)

# Simula dati osservati (10 punti random)
np.random.seed(42)
X_obs = np.random.rand(10, dim)
y_obs = np.sum((X_obs - 0.5)**2, axis=1)

# Sort by score (lower is better)
order = np.argsort(y_obs)
top_k_pts = X_obs[order[:3]]  # Top 3
center = np.mean(X_obs, axis=0)

# Gradient direction (simplified: direction from worst to best)
best_pt = X_obs[order[0]]
worst_pt = X_obs[order[-1]]
gradient_dir = best_pt - worst_pt
gradient_dir = gradient_dir / (np.linalg.norm(gradient_dir) + 1e-8)

print(f"\nObserved data:")
print(f"  Best point: {best_pt[:3]}... -> y={y_obs[order[0]]:.4f}")
print(f"  Worst point: {worst_pt[:3]}... -> y={y_obs[order[-1]]:.4f}")
print(f"  Center: {center[:3]}...")
print(f"  Gradient dir: {gradient_dir[:3]}...")

# Generate samples from each strategy
n_test = 100
rng = np.random.default_rng(123)

# Top-k perturbation
topk_samples = []
for _ in range(n_test):
    base = top_k_pts[rng.integers(len(top_k_pts))]
    noise = rng.normal(0, 0.15, dim)
    x = np.clip(base + noise, 0, 1)
    topk_samples.append(x)

# Gradient
grad_samples = []
for _ in range(n_test):
    top_center = top_k_pts.mean(axis=0)
    step = rng.uniform(0.05, 0.3)
    noise = rng.normal(0, 0.05, dim)
    x = np.clip(top_center + step * gradient_dir + noise, 0, 1)
    grad_samples.append(x)

# Center perturbation
center_samples = []
for _ in range(n_test):
    noise = rng.normal(0, 0.2, dim)
    x = np.clip(center + noise, 0, 1)
    center_samples.append(x)

# Uniform
uniform_samples = []
for _ in range(n_test):
    x = rng.uniform(0, 1, dim)
    uniform_samples.append(x)

# Copula-like (perturb elite with correlation)
# For simplicity, use smaller noise around top points
copula_samples = []
for _ in range(n_test):
    base = top_k_pts[rng.integers(len(top_k_pts))]
    # Tighter noise than top-k (copula is more concentrated)
    noise = rng.normal(0, 0.08, dim)
    x = np.clip(base + noise, 0, 1)
    copula_samples.append(x)

# Evaluate
def eval_samples(samples, name):
    scores = [np.sum((x - 0.5)**2) for x in samples]
    dists = [np.linalg.norm(x - 0.5) for x in samples]
    print(f"\n{name}:")
    print(f"  Score: mean={np.mean(scores):.4f}, min={np.min(scores):.4f}")
    print(f"  Dist:  mean={np.mean(dists):.4f}, min={np.min(dists):.4f}")
    print(f"  <0.2 dist: {sum(1 for d in dists if d < 0.2)}/{n_test}")
    return np.min(scores)

best_topk = eval_samples(topk_samples, "Top-k Perturbation")
best_grad = eval_samples(grad_samples, "Gradient Direction")
best_center = eval_samples(center_samples, "Center Perturbation")
best_uniform = eval_samples(uniform_samples, "Uniform Random")
best_copula = eval_samples(copula_samples, "Copula-like (tight noise)")

print("\n" + "="*80)
print("RANKING STRATEGIE (by best sample found)")
print("="*80)

results = [
    ("Copula-like", best_copula),
    ("Top-k", best_topk),
    ("Gradient", best_grad),
    ("Center", best_center),
    ("Uniform", best_uniform),
]
results.sort(key=lambda x: x[1])

for i, (name, score) in enumerate(results, 1):
    print(f"  {i}. {name:20s}: {score:.4f}")

# Cosa cambia se abbiamo punti osservati vicini all'ottimo?
print("\n" + "="*80)
print("CON PUNTI OSSERVATI VICINI ALL'OTTIMO")
print("="*80)

# Nuovi dati: alcuni punti vicini a 0.5
X_obs2 = np.array([
    [0.4, 0.5, 0.6, 0.5, 0.4],
    [0.5, 0.6, 0.5, 0.4, 0.5],
    [0.6, 0.4, 0.5, 0.5, 0.6],
    [0.3, 0.3, 0.3, 0.7, 0.7],
    [0.8, 0.2, 0.8, 0.2, 0.8],
])
y_obs2 = np.sum((X_obs2 - 0.5)**2, axis=1)

order2 = np.argsort(y_obs2)
top_k_pts2 = X_obs2[order2[:3]]

print(f"\nNew top-3 points:")
for i in range(3):
    idx = order2[i]
    print(f"  {X_obs2[idx][:3]}... -> y={y_obs2[idx]:.4f}")

# Copula-like with new data
copula2_samples = []
for _ in range(n_test):
    base = top_k_pts2[rng.integers(len(top_k_pts2))]
    noise = rng.normal(0, 0.08, dim)
    x = np.clip(base + noise, 0, 1)
    copula2_samples.append(x)

eval_samples(copula2_samples, "Copula-like (with good elite)")

# Compare with top-k
topk2_samples = []
for _ in range(n_test):
    base = top_k_pts2[rng.integers(len(top_k_pts2))]
    noise = rng.normal(0, 0.15, dim)
    x = np.clip(base + noise, 0, 1)
    topk2_samples.append(x)

eval_samples(topk2_samples, "Top-k (with good elite)")

print("\n" + "="*80)
print("CONCLUSIONE")
print("="*80)
print("""
La differenza chiave tra Copula e LGS:
- Copula usa noise più stretto (0.08 vs 0.15) attorno agli elite
- Questo è meglio quando gli elite sono già vicini all'ottimo
- Nelle prime fasi, entrambi sono simili (esplorazione)
- Nelle fasi finali, Copula "sfrutta" meglio concentrando i sample

Il 10% copula samples in più NON è il vantaggio principale.
Il vantaggio è che la distribuzione copula è più CONCENTRATA.
""")
