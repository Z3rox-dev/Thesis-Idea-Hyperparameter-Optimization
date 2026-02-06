#!/usr/bin/env python3
"""
Debug convergenza su funzione sintetica "a scalini" (simula RF surrogate)
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

from alba_framework_potential import ALBA as ALBA_LGS
from alba_framework_copula.optimizer import ALBA as ALBACopula

# =============================================================================
# Funzione "a scalini" che simula il comportamento di un RF surrogate
# =============================================================================

def staircase_function(x, n_steps=10, noise=0.0):
    """
    Funzione a scalini in ogni dimensione.
    Simula il comportamento piecewise-constant di un Random Forest.
    """
    dim = len(x)
    # Quantizza ogni dimensione
    x_quantized = np.floor(x * n_steps) / n_steps
    
    # Combinazione non-lineare
    score = 0.0
    for i in range(dim):
        # Ogni dimensione ha un contributo diverso
        score += (x_quantized[i] - 0.5)**2 * (1 + 0.5 * (i % 2))
    
    # Aggiungi discontinuità nette
    if np.sum(x_quantized) > dim * 0.6:
        score += 0.5  # Penalità per valori alti
    
    return score + np.random.normal(0, noise)

# Versione vettoriale per visualizzazione
def staircase_vector(X, n_steps=10):
    return np.array([staircase_function(x, n_steps) for x in X])

# Test: visualizza la superficie
print("="*70)
print("Test funzione a scalini")
print("="*70)

dim = 5
bounds = np.array([[0, 1]] * dim)

# Sample some points
test_x = np.random.rand(20, dim)
test_y = staircase_vector(test_x)
print(f"Score range: [{test_y.min():.4f}, {test_y.max():.4f}]")
print(f"Optimum teorico: ~0.0 a x=[0.5, 0.5, ...]")
opt_test = np.array([0.5] * dim)
print(f"Score at [0.5, ...]: {staircase_function(opt_test):.4f}")

# =============================================================================
# Confronto LGS vs Copula
# =============================================================================

def run_optimizer(OptimizerClass, name, seed=42, budget=80):
    np.random.seed(seed)
    
    opt = OptimizerClass(
        bounds=bounds.tolist(),
        seed=seed,
        total_budget=budget,
        maximize=False,  # Minimize staircase function
    )
    
    history = []
    for i in range(budget):
        x = opt.ask()
        y = staircase_function(x, n_steps=10)
        opt.tell(x, y)
        history.append(opt.best_y)
    
    return history

print("\n" + "="*70)
print("Running comparison...")
print("="*70)

seeds = [0, 1, 2]
budget = 80

all_lgs = []
all_cop = []

for seed in seeds:
    print(f"\nSeed {seed}:")
    h_lgs = run_optimizer(ALBA_LGS, "LGS", seed, budget)
    h_cop = run_optimizer(ALBACopula, "Copula", seed, budget)
    
    all_lgs.append(h_lgs)
    all_cop.append(h_cop)
    
    print(f"  LGS final:    {h_lgs[-1]:.4f}")
    print(f"  Copula final: {h_cop[-1]:.4f}")
    print(f"  Winner: {'LGS' if h_lgs[-1] < h_cop[-1] else 'Copula'}")

# Average convergence
print("\n" + "="*70)
print("Average convergence over 3 seeds")
print("="*70)

avg_lgs = np.mean(all_lgs, axis=0)
avg_cop = np.mean(all_cop, axis=0)

print(f"\n{'Iter':<8} {'LGS':<12} {'Copula':<12} {'Diff':<12}")
for i in [0, 10, 20, 40, 60, 79]:
    diff = avg_lgs[i] - avg_cop[i]
    better = "LGS +" if diff < 0 else "Cop +"
    print(f"{i:<8} {avg_lgs[i]:<12.4f} {avg_cop[i]:<12.4f} {diff:+.4f} ({better})")

print(f"\nFinal averages:")
print(f"  LGS:    {avg_lgs[-1]:.4f}")
print(f"  Copula: {avg_cop[-1]:.4f}")

# Chi vince?
lgs_wins = sum(1 for l, c in zip([h[-1] for h in all_lgs], [h[-1] for h in all_cop]) if l < c)
cop_wins = 3 - lgs_wins
print(f"\nWins: LGS={lgs_wins}, Copula={cop_wins}")
