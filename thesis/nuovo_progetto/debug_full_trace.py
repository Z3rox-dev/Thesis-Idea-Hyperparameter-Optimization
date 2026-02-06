#!/usr/bin/env python3
"""
DEBUG: Trace completo ALBA vs ALBA-Copula
Step-by-step cosa succede ad ogni iterazione
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

# Import optimizers
from alba_framework.optimizer import ALBA as ALBA_Original
from alba_framework_copula.optimizer import ALBA as ALBA_Copula

print("="*80)
print("TRACE COMPLETO: ALBA vs ALBA-Copula step-by-step")
print("="*80)

# Staircase function
def staircase(x, n_steps=10):
    """Staircase function - mimics RF surrogates"""
    x = np.array([x[k] for k in sorted(x.keys())])
    x_q = np.floor(x * n_steps) / n_steps
    return float(np.sum((x_q - 0.5)**2))

# Sphere function
def sphere(x):
    x = np.array([x[k] for k in sorted(x.keys())])
    return float(np.sum((x - 0.5)**2))

# Configurazione
dim = 5
n_iter = 50
seed = 42

# Param space
param_space = {f"x{i}": (0.0, 1.0) for i in range(dim)}

print(f"\nParam space: {dim}D continuous")
print(f"Iterations: {n_iter}")

# Run on staircase
print(f"\n{'='*80}")
print("TEST 1: STAIRCASE (RF-like)")
print("="*80)

orig = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)
copula = ALBA_Copula(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)

orig_best = float('inf')
cop_best = float('inf')

for i in range(n_iter):
    # Original
    x_orig = orig.ask()
    y_orig = staircase(x_orig)
    orig.tell(x_orig, y_orig)
    if y_orig < orig_best:
        orig_best = y_orig
    
    # Copula
    x_cop = copula.ask()
    y_cop = staircase(x_cop)
    copula.tell(x_cop, y_cop)
    if y_cop < cop_best:
        cop_best = y_cop
    
    if i % 10 == 9:
        print(f"  Iter {i+1}: Orig={orig_best:.4f}, Copula={cop_best:.4f}")

print(f"\n  Final: Orig={orig_best:.4f}, Copula={cop_best:.4f}")
if cop_best < orig_best:
    print(f"  ✅ COPULA WINS on staircase!")
elif cop_best > orig_best:
    print(f"  ❌ Original wins on staircase")
else:
    print(f"  ⚖️ TIE on staircase")

# Run on sphere
print(f"\n{'='*80}")
print("TEST 2: SPHERE (smooth)")
print("="*80)

orig = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)
copula = ALBA_Copula(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)

orig_best = float('inf')
cop_best = float('inf')

for i in range(n_iter):
    x_orig = orig.ask()
    y_orig = sphere(x_orig)
    orig.tell(x_orig, y_orig)
    if y_orig < orig_best:
        orig_best = y_orig
    
    x_cop = copula.ask()
    y_cop = sphere(x_cop)
    copula.tell(x_cop, y_cop)
    if y_cop < cop_best:
        cop_best = y_cop
    
    if i % 10 == 9:
        print(f"  Iter {i+1}: Orig={orig_best:.4f}, Copula={cop_best:.4f}")

print(f"\n  Final: Orig={orig_best:.4f}, Copula={cop_best:.4f}")
if cop_best < orig_best:
    print(f"  ✅ COPULA WINS on sphere!")
elif cop_best > orig_best:
    print(f"  ❌ Original wins on sphere")
else:
    print(f"  ⚖️ TIE on sphere")

# Multiple seeds
print(f"\n{'='*80}")
print("MULTI-SEED TEST (5 seeds)")
print("="*80)

for func_name, func in [("Staircase", staircase), ("Sphere", sphere)]:
    orig_wins = 0
    cop_wins = 0
    ties = 0
    
    for s in range(5):
        orig = ALBA_Original(param_space=param_space, seed=s, maximize=False, total_budget=n_iter)
        copula = ALBA_Copula(param_space=param_space, seed=s, maximize=False, total_budget=n_iter)
        
        orig_best = float('inf')
        cop_best = float('inf')
        
        for i in range(n_iter):
            x_orig = orig.ask()
            y_orig = func(x_orig)
            orig.tell(x_orig, y_orig)
            if y_orig < orig_best:
                orig_best = y_orig
            
            x_cop = copula.ask()
            y_cop = func(x_cop)
            copula.tell(x_cop, y_cop)
            if y_cop < cop_best:
                cop_best = y_cop
        
        if cop_best < orig_best - 0.001:
            cop_wins += 1
            winner = "C✅"
        elif orig_best < cop_best - 0.001:
            orig_wins += 1
            winner = "O❌"
        else:
            ties += 1
            winner = "=⚖️"
        
        print(f"  {func_name} seed {s}: Orig={orig_best:.4f}, Copula={cop_best:.4f} {winner}")
    
    print(f"  {func_name} TOTAL: Copula {cop_wins} - {orig_wins} Orig ({ties} ties)")
    print()
