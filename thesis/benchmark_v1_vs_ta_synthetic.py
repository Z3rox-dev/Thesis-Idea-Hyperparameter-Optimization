#!/usr/bin/env python3
"""
Benchmark sintetico per confrontare V1 vs Thompson_All
su funzioni con molti continui (dove V1 dovrebbe eccellere)
"""

import sys
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')

from ALBA_V1 import ALBA as ALBA_V1
from ALBA_V1_thompson_all import ALBA as ALBA_THOMPSON_ALL


def rosenbrock(x):
    """Rosenbrock function - smooth continuo, ottimo a (1,1,...,1)"""
    total = 0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total


def rastrigin(x):
    """Rastrigin function - multimodale"""
    A = 10
    n = len(x)
    # Scale x from [0,1] to [-5.12, 5.12]
    x_scaled = (np.array(x) - 0.5) * 10.24
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x_scaled)


def mixed_categorical_continuous(x, optimal_cats=[1, 0, 2]):
    """
    Funzione mista: 3 categorici + N continui
    x[0:3] categorici (codificati come continui 0-1)
    x[3:] continui (sfera)
    """
    # Decode categorici
    cat_bonus = 0
    for i, opt in enumerate(optimal_cats):
        cat_val = int(round(x[i] * 2))  # 3 scelte: 0, 1, 2
        if cat_val == opt:
            cat_bonus -= 0.5  # bonus
        else:
            cat_bonus += 0.2 * abs(cat_val - opt)
    
    # Continui: sfera con ottimo a 0.5
    cont_part = sum((xi - 0.5)**2 for xi in x[3:])
    
    return cat_bonus + cont_part + np.random.normal(0, 0.01)


print("="*70)
print("Benchmark SINTETICO: V1 vs Thompson_All")
print("="*70)

functions = {
    'Rosenbrock_5D': (rosenbrock, 5, []),  # Solo continui
    'Rosenbrock_10D': (rosenbrock, 10, []),  # Solo continui
    'Rastrigin_5D': (rastrigin, 5, []),  # Solo continui
    'Mixed_3cat_5cont': (mixed_categorical_continuous, 8, [(0,3), (1,3), (2,3)]),  # Misto
    'Mixed_3cat_10cont': (
        lambda x: mixed_categorical_continuous(x, [1, 0, 2]), 
        13, 
        [(0,3), (1,3), (2,3)]
    ),
}

BUDGET = 200
N_SEEDS = 10

all_v1_wins = 0
all_ta_wins = 0

for func_name, (func, dim, cat_dims) in functions.items():
    print(f"\n>>> {func_name} (dim={dim}, cat={len(cat_dims)})")
    
    results_v1 = []
    results_ta = []
    
    for seed in range(N_SEEDS):
        # V1
        opt_v1 = ALBA_V1(
            bounds=[(0.0, 1.0)] * dim,
            maximize=False,
            seed=seed,
            total_budget=BUDGET,
            categorical_dims=cat_dims if cat_dims else None,
        )
        best_v1 = float('inf')
        for _ in range(BUDGET):
            x = opt_v1.ask()
            y = func(x)
            opt_v1.tell(x, y)
            best_v1 = min(best_v1, y)
        results_v1.append(best_v1)
        
        # Thompson All
        opt_ta = ALBA_THOMPSON_ALL(
            bounds=[(0.0, 1.0)] * dim,
            maximize=False,
            seed=seed,
            total_budget=BUDGET,
        )
        best_ta = float('inf')
        for _ in range(BUDGET):
            x = opt_ta.ask()
            y = func(x)
            opt_ta.tell(x, y)
            best_ta = min(best_ta, y)
        results_ta.append(best_ta)
    
    mean_v1 = np.mean(results_v1)
    mean_ta = np.mean(results_ta)
    std_v1 = np.std(results_v1)
    std_ta = np.std(results_ta)
    
    wins_v1 = sum(1 for a, b in zip(results_v1, results_ta) if a < b - 1e-8)
    wins_ta = sum(1 for a, b in zip(results_v1, results_ta) if b < a - 1e-8)
    
    all_v1_wins += wins_v1
    all_ta_wins += wins_ta
    
    winner = "V1" if wins_v1 > wins_ta else ("TA" if wins_ta > wins_v1 else "TIE")
    print(f"  V1: {mean_v1:.4f} ± {std_v1:.4f}")
    print(f"  TA: {mean_ta:.4f} ± {std_ta:.4f}")
    print(f"  Head-to-head: V1 {wins_v1} - {wins_ta} TA → {winner}")

print("\n" + "="*70)
print("TOTALE SINTETICI")
print("="*70)
print(f"V1: {all_v1_wins} | Thompson_All: {all_ta_wins}")

if all_v1_wins > all_ta_wins:
    print("\n🏆 VINCITORE su sintetici: V1")
elif all_ta_wins > all_v1_wins:
    print("\n🏆 VINCITORE su sintetici: Thompson_All")
else:
    print("\n🤝 PAREGGIO")
