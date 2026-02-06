#!/usr/bin/env python3
"""
Debug dettagliato: Cosa succede iterazione per iterazione?
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

# HPOBench ParamNet
sys.path.insert(0, '/mnt/workspace/HPOBench')
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnTimeBenchmark

from alba_framework_potential import ALBA as ALBA_LGS
from alba_framework_copula.optimizer import ALBA as ALBACopula

# Create benchmark
bench = ParamNetAdultOnTimeBenchmark()

# Get param space
api = bench.get_configuration_space(seed=42)
param_names = list(api.keys())
print(f"Parameters: {param_names}")

# Build proper param_space
param_space = {}
bounds = []
for p in api.values():
    name = p.name
    if hasattr(p, 'choices'):
        param_space[name] = ('cat', list(p.choices))
        bounds.append((0, len(p.choices) - 1))
    else:
        param_space[name] = ('cont', (p.lower, p.upper))
        bounds.append((p.lower, p.upper))

bounds = np.array(bounds, dtype=float)
print(f"Bounds: {bounds}")
print(f"Param space: {param_space}")

def objective(x, fidelity=None):
    config = {}
    for i, p in enumerate(api.values()):
        name = p.name
        if hasattr(p, 'choices'):
            idx = int(round(x[i]))
            idx = max(0, min(idx, len(p.choices) - 1))
            config[name] = p.choices[idx]
        else:
            config[name] = float(x[i])
    
    result = bench.objective_function(config, fidelity={"step": 50})
    return result['function_value']

# Run both optimizers with logging
seed = 0
budget = 50

print("\n" + "="*70)
print(f"ALBA-LGS (seed={seed}, budget={budget})")
print("="*70)

np.random.seed(seed)
opt_lgs = ALBA_LGS(
    objective,
    bounds,
    param_space=param_space,
    n_init=10,
    random_state=seed,
    verbose=True,
)
history_lgs = []
for i in range(budget):
    opt_lgs.step()
    best = min(r[1] for r in opt_lgs.results)
    history_lgs.append(best)
    if i % 10 == 0 or i == budget - 1:
        print(f"  Iter {i}: best={best:.4f}")

print(f"\nFinal best LGS: {min(history_lgs):.4f}")

print("\n" + "="*70)
print(f"ALBA-Copula (seed={seed}, budget={budget})")
print("="*70)

np.random.seed(seed)
opt_cop = ALBACopula(
    objective,
    bounds,
    param_space=param_space,
    n_init=10,
    random_state=seed,
    verbose=True,
)
history_cop = []
for i in range(budget):
    opt_cop.step()
    best = min(r[1] for r in opt_cop.results)
    history_cop.append(best)
    if i % 10 == 0 or i == budget - 1:
        print(f"  Iter {i}: best={best:.4f}")

print(f"\nFinal best Copula: {min(history_cop):.4f}")

# Compare convergence
print("\n" + "="*70)
print("Convergence comparison")
print("="*70)
print(f"{'Iter':<6} {'LGS':<12} {'Copula':<12} {'Better':<10}")
for i in [0, 5, 10, 20, 30, 40, 49]:
    lgs_val = history_lgs[i] if i < len(history_lgs) else history_lgs[-1]
    cop_val = history_cop[i] if i < len(history_cop) else history_cop[-1]
    better = "LGS" if lgs_val < cop_val else "Copula" if cop_val < lgs_val else "Tie"
    print(f"{i:<6} {lgs_val:<12.4f} {cop_val:<12.4f} {better:<10}")
