#!/usr/bin/env python3
"""
TEST: Diverse strategie di sampling su YAHPO RF surrogates

Confrontiamo:
1. LGS originale (25% topk, 15% grad, 15% center, 45% uniform)
2. No gradient (40% topk, 0% grad, 15% center, 45% uniform)  
3. More topk (50% topk, 0% grad, 10% center, 40% uniform)
4. Elite center gradient (usando elite-center invece di LGS grad)
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

from yahpo_gym import BenchmarkSet, local_config

print("="*80)
print("TEST: Sampling Strategies su YAHPO RF")
print("="*80)

# Setup YAHPO
local_config.init_config()
local_config.set_data_path("/mnt/workspace/data")

bench = BenchmarkSet("iaml_ranger")
instance = list(bench.instances)[0]
bench.set_instance(instance)
cs = bench.get_opt_space()

print(f"Instance: {instance}")

# Build param space
param_space = {}
int_params = set()
for hp in cs.get_hyperparameters():
    name = hp.name
    if name == 'task_id':
        continue
    if hasattr(hp, 'choices'):
        param_space[name] = list(hp.choices)
    elif hasattr(hp, 'lower'):
        if 'Integer' in hp.__class__.__name__:
            int_params.add(name)
        if hasattr(hp, 'log') and hp.log:
            param_space[name] = (float(hp.lower), float(hp.upper), 'log')
        else:
            param_space[name] = (float(hp.lower), float(hp.upper))

print(f"Params: {len(param_space)} dims")
print(f"Int params: {int_params}")

def query_yahpo(config):
    cs_config = {'task_id': str(instance)}
    for k, v in config.items():
        if isinstance(param_space[k], list):
            cs_config[k] = v
        elif k in int_params:
            cs_config[k] = int(round(v))
        else:
            cs_config[k] = float(v)
    
    try:
        res = bench.objective_function(cs_config)
        y = 1.0 - res[0].get('mmce', 1.0)  # Use mmce (misclassification), minimize
        if np.isnan(y):
            y = 0.0
        return 1.0 - y  # Convert back to minimization target
    except Exception as e:
        return 1.0

# Now test different sampling strategies using ALBA
from alba_framework.optimizer import ALBA as ALBA_Original

n_iter = 50
n_seeds = 3

print(f"\nRunning {n_iter} iterations x {n_seeds} seeds...")

orig_results = []
for seed in range(n_seeds):
    opt = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)
    best = float('inf')
    
    for i in range(n_iter):
        config = opt.ask()
        y = query_yahpo(config)
        opt.tell(config, y)
        if y < best:
            best = y
    
    orig_results.append(best)
    print(f"  Seed {seed}: best={best:.4f}")

print(f"\nOriginal ALBA: mean={np.mean(orig_results):.4f}")
print(f"\nNota: per testare altre strategie, dovremmo modificare MixtureCandidateGenerator")
