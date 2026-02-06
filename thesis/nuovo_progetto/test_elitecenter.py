#!/usr/bin/env python3
"""
TEST: ALBA con Elite-Center Gradient vs Original su YAHPO
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
print("TEST: ALBA Original vs Elite-Center Gradient")
print("="*80)

# Setup YAHPO
local_config.init_config()
local_config.set_data_path("/mnt/workspace/data")

bench = BenchmarkSet("iaml_ranger")
instances = list(bench.instances)[:3]

# Build param space
bench.set_instance(instances[0])
cs = bench.get_opt_space()

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

def query_yahpo(instance, config):
    bench.set_instance(instance)
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
        y = res[0].get('mmce', 1.0)  # Misclassification error to minimize
        if np.isnan(y):
            y = 1.0
        return y
    except Exception as e:
        return 1.0

from alba_framework.optimizer import ALBA as ALBA_Original
from alba_framework_elitecenter.optimizer import ALBA as ALBA_EliteCenter

n_iter = 50
n_seeds = 5

results = []

for instance in instances:
    print(f"\n{'='*60}")
    print(f"Instance: {instance}")
    print("="*60)
    
    for seed in range(n_seeds):
        # Original
        opt_orig = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)
        best_orig = float('inf')
        
        for i in range(n_iter):
            config = opt_orig.ask()
            y = query_yahpo(instance, config)
            opt_orig.tell(config, y)
            if y < best_orig:
                best_orig = y
        
        # Elite-Center
        opt_elite = ALBA_EliteCenter(param_space=param_space, seed=seed, maximize=False, total_budget=n_iter)
        best_elite = float('inf')
        
        for i in range(n_iter):
            config = opt_elite.ask()
            y = query_yahpo(instance, config)
            opt_elite.tell(config, y)
            if y < best_elite:
                best_elite = y
        
        if best_elite < best_orig - 0.001:
            winner = "Elite ✅"
        elif best_orig < best_elite - 0.001:
            winner = "Orig ❌"
        else:
            winner = "Tie"
        
        print(f"  Seed {seed}: Orig={best_orig:.4f}, Elite={best_elite:.4f} -> {winner}")
        results.append({
            'instance': instance,
            'seed': seed,
            'orig': best_orig,
            'elite': best_elite,
            'winner': 'elite' if best_elite < best_orig - 0.001 else ('orig' if best_orig < best_elite - 0.001 else 'tie')
        })

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

elite_wins = sum(1 for r in results if r['winner'] == 'elite')
orig_wins = sum(1 for r in results if r['winner'] == 'orig')
ties = sum(1 for r in results if r['winner'] == 'tie')

orig_mean = np.mean([r['orig'] for r in results])
elite_mean = np.mean([r['elite'] for r in results])

print(f"\n  Original mean: {orig_mean:.4f}")
print(f"  Elite-Center mean: {elite_mean:.4f}")
print(f"\n  Elite wins: {elite_wins}")
print(f"  Original wins: {orig_wins}")
print(f"  Ties: {ties}")

if elite_mean < orig_mean:
    print(f"\n  ✅ Elite-Center is {(orig_mean - elite_mean)/orig_mean*100:.2f}% better!")
else:
    print(f"\n  ❌ Original is {(elite_mean - orig_mean)/elite_mean*100:.2f}% better")
