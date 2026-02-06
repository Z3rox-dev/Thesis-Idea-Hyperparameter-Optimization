#!/usr/bin/env python3
"""
Benchmark: ALBA_V1 vs ALBA_V1_thompson_all su YAHPO Gym
"""

import sys
import warnings
import numpy as np
import time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')

from ALBA_V1 import ALBA as ALBA_V1
from ALBA_V1_thompson_all import ALBA as ALBA_THOMPSON_ALL

from yahpo_gym import BenchmarkSet

scenarios = ['rbv2_xgboost', 'rbv2_ranger', 'rbv2_svm', 'iaml_xgboost']

BUDGET = 200
N_SEEDS = 3
N_INSTANCES = 2

print("="*70)
print("YAHPO: ALBA_V1 vs Thompson_All")
print(f"Budget: {BUDGET}, Seeds: {N_SEEDS}, Instances: {N_INSTANCES}")
print("="*70)

all_results = {'V1': 0, 'Thompson_All': 0, 'ties': 0}

for scenario_name in scenarios:
    print(f"\n>>> Scenario: {scenario_name}")
    
    bench = BenchmarkSet(scenario_name)
    cs = bench.get_opt_space()
    instances = list(bench.instances)[:N_INSTANCES]
    
    obj_name = bench.config.y_names[0]
    maximize = 'acc' in obj_name.lower() or 'auc' in obj_name.lower()
    
    hps = list(cs.values())
    dim = len(hps)
    
    # Detect categoricals for V1
    categorical_dims = []
    for i, hp in enumerate(hps):
        if hasattr(hp, 'choices'):
            categorical_dims.append((i, len(hp.choices)))
    
    n_cat = len(categorical_dims)
    n_cont = dim - n_cat
    print(f"  Dim: {dim} ({n_cont} cont + {n_cat} cat), Obj: {obj_name} ({'max' if maximize else 'min'})")
    
    def config_from_x(x, instance_id):
        config = {'trainsize': 1.0, 'repl': 1}
        if hasattr(bench.config, 'instance_names'):
            inst_name = bench.config.instance_names
            if inst_name:
                config[inst_name] = instance_id
        
        for i, hp in enumerate(hps):
            name = hp.name
            if name in config:
                continue
            
            # Handle conditionals for SVM
            if scenario_name == 'rbv2_svm':
                if name == 'degree':
                    kernel_idx = next((j for j, h in enumerate(hps) if h.name == 'kernel'), None)
                    if kernel_idx is not None:
                        kernel_hp = hps[kernel_idx]
                        kernel_choices = list(kernel_hp.choices)
                        k_idx = int(np.round(x[kernel_idx] * (len(kernel_choices) - 1)))
                        k_idx = max(0, min(len(kernel_choices) - 1, k_idx))
                        kernel_val = kernel_choices[k_idx]
                        if kernel_val != 'polynomial':
                            continue
                elif name == 'gamma':
                    kernel_idx = next((j for j, h in enumerate(hps) if h.name == 'kernel'), None)
                    if kernel_idx is not None:
                        kernel_hp = hps[kernel_idx]
                        kernel_choices = list(kernel_hp.choices)
                        k_idx = int(np.round(x[kernel_idx] * (len(kernel_choices) - 1)))
                        k_idx = max(0, min(len(kernel_choices) - 1, k_idx))
                        kernel_val = kernel_choices[k_idx]
                        if kernel_val != 'radial':
                            continue
                
            if hasattr(hp, 'choices'):
                choices = list(hp.choices)
                idx = int(np.round(x[i] * (len(choices) - 1)))
                idx = max(0, min(len(choices) - 1, idx))
                config[name] = choices[idx]
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                lo, hi = hp.lower, hp.upper
                if getattr(hp, 'log', False):
                    val = np.exp(np.log(lo) + x[i] * (np.log(hi) - np.log(lo)))
                else:
                    val = lo + x[i] * (hi - lo)
                
                if 'int' in str(type(hp)).lower():
                    val = int(np.round(val))
                config[name] = val
        
        return config
    
    def run_alba(alba_cls, instance_id, budget, seed, cat_dims=None):
        opt = alba_cls(
            bounds=[(0.0, 1.0)] * dim,
            maximize=maximize,
            seed=seed,
            total_budget=budget,
            categorical_dims=cat_dims,
        )
        
        best = -np.inf if maximize else np.inf
        
        for _ in range(budget):
            x = opt.ask()
            config = config_from_x(x, instance_id)
            
            try:
                result = bench.objective_function(configuration=config, seed=seed)
                score = result[0][obj_name]
            except Exception as e:
                score = 0.0 if maximize else 1.0
            
            opt.tell(x, score)
            
            if maximize:
                best = max(best, score)
            else:
                best = min(best, score)
        
        return best
    
    scenario_v1 = 0
    scenario_ta = 0
    scenario_ties = 0
    
    for inst in instances:
        for seed in range(N_SEEDS):
            t0 = time.time()
            score_v1 = run_alba(ALBA_V1, inst, BUDGET, seed, categorical_dims)
            t1 = time.time()
            score_ta = run_alba(ALBA_THOMPSON_ALL, inst, BUDGET, seed, None)
            t2 = time.time()
            
            if maximize:
                if score_v1 > score_ta + 1e-6:
                    all_results['V1'] += 1
                    scenario_v1 += 1
                    mark = "V1 ✓"
                elif score_ta > score_v1 + 1e-6:
                    all_results['Thompson_All'] += 1
                    scenario_ta += 1
                    mark = "TA ✓"
                else:
                    all_results['ties'] += 1
                    scenario_ties += 1
                    mark = "="
            else:
                if score_v1 < score_ta - 1e-6:
                    all_results['V1'] += 1
                    scenario_v1 += 1
                    mark = "V1 ✓"
                elif score_ta < score_v1 - 1e-6:
                    all_results['Thompson_All'] += 1
                    scenario_ta += 1
                    mark = "TA ✓"
                else:
                    all_results['ties'] += 1
                    scenario_ties += 1
                    mark = "="
            
            print(f"  {inst}/s{seed}: V1={score_v1:.4f} ({t1-t0:.1f}s) | TA={score_ta:.4f} ({t2-t1:.1f}s) | {mark}")
    
    print(f"  Scenario total: V1 {scenario_v1} - {scenario_ta} TA (ties: {scenario_ties})")

print("\n" + "="*70)
print("RISULTATO FINALE YAHPO")
print("="*70)
print(f"V1: {all_results['V1']} | Thompson_All: {all_results['Thompson_All']} | Ties: {all_results['ties']}")

if all_results['V1'] > all_results['Thompson_All']:
    print("\n🏆 VINCITORE: ALBA_V1 (Thompson cat + LGS cont)")
elif all_results['Thompson_All'] > all_results['V1']:
    print("\n🏆 VINCITORE: Thompson_All")
else:
    print("\n🤝 PAREGGIO")
