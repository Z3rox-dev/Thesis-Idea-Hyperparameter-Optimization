#!/usr/bin/env python3
"""
BENCHMARK: ALBA vs ALBA-Copula su YAHPO (RF surrogates)
Test definitivo
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

from alba_framework.optimizer import ALBA as ALBA_Original
from alba_framework_copula.optimizer import ALBA as ALBA_Copula

# YAHPO
from yahpo_gym import BenchmarkSet, local_config

import json
from datetime import datetime

print("="*80)
print("BENCHMARK: ALBA vs ALBA-Copula su YAHPO (tabular)")
print("="*80)

# Config
local_config.init_config()
local_config.set_data_path("/mnt/workspace/data")

SCENARIOS = ["rbv2_xgboost", "iaml_xgboost"]
SEEDS = range(3)
N_ITER = 50

results = []

for scenario_name in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print("="*60)
    
    bench = BenchmarkSet(scenario_name)
    instances = list(bench.instances)[:2]  # First 2 instances
    
    for instance in instances:
        bench.set_instance(instance)
        cs = bench.get_opt_space()
        target = bench.config.y_names[0]
        
        # Build param space (exclude constant task_id)
        param_space = {}
        int_params = set()  # Track integer params
        for hp in cs.get_hyperparameters():
            name = hp.name
            if name == 'task_id':  # Skip constant
                continue
            if hasattr(hp, 'choices'):
                param_space[name] = list(hp.choices)
            elif hasattr(hp, 'lower'):
                # Check if integer type
                if 'Integer' in hp.__class__.__name__:
                    int_params.add(name)
                if hasattr(hp, 'log') and hp.log:
                    param_space[name] = (float(hp.lower), float(hp.upper), 'log')
                else:
                    param_space[name] = (float(hp.lower), float(hp.upper))
        
        print(f"\n  Instance: {instance} ({len(param_space)} dims)")
        
        for seed in SEEDS:
            # Run Original
            orig = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=N_ITER)
            orig_best = float('inf')
            
            for i in range(N_ITER):
                config = orig.ask()
                
                # Convert to ConfigSpace format - add required task_id
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
                    # Use 1-acc for minimization (acc is higher=better)
                    y = 1.0 - res[0].get('acc', 0.0)
                    if np.isnan(y):
                        y = 1.0
                    if i == 0 and seed == 0:  # Debug first query
                        print(f"    DEBUG: y={y:.4f}, config keys={list(cs_config.keys())[:3]}...")
                except Exception as e:
                    if i == 0 and seed == 0:
                        print(f"    DEBUG ERROR: {e}")
                    y = 1.0
                
                orig.tell(config, y)
                if y < orig_best:
                    orig_best = y
            
            # Run Copula
            copula = ALBA_Copula(param_space=param_space, seed=seed, maximize=False, total_budget=N_ITER)
            cop_best = float('inf')
            
            for i in range(N_ITER):
                config = copula.ask()
                
                # Convert to ConfigSpace format - add required task_id
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
                    y = 1.0 - res[0].get('acc', 0.0)
                    if np.isnan(y):
                        y = 1.0
                except Exception as e:
                    y = 1.0
                
                copula.tell(config, y)
                if y < cop_best:
                    cop_best = y
            
            if cop_best < orig_best - 0.001:
                winner = "Copula"
            elif orig_best < cop_best - 0.001:
                winner = "Original"
            else:
                winner = "Tie"
            
            print(f"    Seed {seed}: Orig={orig_best:.4f}, Copula={cop_best:.4f} -> {winner}")
            
            results.append({
                'scenario': scenario_name,
                'instance': str(instance),
                'seed': seed,
                'orig_best': orig_best,
                'cop_best': cop_best,
                'winner': winner
            })

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

cop_wins = sum(1 for r in results if r['winner'] == 'Copula')
orig_wins = sum(1 for r in results if r['winner'] == 'Original')
ties = sum(1 for r in results if r['winner'] == 'Tie')

print(f"\n  Copula wins: {cop_wins}")
print(f"  Original wins: {orig_wins}")
print(f"  Ties: {ties}")
print(f"  Win rate: Copula {100*cop_wins/(cop_wins+orig_wins+ties):.1f}%")

if cop_wins > orig_wins:
    print(f"\n  ✅ COPULA WINS OVERALL!")
elif orig_wins > cop_wins:
    print(f"\n  ❌ Original wins overall")
else:
    print(f"\n  ⚖️ TIE overall")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'/mnt/workspace/thesis/nuovo_progetto/yahpo_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to yahpo_results_{timestamp}.json")
