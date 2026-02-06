#!/usr/bin/env python3
"""
BENCHMARK: ALBA vs ALBA-Copula su ParamNet (RF surrogates)
Test definitivo
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/HPOBench')

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

from alba_framework.optimizer import ALBA as ALBA_Original
from alba_framework_copula.optimizer import ALBA as ALBA_Copula
from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
)

import json
from datetime import datetime

print("="*80)
print("BENCHMARK: ALBA vs ALBA-Copula su ParamNet")
print("="*80)

# Config - map task names to benchmark classes
BENCHMARK_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
}
TASKS = ["adult", "higgs", "letter"]
SEEDS = range(3)  # 3 seeds
N_ITER = 50
STEP = 50  # Training step for ParamNet

results = []

for task in TASKS:
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print("="*60)
    
    # Load benchmark
    BenchClass = BENCHMARK_MAP[task]
    bench = BenchClass(rng=0)
    config_space = bench.get_configuration_space(seed=0)
    
    # Build param space
    param_space = {}
    for hp in config_space.get_hyperparameters():
        name = hp.name
        if hasattr(hp, 'choices'):
            param_space[name] = list(hp.choices)
        elif hasattr(hp, 'lower'):
            if hp.log:
                param_space[name] = (float(hp.lower), float(hp.upper), 'log')
            else:
                param_space[name] = (float(hp.lower), float(hp.upper))
    
    print(f"Param space: {len(param_space)} dims")
    
    for seed in SEEDS:
        # Run Original
        orig = ALBA_Original(param_space=param_space, seed=seed, maximize=False, total_budget=N_ITER)
        orig_best = float('inf')
        
        for i in range(N_ITER):
            config = orig.ask()
            
            # Convert to HPOBench format
            hpobench_config = {}
            for k, v in config.items():
                if isinstance(param_space[k], list):
                    hpobench_config[k] = str(v)
                else:
                    hpobench_config[k] = float(v)
            
            try:
                res = bench.objective_function(configuration=hpobench_config, fidelity={'step': STEP})
                y = res['function_value']
            except Exception as e:
                y = 1.0  # worst case
            
            orig.tell(config, y)
            if y < orig_best:
                orig_best = y
        
        # Run Copula
        copula = ALBA_Copula(param_space=param_space, seed=seed, maximize=False, total_budget=N_ITER)
        cop_best = float('inf')
        
        for i in range(N_ITER):
            config = copula.ask()
            
            hpobench_config = {}
            for k, v in config.items():
                if isinstance(param_space[k], list):
                    hpobench_config[k] = str(v)
                else:
                    hpobench_config[k] = float(v)
            
            try:
                res = bench.objective_function(configuration=hpobench_config, fidelity={'step': STEP})
                y = res['function_value']
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
        
        print(f"  Seed {seed}: Orig={orig_best:.4f}, Copula={cop_best:.4f} -> {winner}")
        
        results.append({
            'task': task,
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

if cop_wins > orig_wins:
    print(f"\n  ✅ COPULA WINS OVERALL!")
elif orig_wins > cop_wins:
    print(f"\n  ❌ Original wins overall")
else:
    print(f"\n  ⚖️ TIE overall")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'/mnt/workspace/thesis/nuovo_progetto/paramnet_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to paramnet_results_{timestamp}.json")
