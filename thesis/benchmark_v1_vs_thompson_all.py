#!/usr/bin/env python3
"""
Benchmark: ALBA_V1 (Thompson cat + LGS cont) vs ALBA_V1_thompson_all (Thompson tutto)
Su ParamNet e YAHPO
"""

import sys
import warnings
import numpy as np
import time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')

from ALBA_V1 import ALBA as ALBA_V1
from ALBA_V1_thompson_all import ALBA as ALBA_THOMPSON_ALL

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

#############################################
# ParamNet Benchmark
#############################################

def run_paramnet_benchmark():
    print("="*70)
    print("BENCHMARK 1: ParamNet Adult")
    print("="*70)
    
    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnTimeBenchmark
    
    benchmark = ParamNetAdultOnTimeBenchmark()
    cs = benchmark.get_configuration_space()
    hps = list(cs.values())
    dim = len(hps)
    
    print(f"Dimensioni: {dim}")
    
    # Detect categorical dims
    categorical_dims = []
    for i, hp in enumerate(hps):
        if hasattr(hp, 'choices'):
            categorical_dims.append((i, len(hp.choices)))
    print(f"Categorici: {categorical_dims}")
    
    def config_from_x(x):
        config = {}
        for i, hp in enumerate(hps):
            name = hp.name
            if hasattr(hp, 'choices'):
                choices = list(hp.choices)
                idx = int(np.round(x[i] * (len(choices) - 1)))
                idx = max(0, min(len(choices) - 1, idx))
                config[name] = choices[idx]
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                lo, hi = hp.lower, hp.upper
                if hasattr(hp, 'q') or 'int' in str(type(hp)).lower() or isinstance(lo, int):
                    val = int(np.round(lo + x[i] * (hi - lo)))
                    val = max(lo, min(hi, val))
                elif getattr(hp, 'log', False):
                    val = np.exp(np.log(lo) + x[i] * (np.log(hi) - np.log(lo)))
                else:
                    val = lo + x[i] * (hi - lo)
                config[name] = val
        return config
    
    def run_alba(alba_cls, budget, seed, cat_dims=None):
        opt = alba_cls(
            bounds=[(0.0, 1.0)] * dim,
            maximize=False,
            seed=seed,
            total_budget=budget,
            categorical_dims=cat_dims,
        )
        best = float('inf')
        for _ in range(budget):
            x = opt.ask()
            config = config_from_x(x)
            result = benchmark.objective_function(configuration=config)
            loss = result['function_value']
            opt.tell(x, loss)
            best = min(best, loss)
        return best
    
    BUDGET = 300
    SEEDS = [42, 43, 44, 45, 46]
    
    results = {'V1': [], 'Thompson_All': []}
    
    print(f"\nBudget: {BUDGET}, Seeds: {SEEDS}")
    print("-"*50)
    
    for seed in SEEDS:
        t0 = time.time()
        loss_v1 = run_alba(ALBA_V1, BUDGET, seed, categorical_dims)
        t1 = time.time()
        loss_ta = run_alba(ALBA_THOMPSON_ALL, BUDGET, seed, None)  # thompson_all non usa cat_dims
        t2 = time.time()
        
        results['V1'].append(loss_v1)
        results['Thompson_All'].append(loss_ta)
        
        winner = "V1" if loss_v1 < loss_ta else ("Thompson_All" if loss_ta < loss_v1 else "TIE")
        print(f"Seed {seed}: V1={loss_v1:.5f} ({t1-t0:.1f}s) | Thompson_All={loss_ta:.5f} ({t2-t1:.1f}s) | {winner}")
    
    print("-"*50)
    print(f"Mean: V1={np.mean(results['V1']):.5f} | Thompson_All={np.mean(results['Thompson_All']):.5f}")
    
    wins_v1 = sum(1 for a, b in zip(results['V1'], results['Thompson_All']) if a < b - 1e-8)
    wins_ta = sum(1 for a, b in zip(results['V1'], results['Thompson_All']) if b < a - 1e-8)
    print(f"HEAD-TO-HEAD: V1 {wins_v1} - {wins_ta} Thompson_All")
    
    return results


#############################################
# YAHPO Benchmark
#############################################

def run_yahpo_benchmark():
    print("\n" + "="*70)
    print("BENCHMARK 2: YAHPO Gym")
    print("="*70)
    
    from yahpo_gym import BenchmarkSet
    from yahpo_gym.configuration import list_scenarios
    
    scenarios = ['rbv2_xgboost', 'rbv2_ranger', 'rbv2_svm']
    
    all_results = {'V1': 0, 'Thompson_All': 0, 'ties': 0}
    
    for scenario_name in scenarios:
        print(f"\n>>> Scenario: {scenario_name}")
        
        bench = BenchmarkSet(scenario_name)
        cs = bench.get_opt_space()
        instances = list(bench.instances)[:2]  # Solo 2 instances per velocità
        
        obj_name = bench.config.y_names[0]
        maximize = 'acc' in obj_name.lower() or 'auc' in obj_name.lower()
        
        hps = list(cs.values())
        dim = len(hps)
        
        # Detect categoricals
        categorical_dims = []
        for i, hp in enumerate(hps):
            if hasattr(hp, 'choices'):
                categorical_dims.append((i, len(hp.choices)))
        
        print(f"  Dim: {dim}, Cat: {len(categorical_dims)}, Obj: {obj_name} ({'max' if maximize else 'min'})")
        
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
                except:
                    score = 0.0 if maximize else 1.0
                
                opt.tell(x, score)
                
                if maximize:
                    best = max(best, score)
                else:
                    best = min(best, score)
            
            return best
        
        BUDGET = 100
        SEEDS = [42, 43]
        
        for inst in instances:
            for seed in SEEDS:
                score_v1 = run_alba(ALBA_V1, inst, BUDGET, seed, categorical_dims)
                score_ta = run_alba(ALBA_THOMPSON_ALL, inst, BUDGET, seed, None)
                
                if maximize:
                    if score_v1 > score_ta + 1e-6:
                        all_results['V1'] += 1
                        mark = "V1 ✓"
                    elif score_ta > score_v1 + 1e-6:
                        all_results['Thompson_All'] += 1
                        mark = "TA ✓"
                    else:
                        all_results['ties'] += 1
                        mark = "="
                else:
                    if score_v1 < score_ta - 1e-6:
                        all_results['V1'] += 1
                        mark = "V1 ✓"
                    elif score_ta < score_v1 - 1e-6:
                        all_results['Thompson_All'] += 1
                        mark = "TA ✓"
                    else:
                        all_results['ties'] += 1
                        mark = "="
                
                print(f"  {inst}/s{seed}: V1={score_v1:.4f} | TA={score_ta:.4f} | {mark}")
    
    print("\n" + "="*70)
    print("YAHPO TOTALE:")
    print(f"  V1: {all_results['V1']} | Thompson_All: {all_results['Thompson_All']} | Ties: {all_results['ties']}")
    
    return all_results


#############################################
# Main
#############################################

if __name__ == "__main__":
    print("ALBA_V1 (Thompson cat + LGS cont) vs ALBA_V1_thompson_all (Thompson tutto)")
    print("="*70)
    
    # ParamNet
    paramnet_results = run_paramnet_benchmark()
    
    # YAHPO
    yahpo_results = run_yahpo_benchmark()
    
    print("\n" + "="*70)
    print("RIEPILOGO FINALE")
    print("="*70)
    
    v1_mean = np.mean(paramnet_results['V1'])
    ta_mean = np.mean(paramnet_results['Thompson_All'])
    print(f"\nParamNet: V1={v1_mean:.5f} vs Thompson_All={ta_mean:.5f}")
    
    v1_wins = sum(1 for a, b in zip(paramnet_results['V1'], paramnet_results['Thompson_All']) if a < b - 1e-8)
    ta_wins = sum(1 for a, b in zip(paramnet_results['V1'], paramnet_results['Thompson_All']) if b < a - 1e-8)
    print(f"  Head-to-head: V1 {v1_wins} - {ta_wins} Thompson_All")
    
    print(f"\nYAHPO: V1 {yahpo_results['V1']} - {yahpo_results['Thompson_All']} Thompson_All (ties: {yahpo_results['ties']})")
    
    total_v1 = v1_wins + yahpo_results['V1']
    total_ta = ta_wins + yahpo_results['Thompson_All']
    
    print(f"\nTOTALE: V1 {total_v1} - {total_ta} Thompson_All")
    
    if total_v1 > total_ta:
        print("\n🏆 VINCITORE: ALBA_V1 (Thompson cat + LGS cont)")
    elif total_ta > total_v1:
        print("\n🏆 VINCITORE: Thompson_All")
    else:
        print("\n🤝 PAREGGIO")
