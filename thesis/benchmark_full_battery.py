#!/usr/bin/env python3
"""
BATTERIA COMPLETA DI BENCHMARK: ALBA_V1 vs Optuna TPE
- YAHPO Gym (4 scenari)
- JAHS-Bench-201 (3 dataset)
- 10 seeds per dataset
- Budget 2000 con checkpoint ogni 500
- Salvataggio risultati intermedi
"""

import sys
import os
import json
import time
import numpy as np
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Paths
sys.path.insert(0, '/mnt/workspace/thesis')
RESULTS_DIR = '/mnt/workspace/thesis/benchmark_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config
BUDGET = 2000
CHECKPOINTS = [100, 250, 500, 1000, 1500, 2000]
N_SEEDS = 10
SEEDS = list(range(42, 42 + N_SEEDS))

# Timestamp per questa run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f"{RESULTS_DIR}/full_benchmark_{TIMESTAMP}.json"

print("="*80)
print(f"BATTERIA COMPLETA DI BENCHMARK")
print(f"Budget: {BUDGET}, Seeds: {N_SEEDS}, Checkpoints: {CHECKPOINTS}")
print(f"Results file: {RESULTS_FILE}")
print("="*80)

# Risultati globali
all_results = {
    'config': {
        'budget': BUDGET,
        'checkpoints': CHECKPOINTS,
        'n_seeds': N_SEEDS,
        'seeds': SEEDS,
        'timestamp': TIMESTAMP,
    },
    'yahpo': {},
    'jahs': {},
}


def save_results():
    """Salva risultati intermedi."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [Saved to {RESULTS_FILE}]")


#############################################
# YAHPO BENCHMARK
#############################################

def run_yahpo_benchmarks():
    print("\n" + "="*80)
    print("PARTE 1: YAHPO GYM")
    print("="*80)
    
    try:
        from yahpo_gym import BenchmarkSet
        from ALBA_V1 import ALBA as ALBA_V1
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except ImportError as e:
        print(f"  SKIP YAHPO: {e}")
        return
    
    scenarios = ['rbv2_xgboost', 'rbv2_ranger', 'rbv2_svm', 'iaml_xgboost']
    
    for scenario_name in scenarios:
        print(f"\n>>> YAHPO Scenario: {scenario_name}")
        
        try:
            bench = BenchmarkSet(scenario_name)
            cs = bench.get_opt_space()
            instances = list(bench.instances)[:3]  # 3 instances per scenario
            
            obj_name = bench.config.y_names[0]
            maximize = 'acc' in obj_name.lower() or 'auc' in obj_name.lower()
            
            hps = list(cs.values())
            dim = len(hps)
            
            # Detect categoricals
            categorical_dims = []
            for i, hp in enumerate(hps):
                if hasattr(hp, 'choices'):
                    categorical_dims.append((i, len(hp.choices)))
            
            print(f"  Dim: {dim}, Cat: {len(categorical_dims)}, Instances: {len(instances)}")
            
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
                                if kernel_choices[k_idx] != 'polynomial':
                                    continue
                        elif name == 'gamma':
                            kernel_idx = next((j for j, h in enumerate(hps) if h.name == 'kernel'), None)
                            if kernel_idx is not None:
                                kernel_hp = hps[kernel_idx]
                                kernel_choices = list(kernel_hp.choices)
                                k_idx = int(np.round(x[kernel_idx] * (len(kernel_choices) - 1)))
                                k_idx = max(0, min(len(kernel_choices) - 1, k_idx))
                                if kernel_choices[k_idx] != 'radial':
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
            
            scenario_results = {
                'dim': dim,
                'n_cat': len(categorical_dims),
                'objective': obj_name,
                'maximize': maximize,
                'instances': {},
            }
            
            for inst in instances:
                print(f"\n  Instance: {inst}")
                inst_results = {'alba': {}, 'optuna': {}}
                
                for seed in SEEDS:
                    # ALBA run with checkpoints
                    opt = ALBA_V1(
                        bounds=[(0.0, 1.0)] * dim,
                        maximize=maximize,
                        seed=seed,
                        total_budget=BUDGET,
                        categorical_dims=categorical_dims,
                    )
                    
                    alba_best = -np.inf if maximize else np.inf
                    alba_checkpoints = {}
                    
                    for it in range(BUDGET):
                        x = opt.ask()
                        config = config_from_x(x, inst)
                        try:
                            result = bench.objective_function(configuration=config, seed=seed)
                            score = result[0][obj_name]
                        except:
                            score = 0.0 if maximize else 1.0
                        
                        opt.tell(x, score)
                        if maximize:
                            alba_best = max(alba_best, score)
                        else:
                            alba_best = min(alba_best, score)
                        
                        if (it + 1) in CHECKPOINTS:
                            alba_checkpoints[it + 1] = float(alba_best)
                    
                    inst_results['alba'][seed] = alba_checkpoints
                    
                    # Optuna run with checkpoints
                    optuna_best = -np.inf if maximize else np.inf
                    optuna_checkpoints = {}
                    optuna_iter = [0]
                    
                    def optuna_objective(trial):
                        x = np.array([trial.suggest_float(f'x{i}', 0, 1) for i in range(dim)])
                        config = config_from_x(x, inst)
                        try:
                            result = bench.objective_function(configuration=config, seed=seed)
                            score = result[0][obj_name]
                        except:
                            score = 0.0 if maximize else 1.0
                        
                        nonlocal optuna_best
                        if maximize:
                            optuna_best = max(optuna_best, score)
                        else:
                            optuna_best = min(optuna_best, score)
                        
                        optuna_iter[0] += 1
                        if optuna_iter[0] in CHECKPOINTS:
                            optuna_checkpoints[optuna_iter[0]] = float(optuna_best)
                        
                        return score
                    
                    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
                    direction = 'maximize' if maximize else 'minimize'
                    study = optuna.create_study(direction=direction, sampler=sampler)
                    study.optimize(optuna_objective, n_trials=BUDGET, show_progress_bar=False)
                    
                    inst_results['optuna'][seed] = optuna_checkpoints
                    
                    # Print progress
                    alba_final = alba_checkpoints.get(BUDGET, alba_best)
                    optuna_final = optuna_checkpoints.get(BUDGET, optuna_best)
                    winner = "ALBA" if (maximize and alba_final > optuna_final) or (not maximize and alba_final < optuna_final) else "Optuna"
                    print(f"    Seed {seed}: ALBA={alba_final:.4f} Optuna={optuna_final:.4f} → {winner}")
                
                scenario_results['instances'][str(inst)] = inst_results
            
            all_results['yahpo'][scenario_name] = scenario_results
            save_results()
            
        except Exception as e:
            print(f"  ERROR in {scenario_name}: {e}")
            import traceback
            traceback.print_exc()


#############################################
# JAHS BENCHMARK  
#############################################

def run_jahs_benchmarks():
    print("\n" + "="*80)
    print("PARTE 2: JAHS-BENCH-201")
    print("="*80)
    
    try:
        from benchmark_jahs import JAHSBenchWrapper, run_optuna_tpe
        from ALBA_V1 import ALBA as ALBA_V1
    except ImportError as e:
        print(f"  SKIP JAHS: {e}")
        return
    
    tasks = ['cifar10', 'fashion_mnist', 'colorectal_histology']
    
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]
    
    for task in tasks:
        print(f"\n>>> JAHS Task: {task}")
        
        try:
            wrapper = JAHSBenchWrapper(task=task)
            dim = wrapper.dim
            
            print(f"  Dim: {dim}")
            
            task_results = {
                'dim': dim,
                'alba': {},
                'optuna': {},
            }
            
            for seed in SEEDS:
                # ALBA run with checkpoints
                opt = ALBA_V1(
                    bounds=[(0.0, 1.0)] * dim,
                    maximize=False,
                    seed=seed,
                    split_depth_max=8,
                    total_budget=BUDGET,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    categorical_dims=categorical_dims,
                )
                
                wrapper.reset()
                alba_best = float('inf')
                alba_checkpoints = {}
                
                for it in range(BUDGET):
                    x = opt.ask()
                    y = wrapper.evaluate_array(x)
                    opt.tell(x, y)
                    alba_best = min(alba_best, y)
                    
                    if (it + 1) in CHECKPOINTS:
                        alba_checkpoints[it + 1] = float(alba_best)
                
                task_results['alba'][seed] = alba_checkpoints
                
                # Optuna run with checkpoints
                import optuna
                optuna.logging.set_verbosity(optuna.logging.ERROR)
                
                wrapper.reset()
                optuna_best = float('inf')
                optuna_checkpoints = {}
                optuna_iter = [0]
                
                def optuna_objective(trial):
                    x = np.array([trial.suggest_float(f'x{i}', 0, 1) for i in range(dim)])
                    y = wrapper.evaluate_array(x)
                    
                    nonlocal optuna_best
                    optuna_best = min(optuna_best, y)
                    
                    optuna_iter[0] += 1
                    if optuna_iter[0] in CHECKPOINTS:
                        optuna_checkpoints[optuna_iter[0]] = float(optuna_best)
                    
                    return y
                
                sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
                study = optuna.create_study(direction='minimize', sampler=sampler)
                study.optimize(optuna_objective, n_trials=BUDGET, show_progress_bar=False)
                
                task_results['optuna'][seed] = optuna_checkpoints
                
                # Print progress
                alba_final = alba_checkpoints.get(BUDGET, alba_best)
                optuna_final = optuna_checkpoints.get(BUDGET, optuna_best)
                alba_acc = (1 - alba_final) * 100
                optuna_acc = (1 - optuna_final) * 100
                winner = "ALBA" if alba_final < optuna_final else "Optuna"
                print(f"    Seed {seed}: ALBA={alba_acc:.2f}% Optuna={optuna_acc:.2f}% → {winner}")
            
            all_results['jahs'][task] = task_results
            save_results()
            
        except Exception as e:
            print(f"  ERROR in {task}: {e}")
            import traceback
            traceback.print_exc()


#############################################
# SUMMARY
#############################################

def generate_summary():
    print("\n" + "="*80)
    print("RIEPILOGO FINALE")
    print("="*80)
    
    summary = {
        'yahpo': {'alba_wins': 0, 'optuna_wins': 0, 'ties': 0, 'by_scenario': {}},
        'jahs': {'alba_wins': 0, 'optuna_wins': 0, 'ties': 0, 'by_task': {}},
        'total': {'alba_wins': 0, 'optuna_wins': 0, 'ties': 0},
        'checkpoint_analysis': {},
    }
    
    # YAHPO summary
    print("\n--- YAHPO GYM ---")
    for scenario, data in all_results.get('yahpo', {}).items():
        maximize = data.get('maximize', True)
        scenario_alba = 0
        scenario_optuna = 0
        scenario_ties = 0
        
        for inst, inst_data in data.get('instances', {}).items():
            for seed in SEEDS:
                alba_final = inst_data['alba'].get(seed, {}).get(BUDGET, None)
                optuna_final = inst_data['optuna'].get(seed, {}).get(BUDGET, None)
                
                if alba_final is None or optuna_final is None:
                    continue
                
                if maximize:
                    if alba_final > optuna_final + 1e-6:
                        scenario_alba += 1
                    elif optuna_final > alba_final + 1e-6:
                        scenario_optuna += 1
                    else:
                        scenario_ties += 1
                else:
                    if alba_final < optuna_final - 1e-6:
                        scenario_alba += 1
                    elif optuna_final < alba_final - 1e-6:
                        scenario_optuna += 1
                    else:
                        scenario_ties += 1
        
        summary['yahpo']['by_scenario'][scenario] = {
            'alba': scenario_alba, 'optuna': scenario_optuna, 'ties': scenario_ties
        }
        summary['yahpo']['alba_wins'] += scenario_alba
        summary['yahpo']['optuna_wins'] += scenario_optuna
        summary['yahpo']['ties'] += scenario_ties
        
        print(f"  {scenario}: ALBA {scenario_alba} - {scenario_optuna} Optuna (ties: {scenario_ties})")
    
    print(f"\n  YAHPO TOTAL: ALBA {summary['yahpo']['alba_wins']} - {summary['yahpo']['optuna_wins']} Optuna")
    
    # JAHS summary
    print("\n--- JAHS-BENCH-201 ---")
    for task, data in all_results.get('jahs', {}).items():
        task_alba = 0
        task_optuna = 0
        task_ties = 0
        
        alba_errors = []
        optuna_errors = []
        
        for seed in SEEDS:
            alba_final = data['alba'].get(seed, {}).get(BUDGET, None)
            optuna_final = data['optuna'].get(seed, {}).get(BUDGET, None)
            
            if alba_final is None or optuna_final is None:
                continue
            
            alba_errors.append(alba_final)
            optuna_errors.append(optuna_final)
            
            if alba_final < optuna_final - 1e-6:
                task_alba += 1
            elif optuna_final < alba_final - 1e-6:
                task_optuna += 1
            else:
                task_ties += 1
        
        summary['jahs']['by_task'][task] = {
            'alba': task_alba, 'optuna': task_optuna, 'ties': task_ties,
            'alba_mean_acc': (1 - np.mean(alba_errors)) * 100 if alba_errors else 0,
            'optuna_mean_acc': (1 - np.mean(optuna_errors)) * 100 if optuna_errors else 0,
        }
        summary['jahs']['alba_wins'] += task_alba
        summary['jahs']['optuna_wins'] += task_optuna
        summary['jahs']['ties'] += task_ties
        
        alba_acc = summary['jahs']['by_task'][task]['alba_mean_acc']
        optuna_acc = summary['jahs']['by_task'][task]['optuna_mean_acc']
        print(f"  {task}: ALBA {task_alba} - {task_optuna} Optuna | ALBA={alba_acc:.2f}% Optuna={optuna_acc:.2f}%")
    
    print(f"\n  JAHS TOTAL: ALBA {summary['jahs']['alba_wins']} - {summary['jahs']['optuna_wins']} Optuna")
    
    # Grand total
    summary['total']['alba_wins'] = summary['yahpo']['alba_wins'] + summary['jahs']['alba_wins']
    summary['total']['optuna_wins'] = summary['yahpo']['optuna_wins'] + summary['jahs']['optuna_wins']
    summary['total']['ties'] = summary['yahpo']['ties'] + summary['jahs']['ties']
    
    print("\n" + "="*80)
    print("GRAND TOTAL")
    print("="*80)
    print(f"ALBA: {summary['total']['alba_wins']} wins")
    print(f"Optuna: {summary['total']['optuna_wins']} wins")
    print(f"Ties: {summary['total']['ties']}")
    
    if summary['total']['alba_wins'] > summary['total']['optuna_wins']:
        print("\n🏆 VINCITORE ASSOLUTO: ALBA")
    elif summary['total']['optuna_wins'] > summary['total']['alba_wins']:
        print("\n🏆 VINCITORE ASSOLUTO: Optuna")
    else:
        print("\n🤝 PAREGGIO")
    
    # Checkpoint analysis
    print("\n" + "="*80)
    print("ANALISI PER CHECKPOINT (convergenza)")
    print("="*80)
    
    for cp in CHECKPOINTS:
        alba_wins_cp = 0
        optuna_wins_cp = 0
        
        # YAHPO
        for scenario, data in all_results.get('yahpo', {}).items():
            maximize = data.get('maximize', True)
            for inst, inst_data in data.get('instances', {}).items():
                for seed in SEEDS:
                    alba_val = inst_data['alba'].get(seed, {}).get(cp, None)
                    optuna_val = inst_data['optuna'].get(seed, {}).get(cp, None)
                    if alba_val is None or optuna_val is None:
                        continue
                    if maximize:
                        if alba_val > optuna_val + 1e-6:
                            alba_wins_cp += 1
                        elif optuna_val > alba_val + 1e-6:
                            optuna_wins_cp += 1
                    else:
                        if alba_val < optuna_val - 1e-6:
                            alba_wins_cp += 1
                        elif optuna_val < alba_val - 1e-6:
                            optuna_wins_cp += 1
        
        # JAHS
        for task, data in all_results.get('jahs', {}).items():
            for seed in SEEDS:
                alba_val = data['alba'].get(seed, {}).get(cp, None)
                optuna_val = data['optuna'].get(seed, {}).get(cp, None)
                if alba_val is None or optuna_val is None:
                    continue
                if alba_val < optuna_val - 1e-6:
                    alba_wins_cp += 1
                elif optuna_val < alba_val - 1e-6:
                    optuna_wins_cp += 1
        
        summary['checkpoint_analysis'][cp] = {'alba': alba_wins_cp, 'optuna': optuna_wins_cp}
        print(f"  Budget {cp:4d}: ALBA {alba_wins_cp:3d} - {optuna_wins_cp:3d} Optuna")
    
    all_results['summary'] = summary
    save_results()
    
    return summary


#############################################
# MAIN
#############################################

if __name__ == "__main__":
    start_time = time.time()
    
    # Run benchmarks
    run_yahpo_benchmarks()
    run_jahs_benchmarks()
    
    # Generate summary
    summary = generate_summary()
    
    elapsed = time.time() - start_time
    print(f"\n\nTempo totale: {elapsed/60:.1f} minuti")
    print(f"Risultati salvati in: {RESULTS_FILE}")
