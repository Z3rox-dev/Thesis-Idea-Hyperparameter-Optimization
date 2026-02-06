#!/usr/bin/env python3
"""
Benchmark Triple: HPO_v5s (Bayesian+Trust) vs Random vs Optuna vs HPO_v3
on HPOBench XGBoost Tabular Benchmark
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace')

# Import HPOBench
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

# Import optimizers
from thesis.hpo_v5s_more_novelty_standalone import HPOptimizerV5s as HPOptimizerMain
from thesis.hpo_lgs_v3 import HPOptimizer as HPOptimizerV3
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_hpo_main(benchmark, cs, budget, seed, verbose=False):
    """Run HPO_v5s (Bayesian+Trust)"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    dim = len(hp_names)
    
    optimizer = HPOptimizerMain(
        bounds=[(0, 1)] * dim,
        seed=seed,
        total_budget=budget,
        maximize=False
    )
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            hp = hps[i]
            val_norm = x[i]
            
            if hasattr(hp, 'sequence'): # Ordinal
                idx = int(np.clip(val_norm * len(hp.sequence), 0, len(hp.sequence) - 1))
                val = hp.sequence[idx]
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'): # Numerical
                if hp.log:
                    log_lower = np.log(hp.lower)
                    log_upper = np.log(hp.upper)
                    val = np.exp(log_lower + val_norm * (log_upper - log_lower))
                else:
                    val = hp.lower + val_norm * (hp.upper - hp.lower)
                
                if 'Integer' in str(type(hp)):
                    val = int(round(val))
            else:
                # Fallback for categorical if needed (not common in this benchmark)
                val = hp.choices[int(np.clip(val_norm * len(hp.choices), 0, len(hp.choices) - 1))]
                
            config[name] = val
            
        res = benchmark.objective_function(configuration=config)
        return res['function_value']

    best_x, best_y = optimizer.optimize(objective, budget=budget)
    return best_y

def run_random_search(benchmark, cs, budget, seed, verbose=False):
    """Run Random Search"""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    
    best_y = float('inf')
    
    for _ in range(budget):
        config = {}
        for hp in hps:
            if hasattr(hp, 'sequence'):
                config[hp.name] = rng.choice(hp.sequence)
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                if hp.log:
                    val = np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper)))
                else:
                    val = rng.uniform(hp.lower, hp.upper)
                
                if 'Integer' in str(type(hp)):
                    val = int(round(val))
                config[hp.name] = val
            else:
                config[hp.name] = rng.choice(hp.choices)
                
        res = benchmark.objective_function(configuration=config)
        y = res['function_value']
        if y < best_y:
            best_y = y
            
    return best_y

def run_hpo_v3(benchmark, cs, budget, seed, verbose=False):
    """Run HPO v3 (Previous Baseline)"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    dim = len(hp_names)
    
    optimizer = HPOptimizerV3(
        bounds=[(0, 1)] * dim,
        seed=seed,
        # total_budget=budget, # Removed as it's not supported in v3
        maximize=False
    )
    optimizer.exploration_budget = int(budget * (1 - optimizer.local_search_ratio)) # Manually set exploration budget
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            hp = hps[i]
            val_norm = x[i]
            
            if hasattr(hp, 'sequence'):
                idx = int(np.clip(val_norm * len(hp.sequence), 0, len(hp.sequence) - 1))
                val = hp.sequence[idx]
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                if hp.log:
                    log_lower = np.log(hp.lower)
                    log_upper = np.log(hp.upper)
                    val = np.exp(log_lower + val_norm * (log_upper - log_lower))
                else:
                    val = hp.lower + val_norm * (hp.upper - hp.lower)
                
                if 'Integer' in str(type(hp)):
                    val = int(round(val))
            else:
                val = hp.choices[int(np.clip(val_norm * len(hp.choices), 0, len(hp.choices) - 1))]
                
            config[name] = val
            
        res = benchmark.objective_function(configuration=config)
        return res['function_value']

    best_x, best_y = optimizer.optimize(objective, budget=budget)
    return best_y

def run_optuna(benchmark, cs, budget, seed, verbose=False):
    """Run Optuna (TPE Multivariate)"""
    
    def objective(trial):
        config = {}
        for hp in cs.values():
            if hasattr(hp, 'sequence'):
                # Optuna doesn't support sequence directly well, map to categorical or int
                # Using float/int index is better for order
                idx = trial.suggest_int(hp.name, 0, len(hp.sequence) - 1)
                config[hp.name] = hp.sequence[idx]
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                if 'Integer' in str(type(hp)):
                    config[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper, log=hp.log)
                else:
                    config[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper, log=hp.log)
            else:
                config[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
        
        res = benchmark.objective_function(configuration=config)
        return res['function_value']

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget)
    return study.best_value

def main():
    parser = argparse.ArgumentParser(description='Benchmark XGBoost')
    parser.add_argument('--budget', type=int, default=200, help='Number of evaluations per seed')
    parser.add_argument('--seeds', type=str, default="10,11,12,13,14", help='Comma-separated seeds')
    parser.add_argument('--task_id', type=int, default=167149, help='OpenML task ID (default: 167149)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-trial output')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    budget = args.budget
    task_id = args.task_id
    
    print("=" * 80)
    print("Triple Benchmark XGBoost: HPO_v5s (Bayesian) vs Random vs Optuna vs HPO_v3")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {budget}")
    print(f"Task ID: {task_id}")
    
    # Create benchmark
    benchmark = TabularBenchmark(model='xgb', task_id=task_id)
    cs = benchmark.get_configuration_space()
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    print(f"Dimensions: {len(hp_names)}")
    print(f"Hyperparameters: {hp_names}")
    print()
    
    results_main = []
    results_random = []
    results_v3 = []
    results_optuna = []
    
    for seed in seeds:
        if args.verbose:
            print(f"\n--- Seed {seed} ---")
        
        # Run HPO Main (v5s)
        best_main = run_hpo_main(benchmark, cs, budget, seed, verbose=args.verbose)
        results_main.append(best_main)
        
        # Run Random Search
        best_random = run_random_search(benchmark, cs, budget, seed, verbose=args.verbose)
        results_random.append(best_random)

        # Run HPO v3
        best_v3 = run_hpo_v3(benchmark, cs, budget, seed, verbose=args.verbose)
        results_v3.append(best_v3)
        
        # Run Optuna
        best_optuna = run_optuna(benchmark, cs, budget, seed, verbose=args.verbose)
        results_optuna.append(best_optuna)
        
        print(f"[Seed {seed:3d}] HPO_v5s: {best_main:.6f} | Random: {best_random:.6f} | HPO_v3: {best_v3:.6f} | Optuna: {best_optuna:.6f}")
    
    # Summary statistics
    mean_main = np.mean(results_main)
    std_main = np.std(results_main)
    mean_random = np.mean(results_random)
    std_random = np.std(results_random)
    mean_v3 = np.mean(results_v3)
    std_v3 = np.std(results_v3)
    mean_optuna = np.mean(results_optuna)
    std_optuna = np.std(results_optuna)
    
    print()
    print(f"HPO_v5s (Bayesian) mean best loss: {mean_main:.6f} ± {std_main:.6f}")
    print(f"Random Search      mean best loss: {mean_random:.6f} ± {std_random:.6f}")
    print(f"HPO_v3             mean best loss: {mean_v3:.6f} ± {std_v3:.6f}")
    print(f"Optuna (multi)     mean best loss: {mean_optuna:.6f} ± {std_optuna:.6f}")
    print()
    
    # Determine winner
    results = [
        ("HPO_v5s", mean_main),
        ("Random", mean_random),
        ("HPO_v3", mean_v3),
        ("Optuna", mean_optuna)
    ]
    winner = min(results, key=lambda x: x[1])
    print(f">>> WINNER: {winner[0]} with mean loss = {winner[1]:.6f}")
    
    # Win counts
    wins_main = sum(1 for i in range(len(seeds)) if results_main[i] <= min(results_random[i], results_v3[i], results_optuna[i]))
    wins_random = sum(1 for i in range(len(seeds)) if results_random[i] <= min(results_main[i], results_v3[i], results_optuna[i]))
    wins_v3 = sum(1 for i in range(len(seeds)) if results_v3[i] <= min(results_main[i], results_random[i], results_optuna[i]))
    wins_optuna = sum(1 for i in range(len(seeds)) if results_optuna[i] <= min(results_main[i], results_random[i], results_v3[i]))
    print(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/mnt/workspace/thesis/tests/benchmark_xgb_v5s_{timestamp}.txt"
    
    with open(log_path, 'w') as f:
        f.write("Triple Benchmark XGBoost: HPO_v5s (Bayesian) vs Random vs HPO_v3 vs Optuna\n")
        f.write("=" * 60 + "\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Hyperparameters: {hp_names}\n\n")
        
        for i, seed in enumerate(seeds):
            f.write(f"[Seed {seed:3d}] HPO_v5s: {results_main[i]:.6f} | ")
            f.write(f"Random: {results_random[i]:.6f} | HPO_v3: {results_v3[i]:.6f} | Optuna: {results_optuna[i]:.6f}\n")
        
        f.write(f"\nHPO_v5s (Bayesian): {mean_main:.6f} ± {std_main:.6f}\n")
        f.write(f"Random Search:      {mean_random:.6f} ± {std_random:.6f}\n")
        f.write(f"HPO_v3:             {mean_v3:.6f} ± {std_v3:.6f}\n")
        f.write(f"Optuna (multi):     {mean_optuna:.6f} ± {std_optuna:.6f}\n")
        f.write(f"\nWinner: {winner[0]}\n")
        f.write(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}\n")
    
    print(f"\nLog salvato in: {log_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
