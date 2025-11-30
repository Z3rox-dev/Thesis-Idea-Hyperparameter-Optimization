#!/usr/bin/env python3
"""
Benchmark QuadHPO vs Random vs Optuna on HPOBench NN TabularBenchmark
Uses pre-computed results from HPOBench tabular data
"""

import sys
import os
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import argparse
from datetime import datetime

# Import HPOBench Tabular benchmark
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

# Import optimizers  
from hpo_main import HPOptimizer
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_quadhpo(benchmark, cs, budget, seed, verbose=False):
    """Run QuadHPO optimizer"""
    np.random.seed(seed)
    
    # Get hyperparameter names and their sequences
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    
    optimizer = HPOptimizer(
        bounds=[(0, 1)] * dim,
        seed=seed
    )
    
    # Get max fidelity
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(x):
        # Map normalized [0,1] to discrete config values
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    QuadHPO trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    best_x, best_y = optimizer.optimize(objective, budget=budget)
    return best_y


def run_random(benchmark, cs, budget, seed, verbose=False):
    """Run Random Search"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    max_fidelity = benchmark.get_max_fidelity()
    
    best_y = float('inf')
    
    for trial in range(budget):
        x = np.random.uniform(0, 1, dim)
        
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        if y < best_y:
            best_y = y
        
        if verbose:
            print(f"    Random  trial {trial+1:3d}: loss={y:.6f} (best={best_y:.6f})")
    
    return best_y


def run_optuna(benchmark, cs, budget, seed, verbose=False):
    """Run Optuna TPE"""
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(trial):
        config = {}
        for name in hp_names:
            seq = hp_seqs[name]
            config[name] = trial.suggest_categorical(name, seq)
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    Optuna  trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value


def main():
    parser = argparse.ArgumentParser(description='Benchmark QuadHPO on HPOBench NN TabularBenchmark')
    parser.add_argument('--budget', type=int, default=50, help='Number of evaluations per seed')
    parser.add_argument('--seeds', type=str, default="0,1,2,3,4", help='Comma-separated seeds')
    parser.add_argument('--task_id', type=int, default=31, help='OpenML task ID')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-trial output')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    budget = args.budget
    task_id = args.task_id
    
    print("=" * 78)
    print("HPOBench NN TabularBenchmark: QuadHPO vs Random vs Optuna")
    print("=" * 78)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {budget}")
    print(f"Task ID: {task_id}")
    
    # Create benchmark
    benchmark = TabularBenchmark(model='nn', task_id=task_id)
    cs = benchmark.get_configuration_space()
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    print(f"Dimensions: {len(hp_names)}")
    print(f"Hyperparameters: {hp_names}")
    print()
    
    results_quadhpo = []
    results_random = []
    results_optuna = []
    
    for seed in seeds:
        if args.verbose:
            print(f"\n--- Seed {seed} ---")
        
        # Run QuadHPO
        if args.verbose:
            print("  QuadHPO:")
        best_quadhpo = run_quadhpo(benchmark, cs, budget, seed, verbose=args.verbose)
        results_quadhpo.append(best_quadhpo)
        
        # Run Random
        if args.verbose:
            print("  Random:")
        best_random = run_random(benchmark, cs, budget, seed, verbose=args.verbose)
        results_random.append(best_random)
        
        # Run Optuna
        if args.verbose:
            print("  Optuna:")
        best_optuna = run_optuna(benchmark, cs, budget, seed, verbose=args.verbose)
        results_optuna.append(best_optuna)
        
        print(f"[Seed {seed:3d}] QuadHPO: {best_quadhpo:.6f} | Rand: {best_random:.6f} | Optuna: {best_optuna:.6f}")
    
    # Summary statistics
    mean_quadhpo = np.mean(results_quadhpo)
    std_quadhpo = np.std(results_quadhpo)
    mean_random = np.mean(results_random)
    std_random = np.std(results_random)
    mean_optuna = np.mean(results_optuna)
    std_optuna = np.std(results_optuna)
    
    print()
    print(f"QuadHPO mean best loss (nn/tabular): {mean_quadhpo:.6f} ± {std_quadhpo:.6f}")
    print(f"Random  mean best loss (nn/tabular): {mean_random:.6f} ± {std_random:.6f}")
    print(f"Optuna  mean best loss (nn/tabular): {mean_optuna:.6f} ± {std_optuna:.6f}")
    print()
    
    # Determine winner
    winner = "QuadHPO" if mean_quadhpo <= min(mean_random, mean_optuna) else \
             ("Random" if mean_random <= mean_optuna else "Optuna")
    best_loss = min(mean_quadhpo, mean_random, mean_optuna)
    print(f">>> WINNER: {winner} with mean loss = {best_loss:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/mnt/workspace/thesis/tests/benchmark_nn_tabular_results_{timestamp}.txt"
    
    with open(log_path, 'w') as f:
        f.write("HPOBench NN TabularBenchmark Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Hyperparameters: {hp_names}\n\n")
        
        for i, seed in enumerate(seeds):
            f.write(f"[Seed {seed:3d}] QuadHPO: {results_quadhpo[i]:.6f} | ")
            f.write(f"Rand: {results_random[i]:.6f} | Optuna: {results_optuna[i]:.6f}\n")
        
        f.write(f"\nQuadHPO: {mean_quadhpo:.6f} ± {std_quadhpo:.6f}\n")
        f.write(f"Random:  {mean_random:.6f} ± {std_random:.6f}\n")
        f.write(f"Optuna:  {mean_optuna:.6f} ± {std_optuna:.6f}\n")
        f.write(f"\nWinner: {winner}\n")
    
    print(f"\nLog salvato in: {log_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
