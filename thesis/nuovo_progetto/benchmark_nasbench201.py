#!/usr/bin/env python3
"""
Benchmark CopulaHPO v2 vs Optuna vs baselines on HPOBench NASBench-201.
NASBench-201 has 6 categorical hyperparameters (cell operations).
"""
import sys
sys.path.insert(0, '/mnt/workspace/HPOBench')

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import argparse
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

# HPOBench NASBench-201
from hpobench.benchmarks.nas.nasbench_201 import NasBench201BaseBenchmark

# CopulaHPO
from copula_hpo_v2 import CopulaHPO

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def convert_config_space_to_copula(cs):
    """Convert ConfigSpace to CopulaHPO format (list of HyperparameterSpec)."""
    from copula_hpo_v2 import HyperparameterSpec
    
    specs = []
    for hp in cs.get_hyperparameters():
        name = hp.name
        if hasattr(hp, 'choices'):  # Categorical
            specs.append(HyperparameterSpec(
                name=name,
                type='categorical',
                bounds=list(hp.choices)
            ))
        elif hasattr(hp, 'lower'):  # Float or Integer
            if hp.__class__.__name__ == 'UniformFloatHyperparameter':
                specs.append(HyperparameterSpec(
                    name=name,
                    type='continuous',
                    bounds=(hp.lower, hp.upper)
                ))
            else:  # Integer
                specs.append(HyperparameterSpec(
                    name=name,
                    type='integer',
                    bounds=(hp.lower, hp.upper)
                ))
    return specs


def objective_wrapper(bench, config_dict, epoch=199):
    """Evaluate a configuration on NASBench-201."""
    try:
        result = bench.objective_function(config_dict, fidelity={"epoch": epoch})
        # NASBench-201 returns valid_precision (0-100 scale) - we want to minimize error
        valid_precision = result['function_value']  # e.g., 91.5 for 91.5%
        return 100.0 - valid_precision  # Convert to error (minimize)
    except Exception as e:
        print(f"Error evaluating config: {e}")
        return 100.0  # Worst case


def run_copula(bench, cs, n_trials, seed, mode='elite'):
    """Run CopulaHPO on NASBench-201."""
    space = convert_config_space_to_copula(cs)
    opt = CopulaHPO(space, mode=mode, seed=seed)
    
    best_value = float('inf')
    for _ in range(n_trials):
        x = opt.ask()
        y = objective_wrapper(bench, x)
        opt.tell(x, y)
        best_value = min(best_value, y)
    
    return 100.0 - best_value  # Return precision (%)


def run_optuna(bench, cs, n_trials, seed):
    """Run Optuna TPE on NASBench-201."""
    def objective(trial):
        config = {}
        for hp in cs.get_hyperparameters():
            name = hp.name
            if hasattr(hp, 'choices'):
                config[name] = trial.suggest_categorical(name, list(hp.choices))
            elif hasattr(hp, 'lower'):
                if hp.__class__.__name__ == 'UniformFloatHyperparameter':
                    config[name] = trial.suggest_float(name, hp.lower, hp.upper)
                else:
                    config[name] = trial.suggest_int(name, hp.lower, hp.upper)
        
        return objective_wrapper(bench, config)  # Minimize error
    
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        constant_liar=True
    )
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return 100.0 - study.best_value  # Return precision (%)


def run_random(bench, cs, n_trials, seed):
    """Run random search on NASBench-201."""
    rng = np.random.default_rng(seed)
    
    best_value = float('inf')
    for _ in range(n_trials):
        config = {}
        for hp in cs.get_hyperparameters():
            name = hp.name
            if hasattr(hp, 'choices'):
                config[name] = rng.choice(list(hp.choices))
            elif hasattr(hp, 'lower'):
                if hp.__class__.__name__ == 'UniformFloatHyperparameter':
                    config[name] = rng.uniform(hp.lower, hp.upper)
                else:
                    config[name] = rng.integers(hp.lower, hp.upper+1)
        
        y = objective_wrapper(bench, config)
        best_value = min(best_value, y)
    
    return 100.0 - best_value  # Return precision (%)


def benchmark_dataset(dataset, n_trials, seeds, methods):
    """Benchmark on a single NASBench-201 dataset."""
    print(f"\n{'='*60}")
    print(f"NASBench-201 - Dataset: {dataset}")
    print(f"{'='*60}")
    
    bench = NasBench201BaseBenchmark(dataset=dataset)
    cs = bench.get_configuration_space()
    
    print(f"Config space: {len(list(cs.get_hyperparameters()))} hyperparameters")
    
    results = []
    wins = defaultdict(int)
    
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        scores = {}
        
        if 'Copula' in methods:
            scores['Copula'] = run_copula(bench, cs, n_trials, seed)
        
        if 'Optuna' in methods:
            scores['Optuna'] = run_optuna(bench, cs, n_trials, seed)
        
        if 'Random' in methods:
            scores['Random'] = run_random(bench, cs, n_trials, seed)
        
        # Determine winner
        best_method = max(scores, key=scores.get)
        wins[best_method] += 1
        
        score_str = " ".join([f"{m}={v:.4f}" for m, v in scores.items()])
        print(f"    {score_str} [Winner: {best_method}]")
        
        results.append({
            'dataset': dataset,
            'seed': seed,
            'scores': scores,
            'winner': best_method
        })
    
    return results, wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='cifar10-valid,cifar100,ImageNet16-120',
                        help='Comma-separated list of datasets')
    parser.add_argument('--budget', type=int, default=100, help='Number of trials')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--methods', type=str, default='Copula,Optuna,Random',
                        help='Comma-separated list of methods')
    parser.add_argument('--output', type=str, default='results_nasbench201.json',
                        help='Output file path')
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]
    seeds = list(range(args.seeds))
    
    print(f"Benchmark: NASBench-201 (HPOBench)")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Budget: {args.budget} trials")
    print(f"Seeds: {seeds}")
    
    all_results = []
    total_wins = defaultdict(int)
    
    for dataset in datasets:
        results, wins = benchmark_dataset(dataset, args.budget, seeds, methods)
        all_results.extend(results)
        for m, w in wins.items():
            total_wins[m] += w
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - NASBench-201")
    print("="*60)
    
    total_runs = len(all_results)
    for method in methods:
        pct = (total_wins[method] / total_runs) * 100 if total_runs > 0 else 0
        print(f"  {method}: {total_wins[method]}/{total_runs} wins ({pct:.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_data = {
        'benchmark': 'NASBench-201',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'datasets': datasets,
            'budget': args.budget,
            'seeds': seeds,
            'methods': methods
        },
        'results': all_results,
        'total_wins': dict(total_wins)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
