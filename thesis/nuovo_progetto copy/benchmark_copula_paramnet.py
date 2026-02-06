#!/usr/bin/env python3
"""
Benchmark CopulaHPO su HPOBench ParamNet.

ParamNet (Full) ha 8 hyperparameters:
- initial_lr_log10: [-6, -2]
- batch_size_log2: [3, 8]
- average_units_per_layer_log2: [4, 8]
- final_lr_fraction_log2: [-4, 0]
- shape_parameter_1: [0, 1]
- num_layers: [1, 5] (int)
- dropout_0: [0, 0.5]
- dropout_1: [0, 0.5]

Usage:
    python benchmark_copula_paramnet.py --budget 400 --seeds 3
"""

import argparse
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetMnistOnStepsBenchmark,
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)

from copula_hpo_v2 import CopulaHPO, HyperparameterSpec


BENCHMARK_CLASSES = {
    'mnist': ParamNetMnistOnStepsBenchmark,
    'adult': ParamNetAdultOnStepsBenchmark,
    'higgs': ParamNetHiggsOnStepsBenchmark,
    'letter': ParamNetLetterOnStepsBenchmark,
    'optdigits': ParamNetOptdigitsOnStepsBenchmark,
    'poker': ParamNetPokerOnStepsBenchmark,
}


def get_paramnet_specs():
    """Return HyperparameterSpec for ParamNet Full space (8 dims)."""
    return [
        HyperparameterSpec('initial_lr_log10', 'continuous', (-6.0, -2.0)),
        HyperparameterSpec('batch_size_log2', 'continuous', (3.0, 8.0)),
        HyperparameterSpec('average_units_per_layer_log2', 'continuous', (4.0, 8.0)),
        HyperparameterSpec('final_lr_fraction_log2', 'continuous', (-4.0, 0.0)),
        HyperparameterSpec('shape_parameter_1', 'continuous', (0.0, 1.0)),
        HyperparameterSpec('num_layers', 'integer', (1, 5)),
        HyperparameterSpec('dropout_0', 'continuous', (0.0, 0.5)),
        HyperparameterSpec('dropout_1', 'continuous', (0.0, 0.5)),
    ]


def copula_to_paramnet_config(config, cs):
    """Convert CopulaHPO config to ParamNet ConfigSpace config."""
    import ConfigSpace as CS
    
    cfg_dict = {
        'initial_lr_log10': float(config['initial_lr_log10']),
        'batch_size_log2': float(config['batch_size_log2']),
        'average_units_per_layer_log2': float(config['average_units_per_layer_log2']),
        'final_lr_fraction_log2': float(config['final_lr_fraction_log2']),
        'shape_parameter_1': float(config['shape_parameter_1']),
        'num_layers': int(config['num_layers']),
        'dropout_0': float(config['dropout_0']),
        'dropout_1': float(config['dropout_1']),
    }
    
    return CS.Configuration(cs, values=cfg_dict)


def run_benchmark(dataset: str, budget: int, seeds: list):
    """Run CopulaHPO on ParamNet."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    print(f"\n{'=' * 60}")
    print(f"HPOBench ParamNet - {dataset}")
    print(f"Budget: {budget}, Seeds: {seeds}")
    print("=" * 60)
    
    BenchClass = BENCHMARK_CLASSES[dataset]
    
    specs = get_paramnet_specs()
    
    results = {'copula': [], 'random': [], 'optuna': []}
    
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        
        bench = BenchClass(rng=seed)
        cs = bench.get_configuration_space(seed=seed)
        
        # CopulaHPO
        opt = CopulaHPO(specs, seed=seed)
        best_copula = float('inf')
        
        for i in range(budget):
            config = opt.ask()
            cfg = copula_to_paramnet_config(config, cs)
            result = bench.objective_function(cfg)
            y = float(result['function_value'])
            opt.tell(config, y)
            best_copula = min(best_copula, y)
        
        results['copula'].append(best_copula)
        
        # Optuna TPE
        def optuna_objective(trial):
            cfg_dict = {
                'initial_lr_log10': trial.suggest_float('initial_lr_log10', -6.0, -2.0),
                'batch_size_log2': trial.suggest_float('batch_size_log2', 3.0, 8.0),
                'average_units_per_layer_log2': trial.suggest_float('average_units_per_layer_log2', 4.0, 8.0),
                'final_lr_fraction_log2': trial.suggest_float('final_lr_fraction_log2', -4.0, 0.0),
                'shape_parameter_1': trial.suggest_float('shape_parameter_1', 0.0, 1.0),
                'num_layers': trial.suggest_int('num_layers', 1, 5),
                'dropout_0': trial.suggest_float('dropout_0', 0.0, 0.5),
                'dropout_1': trial.suggest_float('dropout_1', 0.0, 0.5),
            }
            import ConfigSpace as CS
            cfg = CS.Configuration(cs, values=cfg_dict)
            result = bench.objective_function(cfg)
            return float(result['function_value'])
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
        results['optuna'].append(study.best_value)
        
        # Random baseline
        rng = np.random.default_rng(seed)
        best_random = float('inf')
        
        for i in range(budget):
            # Sample from ConfigSpace directly
            rand_cfg = cs.sample_configuration()
            result = bench.objective_function(rand_cfg)
            y = float(result['function_value'])
            best_random = min(best_random, y)
        
        results['random'].append(best_random)
        
        print(f"Copula: {best_copula:.4f}, Optuna: {study.best_value:.4f}, Random: {best_random:.4f}")
    
    print(f"\n  Summary:")
    print(f"    CopulaHPO: {np.mean(results['copula']):.4f} ± {np.std(results['copula']):.4f}")
    print(f"    Optuna:    {np.mean(results['optuna']):.4f} ± {np.std(results['optuna']):.4f}")
    print(f"    Random:    {np.mean(results['random']):.4f} ± {np.std(results['random']):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--datasets', type=str, default='adult,mnist,letter',
                       help='Comma-separated list of datasets')
    args = parser.parse_args()
    
    seeds = list(range(42, 42 + args.seeds))
    datasets = args.datasets.split(',')
    
    print("=" * 60)
    print("CopulaHPO v2 - ParamNet Benchmark")
    print("=" * 60)
    
    all_results = {}
    for dataset in datasets:
        if dataset in BENCHMARK_CLASSES:
            all_results[dataset] = run_benchmark(dataset, args.budget, seeds)
        else:
            print(f"Unknown dataset: {dataset}")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for dataset, res in all_results.items():
        cop = np.mean(res['copula'])
        opt = np.mean(res['optuna'])
        rnd = np.mean(res['random'])
        best = min(cop, opt, rnd)
        winner = "CopulaHPO" if cop == best else ("Optuna" if opt == best else "Random")
        print(f"  {dataset}: Copula={cop:.4f}, Optuna={opt:.4f}, Random={rnd:.4f} -> {winner}")


if __name__ == '__main__':
    main()
