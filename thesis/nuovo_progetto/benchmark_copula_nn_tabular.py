#!/usr/bin/env python
"""Benchmark CopulaHPO su HPOBench TabularBenchmark (NN)."""
import sys
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

import numpy as np
import warnings
import argparse
warnings.filterwarnings('ignore')
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

import ConfigSpace as CS
from hpobench.benchmarks.ml import TabularBenchmark
from copula_hpo_v2 import CopulaHPO, HyperparameterSpec
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)


def build_specs_from_configspace(cs):
    """Convert ConfigSpace to CopulaHPO specs."""
    specs = []
    hps = cs.get_hyperparameters()
    
    for hp in hps:
        if hasattr(hp, 'sequence') and hp.sequence:
            # Ordinal as categorical
            seq = list(hp.sequence)
            specs.append(HyperparameterSpec(hp.name, 'categorical', seq))
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            specs.append(HyperparameterSpec(hp.name, 'continuous', (hp.lower, hp.upper)))
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            specs.append(HyperparameterSpec(hp.name, 'integer', (hp.lower, hp.upper)))
        elif isinstance(hp, CS.CategoricalHyperparameter):
            specs.append(HyperparameterSpec(hp.name, 'categorical', list(hp.choices)))
        else:
            raise ValueError(f'Unsupported HP: {hp}')
    
    return specs, hps


def copula_config_to_cs(config, cs, hps):
    """Convert CopulaHPO config to ConfigSpace config."""
    values = {}
    for hp in hps:
        val = config[hp.name]
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            val = int(val)
        values[hp.name] = val
    return CS.Configuration(cs, values=values)


def run_benchmark(model: str, task_id: int, budget: int, seeds: list):
    """Run benchmark on TabularBenchmark."""
    print(f"\n{'=' * 70}")
    print(f"HPOBench TabularBenchmark - model={model}, task_id={task_id}")
    print(f"Budget: {budget}, Seeds: {seeds}")
    print("=" * 70)
    
    bench = TabularBenchmark(model=model, task_id=task_id)
    cs = bench.get_configuration_space()
    max_fid = bench.get_max_fidelity()
    
    specs, hps = build_specs_from_configspace(cs)
    
    print(f"Dimensioni: {len(specs)} iperparametri")
    for s in specs:
        print(f"  - {s.name}: {s.type}")
    
    def objective(cfg):
        res = bench.objective_function(cfg, fidelity=max_fid, metric='acc')
        return float(res['function_value'])
    
    results = {'copula': [], 'optuna': [], 'random': []}
    
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        
        # CopulaHPO
        opt = CopulaHPO(specs, seed=seed)
        best_copula = float('inf')
        
        for i in range(budget):
            config = opt.ask()
            cfg = copula_config_to_cs(config, cs, hps)
            y = objective(cfg)
            opt.tell(config, y)
            best_copula = min(best_copula, y)
        
        results['copula'].append(best_copula)
        
        # Optuna TPE
        def optuna_obj(trial):
            values = {}
            for hp in hps:
                if hasattr(hp, 'sequence') and hp.sequence:
                    values[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
                elif isinstance(hp, CS.UniformFloatHyperparameter):
                    values[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper)
                elif isinstance(hp, CS.UniformIntegerHyperparameter):
                    values[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper)
                elif isinstance(hp, CS.CategoricalHyperparameter):
                    values[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            cfg = CS.Configuration(cs, values=values)
            return objective(cfg)
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(optuna_obj, n_trials=budget, show_progress_bar=False)
        results['optuna'].append(study.best_value)
        
        # Random
        rng = np.random.default_rng(seed)
        best_random = float('inf')
        for _ in range(budget):
            cfg = cs.sample_configuration()
            y = objective(cfg)
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
    parser.add_argument('--model', type=str, default='nn', choices=['nn', 'rf', 'xgb', 'svm'])
    parser.add_argument('--task_ids', type=str, default='31,3945,7593',
                       help='Comma-separated task IDs')
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    
    seeds = list(range(42, 42 + args.seeds))
    task_ids = [int(t) for t in args.task_ids.split(',')]
    
    print("=" * 70)
    print(f"CopulaHPO v2 - TabularBenchmark ({args.model.upper()})")
    print("=" * 70)
    
    all_results = {}
    for task_id in task_ids:
        try:
            all_results[task_id] = run_benchmark(args.model, task_id, args.budget, seeds)
        except Exception as e:
            print(f"  Error on task {task_id}: {e}")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for task_id, res in all_results.items():
        cop = np.mean(res['copula'])
        opt = np.mean(res['optuna'])
        rnd = np.mean(res['random'])
        best = min(cop, opt, rnd)
        winner = "CopulaHPO" if cop == best else ("Optuna" if opt == best else "Random")
        print(f"  Task {task_id}: Copula={cop:.4f}, Optuna={opt:.4f}, Random={rnd:.4f} -> {winner}")


if __name__ == '__main__':
    main()
