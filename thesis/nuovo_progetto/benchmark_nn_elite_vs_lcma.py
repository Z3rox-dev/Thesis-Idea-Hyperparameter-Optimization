#!/usr/bin/env python
"""Benchmark CopulaHPO Elite vs Latent-CMA su HPOBench TabularBenchmark (NN)."""
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
    
    def objective(cfg):
        res = bench.objective_function(cfg, fidelity=max_fid, metric='acc')
        return float(res['function_value'])
    
    results = {'elite': [], 'lcma': [], 'optuna': [], 'random': []}
    
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        
        # CopulaHPO Elite
        opt = CopulaHPO(specs, seed=seed, mode="elite", budget=budget)
        best_elite = float('inf')
        for i in range(budget):
            config = opt.ask()
            cfg = copula_config_to_cs(config, cs, hps)
            y = objective(cfg)
            opt.tell(config, y)
            best_elite = min(best_elite, y)
        results['elite'].append(best_elite)
        
        # CopulaHPO Latent-CMA
        opt2 = CopulaHPO(specs, seed=seed, mode="latent_cma", budget=budget)
        best_lcma = float('inf')
        for i in range(budget):
            config = opt2.ask()
            cfg = copula_config_to_cs(config, cs, hps)
            y = objective(cfg)
            opt2.tell(config, y)
            best_lcma = min(best_lcma, y)
        results['lcma'].append(best_lcma)
        
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
        best_random = float('inf')
        for _ in range(budget):
            cfg = cs.sample_configuration()
            y = objective(cfg)
            best_random = min(best_random, y)
        results['random'].append(best_random)
        
        print(f"Elite={best_elite:.4f}, L-CMA={best_lcma:.4f}, Opt={study.best_value:.4f}, Rand={best_random:.4f}")
    
    print(f"\n  Summary:")
    print(f"    Elite:      {np.mean(results['elite']):.4f} ± {np.std(results['elite']):.4f}")
    print(f"    Latent-CMA: {np.mean(results['lcma']):.4f} ± {np.std(results['lcma']):.4f}")
    print(f"    Optuna:     {np.mean(results['optuna']):.4f} ± {np.std(results['optuna']):.4f}")
    print(f"    Random:     {np.mean(results['random']):.4f} ± {np.std(results['random']):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nn', choices=['nn', 'rf', 'xgb', 'svm'])
    parser.add_argument('--task_ids', type=str, default='10101,146818,146821,146822,31,3917,53,9952')
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    
    seeds = list(range(42, 42 + args.seeds))
    task_ids = [int(t) for t in args.task_ids.split(',')]
    
    print("=" * 70)
    print(f"CopulaHPO Elite vs Latent-CMA - TabularBenchmark ({args.model.upper()})")
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
        elite = np.mean(res['elite'])
        lcma = np.mean(res['lcma'])
        opt = np.mean(res['optuna'])
        rnd = np.mean(res['random'])
        best = min(elite, lcma, opt, rnd)
        winner = "Elite" if elite == best else ("L-CMA" if lcma == best else ("Optuna" if opt == best else "Random"))
        print(f"  Task {task_id}: Elite={elite:.4f}, L-CMA={lcma:.4f}, Opt={opt:.4f}, Rand={rnd:.4f} -> {winner}")


if __name__ == '__main__':
    main()
