#!/usr/bin/env python3
"""
Benchmark CopulaHPO su JAHS-Bench-201 e HPOBench ParamNet.

Ambiente: py39 (per JAHS) o hpobench

Usage:
    # JAHS
    source /mnt/workspace/miniconda3/bin/activate py39
    python benchmark_copula_real_hpo.py --benchmark jahs --budget 200 --seeds 3
    
    # HPOBench ParamNet
    python benchmark_copula_real_hpo.py --benchmark paramnet --budget 200 --seeds 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

# Add paths
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from copula_hpo_v2 import CopulaHPO, HyperparameterSpec


# =============================================================================
# JAHS-Bench-201 Adapter
# =============================================================================

def get_jahs_specs() -> List[HyperparameterSpec]:
    """Return HyperparameterSpec list for JAHS-Bench-201.
    
    JAHS space:
    - 2 float (log-scale): LearningRate, WeightDecay
    - 3 ordinal: N, W, Resolution
    - 8 categorical: Activation, TrivialAugment, Op1-Op6
    
    We use integer type for ordinals (treated as ordered indices).
    """
    return [
        # Continuous (log-scale - stored as log values)
        HyperparameterSpec('LearningRate', 'continuous', (np.log(0.001), np.log(1.0))),
        HyperparameterSpec('WeightDecay', 'continuous', (np.log(1e-5), np.log(0.01))),
        # Ordinal as integer indices
        HyperparameterSpec('N', 'integer', (0, 2)),           # maps to [1, 3, 5]
        HyperparameterSpec('W', 'integer', (0, 2)),           # maps to [4, 8, 16]
        HyperparameterSpec('Resolution', 'integer', (0, 2)),  # maps to [0.25, 0.5, 1.0]
        # Categorical
        HyperparameterSpec('Activation', 'categorical', ['ReLU', 'Hardswish', 'Mish']),
        HyperparameterSpec('TrivialAugment', 'categorical', [True, False]),
        # Operations (NAS) - 5 choices each
        HyperparameterSpec('Op1', 'categorical', [0, 1, 2, 3, 4]),
        HyperparameterSpec('Op2', 'categorical', [0, 1, 2, 3, 4]),
        HyperparameterSpec('Op3', 'categorical', [0, 1, 2, 3, 4]),
        HyperparameterSpec('Op4', 'categorical', [0, 1, 2, 3, 4]),
        HyperparameterSpec('Op5', 'categorical', [0, 1, 2, 3, 4]),
        HyperparameterSpec('Op6', 'categorical', [0, 1, 2, 3, 4]),
    ]

# Mappings for ordinal HPs
N_CHOICES = [1, 3, 5]
W_CHOICES = [4, 8, 16]
RESOLUTION_CHOICES = [0.25, 0.5, 1.0]


def copula_config_to_jahs(config: Dict) -> Dict:
    """Convert CopulaHPO config to JAHS format."""
    jahs_config = {}
    # LearningRate and WeightDecay are log-scale
    jahs_config['LearningRate'] = np.exp(config['LearningRate'])
    jahs_config['WeightDecay'] = np.exp(config['WeightDecay'])
    # Ordinal mappings
    jahs_config['N'] = N_CHOICES[int(config['N'])]
    jahs_config['W'] = W_CHOICES[int(config['W'])]
    jahs_config['Resolution'] = RESOLUTION_CHOICES[int(config['Resolution'])]
    # Direct categorical
    for key in ['Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']:
        jahs_config[key] = config[key]
    # Fixed params
    jahs_config['Optimizer'] = 'SGD'
    jahs_config['epoch'] = 200
    return jahs_config


def run_jahs_benchmark(task: str, budget: int, seeds: List[int]) -> Dict:
    """Run CopulaHPO on JAHS-Bench-201."""
    from benchmark_jahs import JAHSBenchWrapper
    
    print(f"\n  JAHS-Bench-201 - {task}")
    print(f"  Budget: {budget}, Seeds: {seeds}")
    
    specs = get_jahs_specs()
    
    results = {
        'copula': [],
        'random': [],
    }
    
    for seed in seeds:
        print(f"    Seed {seed}...", end=" ", flush=True)
        wrapper = JAHSBenchWrapper(task=task)
        
        # CopulaHPO
        opt = CopulaHPO(specs, seed=seed)
        wrapper.reset()
        
        best_error = float('inf')
        for i in range(budget):
            config = opt.ask()
            jahs_config = copula_config_to_jahs(config)
            error = wrapper.evaluate(jahs_config)
            opt.tell(config, error)
            best_error = min(best_error, error)
        
        results['copula'].append(best_error)
        
        # Random baseline
        wrapper.reset()
        rng = np.random.default_rng(seed)
        best_random = float('inf')
        for i in range(budget):
            config = wrapper.sample_random(rng)
            error = wrapper.evaluate(config)
            best_random = min(best_random, error)
        
        results['random'].append(best_random)
        
        print(f"Copula: {best_error:.4f}, Random: {best_random:.4f}")
    
    return results


# =============================================================================
# HPOBench ParamNet Adapter
# =============================================================================

def get_paramnet_specs(dim: int) -> List[HyperparameterSpec]:
    """Return specs for ParamNet (all continuous/integer mapped to [0,1])."""
    # ParamNet is typically all continuous after normalization
    return [
        HyperparameterSpec(f'x{i}', 'continuous', (0.0, 1.0))
        for i in range(dim)
    ]


def run_paramnet_benchmark(dataset: str, budget: int, seeds: List[int]) -> Dict:
    """Run CopulaHPO on HPOBench ParamNet."""
    # Import HPOBench components
    try:
        from benchmark_hpobench_paramnet_curv import (
            build_paramnet_adapter,
            xnorm_to_config,
        )
    except ImportError as e:
        print(f"  Cannot import HPOBench: {e}")
        return {'error': str(e)}
    
    print(f"\n  HPOBench ParamNet - {dataset}")
    print(f"  Budget: {budget}, Seeds: {seeds}")
    
    # Build adapter
    try:
        bench, cs, hps, bounds, types = build_paramnet_adapter(dataset)
    except Exception as e:
        print(f"  Cannot build adapter: {e}")
        return {'error': str(e)}
    
    dim = len(hps)
    specs = get_paramnet_specs(dim)
    
    results = {
        'copula': [],
        'random': [],
    }
    
    def objective(x_norm: np.ndarray) -> float:
        cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
        res = bench.objective_function(cfg)
        return float(res['function_value'])
    
    for seed in seeds:
        print(f"    Seed {seed}...", end=" ", flush=True)
        
        # CopulaHPO (using continuous wrapper)
        from copula_hpo_v2 import CopulaHPO_Continuous
        opt = CopulaHPO_Continuous([(0.0, 1.0)] * dim, seed=seed)
        
        best_copula = float('inf')
        for i in range(budget):
            x = opt.ask()
            y = objective(x)
            opt.tell(x, y)
            best_copula = min(best_copula, y)
        
        results['copula'].append(best_copula)
        
        # Random
        rng = np.random.default_rng(seed)
        best_random = float('inf')
        for i in range(budget):
            x = rng.random(dim)
            y = objective(x)
            best_random = min(best_random, y)
        
        results['random'].append(best_random)
        
        print(f"Copula: {best_copula:.4f}, Random: {best_random:.4f}")
    
    return results


# =============================================================================
# Optuna Baseline
# =============================================================================

def run_jahs_optuna(task: str, budget: int, seeds: List[int]) -> List[float]:
    """Run Optuna TPE on JAHS for comparison."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except ImportError:
        return []
    
    from benchmark_jahs import JAHSBenchWrapper
    
    results = []
    wrapper = JAHSBenchWrapper(task=task)
    
    HP_SPACE = wrapper.HP_SPACE
    
    for seed in seeds:
        wrapper.reset()
        
        def objective(trial):
            config = {}
            for hp_name, hp_spec in HP_SPACE.items():
                if hp_spec['type'] == 'float':
                    if hp_spec.get('log', False):
                        config[hp_name] = trial.suggest_float(
                            hp_name, hp_spec['low'], hp_spec['high'], log=True
                        )
                    else:
                        config[hp_name] = trial.suggest_float(
                            hp_name, hp_spec['low'], hp_spec['high']
                        )
                elif hp_spec['type'] in ['ordinal', 'categorical']:
                    config[hp_name] = trial.suggest_categorical(
                        hp_name, hp_spec['choices']
                    )
            config['Optimizer'] = 'SGD'
            config['epoch'] = 200
            return wrapper.evaluate(config)
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        
        results.append(study.best_value)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='jahs',
                       choices=['jahs', 'paramnet', 'both'])
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--task', type=str, default='cifar10',
                       help='JAHS task or ParamNet dataset')
    args = parser.parse_args()
    
    seeds = list(range(42, 42 + args.seeds))
    
    print("=" * 70)
    print("CopulaHPO v2 - Real HPO Benchmark")
    print("=" * 70)
    print(f"Benchmark: {args.benchmark}")
    print(f"Budget: {args.budget}")
    print(f"Seeds: {seeds}")
    
    all_results = {}
    
    if args.benchmark in ['jahs', 'both']:
        tasks = ['cifar10', 'fashion_mnist', 'colorectal_histology']
        for task in tasks:
            print(f"\n{'=' * 70}")
            print(f"JAHS-Bench-201: {task}")
            print("=" * 70)
            
            results = run_jahs_benchmark(task, args.budget, seeds)
            
            # Also run Optuna for comparison
            print("    Running Optuna TPE...")
            optuna_results = run_jahs_optuna(task, args.budget, seeds)
            results['optuna'] = optuna_results
            
            all_results[f'jahs_{task}'] = results
            
            # Summary
            print(f"\n  Summary for {task}:")
            print(f"    CopulaHPO: {np.mean(results['copula']):.4f} ± {np.std(results['copula']):.4f}")
            print(f"    Random:    {np.mean(results['random']):.4f} ± {np.std(results['random']):.4f}")
            if optuna_results:
                print(f"    Optuna:    {np.mean(optuna_results):.4f} ± {np.std(optuna_results):.4f}")
    
    if args.benchmark in ['paramnet', 'both']:
        datasets = ['adult', 'higgs', 'letter', 'mnist', 'optdigits', 'poker']
        for dataset in datasets:
            print(f"\n{'=' * 70}")
            print(f"HPOBench ParamNet: {dataset}")
            print("=" * 70)
            
            results = run_paramnet_benchmark(dataset, args.budget, seeds)
            all_results[f'paramnet_{dataset}'] = results
            
            if 'error' not in results:
                print(f"\n  Summary for {dataset}:")
                print(f"    CopulaHPO: {np.mean(results['copula']):.4f} ± {np.std(results['copula']):.4f}")
                print(f"    Random:    {np.mean(results['random']):.4f} ± {np.std(results['random']):.4f}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
