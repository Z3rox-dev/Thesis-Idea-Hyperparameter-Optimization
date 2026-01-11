#!/usr/bin/env python3
"""
Test ALBA with LLR Gradient vs ALBA baseline vs Random vs Optuna
================================================================

Confronto su:
1. Funzioni sintetiche (Sphere, Rastrigin, Ackley, Levy, Rosenbrock)
2. JAHS-Bench-201 (con conda py39)
3. ParamNet (con paramnetvenv)
4. NN Tabular (con py39 o yahpo)

Usage:
    # Funzioni sintetiche
    python thesis/test_alba_llr_gradient.py --mode synthetic
    
    # JAHS (richiede conda py39)
    source /mnt/workspace/miniconda3/bin/activate py39
    python thesis/test_alba_llr_gradient.py --mode jahs
    
    # ParamNet
    source /mnt/workspace/miniconda3/bin/activate paramnetvenv
    python thesis/test_alba_llr_gradient.py --mode paramnet
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Add thesis to path
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_gravity import ALBA


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function (minimum at 0)."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (many local minima)."""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function."""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)


def levy(x: np.ndarray) -> float:
    """Levy function."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function."""
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


SYNTHETIC_FUNCTIONS = {
    "sphere": (sphere, [(-5.12, 5.12)] * 10),
    "rastrigin": (rastrigin, [(-5.12, 5.12)] * 10),
    "ackley": (ackley, [(-5, 5)] * 10),
    "levy": (levy, [(-10, 10)] * 10),
    "rosenbrock": (rosenbrock, [(-5, 10)] * 10),
}


# =============================================================================
# OPTIMIZER WRAPPERS
# =============================================================================

def run_alba(bounds: List[Tuple[float, float]], objective: Callable, budget: int, 
             seed: int, llr_gradient: bool = False) -> Tuple[float, List[float]]:
    """Run ALBA optimization."""
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        llr_gradient=llr_gradient,
        llr_gradient_weight=0.7 if llr_gradient else 0.0,
    )
    
    trace = []
    best_so_far = float('inf')
    
    for _ in range(budget):
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
        best_so_far = min(best_so_far, y)
        trace.append(best_so_far)
    
    return opt.best_y, trace


def run_random(bounds: List[Tuple[float, float]], objective: Callable, 
               budget: int, seed: int) -> Tuple[float, List[float]]:
    """Run Random Search."""
    rng = np.random.default_rng(seed)
    
    trace = []
    best_so_far = float('inf')
    best_x = None
    
    for _ in range(budget):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        y = objective(x)
        if y < best_so_far:
            best_so_far = y
            best_x = x
        trace.append(best_so_far)
    
    return best_so_far, trace


def run_optuna(bounds: List[Tuple[float, float]], objective: Callable, 
               budget: int, seed: int) -> Tuple[float, List[float]]:
    """Run Optuna TPE."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    trace = []
    best_so_far = float('inf')
    
    def optuna_objective(trial):
        nonlocal best_so_far
        x = np.array([
            trial.suggest_float(f"x{i}", lo, hi)
            for i, (lo, hi) in enumerate(bounds)
        ])
        y = objective(x)
        best_so_far = min(best_so_far, y)
        trace.append(best_so_far)
        return y
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value, trace


# =============================================================================
# SYNTHETIC BENCHMARK
# =============================================================================

def run_synthetic_benchmark(budget: int = 200, seeds: List[int] = [42, 43, 44]):
    """Run synthetic function benchmark."""
    print("=" * 70)
    print("SYNTHETIC FUNCTIONS BENCHMARK")
    print("=" * 70)
    print(f"Budget: {budget}, Seeds: {seeds}")
    print()
    
    results = {}
    
    for func_name, (func, bounds) in SYNTHETIC_FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"Function: {func_name.upper()} (dim={len(bounds)})")
        print('='*60)
        
        alba_results = []
        alba_llr_results = []
        random_results = []
        optuna_results = []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # ALBA baseline
            y_alba, _ = run_alba(bounds, func, budget, seed, llr_gradient=False)
            alba_results.append(y_alba)
            
            # ALBA + LLR
            y_alba_llr, _ = run_alba(bounds, func, budget, seed, llr_gradient=True)
            alba_llr_results.append(y_alba_llr)
            
            # Random
            y_random, _ = run_random(bounds, func, budget, seed)
            random_results.append(y_random)
            
            # Optuna
            y_optuna, _ = run_optuna(bounds, func, budget, seed)
            optuna_results.append(y_optuna)
            
            print(f"ALBA={y_alba:.4f}, ALBA+LLR={y_alba_llr:.4f}, Random={y_random:.4f}, Optuna={y_optuna:.4f}")
        
        results[func_name] = {
            "alba": alba_results,
            "alba_llr": alba_llr_results,
            "random": random_results,
            "optuna": optuna_results,
        }
        
        # Summary
        print(f"\n  Summary:")
        print(f"    ALBA:       mean={np.mean(alba_results):.4f} ± {np.std(alba_results):.4f}")
        print(f"    ALBA+LLR:   mean={np.mean(alba_llr_results):.4f} ± {np.std(alba_llr_results):.4f}")
        print(f"    Random:     mean={np.mean(random_results):.4f} ± {np.std(random_results):.4f}")
        print(f"    Optuna:     mean={np.mean(optuna_results):.4f} ± {np.std(optuna_results):.4f}")
        
        # Improvement
        alba_mean = np.mean(alba_results)
        llr_mean = np.mean(alba_llr_results)
        improvement = (alba_mean - llr_mean) / (alba_mean + 1e-10) * 100
        print(f"\n    LLR vs ALBA improvement: {improvement:+.1f}%")
    
    return results


# =============================================================================
# JAHS BENCHMARK
# =============================================================================

def run_jahs_benchmark(budget: int = 200, seeds: List[int] = [70, 71, 72], 
                       tasks: List[str] = ["cifar10", "fashion_mnist"]):
    """Run JAHS-Bench-201 benchmark."""
    print("=" * 70)
    print("JAHS-BENCH-201 BENCHMARK")
    print("=" * 70)
    
    try:
        from jahs_bench import Benchmark
    except ImportError:
        print("ERROR: jahs_bench not available. Run with conda py39.")
        return None
    
    results = {}
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)
        
        # Initialize benchmark
        bench = Benchmark(
            task=task,
            kind='surrogate',
            download=True,
            save_dir='/mnt/workspace/jahs_bench_data',
            metrics=['valid-acc']
        )
        
        # JAHS space definition (simplified to [0,1]^13)
        dim = 13
        bounds = [(0.0, 1.0)] * dim
        
        def jahs_objective(x_norm):
            """Convert normalized x to JAHS config and evaluate."""
            # Map normalized to actual values
            config = {
                'LearningRate': 10 ** (np.log10(0.001) + x_norm[0] * (np.log10(1.0) - np.log10(0.001))),
                'WeightDecay': 10 ** (np.log10(1e-5) + x_norm[1] * (np.log10(0.01) - np.log10(1e-5))),
                'N': [1, 3, 5][int(x_norm[2] * 2.99)],
                'W': [4, 8, 16][int(x_norm[3] * 2.99)],
                'Resolution': [0.25, 0.5, 1.0][int(x_norm[4] * 2.99)],
                'Activation': ['ReLU', 'Hardswish', 'Mish'][int(x_norm[5] * 2.99)],
                'TrivialAugment': [True, False][int(x_norm[6] * 1.99)],
                'Op1': int(x_norm[7] * 4.99),
                'Op2': int(x_norm[8] * 4.99),
                'Op3': int(x_norm[9] * 4.99),
                'Op4': int(x_norm[10] * 4.99),
                'Op5': int(x_norm[11] * 4.99),
                'Op6': int(x_norm[12] * 4.99),
                # Required fixed params
                'Optimizer': 'SGD',
                'epoch': 200,
            }
            
            result = bench(config)
            # Result is {epoch: {metrics}}
            last_epoch = max(result.keys())
            valid_acc = result[last_epoch]['valid-acc']
            return 1.0 - valid_acc / 100.0  # Return error (minimize)
        
        alba_results = []
        alba_llr_results = []
        random_results = []
        optuna_results = []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # ALBA baseline
            y_alba, _ = run_alba(bounds, jahs_objective, budget, seed, llr_gradient=False)
            alba_results.append(1.0 - y_alba)  # Convert error to accuracy
            
            # ALBA + LLR
            y_alba_llr, _ = run_alba(bounds, jahs_objective, budget, seed, llr_gradient=True)
            alba_llr_results.append(1.0 - y_alba_llr)
            
            # Random
            y_random, _ = run_random(bounds, jahs_objective, budget, seed)
            random_results.append(1.0 - y_random)
            
            # Optuna
            y_optuna, _ = run_optuna(bounds, jahs_objective, budget, seed)
            optuna_results.append(1.0 - y_optuna)
            
            print(f"ALBA={1-y_alba:.4f}, ALBA+LLR={1-y_alba_llr:.4f}, Random={1-y_random:.4f}, Optuna={1-y_optuna:.4f}")
        
        results[task] = {
            "alba": alba_results,
            "alba_llr": alba_llr_results,
            "random": random_results,
            "optuna": optuna_results,
        }
        
        # Summary
        print(f"\n  Summary (validation accuracy):")
        print(f"    ALBA:       mean={np.mean(alba_results):.4f} ± {np.std(alba_results):.4f}")
        print(f"    ALBA+LLR:   mean={np.mean(alba_llr_results):.4f} ± {np.std(alba_llr_results):.4f}")
        print(f"    Random:     mean={np.mean(random_results):.4f} ± {np.std(random_results):.4f}")
        print(f"    Optuna:     mean={np.mean(optuna_results):.4f} ± {np.std(optuna_results):.4f}")
        
        # Winner
        means = {
            'ALBA': np.mean(alba_results),
            'ALBA+LLR': np.mean(alba_llr_results),
            'Random': np.mean(random_results),
            'Optuna': np.mean(optuna_results),
        }
        winner = max(means, key=means.get)
        print(f"    Winner: {winner}")
    
    return results


# =============================================================================
# PARAMNET BENCHMARK
# =============================================================================

def run_paramnet_benchmark(budget: int = 200, seeds: List[int] = [70, 71, 72],
                          datasets: List[str] = ["adult", "higgs"]):
    """Run ParamNet benchmark."""
    print("=" * 70)
    print("PARAMNET BENCHMARK")
    print("=" * 70)
    
    sys.path.insert(0, "/mnt/workspace/HPOBench")
    
    try:
        from hpobench.benchmarks.surrogates.paramnet_benchmark import (
            ParamNetAdultOnStepsBenchmark,
            ParamNetHiggsOnStepsBenchmark,
            ParamNetLetterOnStepsBenchmark,
            ParamNetMnistOnStepsBenchmark,
        )
    except ImportError as e:
        print(f"ERROR: HPOBench ParamNet not available: {e}")
        return None
    
    BENCH_MAP = {
        "adult": ParamNetAdultOnStepsBenchmark,
        "higgs": ParamNetHiggsOnStepsBenchmark,
        "letter": ParamNetLetterOnStepsBenchmark,
        "mnist": ParamNetMnistOnStepsBenchmark,
    }
    
    results = {}
    
    for dataset in datasets:
        if dataset not in BENCH_MAP:
            print(f"Skipping unknown dataset: {dataset}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        bench_cls = BENCH_MAP[dataset]
        bench = bench_cls()
        cs = bench.get_configuration_space()
        hps = cs.get_hyperparameters()
        
        # Build bounds from ConfigSpace
        bounds = []
        types = []
        for hp in hps:
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                bounds.append((float(hp.lower), float(hp.upper)))
                types.append('int' if 'Integer' in str(type(hp)) else 'float')
            else:
                print(f"Warning: Unsupported HP type: {hp}")
                continue
        
        dim = len(bounds)
        print(f"  Dimension: {dim}")
        
        def paramnet_objective(x_norm):
            """Evaluate ParamNet with normalized input."""
            values = {}
            for i, (hp, (lo, hi), t) in enumerate(zip(hps, bounds, types)):
                v = lo + x_norm[i] * (hi - lo)
                if t == 'int':
                    v = int(round(v))
                values[hp.name] = v
            
            config = cs.sample_configuration()
            for k, v in values.items():
                config[k] = v
            
            result = bench.objective_function(config)
            return result['function_value']
        
        # Normalize bounds to [0,1]
        norm_bounds = [(0.0, 1.0)] * dim
        
        alba_results = []
        alba_llr_results = []
        random_results = []
        optuna_results = []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # ALBA baseline
            y_alba, _ = run_alba(norm_bounds, paramnet_objective, budget, seed, llr_gradient=False)
            alba_results.append(y_alba)
            
            # ALBA + LLR
            y_alba_llr, _ = run_alba(norm_bounds, paramnet_objective, budget, seed, llr_gradient=True)
            alba_llr_results.append(y_alba_llr)
            
            # Random
            y_random, _ = run_random(norm_bounds, paramnet_objective, budget, seed)
            random_results.append(y_random)
            
            # Optuna
            y_optuna, _ = run_optuna(norm_bounds, paramnet_objective, budget, seed)
            optuna_results.append(y_optuna)
            
            print(f"ALBA={y_alba:.4f}, LLR={y_alba_llr:.4f}, Rnd={y_random:.4f}, Opt={y_optuna:.4f}")
        
        results[dataset] = {
            "alba": alba_results,
            "alba_llr": alba_llr_results,
            "random": random_results,
            "optuna": optuna_results,
        }
        
        # Summary
        print(f"\n  Summary:")
        print(f"    ALBA:       mean={np.mean(alba_results):.6f}")
        print(f"    ALBA+LLR:   mean={np.mean(alba_llr_results):.6f}")
        print(f"    Random:     mean={np.mean(random_results):.6f}")
        print(f"    Optuna:     mean={np.mean(optuna_results):.6f}")
        
        # Winner
        means = {
            'ALBA': np.mean(alba_results),
            'ALBA+LLR': np.mean(alba_llr_results),
            'Random': np.mean(random_results),
            'Optuna': np.mean(optuna_results),
        }
        winner = min(means, key=means.get)
        print(f"    Winner (lowest): {winner}")
    
    return results


# =============================================================================
# NN TABULAR BENCHMARK (HPOBench)
# =============================================================================

def run_nn_tabular_benchmark(budget: int = 200, seeds: List[int] = [70, 71, 72],
                              task_ids: List[int] = [31, 53, 3917, 9952]):
    """Run HPOBench NN Tabular benchmark."""
    print("=" * 70)
    print("NN TABULAR BENCHMARK (HPOBench)")
    print("=" * 70)
    
    sys.path.insert(0, "/mnt/workspace/HPOBench")
    
    try:
        from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    except ImportError as e:
        print(f"ERROR: HPOBench TabularBenchmark not available: {e}")
        return None
    
    results = {}
    
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"Task ID: {task_id}")
        print('='*60)
        
        try:
            bench = TabularBenchmark(model="nn", task_id=task_id)
        except Exception as e:
            print(f"  ERROR loading task {task_id}: {e}")
            continue
        
        cs = bench.get_configuration_space()
        hps = list(cs.values())
        hp_names = [hp.name for hp in hps]
        hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
        max_fidelity = bench.get_max_fidelity()
        
        dim = len(hp_names)
        print(f"  Dimension: {dim}")
        
        def nn_objective(x_norm):
            """Evaluate NN Tabular with normalized input."""
            cfg = {}
            for i, name in enumerate(hp_names):
                seq = hp_seqs[name]
                k = len(seq)
                if k <= 1:
                    cfg[name] = seq[0]
                else:
                    idx = int(np.floor(float(x_norm[i]) * k))
                    idx = max(0, min(k-1, idx))
                    cfg[name] = seq[idx]
            
            res = bench.objective_function(configuration=cfg, fidelity=max_fidelity)
            return float(res["function_value"])
        
        # Normalize bounds to [0,1]
        norm_bounds = [(0.0, 1.0)] * dim
        
        alba_results = []
        alba_llr_results = []
        random_results = []
        optuna_results = []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # ALBA baseline
            y_alba, _ = run_alba(norm_bounds, nn_objective, budget, seed, llr_gradient=False)
            alba_results.append(y_alba)
            
            # ALBA + LLR
            y_alba_llr, _ = run_alba(norm_bounds, nn_objective, budget, seed, llr_gradient=True)
            alba_llr_results.append(y_alba_llr)
            
            # Random
            y_random, _ = run_random(norm_bounds, nn_objective, budget, seed)
            random_results.append(y_random)
            
            # Optuna
            y_optuna, _ = run_optuna(norm_bounds, nn_objective, budget, seed)
            optuna_results.append(y_optuna)
            
            print(f"ALBA={y_alba:.4f}, LLR={y_alba_llr:.4f}, Rnd={y_random:.4f}, Opt={y_optuna:.4f}")
        
        results[str(task_id)] = {
            "alba": alba_results,
            "alba_llr": alba_llr_results,
            "random": random_results,
            "optuna": optuna_results,
        }
        
        # Summary
        print(f"\n  Summary (error - lower is better):")
        print(f"    ALBA:       mean={np.mean(alba_results):.6f}")
        print(f"    ALBA+LLR:   mean={np.mean(alba_llr_results):.6f}")
        print(f"    Random:     mean={np.mean(random_results):.6f}")
        print(f"    Optuna:     mean={np.mean(optuna_results):.6f}")
        
        # Winner
        means = {
            'ALBA': np.mean(alba_results),
            'ALBA+LLR': np.mean(alba_llr_results),
            'Random': np.mean(random_results),
            'Optuna': np.mean(optuna_results),
        }
        winner = min(means, key=means.get)
        print(f"    Winner (lowest error): {winner}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test ALBA with LLR Gradient')
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'jahs', 'paramnet', 'nn_tabular', 'all'],
                       help='Benchmark mode')
    parser.add_argument('--budget', type=int, default=200, help='Evaluation budget')
    parser.add_argument('--seeds', type=str, default='70,71,72', help='Random seeds')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    all_results = {}
    
    if args.mode in ['synthetic', 'all']:
        all_results['synthetic'] = run_synthetic_benchmark(args.budget, seeds)
    
    if args.mode in ['jahs', 'all']:
        all_results['jahs'] = run_jahs_benchmark(args.budget, seeds)
    
    if args.mode in ['paramnet', 'all']:
        all_results['paramnet'] = run_paramnet_benchmark(args.budget, seeds)
    
    if args.mode in ['nn_tabular', 'all']:
        all_results['nn_tabular'] = run_nn_tabular_benchmark(args.budget, seeds)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/mnt/workspace/thesis/benchmark_results/alba_llr_test_{args.mode}_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if 'synthetic' in all_results and all_results['synthetic']:
        print("\nSYNTHETIC FUNCTIONS:")
        llr_wins = 0
        for func_name, data in all_results['synthetic'].items():
            alba_mean = np.mean(data['alba'])
            llr_mean = np.mean(data['alba_llr'])
            winner = "LLR" if llr_mean < alba_mean else "ALBA"
            if llr_mean < alba_mean:
                llr_wins += 1
            imp = (alba_mean - llr_mean) / (alba_mean + 1e-10) * 100
            print(f"  {func_name:15s}: {winner:5s} ({imp:+.1f}%)")
        print(f"  LLR win rate: {llr_wins}/{len(all_results['synthetic'])}")


if __name__ == "__main__":
    main()
