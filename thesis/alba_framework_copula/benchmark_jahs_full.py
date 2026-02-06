#!/usr/bin/env python3
"""
Full JAHS-Bench-201 Benchmark
=============================
Datasets: cifar10, colorectal_histology, fashion_mnist
Optimizers: Random, Optuna (TPE), CMA (Nevergrad), ALBA (Std, Cov, Hybrid)
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np
from typing import Dict, List, Tuple

# Path Setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/workspace/thesis')
# Add py39 site-packages for heavy libs
sys.path.append('/mnt/workspace/miniconda3/envs/py39/lib/python3.9/site-packages')

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# IMPORTS (Lazy / Robust)
# -----------------------------------------------------------------------------
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
except ImportError:
    optuna = None

try:
    import nevergrad as ng
except ImportError:
    ng = None

try:
    from jahs_bench import Benchmark
    HAS_JAHS = True
except ImportError:
    HAS_JAHS = False
    print("WARNING: jahs_bench not found. Using Mock.")

# ALBA Imports
try:
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.local_search import CovarianceLocalSearchSampler
except ImportError:
    sys.path.insert(0, '/mnt/workspace/thesis/alba_framework_potential')
    from optimizer import ALBA
    from local_search import CovarianceLocalSearchSampler

# -----------------------------------------------------------------------------
# WRAPPER
# -----------------------------------------------------------------------------
class JAHSBenchWrapper:
    TASKS = ['cifar10', 'colorectal_histology', 'fashion_mnist']
    
    HP_SPACE = {
        'LearningRate': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
        'WeightDecay': {'type': 'float', 'low': 1e-5, 'high': 0.01, 'log': True},
        'N': {'type': 'ordinal', 'choices': [1, 3, 5]},
        'W': {'type': 'ordinal', 'choices': [4, 8, 16]},
        'Resolution': {'type': 'ordinal', 'choices': [0.25, 0.5, 1.0]},
        'Activation': {'type': 'categorical', 'choices': ['ReLU', 'Hardswish', 'Mish']},
        'TrivialAugment': {'type': 'categorical', 'choices': [True, False]},
        'Op1': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op2': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op3': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op4': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op5': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op6': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
    }
    
    HP_ORDER = ['LearningRate', 'WeightDecay', 'N', 'W', 'Resolution', 
                'Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']

    def __init__(self, task='cifar10'):
        self.task = task
        self.bench = None
        if HAS_JAHS:
            try:
                self.bench = Benchmark(task=task, kind='surrogate', download=True, save_dir='/mnt/workspace/jahs_bench_data', metrics=['valid-acc'])
            except Exception as e:
                print(f"Error loading JAHS task {task}: {e}")

    @property
    def dim(self):
        return len(self.HP_ORDER)

    def _array_to_dict(self, arr):
        config = {}
        for i, hp_name in enumerate(self.HP_ORDER):
            hp_spec = self.HP_SPACE[hp_name]
            val = np.clip(arr[i], 0, 1)
            
            if hp_spec['type'] == 'float':
                if hp_spec.get('log', False):
                    low_log = np.log(hp_spec['low'])
                    high_log = np.log(hp_spec['high'])
                    config[hp_name] = np.exp(low_log + val * (high_log - low_log))
                else:
                    config[hp_name] = hp_spec['low'] + val * (hp_spec['high'] - hp_spec['low'])
            elif hp_spec['type'] in ['ordinal', 'categorical']:
                choices = hp_spec['choices']
                idx = int(round(val * (len(choices) - 1)))
                idx = np.clip(idx, 0, len(choices) - 1)
                config[hp_name] = choices[idx]
        
        config['Optimizer'] = 'SGD'
        config['epoch'] = 200
        return config

    def evaluate(self, x):
        # x is [0,1]^d array
        config = self._array_to_dict(x)
        if self.bench:
            res = self.bench(config)
            last_epoch = max(res.keys())
            acc = res[last_epoch]['valid-acc']
            return 100.0 - acc
        else:
            # Mock
            return np.sum((x - 0.5)**2) * 100

# -----------------------------------------------------------------------------
# OPTIMIZERS
# -----------------------------------------------------------------------------

def run_random(wrapper, n_evals, seed):
    rng = np.random.default_rng(seed)
    best = float('inf')
    for _ in range(n_evals):
        x = rng.random(wrapper.dim)
        y = wrapper.evaluate(x)
        best = min(best, y)
    return best

def run_optuna(wrapper, n_evals, seed):
    if optuna is None: return float('inf')
    
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(wrapper.dim)])
        return wrapper.evaluate(x)
        
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_trials=n_evals)
    return study.best_value

try:
    import cma
except ImportError:
    cma = None

def run_cma_direct(wrapper, n_evals, seed):
    """Run CMA-ES using the official 'cma' library."""
    if cma is None: return float('inf')
    
    # CMS-ES minimizes by default
    # Initial point: center of [0,1]^d
    x0 = [0.5] * wrapper.dim
    sigma0 = 0.2  # Covers [0,1] reasonably well (0.5 +/- 2*0.2)
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': [0, 1], 
        'seed': seed, 
        'maxfevals': n_evals,
        'verbose': -9
    })
    
    best = float('inf')
    
    while not es.stop():
        X = es.ask()
        Y = [wrapper.evaluate(x) for x in X]
        es.tell(X, Y)
        best = min(best, min(Y))
        
        if es.result.evaluations >= n_evals:
            break
            
    return best

def run_cma_ng(wrapper, n_evals, seed):
    """Run CMA-ES via Nevergrad (Fallback)."""
    if ng is None: return float('inf')
    
    # Parametrization
    param = ng.p.Array(shape=(wrapper.dim,)).set_bounds(lower=0.0, upper=1.0)
    optimizer = ng.optimizers.CMA(parametrization=param, budget=n_evals, num_workers=1)
    # Seed? Nevergrad seeding is tricky, usually via parametrization or random state
    # We'll rely on numpy seed if NG uses it, but NG has internal seeding.
    
    best = float('inf')
    for _ in range(n_evals):
        x_ng = optimizer.ask()
        x = x_ng.value
        y = wrapper.evaluate(x)
        optimizer.tell(x_ng, y)
        best = min(best, y)
    return best

def run_alba(wrapper, n_evals, seed, mode='std'):
    use_drilling = (mode == 'hybrid')
    use_cov = (mode != 'std')
    
    sampler = None
    if use_cov:
        sampler = CovarianceLocalSearchSampler(
            radius_start=0.15, radius_end=0.01, top_k_fraction=0.15, min_points_fit=10
        )
        
    opt = ALBA(
        bounds=[(0.0, 1.0)]*wrapper.dim,
        total_budget=n_evals,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        use_drilling=use_drilling,
        seed=seed
    )
    
    best = float('inf')
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate(x)
        opt.tell(x, y)
        best = min(best, y)
    return best

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evals', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    tasks = ['cifar10', 'colorectal_histology', 'fashion_mnist']
    optimizers = {
        'Random': run_random,
        'Optuna': run_optuna,
        'CMA_Direct': run_cma_direct,  # Switched to official 'cma' library
        'ALBA_Std': lambda w, n, s: run_alba(w, n, s, 'std'),
        'ALBA_Cov': lambda w, n, s: run_alba(w, n, s, 'cov'),
        'ALBA_Hybrid': lambda w, n, s: run_alba(w, n, s, 'hybrid'),
    }

    print(f"FULL JAHS BENCHMARK (Evals={args.evals}, Seeds={args.seeds})")
    
    for task in tasks:
        print(f"\n=== TASK: {task} ===")
        wrapper = JAHSBenchWrapper(task)
        if not wrapper.bench and HAS_JAHS:
            print(f"Skipping {task} (Load failed)")
            continue
            
        results = {}
        for name, func in optimizers.items():
            print(f"  Running {name}...", end='', flush=True)
            vals = []
            t0 = time.time()
            for s in range(args.seeds):
                v = func(wrapper, args.evals, s)
                vals.append(v)
            dt = time.time() - t0
            
            mean = np.mean(vals)
            std = np.std(vals)
            results[name] = (mean, std)
            print(f" {mean:.4f} +/- {std:.4f} ({dt:.1f}s)")
            
        # Summary for Task
        print(f"  --- Winner: {min(results, key=lambda k: results[k][0])} ---")

if __name__ == "__main__":
    main()
