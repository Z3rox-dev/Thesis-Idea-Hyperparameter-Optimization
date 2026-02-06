#!/usr/bin/env python3
"""
Benchmark JAHS-Bench-201 with ALBA Potential + Adaptive Drilling
================================================================
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import New Framework
try:
    # Try package import first (when running as module)
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.local_search import CovarianceLocalSearchSampler
except ImportError:
    # Fallback for local run
    try:
        from optimizer import ALBA
        from local_search import CovarianceLocalSearchSampler
    except ImportError:
        # Last resort path hacking
        sys.path.insert(0, '/mnt/workspace/thesis/alba_framework_potential')
        from optimizer import ALBA
        from local_search import CovarianceLocalSearchSampler

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# JAHS-BENCH-201 INTERFACE (Copied from benchmark_jahs.py)
# =============================================================================

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
    
    def __init__(self, task: str = 'cifar10', save_dir: str = '/mnt/workspace/jahs_bench_data'):
        # Add path to JAHS Bench if needed
        sys.path.append('/mnt/workspace/miniconda3/envs/py39/lib/python3.9/site-packages')
        
        try:
            from jahs_bench import Benchmark
            self.bench = Benchmark(task=task, kind='surrogate', download=True, save_dir=save_dir, metrics=['valid-acc'])
        except ImportError:
            print("JAHS-Bench not found (even after path add). Creating Mock Wrapper.")
            self.bench = None
            
        self.task = task
        self.n_evals = 0
    
    @property
    def dim(self) -> int:
        return len(self.HP_ORDER)
    
    def evaluate(self, config: Dict) -> float:
        if self.bench:
            result = self.bench(config)
            last_epoch = max(result.keys())
            valid_acc = result[last_epoch]['valid-acc']
            return 100.0 - valid_acc # Minimization
        else:
            # Mock Landscape (Sphere-like)
            val = 0.0
            for k, v in config.items():
                if isinstance(v, (int, float)):
                    val += (v - 0.5)**2
            return val

    def _array_to_dict(self, arr: np.ndarray) -> Dict:
        """Converts normalized array to config dict."""
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

    def evaluate_array(self, x: np.ndarray) -> float:
        config = self._array_to_dict(x)
        return self.evaluate(config)
        
    def reset(self):
        self.n_evals = 0

# =============================================================================
# OPTIMIZERS
# =============================================================================

def run_alba_std(wrapper, n_evals, seed):
    """Standard ALBA (No Drilling)."""
    wrapper.reset()
    dim = wrapper.dim
    
    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        total_budget=n_evals,
        local_search_ratio=0.3,
        use_drilling=False,
        seed=seed
    )
    
    best_err = float('inf')
    history = []
    
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_err = min(best_err, y)
        history.append(best_err)
        
    return history, best_err

def run_alba_cov(wrapper, n_evals, seed):
    """ALBA Weighted Covariance."""
    wrapper.reset()
    dim = wrapper.dim
    
    cov_sampler = CovarianceLocalSearchSampler(
        radius_start=0.15,
        radius_end=0.01,
        top_k_fraction=0.15,
        min_points_fit=10
    )
    
    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        total_budget=n_evals,
        local_search_ratio=0.3,
        local_search_sampler=cov_sampler,
        use_drilling=False,
        seed=seed
    )
    
    best_err = float('inf')
    history = []
    
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_err = min(best_err, y)
        history.append(best_err)
        
    return history, best_err

def run_alba_hybrid(wrapper, n_evals, seed):
    """ALBA Hybrid Adaptive (The Quadra)."""
    wrapper.reset()
    dim = wrapper.dim
    
    cov_sampler = CovarianceLocalSearchSampler(
        radius_start=0.15,
        radius_end=0.01,
        top_k_fraction=0.15, 
        min_points_fit=10
    )
    
    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        total_budget=n_evals,
        local_search_ratio=0.3,
        local_search_sampler=cov_sampler,
        use_drilling=True, # Enable Adaptive Drilling
        seed=seed
    )
    
    best_err = float('inf')
    history = []
    
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_err = min(best_err, y)
        history.append(best_err)
        
    return history, best_err

# =============================================================================
# RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evals', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    
    wrapper = JAHSBenchWrapper(task='cifar10')
    if wrapper.bench is None:
        print("WARNING: Running in MOCK mode (JAHS not found).")
    
    optimizers = {
        "ALBA_Std": run_alba_std,
        "ALBA_Cov": run_alba_cov,
        "ALBA_Hybrid": run_alba_hybrid
    }
    
    print(f"Running JAHS Benchmark (Eval={args.evals}, Seeds={args.seeds})")
    print("-" * 60)
    
    results = {}
    
    for name, func in optimizers.items():
        print(f"Running {name}...")
        errors = []
        times = []
        
        for s in range(args.seeds):
            t0 = time.time()
            _, err = func(wrapper, args.evals, s)
            defaults_t = time.time() - t0
            errors.append(err)
            times.append(defaults_t)
            print(f"  Seed {s}: {err:.4f} ({defaults_t:.2f}s)")
            
        mean = np.mean(errors)
        std = np.std(errors)
        results[name] = mean
        print(f"  Result: {mean:.4f} +/- {std:.4f}")
        print("-" * 60)
        
    print("\nFINAL SUMMARY")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
