#!/usr/bin/env python3
"""
Benchmark JAHS-Bench-201: Random Search vs Optuna TPE vs TuRBO-M vs ALBA
================================================================================

Questo benchmark confronta 4 ottimizzatori su JAHS-Bench-201 (surrogate mode):
1. Random Search
2. Optuna TPE (gestisce nativamente i categorici)
3. TuRBO-M (implementazione ufficiale Uber con multi trust regions)
4. ALBA (il nostro algoritmo custom)

NOTA SULLE VARIABILI CATEGORICHE:
- Lo spazio JAHS ha 13 dimensioni: 2 float (log), 3 ordinal, 8 categorical
- Per Random/TuRBO/ALBA normalizziamo tutto a [0,1]^13 e arrotondiamo
- Questo crea funzioni a gradini (non lisce) per le dimensioni discrete
- TPE invece gestisce nativamente i categorici -> potenziale vantaggio
- GP/LGS vedono variabili "continue" che in realtà sono discrete

Usage:
    python benchmark_jahs.py --task cifar10 --n_evals 200 --n_seeds 10
"""

import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# JAHS-BENCH-201 INTERFACE
# =============================================================================

class JAHSBenchWrapper:
    """Wrapper per JAHS-Bench-201 che standardizza l'interfaccia."""
    
    TASKS = ['cifar10', 'colorectal_histology', 'fashion_mnist']
    
    # Definizione dello spazio di ricerca
    HP_SPACE = {
        # Continuous (log-scale)
        'LearningRate': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
        'WeightDecay': {'type': 'float', 'low': 1e-5, 'high': 0.01, 'log': True},
        # Ordinal (trattati come categorici per semplicità)
        'N': {'type': 'ordinal', 'choices': [1, 3, 5]},
        'W': {'type': 'ordinal', 'choices': [4, 8, 16]},
        'Resolution': {'type': 'ordinal', 'choices': [0.25, 0.5, 1.0]},
        # Categorical
        'Activation': {'type': 'categorical', 'choices': ['ReLU', 'Hardswish', 'Mish']},
        'TrivialAugment': {'type': 'categorical', 'choices': [True, False]},
        # Operations (NAS)
        'Op1': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op2': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op3': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op4': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op5': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
        'Op6': {'type': 'categorical', 'choices': [0, 1, 2, 3, 4]},
    }
    
    # Ordine dei parametri per conversione array
    HP_ORDER = ['LearningRate', 'WeightDecay', 'N', 'W', 'Resolution', 
                'Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']
    
    def __init__(self, task: str = 'cifar10', save_dir: str = '/mnt/workspace/jahs_bench_data'):
        from jahs_bench import Benchmark
        
        if task not in self.TASKS:
            raise ValueError(f"Task must be one of {self.TASKS}")
        
        self.task = task
        self.bench = Benchmark(
            task=task, 
            kind='surrogate', 
            download=True, 
            save_dir=save_dir,
            metrics=['valid-acc']  # Solo validation accuracy per efficienza
        )
        self.n_evals = 0
    
    @property
    def dim(self) -> int:
        return len(self.HP_ORDER)
    
    def _dict_to_array(self, config: Dict) -> np.ndarray:
        """Converte un dizionario config in array normalizzato [0,1]^d."""
        arr = np.zeros(self.dim)
        for i, hp_name in enumerate(self.HP_ORDER):
            hp_spec = self.HP_SPACE[hp_name]
            val = config[hp_name]
            
            if hp_spec['type'] == 'float':
                if hp_spec.get('log', False):
                    # Log scale normalization
                    low_log = np.log(hp_spec['low'])
                    high_log = np.log(hp_spec['high'])
                    arr[i] = (np.log(val) - low_log) / (high_log - low_log)
                else:
                    arr[i] = (val - hp_spec['low']) / (hp_spec['high'] - hp_spec['low'])
            elif hp_spec['type'] in ['ordinal', 'categorical']:
                choices = hp_spec['choices']
                idx = choices.index(val)
                arr[i] = idx / (len(choices) - 1) if len(choices) > 1 else 0.5
        
        return np.clip(arr, 0, 1)
    
    def _array_to_dict(self, arr: np.ndarray) -> Dict:
        """Converte un array normalizzato [0,1]^d in dizionario config."""
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
        
        # Parametri fissi
        config['Optimizer'] = 'SGD'
        config['epoch'] = 200  # Full training
        
        return config
    
    def evaluate(self, config: Dict) -> float:
        """Valuta una configurazione, ritorna l'errore (1 - valid_acc/100)."""
        result = self.bench(config)
        # Il risultato è {epoch: {metrics}}
        # Prendiamo l'ultimo epoch disponibile
        last_epoch = max(result.keys())
        valid_acc = result[last_epoch]['valid-acc']
        self.n_evals += 1
        # Ritorniamo l'errore da minimizzare
        return 1.0 - valid_acc / 100.0
    
    def evaluate_array(self, x: np.ndarray) -> float:
        """Valuta un array normalizzato [0,1]^d."""
        config = self._array_to_dict(x)
        return self.evaluate(config)
    
    def sample_random(self, rng: np.random.Generator) -> Dict:
        """Campiona una configurazione random."""
        config = {}
        for hp_name in self.HP_ORDER:
            hp_spec = self.HP_SPACE[hp_name]
            
            if hp_spec['type'] == 'float':
                if hp_spec.get('log', False):
                    log_val = rng.uniform(np.log(hp_spec['low']), np.log(hp_spec['high']))
                    config[hp_name] = np.exp(log_val)
                else:
                    config[hp_name] = rng.uniform(hp_spec['low'], hp_spec['high'])
            elif hp_spec['type'] in ['ordinal', 'categorical']:
                config[hp_name] = rng.choice(hp_spec['choices'])
        
        config['Optimizer'] = 'SGD'
        config['epoch'] = 200
        return config
    
    def reset(self):
        """Reset evaluation counter."""
        self.n_evals = 0


# =============================================================================
# OPTIMIZERS
# =============================================================================

def run_random_search(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """Random Search baseline."""
    rng = np.random.default_rng(seed)
    wrapper.reset()
    
    best_error = float('inf')
    history = []
    
    for _ in range(n_evals):
        config = wrapper.sample_random(rng)
        error = wrapper.evaluate(config)
        best_error = min(best_error, error)
        history.append(best_error)
    
    return history, best_error


def run_optuna_tpe(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """Optuna TPE optimizer."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    wrapper.reset()
    history = []
    best_error = float('inf')
    
    def objective(trial):
        nonlocal best_error
        
        config = {}
        for hp_name in wrapper.HP_ORDER:
            hp_spec = wrapper.HP_SPACE[hp_name]
            
            if hp_spec['type'] == 'float':
                config[hp_name] = trial.suggest_float(
                    hp_name, hp_spec['low'], hp_spec['high'], 
                    log=hp_spec.get('log', False)
                )
            elif hp_spec['type'] == 'ordinal':
                config[hp_name] = trial.suggest_categorical(hp_name, hp_spec['choices'])
            elif hp_spec['type'] == 'categorical':
                config[hp_name] = trial.suggest_categorical(hp_name, hp_spec['choices'])
        
        config['Optimizer'] = 'SGD'
        config['epoch'] = 200
        
        error = wrapper.evaluate(config)
        best_error = min(best_error, error)
        history.append(best_error)
        
        return error
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_evals, show_progress_bar=False)
    
    return history, best_error


def run_turbo_m(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """
    TuRBO-M optimizer (official Uber implementation with multiple trust regions).
    
    Uses the official uber-research/TuRBO implementation.
    TuRBO-M maintains multiple independent trust regions that compete
    for samples, providing better exploration than single-TR TuRBO-1.
    """
    from turbo import TurboM
    import torch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    wrapper.reset()
    dim = wrapper.dim
    
    # Wrapper function for TuRBO (expects (d,) -> scalar)
    eval_count = [0]
    history = []
    best_error = [float('inf')]
    
    class JAHSObjective:
        def __init__(self):
            self.dim = dim
            self.lb = np.zeros(dim)
            self.ub = np.ones(dim)
        
        def __call__(self, x):
            error = wrapper.evaluate_array(x)
            eval_count[0] += 1
            best_error[0] = min(best_error[0], error)
            history.append(best_error[0])
            return error
    
    f = JAHSObjective()
    
    # TuRBO-M parameters
    n_trust_regions = min(5, max(2, n_evals // 50))  # 2-5 trust regions
    n_init = min(2 * dim, n_evals // (n_trust_regions * 3))  # Per trust region
    n_init = max(n_init, dim + 1)  # At least dim+1
    
    # Run TuRBO-M
    turbo_m = TurboM(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=n_evals,
        n_trust_regions=n_trust_regions,
        batch_size=1,
        verbose=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        device="cpu",
        dtype="float64",
    )
    
    turbo_m.optimize()
    
    # Ensure history has correct length
    while len(history) < n_evals:
        history.append(best_error[0])
    
    return history[:n_evals], best_error[0]


def run_alba(wrapper: JAHSBenchWrapper, n_evals: int, seed: int) -> Tuple[List[float], float]:
    """
    ALBA (Adaptive Local Bayesian Algorithm) - il nostro algoritmo custom.
    
    Note: ALBA lavora su [0,1]^d e vede le variabili categoriche come continue.
    Questo può causare problemi perché la LGS lineare assume smoothness,
    ma le dimensioni categoriche creano funzioni a gradini.
    """
    # Import del nostro optimizer
    sys.path.insert(0, '/mnt/workspace/thesis')
    from ALBA_V1 import ALBA
    
    wrapper.reset()
    dim = wrapper.dim
    
    # Parametri ottimizzatore
    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50
    )
    
    # Run optimization
    best_error = float('inf')
    history = []
    
    for _ in range(n_evals):
        # Suggest next point
        x = opt.ask()
        # Evaluate
        y = wrapper.evaluate_array(x)
        # Tell result
        opt.tell(x, y)
        
        best_error = min(best_error, y)
        history.append(best_error)
    
    return history, best_error


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

OPTIMIZERS = {
    'Random': run_random_search,
    'Optuna_TPE': run_optuna_tpe,
    'TuRBO_M': run_turbo_m,
    'ALBA': run_alba,
}


def run_benchmark(
    task: str = 'cifar10',
    n_evals: int = 200,
    n_seeds: int = 10,
    seed_start: int = 0,
    output_dir: str = '/mnt/workspace/thesis/results',
    optimizers: Optional[List[str]] = None
) -> Dict:
    """Run full benchmark."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Select optimizers
    if optimizers is None:
        optimizers = list(OPTIMIZERS.keys())
    
    print("=" * 80)
    print(f"JAHS-BENCH-201 BENCHMARK")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Evaluations: {n_evals}")
    print(f"Seeds: {n_seeds} (starting from {seed_start})")
    print(f"Optimizers: {optimizers}")
    print("=" * 80)
    
    # Initialize wrapper
    print("\nInitializing JAHS-Bench-201...")
    wrapper = JAHSBenchWrapper(task=task)
    print(f"Config space dimension: {wrapper.dim}")
    
    # Results storage
    results = {
        'task': task,
        'n_evals': n_evals,
        'n_seeds': n_seeds,
        'seed_start': seed_start,
        'timestamp': timestamp,
        'optimizers': {}
    }
    
    # Run each optimizer
    for opt_name in optimizers:
        print(f"\n{'=' * 60}")
        print(f"Running {opt_name}...")
        print('=' * 60)
        
        opt_func = OPTIMIZERS[opt_name]
        opt_results = {
            'final_errors': [],
            'histories': [],
            'times': []
        }
        
        for i, seed in enumerate(range(seed_start, seed_start + n_seeds)):
            start_time = time.time()
            
            try:
                history, final_error = opt_func(wrapper, n_evals, seed)
                elapsed = time.time() - start_time
                
                opt_results['final_errors'].append(final_error)
                opt_results['histories'].append(history)
                opt_results['times'].append(elapsed)
                
                print(f"  Seed {seed}: error={final_error:.6f}, time={elapsed:.1f}s")
                
            except Exception as e:
                print(f"  Seed {seed}: FAILED - {e}")
                opt_results['final_errors'].append(float('nan'))
                opt_results['histories'].append([])
                opt_results['times'].append(0)
        
        # Compute statistics
        errors = [e for e in opt_results['final_errors'] if not np.isnan(e)]
        if errors:
            opt_results['mean'] = float(np.mean(errors))
            opt_results['std'] = float(np.std(errors))
            opt_results['min'] = float(np.min(errors))
            opt_results['max'] = float(np.max(errors))
            opt_results['median'] = float(np.median(errors))
        
        results['optimizers'][opt_name] = opt_results
        
        print(f"\n{opt_name} Summary:")
        if 'mean' in opt_results:
            print(f"  Mean error: {opt_results['mean']:.6f} ± {opt_results['std']:.6f}")
            print(f"  Min/Max: {opt_results['min']:.6f} / {opt_results['max']:.6f}")
        else:
            print(f"  No successful runs")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    # Sort by mean error
    ranked = sorted(
        [(name, data['mean'], data['std']) 
         for name, data in results['optimizers'].items() 
         if 'mean' in data],
        key=lambda x: x[1]
    )
    
    for rank, (name, mean, std) in enumerate(ranked, 1):
        print(f"  {rank}. {name:20s}: {mean:.6f} ± {std:.6f}")
    
    # Head-to-head comparison
    print("\n" + "-" * 60)
    print("Head-to-Head Comparison (wins on each seed):")
    print("-" * 60)
    
    opt_names = list(results['optimizers'].keys())
    for i, opt1 in enumerate(opt_names):
        for opt2 in opt_names[i+1:]:
            errors1 = results['optimizers'][opt1]['final_errors']
            errors2 = results['optimizers'][opt2]['final_errors']
            
            wins1 = sum(1 for e1, e2 in zip(errors1, errors2) 
                       if not np.isnan(e1) and not np.isnan(e2) and e1 < e2)
            wins2 = sum(1 for e1, e2 in zip(errors1, errors2) 
                       if not np.isnan(e1) and not np.isnan(e2) and e2 < e1)
            ties = sum(1 for e1, e2 in zip(errors1, errors2) 
                      if not np.isnan(e1) and not np.isnan(e2) and abs(e1 - e2) < 1e-8)
            
            print(f"  {opt1} vs {opt2}: {wins1}-{wins2}-{ties}")
    
    # Save results
    json_path = os.path.join(output_dir, f'jahs_benchmark_{task}_{timestamp}.json')
    with open(json_path, 'w') as f:
        # Convert histories to lists for JSON serialization
        json_results = results.copy()
        for opt_name in json_results['optimizers']:
            json_results['optimizers'][opt_name]['histories'] = [
                [float(x) for x in h] for h in json_results['optimizers'][opt_name]['histories']
            ]
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, f'jahs_benchmark_{task}_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"JAHS-BENCH-201 BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Task: {task}\n")
        f.write(f"Evaluations: {n_evals}\n")
        f.write(f"Seeds: {n_seeds} (from {seed_start})\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n")
        
        f.write("RANKING:\n")
        for rank, (name, mean, std) in enumerate(ranked, 1):
            f.write(f"  {rank}. {name:20s}: {mean:.6f} ± {std:.6f}\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write("DETAILED RESULTS PER OPTIMIZER:\n")
        f.write("-" * 60 + "\n")
        
        for opt_name, opt_data in results['optimizers'].items():
            f.write(f"\n{opt_name}:\n")
            if 'mean' in opt_data:
                f.write(f"  Mean: {opt_data['mean']:.6f}\n")
                f.write(f"  Std:  {opt_data['std']:.6f}\n")
                f.write(f"  Min:  {opt_data['min']:.6f}\n")
                f.write(f"  Max:  {opt_data['max']:.6f}\n")
            else:
                f.write(f"  No successful runs\n")
            f.write(f"  Seeds: {opt_data['final_errors']}\n")
    
    print(f"Summary saved to: {txt_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='JAHS-Bench-201 Benchmark')
    parser.add_argument('--task', type=str, default='cifar10',
                       choices=['cifar10', 'colorectal_histology', 'fashion_mnist'],
                       help='Task to benchmark on')
    parser.add_argument('--n_evals', type=int, default=200,
                       help='Number of evaluations per optimizer')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of seeds to run')
    parser.add_argument('--seed_start', type=int, default=0,
                       help='Starting seed')
    parser.add_argument('--output_dir', type=str, default='/mnt/workspace/thesis/results',
                       help='Output directory for results')
    parser.add_argument('--optimizers', type=str, nargs='+', default=None,
                       choices=list(OPTIMIZERS.keys()),
                       help='Optimizers to run (default: all)')
    
    args = parser.parse_args()
    
    run_benchmark(
        task=args.task,
        n_evals=args.n_evals,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        optimizers=args.optimizers
    )


if __name__ == '__main__':
    main()
