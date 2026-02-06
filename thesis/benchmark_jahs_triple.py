#!/usr/bin/env python3
"""
Benchmark confronto triplo su JAHS-Bench-201:
  - hpo_minimal (old version)
  - hpo_lgs_v3 (non-parametric LGS)
  - ALBA (new ALBA with categorical stats)

Task: colorectal_histology
"""

import numpy as np
import time
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict

# Importa i tre optimizer
sys.path.insert(0, '/mnt/workspace/thesis')
from hpo_minimal_improved import HPOptimizer as HPOMinimalImproved
from hpo_lgs_v3 import HPOptimizer as HPOLGSV3
from ALBA_V1 import ALBA

# JAHS-Bench-201
try:
    from jahs_bench import Benchmark
    JAHS_AVAILABLE = True
except ImportError:
    JAHS_AVAILABLE = False
    print("Warning: JAHS-Bench not available")


# ---------- JAHS Wrapper (from benchmark_jahs.py) ----------

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
    
    def __init__(self, task: str = 'colorectal_histology', 
                 save_dir: str = '/mnt/workspace/jahs_bench_data'):
        from jahs_bench import Benchmark
        
        if task not in self.TASKS:
            raise ValueError(f"Task must be one of {self.TASKS}")
        
        self.task = task
        self.bench = Benchmark(
            task=task, 
            kind='surrogate', 
            download=True, 
            save_dir=save_dir,
            metrics=['valid-acc']
        )
        self.n_evals = 0
    
    @property
    def dim(self) -> int:
        return len(self.HP_ORDER)
    
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
        last_epoch = max(result.keys())
        valid_acc = result[last_epoch]['valid-acc']
        self.n_evals += 1
        return 1.0 - valid_acc / 100.0
    
    def evaluate_array(self, x: np.ndarray) -> float:
        """Valuta un array normalizzato [0,1]^d."""
        config = self._array_to_dict(x)
        return self.evaluate(config)
    
    def reset(self):
        """Reset evaluation counter."""
        self.n_evals = 0


# Categorical dims per JAHS (per ALBA)
# HP_ORDER = ['LearningRate', 'WeightDecay', 'N', 'W', 'Resolution', 
#             'Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']
CATEGORICAL_DIMS = [
    (2, 3),   # N: 3 choices [1, 3, 5]
    (3, 3),   # W: 3 choices [4, 8, 16]
    (4, 3),   # Resolution: 3 choices [0.25, 0.5, 1.0]
    (5, 3),   # Activation: 3 choices ['ReLU', 'Hardswish', 'Mish']
    (6, 2),   # TrivialAugment: 2 choices [True, False]
    (7, 5),   # Op1: 5 choices [0, 1, 2, 3, 4]
    (8, 5),   # Op2: 5 choices
    (9, 5),   # Op3: 5 choices
    (10, 5),  # Op4: 5 choices
    (11, 5),  # Op5: 5 choices
    (12, 5),  # Op6: 5 choices
]


# ---------- Runner functions ----------

def run_random(wrapper: JAHSBenchWrapper, n_evals: int, seed: int):
    """Random Search baseline in normalized space [0,1]^d."""
    rng = np.random.default_rng(seed)
    wrapper.reset()

    best_error = float('inf')
    history: List[float] = []

    for _ in range(n_evals):
        x = rng.uniform(0.0, 1.0, size=wrapper.dim)
        err = float(wrapper.evaluate_array(x))
        best_error = min(best_error, err)
        history.append(best_error)

    return history, best_error


def run_hpo_minimal(wrapper: JAHSBenchWrapper, n_evals: int, seed: int):
    """Run hpo_minimal_improved (configured to minimize validation error)."""
    wrapper.reset()
    bounds = [(0.0, 1.0)] * wrapper.dim
    opt = HPOMinimalImproved(bounds=bounds, maximize=False, seed=seed)

    best_error = float('inf')
    history: List[float] = []

    def objective(x: np.ndarray) -> float:
        nonlocal best_error
        err = float(wrapper.evaluate_array(x))
        best_error = min(best_error, err)
        history.append(best_error)
        return err

    opt.optimize(objective, budget=n_evals)
    return history, best_error


def run_lgs_v3(wrapper: JAHSBenchWrapper, n_evals: int, seed: int):
    """Run hpo_lgs_v3 (minimizes validation error)."""
    wrapper.reset()
    bounds = [(0.0, 1.0)] * wrapper.dim
    opt = HPOLGSV3(bounds=bounds, maximize=False, seed=seed)
    opt.exploration_budget = n_evals

    best_error = float('inf')
    history: List[float] = []

    for _ in range(n_evals):
        x = opt.ask()
        err = float(wrapper.evaluate_array(x))
        opt.tell(x, err)
        best_error = min(best_error, err)
        history.append(best_error)

    return history, best_error


def run_alba(wrapper: JAHSBenchWrapper, n_evals: int, seed: int):
    """Run ALBA with categorical-aware sampling (minimizes validation error)."""
    wrapper.reset()
    dim = wrapper.dim

    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=CATEGORICAL_DIMS,
    )

    best_error = float('inf')
    history: List[float] = []

    for _ in range(n_evals):
        x = opt.ask()
        err = float(wrapper.evaluate_array(x))
        opt.tell(x, err)
        best_error = min(best_error, err)
        history.append(best_error)

    return history, best_error


# ---------- Main benchmark ----------

def main():
    # Config
    TASK = 'colorectal_histology'
    BUDGET = 500
    SEEDS = [0, 1, 2, 3, 4]
    CHECKPOINTS = [50, 100, 200, 300, 500]

    # Output streaming log
    out_dir = '/mnt/workspace/thesis/results/jahs_checkpoints'
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(out_dir, f'benchmark_triple_{TASK}_{timestamp}.txt')

    def log_line(msg: str) -> None:
        print(msg)
        with open(log_path, 'a', encoding='utf-8', buffering=1) as f:
            f.write(msg + "\n")
            f.flush()

    log_line("=" * 70)
    log_line("  BENCHMARK TRIPLO: hpo_minimal_improved vs lgs_v3 vs ALBA su JAHS-Bench-201")
    log_line("=" * 70)

    if not JAHS_AVAILABLE:
        log_line("\n❌ JAHS-Bench non disponibile. Installa con: pip install jahs-bench")
        return
    
    # Algorithms
    algorithms = [
        ('Random', run_random),
        ('hpo_minimal_improved', run_hpo_minimal),
        ('lgs_v3', run_lgs_v3),
        ('ALBA', run_alba),
    ]
    
    # Results storage
    results = {name: {cp: [] for cp in CHECKPOINTS} for name, _ in algorithms}
    curves_all = {name: [] for name, _ in algorithms}
    
    log_line(f"\n📊 Task: {TASK}")
    log_line(f"📊 Budget: {BUDGET} evaluations")
    log_line(f"📊 Seeds: {SEEDS}")
    log_line(f"📊 Checkpoints: {CHECKPOINTS}")
    log_line(f"📄 Streaming log: {log_path}")
    
    for seed in SEEDS:
        log_line(f"\n{'='*60}")
        log_line(f"  SEED = {seed}")
        log_line(f"{'='*60}")
        
        for algo_name, algo_fn in algorithms:
            log_line(f"\n🔄 Running {algo_name}...")
            
            # Fresh benchmark wrapper
            wrapper = JAHSBenchWrapper(task=TASK)
            
            t0 = time.time()
            try:
                curve, final_error = algo_fn(wrapper, BUDGET, seed)
                elapsed = time.time() - t0
                
                # Record checkpoints
                for cp in CHECKPOINTS:
                    if cp <= len(curve):
                        results[algo_name][cp].append(curve[cp - 1])
                
                curves_all[algo_name].append(curve)
                
                log_line(f"   ✅ {algo_name}: {final_error:.6f} in {elapsed:.1f}s ({wrapper.n_evals} evals)")
                
                # Progress report
                for cp in [50, 100, 200]:
                    if cp <= len(curve):
                        log_line(f"      @{cp}: {curve[cp-1]:.6f}")

                # Write full checkpoint line for this run
                ck_vals = []
                for cp in CHECKPOINTS:
                    if cp <= len(curve):
                        ck_vals.append(f"@{cp}={curve[cp-1]:.6f}")
                if ck_vals:
                    log_line(f"   📌 {algo_name} checkpoints: " + ", ".join(ck_vals))
                        
            except Exception as e:
                log_line(f"   ❌ {algo_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    # ---------- Summary ----------
    log_line("\n" + "=" * 70)
    log_line("  RISULTATI FINALI")
    log_line("=" * 70)
    
    header = f"\n{'Algorithm':<20}" + "".join([f"  @{cp:<6}" for cp in CHECKPOINTS])
    log_line(header)
    log_line("-" * 70)
    
    for algo_name, _ in algorithms:
        row = f"{algo_name:<20}"
        for cp in CHECKPOINTS:
            vals = results[algo_name][cp]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f"  {mean_val:.4f}±{std_val:.3f}"
            else:
                row += f"  {'N/A':>12}"
        log_line(row)
    
    # Best at each checkpoint
    log_line("\n🏆 Best algorithm at each checkpoint:")
    for cp in CHECKPOINTS:
        best_algo = None
        best_mean = float('inf')
        for algo_name, _ in algorithms:
            vals = results[algo_name][cp]
            if vals and np.mean(vals) < best_mean:
                best_mean = np.mean(vals)
                best_algo = algo_name
        if best_algo:
            log_line(f"   @{cp}: {best_algo} ({best_mean:.6f})")
    
    # Save results
    np.savez('/mnt/workspace/thesis/results/benchmark_triple_jahs_results.npz',
             results=results,
             curves=curves_all,
             seeds=SEEDS,
             checkpoints=CHECKPOINTS,
             budget=BUDGET)
    log_line("\n💾 Results saved to results/benchmark_triple_jahs_results.npz")
    log_line(f"💾 Streaming log saved to {log_path}")


if __name__ == '__main__':
    main()
