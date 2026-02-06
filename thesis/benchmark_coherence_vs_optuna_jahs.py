"""
Coherence vs Optuna TPE on JAHS-Bench-201.

JAHS has mixed parameter space:
- 2 continuous (log-scale): LearningRate, WeightDecay  
- 3 ordinal: N, W, Resolution
- 8 categorical: Activation, TrivialAugment, Op1-Op6

Uses conda env: py39
Run: source /mnt/workspace/miniconda3/bin/activate py39 && python benchmark_coherence_vs_optuna_jahs.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, "/mnt/workspace/thesis")

# ─────────────────────────────────────────────────────────────────────────────
# JAHS wrapper (from benchmark_jahs.py)
# ─────────────────────────────────────────────────────────────────────────────

class JAHSBenchWrapper:
    """Wrapper for JAHS-Bench-201 that normalizes to [0,1]^13."""
    
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
    
    def _array_to_dict(self, arr: np.ndarray) -> dict:
        """Convert normalized [0,1]^d array to config dict."""
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
        """Evaluate normalized array, return error (1 - valid_acc/100)."""
        config = self._array_to_dict(x)
        result = self.bench(config)
        last_epoch = max(result.keys())
        valid_acc = result[last_epoch]['valid-acc']
        self.n_evals += 1
        return 1.0 - valid_acc / 100.0
    
    def reset(self):
        self.n_evals = 0


# ─────────────────────────────────────────────────────────────────────────────
# Coherence optimizer
# ─────────────────────────────────────────────────────────────────────────────

def run_coherence(wrapper, budget: int, seed: int, categorical_dims: list) -> float:
    """Run Coherence optimizer, return best error found."""
    from alba_framework_coherence.optimizer import ALBA
    
    dim = wrapper.dim
    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        categorical_dims=categorical_dims,
    )
    
    wrapper.reset()
    best = float("inf")
    
    for _ in range(budget):
        x = opt.ask()
        y = float(wrapper.evaluate_array(x))
        opt.tell(x, y)
        best = min(best, y)
    
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Optuna TPE optimizer
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna(wrapper, budget: int, seed: int) -> float:
    """Run Optuna TPE, return best error found."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    dim = wrapper.dim
    wrapper.reset()
    best = [float("inf")]
    
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
        y = float(wrapper.evaluate_array(x))
        best[0] = min(best[0], y)
        return y
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--seeds", type=str, default="0-4", help="e.g. 0-4 or 0,1,2")
    parser.add_argument("--tasks", type=str, default="cifar10,fashion_mnist,colorectal_histology")
    args = parser.parse_args()
    
    # Parse seeds
    if "-" in args.seeds:
        s, e = map(int, args.seeds.split("-"))
        seeds = list(range(s, e + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    budget = args.budget
    checkpoints = [100, 200, 400]
    
    # Categorical dims for JAHS (indices 2-12 are ordinal/categorical)
    # Format: (dim_idx, n_choices)
    categorical_dims = [
        (2, 3),   # N: 3 choices
        (3, 3),   # W: 3 choices  
        (4, 3),   # Resolution: 3 choices
        (5, 3),   # Activation: 3 choices
        (6, 2),   # TrivialAugment: 2 choices
        (7, 5),   # Op1: 5 choices
        (8, 5),   # Op2: 5 choices
        (9, 5),   # Op3: 5 choices
        (10, 5),  # Op4: 5 choices
        (11, 5),  # Op5: 5 choices
        (12, 5),  # Op6: 5 choices
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "/mnt/workspace/thesis/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/coherence_vs_optuna_jahs_b{budget}_s{args.seeds.replace(',','-')}_{timestamp}.json"
    
    print(f"JAHS BENCHMARK | budget={budget} | seeds={seeds}")
    print(f"tasks={tasks}")
    print(f"checkpoints={checkpoints}")
    print(f"save={results_file}")
    print("=" * 80)
    
    results = {
        "config": {"budget": budget, "seeds": seeds, "tasks": tasks, "checkpoints": checkpoints},
        "runs": [],
    }
    
    total_coh_wins = 0
    total_opt_wins = 0
    total_ties = 0
    
    for task in tasks:
        print(f"\n{task.upper()}")
        print("-" * 80)
        
        wrapper = JAHSBenchWrapper(task=task)
        
        task_coh_wins = 0
        task_opt_wins = 0
        task_ties = 0
        
        for seed in seeds:
            t0 = time.time()
            
            coh_best = run_coherence(wrapper, budget, seed, categorical_dims)
            opt_best = run_optuna(wrapper, budget, seed)
            
            elapsed = time.time() - t0
            
            if coh_best < opt_best - 1e-9:
                winner = "COH"
                task_coh_wins += 1
            elif opt_best < coh_best - 1e-9:
                winner = "OPT"
                task_opt_wins += 1
            else:
                winner = "TIE"
                task_ties += 1
            
            # Convert to accuracy for readability
            coh_acc = (1.0 - coh_best) * 100
            opt_acc = (1.0 - opt_best) * 100
            
            print(f"  seed={seed:2d} | COH={coh_acc:.2f}% OPT={opt_acc:.2f}% -> {winner} ({elapsed:.1f}s)")
            
            results["runs"].append({
                "task": task,
                "seed": seed,
                "coh_error": coh_best,
                "opt_error": opt_best,
                "coh_acc": coh_acc,
                "opt_acc": opt_acc,
                "winner": winner,
            })
            
            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
        
        print(f"  {task} SUMMARY: COH wins={task_coh_wins}, OPT wins={task_opt_wins}, ties={task_ties}")
        
        total_coh_wins += task_coh_wins
        total_opt_wins += task_opt_wins
        total_ties += task_ties
    
    print("\n" + "=" * 80)
    print(f"FINAL: Coherence wins={total_coh_wins}, Optuna wins={total_opt_wins}, ties={total_ties}")
    winrate = total_coh_wins / (total_coh_wins + total_opt_wins) * 100 if (total_coh_wins + total_opt_wins) > 0 else 0
    print(f"Coherence winrate: {winrate:.1f}%")
    print(f"Saved: {results_file}")


if __name__ == "__main__":
    main()
