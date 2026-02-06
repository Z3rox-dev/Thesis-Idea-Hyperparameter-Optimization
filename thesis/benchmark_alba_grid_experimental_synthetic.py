#!/usr/bin/env python3
"""thesis/benchmark_alba_grid_experimental_synthetic.py

Benchmark: ALBA Framework (alba_framework_grid) vs ALBA Experimental vs Optuna
su funzioni sintetiche di diverse dimensioni (3D, 8D, 15D).
"""

import sys
from pathlib import Path
import numpy as np
import time
import warnings
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure we can import modules from the thesis folder when run from repo root.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Import Grid Version
try:
    from alba_framework_grid.optimizer import ALBA as AlbaFramework
    print("Successfully imported AlbaFramework from alba_framework_grid")
except ImportError as e:
    print(f"Error importing AlbaGrid: {e}")
    sys.exit(1)

# Import Experimental Version
try:
    from ALBA_V1_experimental import ALBA as AlbaExperimental
    print("Successfully imported AlbaExperimental")
except ImportError as e:
    print(f"Error importing AlbaExperimental: {e}")
    sys.exit(1)

# --- Synthetic Functions (standard definitions; we evaluate them on their native domains) ---

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    # Standard Rosenbrock
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    A = 10.0
    return float(A * x.size + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return float(term1 + term2 + 20.0 + np.e)


def levy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    wi = w[:-1]
    term2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2))
    return float(term1 + term2 + term3)


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(1 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))))


def schwefel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def zakharov(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    s1 = np.sum(x ** 2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2 ** 2 + s2 ** 4)


def styblinski_tang(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x))


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi) ** (2 * m))))


def dixon_price(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x1 = x[0]
    rest = x[1:]
    i = np.arange(2, x.size + 1, dtype=float)
    return float((x1 - 1) ** 2 + np.sum(i * (2 * rest ** 2 - x[:-1]) ** 2))


def alpine1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def _eval_normalized(func, x_norm, bounds: tuple[float, float]) -> float:
    """Map x_norm in [0,1]^d to the function native domain given by bounds."""
    lo, hi = bounds
    x_norm = np.asarray(x_norm, dtype=float)
    x = lo + (hi - lo) * x_norm
    return float(func(x))


BENCHMARKS = {
    "Sphere": (sphere, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-2.0, 2.0)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Ackley": (ackley, (-32.768, 32.768)),
    "Levy": (levy, (-10.0, 10.0)),
    "Griewank": (griewank, (-600.0, 600.0)),
    "Schwefel": (schwefel, (-500.0, 500.0)),
    "Zakharov": (zakharov, (-5.0, 10.0)),
    "StyblinskiTang": (styblinski_tang, (-5.0, 5.0)),
    "Michalewicz": (michalewicz, (0.0, float(np.pi))),
    "DixonPrice": (dixon_price, (-10.0, 10.0)),
    "Alpine1": (alpine1, (0.0, 10.0)),
}

dims = [3, 8, 15]
BUDGET = 250
N_SEEDS = 5

def run_optuna(func, dim, budget, seed, bounds):
    def objective(trial):
        x = [trial.suggest_float(f'x{i}', 0.0, 1.0) for i in range(dim)]
        return _eval_normalized(func, x, bounds)
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    # Optuna TPE can be verbose, suppressing...
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_trials=budget)
    return study.best_value

def run_alba(cls, func, dim, budget, seed, bounds):
    opt = cls(
        bounds=[(0.0, 1.0)] * dim,  # normalized domain
        maximize=False,
        seed=seed,
        total_budget=budget,
    )
    best_val = float('inf')
    
    # Check if 'ask' returns a single point or we need a loop logic
    # Assuming standard ask/tell interface from earlier files
    for _ in range(budget):
        try:
            config = opt.ask()
            # config gives a list/array usually in ALBA_V1
            
            # Check format of config. If dict (from param_space) handle it, 
            # but here we init with bounds so it should be list/array.
            # Grid version might return different format?
            
            if isinstance(config, dict):
                # Should not happen with bounds=... list
                vals = list(config.values())
            else:
                vals = config

            score = _eval_normalized(func, vals, bounds)
            opt.tell(config, score)
            
            best_val = min(best_val, score)
        except Exception as e:
            print(f"Exception in ALBA run: {e}")
            break
            
    return best_val


def run_minimal_improved(func, dim, budget, seed, bounds):
    from hpo_minimal_improved import HPOptimizer as HPOMinimalImproved

    def objective(x_norm: np.ndarray) -> float:
        return _eval_normalized(func, x_norm, bounds)

    start = time.perf_counter()
    opt = HPOMinimalImproved(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
    _, best = opt.optimize(objective, budget=budget)
    elapsed = time.perf_counter() - start
    return float(best), float(elapsed)

overall_start = time.perf_counter()
time_totals = {"Framework": 0.0, "Experimental": 0.0, "MinimalImproved": 0.0, "Optuna": 0.0}

print("Benchmark: Framework vs Experimental vs MinimalImproved vs Optuna")
print(f"dims={dims} | budget={BUDGET} | seeds={list(range(N_SEEDS))}")
print(f"functions={list(BENCHMARKS.keys())}")
print("-")

for func_name, (func, bounds) in BENCHMARKS.items():
    for dim in dims:
        bench_name = f"{func_name}_{dim}D"

        scores_framework: list[float] = []
        scores_exp: list[float] = []
        scores_min_impr: list[float] = []
        scores_optuna: list[float] = []

        times_framework: list[float] = []
        times_exp: list[float] = []
        times_min_impr: list[float] = []
        times_optuna: list[float] = []

        for seed in range(N_SEEDS):
            # Framework (alba_framework_grid)
            t0 = time.perf_counter()
            sf = run_alba(AlbaFramework, func, dim, BUDGET, seed, bounds)
            t_fw = time.perf_counter() - t0
            scores_framework.append(float(sf))
            times_framework.append(float(t_fw))
            time_totals["Framework"] += float(t_fw)

            # Experimental (ALBA_V1_experimental)
            t0 = time.perf_counter()
            se = run_alba(AlbaExperimental, func, dim, BUDGET, seed, bounds)
            t_exp = time.perf_counter() - t0
            scores_exp.append(float(se))
            times_exp.append(float(t_exp))
            time_totals["Experimental"] += float(t_exp)

            # Minimal improved (hpo_minimal_improved)
            smin, tmin = run_minimal_improved(func, dim, BUDGET, seed, bounds)
            scores_min_impr.append(float(smin))
            times_min_impr.append(float(tmin))
            time_totals["MinimalImproved"] += float(tmin)

            # Optuna TPE
            t0 = time.perf_counter()
            so = run_optuna(func, dim, BUDGET, seed, bounds)
            t_opt = time.perf_counter() - t0
            scores_optuna.append(float(so))
            times_optuna.append(float(t_opt))
            time_totals["Optuna"] += float(t_opt)

        # Winner by mean (still computed internally), but we print full per-seed lists.
        means = {
            "Framework": float(np.mean(scores_framework)),
            "Experimental": float(np.mean(scores_exp)),
            "MinimalImproved": float(np.mean(scores_min_impr)),
            "Optuna": float(np.mean(scores_optuna)),
        }
        winner = min(means, key=means.get)

        print(f"{bench_name} | bounds={bounds} | winner={winner}")
        print(f"  Framework       scores={scores_framework} | times_s={times_framework}")
        print(f"  Experimental    scores={scores_exp} | times_s={times_exp}")
        print(f"  MinimalImproved scores={scores_min_impr} | times_s={times_min_impr}")
        print(f"  Optuna          scores={scores_optuna} | times_s={times_optuna}")

overall_elapsed = time.perf_counter() - overall_start
print("-")
print(f"TOTAL elapsed (all optimizers together): {overall_elapsed:.2f}s")
for k, v in time_totals.items():
    print(f"TOTAL time {k:14s}: {v:.2f}s")

