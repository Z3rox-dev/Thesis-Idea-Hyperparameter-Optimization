#!/usr/bin/env python3
"""
ALBA Framework — Quick Demo (with baselines)
==============================================

Compares ALBA vs Optuna (TPE), Random Search, and CMA-ES (nevergrad)
on synthetic benchmarks (continuous and mixed continuous+categorical).

Usage:
    python examples/quick_demo.py
"""

from __future__ import annotations

import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")  # suppress optuna/nevergrad verbosity

# ═══════════════════════════════════════════════════════════════════════════
#  CONTINUOUS SYNTHETIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def sphere(x: np.ndarray) -> float:
    """f(x) = sum(x_i^2).  Min = 0 at origin."""
    return float(np.sum(x ** 2))

def rosenbrock(x: np.ndarray) -> float:
    """Min = 0 at (1,1,...,1)."""
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def rastrigin(x: np.ndarray) -> float:
    """Min = 0 at origin.  Highly multimodal."""
    A = 10
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    """Min = 0 at origin."""
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def levy(x: np.ndarray) -> float:
    """Levy function. Min = 0 at (1,1,...,1)."""
    w = 1 + (x - 1) / 4
    t1 = np.sin(np.pi * w[0]) ** 2
    t2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    t3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(t1 + t2 + t3)

def griewank(x: np.ndarray) -> float:
    """Min = 0 at origin. Multimodal with global structure."""
    s = np.sum(x ** 2) / 4000
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(s - p + 1)

def styblinski_tang(x: np.ndarray) -> float:
    """Min ≈ -39.166*d at x_i ≈ -2.903.  Asymmetric multimodal."""
    return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x))


# ═══════════════════════════════════════════════════════════════════════════
#  MIXED CONTINUOUS + CATEGORICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Penalties per categorical choice (lower = better)
_ACT_PENALTIES = {"relu": 0.0, "tanh": 0.5, "gelu": 0.2, "swish": 0.8}
_OPT_SCALES    = {"sgd": 1.0, "adam": 0.3, "rmsprop": 0.6, "adamw": 0.15}
_NORM_OFFSETS  = {"none": 0.0, "batch": -0.1, "layer": -0.15, "group": -0.05}

def mixed_2c_1cat(config: dict) -> float:
    """2 continuous + 1 categorical (activation). Min ≈ 0 at x=0.3, y=0.7, relu."""
    x, y = config["x"], config["y"]
    return (x - 0.3) ** 2 + (y - 0.7) ** 2 + _ACT_PENALTIES[config["activation"]]

def mixed_3c_2cat(config: dict) -> float:
    """3 continuous + 2 categorical (activation + optimizer). Rosenbrock-like + penalties."""
    x1, x2, x3 = config["x1"], config["x2"], config["x3"]
    base = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 + (x3 - 0.5) ** 2
    act_p = _ACT_PENALTIES[config["activation"]]
    opt_s = _OPT_SCALES[config["optimizer"]]
    return base * opt_s + act_p

def mixed_5c_3cat(config: dict) -> float:
    """5 continuous + 3 categorical. Complex landscape."""
    lr = config["lr"]
    wd = config["wd"]
    drop = config["dropout"]
    h1 = config["hidden1"]
    h2 = config["hidden2"]
    act_p = _ACT_PENALTIES[config["activation"]]
    opt_s = _OPT_SCALES[config["optimizer"]]
    norm_o = _NORM_OFFSETS[config["norm"]]
    # simulated validation loss
    base = (np.log10(lr) + 2.5) ** 2 + 10 * wd + (drop - 0.2) ** 2 + \
           ((h1 - 128) / 256) ** 2 + ((h2 - 64) / 128) ** 2
    return base * opt_s + act_p + norm_o + 0.5

# param_space definitions for ALBA
MIXED_SPACES = {
    "Mixed2c1cat": {
        "space": {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "activation": ["relu", "tanh", "gelu", "swish"],
        },
        "func": mixed_2c_1cat,
        "budget": 400,
    },
    "Mixed3c2cat": {
        "space": {
            "x1": (-2.0, 2.0),
            "x2": (-2.0, 2.0),
            "x3": (-1.0, 1.0),
            "activation": ["relu", "tanh", "gelu", "swish"],
            "optimizer": ["sgd", "adam", "rmsprop", "adamw"],
        },
        "func": mixed_3c_2cat,
        "budget": 400,
    },
    "Mixed5c3cat": {
        "space": {
            "lr": (1e-5, 1e-1),
            "wd": (0.0, 0.1),
            "dropout": (0.0, 0.5),
            "hidden1": (32.0, 512.0),
            "hidden2": (16.0, 256.0),
            "activation": ["relu", "tanh", "gelu", "swish"],
            "optimizer": ["sgd", "adam", "rmsprop", "adamw"],
            "norm": ["none", "batch", "layer", "group"],
        },
        "func": mixed_5c_3cat,
        "budget": 400,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_alba(func, bounds, budget, seed):
    from alba_framework import ALBA
    opt = ALBA(bounds=bounds, maximize=False, seed=seed, total_budget=budget)
    t0 = time.perf_counter()
    _, best_y = opt.optimize(func, budget=budget)
    return best_y, time.perf_counter() - t0

def run_alba_mixed(func, param_space, budget, seed):
    from alba_framework import ALBA
    opt = ALBA(param_space=param_space, maximize=False, seed=seed, total_budget=budget)
    t0 = time.perf_counter()
    _, best_y = opt.optimize(func, budget=budget)
    return best_y, time.perf_counter() - t0

def run_optuna(func, bounds, budget, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", lo, hi) for i, (lo, hi) in enumerate(bounds)])
        return func(x)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return study.best_value, time.perf_counter() - t0

def run_optuna_mixed(func, param_space, budget, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        config = {}
        for name, spec in param_space.items():
            if isinstance(spec, tuple):
                config[name] = trial.suggest_float(name, spec[0], spec[1])
            elif isinstance(spec, list):
                config[name] = trial.suggest_categorical(name, spec)
        return func(config)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return study.best_value, time.perf_counter() - t0

def run_random(func, bounds, budget, seed):
    rng = np.random.default_rng(seed)
    bounds_arr = np.array(bounds)
    t0 = time.perf_counter()
    best = float("inf")
    for _ in range(budget):
        x = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        y = func(x)
        if y < best:
            best = y
    return best, time.perf_counter() - t0

def run_random_mixed(func, param_space, budget, seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best = float("inf")
    for _ in range(budget):
        config = {}
        for name, spec in param_space.items():
            if isinstance(spec, tuple):
                config[name] = rng.uniform(spec[0], spec[1])
            elif isinstance(spec, list):
                config[name] = spec[rng.integers(len(spec))]
        y = func(config)
        if y < best:
            best = y
    return best, time.perf_counter() - t0

def run_cmaes(func, bounds, budget, seed):
    import nevergrad as ng
    bounds_arr = np.array(bounds)
    param = ng.p.Array(init=((bounds_arr[:, 0] + bounds_arr[:, 1]) / 2)).set_bounds(bounds_arr[:, 0], bounds_arr[:, 1])
    opt = ng.optimizers.CMA(parametrization=param, budget=budget)
    opt.parametrization.random_state = np.random.RandomState(seed)
    t0 = time.perf_counter()
    for _ in range(budget):
        x = opt.ask()
        y = func(x.value)
        opt.tell(x, y)
    rec = opt.recommend()
    return func(rec.value), time.perf_counter() - t0

def run_cmaes_mixed(func, param_space, budget, seed):
    """CMA-ES doesn't natively handle categoricals – we enumerate all combos
    and run CMA on the continuous sub-problem for each, splitting budget."""
    import nevergrad as ng
    import itertools

    cont_names = [n for n, s in param_space.items() if isinstance(s, tuple)]
    cat_names  = [n for n, s in param_space.items() if isinstance(s, list)]
    cat_values = [param_space[n] for n in cat_names]

    combos = list(itertools.product(*cat_values))
    budget_per = max(10, budget // len(combos))

    best_overall = float("inf")
    t0 = time.perf_counter()
    for combo in combos:
        cat_config = dict(zip(cat_names, combo))
        bounds_arr = np.array([param_space[n] for n in cont_names])
        param = ng.p.Array(init=((bounds_arr[:, 0] + bounds_arr[:, 1]) / 2)).set_bounds(bounds_arr[:, 0], bounds_arr[:, 1])
        opt = ng.optimizers.CMA(parametrization=param, budget=budget_per)
        opt.parametrization.random_state = np.random.RandomState(seed)
        for _ in range(budget_per):
            x = opt.ask()
            config = dict(zip(cont_names, x.value))
            config.update(cat_config)
            y = func(config)
            opt.tell(x, y)
        rec = opt.recommend()
        config = dict(zip(cont_names, rec.value))
        config.update(cat_config)
        y = func(config)
        if y < best_overall:
            best_overall = y
    return best_overall, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 7]

def main():
    print("=" * 90)
    print("  ALBA Framework — Benchmark vs Optuna / Random / CMA-ES")
    print("=" * 90)

    # ── Part 1: Continuous benchmarks ───────────────────────────────────────
    print("\n  ┌─ CONTINUOUS BENCHMARKS (budget=400, avg over 3 seeds) ─┐\n")

    benchmarks = [
        ("Sphere 5-D",       sphere,          [(-5, 5)] * 5,   400),
        ("Sphere 10-D",      sphere,          [(-5, 5)] * 10,  400),
        ("Rosenbrock 5-D",   rosenbrock,      [(-5, 5)] * 5,   400),
        ("Rosenbrock 10-D",  rosenbrock,      [(-5, 5)] * 10,  400),
        ("Rastrigin 5-D",    rastrigin,       [(-5, 5)] * 5,   400),
        ("Rastrigin 10-D",   rastrigin,       [(-5, 5)] * 10,  400),
        ("Ackley 5-D",       ackley,          [(-5, 5)] * 5,   400),
        ("Levy 5-D",         levy,            [(-10, 10)] * 5, 400),
        ("Griewank 5-D",     griewank,        [(-10, 10)] * 5, 400),
        ("StyblinskiT 5-D",  styblinski_tang, [(-5, 5)] * 5,   400),
    ]

    cont_runners = [
        ("ALBA",   run_alba),
        ("Optuna", run_optuna),
        ("Random", run_random),
        ("CMA-ES", run_cmaes),
    ]

    header = f"  {'Benchmark':22s}"
    for rname, _ in cont_runners:
        header += f" | {rname:>12s}"
    print(header)
    print("  " + "-" * (22 + 15 * len(cont_runners)))

    cont_summary = {rn: [] for rn, _ in cont_runners}
    for bname, func, bounds, budget in benchmarks:
        row = f"  {bname:22s}"
        for rname, rfunc in cont_runners:
            vals = [rfunc(func, bounds, budget, s)[0] for s in SEEDS]
            avg = np.mean(vals)
            cont_summary[rname].append(avg)
            row += f" | {avg:12.4f}"
        print(row)

    # continuous wins
    print()
    cont_wins = {rn: 0 for rn, _ in cont_runners}
    for i in range(len(benchmarks)):
        scores = {rn: cont_summary[rn][i] for rn, _ in cont_runners}
        cont_wins[min(scores, key=scores.get)] += 1
    for rn, _ in cont_runners:
        print(f"  {rn:10s}  {cont_wins[rn]:2d}/{len(benchmarks)}  {'█' * cont_wins[rn]}")

    # ── Part 2: Mixed benchmarks ───────────────────────────────────────────
    print(f"\n  ┌─ MIXED CONTINUOUS+CATEGORICAL BENCHMARKS (budget=400) ─┐\n")

    mixed_runners = [
        ("ALBA",   run_alba_mixed),
        ("Optuna", run_optuna_mixed),
        ("Random", run_random_mixed),
        ("CMA-ES", run_cmaes_mixed),
    ]

    header = f"  {'Benchmark':22s}"
    for rname, _ in mixed_runners:
        header += f" | {rname:>12s}"
    print(header)
    print("  " + "-" * (22 + 15 * len(mixed_runners)))

    mixed_summary = {rn: [] for rn, _ in mixed_runners}
    for bname, spec in MIXED_SPACES.items():
        func, space, budget = spec["func"], spec["space"], spec["budget"]
        row = f"  {bname:22s}"
        for rname, rfunc in mixed_runners:
            vals = [rfunc(func, space, budget, s)[0] for s in SEEDS]
            avg = np.mean(vals)
            mixed_summary[rname].append(avg)
            row += f" | {avg:12.4f}"
        print(row)

    # mixed wins
    print()
    mixed_wins = {rn: 0 for rn, _ in mixed_runners}
    for i in range(len(MIXED_SPACES)):
        scores = {rn: mixed_summary[rn][i] for rn, _ in mixed_runners}
        mixed_wins[min(scores, key=scores.get)] += 1
    for rn, _ in mixed_runners:
        n = len(MIXED_SPACES)
        print(f"  {rn:10s}  {mixed_wins[rn]:2d}/{n}  {'█' * mixed_wins[rn]}")

    # ── Grand total ────────────────────────────────────────────────────────
    total = len(benchmarks) + len(MIXED_SPACES)
    print(f"\n  ┌─ OVERALL ({total} benchmarks) ─┐\n")
    all_names = [rn for rn, _ in cont_runners]
    for rn in all_names:
        w = cont_wins.get(rn, 0) + mixed_wins.get(rn, 0)
        print(f"  {rn:10s}  {w:2d}/{total}  {'█' * w}")

    print("\n" + "=" * 90)
    print("  Done!")
    print("=" * 90)


if __name__ == "__main__":
    main()
