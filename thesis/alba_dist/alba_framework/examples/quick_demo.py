#!/usr/bin/env python3
"""
ALBA Framework — Quick Demo
============================

This script demonstrates ALBA on classic synthetic benchmarks.
It runs out of the box with no external data files.

Usage:
    python examples/quick_demo.py

Requirements:
    pip install alba-framework          # core
    pip install alba-framework[examples] # adds matplotlib for plots
"""

from __future__ import annotations

import sys
import time
import numpy as np

# ── synthetic test functions ────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """Sphere function  f(x) = sum(x_i^2).  Min at origin."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function.  Min = 0 at (1,1,...,1)."""
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function.  Min = 0 at origin.  Highly multimodal."""
    A = 10
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function.  Min = 0 at origin."""
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / n))
                 - np.exp(sum2 / n) + 20 + np.e)


# ── mixed continuous + categorical example ──────────────────────────────────

def mixed_objective(config: dict) -> float:
    """
    A simple mixed-type objective: the professor can see how ALBA
    handles categorical + continuous parameters together.
    """
    x = config["x"]
    y = config["y"]
    activation = config["activation"]

    # Each activation function defines a different landscape
    penalties = {"relu": 0.0, "tanh": 0.5, "gelu": 0.2, "swish": 0.8}
    penalty = penalties.get(activation, 1.0)

    return (x - 0.3) ** 2 + (y - 0.7) ** 2 + penalty


# ── runner ──────────────────────────────────────────────────────────────────

def run_benchmark(name: str, func, bounds, budget: int = 150, seed: int = 42):
    """Run ALBA on a single benchmark and print results."""
    from alba_framework_potential import ALBA

    dim = len(bounds)
    opt = ALBA(bounds=bounds, maximize=False, seed=seed, total_budget=budget)

    t0 = time.perf_counter()
    best_x, best_y = opt.optimize(func, budget=budget)
    elapsed = time.perf_counter() - t0

    stats = opt.get_statistics()
    print(f"  {name:20s}  |  dim={dim}  budget={budget}  "
          f"best_y={best_y:12.6f}  leaves={stats['n_leaves']:3d}  "
          f"time={elapsed:.2f}s")
    return best_y


def run_mixed_demo(budget: int = 100, seed: int = 42):
    """Run ALBA on a mixed continuous+categorical problem."""
    from alba_framework_potential import ALBA

    param_space = {
        "x": (0.0, 1.0),                           # continuous
        "y": (0.0, 1.0),                           # continuous
        "activation": ["relu", "tanh", "gelu", "swish"],  # categorical
    }

    opt = ALBA(param_space=param_space, maximize=False, seed=seed,
               total_budget=budget)

    t0 = time.perf_counter()
    best_config, best_y = opt.optimize(mixed_objective, budget=budget)
    elapsed = time.perf_counter() - t0

    print(f"  {'mixed_categorical':20s}  |  budget={budget}  "
          f"best_y={best_y:12.6f}  config={best_config}  time={elapsed:.2f}s")
    return best_config, best_y


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  ALBA Framework — Quick Demo")
    print("=" * 78)
    print()

    benchmarks = [
        ("Sphere 5-D",     sphere,     [(-5, 5)] * 5,   200),
        ("Rosenbrock 5-D", rosenbrock,  [(-5, 5)] * 5,   200),
        ("Rastrigin 5-D",  rastrigin,   [(-5, 5)] * 5,   200),
        ("Ackley 5-D",     ackley,      [(-5, 5)] * 5,   200),
        ("Sphere 10-D",    sphere,      [(-5, 5)] * 10,  300),
        ("Rastrigin 10-D", rastrigin,   [(-5, 5)] * 10,  300),
    ]

    print("  Benchmark             |  Details")
    print("  " + "-" * 72)

    results = {}
    for name, func, bounds, budget in benchmarks:
        best = run_benchmark(name, func, bounds, budget=budget)
        results[name] = best

    print()
    print("  Mixed continuous + categorical:")
    print("  " + "-" * 72)
    run_mixed_demo()

    print()
    print("=" * 78)
    print("  Done!  All benchmarks completed successfully.")
    print("=" * 78)

    # ── optional convergence plot ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
        import matplotlib.pyplot as plt

        print("\n  Generating convergence plot → convergence_demo.png ...")

        from alba_framework_potential import ALBA

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, func, bounds, budget) in zip(
            axes, [benchmarks[0], benchmarks[1], benchmarks[2]]
        ):
            opt = ALBA(bounds=bounds, maximize=False, seed=0, total_budget=budget)
            trace = []
            best_so_far = float("inf")
            for _ in range(budget):
                x = opt.ask()
                y = func(x)
                opt.tell(x, y)
                best_so_far = min(best_so_far, y)
                trace.append(best_so_far)
            ax.plot(trace, linewidth=1.5)
            ax.set_title(name)
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best value")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.grid(True, alpha=0.3)

        fig.suptitle("ALBA — Convergence Traces", fontsize=14)
        fig.tight_layout()
        fig.savefig("convergence_demo.png", dpi=150)
        print("  ✓ Saved convergence_demo.png")

    except ImportError:
        print("\n  (matplotlib not installed — skipping plot. "
              "Install with:  pip install matplotlib)")


if __name__ == "__main__":
    main()
