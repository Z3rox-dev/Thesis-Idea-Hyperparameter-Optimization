#!/usr/bin/env python3
"""
Quick synthetic benchmark: ALBA Framework (grid) vs ALBA Experimental
on Sphere and Rosenbrock (normalized [0,1]^d domain).

This is intentionally small/fast to support iterative diagnostics.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure we can import modules from the thesis folder when run from repo root.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

try:
    from alba_framework_grid.optimizer import ALBA as AlbaFramework
except ImportError as e:
    raise SystemExit(f"Error importing AlbaFramework: {e}")

try:
    from ALBA_V1_experimental import ALBA as AlbaExperimental
except ImportError as e:
    raise SystemExit(f"Error importing AlbaExperimental: {e}")


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


BENCH: Dict[str, Tuple[Callable[[np.ndarray], float], Tuple[float, float]]] = {
    "Sphere": (sphere, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-2.0, 2.0)),
}


def _denorm(x_norm: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    x_norm = np.asarray(x_norm, dtype=float)
    return lo + x_norm * (hi - lo)


def _eval_normalized(func: Callable[[np.ndarray], float], x_norm: np.ndarray, bounds: Tuple[float, float]) -> float:
    x = _denorm(x_norm, bounds)
    return float(func(x))


def run_alba(
    cls,
    func: Callable[[np.ndarray], float],
    dim: int,
    budget: int,
    seed: int,
    bounds: Tuple[float, float],
) -> float:
    opt = cls(bounds=[(0.0, 1.0)] * int(dim), maximize=False, seed=int(seed), total_budget=int(budget))
    best_val = float("inf")
    for _ in range(int(budget)):
        x = opt.ask()
        score = _eval_normalized(func, np.asarray(x, dtype=float), bounds)
        opt.tell(x, score)
        if score < best_val:
            best_val = float(score)
    return float(best_val)


def main() -> int:
    dims = [3, 8]
    budget = 150
    seeds = [70, 71]

    print(f"dims={dims} budget={budget} seeds={seeds}")
    print("Function             | Dim | Framework Mean | Exp Mean   | Winner")
    print("-" * 72)

    for name, (fn, bnd) in BENCH.items():
        for d in dims:
            fw: List[float] = []
            ex: List[float] = []
            for s in seeds:
                fw.append(run_alba(AlbaFramework, fn, d, budget, s, bnd))
                ex.append(run_alba(AlbaExperimental, fn, d, budget, s, bnd))

            fw_mean = float(np.mean(fw)) if fw else float("nan")
            ex_mean = float(np.mean(ex)) if ex else float("nan")
            winner = "FW" if fw_mean < ex_mean else "EXP"
            print(f"{name:<20} | {d:<3} | {fw_mean:<14.6g} | {ex_mean:<10.6g} | {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

