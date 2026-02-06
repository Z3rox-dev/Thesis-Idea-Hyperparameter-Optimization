#!/usr/bin/env python3
"""
Synthetic benchmark: ALBA Coherence vs ALBA Gravity(surrogate) vs Optuna TPE vs Random.

Focus: continuous-only optimization across multiple functions, dimensions and seeds.

Usage:
  python3 thesis/benchmark_coherence_vs_gravity_surrogate_synthetic.py --budget 200 --seeds 0-19 --dims 4,15

Notes:
- Minimization (lower is better).
- Saves incremental JSON results to thesis/benchmark_results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
except Exception:  # pragma: no cover
    optuna = None  # type: ignore

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_gravity import ALBA as ALBA_GRAVITY  # noqa: E402
from alba_framework_coherence import ALBA as ALBA_COHERENCE  # noqa: E402

# Import synthetic functions (repo root ParamSpace.py) if available.
try:
    from ParamSpace import (  # type: ignore
        sphere,
        rosenbrock,
        rastrigin,
        ackley,
        levy,
        griewank,
        schwefel,
        zakharov,
        styblinski_tang,
    )
except Exception:

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def rosenbrock(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def rastrigin(x: np.ndarray) -> float:
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    def ackley(x: np.ndarray) -> float:
        n = len(x)
        return float(
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
            + 20
            + np.e
        )

    def levy(x: np.ndarray) -> float:
        w = 1 + (x - 1) / 4
        return float(
            (np.sin(np.pi * w[0]) ** 2)
            + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        )

    def griewank(x: np.ndarray) -> float:
        return float(
            1
            + np.sum(x**2) / 4000
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        )

    def schwefel(x: np.ndarray) -> float:
        return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def zakharov(x: np.ndarray) -> float:
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return float(sum1 + sum2**2 + sum2**4)

    def styblinski_tang(x: np.ndarray) -> float:
        return float(0.5 * np.sum(x**4 - 16 * x**2 + 5 * x))


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"

SYN_FUNCS: Dict[str, Tuple[Callable[[np.ndarray], float], Tuple[float, float]]] = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-5.0, 5.0)),
    "levy": (levy, (-10.0, 10.0)),
    "griewank": (griewank, (-600.0, 600.0)),
    "schwefel": (schwefel, (-500.0, 500.0)),
    "zakharov": (zakharov, (-5.0, 10.0)),
    "styblinski_tang": (styblinski_tang, (-5.0, 5.0)),
}


def _save(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _parse_seeds(arg: str) -> List[int]:
    arg = str(arg or "").strip()
    if not arg:
        return []
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _run_alba_gravity_surrogate(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
) -> float:
    bounds = [bounds_range] * dim
    opt = ALBA_GRAVITY(
        bounds=bounds,
        maximize=False,
        seed=int(seed),
        total_budget=int(budget),
        split_depth_max=16,
        global_random_prob=0.05,
        stagnation_threshold=50,
        cube_gravity=True,
        cube_gravity_mode="surrogate_gradient",
    )
    if getattr(opt, "_use_cube_gravity", None) is not True:
        raise RuntimeError("Expected ALBA_GRAVITY with cube_gravity=True.")

    best = float("inf")
    for _ in range(int(budget)):
        x = opt.ask()
        y = float(func(np.asarray(x, dtype=float)))
        opt.tell(x, y)
        if y < best:
            best = y
    return float(best)


def _run_alba_coherence(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
    *,
    use_coherence_gating: bool,
) -> float:
    bounds = [bounds_range] * dim
    opt = ALBA_COHERENCE(
        bounds=bounds,
        maximize=False,
        seed=int(seed),
        total_budget=int(budget),
        split_depth_max=16,
        global_random_prob=0.05,
        stagnation_threshold=50,
        use_coherence_gating=bool(use_coherence_gating),
    )
    best = float("inf")
    for _ in range(int(budget)):
        x = opt.ask()
        y = float(func(np.asarray(x, dtype=float)))
        opt.tell(x, y)
        if y < best:
            best = y
    return float(best)


def _run_random(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    lo, hi = float(bounds_range[0]), float(bounds_range[1])
    best = float("inf")
    for _ in range(int(budget)):
        x = rng.uniform(lo, hi, size=int(dim))
        y = float(func(np.asarray(x, dtype=float)))
        if y < best:
            best = y
    return float(best)


def _run_optuna_tpe(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
) -> float:
    if optuna is None:
        return float("inf")

    lo, hi = float(bounds_range[0]), float(bounds_range[1])
    best = float("inf")

    def objective(trial: Any) -> float:
        nonlocal best
        x = np.array([trial.suggest_float(f"x{i}", lo, hi) for i in range(dim)], dtype=float)
        y = float(func(x))
        if y < best:
            best = y
        return y

    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(budget), show_progress_bar=False)
    return float(best)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Synthetic: ALBA coherence vs gravity(surrogate) vs optuna vs random"
    )
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seeds", type=str, default="0-19")
    p.add_argument("--dims", type=str, default="4,15")
    p.add_argument(
        "--funcs",
        type=str,
        default=",".join(SYN_FUNCS.keys()),
        help="Comma-separated function names",
    )
    p.add_argument(
        "--coherence-gating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable coherence gating inside ALBA_COHERENCE",
    )
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    dims = sorted(set(int(x.strip()) for x in str(args.dims).split(",") if x.strip()))
    funcs = [f.strip() for f in str(args.funcs).split(",") if f.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(
        RESULTS_DIR,
        f"synthetic_coherence_vs_gravity_surrogate_optuna_random_b{args.budget}_{ts}.json",
    )

    results: Dict[str, Any] = {
        "benchmark": "synthetic_coherence_vs_gravity_surrogate_optuna_random",
        "methods": {
            "alba_gravity_surrogate": {
                "kind": "alba_framework_gravity",
                "cube_gravity_mode": "surrogate_gradient",
            },
            "alba_coherence": {
                "kind": "alba_framework_coherence",
                "use_coherence_gating": bool(args.coherence_gating),
            },
            "optuna_tpe": {"kind": "optuna_TPE"},
            "random": {"kind": "random_search"},
        },
        "config": {
            "budget": int(args.budget),
            "seeds": seeds,
            "dims": dims,
            "funcs": funcs,
            "coherence_gating": bool(args.coherence_gating),
        },
        "tasks": {},
    }
    _save(out_file, results)

    print("=" * 78)
    print("SYNTHETIC: ALBA coherence vs gravity(surrogate) vs optuna_tpe vs random")
    print(f"budget={args.budget} | seeds={seeds} | dims={dims}")
    print(f"funcs={funcs}")
    print(f"coherence_gating={bool(args.coherence_gating)}")
    print(f"out={out_file}")
    print("=" * 78)
    print()

    for func_name in funcs:
        if func_name not in SYN_FUNCS:
            print(f"Skipping unknown func: {func_name}")
            continue
        func, bounds_range = SYN_FUNCS[func_name]

        for dim in dims:
            task_key = f"{func_name}_{dim}D"
            results["tasks"].setdefault(task_key, [])
            print(f"== {task_key} ==")

            rows: List[Dict[str, Any]] = []
            for seed in seeds:
                row: Dict[str, Any] = {"seed": int(seed)}

                row["alba_gravity_surrogate"] = float(
                    _run_alba_gravity_surrogate(func, dim, bounds_range, seed, int(args.budget))
                )
                row["alba_coherence"] = float(
                    _run_alba_coherence(
                        func,
                        dim,
                        bounds_range,
                        seed,
                        int(args.budget),
                        use_coherence_gating=bool(args.coherence_gating),
                    )
                )
                row["random"] = float(_run_random(func, dim, bounds_range, seed, int(args.budget)))
                row["optuna_tpe"] = float(_run_optuna_tpe(func, dim, bounds_range, seed, int(args.budget)))

                rows.append(row)
                results["tasks"][task_key].append(row)
                _save(out_file, results)

            # Pretty table
            seed_strs = [str(r["seed"]) for r in rows]
            w_seed = max(4, max((len(s) for s in seed_strs), default=4))

            cols = ["alba_gravity_surrogate", "alba_coherence", "random", "optuna_tpe"]
            widths = {c: max(len(c), 12) for c in cols}

            header = f"{'seed':<{w_seed}} | " + " | ".join(f"{c:>{widths[c]}}" for c in cols) + " | best"
            print(header)
            print(f"{'-'*w_seed}-+-" + "-+-".join("-" * widths[c] for c in cols) + "-+------")

            wins: Dict[str, int] = {c: 0 for c in cols}
            for r in rows:
                seed = int(r["seed"])
                vals = {c: float(r[c]) for c in cols}
                best_col = min(vals, key=lambda k: vals[k])
                wins[best_col] += 1
                row_s = f"{seed:<{w_seed}} | " + " | ".join(
                    f"{vals[c]:>{widths[c]}.6f}" for c in cols
                ) + f" | {best_col}"
                print(row_s)

            wins_row = f"{'wins':<{w_seed}} | " + " | ".join(
                f"{wins[c]:>{widths[c]}d}" for c in cols
            ) + " |"
            print(wins_row)
            print()

    print(f"✓ Results saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

