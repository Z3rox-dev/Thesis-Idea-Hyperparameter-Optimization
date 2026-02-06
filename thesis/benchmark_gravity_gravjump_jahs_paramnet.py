#!/usr/bin/env python3
"""
JAHS + ParamNet benchmark: alba_experimental vs Optuna vs alba_framework_gravity.

Runs:
- JAHS-Bench-201 surrogate tasks (cifar10/fashion_mnist/colorectal_histology)
- HPOBench ParamNet surrogates (adult/higgs/letter/mnist/...)

Intended env: conda `py39` (has jahs_bench + HPOBench deps).

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_gravity_gravjump_jahs_paramnet.py \
      --jahs-tasks cifar10,colorectal_histology \
      --paramnet-datasets letter,mnist \
            --budget 400 --seeds 70-79 --paramnet-step 50

Output:
    - Per-problem per-seed table
    - Mean ± std summary for each method
    - JSON saved to thesis/benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure local thesis/ is importable as a package root.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_gravity import ALBA as ALBA_GRAV  # noqa: E402
from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def _save_results(path: str, obj: dict) -> None:
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
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


@dataclass
class RunSummary:
    best: float
    unique_keys: int = 0
    switch_rate: float = 0.0


def _cat_key_from_x(x: np.ndarray, *, categorical_dims: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    key: List[int] = []
    x = np.asarray(x, dtype=float)
    for dim_idx, n_ch in categorical_dims:
        dim_idx_i = int(dim_idx)
        n_ch_i = int(n_ch)
        if n_ch_i <= 1:
            key.append(0)
            continue
        v = float(np.clip(x[dim_idx_i], 0.0, 1.0))
        idx = int(round(v * float(n_ch_i - 1)))
        idx = int(np.clip(idx, 0, n_ch_i - 1))
        key.append(idx)
    return tuple(key)


def _run_alba_bounds(
    objective,
    *,
    bounds: List[Tuple[float, float]],
    seed: int,
    budget: int,
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
    key_dims_for_metrics: Optional[List[Tuple[int, int]]] = None,
    alba_cls: Any = None,
) -> Tuple[RunSummary, List[Tuple[int, ...]]]:
    cls = alba_cls or ALBA_GRAV
    kwargs: Dict[str, Any] = {
        "bounds": bounds,
        "maximize": False,
        "seed": int(seed),
        "total_budget": int(budget),
        "split_depth_max": 8,
        "global_random_prob": 0.05,
        "stagnation_threshold": 50,
        "categorical_dims": categorical_dims,
    }

    # Keep construction resilient across ALBA variants.
    try:
        opt = cls(**kwargs)
    except TypeError:
        # Minimal fallback.
        kwargs_min = {
            "bounds": bounds,
            "maximize": False,
            "seed": int(seed),
            "total_budget": int(budget),
            "categorical_dims": categorical_dims,
        }
        opt = cls(**kwargs_min)

    best = float("inf")
    keys: List[Tuple[int, ...]] = []
    metric_dims = key_dims_for_metrics or categorical_dims or []

    for _ in range(int(budget)):
        x = opt.ask()
        y = float(objective(x))
        opt.tell(x, y)
        best = min(best, y)
        if metric_dims:
            keys.append(_cat_key_from_x(np.asarray(x, dtype=float), categorical_dims=metric_dims))

    if not keys:
        return RunSummary(best=best), keys

    unique = len(set(keys))
    switches = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    switch_rate = float(switches) / float(max(1, len(keys) - 1))
    return RunSummary(best=best, unique_keys=unique, switch_rate=switch_rate), keys


def _run_optuna_bounds(
    objective,
    *,
    dim: int,
    seed: int,
    budget: int,
    key_dims_for_metrics: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[RunSummary, List[Tuple[int, ...]]]:
    try:
        import optuna  # type: ignore
    except Exception as e:
        raise RuntimeError("Optuna not available in this environment") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    keys: List[Tuple[int, ...]] = []
    metric_dims = key_dims_for_metrics or []

    def optuna_objective(trial):
        x = np.asarray([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(int(dim))], dtype=float)
        y = float(objective(x))
        if metric_dims:
            keys.append(_cat_key_from_x(x, categorical_dims=metric_dims))
        return y

    study.optimize(optuna_objective, n_trials=int(budget), show_progress_bar=False)
    best = float(study.best_value)

    if not keys:
        return RunSummary(best=best), keys
    unique = len(set(keys))
    switches = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    switch_rate = float(switches) / float(max(1, len(keys) - 1))
    return RunSummary(best=best, unique_keys=unique, switch_rate=switch_rate), keys


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _fmt(x: Optional[float], *, width: int = 9, prec: int = 4) -> str:
    if x is None or not np.isfinite(float(x)):
        return " " * (width - 3) + "n/a"
    s = f"{float(x):.{prec}f}"
    return s.rjust(width)


def _make_jahs_objective(task: str, *, nepochs: int) -> Any:
    try:
        from jahs_bench import Benchmark  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "jahs_bench not available. Run with conda env py39."
        ) from e

    bench = Benchmark(
        task=str(task),
        kind="surrogate",
        download=True,
        save_dir="/mnt/workspace/jahs_bench_data",
        metrics=["valid-acc"],
    )

    def objective(x_norm: np.ndarray) -> float:
        x_norm = np.asarray(x_norm, dtype=float)
        config = {
            "LearningRate": float(
                10 ** (np.log10(0.001) + float(x_norm[0]) * (np.log10(1.0) - np.log10(0.001)))
            ),
            "WeightDecay": float(
                10 ** (np.log10(1e-5) + float(x_norm[1]) * (np.log10(0.01) - np.log10(1e-5)))
            ),
            "N": [1, 3, 5][int(np.clip(int(float(x_norm[2]) * 2.99), 0, 2))],
            "W": [4, 8, 16][int(np.clip(int(float(x_norm[3]) * 2.99), 0, 2))],
            "Resolution": [0.25, 0.5, 1.0][int(np.clip(int(float(x_norm[4]) * 2.99), 0, 2))],
            "Activation": ["ReLU", "Hardswish", "Mish"][int(np.clip(int(float(x_norm[5]) * 2.99), 0, 2))],
            "TrivialAugment": [True, False][int(np.clip(int(float(x_norm[6]) * 1.99), 0, 1))],
            "Op1": int(np.clip(int(float(x_norm[7]) * 4.99), 0, 4)),
            "Op2": int(np.clip(int(float(x_norm[8]) * 4.99), 0, 4)),
            "Op3": int(np.clip(int(float(x_norm[9]) * 4.99), 0, 4)),
            "Op4": int(np.clip(int(float(x_norm[10]) * 4.99), 0, 4)),
            "Op5": int(np.clip(int(float(x_norm[11]) * 4.99), 0, 4)),
            "Op6": int(np.clip(int(float(x_norm[12]) * 4.99), 0, 4)),
            "Optimizer": "SGD",
            "epoch": int(nepochs),
        }

        result = bench(config)
        last_epoch = max(result.keys())
        valid_acc = float(result[last_epoch]["valid-acc"])
        return 1.0 - valid_acc / 100.0

    return objective


def _make_paramnet_objective(dataset: str, *, step: int) -> Any:
    # ParamNet surrogates shipped with HPOBench are pickled with older scikit-learn
    # module paths (e.g. sklearn.ensemble.forest). Add small import aliases so
    # unpickling works with modern scikit-learn.
    try:
        import sklearn.ensemble._forest as _sk_forest  # type: ignore

        sys.modules.setdefault("sklearn.ensemble.forest", _sk_forest)
    except Exception:
        pass

    try:
        import sklearn.tree._classes as _sk_tree_classes  # type: ignore

        sys.modules.setdefault("sklearn.tree.tree", _sk_tree_classes)
    except Exception:
        pass

    # HPOBench (vendored)
    sys.path.insert(0, "/mnt/workspace/HPOBench")
    import ConfigSpace as CS  # type: ignore

    from hpobench.benchmarks.surrogates.paramnet_benchmark import (  # type: ignore
        ParamNetAdultOnStepsBenchmark,
        ParamNetHiggsOnStepsBenchmark,
        ParamNetLetterOnStepsBenchmark,
        ParamNetMnistOnStepsBenchmark,
        ParamNetOptdigitsOnStepsBenchmark,
        ParamNetPokerOnStepsBenchmark,
    )

    bench_map = {
        "adult": ParamNetAdultOnStepsBenchmark,
        "higgs": ParamNetHiggsOnStepsBenchmark,
        "letter": ParamNetLetterOnStepsBenchmark,
        "mnist": ParamNetMnistOnStepsBenchmark,
        "optdigits": ParamNetOptdigitsOnStepsBenchmark,
        "poker": ParamNetPokerOnStepsBenchmark,
    }
    ds = str(dataset).strip().lower()
    if ds not in bench_map:
        raise ValueError(f"Unknown ParamNet dataset: {dataset}")

    bench = bench_map[ds]()
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()

    def objective(x01: np.ndarray) -> float:
        x01 = np.asarray(x01, dtype=float)
        values: Dict[str, Any] = {}
        for i, hp in enumerate(hps):
            v01 = float(np.clip(x01[i], 0.0, 1.0))
            lo = float(getattr(hp, "lower"))
            hi = float(getattr(hp, "upper"))
            if getattr(hp, "log", False):
                val = float(np.exp(np.log(lo) + v01 * (np.log(hi) - np.log(lo))))
            else:
                val = float(lo + v01 * (hi - lo))
            if isinstance(hp, CS.UniformIntegerHyperparameter):
                val = int(round(val))
                val = max(int(hp.lower), min(int(hp.upper), int(val)))
            values[hp.name] = val

        cfg = CS.Configuration(cs, values=values)
        res = bench.objective_function(configuration=cfg, fidelity={"step": int(step)})
        return float(res["function_value"])

    return objective, len(hps)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--seeds", default="70-79")
    p.add_argument("--jahs-tasks", default="cifar10,colorectal_histology")
    p.add_argument("--jahs-nepochs", type=int, default=200)
    p.add_argument("--paramnet-datasets", default="letter,mnist")
    p.add_argument("--paramnet-step", type=int, default=50)
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")

    # JAHS: 13 dims, first 2 continuous, the rest discrete.
    jahs_bounds = [(0.0, 1.0)] * 13
    jahs_cat_dims = [
        (2, 3),   # N
        (3, 3),   # W
        (4, 3),   # Resolution
        (5, 3),   # Activation
        (6, 2),   # TrivialAugment
        (7, 5),   # Op1
        (8, 5),   # Op2
        (9, 5),   # Op3
        (10, 5),  # Op4
        (11, 5),  # Op5
        (12, 5),  # Op6
    ]

    jahs_tasks = [t.strip() for t in str(args.jahs_tasks).split(",") if t.strip()]
    paramnet_datasets = [d.strip() for d in str(args.paramnet_datasets).split(",") if d.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/gravjump_jahs_paramnet_3way_b{int(args.budget)}_{ts}.json"

    methods = {
        "alba_experimental": {"kind": "ALBA_V1_experimental"},
        "optuna": {"kind": "TPESampler"},
        "alba_framework_gravity": {"kind": "alba_framework_gravity"},
    }

    results: Dict[str, Any] = {
        "config": {
            "budget": int(args.budget),
            "seeds": seeds,
            "jahs_tasks": jahs_tasks,
            "jahs_nepochs": int(args.jahs_nepochs),
            "paramnet_datasets": paramnet_datasets,
            "paramnet_step": int(args.paramnet_step),
            "methods": methods,
            "timestamp": ts,
        },
        "jahs": {},
        "paramnet": {},
    }
    _save_results(out_path, results)

    print("=" * 78)
    print("Benchmark (py39): alba_experimental vs Optuna vs alba_framework_gravity")
    print(f"budget={int(args.budget)} | seeds={seeds} | out={out_path}")
    print(f"JAHS tasks: {jahs_tasks} | ParamNet datasets: {paramnet_datasets}")
    print("=" * 78)

    # ---------------------------------------------------------------------
    # JAHS
    # ---------------------------------------------------------------------
    for task in jahs_tasks:
        objective = _make_jahs_objective(task, nepochs=int(args.jahs_nepochs))

        per_seed: Dict[int, Dict[str, RunSummary]] = {}
        rows: List[Tuple[int, float, float, float]] = []

        print(f"\n== JAHS {task} (nepochs={int(args.jahs_nepochs)}) ==")
        print("seed | alba_experimental |    optuna | gravity(gravjump) | uniq(exp/opt/grav)")
        for seed in seeds:
            t0 = time.time()

            exp, _ = _run_alba_bounds(
                objective,
                bounds=jahs_bounds,
                seed=seed,
                budget=int(args.budget),
                categorical_dims=jahs_cat_dims,
                key_dims_for_metrics=jahs_cat_dims,
                alba_cls=ALBA_EXP,
            )
            opt, _ = _run_optuna_bounds(
                objective,
                dim=13,
                seed=seed,
                budget=int(args.budget),
                key_dims_for_metrics=jahs_cat_dims,
            )
            grav, _ = _run_alba_bounds(
                objective,
                bounds=jahs_bounds,
                seed=seed,
                budget=int(args.budget),
                categorical_dims=jahs_cat_dims,
                key_dims_for_metrics=jahs_cat_dims,
                alba_cls=ALBA_GRAV,
            )
            elapsed = time.time() - t0

            per_seed[int(seed)] = {
                "alba_experimental": exp,
                "optuna": opt,
                "alba_framework_gravity": grav,
            }
            rows.append((int(seed), float(exp.best), float(opt.best), float(grav.best)))

            print(
                f"{int(seed):4d} |"
                f" {_fmt(exp.best, width=15)} |"
                f" {_fmt(opt.best, width=9)} |"
                f" {_fmt(grav.best, width=15)} |"
                f" {exp.unique_keys:4d}/{opt.unique_keys:3d}/{grav.unique_keys:4d}"
                f"  ({elapsed:.1f}s)"
            )

        # Aggregate summary
        exp_vals = [v[1] for v in rows]
        opt_vals = [v[2] for v in rows]
        grav_vals = [v[3] for v in rows]
        m_exp, s_exp = _mean_std(exp_vals)
        m_opt, s_opt = _mean_std(opt_vals)
        m_grav, s_grav = _mean_std(grav_vals)

        wins_exp = sum(1 for _, a, b, c in rows if a <= min(b, c))
        wins_opt = sum(1 for _, a, b, c in rows if b <= min(a, c))
        wins_grav = sum(1 for _, a, b, c in rows if c <= min(a, b))

        print("  Summary (lower is better):")
        print(f"  alba_experimental      mean={m_exp:.4f} ± {s_exp:.4f} | wins={wins_exp}/{len(rows)}")
        print(f"  optuna                mean={m_opt:.4f} ± {s_opt:.4f} | wins={wins_opt}/{len(rows)}")
        print(f"  alba_framework_gravity mean={m_grav:.4f} ± {s_grav:.4f} | wins={wins_grav}/{len(rows)}")

        results["jahs"][str(task)] = {
            "per_seed": {str(k): {m: vars(s) for m, s in v.items()} for k, v in per_seed.items()},
            "summary": {
                "alba_experimental": {"mean": m_exp, "std": s_exp, "wins": wins_exp},
                "optuna": {"mean": m_opt, "std": s_opt, "wins": wins_opt},
                "alba_framework_gravity": {"mean": m_grav, "std": s_grav, "wins": wins_grav},
            },
        }
        _save_results(out_path, results)

    # ---------------------------------------------------------------------
    # ParamNet
    # ---------------------------------------------------------------------
    for ds in paramnet_datasets:
        objective, dim = _make_paramnet_objective(ds, step=int(args.paramnet_step))
        bounds = [(0.0, 1.0)] * int(dim)

        per_seed: Dict[int, Dict[str, RunSummary]] = {}
        rows: List[Tuple[int, float, float, float]] = []

        print(f"\n== ParamNet {ds} (step={int(args.paramnet_step)}) ==")
        print("seed | alba_experimental |    optuna | alba_framework_gravity")
        for seed in seeds:
            t0 = time.time()

            exp, _ = _run_alba_bounds(
                objective,
                bounds=bounds,
                seed=seed,
                budget=int(args.budget),
                categorical_dims=None,
                alba_cls=ALBA_EXP,
            )
            opt, _ = _run_optuna_bounds(
                objective,
                dim=int(dim),
                seed=seed,
                budget=int(args.budget),
                key_dims_for_metrics=None,
            )
            grav, _ = _run_alba_bounds(
                objective,
                bounds=bounds,
                seed=seed,
                budget=int(args.budget),
                categorical_dims=None,
                alba_cls=ALBA_GRAV,
            )
            elapsed = time.time() - t0

            per_seed[int(seed)] = {
                "alba_experimental": exp,
                "optuna": opt,
                "alba_framework_gravity": grav,
            }
            rows.append((int(seed), float(exp.best), float(opt.best), float(grav.best)))

            print(
                f"{int(seed):4d} |"
                f" {_fmt(exp.best, width=15, prec=6)} |"
                f" {_fmt(opt.best, width=9, prec=6)} |"
                f" {_fmt(grav.best, width=15, prec=6)}"
                f"  ({elapsed:.1f}s)"
            )

        exp_vals = [v[1] for v in rows]
        opt_vals = [v[2] for v in rows]
        grav_vals = [v[3] for v in rows]
        m_exp, s_exp = _mean_std(exp_vals)
        m_opt, s_opt = _mean_std(opt_vals)
        m_grav, s_grav = _mean_std(grav_vals)

        wins_exp = sum(1 for _, a, b, c in rows if a <= min(b, c))
        wins_opt = sum(1 for _, a, b, c in rows if b <= min(a, c))
        wins_grav = sum(1 for _, a, b, c in rows if c <= min(a, b))

        print("  Summary (lower is better):")
        print(f"  alba_experimental      mean={m_exp:.6f} ± {s_exp:.6f} | wins={wins_exp}/{len(rows)}")
        print(f"  optuna                mean={m_opt:.6f} ± {s_opt:.6f} | wins={wins_opt}/{len(rows)}")
        print(f"  alba_framework_gravity mean={m_grav:.6f} ± {s_grav:.6f} | wins={wins_grav}/{len(rows)}")

        results["paramnet"][str(ds)] = {
            "dim": int(dim),
            "per_seed": {str(k): {m: vars(s) for m, s in v.items()} for k, v in per_seed.items()},
            "summary": {
                "alba_experimental": {"mean": m_exp, "std": s_exp, "wins": wins_exp},
                "optuna": {"mean": m_opt, "std": s_opt, "wins": wins_opt},
                "alba_framework_gravity": {"mean": m_grav, "std": s_grav, "wins": wins_grav},
            },
        }
        _save_results(out_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
