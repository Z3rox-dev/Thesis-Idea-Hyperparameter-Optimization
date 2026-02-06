#!/usr/bin/env python3
"""
Diagnostics for alba_framework_gravity: warp + cube_gravity (momentum).

Goal
----
Produce under-the-hood metrics (JSONL) + summary (JSON) to assess whether:
- the per-leaf warp is learning a sensible metric
- cube-level gravity + momentum is stable and helpful

This script supports:
- synthetic functions (always available)
- optional JAHS + ParamNet (requires conda env `py39`, see benchmark scripts)

It does NOT require modifying the optimizer implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_gravity import ALBA as ALBA_GRAV  # noqa: E402


try:
    from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402
except Exception:
    ALBA_EXP = None  # type: ignore


try:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.ERROR)
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


# ---------------------------------------------------------------------------
# Synthetic functions (import from ParamSpace.py if present, else fallback)
# ---------------------------------------------------------------------------

try:
    from ParamSpace import sphere, rosenbrock  # type: ignore
except Exception:

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def rosenbrock(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


SYN_FUNCS: Dict[str, Tuple[Callable[[np.ndarray], float], Tuple[float, float]]] = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _save_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _parse_csv(arg: str) -> List[str]:
    return [x.strip() for x in str(arg).split(",") if x.strip()]


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


@contextmanager
def _warp_disabled() -> Iterator[None]:
    """Disable warp in alba_framework_gravity by monkeypatching Cube methods."""
    from alba_framework_gravity import cube as cube_mod  # local import to avoid early import issues

    old_get = cube_mod.Cube.get_warp_multipliers
    old_update = cube_mod.Cube.update_warp

    def get_ones(self: Any) -> np.ndarray:
        return np.ones(len(getattr(self, "bounds", [])), dtype=float)

    def noop(self: Any, x: np.ndarray, score: float) -> None:
        return None

    cube_mod.Cube.get_warp_multipliers = get_ones  # type: ignore[assignment]
    cube_mod.Cube.update_warp = noop  # type: ignore[assignment]
    try:
        yield
    finally:
        cube_mod.Cube.get_warp_multipliers = old_get  # type: ignore[assignment]
        cube_mod.Cube.update_warp = old_update  # type: ignore[assignment]


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


def _safe_np_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "mean": float("nan")}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    best: float
    wall_s: float
    unique_keys: int = 0
    switch_rate: float = 0.0


def _run_optimizer(
    *,
    method_name: str,
    make_opt: Callable[[], Any],
    objective: Callable[[np.ndarray], float],
    budget: int,
    trace_path: str,
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
) -> RunResult:
    opt = make_opt()

    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    f = open(trace_path, "w")

    best = float("inf")
    keys: List[Tuple[int, ...]] = []
    cat_dims = categorical_dims or []

    t0 = time.time()
    for it in range(int(budget)):
        x = opt.ask()
        x_arr = np.asarray(x, dtype=float)
        y = float(objective(x_arr))
        opt.tell(x, y)
        best = min(best, y)

        rec: Dict[str, Any] = {
            "method": str(method_name),
            "iter": int(it),
            "y": float(y),
            "best": float(best),
        }

        # Phase / stagnation (available in ALBA variants)
        try:
            rec["stagnation"] = int(getattr(opt, "stagnation"))
            rec["exploration_budget"] = int(getattr(opt, "exploration_budget"))
            rec["local_search_budget"] = int(getattr(opt, "local_search_budget"))
            rec["phase"] = "exploration" if it < int(getattr(opt, "exploration_budget")) else "local_search"
        except Exception:
            pass

        # Leaf info + warp/gravity metrics for alba_framework_gravity.
        if method_name.startswith("grav"):
            try:
                leaf = getattr(opt, "_last_cube", None)
                if leaf is None and hasattr(opt, "_find_containing_leaf"):
                    leaf = opt._find_containing_leaf(x_arr)
                if leaf is not None:
                    rec["leaf"] = {
                        "id": int(id(leaf)),
                        "depth": int(getattr(leaf, "depth", -1)),
                        "n_trials": int(getattr(leaf, "n_trials", 0)),
                        "good_ratio": float(getattr(leaf, "good_ratio")()) if hasattr(leaf, "good_ratio") else None,
                        "warp_updates": int(getattr(leaf, "warp_updates", 0)),
                    }

                    # Warp multipliers summary.
                    if hasattr(leaf, "get_warp_multipliers"):
                        w = np.asarray(leaf.get_warp_multipliers(), dtype=float)
                        if w.shape == (x_arr.shape[0],):
                            rec["warp"] = {
                                "w_min": float(np.min(w)),
                                "w_med": float(np.median(w)),
                                "w_max": float(np.max(w)),
                            }
            except Exception:
                pass

            # Gravity field / velocity summary.
            try:
                cube_grav = getattr(opt, "_cube_gravity", None)
                leaves = getattr(opt, "leaves", None)
                if cube_grav is not None and leaves:
                    acc = cube_grav.get_drift_vector(x_arr, leaves, normalize=False)
                    acc = np.asarray(acc, dtype=float)
                    rec["gravity"] = {"acc_norm": float(np.linalg.norm(acc))}

                    # Continuous-only norm (ignore categorical dims if known).
                    cat_set = {int(i) for i, _ in getattr(getattr(opt, "_cat_sampler", None), "categorical_dims", [])}
                    cont_idx = [i for i in range(x_arr.shape[0]) if i not in cat_set]
                    if cont_idx:
                        rec["gravity"]["acc_norm_cont"] = float(np.linalg.norm(acc[cont_idx]))

                vmap = getattr(opt, "_gravity_velocity", None)
                if isinstance(vmap, dict) and "leaf" in rec:
                    v = vmap.get(int(rec["leaf"]["id"]))
                    if v is not None:
                        v = np.asarray(v, dtype=float)
                        rec.setdefault("gravity", {})
                        rec["gravity"]["v_norm"] = float(np.linalg.norm(v))
                        cat_set = {int(i) for i, _ in getattr(getattr(opt, "_cat_sampler", None), "categorical_dims", [])}
                        cont_idx = [i for i in range(v.shape[0]) if i not in cat_set]
                        if cont_idx:
                            rec["gravity"]["v_norm_cont"] = float(np.linalg.norm(v[cont_idx]))
            except Exception:
                pass

        if cat_dims:
            key = _cat_key_from_x(x_arr, categorical_dims=cat_dims)
            keys.append(key)
            rec["cat_key"] = list(key)

        f.write(json.dumps(rec) + "\n")

    wall = float(time.time() - t0)
    f.close()

    if not keys:
        return RunResult(best=float(best), wall_s=wall)

    unique = len(set(keys))
    switches = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    switch_rate = float(switches) / float(max(1, len(keys) - 1))
    return RunResult(best=float(best), wall_s=wall, unique_keys=unique, switch_rate=switch_rate)


def _run_optuna(
    *,
    objective: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    seed: int,
    budget: int,
    trace_path: str,
) -> RunResult:
    if optuna is None:
        return RunResult(best=float("inf"), wall_s=0.0)

    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)

    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    f = open(trace_path, "w")

    best = float("inf")
    t0 = time.time()

    def obj(trial: Any) -> float:
        nonlocal best
        x = np.asarray([trial.suggest_float(f"x{i}", float(lo[i]), float(hi[i])) for i in range(int(dim))])
        y = float(objective(x))
        best = min(best, y)
        rec = {"method": "optuna", "iter": int(trial.number), "y": float(y), "best": float(best)}
        f.write(json.dumps(rec) + "\n")
        return y

    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(obj, n_trials=int(budget), show_progress_bar=False)
    wall = float(time.time() - t0)
    f.close()
    return RunResult(best=float(best), wall_s=wall)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def _suite_synthetic(
    *,
    out_dir: str,
    funcs: Sequence[str],
    dims: Sequence[int],
    seeds: Sequence[int],
    budget: int,
) -> Dict[str, Any]:
    tasks: Dict[str, Any] = {}

    for func_name in funcs:
        if func_name not in SYN_FUNCS:
            raise ValueError(f"Unknown synthetic func: {func_name}")
        func, br = SYN_FUNCS[func_name]
        for dim in dims:
            task_name = f"{func_name}_{int(dim)}D"
            bounds = [br] * int(dim)

            per_seed: Dict[str, Any] = {}
            for seed in seeds:
                seed = int(seed)
                run_dir = os.path.join(out_dir, "traces", "synthetic", task_name, f"seed{seed}")
                os.makedirs(run_dir, exist_ok=True)

                def make_grav(cube_gravity: bool) -> Any:
                    return ALBA_GRAV(
                        bounds=bounds,
                        maximize=False,
                        seed=seed,
                        total_budget=int(budget),
                        split_depth_max=16,
                        global_random_prob=0.05,
                        stagnation_threshold=50,
                        cube_gravity=bool(cube_gravity),
                    )

                # Baselines (warp on/off) and gravity (warp on/off).
                with _warp_disabled():
                    r_w0_g0 = _run_optimizer(
                        method_name="grav_warp0_grav0",
                        make_opt=lambda: make_grav(False),
                        objective=lambda x: float(func(np.asarray(x, dtype=float))),
                        budget=budget,
                        trace_path=os.path.join(run_dir, "grav_warp0_grav0.jsonl"),
                    )
                    r_w0_g1 = _run_optimizer(
                        method_name="grav_warp0_grav1",
                        make_opt=lambda: make_grav(True),
                        objective=lambda x: float(func(np.asarray(x, dtype=float))),
                        budget=budget,
                        trace_path=os.path.join(run_dir, "grav_warp0_grav1.jsonl"),
                    )

                r_w1_g0 = _run_optimizer(
                    method_name="grav_warp1_grav0",
                    make_opt=lambda: make_grav(False),
                    objective=lambda x: float(func(np.asarray(x, dtype=float))),
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp1_grav0.jsonl"),
                )
                r_w1_g1 = _run_optimizer(
                    method_name="grav_warp1_grav1",
                    make_opt=lambda: make_grav(True),
                    objective=lambda x: float(func(np.asarray(x, dtype=float))),
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp1_grav1.jsonl"),
                )

                per_seed[str(seed)] = {
                    "grav_warp0_grav0": r_w0_g0.__dict__,
                    "grav_warp0_grav1": r_w0_g1.__dict__,
                    "grav_warp1_grav0": r_w1_g0.__dict__,
                    "grav_warp1_grav1": r_w1_g1.__dict__,
                }

            tasks[task_name] = {"bounds": br, "per_seed": per_seed}

    return tasks


def _suite_jahs_paramnet(
    *,
    out_dir: str,
    seeds: Sequence[int],
    budget: int,
    jahs_tasks: Sequence[str],
    jahs_nepochs: int,
    paramnet_datasets: Sequence[str],
    paramnet_step: int,
) -> Dict[str, Any]:
    # Import objective builders from the existing benchmark script (py39 expected).
    from benchmark_gravity_gravjump_jahs_paramnet import _make_jahs_objective, _make_paramnet_objective

    out: Dict[str, Any] = {"jahs": {}, "paramnet": {}}

    # JAHS: 13 dims, first 2 continuous, the rest discrete.
    jahs_bounds = [(0.0, 1.0)] * 13
    jahs_cat_dims = [
        (2, 3),  # N
        (3, 3),  # W
        (4, 3),  # Resolution
        (5, 3),  # Activation
        (6, 2),  # TrivialAugment
        (7, 5),  # Op1
        (8, 5),  # Op2
        (9, 5),  # Op3
        (10, 5),  # Op4
        (11, 5),  # Op5
        (12, 5),  # Op6
    ]

    # --------------------------
    # JAHS tasks
    # --------------------------
    for task in jahs_tasks:
        objective = _make_jahs_objective(str(task), nepochs=int(jahs_nepochs))
        per_seed: Dict[str, Any] = {}

        for seed in seeds:
            seed = int(seed)
            run_dir = os.path.join(out_dir, "traces", "jahs", str(task), f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            def make_grav(cube_gravity: bool) -> Any:
                return ALBA_GRAV(
                    bounds=jahs_bounds,
                    maximize=False,
                    seed=seed,
                    total_budget=int(budget),
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    categorical_dims=jahs_cat_dims,
                    cube_gravity=bool(cube_gravity),
                )

            with _warp_disabled():
                r_w0_g0 = _run_optimizer(
                    method_name="grav_warp0_grav0",
                    make_opt=lambda: make_grav(False),
                    objective=objective,
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp0_grav0.jsonl"),
                    categorical_dims=jahs_cat_dims,
                )
                r_w0_g1 = _run_optimizer(
                    method_name="grav_warp0_grav1",
                    make_opt=lambda: make_grav(True),
                    objective=objective,
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp0_grav1.jsonl"),
                    categorical_dims=jahs_cat_dims,
                )

            r_w1_g0 = _run_optimizer(
                method_name="grav_warp1_grav0",
                make_opt=lambda: make_grav(False),
                objective=objective,
                budget=budget,
                trace_path=os.path.join(run_dir, "grav_warp1_grav0.jsonl"),
                categorical_dims=jahs_cat_dims,
            )
            r_w1_g1 = _run_optimizer(
                method_name="grav_warp1_grav1",
                make_opt=lambda: make_grav(True),
                objective=objective,
                budget=budget,
                trace_path=os.path.join(run_dir, "grav_warp1_grav1.jsonl"),
                categorical_dims=jahs_cat_dims,
            )

            per_seed[str(seed)] = {
                "grav_warp0_grav0": r_w0_g0.__dict__,
                "grav_warp0_grav1": r_w0_g1.__dict__,
                "grav_warp1_grav0": r_w1_g0.__dict__,
                "grav_warp1_grav1": r_w1_g1.__dict__,
            }

        out["jahs"][str(task)] = {"per_seed": per_seed}

    # --------------------------
    # ParamNet datasets
    # --------------------------
    for ds in paramnet_datasets:
        objective, dim = _make_paramnet_objective(str(ds), step=int(paramnet_step))
        bounds = [(0.0, 1.0)] * int(dim)
        per_seed = {}

        for seed in seeds:
            seed = int(seed)
            run_dir = os.path.join(out_dir, "traces", "paramnet", str(ds), f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            def make_grav(cube_gravity: bool) -> Any:
                return ALBA_GRAV(
                    bounds=bounds,
                    maximize=False,
                    seed=seed,
                    total_budget=int(budget),
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    cube_gravity=bool(cube_gravity),
                )

            with _warp_disabled():
                r_w0_g0 = _run_optimizer(
                    method_name="grav_warp0_grav0",
                    make_opt=lambda: make_grav(False),
                    objective=objective,
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp0_grav0.jsonl"),
                )
                r_w0_g1 = _run_optimizer(
                    method_name="grav_warp0_grav1",
                    make_opt=lambda: make_grav(True),
                    objective=objective,
                    budget=budget,
                    trace_path=os.path.join(run_dir, "grav_warp0_grav1.jsonl"),
                )

            r_w1_g0 = _run_optimizer(
                method_name="grav_warp1_grav0",
                make_opt=lambda: make_grav(False),
                objective=objective,
                budget=budget,
                trace_path=os.path.join(run_dir, "grav_warp1_grav0.jsonl"),
            )
            r_w1_g1 = _run_optimizer(
                method_name="grav_warp1_grav1",
                make_opt=lambda: make_grav(True),
                objective=objective,
                budget=budget,
                trace_path=os.path.join(run_dir, "grav_warp1_grav1.jsonl"),
            )

            per_seed[str(seed)] = {
                "grav_warp0_grav0": r_w0_g0.__dict__,
                "grav_warp0_grav1": r_w0_g1.__dict__,
                "grav_warp1_grav0": r_w1_g0.__dict__,
                "grav_warp1_grav1": r_w1_g1.__dict__,
            }

        out["paramnet"][str(ds)] = {"dim": int(dim), "per_seed": per_seed}

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnostics for alba_framework_gravity (warp + gravity)")
    p.add_argument("--out-dir", type=str, default="/mnt/workspace/thesis/benchmark_results/gravity_diagnostics")
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seeds", type=str, default="70-73")
    p.add_argument("--suite", type=str, default="synthetic", choices=["synthetic", "jahs_paramnet"])
    p.add_argument("--funcs", type=str, default="sphere,rosenbrock")
    p.add_argument("--dims", type=str, default="4,15")
    p.add_argument("--jahs-tasks", type=str, default="cifar10,colorectal_histology")
    p.add_argument("--jahs-nepochs", type=int, default=200)
    p.add_argument("--paramnet-datasets", type=str, default="letter,mnist")
    p.add_argument("--paramnet-step", type=int, default=50)
    args = p.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"summary_{args.suite}_b{int(args.budget)}_{ts}.json")

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")

    summary: Dict[str, Any] = {
        "suite": str(args.suite),
        "budget": int(args.budget),
        "seeds": seeds,
        "timestamp": ts,
        "out_dir": out_dir,
        "results": {},
    }

    if args.suite == "synthetic":
        funcs = _parse_csv(args.funcs)
        dims = [int(x) for x in _parse_csv(args.dims)]
        summary["results"]["synthetic"] = _suite_synthetic(
            out_dir=out_dir,
            funcs=funcs,
            dims=dims,
            seeds=seeds,
            budget=int(args.budget),
        )
    else:
        summary["results"]["jahs_paramnet"] = _suite_jahs_paramnet(
            out_dir=out_dir,
            seeds=seeds,
            budget=int(args.budget),
            jahs_tasks=_parse_csv(args.jahs_tasks),
            jahs_nepochs=int(args.jahs_nepochs),
            paramnet_datasets=_parse_csv(args.paramnet_datasets),
            paramnet_step=int(args.paramnet_step),
        )

    _save_json(summary_path, summary)
    print(f"✓ Wrote summary: {summary_path}")
    print(f"✓ Traces under: {os.path.join(out_dir, 'traces')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
