#!/usr/bin/env python3
"""HPOBench NN-Tabular battery: ALBA Framework vs Optuna(TPE) vs Random vs TuRBO-M.

Runs on all locally available HPOBench TabularBenchmark datasets for model='nn'
(task IDs present on disk), with a fixed evaluation budget per (dataset, seed).

Intended env: conda `py39` (this repo already uses it for JAHS).

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_nn_tabular_battery_alba_optuna_random_turbo.py --budget 400 --seeds 42 43 44

Resume:
  python thesis/benchmark_nn_tabular_battery_alba_optuna_random_turbo.py --results_file thesis/benchmark_results/nn_tabular_b400_....json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# HPOBench (vendored)
sys.path.insert(0, "/mnt/workspace/HPOBench")

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _enable_fast_hpobench_tabular_lookup(benchmark: TabularBenchmark) -> None:
    """Speed up TabularBenchmark lookups without changing results."""
    import pandas as pd
    import types

    df = benchmark.table
    if not hasattr(df, "columns") or "result" not in df.columns:
        return

    idx_cols = [c for c in df.columns if c != "result"]
    indexed = df.set_index(idx_cols, drop=False)
    try:
        indexed = indexed.sort_index()
    except Exception:
        pass

    cache: dict[tuple, object] = {}
    max_cache = 200_000

    def _search_dataframe_fast(self, row_dict, _df_unused):
        key = tuple(row_dict[c] for c in idx_cols)
        hit = cache.get(key)
        if hit is not None:
            return hit

        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise AssertionError(
                    f"The query has resulted into multiple matches. Query={row_dict} matches={len(row)}"
                )
            row = row.iloc[0]
        res = row["result"]

        if len(cache) < max_cache:
            cache[key] = res
        return res

    benchmark._search_dataframe = types.MethodType(_search_dataframe_fast, benchmark)


def _available_hpobench_tabular_task_ids(model: str = "nn") -> List[int]:
    """Return available task_ids for HPOBench TabularBenchmark datasets already present on disk."""
    try:
        import hpobench
    except Exception:
        return []

    base_dir = Path(hpobench.config_file.data_dir) / "TabularData"
    model_dir = base_dir / model
    if not model_dir.exists():
        return []

    task_ids: List[int] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        tid = int(child.name)
        parquet = child / f"{model}_{tid}_data.parquet.gzip"
        meta = child / f"{model}_{tid}_metadata.json"
        if parquet.exists() and meta.exists():
            task_ids.append(tid)

    task_ids.sort()
    return task_ids


def _x_to_config_dict(x: np.ndarray, hp_names: List[str], hp_seqs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Map x in [0,1]^d to a discrete config dict via uniform binning."""
    cfg: Dict[str, Any] = {}
    for i, name in enumerate(hp_names):
        seq = hp_seqs[name]
        k = len(seq)
        if k <= 1:
            cfg[name] = seq[0]
            continue
        # Uniform mapping to categories
        idx = int(np.floor(float(x[i]) * k))
        if idx < 0:
            idx = 0
        elif idx >= k:
            idx = k - 1
        cfg[name] = seq[idx]
    return cfg


def _objective_factory(
    benchmark: TabularBenchmark, cs
) -> Tuple[int, List[str], Dict[str, List[Any]], List[Tuple[int, int]], Any]:
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    categorical_dims: List[Tuple[int, int]] = [(i, len(hp_seqs[name])) for i, name in enumerate(hp_names)]
    max_fidelity = benchmark.get_max_fidelity()

    def objective_x(x01: np.ndarray) -> float:
        cfg = _x_to_config_dict(x01, hp_names, hp_seqs)
        res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
        return float(res["function_value"])

    return dim, hp_names, hp_seqs, categorical_dims, objective_x


def run_random(objective_x, dim: int, budget: int, seed: int) -> Tuple[float, Dict[int, float]]:
    rng = np.random.default_rng(seed)
    best = float("inf")
    cp: Dict[int, float] = {}
    for it in range(budget):
        x = rng.random(dim)
        y = float(objective_x(x))
        best = min(best, y)
        cp[it + 1] = best
    return best, cp


def run_optuna_tpe(benchmark: TabularBenchmark, cs, budget: int, seed: int) -> Tuple[float, Dict[int, float]]:
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    max_fidelity = benchmark.get_max_fidelity()

    best = float("inf")
    cp: Dict[int, float] = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best
        cfg: Dict[str, Any] = {}
        for name in hp_names:
            cfg[name] = trial.suggest_categorical(name, hp_seqs[name])
        res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
        y = float(res["function_value"])
        best = min(best, y)
        cp[len(cp) + 1] = best
        return y

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return float(study.best_value), cp


def run_turbo_m(objective_x, dim: int, budget: int, seed: int, n_trust_regions: int = 5) -> Tuple[float, Dict[int, float]]:
    try:
        from turbo import TurboM, Turbo1
    except Exception as e:
        raise RuntimeError(
            "TuRBO is not installed in this environment. "
            "Install with: pip install gpytorch botorch && pip install git+https://github.com/uber-research/TuRBO.git"
        ) from e

    if budget <= 2:
        raise ValueError(f"TuRBO requires budget > 2, got budget={budget}")

    np.random.seed(seed)

    best_so_far = [float("inf")]
    cp: Dict[int, float] = {}
    eval_count = [0]

    def f(x: np.ndarray) -> float:
        y = float(objective_x(np.asarray(x, dtype=float)))
        eval_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        cp[eval_count[0]] = best_so_far[0]
        return y

    lb = np.zeros(dim)
    ub = np.ones(dim)

    n_init = max(10, 2 * dim)
    n_init = min(n_init, budget - 1)

    n_tr = max(1, int(n_trust_regions))
    max_tr = max(1, (budget - 1) // max(1, n_init))
    n_tr = min(n_tr, max_tr)

    if n_tr >= 2:
        turbo = TurboM(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=budget,
            n_trust_regions=n_tr,
            batch_size=1,
            verbose=False,
            use_ard=True,
            device="cpu",
            dtype="float64",
        )
    else:
        turbo = Turbo1(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=budget,
            batch_size=1,
            verbose=False,
            use_ard=True,
            device="cpu",
            dtype="float64",
        )

    turbo.optimize()
    return float(np.min(turbo.fX)), cp


def run_alba_framework(
    objective_x,
    dim: int,
    budget: int,
    seed: int,
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[float, Dict[int, float]]:
    sys.path.insert(0, "/mnt/workspace/thesis")
    from alba_framework import ALBA

    opt = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=budget,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims or [],
    )

    best = float("inf")
    cp: Dict[int, float] = {}
    for it in range(budget):
        x = opt.ask()
        y = float(objective_x(x))
        opt.tell(x, y)
        best = min(best, y)
        cp[it + 1] = best
    return best, cp


def _is_done(task_bucket: Dict[str, Any], algo: str, seed: int, budget: int) -> bool:
    seed_key = str(seed)
    cp = task_bucket.get(algo, {}).get(seed_key)
    if not cp:
        return False
    return str(budget) in cp or budget in cp


def _save(results_file: str, obj: Dict[str, Any]) -> None:
    tmp = results_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, results_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPOBench NN Tabular battery: ALBA Framework vs Optuna vs Random vs TuRBO"
    )
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--task_ids", type=int, nargs="*", default=None, help="Optional explicit task IDs")
    parser.add_argument("--list_tasks", action="store_true", help="List locally available task IDs and exit")
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--no_fast_lookup", action="store_true")
    parser.add_argument("--turbo_trust_regions", type=int, default=5)
    args = parser.parse_args()

    budget = int(args.budget)
    seeds = list(args.seeds)

    available = _available_hpobench_tabular_task_ids(model="nn")
    if args.list_tasks:
        print("Available HPOBench TabularBenchmark task IDs for model=nn:")
        print(", ".join(map(str, available)) if available else "(none found locally)")
        return

    task_ids = list(args.task_ids) if args.task_ids else available
    if not task_ids:
        raise SystemExit("No task IDs found. Run with --list_tasks to debug.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.results_file or os.path.join(
        RESULTS_DIR,
        f"nn_tabular_alba_fw_optuna_random_turbo_b{budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json",
    )

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {
            "config": {
                "budget": budget,
                "seeds": seeds,
                "task_ids": task_ids,
                "timestamp": timestamp,
                "turbo_trust_regions": int(args.turbo_trust_regions),
            },
            "hpobench": {},
        }
        _save(results_file, results)

    results.setdefault("hpobench", {})

    print("=" * 80)
    print("NN TABULAR BATTERY: ALBA Framework vs Optuna(TPE) vs Random vs TuRBO-M")
    print(f"Budget: {budget} | Seeds: {seeds}")
    print(f"Task IDs: {task_ids}")
    print(f"Results: {results_file}")
    print("=" * 80)

    for task_id in task_ids:
        task_key = str(task_id)
        task_bucket = results["hpobench"].setdefault(task_key, {})
        task_bucket.setdefault("alba_fw", {})
        task_bucket.setdefault("optuna_tpe", {})
        task_bucket.setdefault("random", {})
        task_bucket.setdefault("turbo_m", {})

        # Create benchmark
        benchmark = TabularBenchmark(model="nn", task_id=int(task_id))
        if not args.no_fast_lookup:
            _enable_fast_hpobench_tabular_lookup(benchmark)
        cs = benchmark.get_configuration_space()
        dim, hp_names, hp_seqs, categorical_dims, objective_x = _objective_factory(benchmark, cs)
        task_bucket.setdefault("dim", dim)
        task_bucket.setdefault("hp_names", hp_names)

        print(f"\n>>> Task {task_id} (dim={dim})")

        for seed in seeds:
            if all(_is_done(task_bucket, a, seed, budget) for a in ["alba_fw", "optuna_tpe", "random", "turbo_m"]):
                print(f"  Seed {seed}: SKIP (done)")
                continue

            print(f"  Seed {seed}:", end=" ", flush=True)
            t0 = time.time()

            # Random
            if not _is_done(task_bucket, "random", seed, budget):
                best_r, cp_r = run_random(objective_x, dim, budget, seed)
                task_bucket["random"][str(seed)] = {str(k): float(v) for k, v in cp_r.items() if k == budget or str(k) == str(budget) or k in cp_r}
                task_bucket["random"][str(seed)][str(budget)] = float(best_r)
                _save(results_file, results)
            else:
                best_r = float(task_bucket["random"][str(seed)].get(str(budget)))

            # Optuna
            if not _is_done(task_bucket, "optuna_tpe", seed, budget):
                best_o, cp_o = run_optuna_tpe(benchmark, cs, budget, seed)
                task_bucket["optuna_tpe"][str(seed)] = {str(budget): float(best_o)}
                _save(results_file, results)
            else:
                best_o = float(task_bucket["optuna_tpe"][str(seed)].get(str(budget)))

            # TuRBO
            if not _is_done(task_bucket, "turbo_m", seed, budget):
                best_t, cp_t = run_turbo_m(objective_x, dim, budget, seed, n_trust_regions=int(args.turbo_trust_regions))
                task_bucket["turbo_m"][str(seed)] = {str(budget): float(best_t)}
                _save(results_file, results)
            else:
                best_t = float(task_bucket["turbo_m"][str(seed)].get(str(budget)))

            # ALBA
            if not _is_done(task_bucket, "alba_fw", seed, budget):
                best_a, cp_a = run_alba_framework(objective_x, dim, budget, seed, categorical_dims=categorical_dims)
                task_bucket["alba_fw"][str(seed)] = {str(budget): float(best_a)}
                _save(results_file, results)
            else:
                best_a = float(task_bucket["alba_fw"][str(seed)].get(str(budget)))

            elapsed = time.time() - t0
            print(f"R={best_r:.6f} O={best_o:.6f} T={best_t:.6f} A={best_a:.6f} ({elapsed:.1f}s)")

    print("\nDone.")
    print(f"Saved: {results_file}")


if __name__ == "__main__":
    main()
