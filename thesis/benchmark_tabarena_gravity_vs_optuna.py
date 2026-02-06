#!/usr/bin/env python3
"""TabArena HPO mini-benchmark: ALBA(Gravity) vs Optuna(TPE).

This script runs a small hyperparameter optimization loop where each trial is a
single AutoGluon model fit executed through TabArena (so we measure the same
`metric_error` TabArena reports).

Default settings are intentionally small / local-friendly.

Run:
  source /mnt/workspace/.venv-tabarena/bin/activate
  python thesis/benchmark_tabarena_gravity_vs_optuna.py --dataset anneal --fold 0 --trials 10 --time-limit 60

Outputs:
  thesis/tabarena_runs/hpo_gravity_vs_optuna/<timestamp>/{results.csv,results.json}
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


def _ensure_thesis_on_path() -> None:
    # Makes `import alba_framework_gravity` work when running from workspace root.
    import sys

    thesis_dir = Path(__file__).resolve().parent
    if str(thesis_dir) not in sys.path:
        sys.path.insert(0, str(thesis_dir))


@contextlib.contextmanager
def _suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield


@dataclass
class TrialRecord:
    method: str
    trial: int
    metric_error: float
    best_so_far: float
    wall_time_s: float
    params: Dict[str, Any]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_ints(spec: str) -> List[int]:
    """Parse a seed spec like "1,2,3" or "80-90" or "80-90,95"."""
    out: List[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a_str, b_str = [x.strip() for x in part.split("-", 1)]
            a = int(a_str)
            b = int(b_str)
            if b < a:
                raise ValueError(f"Invalid range: {part!r}")
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    # preserve order, drop duplicates
    seen = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _parse_csv_list(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _make_lgb_param_space() -> Dict[str, Any]:
    # Keys chosen to match TabArena demo config keys (see tabflow/configs/configs_lightgbm_demo.yaml)
    # Keep it small-ish for local runs.
    return {
        "learning_rate": (1e-3, 0.2, "log"),
        "num_leaves": (4, 256, "int"),
        "feature_fraction": (0.3, 1.0, "float"),
        "min_data_in_leaf": (5, 100, "int"),
        "lambda_l2": (1e-8, 10.0, "log"),
        "extra_trees": [False, True],
    }


def _eval_one_config(
    *,
    runner,
    dataset: str,
    fold: int,
    model_cls,
    model_hyperparameters: Dict[str, Any],
    predictor_root: Path,
    wrapper_fit_kwargs: Dict[str, Any] | None,
    num_bag_folds: int,
    num_bag_sets: int,
    time_limit_s: int,
    method_name: str,
    ignore_cache: bool,
    quiet: bool,
) -> float:
    from tabarena.benchmark.experiment import AGModelBagExperiment
    from tabarena.benchmark.result import BaselineResult

    method_kwargs: Dict[str, Any] = {
        "init_kwargs": {
            # Keep AutoGluon artifacts contained inside our run folder.
            "path": str(predictor_root / method_name),
        },
    }
    if wrapper_fit_kwargs:
        method_kwargs["fit_kwargs"] = dict(wrapper_fit_kwargs)

    experiment = AGModelBagExperiment(
        name=method_name,
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        method_kwargs=method_kwargs,
        num_bag_folds=num_bag_folds,
        num_bag_sets=num_bag_sets,
        time_limit=time_limit_s,
        raise_on_model_failure=True,
    )

    with _suppress_output(enabled=quiet):
        results_lst = runner.run(
            datasets=[dataset],
            folds=[fold],
            methods=[experiment],
            ignore_cache=ignore_cache,
            raise_on_failure=True,
        )

    if not results_lst:
        raise RuntimeError("TabArena returned no results.")

    result_obj = BaselineResult.from_dict(results_lst[0])
    return float(result_obj.result["metric_error"])


def _write_records(df_path: Path, records: Sequence[TrialRecord]) -> None:
    df = pd.DataFrame(
        [
            {
                "method": r.method,
                "trial": r.trial,
                "metric_error": r.metric_error,
                "best_so_far": r.best_so_far,
                "wall_time_s": r.wall_time_s,
                "dataset": r.params.get("__dataset__"),
                "seed": r.params.get("__seed__"),
                "params": json.dumps({k: v for k, v in r.params.items() if not k.startswith("__")}, sort_keys=True),
            }
            for r in records
        ]
    )
    df.to_csv(df_path, index=False)


def _best_table(records: Iterable[TrialRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "dataset": r.params.get("__dataset__"),
                "seed": r.params.get("__seed__"),
                "method": r.method,
                "metric_error": r.metric_error,
                "best_so_far": r.best_so_far,
                "trial": r.trial,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    best = (
        df.sort_values(["dataset", "seed", "method", "best_so_far", "trial"])
        .groupby(["dataset", "seed", "method"], as_index=False)
        .first()
    )
    pivot = best.pivot_table(index=["dataset", "seed"], columns="method", values="best_so_far", aggfunc="min")
    pivot = pivot.reset_index()
    # winner (lower is better)
    method_cols = [c for c in pivot.columns if c not in ("dataset", "seed")]
    if method_cols:
        pivot["winner"] = pivot[method_cols].idxmin(axis=1)
    return pivot


def run_gravity(
    *,
    runner,
    dataset: str,
    fold: int,
    model_cls,
    base_ag_args_fit: Dict[str, Any],
    base_ag_args_ensemble: Dict[str, Any],
    predictor_root: Path,
    wrapper_fit_kwargs: Dict[str, Any] | None,
    trials: int,
    seed: int,
    num_bag_folds: int,
    num_bag_sets: int,
    time_limit_s: int,
    ignore_cache: bool,
    quiet: bool,
) -> List[TrialRecord]:
    _ensure_thesis_on_path()
    from alba_framework_gravity import ALBA

    opt = ALBA(
        param_space=_make_lgb_param_space(),
        maximize=False,
        seed=seed,
        total_budget=trials,
        cube_gravity=True,
    )

    use_cube_gravity = getattr(opt, "_use_cube_gravity", None)
    if use_cube_gravity is not True:
        raise RuntimeError(
            "Gravity is not enabled in alba_framework_gravity (expected _use_cube_gravity=True)."
        )

    history: List[TrialRecord] = []
    best = float("inf")

    for t in range(trials):
        params = opt.ask()
        assert isinstance(params, dict)

        method_name = f"LGB_gravity_{dataset}_t{t:04d}_seed{seed}"

        model_hparams = {
            **params,
            "ag_args_fit": dict(base_ag_args_fit),
            "ag_args_ensemble": dict(base_ag_args_ensemble),
        }

        start = time.time()
        metric_error = _eval_one_config(
            runner=runner,
            dataset=dataset,
            fold=fold,
            model_cls=model_cls,
            model_hyperparameters=model_hparams,
            predictor_root=predictor_root,
            wrapper_fit_kwargs=wrapper_fit_kwargs,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            time_limit_s=time_limit_s,
            method_name=method_name,
            ignore_cache=ignore_cache,
            quiet=quiet,
        )
        wall = time.time() - start

        opt.tell(params, metric_error)

        if metric_error < best:
            best = metric_error

        history.append(
            TrialRecord(
                method="gravity",
                trial=t,
                metric_error=float(metric_error),
                best_so_far=float(best),
                wall_time_s=float(wall),
                params={"__dataset__": dataset, "__seed__": seed, **params},
            )
        )

        print(f"gravity\ttrial={t:03d}\tmetric_error={metric_error:.6f}\tbest={best:.6f}", flush=True)

    return history


def run_optuna(
    *,
    runner,
    dataset: str,
    fold: int,
    model_cls,
    base_ag_args_fit: Dict[str, Any],
    base_ag_args_ensemble: Dict[str, Any],
    predictor_root: Path,
    wrapper_fit_kwargs: Dict[str, Any] | None,
    trials: int,
    seed: int,
    num_bag_folds: int,
    num_bag_sets: int,
    time_limit_s: int,
    ignore_cache: bool,
    quiet: bool,
) -> List[TrialRecord]:
    import optuna

    if quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: List[TrialRecord] = []
    best = float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        }

        method_name = f"LGB_optuna_{dataset}_t{trial.number:04d}_seed{seed}"

        model_hparams = {
            **params,
            "ag_args_fit": dict(base_ag_args_fit),
            "ag_args_ensemble": dict(base_ag_args_ensemble),
        }

        start = time.time()
        metric_error = _eval_one_config(
            runner=runner,
            dataset=dataset,
            fold=fold,
            model_cls=model_cls,
            model_hyperparameters=model_hparams,
            predictor_root=predictor_root,
            wrapper_fit_kwargs=wrapper_fit_kwargs,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            time_limit_s=time_limit_s,
            method_name=method_name,
            ignore_cache=ignore_cache,
            quiet=quiet,
        )
        wall = time.time() - start

        if metric_error < best:
            best = metric_error

        history.append(
            TrialRecord(
                method="optuna",
                trial=int(trial.number),
                metric_error=float(metric_error),
                best_so_far=float(best),
                wall_time_s=float(wall),
                params={"__dataset__": dataset, "__seed__": seed, **params},
            )
        )

        print(f"optuna\ttrial={trial.number:03d}\tmetric_error={metric_error:.6f}\tbest={best:.6f}", flush=True)
        return float(metric_error)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with _suppress_output(enabled=quiet):
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="TabArena: gravity vs optuna (trial-by-trial)")
    parser.add_argument(
        "--datasets",
        type=str,
        default="anneal,credit-g,diabetes",
        help="Comma-separated dataset names (TabArena/OpenML).",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--bag-folds", type=int, default=2)
    parser.add_argument("--bag-sets", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Seed list/range like '0,1,2' or '80-90'.",
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--no-quiet", action="store_true", help="Do not suppress TabArena/AutoGluon logs")
    parser.add_argument(
        "--methods",
        type=str,
        default="gravity,optuna",
        help="Comma-separated list: gravity,optuna",
    )
    args = parser.parse_args()

    quiet = not args.no_quiet

    if args.trials <= 0:
        raise ValueError("--trials must be > 0")

    out_root = (
        Path(args.out_root)
        if args.out_root
        else (Path(__file__).parent / "tabarena_runs" / "hpo_gravity_vs_optuna" / _timestamp())
    )
    out_root.mkdir(parents=True, exist_ok=True)

    expname = str(out_root / "experiments")
    predictor_root = out_root / "AutogluonModels"
    predictor_root.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / "results.csv"
    out_json = out_root / "results.json"
    out_table_csv = out_root / "best_table.csv"

    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from tabarena.benchmark.experiment import ExperimentBatchRunner

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=task_metadata,
        debug_mode=True,
    )

    from autogluon.tabular.models import LGBModel

    base_ag_args_fit = {"num_cpus": 1, "num_gpus": 0}
    base_ag_args_ensemble = {
        # model_random_seed will be overridden per-seed in the loop
        "model_random_seed": 0,
        "vary_seed_across_folds": False,
        # Avoid multiprocessing fold fitting (keeps logs + debugging simpler).
        "fold_fitting_strategy": "sequential_local",
    }

    # This goes directly into TabularPredictor.fit(...).
    # When quiet=True, reduce AutoGluon logging substantially.
    wrapper_fit_kwargs = {"verbosity": 0} if quiet else None

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_ints(args.seeds)

    all_records: List[TrialRecord] = []

    # Run loop
    for dataset in datasets:
        for seed in seeds:
            print(f"\n=== dataset={dataset} fold={args.fold} seed={seed} trials={args.trials} ===", flush=True)
            base_ag_args_ensemble_local = dict(base_ag_args_ensemble)
            base_ag_args_ensemble_local["model_random_seed"] = int(seed)

            if "gravity" in methods:
                all_records.extend(
                    run_gravity(
                        runner=runner,
                        dataset=dataset,
                        fold=args.fold,
                        model_cls=LGBModel,
                        base_ag_args_fit=base_ag_args_fit,
                        base_ag_args_ensemble=base_ag_args_ensemble_local,
                        predictor_root=predictor_root,
                        wrapper_fit_kwargs=wrapper_fit_kwargs,
                        trials=args.trials,
                        seed=seed,
                        num_bag_folds=args.bag_folds,
                        num_bag_sets=args.bag_sets,
                        time_limit_s=args.time_limit,
                        ignore_cache=True,
                        quiet=quiet,
                    )
                )

                _write_records(out_csv, all_records)
                out_json.write_text(
                    json.dumps(
                        {
                            "datasets": datasets,
                            "fold": args.fold,
                            "seeds": seeds,
                            "trials": args.trials,
                            "time_limit_s": args.time_limit,
                            "bag_folds": args.bag_folds,
                            "bag_sets": args.bag_sets,
                            "records": [r.__dict__ for r in all_records],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

            if "optuna" in methods:
                all_records.extend(
                    run_optuna(
                        runner=runner,
                        dataset=dataset,
                        fold=args.fold,
                        model_cls=LGBModel,
                        base_ag_args_fit=base_ag_args_fit,
                        base_ag_args_ensemble=base_ag_args_ensemble_local,
                        predictor_root=predictor_root,
                        wrapper_fit_kwargs=wrapper_fit_kwargs,
                        trials=args.trials,
                        seed=seed,
                        num_bag_folds=args.bag_folds,
                        num_bag_sets=args.bag_sets,
                        time_limit_s=args.time_limit,
                        ignore_cache=True,
                        quiet=quiet,
                    )
                )

                _write_records(out_csv, all_records)
                out_json.write_text(
                    json.dumps(
                        {
                            "datasets": datasets,
                            "fold": args.fold,
                            "seeds": seeds,
                            "trials": args.trials,
                            "time_limit_s": args.time_limit,
                            "bag_folds": args.bag_folds,
                            "bag_sets": args.bag_sets,
                            "records": [r.__dict__ for r in all_records],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

    # Final aggregate table
    pivot = _best_table(all_records)
    if not pivot.empty:
        pivot.to_csv(out_table_csv, index=False)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 140):
            print("\nBest metric_error per dataset/seed (lower is better):")
            print(pivot)

        if "winner" in pivot.columns:
            win_counts = pivot["winner"].value_counts().to_dict()
            print("\nWin counts (by dataset/seed):")
            for k, v in win_counts.items():
                print(f"{k}\twins={v}", flush=True)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_table_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
