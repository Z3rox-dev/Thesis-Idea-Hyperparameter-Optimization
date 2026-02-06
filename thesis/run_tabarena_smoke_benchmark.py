#!/usr/bin/env python3
"""Tiny TabArena benchmark smoke test.

Runs a single lightweight AutoGluon model (LinearModel) on a single TabArena dataset
(anneal) for fold 0, with a very small time limit and minimal bagging.

Intended to validate that TabArena + AutoGluon are installed and the benchmark
pipeline works end-to-end.

Run:
  source /mnt/workspace/.venv-tabarena/bin/activate
  python thesis/run_tabarena_smoke_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def main() -> int:
    out_root = Path(__file__).parent / "tabarena_runs" / "smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    expname = str(out_root / "experiments")
    eval_dir = out_root / "eval"

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    datasets = ["anneal"]
    folds = [0]

    from autogluon.tabular.models import LinearModel

    methods = [
        AGModelBagExperiment(
            name="LinearModel_demo_BAG2",
            model_cls=LinearModel,
            model_hyperparameters={
                # Friendly local execution (avoid nested parallelism)
                "ag_args_fit": {"num_cpus": 1},
            },
            num_bag_folds=2,
            num_bag_sets=1,
            time_limit=60,
            raise_on_model_failure=True,
        )
    ]

    runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=task_metadata,
        debug_mode=True,
    )

    results_lst = runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=True,
        raise_on_failure=True,
    )

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print("Configs hyperparameters:")
    print(end_to_end.configs_hyperparameters())

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 140):
        print("Model results (head):")
        print(end_to_end_results.model_results.head(50))

    # Also write a small leaderboard csv for convenience
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = end_to_end_results.model_results.copy()
    df.to_csv(eval_dir / "model_results.csv", index=False)
    print(f"Saved: {eval_dir / 'model_results.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
