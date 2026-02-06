#!/usr/bin/env python3
"""Summarize TabArena HPO run outputs produced by benchmark_tabarena_gravity_vs_optuna.py.

Usage:
  python thesis/summarize_tabarena_hpo_results.py thesis/tabarena_runs/hpo_gravity_vs_optuna/<timestamp>/results.csv

It prints a per-(dataset,seed) table of best_so_far per method and win counts.
"""

from __future__ import annotations

import argparse
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=str, help="Path to results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("No rows.")
        return 0

    # Expect columns: method, trial, metric_error, best_so_far, wall_time_s, dataset, seed
    need = {"method", "trial", "best_so_far", "dataset", "seed"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    best = (
        df.sort_values(["dataset", "seed", "method", "best_so_far", "trial"])
        .groupby(["dataset", "seed", "method"], as_index=False)
        .first()
    )

    pivot = best.pivot_table(index=["dataset", "seed"], columns="method", values="best_so_far", aggfunc="min")
    pivot = pivot.reset_index()
    method_cols = [c for c in pivot.columns if c not in ("dataset", "seed")]
    if method_cols:
        pivot["winner"] = pivot[method_cols].idxmin(axis=1)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(pivot)

    if "winner" in pivot.columns:
        print("\nWin counts:")
        vc = pivot["winner"].value_counts()
        for k, v in vc.items():
            print(f"{k}\twins={int(v)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
