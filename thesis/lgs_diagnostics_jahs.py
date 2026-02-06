#!/usr/bin/env python3
"""LGS diagnostics on JAHS-Bench-201 (surrogate).

This script runs alba_framework on a JAHS task while recording, at each LGS step:
- the top-K candidates by predicted score (mu + beta*sigma - penalty)
- the true objective evaluated on the top-M of those candidates (extra budget)
- simple metrics: regret, rank, correlation between predictions and truth

It is meant for *analysis*, not for fair benchmarking.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/lgs_diagnostics_jahs.py --task cifar10 --budget 200 --seed 0 --trace-top-k 64 --eval-top-m 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser(description="LGS diagnostics on JAHS-Bench-201 (surrogate).")
    p.add_argument("--task", default="cifar10", choices=["cifar10", "fashion_mnist", "colorectal_histology"])
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-bins", type=int, default=8)
    p.add_argument("--grid-batch-size", type=int, default=512)
    p.add_argument("--grid-batches", type=int, default=4)
    p.add_argument(
        "--grid-sampling",
        default="grid_random",
        choices=["grid_random", "grid_halton", "halton", "heatmap_ucb"],
    )
    p.add_argument("--grid-jitter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grid-penalty-lambda", type=float, default=0.10)
    p.add_argument("--heatmap-ucb-beta", type=float, default=1.0)
    p.add_argument("--heatmap-ucb-explore-prob", type=float, default=0.25)
    p.add_argument("--heatmap-ucb-temperature", type=float, default=1.0)
    p.add_argument("--novelty-weight", type=float, default=0.4)
    p.add_argument("--global-random-prob", type=float, default=0.0)
    p.add_argument(
        "--split-depth-max",
        type=int,
        default=16,
        help="Maximum split depth (0 disables splitting).",
    )
    p.add_argument(
        "--surrogate",
        default="lgs",
        choices=["lgs", "knn", "knn_lgs", "lgs_catadd", "lgs_heatmap_blend"],
        help="Surrogate used to score grid candidates.",
    )
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-cat-weight", type=float, default=1.0)
    p.add_argument("--knn-sigma-mode", default="var", choices=["var", "dist", "var+dist"])
    p.add_argument("--knn-dist-scale", type=float, default=0.25)
    p.add_argument("--catadd-smoothing", type=float, default=1.0)
    p.add_argument("--heatmap-blend-tau", type=float, default=1e9)
    p.add_argument(
        "--cat-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable categorical sampling step (default: enabled). Disable to study LGS with categoricals fixed.",
    )
    p.add_argument(
        "--cat-stage",
        default="post",
        choices=["post", "pre"],
        help="When categorical sampling is enabled: apply categoricals post-LGS (post) or pick them before grid/LGS (pre).",
    )
    p.add_argument("--trace-top-k", type=int, default=64, help="Store top-K predicted candidates per iter (0 disables).")
    p.add_argument("--eval-top-m", type=int, default=16, help="Evaluate true objective on top-M predicted candidates.")
    p.add_argument("--out", default=None, help="Output JSONL path (default: RESULTS_DIR with timestamp).")
    args = p.parse_args()

    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")
    if args.trace_top_k < 0:
        raise SystemExit("--trace-top-k must be >= 0")
    if args.eval_top_m < 0:
        raise SystemExit("--eval-top-m must be >= 0")
    if args.trace_top_k > 0 and args.eval_top_m > args.trace_top_k:
        raise SystemExit("--eval-top-m cannot exceed --trace-top-k")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"{RESULTS_DIR}/lgs_diag_jahs_{args.task}_b{args.budget}_s{args.seed}_{stamp}.jsonl"

    sys.path.insert(0, "/mnt/workspace")
    sys.path.insert(0, "/mnt/workspace/thesis")

    from benchmark_jahs import JAHSBenchWrapper
    from alba_framework_grid import ALBA as ALBA_FW

    wrapper = JAHSBenchWrapper(task=args.task)
    dim = wrapper.dim

    categorical_dims = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 2),
        (7, 5),
        (8, 5),
        (9, 5),
        (10, 5),
        (11, 5),
        (12, 5),
    ]

    last_trace: Optional[Dict[str, Any]] = None

    def trace_hook(trace: Dict[str, Any]) -> None:
        nonlocal last_trace
        last_trace = trace

    opt = ALBA_FW(
        bounds=[(0.0, 1.0)] * dim,
        categorical_dims=categorical_dims,
        seed=args.seed,
        maximize=False,  # wrapper returns error (minimize)
        total_budget=args.budget,
        split_depth_max=int(args.split_depth_max),
        global_random_prob=float(args.global_random_prob),
        novelty_weight=float(args.novelty_weight),
        grid_bins=args.grid_bins,
        grid_batch_size=args.grid_batch_size,
        grid_batches=args.grid_batches,
        grid_sampling=args.grid_sampling,
        grid_jitter=args.grid_jitter,
        grid_penalty_lambda=args.grid_penalty_lambda,
        heatmap_ucb_beta=float(args.heatmap_ucb_beta),
        heatmap_ucb_explore_prob=float(args.heatmap_ucb_explore_prob),
        heatmap_ucb_temperature=float(args.heatmap_ucb_temperature),
        surrogate=str(args.surrogate),
        knn_k=int(args.knn_k),
        knn_cat_weight=float(args.knn_cat_weight),
        knn_sigma_mode=str(args.knn_sigma_mode),
        knn_dist_scale=float(args.knn_dist_scale),
        catadd_smoothing=float(args.catadd_smoothing),
        heatmap_blend_tau=float(args.heatmap_blend_tau),
        trace_top_k=args.trace_top_k,
        trace_hook=trace_hook,
        categorical_sampling=bool(args.cat_sampler),
        categorical_stage=str(args.cat_stage),
    )

    n_lgs_iters = 0
    n_lgs_chosen_best = 0
    regrets: list[float] = []

    t0 = time.time()
    with open(out_path, "w", buffering=1) as f:
        header = {
            "type": "config",
            "task": args.task,
            "budget": int(args.budget),
            "seed": int(args.seed),
            "grid": {
                "bins": int(args.grid_bins),
                "batch_size": int(args.grid_batch_size),
                "batches": int(args.grid_batches),
                "sampling": str(args.grid_sampling),
                "jitter": bool(args.grid_jitter),
                "penalty_lambda": float(args.grid_penalty_lambda),
                "heatmap_ucb": {
                    "beta": float(args.heatmap_ucb_beta),
                    "explore_prob": float(args.heatmap_ucb_explore_prob),
                    "temperature": float(args.heatmap_ucb_temperature),
                },
            },
            "novelty_weight": float(args.novelty_weight),
            "global_random_prob": float(args.global_random_prob),
            "surrogate": {
                "kind": str(args.surrogate),
                "knn_k": int(args.knn_k),
                "knn_cat_weight": float(args.knn_cat_weight),
                "knn_sigma_mode": str(args.knn_sigma_mode),
                "knn_dist_scale": float(args.knn_dist_scale),
                "catadd_smoothing": float(args.catadd_smoothing),
                "heatmap_blend_tau": float(args.heatmap_blend_tau),
            },
            "categorical_sampling": bool(args.cat_sampler),
            "categorical_stage": str(args.cat_stage),
            "trace_top_k": int(args.trace_top_k),
            "eval_top_m": int(args.eval_top_m),
            "timestamp": stamp,
        }
        f.write(json.dumps(header, default=_json_default) + "\n")

        for it in range(args.budget):
            last_trace = None
            x = np.asarray(opt.ask(), dtype=float)
            err_final = float(wrapper.evaluate_array(x))
            opt.tell(x, err_final)

            rec: Dict[str, Any] = {
                "type": "iter",
                "iter": int(it),
                "x_final": x,
                "err_final": float(err_final),
            }

            # Only available when LGS+grid was actually used.
            if last_trace is not None and "top" in last_trace and args.eval_top_m > 0:
                n_lgs_iters += 1

                top = last_trace["top"]
                X_top = np.asarray(top["x"], dtype=float)
                m = min(int(args.eval_top_m), int(X_top.shape[0]))

                errs = np.zeros(m, dtype=float)
                for j in range(m):
                    errs[j] = float(wrapper.evaluate_array(X_top[j]))

                # "Chosen raw" should coincide with top[0] (highest predicted score).
                err_chosen_raw = float(errs[0])
                err_best = float(np.min(errs))
                regret = float(err_chosen_raw - err_best)
                regrets.append(regret)
                if abs(regret) < 1e-12:
                    n_lgs_chosen_best += 1

                # Correlations vs *internal* score (higher is better).
                y_internal = -errs  # because err is minimized
                mu = np.asarray(top["mu"][:m], dtype=float)
                score = np.asarray(top["score"][:m], dtype=float)

                rec.update(
                    {
                        "lgs_trace": {
                            "x_chosen_raw": np.asarray(last_trace["x_chosen_raw"], dtype=float),
                            "chosen_pred": last_trace.get("chosen_pred", {}),
                            "lgs": last_trace.get("lgs", {}),
                            "n_scored": int(last_trace.get("n_scored", 0)),
                            "eval_top_m": int(m),
                            "err_chosen_raw": float(err_chosen_raw),
                            "err_best_top_m": float(err_best),
                            "regret_top_m": float(regret),
                            "delta_cat_sampler": float(err_final - err_chosen_raw),
                            "corr_mu_y_internal": _corr(mu, y_internal),
                            "corr_score_y_internal": _corr(score, y_internal),
                            "top_m": {
                                "mu": mu,
                                "sigma": np.asarray(top["sigma"][:m], dtype=float),
                                "score": score,
                                "visits": np.asarray(top["visits"][:m], dtype=float),
                                "err": errs,
                            },
                        }
                    }
                )

            f.write(json.dumps(rec, default=_json_default) + "\n")

        summary = {
            "type": "summary",
            "elapsed_s": float(time.time() - t0),
            "n_lgs_iters_with_trace": int(n_lgs_iters),
            "n_lgs_chosen_best_top_m": int(n_lgs_chosen_best),
            "frac_chosen_best_top_m": (float(n_lgs_chosen_best) / float(n_lgs_iters)) if n_lgs_iters > 0 else None,
            "regret_top_m_mean": float(np.mean(regrets)) if regrets else None,
            "regret_top_m_median": float(np.median(regrets)) if regrets else None,
            "regret_top_m_max": float(np.max(regrets)) if regrets else None,
        }
        f.write(json.dumps(summary, default=_json_default) + "\n")

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
