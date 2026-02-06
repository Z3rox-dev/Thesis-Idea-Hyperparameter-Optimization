#!/usr/bin/env python3
"""LGS diagnostics on HPOBench ParamNet surrogate (steps fidelity).

Runs alba_framework in array mode ([0,1]^d) on a ParamNet dataset while recording,
at each LGS step:
- top-K candidates by predicted score (mu + beta*sigma - penalty)
- true objective on top-M candidates (extra budget)
- simple metrics: regret, rank, correlation between predictions and truth

This is meant for *analysis*, not for fair benchmarking.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/lgs_diagnostics_paramnet.py --dataset adult --budget 200 --seed 0 --step 50 \
      --trace-top-k 64 --eval-top-m 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
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


def _x01_to_cs_config(x01: np.ndarray, cs) -> Any:
    """Map x in [0,1]^d to a ConfigSpace Configuration, respecting log-scale."""
    import ConfigSpace as CS

    x01 = np.asarray(x01, dtype=float)
    hps = cs.get_hyperparameters()
    if len(x01) != len(hps):
        raise ValueError(f"Expected x of length {len(hps)}, got {len(x01)}")

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

    return CS.Configuration(cs, values=values)


def main() -> int:
    p = argparse.ArgumentParser(description="LGS diagnostics on HPOBench ParamNet surrogate.")
    p.add_argument("--dataset", default="adult", choices=["adult", "higgs", "letter", "mnist", "optdigits", "poker"])
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=50)
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
    p.add_argument("--trace-top-k", type=int, default=64)
    p.add_argument("--eval-top-m", type=int, default=16)
    p.add_argument("--out", default=None, help="Output JSONL path (default: RESULTS_DIR with timestamp).")
    args = p.parse_args()

    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")
    if args.step < 1:
        raise SystemExit("--step must be >= 1")
    if args.trace_top_k < 0:
        raise SystemExit("--trace-top-k must be >= 0")
    if args.eval_top_m < 0:
        raise SystemExit("--eval-top-m must be >= 0")
    if args.trace_top_k > 0 and args.eval_top_m > args.trace_top_k:
        raise SystemExit("--eval-top-m cannot exceed --trace-top-k")

    # ParamNet surrogates shipped with HPOBench are pickled with older scikit-learn
    # module paths (e.g. sklearn.ensemble.forest). Add an import alias so unpickling
    # works with modern scikit-learn (same approach as thesis/benchmark_paramnet_*).
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

    # Compat for older deps that use np.float/np.int/np.bool without triggering FutureWarning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.bool.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.int.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.float.*")

        _np_dict = np.__dict__
        if "float" not in _np_dict:
            setattr(np, "float", float)  # type: ignore[attr-defined]
        if "int" not in _np_dict:
            setattr(np, "int", int)  # type: ignore[attr-defined]
        if "bool" not in _np_dict:
            setattr(np, "bool", bool)  # type: ignore[attr-defined]

    from hpobench.benchmarks.surrogates.paramnet_benchmark import (  # type: ignore[import]
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

    BenchCls = bench_map[args.dataset]
    bench = BenchCls()
    cs = bench.get_configuration_space()
    dim = len(cs.get_hyperparameters())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (
        f"{RESULTS_DIR}/lgs_diag_paramnet_{args.dataset}_step{args.step}_"
        f"b{args.budget}_s{args.seed}_{stamp}.jsonl"
    )

    sys.path.insert(0, "/mnt/workspace/thesis")
    from alba_framework_grid import ALBA as ALBA_FW

    last_trace: Optional[Dict[str, Any]] = None

    def trace_hook(trace: Dict[str, Any]) -> None:
        nonlocal last_trace
        last_trace = trace

    opt = ALBA_FW(
        bounds=[(0.0, 1.0)] * dim,
        seed=args.seed,
        maximize=False,  # minimize loss
        total_budget=args.budget,
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
    )

    n_lgs_iters = 0
    n_lgs_chosen_best = 0
    regrets: list[float] = []

    t0 = time.time()
    with open(out_path, "w", buffering=1) as f:
        header = {
            "type": "config",
            "dataset": args.dataset,
            "dim": int(dim),
            "step": int(args.step),
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
            "trace_top_k": int(args.trace_top_k),
            "eval_top_m": int(args.eval_top_m),
            "timestamp": stamp,
        }
        f.write(json.dumps(header, default=_json_default) + "\n")

        # Simple memoization to reduce repeated surrogate calls for identical configs.
        cache: Dict[str, float] = {}

        def eval_x(x01: np.ndarray) -> float:
            cfg = _x01_to_cs_config(x01, cs)
            key = str(cfg)
            if key in cache:
                return cache[key]
            res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
            y = float(res["function_value"])
            cache[key] = y
            return y

        best = float("inf")
        for it in range(args.budget):
            last_trace = None
            x = np.asarray(opt.ask_array(), dtype=float)
            y = eval_x(x)
            opt.tell(x, y)
            best = min(best, float(y))

            rec: Dict[str, Any] = {
                "type": "iter",
                "iter": int(it),
                "x_final": x,
                "loss_final": float(y),
                "best_loss_so_far": float(best),
            }

            if last_trace is not None and "top" in last_trace and args.eval_top_m > 0:
                n_lgs_iters += 1
                top = last_trace["top"]
                X_top = np.asarray(top["x"], dtype=float)
                m = min(int(args.eval_top_m), int(X_top.shape[0]))

                losses = np.zeros(m, dtype=float)
                for j in range(m):
                    losses[j] = eval_x(X_top[j])

                loss_chosen_raw = float(losses[0])
                loss_best_top_m = float(np.min(losses))
                regret = float(loss_chosen_raw - loss_best_top_m)
                regrets.append(regret)
                if abs(regret) < 1e-12:
                    n_lgs_chosen_best += 1

                # correlations vs internal score (higher is better)
                y_internal = -losses
                mu = np.asarray(top["mu"][:m], dtype=float)
                score = np.asarray(top["score"][:m], dtype=float)

                rec["lgs_trace"] = {
                    "x_chosen_raw": np.asarray(last_trace["x_chosen_raw"], dtype=float),
                    "chosen_pred": last_trace.get("chosen_pred", {}),
                    "lgs": last_trace.get("lgs", {}),
                    "n_scored": int(last_trace.get("n_scored", 0)),
                    "eval_top_m": int(m),
                    "loss_chosen_raw": float(loss_chosen_raw),
                    "loss_best_top_m": float(loss_best_top_m),
                    "regret_top_m": float(regret),
                    "delta_sampler": float(y - loss_chosen_raw),  # should be ~0 (no categoricals)
                    "corr_mu_y_internal": _corr(mu, y_internal),
                    "corr_score_y_internal": _corr(score, y_internal),
                    "top_m": {
                        "mu": mu,
                        "sigma": np.asarray(top["sigma"][:m], dtype=float),
                        "score": score,
                        "visits": np.asarray(top["visits"][:m], dtype=float),
                        "loss": losses,
                    },
                }

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
