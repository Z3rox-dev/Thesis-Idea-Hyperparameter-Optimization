#!/usr/bin/env python3
"""Synthetic categorical benchmark for ALBA (grid version).

Goal
----
Stress-test categorical handling (thrashing vs. within-key optimization) with a
fully controlled objective where we can compute true regrets and key-level stats.

We create a mixed space:
  - 1 categorical dim with K choices (encoded in [0,1] and discretized)
  - d_cont continuous dims in [0,1]

Objective (minimize):
  f(key, x) = bias[key] + ||x - center[key]||^2

Where:
  - one (or a few) keys are truly better (lower bias)
  - within a key, there is still continuous structure to exploit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from alba_framework_grid.diagnostics import make_jsonl_trace_hooks
from alba_framework_grid.optimizer import ALBA


def _discretize_cat(x_val: float, n_choices: int) -> int:
    n = int(n_choices)
    if n <= 1:
        return 0
    idx = int(np.round(float(x_val) * float(n - 1)))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


@dataclass(frozen=True)
class SynthCatObjective:
    n_choices: int
    d_cont: int
    seed: int
    bias_scale: float = 0.35

    def make(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(self.seed))
        centers = rng.uniform(0.0, 1.0, size=(int(self.n_choices), int(self.d_cont))).astype(float)

        # Biases: a few good keys, many mediocre/bad keys.
        raw = rng.random(int(self.n_choices))
        raw = raw - float(np.min(raw))
        if float(np.max(raw)) > 1e-12:
            raw = raw / float(np.max(raw))
        biases = (float(self.bias_scale) * raw).astype(float)
        # Ensure there is a unique best key with bias=0 for clean diagnostics.
        best_k = int(np.argmin(biases))
        biases = biases - float(biases[best_k])
        return centers, biases


def eval_objective(
    x: np.ndarray,
    *,
    centers: np.ndarray,
    biases: np.ndarray,
    cat_dim: int,
    n_choices: int,
) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    k = _discretize_cat(float(x[int(cat_dim)]), int(n_choices))
    x_cont = np.delete(x, int(cat_dim))
    c = centers[int(k)]
    return float(biases[int(k)] + float(np.sum((x_cont - c) ** 2)))


def run_one(
    *,
    out_path: Path,
    seed: int,
    budget: int,
    n_choices: int,
    d_cont: int,
    centers: np.ndarray,
    biases: np.ndarray,
    categorical_stage: str,
    trace_top_k: int,
) -> Dict[str, Any]:
    dim = 1 + int(d_cont)
    bounds = [(0.0, 1.0)] * dim
    cat_dim = 0

    trace_hook, trace_hook_tell, writer = make_jsonl_trace_hooks(out_path, flush=True)

    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=int(seed),
        total_budget=int(budget),
        categorical_dims=[(cat_dim, int(n_choices))],
        categorical_sampling=True,
        categorical_stage=str(categorical_stage),
        categorical_pre_n=min(8, int(n_choices)),
        trace_top_k=int(trace_top_k),
        trace_hook=trace_hook,
        trace_hook_tell=trace_hook_tell,
    )

    best = float("inf")
    best_x: Optional[np.ndarray] = None
    for _ in range(int(budget)):
        x = opt.ask()
        y = eval_objective(x, centers=centers, biases=biases, cat_dim=cat_dim, n_choices=n_choices)
        opt.tell(x, float(y))
        if float(y) < best or best_x is None:
            best = float(y)
            best_x = np.asarray(x, dtype=float).copy()

    writer.close()
    return {"seed": int(seed), "best": float(best), "best_x": (best_x.tolist() if best_x is not None else None)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="thesis/_tmp_traces_synth_cat", help="Output directory for JSONL traces.")
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--seeds", type=str, default="70,71,72,73")
    p.add_argument("--n-choices", type=int, default=8)
    p.add_argument("--d-cont", type=int, default=6)
    p.add_argument("--obj-seed", type=int, default=0, help="Seed for the synthetic objective definition.")
    p.add_argument("--trace-top-k", type=int, default=64)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = SynthCatObjective(n_choices=int(args.n_choices), d_cont=int(args.d_cont), seed=int(args.obj_seed))
    centers, biases = obj.make()

    variants = [
        ("pre", "pre_enum"),
        ("auto", "auto_ts_masked"),
    ]

    results: List[Dict[str, Any]] = []
    for stage, tag in variants:
        for seed in seeds:
            out_path = out_dir / f"trace_synthcat_fw_{tag}_seed{seed}_b{int(args.budget)}.jsonl"
            r = run_one(
                out_path=out_path,
                seed=seed,
                budget=int(args.budget),
                n_choices=int(args.n_choices),
                d_cont=int(args.d_cont),
                centers=centers,
                biases=biases,
                categorical_stage=stage,
                trace_top_k=int(args.trace_top_k),
            )
            r["variant"] = tag
            r["trace"] = str(out_path)
            results.append(r)

    meta = {
        "n_choices": int(args.n_choices),
        "d_cont": int(args.d_cont),
        "obj_seed": int(args.obj_seed),
        "budget": int(args.budget),
        "seeds": seeds,
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"Wrote {len(results)} traces to {out_dir}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()

