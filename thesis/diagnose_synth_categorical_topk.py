#!/usr/bin/env python3
"""Diagnose categorical regret decomposition from ALBA JSONL traces.

This is tailored to traces produced by `thesis/benchmark_synth_categorical.py`.

We compute (per ask event with top-K candidates in trace):
  - regret_total = y(chosen) - min(y(topK))
  - regret_cat   = min_y_in_chosen_key(topK) - min_y(topK)
  - regret_cont  = y(chosen) - min_y_in_chosen_key(topK)
  - hit@1_key    = 1{argmin_y(topK).key == chosen_key}
  - switch_rate  based on chosen_key changes across asks

All regrets are in raw objective units (minimize).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


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
        raw = rng.random(int(self.n_choices))
        raw = raw - float(np.min(raw))
        if float(np.max(raw)) > 1e-12:
            raw = raw / float(np.max(raw))
        biases = (float(self.bias_scale) * raw).astype(float)
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
    return float(biases[int(k)] + float(np.sum((x_cont - centers[int(k)]) ** 2)))


def _pct(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def iter_events(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def diagnose_trace(
    path: Path,
    *,
    centers: np.ndarray,
    biases: np.ndarray,
    cat_dim: int,
    n_choices: int,
) -> Dict[str, Any]:
    regrets = []
    regrets_cat = []
    regrets_cont = []
    hit1_key = []
    chosen_keys = []

    n_ask = 0
    n_ask_with_top = 0

    for ev in iter_events(path):
        if ev.get("event") != "ask":
            continue
        n_ask += 1

        # chosen point (final, after categorical stage)
        x_chosen = ev.get("x_final") or ev.get("x_chosen_raw")
        if x_chosen is None:
            continue
        x_chosen = np.asarray(x_chosen, dtype=float)
        key_chosen = int(_discretize_cat(float(x_chosen[int(cat_dim)]), int(n_choices)))
        chosen_keys.append(key_chosen)
        y_chosen = eval_objective(x_chosen, centers=centers, biases=biases, cat_dim=cat_dim, n_choices=n_choices)

        top = ev.get("top")
        if not isinstance(top, dict) or "x" not in top:
            continue
        X_top = np.asarray(top.get("x"), dtype=float)
        if X_top.ndim != 2 or X_top.shape[0] == 0:
            continue
        n_ask_with_top += 1

        y_top = np.array(
            [eval_objective(xx, centers=centers, biases=biases, cat_dim=cat_dim, n_choices=n_choices) for xx in X_top],
            dtype=float,
        )
        keys_top = np.array([_discretize_cat(float(xx[int(cat_dim)]), int(n_choices)) for xx in X_top], dtype=int)

        best_overall = float(np.min(y_top))
        # best achievable within chosen key (inside this top-K slice).
        mask = keys_top == key_chosen
        if bool(np.any(mask)):
            best_in_key = float(np.min(y_top[mask]))
        else:
            best_in_key = float(y_chosen)

        regret_total = float(max(0.0, y_chosen - best_overall))
        regret_cat = float(max(0.0, best_in_key - best_overall))
        regret_cont = float(max(0.0, y_chosen - best_in_key))

        regrets.append(regret_total)
        regrets_cat.append(regret_cat)
        regrets_cont.append(regret_cont)

        k_best = int(keys_top[int(np.argmin(y_top))])
        hit1_key.append(1.0 if k_best == key_chosen else 0.0)

    chosen_keys_arr = np.asarray(chosen_keys, dtype=int)
    switches = int(np.sum(chosen_keys_arr[1:] != chosen_keys_arr[:-1])) if chosen_keys_arr.size >= 2 else 0
    switch_rate = float(switches) / float(max(1, int(chosen_keys_arr.size - 1))) if chosen_keys_arr.size else 0.0

    return {
        "trace": str(path),
        "n_ask": int(n_ask),
        "n_ask_with_top": int(n_ask_with_top),
        "switch_rate": float(switch_rate),
        "unique_keys": int(len(set(int(k) for k in chosen_keys_arr.tolist()))) if chosen_keys_arr.size else 0,
        "hit@1_key": float(np.mean(np.asarray(hit1_key, dtype=float))) if hit1_key else 0.0,
        "regret": _pct(np.asarray(regrets, dtype=float)),
        "regret_cat": _pct(np.asarray(regrets_cat, dtype=float)),
        "regret_cont": _pct(np.asarray(regrets_cont, dtype=float)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="Optional output JSON path.")
    p.add_argument("--n-choices", type=int, default=8)
    p.add_argument("--d-cont", type=int, default=6)
    p.add_argument("--obj-seed", type=int, default=0)
    p.add_argument("traces", nargs="+", help="One or more JSONL trace paths (glob supported by shell).")
    args = p.parse_args()

    obj = SynthCatObjective(n_choices=int(args.n_choices), d_cont=int(args.d_cont), seed=int(args.obj_seed))
    centers, biases = obj.make()
    cat_dim = 0

    results = []
    for t in args.traces:
        results.append(
            diagnose_trace(
                Path(t),
                centers=centers,
                biases=biases,
                cat_dim=cat_dim,
                n_choices=int(args.n_choices),
            )
        )

    out = {"meta": {"n_choices": int(args.n_choices), "d_cont": int(args.d_cont), "obj_seed": int(args.obj_seed)}, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

