#!/usr/bin/env python3
"""Diagnose categorical bandit masking from ALBA JSONL traces.

This focuses on "cross-key vs chosen-key" diagnostics when using hierarchical
categorical handling (e.g. `stage_effective='auto_enum_ts_masked'`).

It requires traces that log, per ask event:
- `top`: top-K candidates across keys (cross-key)
- `top_within_key`: top-K candidates restricted to the chosen key

Metrics
-------
- `hit@k_key`: whether the chosen-key best would appear in cross-key top-k.
- `regret_cat_pred`: score(best_cross_key) - score(best_within_chosen_key) (>=0).
- `rank_within_in_cross_topK`: rank of chosen-key best within cross-key top-K (or >K).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_floats(xs: Any) -> np.ndarray:
    if xs is None:
        return np.zeros(0, dtype=float)
    if isinstance(xs, (int, float)):
        v = float(xs)
        return np.asarray([v], dtype=float) if np.isfinite(v) else np.zeros(0, dtype=float)
    if not isinstance(xs, list):
        return np.zeros(0, dtype=float)
    out: List[float] = []
    for x in xs:
        try:
            v = float(x)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            continue
    return np.asarray(out, dtype=float)


def _pct(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    p50, p90, p99 = np.percentile(x, [50, 90, 99])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(np.max(x)),
    }


def _summarize_one(path: Path, *, ks: List[int], tol: float = 1e-12) -> Dict[str, Any]:
    regrets: List[float] = []
    ranks: List[int] = []
    hit_counts = {k: 0 for k in ks}
    n_used = 0
    n_missing = 0

    for e in _iter_jsonl(path):
        if str(e.get("event", "")) != "ask":
            continue
        top = e.get("top")
        topw = e.get("top_within_key")
        if not isinstance(top, dict) or not isinstance(topw, dict):
            n_missing += 1
            continue
        sc = _as_floats(top.get("score"))
        sw = _as_floats(topw.get("score"))
        if sc.size == 0 or sw.size == 0:
            n_missing += 1
            continue

        best_cross = float(sc[0])
        best_within = float(sw[0])
        if not (np.isfinite(best_cross) and np.isfinite(best_within)):
            n_missing += 1
            continue

        regret = float(best_cross - best_within)
        if regret < 0.0 and abs(regret) <= 1e-9:
            regret = 0.0
        regrets.append(regret)

        # Rank of best-within among cross-key top-K list.
        # If best_within is worse than the K-th score, we only know rank > K.
        K = int(sc.size)
        better = int(np.sum(sc > (best_within + tol)))
        rank = 1 + better
        if rank > K and best_within < float(sc[-1]) - tol:
            rank = K + 1
        ranks.append(int(rank))

        for k in ks:
            kk = int(k)
            if kk <= 0:
                continue
            if kk == 1:
                hit = bool(abs(regret) <= tol)
            elif kk <= K:
                hit = bool(best_within >= float(sc[kk - 1]) - tol)
            else:
                hit = False
            if hit:
                hit_counts[kk] = hit_counts.get(kk, 0) + 1

        n_used += 1

    regrets_arr = np.asarray(regrets, dtype=float)
    ranks_arr = np.asarray(ranks, dtype=float)
    return {
        "trace": str(path),
        "n_ask_used": int(n_used),
        "n_ask_missing_top": int(n_missing),
        "regret_cat_pred": _pct(regrets_arr),
        "rank_within_in_cross_topK": _pct(ranks_arr),
        "hit_at_k_key": {str(k): (float(hit_counts.get(int(k), 0)) / float(max(1, n_used))) for k in ks},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose categorical masking via cross-key vs within-key top-K traces.")
    p.add_argument("traces", nargs="+", help="One or more JSONL trace paths (globs ok via shell).")
    p.add_argument("--ks", default="1,5,10,32", help="Comma-separated k values for hit@k (default: 1,5,10,32).")
    p.add_argument("--out", default=None, help="Optional output JSON path for machine-readable summary.")
    args = p.parse_args()

    ks: List[int] = []
    for part in str(args.ks).split(","):
        part = part.strip()
        if not part:
            continue
        ks.append(int(part))
    ks = sorted(set([k for k in ks if k > 0]))
    if not ks:
        raise SystemExit("--ks must contain at least one positive integer")

    results: List[Dict[str, Any]] = []
    for raw in args.traces:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"Missing trace: {path}")

        rec = _summarize_one(path, ks=ks)
        results.append(rec)

        print(path.name)
        print(f"  n_ask_used={rec.get('n_ask_used')} missing_top={rec.get('n_ask_missing_top')}")
        print(f"  regret_cat_pred={rec.get('regret_cat_pred')}")
        print(f"  rank_within_in_cross_topK={rec.get('rank_within_in_cross_topK')}")
        print(f"  hit@k_key={rec.get('hit_at_k_key')}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2))
        print(f"\nwrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

