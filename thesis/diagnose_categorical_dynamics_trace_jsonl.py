#!/usr/bin/env python3
"""Diagnose categorical key dynamics from ALBA JSONL traces.

Focus:
- key thrashing (switch rate, run-length distribution)
- exploration vs stabilization (cumulative unique keys over time)
- which per-key statistic is most predictive of final best (mean vs best vs good-rate)

Works on any trace that logs:
- ask events with `categorical.key_final` or `categorical.key_chosen`
- tell events with `y_internal` and `gamma`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


Key = Tuple[int, ...]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_key(obj: Any) -> Optional[Key]:
    if obj is None:
        return None
    if isinstance(obj, tuple):
        try:
            return tuple(int(x) for x in obj)
        except Exception:
            return None
    if isinstance(obj, list):
        try:
            return tuple(int(x) for x in obj)
        except Exception:
            return None
    return None


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


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 5:
        return None
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    ra = (ra - float(np.mean(ra))) / sa
    rb = (rb - float(np.mean(rb))) / sb
    return float(np.mean(ra * rb))


def _summarize_one(path: Path, *, m: int = 3) -> Dict[str, Any]:
    key_by_iter: Dict[int, Key] = {}
    y_by_iter: Dict[int, float] = {}
    gamma_by_iter: Dict[int, float] = {}

    for e in _iter_jsonl(path):
        ev = str(e.get("event", ""))
        it = e.get("iteration")
        if not isinstance(it, int):
            continue

        if ev == "ask":
            cat = e.get("categorical", {})
            if isinstance(cat, dict):
                k = _as_key(cat.get("key_final"))
                if k is None:
                    k = _as_key(cat.get("key_chosen"))
                if k is not None:
                    key_by_iter[int(it)] = k
        elif ev == "tell":
            yi = e.get("y_internal")
            if yi is not None:
                try:
                    y_by_iter[int(it)] = float(yi)
                except Exception:
                    pass
            g = e.get("gamma")
            if g is not None:
                try:
                    gamma_by_iter[int(it)] = float(g)
                except Exception:
                    pass

    iters = sorted(key_by_iter.keys())
    keys = [key_by_iter[i] for i in iters]
    n = int(len(keys))

    out: Dict[str, Any] = {"trace": str(path), "n_ask_with_key": n, "n_tell": int(len(y_by_iter))}
    if n == 0:
        return out

    switches = int(sum(1 for a, b in zip(keys, keys[1:]) if a != b))
    out["switch_rate"] = float(switches / max(1, n - 1))
    out["unique_keys_total"] = int(len(set(keys)))

    # Run lengths
    run_lens: List[int] = []
    cur = keys[0]
    run = 1
    for k in keys[1:]:
        if k == cur:
            run += 1
        else:
            run_lens.append(run)
            cur = k
            run = 1
    run_lens.append(run)
    out["run_lengths"] = _pct(np.asarray(run_lens, dtype=float))

    # Cumulative unique keys at common milestones
    milestones = [10, 25, 50, 100, 150, 200, 300, n]
    milestones = [int(x) for x in milestones if 1 <= int(x) <= n]
    seen: set[Key] = set()
    cum: Dict[str, int] = {}
    for j, k in enumerate(keys, start=1):
        seen.add(k)
        if j in milestones:
            cum[str(j)] = int(len(seen))
    out["cum_unique_keys"] = cum

    # Per-key stats (use tell-side y_internal; pair via iteration).
    ys_by_key: Dict[Key, List[float]] = {}
    for it in iters:
        k = key_by_iter[it]
        y = y_by_iter.get(it)
        if y is None or not np.isfinite(y):
            continue
        ys_by_key.setdefault(k, []).append(float(y))

    if not ys_by_key:
        return out

    gamma_final: Optional[float] = None
    if gamma_by_iter:
        gamma_final = float(gamma_by_iter[max(gamma_by_iter.keys())])
    out["gamma_final"] = gamma_final

    # Predictiveness: early statistic after m visits vs final best in that key.
    best_m: List[float] = []
    mean_m: List[float] = []
    good_m: List[float] = []
    final_best: List[float] = []
    for ys in ys_by_key.values():
        if len(ys) < m:
            continue
        y0 = np.asarray(ys[:m], dtype=float)
        best_m.append(float(np.max(y0)))
        mean_m.append(float(np.mean(y0)))
        if gamma_final is not None and np.isfinite(gamma_final):
            good_m.append(float(np.mean(y0 >= float(gamma_final))))
        final_best.append(float(np.max(np.asarray(ys, dtype=float))))

    out["n_keys_with_ge_m_evals"] = int(len(final_best))
    out["spearman_best_m_vs_final_best"] = _spearman(np.asarray(best_m), np.asarray(final_best))
    out["spearman_mean_m_vs_final_best"] = _spearman(np.asarray(mean_m), np.asarray(final_best))
    out["spearman_goodrate_m_vs_final_best"] = (
        _spearman(np.asarray(good_m), np.asarray(final_best)) if good_m else None
    )

    visits = np.asarray([len(v) for v in ys_by_key.values()], dtype=float)
    out["visits_per_key"] = _pct(visits)

    # Stability in the tail: how many distinct keys appear late?
    tail = min(50, n)
    head = min(50, n)
    out["unique_keys_first_k"] = int(len(set(keys[:head])))
    out["unique_keys_last_k"] = int(len(set(keys[-tail:])))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose categorical key dynamics from ALBA trace JSONL files.")
    p.add_argument("traces", nargs="+", help="One or more JSONL trace paths (globs ok via shell).")
    p.add_argument("--m", type=int, default=3, help="Early-visit window per key (default: 3).")
    p.add_argument("--out", default=None, help="Optional output JSON path for machine-readable summary.")
    args = p.parse_args()

    m = int(args.m)
    if m < 1:
        raise SystemExit("--m must be >= 1")

    results: List[Dict[str, Any]] = []
    for raw in args.traces:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"Missing trace: {path}")

        rec = _summarize_one(path, m=m)
        results.append(rec)

        print(path.name)
        print(
            f"  n_ask_with_key={rec.get('n_ask_with_key')} unique_total={rec.get('unique_keys_total')} "
            f"switch_rate={rec.get('switch_rate')}"
        )
        print(f"  run_lengths={rec.get('run_lengths')}")
        print(f"  cum_unique_keys={rec.get('cum_unique_keys')}")
        if rec.get("n_keys_with_ge_m_evals"):
            print(
                f"  key_predictiveness(m={m}): "
                f"best={rec.get('spearman_best_m_vs_final_best')} "
                f"mean={rec.get('spearman_mean_m_vs_final_best')} "
                f"goodrate={rec.get('spearman_goodrate_m_vs_final_best')}"
            )
        print(
            f"  visits_per_key={rec.get('visits_per_key')} "
            f"unique_first_k={rec.get('unique_keys_first_k')} unique_last_k={rec.get('unique_keys_last_k')}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2))
        print(f"\nwrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

