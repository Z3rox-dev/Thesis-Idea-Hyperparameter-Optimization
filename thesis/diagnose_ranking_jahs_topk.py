#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from benchmark_jahs import JAHSBenchWrapper


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _rankdata(a: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Average ranks for ties (1..n)."""
    a = np.asarray(a, dtype=float).reshape(-1)
    n = int(a.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    s = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(float(s[j + 1]) - float(s[i])) <= tol:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    ra = (ra - float(np.mean(ra))) / sa
    rb = (rb - float(np.mean(rb))) / sb
    return float(np.mean(ra * rb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    a0 = (a - float(np.mean(a))) / sa
    b0 = (b - float(np.mean(b))) / sb
    return float(np.mean(a0 * b0))


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


def _as_np_2d(xs: Any) -> np.ndarray:
    X = np.asarray(xs, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={tuple(X.shape)}")
    return X


def _find_row(X: np.ndarray, x: np.ndarray, *, tol: float = 0.0) -> Optional[int]:
    """Return index of row exactly matching x (within tol), else None."""
    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float).reshape(1, -1)
    if X.ndim != 2 or x.shape[1] != X.shape[1]:
        return None
    diffs = np.max(np.abs(X - x), axis=1)
    j = int(np.argmin(diffs))
    if float(diffs[j]) <= float(tol):
        return j
    return None


@dataclass
class IterMetrics:
    k: int
    corr_score: float
    corr_ucb: float
    corr_mu: float
    spear_score: float
    spear_ucb: float
    spear_mu: float
    regret_chosen: float
    regret_bestscore: float
    regret_bestucb: float
    chosen_true_rank: int
    bestscore_true_rank: int
    bestucb_true_rank: int
    hit1_chosen: int
    hit5_chosen: int
    hit10_chosen: int
    regret_cat: float
    regret_cont: float
    n_keys_in_top: int
    chosen_key_rank: int
    hit1_key: int
    key_chosen_k_count: int
    regret_cont_mu: float
    regret_cont_score: float
    corr_mu_within_key: float
    corr_score_within_key: float


def _jahs_key_spec(wrapper: JAHSBenchWrapper) -> List[Tuple[int, int]]:
    spec: List[Tuple[int, int]] = []
    for i, hp_name in enumerate(wrapper.HP_ORDER):
        hp = wrapper.HP_SPACE[hp_name]
        if hp.get("type") in {"ordinal", "categorical"}:
            n_choices = int(len(hp["choices"]))
            spec.append((int(i), n_choices))
    return spec


def _jahs_cat_key(x: np.ndarray, spec: List[Tuple[int, int]]) -> Tuple[int, ...]:
    x = np.asarray(x, dtype=float).reshape(-1)
    out: List[int] = []
    for dim_idx, n_choices in spec:
        v = float(np.clip(x[int(dim_idx)], 0.0, 1.0))
        if n_choices <= 1:
            out.append(0)
            continue
        idx = int(round(v * float(n_choices - 1)))
        idx = int(np.clip(idx, 0, n_choices - 1))
        out.append(idx)
    return tuple(out)


def _analyze_trace(path: Path) -> Dict[str, Any]:
    # Detect task/seed from first event (if present).
    first = None
    for e in _iter_jsonl(path):
        first = e
        break
    if first is None:
        raise ValueError(f"Empty trace: {path}")

    run = first.get("run", {}) if isinstance(first, dict) else {}
    task = None
    seed = None
    budget = None
    if isinstance(run, dict):
        task = run.get("task")
        seed = run.get("seed")
        budget = run.get("budget")
    if task is None:
        # Fallback to parsing filename.
        name = path.name
        for t in JAHSBenchWrapper.TASKS:
            if f"_{t}_" in name:
                task = t
                break
    if task is None:
        raise ValueError(f"Could not infer JAHS task from trace: {path}")

    wrapper = JAHSBenchWrapper(task=str(task))
    key_spec = _jahs_key_spec(wrapper)
    cache: Dict[Tuple[float, ...], float] = {}

    def eval_internal(x_row: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(x_row, dtype=float).reshape(-1), 12).tolist())
        if key in cache:
            return cache[key]
        err = float(wrapper.evaluate_array(np.asarray(x_row, dtype=float)))
        y_int = -err  # framework uses internal score = -error (maximize)
        cache[key] = y_int
        return y_int

    iters: List[IterMetrics] = []
    n_skipped = 0

    for e in _iter_jsonl(path):
        if str(e.get("event")) != "ask":
            continue
        top = e.get("top")
        if not isinstance(top, dict) or "x" not in top:
            n_skipped += 1
            continue

        X_top = _as_np_2d(top["x"])
        k = int(X_top.shape[0])
        score = np.asarray(top.get("score", []), dtype=float).reshape(-1)
        ucb = np.asarray(top.get("ucb", []), dtype=float).reshape(-1)
        mu = np.asarray(top.get("mu", []), dtype=float).reshape(-1)
        if score.size != k or ucb.size != k or mu.size != k:
            n_skipped += 1
            continue

        y_true = np.asarray([eval_internal(X_top[i]) for i in range(k)], dtype=float)
        best_true = float(np.max(y_true))
        true_order = np.argsort(-y_true)  # best first

        keys_top = [_jahs_cat_key(X_top[i], key_spec) for i in range(k)]
        key_to_best: Dict[Tuple[int, ...], float] = {}
        for kk, yi in zip(keys_top, y_true):
            prev = key_to_best.get(kk)
            if prev is None or float(yi) > float(prev):
                key_to_best[kk] = float(yi)
        keys_unique = list(key_to_best.keys())
        n_keys_in_top = int(len(keys_unique))
        keys_best_vals = np.asarray([float(key_to_best[kk]) for kk in keys_unique], dtype=float)
        key_best_order = np.argsort(-keys_best_vals)  # best key first

        x_final = np.asarray(e.get("x_final", []), dtype=float).reshape(-1)
        idx_chosen = _find_row(X_top, x_final, tol=0.0)
        if idx_chosen is None:
            # As a fallback, evaluate chosen independently.
            # With masked categorical selection, the chosen point may not appear in the
            # cross-key top-K by score; use `top_within_key` (if present) to still
            # decompose regret into categorical vs continuous components.
            chosen_true = eval_internal(x_final) if x_final.size else float("nan")
            regret_chosen = best_true - float(chosen_true) if np.isfinite(chosen_true) else float("nan")
            chosen_true_rank = -1

            regret_cat = float("nan")
            regret_cont = float("nan")
            chosen_key_rank = -1
            hit1_key = 0
            key_chosen_k_count = 0
            regret_cont_mu = float("nan")
            regret_cont_score = float("nan")
            corr_mu_within_key = 0.0
            corr_score_within_key = 0.0

            top_within = e.get("top_within_key")
            if isinstance(top_within, dict) and "x" in top_within and x_final.size:
                try:
                    X_w = _as_np_2d(top_within["x"])
                    mu_w = np.asarray(top_within.get("mu", []), dtype=float).reshape(-1)
                    score_w = np.asarray(top_within.get("score", []), dtype=float).reshape(-1)
                    if X_w.shape[0] == mu_w.size == score_w.size and X_w.shape[0] > 0:
                        y_w = np.asarray([eval_internal(X_w[i]) for i in range(int(X_w.shape[0]))], dtype=float)
                        best_in_key = float(np.max(y_w))

                        regret_cat = best_true - best_in_key
                        regret_cont = best_in_key - float(chosen_true)

                        # Rank of chosen key among keys seen in cross-key top-K (+ chosen).
                        key_chosen = _jahs_cat_key(x_final, key_spec)
                        key_to_best_full = dict(key_to_best)
                        prev = key_to_best_full.get(key_chosen)
                        if prev is None or float(best_in_key) > float(prev):
                            key_to_best_full[key_chosen] = float(best_in_key)

                        keys_unique_full = list(key_to_best_full.keys())
                        n_keys_in_top = int(len(keys_unique_full))
                        keys_best_vals_full = np.asarray(
                            [float(key_to_best_full[kk]) for kk in keys_unique_full], dtype=float
                        )
                        key_best_order_full = np.argsort(-keys_best_vals_full)
                        chosen_key_rank = int(
                            1 + int(np.where(key_best_order_full == keys_unique_full.index(key_chosen))[0][0])
                        )
                        hit1_key = int(chosen_key_rank == 1)

                        # Within-key diagnostics from the within-key top list.
                        idx_mu_w = int(np.argmax(mu_w))
                        idx_score_w = int(np.argmax(score_w))
                        regret_cont_mu = best_in_key - float(y_w[idx_mu_w])
                        regret_cont_score = best_in_key - float(y_w[idx_score_w])
                        corr_mu_within_key = _spearman(mu_w, y_w) if mu_w.size >= 2 else 0.0
                        corr_score_within_key = _spearman(score_w, y_w) if score_w.size >= 2 else 0.0
                except Exception:
                    pass
        else:
            chosen_true = float(y_true[int(idx_chosen)])
            regret_chosen = best_true - chosen_true
            chosen_true_rank = int(1 + int(np.where(true_order == int(idx_chosen))[0][0]))

            key_chosen = keys_top[int(idx_chosen)]
            best_in_key = float(key_to_best.get(key_chosen, chosen_true))
            regret_cat = best_true - best_in_key
            regret_cont = best_in_key - chosen_true

            # Rank of chosen key among unique keys in top-K (1..n_keys).
            chosen_key_rank = int(1 + int(np.where(key_best_order == keys_unique.index(key_chosen))[0][0]))
            hit1_key = int(chosen_key_rank == 1)

            idxs_key = [i for i, kk in enumerate(keys_top) if kk == key_chosen]
            key_chosen_k_count = int(len(idxs_key))
            if idxs_key:
                idxs_key_arr = np.asarray(idxs_key, dtype=np.int64)
                y_key = y_true[idxs_key_arr]
                mu_key = mu[idxs_key_arr]
                score_key = score[idxs_key_arr]
                best_in_key_true = float(np.max(y_key))

                idx_mu = int(idxs_key_arr[int(np.argmax(mu_key))])
                idx_score = int(idxs_key_arr[int(np.argmax(score_key))])
                regret_cont_mu = best_in_key_true - float(y_true[idx_mu])
                regret_cont_score = best_in_key_true - float(y_true[idx_score])

                corr_mu_within_key = _spearman(mu_key, y_key) if mu_key.size >= 2 else 0.0
                corr_score_within_key = _spearman(score_key, y_key) if score_key.size >= 2 else 0.0
            else:
                regret_cont_mu = float("nan")
                regret_cont_score = float("nan")
                corr_mu_within_key = 0.0
                corr_score_within_key = 0.0

        idx_bestscore = int(np.argmax(score))
        idx_bestucb = int(np.argmax(ucb))

        regret_bestscore = best_true - float(y_true[idx_bestscore])
        regret_bestucb = best_true - float(y_true[idx_bestucb])

        bestscore_true_rank = int(1 + int(np.where(true_order == idx_bestscore)[0][0]))
        bestucb_true_rank = int(1 + int(np.where(true_order == idx_bestucb)[0][0]))

        corr_score = _pearson(score, y_true)
        corr_ucb = _pearson(ucb, y_true)
        corr_mu = _pearson(mu, y_true)
        spear_score = _spearman(score, y_true)
        spear_ucb = _spearman(ucb, y_true)
        spear_mu = _spearman(mu, y_true)

        hit1 = int(chosen_true_rank == 1)
        hit5 = int(chosen_true_rank != -1 and chosen_true_rank <= 5)
        hit10 = int(chosen_true_rank != -1 and chosen_true_rank <= 10)

        iters.append(
            IterMetrics(
                k=k,
                corr_score=corr_score,
                corr_ucb=corr_ucb,
                corr_mu=corr_mu,
                spear_score=spear_score,
                spear_ucb=spear_ucb,
                spear_mu=spear_mu,
                regret_chosen=float(regret_chosen),
                regret_bestscore=float(regret_bestscore),
                regret_bestucb=float(regret_bestucb),
                chosen_true_rank=int(chosen_true_rank),
                bestscore_true_rank=int(bestscore_true_rank),
                bestucb_true_rank=int(bestucb_true_rank),
                hit1_chosen=hit1,
                hit5_chosen=hit5,
                hit10_chosen=hit10,
                regret_cat=float(regret_cat),
                regret_cont=float(regret_cont),
                n_keys_in_top=int(n_keys_in_top),
                chosen_key_rank=int(chosen_key_rank),
                hit1_key=int(hit1_key),
                key_chosen_k_count=int(key_chosen_k_count),
                regret_cont_mu=float(regret_cont_mu),
                regret_cont_score=float(regret_cont_score),
                corr_mu_within_key=float(corr_mu_within_key),
                corr_score_within_key=float(corr_score_within_key),
            )
        )

    if not iters:
        raise ValueError(f"No usable ask/top events found in trace: {path}")

    ks = np.asarray([m.k for m in iters], dtype=int)
    assert int(np.min(ks)) == int(np.max(ks)), "Inconsistent top-K length within one trace"

    def vec(name: str) -> np.ndarray:
        return np.asarray([float(getattr(m, name)) for m in iters], dtype=float)

    def ivec(name: str) -> np.ndarray:
        return np.asarray([int(getattr(m, name)) for m in iters], dtype=int)

    def nanvec(name: str) -> np.ndarray:
        return np.asarray([float(getattr(m, name)) for m in iters], dtype=float)

    out: Dict[str, Any] = {
        "trace": str(path),
        "task": str(task),
        "seed": (int(seed) if seed is not None else None),
        "budget": (int(budget) if budget is not None else None),
        "n_ask_with_top": int(len(iters)),
        "top_k": int(ks[0]),
        "skipped_ask_events": int(n_skipped),
        "corr": {
            "score": _pct(vec("corr_score")),
            "ucb": _pct(vec("corr_ucb")),
            "mu": _pct(vec("corr_mu")),
        },
        "spearman": {
            "score": _pct(vec("spear_score")),
            "ucb": _pct(vec("spear_ucb")),
            "mu": _pct(vec("spear_mu")),
        },
        "regret": {
            "chosen": _pct(vec("regret_chosen")),
            "bestscore": _pct(vec("regret_bestscore")),
            "bestucb": _pct(vec("regret_bestucb")),
            "cat": _pct(nanvec("regret_cat")),
            "cont": _pct(nanvec("regret_cont")),
            "cont_mu": _pct(nanvec("regret_cont_mu")),
            "cont_score": _pct(nanvec("regret_cont_score")),
        },
        "rank": {
            "chosen": _pct(ivec("chosen_true_rank").astype(float)),
            "bestscore": _pct(ivec("bestscore_true_rank").astype(float)),
            "bestucb": _pct(ivec("bestucb_true_rank").astype(float)),
            "hit@1": float(np.mean(ivec("hit1_chosen"))),
            "hit@5": float(np.mean(ivec("hit5_chosen"))),
            "hit@10": float(np.mean(ivec("hit10_chosen"))),
            "n_keys_in_top": _pct(ivec("n_keys_in_top").astype(float)),
            "key_rank_chosen": _pct(ivec("chosen_key_rank").astype(float)),
            "hit@1_key": float(np.mean(ivec("hit1_key"))),
            "k_count_chosen_key": _pct(ivec("key_chosen_k_count").astype(float)),
            "spearman_mu_within_key": _pct(nanvec("corr_mu_within_key")),
            "spearman_score_within_key": _pct(nanvec("corr_score_within_key")),
        },
        "cache": {"n_unique_x": int(len(cache)), "n_evals": int(wrapper.n_evals)},
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Ranking/regret diagnostics from JAHS top-K traces (offline eval).")
    p.add_argument("paths", nargs="+", help="One or more trace .jsonl files (with ask.top fields).")
    p.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = p.parse_args()

    results: List[Dict[str, Any]] = []
    for sp in args.paths:
        res = _analyze_trace(Path(sp))
        results.append(res)
        print(Path(sp).name)
        print(f"  task={res['task']} seed={res['seed']} budget={res['budget']} ask_with_top={res['n_ask_with_top']} K={res['top_k']}")
        print(f"  regret(chosen): {res['regret']['chosen']}")
        print(f"  regret(cat/cont): cat={res['regret']['cat']} cont={res['regret']['cont']}")
        print(f"  rank(chosen): {res['rank']['chosen']} hit@1={res['rank']['hit@1']:.3f} hit@5={res['rank']['hit@5']:.3f}")
        print(
            f"  keys: n_keys_in_top={res['rank']['n_keys_in_top']} "
            f"key_rank_chosen={res['rank']['key_rank_chosen']} "
            f"hit@1_key={res['rank']['hit@1_key']:.3f}"
        )
        print(f"  spearman(score): {res['spearman']['score']}")
        print(f"  cache: {res['cache']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
