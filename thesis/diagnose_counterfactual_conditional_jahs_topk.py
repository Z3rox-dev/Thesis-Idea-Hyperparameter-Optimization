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


def _rankdata(a: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
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


def _ridge_solve(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_feat = int(Phi.shape[1])
    A = Phi.T @ Phi + float(lam) * np.eye(n_feat, dtype=float)
    b = Phi.T @ y
    return np.linalg.solve(A, b)


def _jahs_key_spec(wrapper: JAHSBenchWrapper) -> List[Tuple[int, int]]:
    spec: List[Tuple[int, int]] = []
    for i, hp_name in enumerate(wrapper.HP_ORDER):
        hp = wrapper.HP_SPACE[hp_name]
        if hp.get("type") in {"ordinal", "categorical"}:
            spec.append((int(i), int(len(hp["choices"]))))
    return spec


def _jahs_cont_dims(wrapper: JAHSBenchWrapper) -> List[int]:
    out: List[int] = []
    for i, hp_name in enumerate(wrapper.HP_ORDER):
        hp = wrapper.HP_SPACE[hp_name]
        if hp.get("type") == "float":
            out.append(int(i))
    return out


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


def _poly2_features(X: np.ndarray) -> np.ndarray:
    """Degree-2 polynomial features with intercept; X shape (n,d)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    feats: List[np.ndarray] = [np.ones((n, 1), dtype=float), X]
    feats.append(X * X)
    if d >= 2:
        cross: List[np.ndarray] = []
        for i in range(d):
            for j in range(i + 1, d):
                cross.append((X[:, i] * X[:, j]).reshape(n, 1))
        if cross:
            feats.append(np.concatenate(cross, axis=1))
    return np.concatenate(feats, axis=1)


@dataclass
class AskRecord:
    iteration: int
    x_final: np.ndarray
    X_top: np.ndarray
    score_top: np.ndarray
    mu_top: np.ndarray


def _analyze_trace(path: Path, *, min_train: int = 12, ridge_lam: float = 1e-3) -> Dict[str, Any]:
    # Infer task/seed from first non-empty event.
    first = None
    for e in _iter_jsonl(path):
        first = e
        break
    if first is None:
        raise ValueError(f"Empty trace: {path}")

    run = first.get("run", {}) if isinstance(first, dict) else {}
    task = str(run.get("task") or "")
    seed = run.get("seed")
    budget = run.get("budget")
    if not task:
        raise ValueError(f"Could not infer task from trace: {path}")

    wrapper = JAHSBenchWrapper(task=task)
    key_spec = _jahs_key_spec(wrapper)
    cont_dims = _jahs_cont_dims(wrapper)
    if not cont_dims:
        raise ValueError("No continuous dims detected for JAHS wrapper")

    cache: Dict[Tuple[float, ...], float] = {}

    def eval_internal(x_row: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(x_row, dtype=float).reshape(-1), 12).tolist())
        if key in cache:
            return cache[key]
        err = float(wrapper.evaluate_array(np.asarray(x_row, dtype=float)))
        y_int = -err
        cache[key] = y_int
        return y_int

    tells: List[Tuple[int, np.ndarray, float, Tuple[int, ...]]] = []
    asks: List[AskRecord] = []

    for e in _iter_jsonl(path):
        if str(e.get("event")) == "tell":
            it = int(e.get("iteration", -1))
            x = np.asarray(e.get("x", []), dtype=float).reshape(-1)
            y = float(e.get("y_internal"))
            k = _jahs_cat_key(x, key_spec)
            tells.append((it, x, y, k))
        elif str(e.get("event")) == "ask":
            top = e.get("top")
            if not isinstance(top, dict) or "x" not in top:
                continue
            X_top = np.asarray(top["x"], dtype=float)
            score_top = np.asarray(top.get("score", []), dtype=float).reshape(-1)
            mu_top = np.asarray(top.get("mu", []), dtype=float).reshape(-1)
            if X_top.ndim != 2 or score_top.size != X_top.shape[0] or mu_top.size != X_top.shape[0]:
                continue
            asks.append(
                AskRecord(
                    iteration=int(e.get("iteration", -1)),
                    x_final=np.asarray(e.get("x_final", []), dtype=float).reshape(-1),
                    X_top=X_top,
                    score_top=score_top,
                    mu_top=mu_top,
                )
            )

    tells.sort(key=lambda t: t[0])

    # For quick prefix slicing.
    tell_iters = np.asarray([t[0] for t in tells], dtype=int)

    regrets_actual: List[float] = []
    regrets_score_greedy: List[float] = []
    regrets_mu_greedy: List[float] = []
    regrets_cf_key_fixed: List[float] = []
    regrets_cf_choose_key: List[float] = []
    spear_cf_within_key: List[float] = []

    n_used = 0
    n_cf_fit = 0

    for a in asks:
        if a.iteration < 0:
            continue
        X = a.X_top
        k = int(X.shape[0])
        if k < 2:
            continue

        # Evaluate true y on all top-K points once.
        y_true = np.asarray([eval_internal(X[i]) for i in range(k)], dtype=float)
        best_true = float(np.max(y_true))

        # chosen idx (x_final is always one of top-K in current trace format)
        diffs = np.max(np.abs(X - a.x_final.reshape(1, -1)), axis=1)
        idx_chosen = int(np.argmin(diffs))
        if float(diffs[idx_chosen]) > 1e-12:
            continue

        regrets_actual.append(best_true - float(y_true[idx_chosen]))
        idx_score = int(np.argmax(a.score_top))
        idx_mu = int(np.argmax(a.mu_top))
        regrets_score_greedy.append(best_true - float(y_true[idx_score]))
        regrets_mu_greedy.append(best_true - float(y_true[idx_mu]))

        # Prepare key groups for candidates.
        keys_top = [_jahs_cat_key(X[i], key_spec) for i in range(k)]
        key_to_indices: Dict[Tuple[int, ...], List[int]] = {}
        for i, kk in enumerate(keys_top):
            key_to_indices.setdefault(kk, []).append(i)

        key_chosen = keys_top[idx_chosen]
        idxs_key = key_to_indices.get(key_chosen, [])
        if len(idxs_key) < 2:
            continue

        # Find tells prefix strictly before this ask iteration.
        t = int(a.iteration)
        pre_n = int(np.searchsorted(tell_iters, t, side="left"))
        pre = tells[:pre_n]
        pre_by_key: Dict[Tuple[int, ...], List[Tuple[np.ndarray, float]]] = {}
        for _, x, y, kk in pre:
            pre_by_key.setdefault(kk, []).append((x, y))

        # --- Counterfactual 1: keep chosen key, but pick continuous point using conditional poly2 fit.
        train = pre_by_key.get(key_chosen, [])
        if len(train) >= min_train:
            Xh = np.stack([np.asarray(x, dtype=float)[cont_dims] for x, _ in train], axis=0)
            yh = np.asarray([float(y) for _, y in train], dtype=float)
            Phi = _poly2_features(Xh)
            w = _ridge_solve(Phi, yh, ridge_lam)

            Xk = X[np.asarray(idxs_key, dtype=np.int64)][:, cont_dims]
            Phi_k = _poly2_features(Xk)
            y_hat = Phi_k @ w
            idx_rel = int(np.argmax(y_hat))
            idx_pick = int(idxs_key[idx_rel])
            regrets_cf_key_fixed.append(best_true - float(y_true[idx_pick]))

            # Diagnostic: ranking quality within key.
            spear_cf_within_key.append(_spearman(y_hat, y_true[np.asarray(idxs_key, dtype=np.int64)]))
            n_cf_fit += 1
        else:
            # If we can't fit, fall back to current choice.
            regrets_cf_key_fixed.append(best_true - float(y_true[idx_chosen]))

        # --- Counterfactual 2: choose key as well: among all keys with enough data, pick best predicted.
        best_idx_pick = idx_chosen
        best_pred = -np.inf
        any_fit = False
        for kk, idxs in key_to_indices.items():
            train = pre_by_key.get(kk, [])
            if len(train) < min_train:
                continue
            any_fit = True
            Xh = np.stack([np.asarray(x, dtype=float)[cont_dims] for x, _ in train], axis=0)
            yh = np.asarray([float(y) for _, y in train], dtype=float)
            Phi = _poly2_features(Xh)
            w = _ridge_solve(Phi, yh, ridge_lam)

            Xk = X[np.asarray(idxs, dtype=np.int64)][:, cont_dims]
            Phi_k = _poly2_features(Xk)
            y_hat = Phi_k @ w
            idx_rel = int(np.argmax(y_hat))
            pred = float(y_hat[idx_rel])
            if pred > best_pred:
                best_pred = pred
                best_idx_pick = int(idxs[idx_rel])

        if any_fit:
            regrets_cf_choose_key.append(best_true - float(y_true[best_idx_pick]))
        else:
            regrets_cf_choose_key.append(best_true - float(y_true[idx_chosen]))

        n_used += 1

    out: Dict[str, Any] = {
        "trace": str(path),
        "task": task,
        "seed": (int(seed) if seed is not None else None),
        "budget": (int(budget) if budget is not None else None),
        "top_k": (int(asks[0].X_top.shape[0]) if asks else None),
        "min_train": int(min_train),
        "ridge_lam": float(ridge_lam),
        "n_ask_total": int(len(asks)),
        "n_ask_used": int(n_used),
        "n_cf_fit_key": int(n_cf_fit),
        "regret": {
            "actual": _pct(np.asarray(regrets_actual, dtype=float)),
            "score_greedy": _pct(np.asarray(regrets_score_greedy, dtype=float)),
            "mu_greedy": _pct(np.asarray(regrets_mu_greedy, dtype=float)),
            "cf_key_fixed": _pct(np.asarray(regrets_cf_key_fixed, dtype=float)),
            "cf_choose_key": _pct(np.asarray(regrets_cf_choose_key, dtype=float)),
        },
        "cf_quality": {
            "spearman_within_key": _pct(np.asarray(spear_cf_within_key, dtype=float)),
        },
        "cache": {"n_unique_x": int(len(cache)), "n_evals": int(wrapper.n_evals)},
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Counterfactual: key-conditioned surrogate on JAHS top-K traces.")
    p.add_argument("paths", nargs="+", help="One or more trace .jsonl files with ask.top stored.")
    p.add_argument("--min-train", type=int, default=12, help="Minimum prior points in key to fit model.")
    p.add_argument("--ridge-lam", type=float, default=1e-3, help="Ridge regularization for poly2 regression.")
    p.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = p.parse_args()

    results: List[Dict[str, Any]] = []
    for sp in args.paths:
        res = _analyze_trace(Path(sp), min_train=int(args.min_train), ridge_lam=float(args.ridge_lam))
        results.append(res)
        name = Path(sp).name
        print(name)
        print(
            f"  task={res['task']} seed={res['seed']} used={res['n_ask_used']}/{res['n_ask_total']} "
            f"cf_fit_key={res['n_cf_fit_key']} cache_evals={res['cache']['n_evals']}"
        )
        print(f"  regret(actual): {res['regret']['actual']}")
        print(f"  regret(score_greedy): {res['regret']['score_greedy']}")
        print(f"  regret(mu_greedy): {res['regret']['mu_greedy']}")
        print(f"  regret(cf_key_fixed): {res['regret']['cf_key_fixed']}")
        print(f"  regret(cf_choose_key): {res['regret']['cf_choose_key']}")
        print(f"  cf spearman(within_key): {res['cf_quality']['spearman_within_key']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()

