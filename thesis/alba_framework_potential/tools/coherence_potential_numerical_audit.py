#!/usr/bin/env python3
"""
Numerical audit for ALBA coherence + potential field
====================================================

Goal
----
Provide a *numerical* sanity check suite for the geometric coherence and the
global potential field reconstruction used by ALBA, without modifying the core
framework modules.

This script:
1) Builds synthetic "leaf" sets (Cube objects with mock LGS gradients).
2) Runs `compute_coherence_scores(...)`.
3) Checks invariances, ranges, and usefulness proxies (e.g., correlation with
   distance-to-optimum in controlled synthetic landscapes).
4) Highlights methodological limitations (WARN) distinct from hard numerical
   failures (FAIL).

Usage
-----
From repo root:
  python3 thesis/alba_framework_potential/tools/coherence_potential_numerical_audit.py

Optional:
  python3 thesis/alba_framework_potential/tools/coherence_potential_numerical_audit.py --report /tmp/audit.md
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Make `alba_framework_potential` importable when executed from repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _THESIS_DIR not in sys.path:
    sys.path.insert(0, _THESIS_DIR)

from alba_framework_potential.cube import Cube  # noqa: E402
from alba_framework_potential.coherence import (  # noqa: E402
    _build_knn_graph,
    _compute_predicted_drops,
    _solve_potential_least_squares,
    compute_coherence_scores,
)


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    details: str


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Minimal rank transform (average ranks for ties) without scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            mean_rank = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _make_grid_leaves(
    n_per_axis: int,
    *,
    dim: int = 2,
    lo: float = -1.0,
    hi: float = 1.0,
    optimum: Optional[np.ndarray] = None,
    set_gradients: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[Cube]:
    if dim not in (1, 2, 3):
        raise ValueError("This audit helper supports dim in {1,2,3} only.")
    if rng is None:
        rng = np.random.default_rng(0)
    if optimum is None:
        optimum = np.zeros(dim, dtype=float)
    optimum = np.asarray(optimum, dtype=float).reshape(dim)

    axes = [np.linspace(lo, hi, n_per_axis) for _ in range(dim)]
    d_step = axes[0][1] - axes[0][0] if n_per_axis > 1 else (hi - lo)

    leaves: List[Cube] = []

    if dim == 1:
        for x0 in axes[0]:
            bounds = [(float(x0 - d_step / 2), float(x0 + d_step / 2))]
            c = Cube(bounds=bounds)
            if set_gradients:
                g = (optimum - c.center()).astype(float)
                if np.linalg.norm(g) < 1e-12:
                    g = np.array([1e-6], dtype=float)
                c.lgs_model = {"grad": g}
            c.n_good = 1
            leaves.append(c)
        return leaves

    if dim == 2:
        for x0 in axes[0]:
            for x1 in axes[1]:
                bounds = [
                    (float(x0 - d_step / 2), float(x0 + d_step / 2)),
                    (float(x1 - d_step / 2), float(x1 + d_step / 2)),
                ]
                c = Cube(bounds=bounds)
                if set_gradients:
                    g = (optimum - c.center()).astype(float)
                    if np.linalg.norm(g) < 1e-12:
                        g = np.array([1e-6, 0.0], dtype=float)
                    c.lgs_model = {"grad": g}
                c.n_good = 1
                leaves.append(c)
        return leaves

    # dim == 3
    for x0 in axes[0]:
        for x1 in axes[1]:
            for x2 in axes[2]:
                bounds = [
                    (float(x0 - d_step / 2), float(x0 + d_step / 2)),
                    (float(x1 - d_step / 2), float(x1 + d_step / 2)),
                    (float(x2 - d_step / 2), float(x2 + d_step / 2)),
                ]
                c = Cube(bounds=bounds)
                if set_gradients:
                    g = (optimum - c.center()).astype(float)
                    if np.linalg.norm(g) < 1e-12:
                        g = np.array([1e-6, 0.0, 0.0], dtype=float)
                    c.lgs_model = {"grad": g}
                c.n_good = 1
                leaves.append(c)
    return leaves


def _centers(leaves: Sequence[Cube]) -> np.ndarray:
    return np.array([c.center() for c in leaves], dtype=float)


def _potentials_array(potentials: Dict[int, float], n: int) -> np.ndarray:
    return np.array([potentials.get(i, 0.5) for i in range(n)], dtype=float)


def check_ls_solver_recovery(rng: np.random.Generator) -> CheckResult:
    n = 40
    u_true = rng.normal(0, 1.0, size=n).astype(float)
    u_true = u_true - u_true[0]  # match gauge u[0]=0
    u_true = u_true / max(1e-9, float(np.std(u_true)))  # scale ~1

    # Ensure connectivity: chain edges + random extra edges.
    edges: List[Tuple[int, int]] = [(i, i + 1) for i in range(n - 1)]
    for _ in range(4 * n):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            edges.append((i, j))

    d = np.array([u_true[j] - u_true[i] for i, j in edges], dtype=float)
    weights = rng.uniform(0.5, 2.0, size=len(edges)).astype(float)

    u_est = _solve_potential_least_squares(n, edges, d, weights)

    rmse = float(np.sqrt(np.mean((u_est - u_true) ** 2)))
    corr = _pearson(u_est, u_true)

    # With damped LSQR, we expect a small bias but still strong agreement.
    ok = (rmse < 0.10) and (corr > 0.99)
    status = "PASS" if ok else "FAIL"
    details = f"rmse={rmse:.4f}, pearson={corr:.4f}, n={n}, |E|={len(edges)}"
    return CheckResult("LS solver recovery (gauge-fixed)", status, details)


def check_potential_monotonicity_2d() -> CheckResult:
    leaves = _make_grid_leaves(9, dim=2, lo=-1.0, hi=1.0, optimum=np.zeros(2))
    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves, k_neighbors=6)

    p = _potentials_array(potentials, len(leaves))
    c = _centers(leaves)
    dist = np.linalg.norm(c, axis=1)
    pear = _pearson(dist, p)
    spear = _spearman(dist, p)

    # In a perfect convex bowl with inward gradients, potentials should rank distance well.
    ok = (pear > 0.95) and (spear > 0.95) and (global_coh > 0.80)
    status = "PASS" if ok else "FAIL"
    details = (
        f"pearson(dist,pot)={pear:.4f}, spearman={spear:.4f}, "
        f"global_coh={global_coh:.3f}, q60={q60:.3f}, q80={q80:.3f}"
    )
    return CheckResult("Potential monotonicity on 2D bowl", status, details)


def check_coherence_valley_flip_1d() -> CheckResult:
    leaves = _make_grid_leaves(41, dim=1, lo=-1.0, hi=1.0, optimum=np.zeros(1))
    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves, k_neighbors=6)
    c = _centers(leaves).reshape(-1)
    coh = np.array([scores[i] for i in range(len(leaves))], dtype=float)

    # Expect a dip around the optimum (gradients on opposite sides point toward x*),
    # which yields negative alignment for some neighbor pairs.
    idx0 = int(np.argmin(np.abs(c - 0.0)))
    coh_at_opt = float(coh[idx0])
    coh_far = float(np.median(np.concatenate([coh[:5], coh[-5:]])))

    # This is a *methodology warning*, not a numerical failure.
    if coh_at_opt < 0.70 and coh_far > 0.90:
        status = "WARN"
        details = (
            f"coh(at x≈0)={coh_at_opt:.3f} vs coh(far)≈{coh_far:.3f} "
            "(alignment-based coherence penalizes sign flips near an extremum)"
        )
    else:
        status = "PASS"
        details = f"coh(at x≈0)={coh_at_opt:.3f}, coh(far)≈{coh_far:.3f}"

    return CheckResult("Coherence dip near optimum in 1D", status, details)


def check_affine_invariance_2d() -> CheckResult:
    # Base grid
    base = _make_grid_leaves(9, dim=2, lo=-1.0, hi=1.0, optimum=np.zeros(2))
    _, pot_base, _, _, _ = compute_coherence_scores(base, k_neighbors=6)
    p0 = _potentials_array(pot_base, len(base))

    # Affine transform x' = a*x + b
    a = 10.0
    b = np.array([123.0, -57.0], dtype=float)
    transformed: List[Cube] = []
    for leaf in base:
        c = leaf.center()
        widths = leaf.widths()
        c2 = a * c + b
        w2 = a * widths
        bounds = [(float(c2[i] - w2[i] / 2), float(c2[i] + w2[i] / 2)) for i in range(2)]
        new_leaf = Cube(bounds=bounds)
        # Gradient direction should be invariant (points toward transformed optimum b).
        g = (b - new_leaf.center()).astype(float)
        if np.linalg.norm(g) < 1e-12:
            g = np.array([1e-6, 0.0], dtype=float)
        new_leaf.lgs_model = {"grad": g}
        new_leaf.n_good = 1
        transformed.append(new_leaf)

    _, pot_aff, _, _, _ = compute_coherence_scores(transformed, k_neighbors=6)
    p1 = _potentials_array(pot_aff, len(transformed))

    max_abs = float(np.max(np.abs(p0 - p1)))
    ok = max_abs < 1e-9
    status = "PASS" if ok else "FAIL"
    details = f"max|Δpotential|={max_abs:.3e} (scale+translation)"
    return CheckResult("Affine invariance (2D)", status, details)


def check_categorical_masking_effect(rng: np.random.Generator) -> CheckResult:
    # 2 continuous dims + 1 "categorical-like" dim (index 2).
    n = 7
    leaves = _make_grid_leaves(n, dim=3, lo=-1.0, hi=1.0, optimum=np.zeros(3), rng=rng)

    centers = _centers(leaves)
    # Inject a dominating gradient component on the "categorical" dim.
    for leaf in leaves:
        c = leaf.center()
        g = np.array([-c[0], -c[1], 50.0], dtype=float)  # huge 3rd component
        leaf.lgs_model = {"grad": g}
        leaf.n_good = 1

    # Without masking, potential is dominated by the fake categorical dim.
    _, pot_nomask, coh_nomask, _, _ = compute_coherence_scores(leaves, categorical_dims=None, k_neighbors=6)
    p_nomask = _potentials_array(pot_nomask, len(leaves))

    # With masking, 3rd dimension is removed from gradients/deltas.
    _, pot_mask, coh_mask, _, _ = compute_coherence_scores(leaves, categorical_dims=[(2, 5)], k_neighbors=6)
    p_mask = _potentials_array(pot_mask, len(leaves))

    # Check: masking should improve correlation with continuous distance.
    dist_xy = np.linalg.norm(centers[:, :2], axis=1)
    corr_nomask = abs(_pearson(dist_xy, p_nomask))
    corr_mask = abs(_pearson(dist_xy, p_mask))

    ok = corr_mask > corr_nomask + 0.30
    status = "PASS" if ok else "WARN"
    details = (
        f"|corr(dist_xy,pot)|: no-mask={corr_nomask:.3f} vs masked={corr_mask:.3f}; "
        f"global_coh(no-mask)={coh_nomask:.3f}, global_coh(masked)={coh_mask:.3f}"
    )
    return CheckResult("Categorical masking reduces spurious potential", status, details)


def check_disconnected_graph_warning(rng: np.random.Generator) -> CheckResult:
    # Two clusters far apart; use k=1 so the kNN graph stays disconnected.
    cluster_a = _make_grid_leaves(4, dim=2, lo=-1.0, hi=-0.5, optimum=np.array([-0.75, -0.75]), rng=rng)
    cluster_b = _make_grid_leaves(4, dim=2, lo=10.0, hi=10.5, optimum=np.array([10.25, 10.25]), rng=rng)
    leaves = cluster_a + cluster_b

    # Verify disconnection directly on the built graph.
    edges = _build_knn_graph(leaves, k=1)
    n = len(leaves)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)

    # BFS from 0
    seen = set([0])
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    connected = len(seen) == n

    # Even if disconnected, compute_coherence_scores returns *a* potential field, but
    # its values across components are not comparable (gauge per component).
    _, pot, global_coh, _, _ = compute_coherence_scores(leaves, k_neighbors=1)
    p = _potentials_array(pot, n)

    # If disconnected, raise a WARN with evidence.
    if not connected:
        # Split by cluster
        p_a = p[: len(cluster_a)]
        p_b = p[len(cluster_a) :]
        details = (
            f"kNN graph disconnected (k=1). potential stats: "
            f"clusterA mean={float(np.mean(p_a)):.3f}, clusterB mean={float(np.mean(p_b)):.3f}, "
            f"global_coh={global_coh:.3f}. Interpretation across components is ill-posed."
        )
        return CheckResult("Disconnected graph behavior", "WARN", details)

    return CheckResult("Disconnected graph behavior", "PASS", "kNN graph connected (unexpected for this setup)")


def check_neutral_when_no_gradients() -> CheckResult:
    # If no leaf has a usable gradient/model, the implementation should return
    # neutral defaults rather than raising or producing NaNs.
    leaves = _make_grid_leaves(6, dim=2, lo=-1.0, hi=1.0, optimum=np.zeros(2), set_gradients=False)
    for leaf in leaves:
        leaf.lgs_model = None
        leaf.n_good = 1

    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves, k_neighbors=6)

    coh = np.array([scores[i] for i in range(len(leaves))], dtype=float)
    pot = _potentials_array(potentials, len(leaves))

    ok = (
        np.allclose(coh, 0.5)
        and np.allclose(pot, 0.5)
        and (global_coh == 0.5)
        and (q60 == 0.5)
        and (q80 == 0.5)
    )
    status = "PASS" if ok else "FAIL"
    details = f"global_coh={global_coh}, q60={q60}, q80={q80}, coh_mean={float(np.mean(coh)):.3f}, pot_std={float(np.std(pot)):.3e}"
    return CheckResult("Neutral outputs when gradients are missing", status, details)


def check_random_field_low_structure(rng: np.random.Generator) -> CheckResult:
    # Random gradients should yield coherence near ~0.5 and potentials with
    # little relation to a geometric optimum.
    leaves = _make_grid_leaves(9, dim=2, lo=-1.0, hi=1.0, optimum=np.zeros(2), rng=rng)
    for leaf in leaves:
        g = rng.normal(0, 1.0, size=2).astype(float)
        if np.linalg.norm(g) < 1e-12:
            g = np.array([1.0, 0.0], dtype=float)
        leaf.lgs_model = {"grad": g}
        leaf.n_good = 1

    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves, k_neighbors=6)
    coh = np.array([scores[i] for i in range(len(leaves))], dtype=float)
    pot = _potentials_array(potentials, len(leaves))

    centers = _centers(leaves)
    dist = np.linalg.norm(centers, axis=1)
    corr = abs(_pearson(dist, pot))

    # "Correct" behavior here is qualitative: coherence should not be high,
    # and potential should not correlate strongly with distance.
    if (0.40 <= global_coh <= 0.60) and (corr < 0.30) and np.isfinite(global_coh):
        status = "PASS"
    else:
        status = "WARN"

    details = (
        f"global_coh={global_coh:.3f}, q60={q60:.3f}, q80={q80:.3f}, "
        f"coh_median={float(np.median(coh)):.3f}, |corr(dist,pot)|={corr:.3f}, pot_std={float(np.std(pot)):.3f}"
    )
    return CheckResult("Random-gradient field yields low structure", status, details)


def check_nan_gradient_propagation_warning() -> CheckResult:
    # If a gradient contains NaN/Inf, coherence can become NaN (no sanitization).
    # In normal ALBA runs, LGS should avoid producing NaN grads; still, this is
    # a useful numerical guard-rail to be aware of.
    leaves = []
    xs = np.linspace(0, 1, 7)
    dx = xs[1] - xs[0]
    for i, x in enumerate(xs):
        c = Cube(bounds=[(float(x - dx / 2), float(x + dx / 2)), (0.0, 1.0)])
        c.n_good = 1
        if i == 0:
            c.lgs_model = {"grad": np.array([np.nan, 0.0], dtype=float)}
        else:
            c.lgs_model = {"grad": np.array([1.0, 0.0], dtype=float)}
        leaves.append(c)

    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves, k_neighbors=2)
    coh_vals = np.array(list(scores.values()), dtype=float)

    if (not np.all(np.isfinite(coh_vals))) or (not np.isfinite(global_coh)):
        status = "WARN"
        details = (
            "Non-finite gradient can propagate to coherence/global metrics (expected with current implementation). "
            f"global_coh={global_coh}, q60={q60}, q80={q80}"
        )
    else:
        status = "PASS"
        details = "Coherence remained finite even with NaN gradient (unexpected)."

    return CheckResult("NaN gradient propagation", status, details)


def run_all_checks() -> List[CheckResult]:
    rng = np.random.default_rng(12345)
    checks = [
        check_ls_solver_recovery(rng),
        check_potential_monotonicity_2d(),
        check_coherence_valley_flip_1d(),
        check_affine_invariance_2d(),
        check_categorical_masking_effect(rng),
        check_disconnected_graph_warning(rng),
        check_neutral_when_no_gradients(),
        check_random_field_low_structure(rng),
        check_nan_gradient_propagation_warning(),
    ]
    return checks


def _render_markdown(results: Sequence[CheckResult]) -> str:
    lines = []
    lines.append("# Coherence + Potential Field — Numerical Audit\n")
    lines.append("This report is generated by `coherence_potential_numerical_audit.py`.\n")
    lines.append("## Summary\n")
    for r in results:
        lines.append(f"- **{r.status}** — {r.name}: {r.details}")
    lines.append("\n## Interpretation\n")
    lines.append(
        "- **PASS**: the check is consistent with the intended numerical/mathematical behavior.\n"
        "- **WARN**: the behavior is numerically stable but highlights a methodological limitation or a regime where interpretation can be misleading.\n"
        "- **FAIL**: indicates a strong numerical inconsistency (e.g., solver not recovering a known signal).\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default=None, help="Optional path to write a Markdown report")
    args = parser.parse_args()

    results = run_all_checks()

    # Console output
    max_name = max(len(r.name) for r in results)
    print("=" * 80)
    print("ALBA Coherence + Potential Field — Numerical Audit")
    print("=" * 80)
    for r in results:
        print(f"[{r.status:<4}] {r.name:<{max_name}}  {r.details}")

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    print("-" * 80)
    print(f"Totals: PASS={len(results) - n_fail - n_warn}, WARN={n_warn}, FAIL={n_fail}")

    if args.report:
        report_md = _render_markdown(results)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote report: {args.report}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
