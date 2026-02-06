#!/usr/bin/env python3
"""
Sanity / coherence checks for CopulaHPO v2 (without modifying the implementation).

Focus:
- marginal round-trip consistency (to_uniform -> from_uniform)
- numeric stability (no NaN/inf in transforms / Cholesky)
- copula sampling sanity (u-marginals approximately uniform; corr PD)
- API invariants for mixed spaces (types / bounds respected)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, "/mnt/workspace/thesis/nuovo_progetto")

from copula_hpo_v2 import (  # noqa: E402
    CategoricalMarginal,
    ContinuousMarginal,
    CopulaHPO,
    CopulaHPO_Continuous,
    HyperparameterSpec,
    IntegerMarginal,
    _nearest_pd,
    _norm_cdf,
    _norm_ppf,
)


@dataclass(frozen=True)
class _TestResult:
    name: str
    ok: bool
    detail: str


def _ok(name: str, detail: str = "") -> _TestResult:
    return _TestResult(name=name, ok=True, detail=detail)


def _bad(name: str, detail: str) -> _TestResult:
    return _TestResult(name=name, ok=False, detail=detail)


def _assert_finite(name: str, x: np.ndarray, *, what: str) -> _TestResult | None:
    if not np.all(np.isfinite(x)):
        bad = np.flatnonzero(~np.isfinite(x))
        head = bad[:10].tolist()
        return _bad(name, f"{what} contains non-finite values at indices {head}")
    return None


def test_continuous_roundtrip() -> _TestResult:
    name = "continuous_roundtrip"
    lower, upper = -5.0, 5.0
    rng = np.random.default_rng(0)

    values = rng.uniform(lower, upper, size=5000)
    m = ContinuousMarginal(lower, upper, u_clip=1e-9)
    m.fit(values)

    probe = rng.uniform(lower * 0.8, upper * 0.8, size=2000)  # avoid clipping tails
    u = m.to_uniform(probe)
    x_back = m.from_uniform(u)

    if (r := _assert_finite(name, u, what="u")) is not None:
        return r
    if (r := _assert_finite(name, x_back, what="x_back")) is not None:
        return r

    mae = float(np.mean(np.abs(x_back - probe)))
    mx = float(np.max(np.abs(x_back - probe)))
    if mae > 1e-10 or mx > 1e-7:
        # With Gaussian cdf/ppf + no clipping, this should be ~exact up to float error.
        return _bad(name, f"round-trip too lossy: mae={mae:.3g}, max_abs={mx:.3g}")
    return _ok(name, f"mae={mae:.3g}, max_abs={mx:.3g}")


def test_categorical_roundtrip() -> _TestResult:
    name = "categorical_roundtrip"
    cats = ["adam", "sgd", "rmsprop", "lion"]
    rng = np.random.default_rng(1)
    values = rng.choice(cats, size=2000, replace=True)

    m = CategoricalMarginal(cats, u_clip=1e-9)
    m.fit(values)
    u = m.to_uniform(values)
    back = m.from_uniform(u)

    if (r := _assert_finite(name, u, what="u")) is not None:
        return r
    if not np.all(back == values):
        mism = int(np.sum(back != values))
        return _bad(name, f"{mism}/{len(values)} mismatches in round-trip")
    return _ok(name, "exact round-trip on fitted categories")


def test_integer_roundtrip() -> _TestResult:
    name = "integer_roundtrip"
    lower, upper = 1, 7
    rng = np.random.default_rng(2)
    values = rng.integers(lower, upper + 1, size=2000)

    m = IntegerMarginal(lower, upper, u_clip=1e-9)
    m.fit(values)
    u = m.to_uniform(values)
    back = m.from_uniform(u)

    if (r := _assert_finite(name, u, what="u")) is not None:
        return r
    if not np.all(back.astype(int) == values.astype(int)):
        mism = int(np.sum(back.astype(int) != values.astype(int)))
        return _bad(name, f"{mism}/{len(values)} mismatches in round-trip")
    return _ok(name, "exact round-trip on fitted integers")


def test_norm_cdf_ppf_consistency() -> _TestResult:
    name = "norm_cdf_ppf_consistency"
    rng = np.random.default_rng(3)
    z = rng.normal(size=20000)
    u = _norm_cdf(z)
    if (r := _assert_finite(name, u, what="u")) is not None:
        return r
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    z2 = _norm_ppf(u)
    if (r := _assert_finite(name, z2, what="z2")) is not None:
        return r
    err = float(np.mean(np.abs(z2 - z)))
    if err > 2e-10:
        return _bad(name, f"unexpected inversion error mean_abs={err:.3g}")
    return _ok(name, f"mean_abs={err:.3g}")


def test_nearest_pd_basic() -> _TestResult:
    name = "nearest_pd_basic"
    rng = np.random.default_rng(4)
    d = 20
    A = rng.normal(size=(d, d))
    A = 0.5 * (A + A.T)
    # Make it indefinite on purpose
    A[0, 1] = 5.0
    A[1, 0] = 5.0

    B = _nearest_pd(A, min_eig=1e-6)
    if (r := _assert_finite(name, B, what="B")) is not None:
        return r
    w = np.linalg.eigvalsh(0.5 * (B + B.T))
    if float(np.min(w)) < 0.0:
        return _bad(name, f"output not PSD: min_eig={float(np.min(w)):.3g}")
    return _ok(name, f"min_eig={float(np.min(w)):.3g}, max_eig={float(np.max(w)):.3g}")


def _fit_corr_from_elite(U: np.ndarray, *, alpha_corr: float, reg: float) -> tuple[np.ndarray, float]:
    U = np.clip(U, 1e-12, 1.0 - 1e-12)
    Z = _norm_ppf(U)
    Z = Z - Z.mean(axis=0, keepdims=True)
    cov = np.cov(Z.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    alpha = max(alpha_corr, min(0.9, corr.shape[0] / max(U.shape[0], 1)))
    corr = (1.0 - alpha) * corr + alpha * np.eye(corr.shape[0])
    corr = corr + reg * np.eye(corr.shape[0])
    corr = _nearest_pd(corr, min_eig=reg)
    w = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    cond = float("inf") if float(np.min(w)) <= 0.0 else float(np.max(w) / np.min(w))
    return corr, cond


def test_copula_u_marginals_are_uniformish() -> _TestResult:
    """
    If corr is (approximately) a correlation matrix, and z ~ N(0, corr),
    then u=Phi(z) should have per-dimension marginals close to Uniform(0,1).
    """
    name = "copula_u_uniformish"
    rng = np.random.default_rng(5)
    n, d = 200, 15

    # Create pseudo-observations U that are (approximately) uniform, but correlated.
    # Use a known corr, then map to U via Phi.
    A = rng.normal(size=(d, d))
    corr_true = np.corrcoef(A @ A.T + 1e-3 * np.eye(d))  # quick SPD-ish seed
    corr_true = 0.5 * (corr_true + corr_true.T)
    np.fill_diagonal(corr_true, 1.0)
    L = np.linalg.cholesky(_nearest_pd(corr_true, min_eig=1e-6))
    z = rng.normal(size=(n, d)) @ L.T
    U = _norm_cdf(z)

    corr_fit, cond = _fit_corr_from_elite(U, alpha_corr=0.1, reg=1e-6)
    try:
        Lfit = np.linalg.cholesky(corr_fit)
    except np.linalg.LinAlgError as e:
        return _bad(name, f"cholesky failed: {type(e).__name__}: {e}")

    # Sample and check u stats.
    z_s = rng.normal(size=(20000, d)) @ Lfit.T
    u_s = _norm_cdf(z_s)
    if (r := _assert_finite(name, u_s, what="u_s")) is not None:
        return r

    mean = u_s.mean(axis=0)
    std = u_s.std(axis=0)
    mean_err = float(np.max(np.abs(mean - 0.5)))
    std_err = float(np.max(np.abs(std - math.sqrt(1.0 / 12.0))))

    # Tolerances: if diag deviates a bit from 1, std shifts; mean stays ~0.5.
    if mean_err > 0.01 or std_err > 0.02:
        return _bad(
            name,
            f"u marginals off: max|mean-0.5|={mean_err:.3g}, max|std-0.289|={std_err:.3g}, corr_cond~{cond:.3g}",
        )
    return _ok(
        name,
        f"max|mean-0.5|={mean_err:.3g}, max|std-0.289|={std_err:.3g}, corr_cond~{cond:.3g}",
    )


def test_optimizer_api_continuous_no_nans() -> _TestResult:
    name = "optimizer_continuous_no_nans"
    rng = np.random.default_rng(6)
    d = 12
    bounds = [(-5.0, 5.0)] * d
    opt = CopulaHPO_Continuous(bounds, seed=0, M=128, top_eval=4)

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    best = float("inf")
    for _ in range(300):
        x = opt.ask()
        if not np.all(np.isfinite(x)):
            return _bad(name, "ask() returned non-finite x")
        if np.any(x < -5.0 - 1e-9) or np.any(x > 5.0 + 1e-9):
            return _bad(name, "ask() returned x outside bounds")
        y = sphere(x)
        opt.tell(x, y)
        best = min(best, y)
        if not np.isfinite(best):
            return _bad(name, "best became non-finite")

    if not np.isfinite(opt.best_y) or opt.best_y < 0.0:
        return _bad(name, f"invalid best_y={opt.best_y}")
    # Not a performance test, but it should usually improve vs random init.
    return _ok(name, f"ran 300 evals; best_y={opt.best_y:.4g}")


def test_optimizer_api_latent_cma_active_no_nans() -> _TestResult:
    """
    Smoke test for latent CMA mode with active/contrastive covariance update.
    Not a performance test; just checks it runs and stays finite.
    """
    name = "optimizer_latent_cma_active_no_nans"
    d = 12
    bounds = [(-5.0, 5.0)] * d
    budget = 250
    opt = CopulaHPO_Continuous(
        bounds,
        seed=0,
        mode="latent_cma",
        budget=budget,
        cma_active=True,
        cma_active_eta=0.3,
    )

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    best = float("inf")
    for _ in range(budget):
        x = opt.ask()
        if not np.all(np.isfinite(x)):
            return _bad(name, "ask() returned non-finite x")
        if np.any(x < -5.0 - 1e-9) or np.any(x > 5.0 + 1e-9):
            return _bad(name, "ask() returned x outside bounds")
        y = sphere(x)
        opt.tell(x, y)
        best = min(best, y)
        if not np.isfinite(best):
            return _bad(name, "best became non-finite")

    if not np.isfinite(opt.best_y) or opt.best_y < 0.0:
        return _bad(name, f"invalid best_y={opt.best_y}")
    return _ok(name, f"ran {budget} evals; best_y={opt.best_y:.4g}")


def test_optimizer_api_mixed_types() -> _TestResult:
    name = "optimizer_mixed_types"
    specs = [
        HyperparameterSpec("lr", "continuous", (1e-5, 1e-1)),
        HyperparameterSpec("opt", "categorical", ["adam", "sgd", "rmsprop"]),
        HyperparameterSpec("layers", "integer", (1, 5)),
        HyperparameterSpec("drop", "continuous", (0.0, 0.5)),
    ]
    opt = CopulaHPO(specs, seed=7)

    def obj(x: dict[str, Any]) -> float:
        return float((math.log10(float(x["lr"])) + 3.0) ** 2 + (int(x["layers"]) - 3) ** 2 + (float(x["drop"]) - 0.2) ** 2)

    for _ in range(200):
        x = opt.ask()
        # Keys
        if set(x.keys()) != {"lr", "opt", "layers", "drop"}:
            return _bad(name, f"wrong keys: {sorted(x.keys())}")
        # Types / bounds
        lr = float(x["lr"])
        if not (1e-5 <= lr <= 1e-1):
            return _bad(name, f"lr out of bounds: {lr}")
        if x["opt"] not in ["adam", "sgd", "rmsprop"]:
            return _bad(name, f"opt invalid: {x['opt']}")
        layers = int(x["layers"])
        if not (1 <= layers <= 5):
            return _bad(name, f"layers out of bounds: {layers}")
        drop = float(x["drop"])
        if not (0.0 <= drop <= 0.5):
            return _bad(name, f"dropout out of bounds: {drop}")

        y = obj(x)
        if not np.isfinite(y):
            return _bad(name, "objective produced non-finite y")
        opt.tell(x, y)

    if opt.best_x is None or not np.isfinite(opt.best_y):
        return _bad(name, "best not tracked correctly")
    return _ok(name, f"ran 200 evals; best_y={opt.best_y:.4g}")


def test_optimizer_api_elite_ratio_mixed_types_no_nans() -> _TestResult:
    """
    Smoke test for CopulaHPO in elite_ratio mode on a mixed space.
    Ensures sampling/scoring doesn't produce NaNs and respects bounds/types.
    """
    name = "optimizer_elite_ratio_mixed_types_no_nans"
    specs = [
        HyperparameterSpec("lr", "continuous", (1e-5, 1e-1)),
        HyperparameterSpec("opt", "categorical", ["adam", "sgd", "rmsprop"]),
        HyperparameterSpec("layers", "integer", (1, 5)),
        HyperparameterSpec("drop", "continuous", (0.0, 0.5)),
    ]
    opt = CopulaHPO(specs, seed=11, mode="elite_ratio", M=128, top_eval=4)

    def obj(x: dict[str, Any]) -> float:
        # Best: lr=0.001, adam, 3 layers, dropout=0.2
        score = 0.0
        score += (math.log10(float(x["lr"])) + 3.0) ** 2
        score += {"adam": 0.0, "sgd": 1.0, "rmsprop": 0.5}[x["opt"]]
        score += (int(x["layers"]) - 3) ** 2
        score += (float(x["drop"]) - 0.2) ** 2
        return float(score)

    for _ in range(250):
        x = opt.ask()
        lr = float(x["lr"])
        if not (1e-5 <= lr <= 1e-1):
            return _bad(name, f"lr out of bounds: {lr}")
        if x["opt"] not in ["adam", "sgd", "rmsprop"]:
            return _bad(name, f"opt invalid: {x['opt']}")
        layers = int(x["layers"])
        if not (1 <= layers <= 5):
            return _bad(name, f"layers out of bounds: {layers}")
        drop = float(x["drop"])
        if not (0.0 <= drop <= 0.5):
            return _bad(name, f"dropout out of bounds: {drop}")

        y = obj(x)
        if not np.isfinite(y):
            return _bad(name, "objective produced non-finite y")
        opt.tell(x, y)

    if opt.best_x is None or not np.isfinite(opt.best_y):
        return _bad(name, "best not tracked correctly")
    return _ok(name, f"ran 250 evals; best_y={opt.best_y:.4g}")


def test_degenerate_continuous_bounds_behavior() -> _TestResult:
    """
    Degenerate bounds (lower==upper) are common in conditional spaces.
    This is a robustness test: ideally should return the fixed value without NaNs.
    """
    name = "degenerate_bounds"
    m = ContinuousMarginal(1.234, 1.234, u_clip=1e-6)
    m.fit(np.array([1.234, 1.234]))
    u = m.to_uniform(np.array([1.234]))
    x = m.from_uniform(np.array([0.5]))
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(x)):
        return _bad(name, f"non-finite with degenerate bounds: u={u}, x={x}")
    if float(x[0]) != 1.234:
        return _bad(name, f"expected fixed value 1.234, got {float(x[0])}")
    return _ok(name, f"u={float(u[0]):.3g}, x={float(x[0]):.3g}")


def main() -> int:
    tests = [
        test_norm_cdf_ppf_consistency,
        test_nearest_pd_basic,
        test_continuous_roundtrip,
        test_categorical_roundtrip,
        test_integer_roundtrip,
        test_copula_u_marginals_are_uniformish,
        test_optimizer_api_continuous_no_nans,
        test_optimizer_api_latent_cma_active_no_nans,
        test_optimizer_api_mixed_types,
        test_optimizer_api_elite_ratio_mixed_types_no_nans,
        test_degenerate_continuous_bounds_behavior,
    ]

    results: list[_TestResult] = []
    for fn in tests:
        try:
            results.append(fn())
        except Exception as e:
            results.append(_bad(fn.__name__, f"{type(e).__name__}: {e}"))

    ok = sum(1 for r in results if r.ok)
    bad = len(results) - ok
    print("\nCopulaHPO v2 sanity report")
    print("-" * 70)
    for r in results:
        status = "OK " if r.ok else "FAIL"
        detail = f" - {r.detail}" if r.detail else ""
        print(f"{status:4s} {r.name}{detail}")
    print("-" * 70)
    print(f"Summary: {ok}/{len(results)} OK, {bad} FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
