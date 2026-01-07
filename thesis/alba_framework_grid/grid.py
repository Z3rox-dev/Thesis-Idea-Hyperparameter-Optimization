"""
ALBA Framework - Grid + Heatmap Module

This module implements a per-leaf sparse heatmap over an implicit grid
defined by cube bounds and a fixed number of bins per dimension (B).

It also provides lightweight candidate sampling utilities that avoid
materializing the full cartesian grid (B^d).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# First primes for Halton bases (enough for typical HPO dims).
_PRIMES: List[int] = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
]


def _radical_inverse(base: int, indices: np.ndarray) -> np.ndarray:
    """Vectorized radical inverse for Halton sequences."""
    idx = np.asarray(indices, dtype=np.int64).copy()
    out = np.zeros_like(idx, dtype=float)
    f = 1.0 / float(base)
    while np.any(idx > 0):
        out += (idx % base) * f
        idx //= base
        f /= float(base)
    return out


def halton_sequence(dim: int, n: int, start_index: int = 0) -> np.ndarray:
    """Generate n points of a Halton sequence in [0,1)^dim."""
    if n <= 0:
        return np.zeros((0, dim), dtype=float)
    if dim < 1:
        raise ValueError("halton_sequence requires dim >= 1")
    if dim > len(_PRIMES):
        raise ValueError(f"halton_sequence dim={dim} exceeds supported primes ({len(_PRIMES)})")

    idx = np.arange(start_index + 1, start_index + n + 1, dtype=np.int64)
    out = np.empty((n, dim), dtype=float)
    for j in range(dim):
        out[:, j] = _radical_inverse(_PRIMES[j], idx)
    return out


@dataclass
class CellStats:
    """Additive statistics for one grid cell."""

    n: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    n_good: float = 0.0
    n_r: float = 0.0
    sum_r: float = 0.0
    sum_r2: float = 0.0

    def add(self, y: float, is_good: bool, resid: Optional[float] = None) -> None:
        yv = float(y)
        self.n += 1.0
        self.sum_y += yv
        self.sum_y2 += yv * yv
        self.n_good += 1.0 if is_good else 0.0
        if resid is not None:
            rv = float(resid)
            if np.isfinite(rv):
                self.n_r += 1.0
                self.sum_r += rv
                self.sum_r2 += rv * rv

    def mean_y(self) -> float:
        if self.n <= 0:
            return 0.0
        return float(self.sum_y / self.n)

    def var_y(self) -> float:
        if self.n <= 1:
            return 0.0
        mean = self.sum_y / self.n
        return float(max(0.0, self.sum_y2 / self.n - mean * mean))

    def mean_r(self) -> float:
        if self.n_r <= 0:
            return 0.0
        return float(self.sum_r / self.n_r)

    def var_r(self) -> float:
        if self.n_r <= 1:
            return 0.0
        mean = self.sum_r / self.n_r
        return float(max(0.0, self.sum_r2 / self.n_r - mean * mean))


CellIndex = Tuple[int, ...]


@dataclass
class LeafGridState:
    """Sparse per-leaf grid state used for heatmap statistics and sampling."""

    bounds: List[Tuple[float, float]]
    B: int
    stats: Dict[CellIndex, CellStats] = field(default_factory=dict)
    total_visits: float = 0.0
    seq_index: int = 0  # for low-discrepancy streaming (Halton)

    _lo: np.ndarray = field(init=False, repr=False)
    _hi: np.ndarray = field(init=False, repr=False)
    _widths: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.B < 1:
            raise ValueError("LeafGridState requires B >= 1")
        self._lo = np.array([lo for lo, _ in self.bounds], dtype=float)
        self._hi = np.array([hi for _, hi in self.bounds], dtype=float)
        self._widths = np.maximum(self._hi - self._lo, 1e-12)

    @property
    def dim(self) -> int:
        return int(len(self.bounds))

    # ---------------------------------------------------------------------
    # Binning
    # ---------------------------------------------------------------------

    def cell_index(self, x: np.ndarray) -> CellIndex:
        x = np.asarray(x, dtype=float)
        idx: List[int] = []
        for j, (lo, hi) in enumerate(self.bounds):
            denom = hi - lo
            if abs(denom) < 1e-12:
                idx.append(0)
                continue
            t = float((x[j] - lo) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            ij = int(np.floor(t * self.B))
            if ij >= self.B:
                ij = self.B - 1
            elif ij < 0:
                ij = 0
            idx.append(ij)
        return tuple(idx)

    def visits_for_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected X shape (n,{self.dim}), got {tuple(X.shape)}")
        out = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            key = self.cell_index(X[i])
            st = self.stats.get(key)
            out[i] = 0.0 if st is None else float(st.n)
        return out

    # ---------------------------------------------------------------------
    # Updates / rebuild
    # ---------------------------------------------------------------------

    def update(
        self,
        x: np.ndarray,
        y_internal: float,
        gamma: float,
        *,
        y_pred: Optional[float] = None,
    ) -> None:
        key = self.cell_index(x)
        st = self.stats.get(key)
        if st is None:
            st = CellStats()
            self.stats[key] = st
        resid = None if y_pred is None else float(y_internal - float(y_pred))
        st.add(y_internal, bool(y_internal >= gamma), resid=resid)
        self.total_visits += 1.0

    def rebuild_from_tested_pairs(
        self,
        tested_pairs: Iterable[Tuple[np.ndarray, float]],
        gamma: float,
    ) -> None:
        self.stats = {}
        self.total_visits = 0.0
        for x, y_internal in tested_pairs:
            self.update(x, float(y_internal), gamma)

    # ---------------------------------------------------------------------
    # Candidate sampling
    # ---------------------------------------------------------------------

    def sample_candidates(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        mode: str = "grid_random",
        jitter: bool = True,
    ) -> np.ndarray:
        """
        Sample candidate points inside the leaf bounds.

        Modes
        -----
        - "grid_random": uniform random cell indices + optional jitter in-cell
        - "grid_halton": Halton-ordered cell centers (+ optional jitter)
        - "halton": continuous Halton points in the cube (ignores B for sampling)
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0, self.dim), dtype=float)

        if mode not in {"grid_random", "grid_halton", "halton"}:
            raise ValueError(f"Unknown sampling mode: {mode}")

        if mode == "halton":
            u = halton_sequence(self.dim, n, start_index=self.seq_index)
            self.seq_index += n
            return self._lo + u * self._widths

        if mode == "grid_random":
            idx = rng.integers(0, self.B, size=(n, self.dim), dtype=np.int64)
        else:  # "grid_halton"
            u = halton_sequence(self.dim, n, start_index=self.seq_index)
            self.seq_index += n
            idx = np.floor(u * self.B).astype(np.int64)
            idx = np.clip(idx, 0, self.B - 1)

        if jitter:
            frac = rng.random((n, self.dim))
        else:
            frac = np.full((n, self.dim), 0.5, dtype=float)

        u_cell = (idx.astype(float) + frac) / float(self.B)
        X = self._lo + u_cell * self._widths
        X = np.minimum(np.maximum(X, self._lo), self._hi)
        return X

    def sample_candidates_heatmap_ucb(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        beta: float = 1.0,
        explore_prob: float = 0.25,
        temperature: float = 1.0,
        jitter: bool = True,
    ) -> np.ndarray:
        """Sample candidates using a heatmap-driven cell bandit (UCB on visited cells).

        Strategy
        --------
        - With probability explore_prob, sample a cell uniformly at random (exploration).
        - Otherwise, sample a visited cell according to a softmax over:
            ucb = mean_y + beta * sqrt(var_y / (n + 1e-12))
        - Then sample uniformly within the chosen cell (or cell center if jitter=False).

        Notes
        -----
        - This uses only additive stats (mean/var) and is independent from gamma.
        - Unvisited cells are not enumerated; exploration is handled by explore_prob.
        """
        n = int(n)
        if n <= 0:
            return np.zeros((0, self.dim), dtype=float)

        explore_prob = float(np.clip(explore_prob, 0.0, 1.0))
        beta = float(beta)
        temperature = float(temperature)

        visited = list(self.stats.items())
        keys: List[CellIndex] = []
        probs: np.ndarray
        if visited and temperature > 0.0:
            keys = [k for k, _ in visited]
            u = np.zeros(len(visited), dtype=float)
            for i, (_, st) in enumerate(visited):
                mean = float(st.mean_y())
                se = float(np.sqrt(max(0.0, st.var_y()) / max(float(st.n), 1e-12)))
                u[i] = mean + beta * se
            u = u - float(np.max(u))
            w = np.exp(u / max(temperature, 1e-12))
            s = float(np.sum(w))
            probs = w / s if s > 0 else np.full_like(w, 1.0 / float(len(w)))
        elif visited:
            # Degenerate temperature: always pick the max-UCB visited cell.
            best_key = None
            best_u = -np.inf
            for k, st in visited:
                mean = float(st.mean_y())
                se = float(np.sqrt(max(0.0, st.var_y()) / max(float(st.n), 1e-12)))
                u = mean + beta * se
                if u > best_u or best_key is None:
                    best_u = u
                    best_key = k
            keys = [best_key] if best_key is not None else []
            probs = np.array([1.0], dtype=float)
        else:
            probs = np.zeros(0, dtype=float)

        idx = np.empty((n, self.dim), dtype=np.int64)
        for i in range(n):
            if not keys or float(rng.random()) < explore_prob:
                idx[i] = rng.integers(0, self.B, size=(self.dim,), dtype=np.int64)
                continue
            if probs.size == 1:
                idx[i] = np.array(keys[0], dtype=np.int64)
            else:
                j = int(rng.choice(len(keys), p=probs))
                idx[i] = np.array(keys[j], dtype=np.int64)

        if jitter:
            frac = rng.random((n, self.dim))
        else:
            frac = np.full((n, self.dim), 0.5, dtype=float)

        u_cell = (idx.astype(float) + frac) / float(self.B)
        X = self._lo + u_cell * self._widths
        X = np.minimum(np.maximum(X, self._lo), self._hi)
        return X
