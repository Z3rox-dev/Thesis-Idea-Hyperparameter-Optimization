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

    def add(
        self,
        y: float,
        is_good: bool,
        resid: Optional[float] = None,
        *,
        weight: float = 1.0,
    ) -> None:
        yv = float(y)
        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return
        self.n += w
        self.sum_y += w * yv
        self.sum_y2 += w * yv * yv
        self.n_good += w if is_good else 0.0
        if resid is not None:
            rv = float(resid)
            if np.isfinite(rv):
                self.n_r += w
                self.sum_r += w * rv
                self.sum_r2 += w * rv * rv

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
    B_index: Optional[int] = None  # bins per indexed dim for cell indexing; defaults to B
    B_index_coarse: Optional[int] = None  # optional coarser indexing (multi-resolution)
    index_dims: Optional[np.ndarray] = None  # indices in [0..dim-1] used for cell indexing
    soft_assignment: bool = False  # update neighboring cells with soft weights
    multi_resolution: bool = False  # maintain coarse residual stats alongside fine stats
    stats: Dict[CellIndex, CellStats] = field(default_factory=dict)
    stats_coarse: Dict[CellIndex, CellStats] = field(default_factory=dict)
    total_visits: float = 0.0
    total_visits_coarse: float = 0.0
    seq_index: int = 0  # for low-discrepancy streaming (Halton)

    _lo: np.ndarray = field(init=False, repr=False)
    _hi: np.ndarray = field(init=False, repr=False)
    _widths: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.B < 1:
            raise ValueError("LeafGridState requires B >= 1")
        if self.B_index is None:
            self.B_index = int(self.B)
        self.B_index = int(self.B_index)
        if self.B_index < 1:
            raise ValueError("LeafGridState requires B_index >= 1")
        if self.B_index > self.B:
            raise ValueError("LeafGridState requires B_index <= B")
        self.soft_assignment = bool(self.soft_assignment)
        self.multi_resolution = bool(self.multi_resolution)
        if self.multi_resolution:
            if self.B_index_coarse is None:
                if self.B_index <= 2:
                    self.B_index_coarse = int(self.B_index)
                else:
                    self.B_index_coarse = max(2, int(self.B_index // 2))
            self.B_index_coarse = int(self.B_index_coarse)
            if self.B_index_coarse < 1:
                raise ValueError("LeafGridState requires B_index_coarse >= 1")
            if self.B_index_coarse > self.B_index:
                raise ValueError("LeafGridState requires B_index_coarse <= B_index")
        self._lo = np.array([lo for lo, _ in self.bounds], dtype=float)
        self._hi = np.array([hi for _, hi in self.bounds], dtype=float)
        self._widths = np.maximum(self._hi - self._lo, 1e-12)
        if self.index_dims is not None:
            idx = np.asarray(self.index_dims, dtype=np.int64).ravel()
            if idx.ndim != 1:
                raise ValueError("index_dims must be 1D")
            if idx.size > 0:
                if int(np.min(idx)) < 0 or int(np.max(idx)) >= self.dim:
                    raise ValueError(f"index_dims must be in [0,{self.dim - 1}]")
                # Preserve order but ensure uniqueness.
                seen = set()
                uniq: List[int] = []
                for j in idx.tolist():
                    jj = int(j)
                    if jj in seen:
                        continue
                    seen.add(jj)
                    uniq.append(jj)
                idx = np.array(uniq, dtype=np.int64)
            self.index_dims = idx

    @property
    def dim(self) -> int:
        return int(len(self.bounds))

    # ---------------------------------------------------------------------
    # Binning
    # ---------------------------------------------------------------------

    def cell_index(self, x: np.ndarray, *, B_index: Optional[int] = None) -> CellIndex:
        x = np.asarray(x, dtype=float)
        idx: List[int] = []
        B_i = int(self.B_index) if B_index is None else int(B_index)
        dims: Iterable[int]
        if self.index_dims is None:
            dims = range(self.dim)
        else:
            dims = (int(j) for j in self.index_dims.tolist())
        for j in dims:
            lo, hi = self.bounds[j]
            denom = hi - lo
            if abs(denom) < 1e-12:
                idx.append(0)
                continue
            t = float((x[int(j)] - lo) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            ij = int(np.floor(t * B_i))
            if ij >= B_i:
                ij = B_i - 1
            elif ij < 0:
                ij = 0
            idx.append(ij)
        return tuple(idx)

    def _soft_keys_and_weights(self, x: np.ndarray, *, B_index: int) -> List[Tuple[CellIndex, float]]:
        x = np.asarray(x, dtype=float)
        B_i = int(max(int(B_index), 1))
        if B_i <= 1:
            return [(self.cell_index(x, B_index=B_i), 1.0)]

        dims: List[int]
        if self.index_dims is None:
            dims = list(range(self.dim))
        else:
            dims = [int(j) for j in self.index_dims.tolist()]

        per_dim: List[List[Tuple[int, float]]] = []
        for j in dims:
            lo, hi = self.bounds[j]
            denom = hi - lo
            if abs(denom) < 1e-12:
                per_dim.append([(0, 1.0)])
                continue
            t = float((x[j] - lo) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            u = t * float(B_i)
            i0 = int(np.floor(u))
            if i0 < 0:
                i0 = 0
                frac = 0.0
            elif i0 >= B_i - 1:
                i0 = B_i - 1
                frac = 0.0
            else:
                frac = float(u - float(i0))
                frac = float(np.clip(frac, 0.0, 1.0))

            i1 = min(i0 + 1, B_i - 1)
            if i1 == i0 or frac <= 1e-12 or frac >= 1.0 - 1e-12:
                per_dim.append([(i0, 1.0)])
            else:
                per_dim.append([(i0, 1.0 - frac), (i1, frac)])

        out: Dict[CellIndex, float] = {}
        keys: List[Tuple[int, ...]] = [tuple()]
        weights: List[float] = [1.0]
        for opts in per_dim:
            new_keys: List[Tuple[int, ...]] = []
            new_w: List[float] = []
            for base_key, base_w in zip(keys, weights):
                for bin_idx, w in opts:
                    new_keys.append(base_key + (int(bin_idx),))
                    new_w.append(float(base_w) * float(w))
            keys, weights = new_keys, new_w

        w_sum = float(np.sum(np.asarray(weights, dtype=float))) if weights else 0.0
        if w_sum <= 0.0:
            return [(self.cell_index(x, B_index=B_i), 1.0)]

        for k, w in zip(keys, weights):
            out[k] = out.get(k, 0.0) + float(w) / w_sum
        return [(k, float(w)) for k, w in out.items()]

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
        resid: Optional[float] = None,
    ) -> None:
        if resid is None:
            resid = None if y_pred is None else float(y_internal - float(y_pred))
        else:
            resid = float(resid)
        is_good = bool(y_internal >= gamma)

        if self.soft_assignment:
            for key, w in self._soft_keys_and_weights(x, B_index=int(self.B_index or self.B)):
                st = self.stats.get(key)
                if st is None:
                    st = CellStats()
                    self.stats[key] = st
                st.add(y_internal, is_good, resid=resid, weight=w)
        else:
            key = self.cell_index(x)
            st = self.stats.get(key)
            if st is None:
                st = CellStats()
                self.stats[key] = st
            st.add(y_internal, is_good, resid=resid)

        if self.multi_resolution:
            B_c = int(self.B_index_coarse or self.B_index or self.B)
            if self.soft_assignment:
                for key, w in self._soft_keys_and_weights(x, B_index=B_c):
                    st = self.stats_coarse.get(key)
                    if st is None:
                        st = CellStats()
                        self.stats_coarse[key] = st
                    st.add(y_internal, is_good, resid=resid, weight=w)
            else:
                key = self.cell_index(x, B_index=B_c)
                st = self.stats_coarse.get(key)
                if st is None:
                    st = CellStats()
                    self.stats_coarse[key] = st
                st.add(y_internal, is_good, resid=resid)

        self.total_visits += 1.0
        if self.multi_resolution:
            self.total_visits_coarse += 1.0

    def rebuild_from_tested_pairs(
        self,
        tested_pairs: Iterable[Tuple[np.ndarray, float]],
        gamma: float,
    ) -> None:
        self.stats = {}
        self.stats_coarse = {}
        self.total_visits = 0.0
        self.total_visits_coarse = 0.0
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
        lo: Optional[np.ndarray] = None,
        hi: Optional[np.ndarray] = None,
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

        lo_s = self._lo
        hi_s = self._hi
        if lo is not None:
            lo_in = np.asarray(lo, dtype=float).reshape(-1)
            if lo_in.shape[0] != self.dim:
                raise ValueError(f"lo must have shape ({self.dim},), got {tuple(lo_in.shape)}")
            lo_s = np.maximum(lo_s, lo_in)
        if hi is not None:
            hi_in = np.asarray(hi, dtype=float).reshape(-1)
            if hi_in.shape[0] != self.dim:
                raise ValueError(f"hi must have shape ({self.dim},), got {tuple(hi_in.shape)}")
            hi_s = np.minimum(hi_s, hi_in)
        widths_s = np.maximum(hi_s - lo_s, 1e-12)

        if mode == "halton":
            u = halton_sequence(self.dim, n, start_index=self.seq_index)
            self.seq_index += n
            return lo_s + u * widths_s

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
        X = lo_s + u_cell * widths_s
        X = np.minimum(np.maximum(X, lo_s), hi_s)
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
        lo: Optional[np.ndarray] = None,
        hi: Optional[np.ndarray] = None,
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

        index_dims = (
            np.arange(self.dim, dtype=np.int64)
            if self.index_dims is None
            else np.asarray(self.index_dims, dtype=np.int64)
        )
        B_index = int(self.B_index) if self.B_index is not None else int(self.B)

        lo_s = self._lo
        hi_s = self._hi
        if lo is not None:
            lo_in = np.asarray(lo, dtype=float).reshape(-1)
            if lo_in.shape[0] != self.dim:
                raise ValueError(f"lo must have shape ({self.dim},), got {tuple(lo_in.shape)}")
            lo_s = np.maximum(lo_s, lo_in)
        if hi is not None:
            hi_in = np.asarray(hi, dtype=float).reshape(-1)
            if hi_in.shape[0] != self.dim:
                raise ValueError(f"hi must have shape ({self.dim},), got {tuple(hi_in.shape)}")
            hi_s = np.minimum(hi_s, hi_in)
        widths_s = np.maximum(hi_s - lo_s, 1e-12)

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

        X = np.empty((n, self.dim), dtype=float)
        for i in range(n):
            explore = (not keys) or float(rng.random()) < explore_prob
            if explore:
                X[i] = self.sample_candidates(
                    rng,
                    1,
                    mode="grid_random",
                    jitter=jitter,
                    lo=lo_s,
                    hi=hi_s,
                )[0]
                continue

            key: CellIndex
            if probs.size == 1:
                key = keys[0]
            else:
                j = int(rng.choice(len(keys), p=probs))
                key = keys[j]

            # Sample within trust-region bounds; additionally constrain the indexed dims to the chosen cell
            # (intersection with trust region). Non-indexed dims remain uniformly sampled in trust region.
            x = lo_s + rng.random(self.dim) * widths_s
            if index_dims.size:
                for pos, dim_j in enumerate(index_dims.tolist()):
                    bin_idx = int(key[pos])
                    cell_lo = self._lo[dim_j] + (float(bin_idx) / float(B_index)) * self._widths[dim_j]
                    cell_hi = self._lo[dim_j] + (float(bin_idx + 1) / float(B_index)) * self._widths[dim_j]
                    lo_j = max(float(lo_s[dim_j]), float(cell_lo))
                    hi_j = min(float(hi_s[dim_j]), float(cell_hi))
                    if hi_j <= lo_j + 1e-15:
                        continue
                    if jitter:
                        x[dim_j] = float(lo_j + rng.random() * (hi_j - lo_j))
                    else:
                        x[dim_j] = float(0.5 * (lo_j + hi_j))
            X[i] = np.minimum(np.maximum(x, lo_s), hi_s)

        return X
