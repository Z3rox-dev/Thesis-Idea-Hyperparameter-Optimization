"""
ALBA Framework - Categorical Sampling Module

This module implements categorical-specific sampling using:
- Exchangeable pooling across symmetric categorical dimensions
- Per-dimension contrastive odds (good vs bad) with curiosity bonuses

These strategies help ALBA explore mixed continuous-categorical spaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cube import Cube


class CategoricalSampler:
    """Categorical sampling with exchangeable pooling and contrastive odds."""

    def __init__(
        self,
        categorical_dims: List[Tuple[int, int]],
        curiosity_bonus: float = 0.3,
    ) -> None:
        """
        Initialize the categorical sampler.

        Parameters
        ----------
        categorical_dims : List[Tuple[int, int]]
            List of (dim_index, n_choices) for each categorical dimension.
        curiosity_bonus : float
            Weight for curiosity bonus (inverse visit count).
        """
        self.categorical_dims = categorical_dims
        self.curiosity_bonus = curiosity_bonus

        # Global categorical stats: {dim_idx: {val_idx: (n_good, n_total)}}
        self._global_stats: Dict[int, Dict[int, Tuple[int, int]]] = {}
        self._dim_group: Dict[int, int] = {}
        self._group_sizes: Dict[int, int] = {}
        self._group_dims: Dict[int, List[int]] = {}
        self._type_stats: Dict[int, Dict[int, Tuple[int, int]]] = {}
        self._group_key_visits: Dict[int, Dict[Tuple[int, ...], int]] = {}
        self._group_key_stats: Dict[int, Dict[Tuple[int, ...], Tuple[int, int]]] = {}
        self._group_elites: Dict[int, List[Tuple[int, ...]]] = {}
        self._group_strength: Dict[int, float] = {}
        self._baseline_good_total = 0
        self._baseline_total = 0
        self._baseline_good_rate = 0.0

        groups: Dict[int, List[int]] = {}
        for dim_idx, n_choices in categorical_dims:
            groups.setdefault(n_choices, []).append(dim_idx)
        self._group_dims = {k: list(v) for k, v in groups.items()}
        for n_choices, dims in groups.items():
            for dim_idx in dims:
                self._dim_group[dim_idx] = n_choices
            self._group_sizes[n_choices] = len(dims)
            if len(dims) >= 2:
                self._group_key_visits.setdefault(n_choices, {})



    def _update_group_strength(self, n_choices: int) -> None:
        stats = self._group_key_stats.get(n_choices)
        baseline = float(self._baseline_good_rate)
        if not stats or baseline >= 1.0:
            self._group_strength[n_choices] = 0.0
            return

        best_mean = -np.inf
        best_total = 0
        for n_good, n_total in stats.values():
            mean = (float(n_good) + 1.0) / (float(n_total) + 2.0)
            if mean > best_mean:
                best_mean = mean
                best_total = int(n_total)

        denom = 1.0 - baseline
        lift = 0.0 if denom <= 0 else max(0.0, (best_mean - baseline) / denom)
        strength = lift * (float(best_total) / (float(best_total) + 1.0))
        self._group_strength[n_choices] = float(np.clip(strength, 0.0, 1.0))

    @property
    def has_categoricals(self) -> bool:
        """Check if there are any categorical dimensions."""
        return len(self.categorical_dims) > 0

    # -------------------------------------------------------------------------
    # Discretization helpers
    # -------------------------------------------------------------------------

    def discretize(self, x_val: float, n_choices: int) -> int:
        """
        Convert continuous [0,1] value to discrete category index.

        Parameters
        ----------
        x_val : float
            Value in [0, 1].
        n_choices : int
            Number of categorical choices.

        Returns
        -------
        int
            Category index in [0, n_choices - 1].
        """
        if n_choices <= 1:
            return 0
        idx = int(np.floor(float(x_val) * n_choices))
        if idx < 0:
            idx = 0
        elif idx >= n_choices:
            idx = n_choices - 1
        return idx

    def to_continuous(self, val_idx: int, n_choices: int) -> float:
        """
        Convert discrete category index to continuous [0,1] value.

        Parameters
        ----------
        val_idx : int
            Category index.
        n_choices : int
            Number of categorical choices.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if n_choices <= 1:
            return 0.5
        return (float(val_idx) + 0.5) / float(n_choices)

    # -------------------------------------------------------------------------
    # Statistics tracking
    # -------------------------------------------------------------------------

    def record_observation(
        self,
        x: np.ndarray,
        is_good: bool,
    ) -> None:
        """
        Record an observation for categorical tracking.

        Parameters
        ----------
        x : np.ndarray
            Evaluated point.
        is_good : bool
            Whether the observation is "good" (above gamma).
        """
        if not self.has_categoricals:
            return

        self._baseline_total += 1
        if is_good:
            self._baseline_good_total += 1
        self._baseline_good_rate = float(self._baseline_good_total) / float(self._baseline_total)

        # Update global categorical stats
        for dim_idx, n_choices in self.categorical_dims:
            val_idx = self.discretize(x[dim_idx], n_choices)
            if dim_idx not in self._global_stats:
                self._global_stats[dim_idx] = {}
            n_g, n_t = self._global_stats[dim_idx].get(val_idx, (0, 0))
            self._global_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

            group_key = self._dim_group.get(dim_idx)
            if group_key is not None and self._group_sizes.get(group_key, 0) >= 2:
                if group_key not in self._type_stats:
                    self._type_stats[group_key] = {}
                g_g, g_t = self._type_stats[group_key].get(val_idx, (0, 0))
                self._type_stats[group_key][val_idx] = (g_g + (1 if is_good else 0), g_t + 1)

        # Track joint keys for exchangeable groups (captures interactions)
        for n_choices, dims in self._group_dims.items():
            if len(dims) < 2:
                continue
            key = tuple(self.discretize(x[d], n_choices) for d in dims)
            visits = self._group_key_visits.setdefault(n_choices, {})
            visits[key] = visits.get(key, 0) + 1
            stats = self._group_key_stats.setdefault(n_choices, {})
            n_good, n_total = stats.get(key, (0, 0))
            stats[key] = (n_good + (1 if is_good else 0), n_total + 1)
            self._update_group_strength(n_choices)


    def recompute_global_stats(
        self,
        X_all: List[np.ndarray],
        y_all: List[float],
        gamma: float,
    ) -> None:
        """Recompute global categorical stats based on current gamma."""
        self._global_stats = {}
        self._type_stats = {}
        self._group_elites = {}
        self._group_key_stats = {}
        self._group_strength = {}
        self._baseline_good_total = 0
        self._baseline_total = 0
        self._baseline_good_rate = 0.0
        if not self.has_categoricals:
            return
        good_pairs: List[Tuple[np.ndarray, float]] = []
        for x, y in zip(X_all, y_all):
            is_good = y >= gamma
            self._baseline_total += 1
            if is_good:
                self._baseline_good_total += 1
            if is_good:
                good_pairs.append((x, float(y)))
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self.discretize(x[dim_idx], n_choices)
                if dim_idx not in self._global_stats:
                    self._global_stats[dim_idx] = {}
                n_g, n_t = self._global_stats[dim_idx].get(val_idx, (0, 0))
                self._global_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

                group_key = self._dim_group.get(dim_idx)
                if group_key is not None and self._group_sizes.get(group_key, 0) >= 2:
                    if group_key not in self._type_stats:
                        self._type_stats[group_key] = {}
                    g_g, g_t = self._type_stats[group_key].get(val_idx, (0, 0))
                    self._type_stats[group_key][val_idx] = (g_g + (1 if is_good else 0), g_t + 1)

            for n_choices, dims in self._group_dims.items():
                if len(dims) < 2:
                    continue
                key = tuple(self.discretize(x[d], n_choices) for d in dims)
                stats = self._group_key_stats.setdefault(n_choices, {})
                n_good, n_total = stats.get(key, (0, 0))
                stats[key] = (n_good + (1 if is_good else 0), n_total + 1)

        if self._baseline_total > 0:
            self._baseline_good_rate = float(self._baseline_good_total) / float(self._baseline_total)

        # Build elite joint keys per exchangeable group (score-ordered, unique)
        if good_pairs:
            for n_choices, dims in self._group_dims.items():
                if len(dims) < 2:
                    continue
                best_score: Dict[Tuple[int, ...], float] = {}
                for x, y in good_pairs:
                    key = tuple(self.discretize(x[d], n_choices) for d in dims)
                    prev = best_score.get(key)
                    if prev is None or y > prev:
                        best_score[key] = y
                if not best_score:
                    continue
                elites = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
                keep = min(12, len(elites))
                self._group_elites[n_choices] = [k for k, _ in elites[:keep]]

        for n_choices in self._group_key_stats:
            self._update_group_strength(n_choices)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def _evidence_weight(self, good: np.ndarray, total: np.ndarray) -> float:
        """Scale exploitation based on the spread and precision of success rates."""
        total_sum = float(np.sum(total))
        if total_sum <= 0:
            return 0.0
        p = (good + 1.0) / (total + 2.0)
        p0 = (float(np.sum(good)) + 1.0) / (total_sum + 2.0)
        weights = total + 1.0
        var = float(np.average((p - p0) ** 2, weights=weights))
        return min(1.0, np.sqrt(var) / 0.5)

    def sample(
        self,
        x: np.ndarray,
        cube: "Cube",
        rng: np.random.Generator,
        stagnating: bool = False,
    ) -> np.ndarray:
        """
        Apply categorical sampling to a candidate point.

        Parameters
        ----------
        x : np.ndarray
            Base candidate point.
        cube : Cube
            Current cube for local statistics.
        rng : np.random.Generator
            Random number generator.
        stagnating : bool
            If True, increase exploration.

        Returns
        -------
        np.ndarray
            Point with categorical dimensions sampled.
        """
        if not self.has_categoricals:
            return x

        x = x.copy()

        # Joint sampling for exchangeable groups (captures interactions); otherwise
        # leave candidate-proposed values untouched for those dims.
        handled_dims = set()
        for n_choices, dims in self._group_dims.items():
            if len(dims) < 2:
                continue
            for dim_idx in dims:
                x[dim_idx] = self.to_continuous(
                    self.discretize(x[dim_idx], n_choices), n_choices
                )
            elites = self._group_elites.get(n_choices, [])
            if len(elites) < 2:
                handled_dims.update(dims)
                continue
            strength = float(self._group_strength.get(n_choices, 0.0))
            if strength <= 0.0:
                handled_dims.update(dims)
                continue
            prob = min(0.25, 1.0 / max(1, len(dims)))
            if stagnating:
                prob = min(0.5, prob * 2.0)
            if rng.random() < prob:
                key = self._sample_group_key(n_choices, dims, rng, stagnating)
                if key is None:
                    handled_dims.update(dims)
                    continue
                max_drop = min(2, len(dims) - 1)
                dropout_prob = float(min(0.5, 1.0 - float(np.sqrt(strength))))
                n_drop = 0
                if max_drop == 1:
                    n_drop = 1 if rng.random() < dropout_prob else 0
                elif max_drop >= 2:
                    n_drop = int(rng.binomial(2, dropout_prob))
                    n_drop = min(n_drop, max_drop)

                drop_dims = set()
                if n_drop > 0:
                    drop_dims = set(int(d) for d in rng.choice(dims, size=n_drop, replace=False))

                for dim_idx, val_idx in zip(dims, key):
                    if dim_idx in drop_dims:
                        continue
                    x[dim_idx] = self.to_continuous(int(val_idx), n_choices)
                handled_dims.update(d for d in dims if d not in drop_dims)
            else:
                handled_dims.update(dims)

        mix = 0.2 if stagnating else 0.1
        for dim_idx, n_choices in self.categorical_dims:
            if dim_idx in handled_dims:
                continue
            stats_local = cube.cat_stats.get(dim_idx, {})
            local_total = sum(v[1] for v in stats_local.values())
            use_local = local_total > 0
            group_key = self._dim_group.get(dim_idx)
            if group_key is not None and self._group_sizes.get(group_key, 0) >= 2:
                stats_ref = self._type_stats.get(group_key, {})
            else:
                stats_ref = self._global_stats.get(dim_idx, {})

            n_good = np.zeros(n_choices, dtype=float)
            n_total = np.zeros(n_choices, dtype=float)
            for v in range(n_choices):
                if use_local:
                    n_g, n_t = stats_local.get(v, (0, 0))
                else:
                    n_g, n_t = 0, 0
                g_g, g_t = stats_ref.get(v, (0, 0))
                n_g += g_g
                n_t += g_t
                n_good[v] = float(n_g)
                n_total[v] = float(n_t)

            if n_choices <= 1:
                choice = 0
            else:
                signal = self._evidence_weight(n_good, n_total)
                bad = n_total - n_good
                odds = (n_good + 1.0) / (bad + 1.0)
                scores = (1.0 - signal) + signal * odds
                scores = scores + self.curiosity_bonus / (1.0 + n_total)
                total = float(scores.sum())
                if not np.isfinite(total) or total <= 0:
                    probs = np.full(n_choices, 1.0 / n_choices)
                else:
                    probs = scores / total
                if mix > 0:
                    probs = (1.0 - mix) * probs + mix / n_choices
                    probs = probs / probs.sum()
                choice = int(rng.choice(n_choices, p=probs))

            x[dim_idx] = self.to_continuous(choice, n_choices)

        return x

    def _sample_group_key(
        self,
        n_choices: int,
        dims: List[int],
        rng: np.random.Generator,
        stagnating: bool,
    ) -> Optional[Tuple[int, ...]]:
        elites = self._group_elites.get(n_choices, [])
        n_pos = len(dims)
        if n_pos <= 0 or n_choices <= 0:
            return None

        visits = self._group_key_visits.get(n_choices, {})
        stats = self._group_key_stats.get(n_choices, {})
        crossover_prob = 1.0 / max(1, n_pos)
        if stagnating:
            crossover_prob = min(1.0, crossover_prob * 2.0)
        mutation_prob = 0.5 / max(1, n_pos)
        if stagnating:
            mutation_prob = min(0.5, mutation_prob * 2.0)

        candidates: List[Tuple[int, ...]] = []
        n_candidates = 5
        for _ in range(n_candidates):
            if elites and len(elites) >= 2 and rng.random() < crossover_prob:
                n_elite = len(elites)
                weights = 1.0 / np.arange(1, n_elite + 1, dtype=float)
                weights = weights / weights.sum()
                i1, i2 = rng.choice(n_elite, size=2, replace=False, p=weights)
                p1 = elites[int(i1)]
                p2 = elites[int(i2)]
                key = []
                for j in range(n_pos):
                    v = int(p1[j] if rng.random() < 0.5 else p2[j])
                    if rng.random() < mutation_prob:
                        v = int(rng.integers(0, n_choices))
                    key.append(v)
                candidates.append(tuple(key))
                continue

            if elites:
                n_elite = len(elites)
                weights = 1.0 / np.arange(1, n_elite + 1, dtype=float)
                weights = weights / weights.sum()
                parent = elites[int(rng.choice(n_elite, p=weights))]
                key = list(parent)
                for j in range(n_pos):
                    if rng.random() < mutation_prob:
                        key[j] = int(rng.integers(0, n_choices))
                candidates.append(tuple(int(v) for v in key))
                continue

            candidates.append(tuple(int(rng.integers(0, n_choices)) for _ in range(n_pos)))

        if not candidates:
            return None
        scores: List[float] = []
        for key in candidates:
            n_good, n_total = stats.get(key, (0, 0))
            a = float(n_good) + 1.0
            b = float(max(0, n_total - n_good)) + 1.0
            scores.append(float(rng.beta(a, b)))

        order = np.argsort(np.asarray(scores))[::-1]
        top_k = 5 if stagnating else 3
        top_k = int(min(top_k, len(candidates)))
        top = [candidates[int(i)] for i in order[:top_k]]
        min_visit = min(visits.get(k, 0) for k in top)
        best = [k for k in top if visits.get(k, 0) == min_visit]
        return best[int(rng.integers(0, len(best)))]

    def update_cube_stats(
        self,
        cube: "Cube",
        x: np.ndarray,
        is_good: bool,
    ) -> None:
        """Update categorical stats for a cube with a single observation."""
        for dim_idx, n_choices in self.categorical_dims:
            val_idx = self.discretize(x[dim_idx], n_choices)
            if dim_idx not in cube.cat_stats:
                cube.cat_stats[dim_idx] = {}
            n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
            cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

    def recompute_cube_cat_stats(
        self,
        cube: "Cube",
        gamma: float,
    ) -> None:
        """
        Recompute categorical statistics for a cube from its tested pairs.

        Parameters
        ----------
        cube : Cube
            The cube to update.
        gamma : float
            Current threshold for "good" points.
        """
        cube.cat_stats = {}
        for dim_idx, _ in self.categorical_dims:
            cube.cat_stats[dim_idx] = {}

        for pt, sc in cube.tested_pairs:
            is_good = sc >= gamma
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self.discretize(pt[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (
                    n_g + (1 if is_good else 0),
                    n_t + 1,
                )
