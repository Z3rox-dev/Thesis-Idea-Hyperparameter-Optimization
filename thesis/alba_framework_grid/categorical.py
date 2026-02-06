"""
ALBA Framework - Categorical Sampling Module

This module implements categorical-specific sampling strategies including:
- Curiosity-driven exploration (bonus for unvisited combinations)
- Thompson Sampling for categorical dimensions
- Elite crossover (genetic-style recombination of top configurations)

These strategies help ALBA efficiently explore mixed continuous-categorical
search spaces common in hyperparameter optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cube import Cube


class CategoricalSampler:
    """
    Categorical sampling with curiosity-driven exploration and elite crossover.

    This sampler maintains statistics about visited categorical combinations
    and uses Thompson Sampling with curiosity bonuses to balance exploration
    and exploitation in categorical dimensions.

    Attributes
    ----------
    categorical_dims : List[Tuple[int, int]]
        List of (dim_index, n_choices) for each categorical dimension.
    elite_size : int
        Maximum number of elite configurations to maintain.
    curiosity_bonus : float
        Bonus weight for unvisited combinations.
    crossover_rate : float
        Base probability of using elite crossover.
    """

    def __init__(
        self,
        categorical_dims: List[Tuple[int, int]],
        elite_size: int = 10,
        curiosity_bonus: float = 0.3,
        crossover_rate: float = 0.15,
    ) -> None:
        """
        Initialize the categorical sampler.

        Parameters
        ----------
        categorical_dims : List[Tuple[int, int]]
            List of (dim_index, n_choices) for each categorical dimension.
        elite_size : int
            Maximum number of elite configurations to track.
        curiosity_bonus : float
            Weight for curiosity bonus (inverse visit count).
        crossover_rate : float
            Base probability of elite crossover.
        """
        self.categorical_dims = categorical_dims
        self.elite_size = elite_size
        self.curiosity_bonus = curiosity_bonus
        self.crossover_rate = crossover_rate

        # Track visit counts for curiosity
        self._visit_counts: Dict[Tuple[int, ...], int] = {}

        # Per-key score history (internal score y; higher is better).
        # Used for key-level Thompson Sampling on "good-rate" vs current gamma.
        self._key_scores: Dict[Tuple[int, ...], List[float]] = {}
        self._all_scores: List[float] = []

        # Elite pool: [(cat_key, score), ...] sorted by score descending
        self._elite_configs: List[Tuple[Tuple[int, ...], float]] = []

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
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)

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
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5

    def get_cat_key(self, x: np.ndarray) -> Tuple[int, ...]:
        """
        Extract tuple of categorical indices from vector x.

        Parameters
        ----------
        x : np.ndarray
            Point in normalized space.

        Returns
        -------
        Tuple[int, ...]
            Tuple of category indices.
        """
        return tuple(
            self.discretize(x[dim_idx], n_ch)
            for dim_idx, n_ch in self.categorical_dims
        )

    def apply_cat_key(
        self,
        cat_key: Tuple[int, ...],
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Apply categorical key to vector x.

        Parameters
        ----------
        cat_key : Tuple[int, ...]
            Tuple of category indices.
        x : np.ndarray
            Original point.

        Returns
        -------
        np.ndarray
            Modified point with categorical values set.
        """
        x = x.copy()
        for i, (dim_idx, n_ch) in enumerate(self.categorical_dims):
            x[dim_idx] = self.to_continuous(cat_key[i], n_ch)
        return x

    # -------------------------------------------------------------------------
    # Statistics tracking
    # -------------------------------------------------------------------------

    def record_observation(
        self,
        x: np.ndarray,
        score: float,
    ) -> None:
        """
        Record an observation for categorical tracking.

        Parameters
        ----------
        x : np.ndarray
            Evaluated point.
        score : float
            Internal score (higher is better).
        """
        if not self.has_categoricals:
            return

        cat_key = self.get_cat_key(x)

        # Update visit counts
        self._visit_counts[cat_key] = self._visit_counts.get(cat_key, 0) + 1

        # Track score history for key-level TS.
        try:
            self._all_scores.append(float(score))
            self._key_scores.setdefault(cat_key, []).append(float(score))
        except Exception:
            pass

        # Update elite pool (unique keys; keep best score per key).
        # Duplicates here can cause downstream key enumeration to waste slots and
        # fall back to sampling many unseen keys, increasing key thrashing.
        best_by_key: Dict[Tuple[int, ...], float] = {}
        for k, s in self._elite_configs:
            prev = best_by_key.get(k)
            if prev is None or float(s) > float(prev):
                best_by_key[k] = float(s)
        prev = best_by_key.get(cat_key)
        if prev is None or float(score) > float(prev):
            best_by_key[cat_key] = float(score)
        self._elite_configs = sorted(best_by_key.items(), key=lambda p: p[1], reverse=True)[: self.elite_size]

    def get_visit_count(self, cat_key: Tuple[int, ...]) -> int:
        """Get the number of times a categorical combination was visited."""
        return self._visit_counts.get(cat_key, 0)

    def choose_key_ts_goodrate(
        self,
        candidate_keys: List[Tuple[int, ...]],
        *,
        gamma: float,
        rng: np.random.Generator,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """
        Choose a categorical key via Thompson Sampling on good-rate.

        We treat each evaluated point as a Bernoulli outcome:
            success = 1{score >= gamma}

        For each key k, we maintain a history of observed internal scores and
        compute n_good/n_total relative to the *current* gamma (so gamma updates
        do not require rebuilding persistent counters).

        Returns
        -------
        (key, info)
            key: chosen categorical key.
            info: diagnostics for tracing.
        """
        if not candidate_keys:
            return tuple(), {"method": "ts_goodrate", "chosen": None, "candidates": []}

        # Empirical Bayes prior: anchor unvisited keys to the observed global
        # good-rate under the *current* gamma to avoid pathological optimism
        # (e.g., Beta(1,1) mean=0.5 when true success rates are ~0.1).
        g = float(gamma)
        scores_all = np.asarray(self._all_scores, dtype=float)
        scores_all = scores_all[np.isfinite(scores_all)]
        n_all = int(scores_all.size)
        p0 = float(np.mean(scores_all >= g)) if n_all > 0 else 0.2
        p0 = float(np.clip(p0, 1e-3, 1.0 - 1e-3))

        strength = float(np.sqrt(float(n_all))) if n_all > 0 else 4.0
        strength = float(np.clip(strength, 4.0, 20.0))
        alpha0 = 1.0 + p0 * strength
        beta0 = 1.0 + (1.0 - p0) * strength

        cand_info: List[Dict[str, Any]] = []
        best_key = candidate_keys[0]
        best_sample = -1.0

        for k in candidate_keys:
            scores = self._key_scores.get(k, [])
            n_total = int(len(scores))
            if n_total > 0:
                try:
                    n_good = int(np.sum(np.asarray(scores, dtype=float) >= g))
                except Exception:
                    n_good = int(sum(1 for s in scores if float(s) >= g))
            else:
                n_good = 0

            a = float(alpha0 + float(n_good))
            b = float(beta0 + float(max(n_total - n_good, 0)))
            p = float(rng.beta(a, b))

            cand_info.append(
                {
                    "key": list(k),
                    "n_total": int(n_total),
                    "n_good": int(n_good),
                    "alpha": float(a),
                    "beta": float(b),
                    "p_sample": float(p),
                }
            )
            if p > best_sample or best_key is None:
                best_sample = p
                best_key = k

        info: Dict[str, Any] = {
            "method": "ts_goodrate",
            "prior": {
                "alpha0": float(alpha0),
                "beta0": float(beta0),
                "p0": float(p0),
                "strength": float(strength),
                "n_all": int(n_all),
            },
            "gamma": float(gamma),
            "chosen": list(best_key),
            "p_chosen": float(best_sample),
            "candidates": cand_info,
        }
        return best_key, info

    def elite_keys(self) -> List[Tuple[int, ...]]:
        """Return current elite categorical keys (best first)."""
        return [k for k, _ in self._elite_configs]

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def elite_crossover(
        self,
        rng: np.random.Generator,
        stagnating: bool = False,
    ) -> Optional[Tuple[int, ...]]:
        """
        Create new categorical config by crossing over two elite parents.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        stagnating : bool
            If True, use higher mutation rate.

        Returns
        -------
        Optional[Tuple[int, ...]]
            New categorical key, or None if not enough elites.
        """
        if len(self._elite_configs) < 2:
            return None

        # Select two parents (bias towards better ones)
        n = len(self._elite_configs)
        weights = np.array([1.0 / (i + 1) for i in range(n)])
        weights = weights / weights.sum()

        idx1, idx2 = rng.choice(n, size=2, replace=False, p=weights)
        parent1 = self._elite_configs[idx1][0]
        parent2 = self._elite_configs[idx2][0]

        # Adaptive mutation rate
        mutation_rate = 0.25 if stagnating else 0.1

        # Uniform crossover with mutation
        child = []
        for i, (dim_idx, n_ch) in enumerate(self.categorical_dims):
            if rng.random() < 0.5:
                val = parent1[i]
            else:
                val = parent2[i]

            if rng.random() < mutation_rate:
                val = rng.integers(0, n_ch)

            child.append(val)

        return tuple(child)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        x: np.ndarray,
        cube: "Cube",
        rng: np.random.Generator,
        stagnating: bool = False,
    ) -> np.ndarray:
        """
        Apply categorical sampling to a candidate point.

        Uses a combination of:
        1. Elite crossover (with probability crossover_rate)
        2. Thompson Sampling per dimension
        3. Curiosity-driven selection among candidates

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

        # === ELITE CROSSOVER ===
        crossover_prob = self.crossover_rate
        if stagnating:
            crossover_prob *= 2

        if rng.random() < crossover_prob and len(self._elite_configs) >= 2:
            child_key = self.elite_crossover(rng, stagnating)
            if child_key:
                return self.apply_cat_key(child_key, x)

        # === CURIOSITY-DRIVEN THOMPSON SAMPLING ===
        exploration_boost = 2.0 if stagnating else 1.0

        # Generate multiple candidate categorical configs via Thompson Sampling
        n_candidates = 5
        candidates = []

        for _ in range(n_candidates):
            cat_vals = []
            for dim_idx, n_choices in self.categorical_dims:
                stats = cube.cat_stats.get(dim_idx, {})

                # Thompson Sampling: sample from Beta distribution for each category
                samples = []
                K = n_choices * exploration_boost
                for v in range(n_choices):
                    n_g, n_t = stats.get(v, (0, 0))
                    alpha = n_g + 1
                    beta_param = (n_t - n_g) + K
                    sample = rng.beta(alpha, beta_param)
                    samples.append(sample)

                # Pick category with highest sampled value
                chosen = int(np.argmax(samples))
                cat_vals.append(chosen)

            candidates.append(tuple(cat_vals))

        # Score candidates by curiosity (inverse visit count)
        scores = []
        for cat_key in candidates:
            visit_count = self._visit_counts.get(cat_key, 0)
            curiosity = self.curiosity_bonus / (1 + visit_count)
            scores.append(curiosity)

        # Select based on curiosity score (softmax)
        scores = np.array(scores)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        probs = np.exp(scores * 3)
        probs = probs / probs.sum()

        chosen_idx = rng.choice(len(candidates), p=probs)
        chosen_key = candidates[chosen_idx]

        return self.apply_cat_key(chosen_key, x)

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
