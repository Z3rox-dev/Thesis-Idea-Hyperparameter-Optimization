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

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
        # FIX Finding 28: Handle NaN/Inf and out-of-range values
        if not np.isfinite(x_val):
            return 0  # Default to first category
        # Clamp to [0, 1] then discretize
        x_clamped = max(0.0, min(1.0, x_val))
        idx = int(round(x_clamped * (n_choices - 1)))
        return max(0, min(idx, n_choices - 1))

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

        # Update elite pool
        self._elite_configs.append((cat_key, score))
        self._elite_configs.sort(key=lambda p: p[1], reverse=True)
        self._elite_configs = self._elite_configs[: self.elite_size]

    def get_visit_count(self, cat_key: Tuple[int, ...]) -> int:
        """Get the number of times a categorical combination was visited."""
        return self._visit_counts.get(cat_key, 0)

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
            # FIX Finding 28: Skip invalid scores
            if not np.isfinite(sc):
                continue
            is_good = sc >= gamma
            for dim_idx, n_choices in self.categorical_dims:
                # discretize already handles NaN in pt[dim_idx]
                val_idx = self.discretize(pt[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (
                    n_g + (1 if is_good else 0),
                    n_t + 1,
                )
