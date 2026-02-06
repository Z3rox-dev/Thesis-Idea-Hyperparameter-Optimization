#!/usr/bin/env python3
"""
ALBA_V1_thompson_all - Thompson Sampling for ALL dimensions (continuous + categorical)

Experimental version that applies Thompson Sampling approach to continuous dimensions
by discretizing them into bins, similar to how categorical dimensions are handled.

Key idea: Instead of LGS/surrogate model for continuous, we discretize each continuous
dimension into N bins and apply Thompson Sampling to select promising bins.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


@dataclass(eq=False)
class Cube:
    """A hyperrectangle region of the search space."""
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    depth: int = 0
    # Dimension stats: {dim_idx: {bin_idx: (n_good, n_total)}}
    dim_stats: dict = field(default_factory=dict)

    def _widths(self) -> np.ndarray:
        return np.array([abs(hi - lo) for lo, hi in self.bounds], dtype=float)

    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds], dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True

    def good_ratio(self) -> float:
        """Beta prior estimation: (good + 1) / (trials + 2)"""
        return (self.n_good + 1) / (self.n_trials + 2)

    def get_split_axis(self) -> int:
        """Choose split axis based on widest dimension."""
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int, rng: np.random.Generator = None) -> List["Cube"]:
        """Split cube into two children along the chosen axis."""
        axis = self.get_split_axis()
        lo, hi = self.bounds[axis]
        good_pts = [p[axis] for p, s in self._tested_pairs if s >= gamma]

        if len(good_pts) >= 3:
            cut = float(np.median(good_pts))
            margin = 0.15 * (hi - lo)
            cut = np.clip(cut, lo + margin, hi - margin)
        else:
            cut = (lo + hi) / 2

        bounds_lo = list(self.bounds)
        bounds_hi = list(self.bounds)
        bounds_lo[axis] = (lo, cut)
        bounds_hi[axis] = (cut, hi)

        child_lo = Cube(bounds=bounds_lo, parent=self)
        child_hi = Cube(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1

        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1
            if sc > child.best_score:
                child.best_score = sc
                child.best_x = pt.copy()

        return [child_lo, child_hi]


class ALBA:
    """
    ALBA with Thompson Sampling for ALL dimensions.
    
    Key innovation: Discretize continuous dimensions into bins and use
    Thompson Sampling (Beta distribution) to select promising regions.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        maximize: bool = False,
        seed: int = 42,
        gamma_quantile: float = 0.20,
        gamma_quantile_start: float = 0.15,
        local_search_ratio: float = 0.30,
        n_candidates: int = 25,
        split_trials_min: int = 15,
        split_depth_max: int = 16,
        split_trials_factor: float = 3.0,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        global_random_prob: float = 0.05,
        stagnation_threshold: int = 50,
        categorical_dims: List[Tuple[int, int]] = None,
        n_bins_continuous: int = 10,  # Number of bins for continuous dims
    ) -> None:
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = gamma_quantile_start
        self.local_search_ratio = local_search_ratio
        self.n_candidates = n_candidates
        self.total_budget = total_budget
        self.categorical_dims = categorical_dims or []
        self.n_bins_continuous = n_bins_continuous
        
        # Build dimension info: for each dim, number of bins/choices
        self.dim_info = []  # [(is_categorical, n_bins/n_choices), ...]
        cat_dim_set = {d for d, _ in self.categorical_dims}
        for i in range(self.dim):
            is_cat = i in cat_dim_set
            if is_cat:
                n_ch = next(n for d, n in self.categorical_dims if d == i)
                self.dim_info.append((True, n_ch))
            else:
                self.dim_info.append((False, n_bins_continuous))
        
        # Curiosity and elite tracking
        self._visit_counts = {}  # Track visit counts for combinations
        self._elite_configs = []
        self._elite_size = 10
        self._curiosity_bonus = 0.3
        self._crossover_rate = 0.15

        self.root = Cube(bounds=list(bounds))
        self.leaves: List[Cube] = [self.root]

        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.best_y_internal = -np.inf
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0
        
        self.stagnation = 0
        self.last_improvement_iter = 0

        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        self.global_widths = np.array([hi - lo for lo, hi in bounds])
        self.last_cube: Optional[Cube] = None

        self._split_trials_min = split_trials_min
        self._split_depth_max = split_depth_max
        self._split_trials_factor = split_trials_factor
        self._split_trials_offset = split_trials_offset
        self._novelty_weight = novelty_weight
        self._global_random_prob = global_random_prob
        self._stagnation_threshold = stagnation_threshold

    def _to_internal(self, y_raw: float) -> float:
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        return y_internal if self.maximize else -y_internal

    def _discretize(self, x_val: float, dim_idx: int, cube: Cube) -> int:
        """Convert continuous value to bin index within cube bounds."""
        is_cat, n_bins = self.dim_info[dim_idx]
        lo, hi = cube.bounds[dim_idx]
        
        if is_cat:
            # Categorical: just discretize to n_choices
            return min(int(round(x_val * (n_bins - 1))), n_bins - 1)
        else:
            # Continuous: bin within cube bounds
            if hi - lo < 1e-9:
                return 0
            normalized = (x_val - lo) / (hi - lo)
            bin_idx = int(normalized * n_bins)
            return min(max(bin_idx, 0), n_bins - 1)
    
    def _bin_to_value(self, bin_idx: int, dim_idx: int, cube: Cube) -> float:
        """Convert bin index back to continuous value."""
        is_cat, n_bins = self.dim_info[dim_idx]
        lo, hi = cube.bounds[dim_idx]
        
        if is_cat:
            return bin_idx / (n_bins - 1) if n_bins > 1 else 0.5
        else:
            # Return center of bin with small noise
            bin_lo = lo + (hi - lo) * bin_idx / n_bins
            bin_hi = lo + (hi - lo) * (bin_idx + 1) / n_bins
            # Random within bin for diversity
            return self.rng.uniform(bin_lo, bin_hi)
    
    def _get_config_key(self, x: np.ndarray, cube: Cube) -> tuple:
        """Get discretized key for configuration."""
        return tuple(self._discretize(x[i], i, cube) for i in range(self.dim))

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)

        # Global random for diversity
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self.last_cube = self._find_containing_leaf(x)
            return x

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            self.last_cube = self._select_leaf()
            return self._sample_thompson(self.last_cube)

        # Local search phase
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress
        
        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self.last_cube = self._find_containing_leaf(x)
        else:
            self._update_gamma()
            self._recount_good()
            self.last_cube = self._select_leaf()
            x = self._sample_thompson(self.last_cube)
        
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        y = self._to_internal(y_raw)
        
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
        else:
            self.stagnation += 1

        self.X_all.append(x.copy())
        self.y_all.append(y)
        
        if self.last_cube is not None:
            cube = self.last_cube
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            is_good = y >= self.gamma
            if is_good:
                cube.n_good += 1
            
            if y > cube.best_score:
                cube.best_score = y
                cube.best_x = x.copy()
            
            # Update dimension stats for ALL dimensions
            for dim_idx in range(self.dim):
                _, n_bins = self.dim_info[dim_idx]
                bin_idx = self._discretize(x[dim_idx], dim_idx, cube)
                if dim_idx not in cube.dim_stats:
                    cube.dim_stats[dim_idx] = {}
                n_g, n_t = cube.dim_stats[dim_idx].get(bin_idx, (0, 0))
                cube.dim_stats[dim_idx][bin_idx] = (n_g + (1 if is_good else 0), n_t + 1)
            
            # Track visit counts
            config_key = self._get_config_key(x, cube)
            self._visit_counts[config_key] = self._visit_counts.get(config_key, 0) + 1
            
            # Update elite pool
            self._elite_configs.append((config_key, y, cube))
            self._elite_configs.sort(key=lambda p: p[1], reverse=True)
            self._elite_configs = self._elite_configs[:self._elite_size]

            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim, self.rng)
                for child in children:
                    self._recompute_dim_stats(child)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)

            self.last_cube = None

    def _sample_thompson(self, cube: Cube) -> np.ndarray:
        """
        Thompson Sampling for ALL dimensions.
        For each dimension, sample from Beta distribution for each bin,
        then pick the bin with highest sampled value.
        """
        exploration_boost = 2.0 if self.stagnation > self._stagnation_threshold else 1.0
        
        # === ELITE CROSSOVER ===
        crossover_prob = self._crossover_rate
        if self.stagnation > self._stagnation_threshold:
            crossover_prob *= 2
        
        if self.rng.random() < crossover_prob and len(self._elite_configs) >= 2:
            x = self._elite_crossover(cube)
            if x is not None:
                return x
        
        # === THOMPSON SAMPLING FOR EACH DIMENSION ===
        x = np.zeros(self.dim)
        
        for dim_idx in range(self.dim):
            _, n_bins = self.dim_info[dim_idx]
            stats = cube.dim_stats.get(dim_idx, {})
            
            # Thompson Sampling: sample from Beta for each bin
            samples = []
            K = n_bins * exploration_boost  # Prior strength
            
            for bin_idx in range(n_bins):
                n_g, n_t = stats.get(bin_idx, (0, 0))
                # Beta(alpha, beta) where:
                # alpha = successes + 1
                # beta = failures + K (encourages exploration of unseen bins)
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                sample = self.rng.beta(alpha, beta_param)
                samples.append(sample)
            
            # Pick bin with highest sampled value
            chosen_bin = int(np.argmax(samples))
            x[dim_idx] = self._bin_to_value(chosen_bin, dim_idx, cube)
        
        # Apply curiosity bonus: occasionally explore rare combinations
        if self.rng.random() < self._curiosity_bonus:
            # With some probability, mutate towards less-visited bins
            for dim_idx in range(self.dim):
                if self.rng.random() < 0.3:  # Per-dimension mutation
                    _, n_bins = self.dim_info[dim_idx]
                    # Pick a random bin
                    new_bin = self.rng.integers(0, n_bins)
                    x[dim_idx] = self._bin_to_value(new_bin, dim_idx, cube)
        
        return self._clip_to_cube(x, cube)
    
    def _elite_crossover(self, cube: Cube) -> Optional[np.ndarray]:
        """Crossover two elite configurations."""
        if len(self._elite_configs) < 2:
            return None
        
        n = len(self._elite_configs)
        weights = np.array([1.0 / (i + 1) for i in range(n)])
        weights = weights / weights.sum()
        
        idx1, idx2 = self.rng.choice(n, size=2, replace=False, p=weights)
        key1, _, cube1 = self._elite_configs[idx1]
        key2, _, cube2 = self._elite_configs[idx2]
        
        # Crossover
        x = np.zeros(self.dim)
        for dim_idx in range(self.dim):
            if self.rng.random() < 0.5:
                x[dim_idx] = self._bin_to_value(key1[dim_idx], dim_idx, cube)
            else:
                x[dim_idx] = self._bin_to_value(key2[dim_idx], dim_idx, cube)
            
            # Mutation
            if self.rng.random() < 0.1:
                _, n_bins = self.dim_info[dim_idx]
                x[dim_idx] = self._bin_to_value(self.rng.integers(0, n_bins), dim_idx, cube)
        
        return self._clip_to_cube(x, cube)

    def _update_gamma(self) -> None:
        if len(self.y_all) < 10:
            self.gamma = 0.0
            return
        progress = min(1.0, self.iteration / max(1, self.exploration_budget * 0.5))
        current_quantile = self.gamma_quantile_start - progress * (
            self.gamma_quantile_start - self.gamma_quantile
        )
        self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))

    def _recount_good(self) -> None:
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
            self._recompute_dim_stats(leaf)
    
    def _recompute_dim_stats(self, cube: Cube) -> None:
        """Recompute dimension stats for a cube."""
        cube.dim_stats = {}
        for pt, sc in cube._tested_pairs:
            is_good = sc >= self.gamma
            for dim_idx in range(self.dim):
                _, n_bins = self.dim_info[dim_idx]
                bin_idx = self._discretize(pt[dim_idx], dim_idx, cube)
                if dim_idx not in cube.dim_stats:
                    cube.dim_stats[dim_idx] = {}
                n_g, n_t = cube.dim_stats[dim_idx].get(bin_idx, (0, 0))
                cube.dim_stats[dim_idx][bin_idx] = (n_g + (1 if is_good else 0), n_t + 1)

    def _select_leaf(self) -> Cube:
        if not self.leaves:
            return self.root

        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            if self.stagnation > self._stagnation_threshold:
                exploration *= 2.0
            scores.append(ratio + exploration)

        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        temperature = 1.5 if self.stagnation > self._stagnation_threshold else 3.0
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.leaves), p=probs)
        return self.leaves[int(idx)]

    def _clip_to_cube(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        return np.array([
            np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1])
            for i in range(self.dim)
        ])

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(x[i], self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dim)
        ])

    def _find_containing_leaf(self, x: np.ndarray) -> Cube:
        for leaf in self.leaves:
            if leaf.contains(x):
                return leaf
        min_dist = float('inf')
        closest = self.leaves[0] if self.leaves else self.root
        for leaf in self.leaves:
            dist = np.linalg.norm(x - leaf.center())
            if dist < min_dist:
                min_dist = dist
                closest = leaf
        return closest

    def _local_search_sample(self, progress: float) -> np.ndarray:
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        radius = 0.15 * (1 - progress) + 0.03
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = self.best_x + noise
        return self._clip_to_bounds(x)

    def _should_split(self, cube: Cube) -> bool:
        if cube.n_trials < self._split_trials_min:
            return False
        if cube.depth >= self._split_depth_max:
            return False
        return cube.n_trials >= self._split_trials_factor * self.dim + self._split_trials_offset

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        return self.best_x, self._to_raw(self.best_y_internal)
