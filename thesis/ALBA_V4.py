#!/usr/bin/env python3
"""
ALBA V4 - Exploitation-Boosted Fast Start

Key insight from V3 analysis:
- LHS warm-up helps initial exploration
- BUT it doesn't exploit good regions found early
- Result: V3 leads early, but Original catches up

V4 Strategy:
1. LHS warm-up (like V3) for initial coverage
2. IMMEDIATELY after warm-up: exploit top-k warm-up points
3. Early tree growth: split cubes containing good warm-up points
4. Faster transition to model-based sampling

This combines the best of both:
- Good initial coverage (LHS)
- Fast exploitation of promising regions (immediate refinement)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import heapq
import warnings
warnings.filterwarnings('ignore')


def latin_hypercube_sample(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Latin Hypercube samples in [0,1]^dim."""
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        bins = np.linspace(0, 1, n_samples + 1)
        for i in range(n_samples):
            samples[i, d] = rng.uniform(bins[i], bins[i+1])
        rng.shuffle(samples[:, d])
    return samples


@dataclass(eq=False)
class Cube:
    """A hyperrectangle region of the search space with local surrogate model."""
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    cat_stats: dict = field(default_factory=dict)

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
        return (self.n_good + 1) / (self.n_trials + 2)

    def fit_lgs_model(self, gamma: float, dim: int, rng: np.random.Generator = None) -> None:
        pairs = list(self._tested_pairs)

        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, "_tested_pairs", [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
                if rng is not None:
                    extra = list(extra)
                    rng.shuffle(extra)
                pairs = pairs + extra[:needed]

        if len(pairs) < dim + 2:
            self.lgs_model = None
            return

        all_pts = np.array([p for p, s in pairs])
        all_scores = np.array([s for p, s in pairs])

        k = max(3, len(pairs) // 5)
        top_k_idx = np.argsort(all_scores)[-k:]
        top_k_pts = all_pts[top_k_idx]

        gradient_dir = None
        grad = None
        inv_cov = None
        y_mean = 0.0
        noise_var = 1.0
        
        widths = np.maximum(self._widths(), 1e-9)
        center = self.center()

        if len(pairs) >= dim + 3:
            X_norm = (all_pts - center) / widths
            y_mean = all_scores.mean()
            y_centered = all_scores - y_mean
            
            try:
                dists_sq = np.sum(X_norm**2, axis=1)
                sigma_sq = np.mean(dists_sq) + 1e-6
                weights = np.exp(-dists_sq / (2 * sigma_sq))
                W = np.diag(weights)

                lambda_reg = 0.1
                XtWX = X_norm.T @ W @ X_norm + lambda_reg * np.eye(dim)
                inv_cov = np.linalg.inv(XtWX)
                grad = inv_cov @ (X_norm.T @ W @ y_centered)
                
                y_pred = X_norm @ grad
                residuals = y_centered - y_pred
                noise_var = np.clip(
                    np.average(residuals**2, weights=weights) + 1e-6,
                    1e-4, 10.0
                )
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-9:
                    gradient_dir = grad / grad_norm
            except Exception:
                pass

        self.lgs_model = {
            "all_pts": all_pts,
            "top_k_pts": top_k_pts,
            "gradient_dir": gradient_dir,
            "grad": grad,
            "inv_cov": inv_cov,
            "y_mean": y_mean,
            "noise_var": noise_var,
            "widths": widths,
            "center": center,
        }

    def predict_bayesian(self, candidates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self.lgs_model is None or self.lgs_model.get("inv_cov") is None:
            return np.zeros(len(candidates)), np.ones(len(candidates))

        model = self.lgs_model
        widths = model["widths"]
        center = model["center"]
        grad = model["grad"]
        inv_cov = model["inv_cov"]
        noise_var = model["noise_var"]
        y_mean = model["y_mean"]

        C_norm = (np.array(candidates) - center) / widths
        mu = y_mean + C_norm @ grad
        
        model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0, 10.0)
        total_var = noise_var * (1.0 + model_var)
        sigma = np.sqrt(total_var)
        
        return mu, sigma

    def get_split_axis(self) -> int:
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int, rng: np.random.Generator = None) -> List["Cube"]:
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

        for ch in (child_lo, child_hi):
            ch.fit_lgs_model(gamma, dim, rng)

        return [child_lo, child_hi]


class ALBA:
    """
    ALBA V4 - Exploitation-Boosted Fast Start.
    
    Phases:
    1. LHS Warm-up (exploration)
    2. Exploitation burst (refine top warm-up points)
    3. Standard ALBA with faster model fitting
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
        # V4-specific parameters
        warm_up_fraction: float = 0.08,  # Slightly less LHS than V3
        exploitation_fraction: float = 0.07,  # Exploitation burst after warm-up
        n_top_exploit: int = 3,  # Number of top points to exploit
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
        
        # V4 parameters
        self._warm_up_budget = max(5, int(total_budget * warm_up_fraction))
        self._exploitation_budget = max(3, int(total_budget * exploitation_fraction))
        self._n_top_exploit = n_top_exploit
        
        # LHS samples for warm-up
        self._lhs_samples = latin_hypercube_sample(self._warm_up_budget, self.dim, self.rng)
        self._lhs_index = 0
        for i in range(self.dim):
            lo, hi = self.bounds[i]
            self._lhs_samples[:, i] = lo + self._lhs_samples[:, i] * (hi - lo)
        
        # Track warm-up results for exploitation phase
        self._warmup_results: List[Tuple[np.ndarray, float]] = []
        self._exploitation_index = 0

    def _to_internal(self, y_raw: float) -> float:
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        return y_internal if self.maximize else -y_internal
    
    def _phase(self) -> str:
        """Determine current phase."""
        if self.iteration < self._warm_up_budget:
            return 'warmup'
        elif self.iteration < self._warm_up_budget + self._exploitation_budget:
            return 'exploitation'
        else:
            return 'standard'

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)
        phase = self._phase()

        # Phase 1: LHS warm-up
        if phase == 'warmup':
            x = self._lhs_samples[self._lhs_index].copy()
            self._lhs_index += 1
            x = self._apply_categorical_sampling_warmup(x)
            self.last_cube = self._find_containing_leaf(x)
            return x
        
        # Phase 2: Exploitation burst
        if phase == 'exploitation':
            x = self._exploitation_sample()
            self.last_cube = self._find_containing_leaf(x)
            return x

        # Phase 3: Standard ALBA
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self.last_cube = self._find_containing_leaf(x)
            return x

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            return self._sample_in_cube(self.last_cube)

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
            x = self._sample_in_cube(self.last_cube)
        
        return x
    
    def _exploitation_sample(self) -> np.ndarray:
        """
        Exploitation phase: sample near top warm-up points.
        Uses decreasing radius for progressively finer search.
        """
        if not self._warmup_results:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        
        # Sort by score, get top k
        sorted_results = sorted(self._warmup_results, key=lambda x: x[1], reverse=True)
        top_k = sorted_results[:self._n_top_exploit]
        
        # Progress within exploitation phase
        exploit_iter = self.iteration - self._warm_up_budget
        progress = exploit_iter / max(1, self._exploitation_budget - 1)
        
        # Radius decreases as we progress
        radius = 0.15 * (1 - progress * 0.7) + 0.05
        
        # Select which top point to refine (round robin with some randomness)
        if self.rng.random() < 0.7:
            # Prefer best point
            base_x, base_y = top_k[0]
        else:
            # Sometimes explore other top points
            idx = self.rng.integers(len(top_k))
            base_x, base_y = top_k[idx]
        
        # Sample near the base point
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = base_x + noise
        x = self._clip_to_bounds(x)
        
        # Apply categorical discretization
        x = self._apply_categorical_sampling_warmup(x)
        
        return x
    
    def _apply_categorical_sampling_warmup(self, x: np.ndarray) -> np.ndarray:
        """Discretize categorical dimensions."""
        if not self.categorical_dims:
            return x
        
        x = x.copy()
        for dim_idx, n_choices in self.categorical_dims:
            val_idx = min(int(round(x[dim_idx] * (n_choices - 1))), n_choices - 1)
            x[dim_idx] = val_idx / (n_choices - 1) if n_choices > 1 else 0.5
        
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
        
        # Track warm-up results
        if self._phase() == 'warmup' or self.iteration < self._warm_up_budget:
            self._warmup_results.append((x.copy(), y))

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
            
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

            # Skip model fitting during warm-up
            if self._phase() != 'warmup':
                cube.fit_lgs_model(self.gamma, self.dim, self.rng)

            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim, self.rng)
                for child in children:
                    self._recompute_cat_stats(child)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)

            self.last_cube = None

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
            self._recompute_cat_stats(leaf)

    def _select_leaf(self) -> Cube:
        if not self.leaves:
            return self.root

        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            
            if self.stagnation > self._stagnation_threshold:
                exploration *= 2.0
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= self.dim + 2:
                    model_bonus = 0.1
            scores.append(ratio + exploration + model_bonus)

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

    def _generate_candidates(self, cube: Cube, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube._widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = self.rng.random()

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = self.rng.integers(len(model["top_k_pts"]))
                x = model["top_k_pts"][idx] + self.rng.normal(0, 0.15, self.dim) * widths
            elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
                top_center = model["top_k_pts"].mean(axis=0)
                step = self.rng.uniform(0.05, 0.3)
                x = top_center + step * model["gradient_dir"] * widths
                x = x + self.rng.normal(0, 0.05, self.dim) * widths
            elif strategy < 0.55:
                x = center + self.rng.normal(0, 0.2, self.dim) * widths
            else:
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

            x = self._clip_to_cube(x, cube)
            candidates.append(x)

        return candidates

    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        candidates = self._generate_candidates(cube, self.n_candidates)
        mu, sigma = cube.predict_bayesian(candidates)
        
        beta = self._novelty_weight * 2.0
        score = mu + beta * sigma
        
        if score.std() > 1e-9:
            score_z = (score - score.mean()) / score.std()
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * 3.0)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _discretize_cat(self, x_val: float, n_choices: int) -> int:
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)
    
    def _cat_value_to_continuous(self, val_idx: int, n_choices: int) -> float:
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5
    
    def _recompute_cat_stats(self, cube: Cube) -> None:
        cube.cat_stats = {}
        for dim_idx, n_choices in self.categorical_dims:
            cube.cat_stats[dim_idx] = {}
        
        for pt, sc in cube._tested_pairs:
            is_good = sc >= self.gamma
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(pt[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)
    
    def _apply_categorical_sampling(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        if not self.categorical_dims:
            return x
        
        x = x.copy()
        exploration_boost = 2.0 if self.stagnation > self._stagnation_threshold else 1.0
        
        for dim_idx, n_choices in self.categorical_dims:
            stats = cube.cat_stats.get(dim_idx, {})
            
            probs = []
            for v in range(n_choices):
                n_g, n_t = stats.get(v, (0, 0))
                K = n_choices * exploration_boost
                prob = (n_g + 1) / (n_t + K)
                probs.append(prob)
            
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            chosen = self.rng.choice(n_choices, p=probs)
            x[dim_idx] = self._cat_value_to_continuous(chosen, n_choices)
        
        return x

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        if self.iteration < 15:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        else:
            x = self._sample_with_lgs(cube)
        
        x = self._apply_categorical_sampling(x, cube)
        x = self._clip_to_cube(x, cube)
        return x

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

    def _update_all_models(self) -> None:
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget
            
            self._warm_up_budget = max(5, int(budget * 0.08))
            self._exploitation_budget = max(3, int(budget * 0.07))
            
            self._lhs_samples = latin_hypercube_sample(self._warm_up_budget, self.dim, self.rng)
            self._lhs_index = 0
            for i in range(self.dim):
                lo, hi = self.bounds[i]
                self._lhs_samples[:, i] = lo + self._lhs_samples[:, i] * (hi - lo)

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        return self.best_x, self._to_raw(self.best_y_internal)
