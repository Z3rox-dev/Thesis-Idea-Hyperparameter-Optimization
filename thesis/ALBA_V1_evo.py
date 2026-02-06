#!/usr/bin/env python3
"""
ALBA (Adaptive Local Bayesian Algorithm) - Version 1.1 (Evolutionary)

A novel hyperparameter optimization algorithm based on:
- Adaptive cube partitioning of the search space
- Local Gradient Surrogate (LGS) models within each cube
- UCB-style acquisition with exploration bonuses
- Two-phase optimization: exploration + local search refinement

Evolutionary Features (v1.1):
- Active/Suspended leaf pools (seed bank for soft pruning)
- Thompson sampling with Beta posterior for leaf selection
- Tournament selection (size=2) among cubes
- Resurrection trials: suspended cubes can be revived
- No hard pruning: cubes are suspended, not deleted

Key Features:
- Beta prior for good_ratio estimation (avoids single-point dominance)
- Stagnation-aware exploration (increases randomness when stuck)
- Gradient-guided candidate generation
- Bayesian uncertainty estimation for acquisition
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


@dataclass(eq=False)
class Cube:
    """
    A hyperrectangle region of the search space with local surrogate model.
    """
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    # Categorical stats: {dim_idx: {value_idx: (n_good, n_total)}}
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
        """Beta prior estimation: (good + 1) / (trials + 2)"""
        return (self.n_good + 1) / (self.n_trials + 2)

    def fit_lgs_model(self, gamma: float, dim: int, rng: np.random.Generator = None) -> None:
        """Fit Local Gradient Surrogate model using weighted linear regression."""
        pairs = list(self._tested_pairs)

        # Parent backfill with shuffle for diversity
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
        """Predict mean and uncertainty for candidate points."""
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
        """Choose split axis based on gradient direction or widest dimension."""
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
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
            # Update child's best score
            if sc > child.best_score:
                child.best_score = sc
                child.best_x = pt.copy()

        for ch in (child_lo, child_hi):
            ch.fit_lgs_model(gamma, dim, rng)

        return [child_lo, child_hi]


class ALBA:
    """
    ALBA (Adaptive Local Bayesian Allocator) for Hyperparameter Optimization.
    
    Parameters
    ----------
    bounds : List[Tuple[float, float]]
        Search space bounds for each dimension.
    maximize : bool, default=False
        Whether to maximize (True) or minimize (False) the objective.
    seed : int, default=42
        Random seed for reproducibility.
    gamma_quantile : float, default=0.20
        Final quantile threshold for "good" points.
    gamma_quantile_start : float, default=0.15
        Initial quantile threshold (more selective at start).
    local_search_ratio : float, default=0.30
        Fraction of budget for local search phase.
    n_candidates : int, default=25
        Number of candidates to generate per iteration.
    split_trials_min : int, default=15
        Minimum trials before cube can split.
    split_depth_max : int, default=8
        Maximum tree depth.
    split_trials_factor : float, default=3.0
        Factor for split threshold: factor * dim + offset.
    split_trials_offset : int, default=6
        Offset for split threshold.
    novelty_weight : float, default=0.4
        Weight for exploration in UCB acquisition.
    total_budget : int, default=200
        Total optimization budget.
    global_random_prob : float, default=0.05
        Probability of pure random sampling for diversity.
    stagnation_threshold : int, default=50
        Iterations without improvement before increasing exploration.
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
        categorical_dims: List[Tuple[int, int]] = None,  # [(dim_idx, n_choices), ...]
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
        self.categorical_dims = categorical_dims or []  # List of (dim_idx, n_choices)

        self.root = Cube(bounds=list(bounds))
        # Evolutionary leaf management: active vs suspended (seed bank)
        self.active_leaves: List[Cube] = [self.root]
        self.suspended_leaves: List[Cube] = []  # Dormant cubes (seed bank)
        # Legacy alias for compatibility
        self.leaves: List[Cube] = self.active_leaves
        
        # Evolutionary parameters (hardcoded)
        self._resurrection_interval = 10  # Try to resurrect every N iterations
        # Aggressive max_active to ensure suspension/resurrection mechanism is exercised
        # Formula: sqrt(budget) capped at 16, minimum 4
        self._max_active_leaves = min(16, max(4, int(np.sqrt(total_budget))))
        
        # Debug and stability parameters
        self._debug = False  # Set to True for verbose logging
        self._warmup_iters = 2 * len(bounds)  # Bypass Thompson early
        
        # Success tracking for debug
        self._recent_successes = deque(maxlen=20)  # Track last 20 success booleans
        self._resurrection_attempts = 0
        self._resurrection_successes = 0
        self._splits_triggered = 0
        self._last_selected_pool = "active"  # Track which pool was selected from
        self._last_split_axis = -1
        
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []  # Always stored as "higher is better"
        self.best_y_internal = -np.inf  # Internal score (always maximize)
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0
        
        self.stagnation = 0
        self.last_improvement_iter = 0
        
        # Track if last evaluated cube was suspended (for resurrection)
        self._last_cube_was_suspended = False

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
        """Convert raw objective value to internal score (higher is better)."""
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        """Convert internal score back to raw objective value."""
        return y_internal if self.maximize else -y_internal

    def _all_leaves(self) -> List[Cube]:
        """Return all leaves (active + suspended)."""
        return self.active_leaves + self.suspended_leaves

    def _check_invariants(self) -> None:
        """Check that leaf pools are disjoint and contain no duplicates."""
        active_ids = set(map(id, self.active_leaves))
        suspended_ids = set(map(id, self.suspended_leaves))
        
        # No overlap between pools
        overlap = active_ids.intersection(suspended_ids)
        assert len(overlap) == 0, f"Overlap between active/suspended: {len(overlap)} cubes"
        
        # No duplicates within pools
        all_leaves = self._all_leaves()
        assert len(all_leaves) == len(set(map(id, all_leaves))), "Duplicate cubes detected"

    def _debug_log(self, cube: Cube, split_triggered: bool = False) -> None:
        """Print debug info every 20 iterations."""
        if not self._debug:
            return
        if self.iteration % 20 != 0:
            return
        
        best_raw = self._to_raw(self.best_y_internal)
        success_rate = sum(self._recent_successes) / max(1, len(self._recent_successes))
        
        print(f"\n[DEBUG iter={self.iteration}]")
        print(f"  best_y_raw={best_raw:.6f}, gamma={self.gamma:.4f}")
        print(f"  active={len(self.active_leaves)}, suspended={len(self.suspended_leaves)}")
        print(f"  selected: pool={self._last_selected_pool}, depth={cube.depth}, "
              f"n_trials={cube.n_trials}, n_good={cube.n_good}, ratio={cube.good_ratio():.3f}")
        print(f"  split_triggered={split_triggered}, split_axis={self._last_split_axis}")
        print(f"  last20_success_rate={success_rate:.2f}")
        print(f"  resurrection: attempts={self._resurrection_attempts}, "
              f"successes={self._resurrection_successes}")
        print(f"  total_splits={self._splits_triggered}")

    def ask(self) -> np.ndarray:
        """Suggest the next point to evaluate."""
        self.iteration = len(self.X_all)

        # Global random for diversity - but still assign to a cube
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self.last_cube = self._find_containing_leaf(x)
            self._last_selected_pool = "random"
            return x

        # ALWAYS check for resurrection opportunity (regardless of phase)
        if (self.iteration % self._resurrection_interval == 0 and 
            self.iteration > 0 and 
            len(self.suspended_leaves) > 0):
            self._update_gamma()
            self._recount_good()
            self._last_cube_was_suspended = True
            self._last_selected_pool = "suspended"
            self.last_cube = self._tournament_select(self.suspended_leaves)
            return self._sample_in_cube(self.last_cube)

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            return self._sample_in_cube(self.last_cube)

        # Local search phase with tree integration
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress
        
        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self.last_cube = self._find_containing_leaf(x)  # Assign to tree
            self._last_selected_pool = "local_search"
        else:
            self._update_gamma()
            self._recount_good()
            self.last_cube = self._select_leaf()
            x = self._sample_in_cube(self.last_cube)
        
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        """Report the objective value for a point."""
        y = self._to_internal(y_raw)
        
        # Store previous best for success definition
        prev_best = self.best_y_internal
        
        # Track if this improved global best (for resurrection logic)
        improved_global_best = y > prev_best + 1e-12
        
        # Update best (internal score, always maximize)
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
        else:
            self.stagnation += 1

        self.X_all.append(x.copy())
        self.y_all.append(y)

        split_triggered = False
        
        if self.last_cube is not None:
            cube = self.last_cube
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            
            # Success = gamma-based OR improved global best OR close-to-gamma (lenient)
            is_good = y >= self.gamma
            # For resurrection, be more lenient: within 80% of gamma threshold is "promising"
            gamma_margin = 0.2 * abs(self.gamma) if self.gamma != 0 else 0.01
            is_promising = y >= self.gamma - gamma_margin
            success = is_good or improved_global_best
            
            # Track success for debug
            self._recent_successes.append(success)
            
            # Increment n_good based on gamma (keep consistent with original)
            if is_good:
                cube.n_good += 1
            
            # Update cube's best score
            if y > cube.best_score:
                cube.best_score = y
                cube.best_x = x.copy()
            
            # Resurrection logic: if cube was suspended, check for revival
            # Use more lenient "promising" criterion for resurrection (success is too strict)
            if self._last_cube_was_suspended:
                self._resurrection_attempts += 1
                resurrection_success = is_promising or improved_global_best
                if resurrection_success and cube in self.suspended_leaves:
                    self.suspended_leaves.remove(cube)
                    self.active_leaves.append(cube)
                    self._resurrection_successes += 1
                    # After resurrection, maybe suspend worst to maintain balance
                    self._maybe_suspend_worst()
            
            # Update categorical stats
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

            cube.fit_lgs_model(self.gamma, self.dim, self.rng)

            if self._should_split(cube):
                split_triggered = True
                self._splits_triggered += 1
                self._last_split_axis = cube.get_split_axis()
                children = cube.split(self.gamma, self.dim, self.rng)
                # Propagate categorical stats to children
                for child in children:
                    self._recompute_cat_stats(child)
                # Remove parent from active/suspended and add children to active
                if cube in self.active_leaves:
                    self.active_leaves.remove(cube)
                elif cube in self.suspended_leaves:
                    self.suspended_leaves.remove(cube)
                self.active_leaves.extend(children)
                # After split (reproduction), maybe suspend worst to control growth
                self._maybe_suspend_worst()
            
            # Debug logging
            self._debug_log(cube, split_triggered)
            
            # Check invariants in debug mode
            if self._debug:
                self._check_invariants()

            self.last_cube = None
            self._last_cube_was_suspended = False

    def _maybe_suspend_worst(self) -> None:
        """
        Soft pruning: if active_leaves exceeds max, suspend the worst cube.
        Only consider cubes with n_trials >= dim (avoid suspending young cubes).
        Worst = lowest good_ratio().
        """
        while len(self.active_leaves) > self._max_active_leaves:
            # Find candidates for suspension (mature cubes)
            candidates = [c for c in self.active_leaves if c.n_trials >= self.dim]
            
            if not candidates:
                # All cubes are young, can't suspend any
                break
            
            # Find worst by good_ratio
            worst = min(candidates, key=lambda c: c.good_ratio())
            
            # Move to suspended (seed bank)
            self.active_leaves.remove(worst)
            self.suspended_leaves.append(worst)

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
        # Recount for both active and suspended leaves
        for leaf in self._all_leaves():
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
            # Recompute cat_stats with current gamma to avoid stale stats
            self._recompute_cat_stats(leaf)

    def _thompson_sample(self, cube: Cube) -> float:
        """
        Thompson sampling using Beta posterior.
        Returns a sample from Beta(n_good + 1, n_failures + 1).
        """
        a = cube.n_good + 1
        b = (cube.n_trials - cube.n_good) + 1
        return self.rng.beta(a, b)

    def _tournament_select(self, pool: List[Cube]) -> Cube:
        """
        Tournament selection (size=2) using Thompson sampling.
        Sample 2 cubes, draw theta for each, return winner (higher theta).
        """
        if len(pool) == 0:
            return self.root
        if len(pool) == 1:
            return pool[0]
        
        # Sample 2 distinct cubes
        idx = self.rng.choice(len(pool), size=min(2, len(pool)), replace=False)
        candidates = [pool[i] for i in idx]
        
        # Thompson sample for each
        thetas = [self._thompson_sample(c) for c in candidates]
        
        # Return cube with highest theta
        winner_idx = np.argmax(thetas)
        return candidates[winner_idx]

    def _select_leaf(self) -> Cube:
        """
        Evolutionary leaf selection from active pool:
        - During warm-up: uniform random selection (avoid Thompson chaos)
        - Otherwise: tournament selection using Thompson sampling
        Note: resurrection is handled in ask() directly.
        """
        self._last_cube_was_suspended = False
        
        # Warm-up: bypass Thompson sampling early to gather data
        if self.iteration < self._warmup_iters:
            self._last_selected_pool = "warmup"
            if not self.active_leaves:
                return self.root
            # Uniform random selection during warm-up
            return self.rng.choice(self.active_leaves)
        
        # Normal selection from active leaves using tournament
        self._last_selected_pool = "active"
        if not self.active_leaves:
            return self.root
        
        return self._tournament_select(self.active_leaves)

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
        """Find the leaf cube that contains point x (search both active and suspended)."""
        for leaf in self._all_leaves():
            if leaf.contains(x):
                return leaf
        # Fallback: return leaf with closest center (shouldn't happen normally)
        min_dist = float('inf')
        closest = self.active_leaves[0] if self.active_leaves else self.root
        for leaf in self._all_leaves():
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
        """Convert continuous [0,1] value to discrete category index."""
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)
    
    def _cat_value_to_continuous(self, val_idx: int, n_choices: int) -> float:
        """Convert discrete category index back to continuous [0,1]."""
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5
    
    def _recompute_cat_stats(self, cube: Cube) -> None:
        """Recompute categorical stats for a cube from its tested pairs."""
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
        """
        Apply TPE-like categorical sampling based on cube's category stats.
        Sample categorical values proportional to (n_good + 1) / (n_total + K).
        """
        if not self.categorical_dims:
            return x
        
        x = x.copy()
        
        # Increase exploration factor during stagnation
        exploration_boost = 2.0 if self.stagnation > self._stagnation_threshold else 1.0
        
        for dim_idx, n_choices in self.categorical_dims:
            stats = cube.cat_stats.get(dim_idx, {})
            
            # Compute probability for each categorical value
            probs = []
            for v in range(n_choices):
                n_g, n_t = stats.get(v, (0, 0))
                # Beta-like prior: (n_good + 1) / (n_total + K)
                # K = n_choices for uniform prior, increase for more exploration
                K = n_choices * exploration_boost
                prob = (n_g + 1) / (n_t + K)
                probs.append(prob)
            
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            # Sample categorical value
            chosen = self.rng.choice(n_choices, p=probs)
            x[dim_idx] = self._cat_value_to_continuous(chosen, n_choices)
        
        return x

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        if self.iteration < 15:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        else:
            x = self._sample_with_lgs(cube)
        
        # Apply categorical sampling based on cube stats
        x = self._apply_categorical_sampling(x, cube)
        
        # Re-clip to cube bounds (categorical sampling may have moved outside)
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
        # Evolutionary gate: require some successes to avoid noisy splits
        if cube.n_good < 2:
            return False
        return cube.n_trials >= self._split_trials_factor * self.dim + self._split_trials_offset

    def _update_all_models(self) -> None:
        # Update models for all leaves (both active and suspended)
        for leaf in self._all_leaves():
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        """
        Run optimization loop.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Function to optimize.
        budget : int, default=100
            Number of function evaluations.
            
        Returns
        -------
        best_x : np.ndarray
            Best configuration found.
        best_y : float
            Best objective value found.
        """
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        return self.best_x, self._to_raw(self.best_y_internal)
