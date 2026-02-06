#!/usr/bin/env python3
"""
ALBA_V1_debug - Debug version with extensive logging to understand stagnation issues.
Created for debugging seed 100 on JAHS-Bench-201 cifar10.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


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
    cube_id: int = 0  # DEBUG: unique ID for tracking

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
                noise_var = np.clip(np.average(residuals**2, weights=weights) + 1e-6, 1e-4, 10.0)
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

    def split(self, gamma: float, dim: int, rng: np.random.Generator = None, next_id: int = 0) -> Tuple[List["Cube"], int]:
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

        child_lo = Cube(bounds=bounds_lo, parent=self, cube_id=next_id)
        child_hi = Cube(bounds=bounds_hi, parent=self, cube_id=next_id + 1)
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

        return [child_lo, child_hi], next_id + 2


class ALBA:
    """ALBA with extensive debug logging."""

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
        debug: bool = True,  # DEBUG MODE
        debug_milestones: List[int] = None,
    ) -> None:
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = gamma_quantile_start
        self.local_search_ratio = local_search_ratio
        self.n_candidates = n_candidates
        self.total_budget = total_budget
        self.categorical_dims = categorical_dims or []
        
        # DEBUG flags
        self.debug = debug
        self.debug_milestones = debug_milestones or [100, 200, 500, 1000, 1500, 2000]
        self.debug_log = []

        self.root = Cube(bounds=list(bounds), cube_id=0)
        self.leaves: List[Cube] = [self.root]
        self.next_cube_id = 1

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
        
        # DEBUG: tracking
        self.split_history = []  # (iter, parent_id, child1_id, child2_id, axis)
        self.leaf_selection_history = []  # (iter, cube_id, n_leaves)
        self.sample_strategy_history = []  # (iter, strategy_type)
        self.improvements = []  # (iter, old_best, new_best)

    def _log(self, msg: str):
        if self.debug:
            self.debug_log.append(f"[iter={self.iteration}] {msg}")
    
    def _log_milestone(self):
        if not self.debug:
            return
        if self.iteration not in self.debug_milestones:
            return
            
        best_raw = self._to_raw(self.best_y_internal)
        print(f"\n{'='*60}")
        print(f"MILESTONE: iter={self.iteration}, best_error={best_raw:.4f} ({(1-best_raw)*100:.2f}%)")
        print(f"{'='*60}")
        print(f"  Stagnation: {self.stagnation} iters since improvement")
        print(f"  Last improvement at iter: {self.last_improvement_iter}")
        print(f"  Gamma (good threshold): {self.gamma:.4f}")
        print(f"  Number of leaves: {len(self.leaves)}")
        print(f"  Tree depth range: {min(c.depth for c in self.leaves)} - {max(c.depth for c in self.leaves)}")
        
        # Leaf stats
        print(f"\n  Top 5 leaves by good_ratio:")
        sorted_leaves = sorted(self.leaves, key=lambda c: c.good_ratio(), reverse=True)[:5]
        for c in sorted_leaves:
            print(f"    Cube {c.cube_id}: depth={c.depth}, trials={c.n_trials}, good={c.n_good}, ratio={c.good_ratio():.3f}")
            if c.best_x is not None:
                best_err = -c.best_score
                print(f"      best_in_cube: error={best_err:.4f}, x[:4]={c.best_x[:4]}")
        
        # Check if best point is in a leaf
        if self.best_x is not None:
            containing = None
            for leaf in self.leaves:
                if leaf.contains(self.best_x):
                    containing = leaf
                    break
            if containing:
                print(f"\n  Best point is in Cube {containing.cube_id} (depth={containing.depth}, trials={containing.n_trials})")
                print(f"    Cube bounds widths: {containing._widths()[:4]}...")
            else:
                print(f"\n  WARNING: Best point not contained in any leaf!")
        
        # Sample diversity check
        if len(self.X_all) > 0:
            recent = np.array(self.X_all[-100:]) if len(self.X_all) >= 100 else np.array(self.X_all)
            print(f"\n  Recent sample stats (last {len(recent)}):")
            for i in range(min(4, self.dim)):
                vals = recent[:, i]
                print(f"    dim {i}: mean={vals.mean():.3f}, std={vals.std():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]")
        
        # Strategy distribution
        if self.sample_strategy_history:
            recent_strats = [s[1] for s in self.sample_strategy_history[-100:]]
            from collections import Counter
            counts = Counter(recent_strats)
            print(f"\n  Recent sampling strategies: {dict(counts)}")

    def _to_internal(self, y_raw: float) -> float:
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        return y_internal if self.maximize else -y_internal

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)
        
        # Log milestone
        if self.debug and self.iteration in self.debug_milestones:
            self._log_milestone()

        # Global random
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self.last_cube = self._find_containing_leaf(x)
            self.sample_strategy_history.append((self.iteration, 'global_random'))
            self._log(f"GLOBAL_RANDOM: cube={self.last_cube.cube_id}")
            return x

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            self.leaf_selection_history.append((self.iteration, self.last_cube.cube_id, len(self.leaves)))
            x = self._sample_in_cube(self.last_cube)
            return x

        # Local search phase
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress
        
        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self.last_cube = self._find_containing_leaf(x)
            self.sample_strategy_history.append((self.iteration, 'local_search'))
            self._log(f"LOCAL_SEARCH: x[:4]={x[:4]}")
        else:
            self._update_gamma()
            self._recount_good()
            self.last_cube = self._select_leaf()
            self.leaf_selection_history.append((self.iteration, self.last_cube.cube_id, len(self.leaves)))
            x = self._sample_in_cube(self.last_cube)
        
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        y = self._to_internal(y_raw)
        
        old_best = self.best_y_internal
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.improvements.append((self.iteration, -old_best, -y))  # Store as errors
            self._log(f"NEW_BEST: error={y_raw:.4f} (was {-old_best:.4f}), x[:4]={x[:4]}")
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
            
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

            cube.fit_lgs_model(self.gamma, self.dim, self.rng)

            if self._should_split(cube):
                parent_id = cube.cube_id
                children, self.next_cube_id = cube.split(self.gamma, self.dim, self.rng, self.next_cube_id)
                for child in children:
                    self._recompute_cat_stats(child)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)
                
                axis = cube.get_split_axis()
                self.split_history.append((self.iteration, parent_id, children[0].cube_id, children[1].cube_id, axis))
                self._log(f"SPLIT: cube {parent_id} -> ({children[0].cube_id}, {children[1].cube_id}) on axis {axis}")

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
        return np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(self.dim)])

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.clip(x[i], self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)])

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

    def _generate_candidates(self, cube: Cube, n: int) -> Tuple[List[np.ndarray], List[str]]:
        candidates: List[np.ndarray] = []
        strategies: List[str] = []
        widths = cube._widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = self.rng.random()

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = self.rng.integers(len(model["top_k_pts"]))
                x = model["top_k_pts"][idx] + self.rng.normal(0, 0.15, self.dim) * widths
                strat_name = 'top_k_perturb'
            elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
                top_center = model["top_k_pts"].mean(axis=0)
                step = self.rng.uniform(0.05, 0.3)
                x = top_center + step * model["gradient_dir"] * widths
                x = x + self.rng.normal(0, 0.05, self.dim) * widths
                strat_name = 'gradient_guided'
            elif strategy < 0.55:
                x = center + self.rng.normal(0, 0.2, self.dim) * widths
                strat_name = 'center_perturb'
            else:
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
                strat_name = 'uniform'

            x = self._clip_to_cube(x, cube)
            candidates.append(x)
            strategies.append(strat_name)

        return candidates, strategies

    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        if cube.lgs_model is None or cube.n_trials < 5:
            strat = 'cube_uniform'
            self.sample_strategy_history.append((self.iteration, strat))
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        candidates, strategies = self._generate_candidates(cube, self.n_candidates)
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
        
        chosen_strat = strategies[idx]
        self.sample_strategy_history.append((self.iteration, f'lgs_{chosen_strat}'))
        
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
            self.sample_strategy_history.append((self.iteration, 'early_random'))
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

    def get_debug_summary(self) -> dict:
        """Return comprehensive debug info."""
        return {
            'seed': self.seed,
            'total_evals': len(self.X_all),
            'best_error': self._to_raw(self.best_y_internal),
            'final_stagnation': self.stagnation,
            'last_improvement_iter': self.last_improvement_iter,
            'n_leaves': len(self.leaves),
            'n_splits': len(self.split_history),
            'n_improvements': len(self.improvements),
            'improvements': self.improvements,
            'split_history': self.split_history,
            'leaf_depths': [c.depth for c in self.leaves],
            'leaf_trials': [c.n_trials for c in self.leaves],
        }
    
    def print_final_analysis(self):
        """Print detailed analysis at end of optimization."""
        print(f"\n{'='*70}")
        print(f"FINAL ANALYSIS - Seed {self.seed}")
        print(f"{'='*70}")
        
        best_err = self._to_raw(self.best_y_internal)
        print(f"Best error: {best_err:.4f} ({(1-best_err)*100:.2f}%)")
        print(f"Total evaluations: {len(self.X_all)}")
        print(f"Final stagnation: {self.stagnation} iters")
        print(f"Last improvement at iter: {self.last_improvement_iter}")
        
        print(f"\n--- Improvement History ---")
        for i, (it, old, new) in enumerate(self.improvements):
            print(f"  {i+1}. iter={it}: {old:.4f} -> {new:.4f} (delta={old-new:.4f})")
        
        print(f"\n--- Tree Statistics ---")
        print(f"Number of leaves: {len(self.leaves)}")
        print(f"Total splits: {len(self.split_history)}")
        depths = [c.depth for c in self.leaves]
        trials = [c.n_trials for c in self.leaves]
        print(f"Leaf depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
        print(f"Leaf trials: min={min(trials)}, max={max(trials)}, mean={np.mean(trials):.1f}")
        
        # Where was best found?
        if self.best_x is not None:
            containing = None
            for leaf in self.leaves:
                if leaf.contains(self.best_x):
                    containing = leaf
                    break
            if containing:
                print(f"\nBest point in Cube {containing.cube_id}:")
                print(f"  depth={containing.depth}, trials={containing.n_trials}, good={containing.n_good}")
                print(f"  bounds widths: {containing._widths()[:4]}...")
        
        # Analyze stagnation
        print(f"\n--- Stagnation Analysis ---")
        if self.improvements:
            gaps = []
            for i in range(1, len(self.improvements)):
                gap = self.improvements[i][0] - self.improvements[i-1][0]
                gaps.append(gap)
            if gaps:
                print(f"Gaps between improvements: {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
                print(f"Mean gap: {np.mean(gaps):.1f}, Max gap: {max(gaps)}")
        
        # Sample diversity
        if len(self.X_all) > 100:
            last_500 = np.array(self.X_all[-500:]) if len(self.X_all) >= 500 else np.array(self.X_all)
            print(f"\n--- Sample Diversity (last {len(last_500)}) ---")
            for i in range(min(6, self.dim)):
                vals = last_500[:, i]
                print(f"  dim {i}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)
        
        if self.debug:
            self.print_final_analysis()

        return self.best_x, self._to_raw(self.best_y_internal)
