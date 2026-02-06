#!/usr/bin/env python3
"""
FIX CONSERVATIVE: modifiche che NON alterano la sequenza random rispetto all'originale.

Questo permette di isolare l'effetto di ogni singola modifica.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import sys
import types
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# VERSIONE A: Solo Beta Prior (non altera RNG)
# =============================================================================
@dataclass(eq=False)
class CubeA:
    """Solo modifica: Beta prior per good_ratio"""
    bounds: List[Tuple[float, float]]
    parent: Optional["CubeA"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[dict] = field(default=None, init=False)
    depth: int = 0

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
        # FIX: Beta prior (1,1)
        return (self.n_good + 1) / (self.n_trials + 2)

    def fit_lgs_model(self, gamma: float, dim: int) -> None:
        # IDENTICO all'originale
        pairs = list(self._tested_pairs)
        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, "_tested_pairs", [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
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
                noise_var = np.average(residuals**2, weights=weights) + 1e-6
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-9:
                    gradient_dir = grad / grad_norm
            except Exception:
                pass

        self.lgs_model = {
            "all_pts": all_pts, "top_k_pts": top_k_pts, "gradient_dir": gradient_dir,
            "grad": grad, "inv_cov": inv_cov, "y_mean": y_mean, "noise_var": noise_var,
            "widths": widths, "center": center,
        }

    def predict_bayesian(self, candidates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self.lgs_model is None or self.lgs_model.get("inv_cov") is None:
            return np.zeros(len(candidates)), np.ones(len(candidates))
        model = self.lgs_model
        C_norm = (np.array(candidates) - model["center"]) / model["widths"]
        mu = model["y_mean"] + C_norm @ model["grad"]
        model_var = np.sum((C_norm @ model["inv_cov"]) * C_norm, axis=1)
        sigma = np.sqrt(model["noise_var"] * (1.0 + model_var))
        return mu, sigma

    def get_split_axis(self) -> int:
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int) -> List["CubeA"]:
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

        child_lo = CubeA(bounds=bounds_lo, parent=self)
        child_hi = CubeA(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1

        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1

        for ch in (child_lo, child_hi):
            ch.fit_lgs_model(gamma, dim)

        return [child_lo, child_hi]


class HPOptimizerV5s_BetaPrior:
    """Solo FIX: Beta prior per good_ratio - NON altera sequenza random"""

    def __init__(self, bounds, maximize=False, seed=42, total_budget=200, split_depth_max=8, **kwargs):
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.total_budget = total_budget
        self.gamma_quantile = 0.20
        self.gamma_quantile_start = 0.15
        self.local_search_ratio = 0.30
        self.n_candidates = 25

        self.root = CubeA(bounds=list(bounds))
        self.leaves: List[CubeA] = [self.root]
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.best_y = -np.inf if maximize else np.inf
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0

        self.exploration_budget = int(total_budget * (1 - self.local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget
        self.global_widths = np.array([hi - lo for lo, hi in bounds])
        self.last_cube: Optional[CubeA] = None

        self._v5s_split_trials_min = 15
        self._v5s_split_depth_max = split_depth_max
        self._v5s_split_trials_factor = 3.0
        self._v5s_split_trials_offset = 6
        self._v5s_novelty_weight = 0.4

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)
        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()
            if self.iteration % 5 == 0:
                self._update_all_models()
            self.last_cube = self._select_leaf()
            return self._sample_in_cube(self.last_cube)
        
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        self.last_cube = None
        return self._local_search_sample(progress)

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        y = y_raw if self.maximize else -y_raw
        self._update_best(x, y_raw)
        self.X_all.append(x.copy())
        self.y_all.append(y)

        if self.last_cube is not None:
            cube = self.last_cube
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            if y >= self.gamma:
                cube.n_good += 1
            cube.fit_lgs_model(self.gamma, self.dim)
            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)
            self.last_cube = None

    def _update_gamma(self) -> None:
        if len(self.y_all) < 10:
            self.gamma = 0.0
            return
        progress = min(1.0, self.iteration / max(1, self.exploration_budget * 0.5))
        current_quantile = self.gamma_quantile_start - progress * (self.gamma_quantile_start - self.gamma_quantile)
        self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))

    def _recount_good(self) -> None:
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)

    def _update_best(self, x: np.ndarray, y_raw: float) -> None:
        if self.maximize:
            if y_raw > self.best_y:
                self.best_y, self.best_x = y_raw, x.copy()
        else:
            if y_raw < self.best_y:
                self.best_y, self.best_x = y_raw, x.copy()

    def _select_leaf(self) -> CubeA:
        if not self.leaves:
            return self.root
        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()  # USA BETA PRIOR!
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            model_bonus = 0.1 if c.lgs_model and len(c.lgs_model.get("all_pts", [])) >= self.dim + 2 else 0.0
            scores.append(ratio + exploration + model_bonus)
        scores_arr = np.array(scores) - max(scores)
        probs = np.exp(scores_arr * 3)
        probs /= probs.sum()
        return self.leaves[int(self.rng.choice(len(self.leaves), p=probs))]

    def _clip_to_cube(self, x, cube):
        return np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(self.dim)])

    def _clip_to_bounds(self, x):
        return np.array([np.clip(x[i], self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)])

    def _generate_candidates(self, cube, n):
        candidates = []
        widths = cube._widths()
        center = cube.center()
        model = cube.lgs_model
        for _ in range(n):
            strategy = self.rng.random()
            if strategy < 0.25 and model and len(model["top_k_pts"]) > 0:
                idx = self.rng.integers(len(model["top_k_pts"]))
                x = model["top_k_pts"][idx] + self.rng.normal(0, 0.15, self.dim) * widths
            elif strategy < 0.40 and model and model["gradient_dir"] is not None:
                top_center = model["top_k_pts"].mean(axis=0)
                step = self.rng.uniform(0.05, 0.3)
                x = top_center + step * model["gradient_dir"] * widths + self.rng.normal(0, 0.05, self.dim) * widths
            elif strategy < 0.55:
                x = center + self.rng.normal(0, 0.2, self.dim) * widths
            else:
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            candidates.append(self._clip_to_cube(x, cube))
        return candidates

    def _sample_with_lgs(self, cube):
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        candidates = self._generate_candidates(cube, self.n_candidates)
        mu, sigma = cube.predict_bayesian(candidates)
        score = mu + self._v5s_novelty_weight * 2.0 * sigma
        score_z = (score - score.mean()) / score.std() if score.std() > 1e-9 else np.zeros_like(score)
        probs = np.exp(score_z * 3.0)
        probs /= probs.sum()
        return candidates[int(self.rng.choice(len(candidates), p=probs))]

    def _sample_in_cube(self, cube):
        if self.iteration < 15:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        return self._sample_with_lgs(cube)

    def _local_search_sample(self, progress):
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        radius = 0.10 * (1 - progress) + 0.02
        return self._clip_to_bounds(self.best_x + self.rng.normal(0, radius, self.dim) * self.global_widths)

    def _should_split(self, cube):
        if cube.n_trials < self._v5s_split_trials_min or cube.depth >= self._v5s_split_depth_max:
            return False
        return cube.n_trials >= self._v5s_split_trials_factor * self.dim + self._v5s_split_trials_offset

    def _update_all_models(self):
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim)

    def optimize(self, objective, budget=100):
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget
        for _ in range(budget):
            x = self.ask()
            self.tell(x, objective(x))
        return self.best_x, self.best_y


# =============================================================================
# TEST
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=600)
    parser.add_argument('--seeds', type=str, default='10,11,23,24,25')
    parser.add_argument('--task_id', type=int, default=31)
    parser.add_argument('--target_loss', type=float, default=0.092)
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    sys.path.insert(0, '/mnt/workspace/HPOBench')
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    from ConfigSpace import Configuration
    
    print("Loading benchmark...")
    bench = TabularBenchmark(model='nn', task_id=args.task_id)
    
    # Fast lookup
    import pandas as pd
    df = bench.table
    idx_cols = [c for c in df.columns if c != 'result']
    indexed = df.set_index(idx_cols, drop=False).sort_index()
    cache = {}
    def _search_fast(self, row_dict, _df):
        key = tuple(row_dict[c] for c in idx_cols)
        if key in cache: return cache[key]
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        res = row['result']
        cache[key] = res
        return res
    bench._search_dataframe = types.MethodType(_search_fast, bench)
    
    cs = bench.get_configuration_space()
    hp_names = list(cs.get_hyperparameter_names())
    
    bounds = []
    for hp_name in hp_names:
        hp = cs.get_hyperparameter(hp_name)
        if hasattr(hp, 'sequence'):
            bounds.append((0.0, float(len(hp.sequence) - 1)))
        else:
            bounds.append((float(hp.lower), float(hp.upper)))
    
    def objective(x):
        config_dict = {}
        for i, hp_name in enumerate(hp_names):
            hp = cs.get_hyperparameter(hp_name)
            if hasattr(hp, 'sequence'):
                idx = int(round(np.clip(x[i], 0, len(hp.sequence) - 1)))
                config_dict[hp_name] = hp.sequence[idx]
            else:
                config_dict[hp_name] = float(np.clip(x[i], hp.lower, hp.upper))
        config = Configuration(cs, values=config_dict)
        return float(bench.objective_function(config)['function_value'])
    
    sys.path.insert(0, '/mnt/workspace/thesis')
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s
    
    print("=" * 80)
    print("TEST: ORIGINAL vs BETA_PRIOR_ONLY")
    print("=" * 80)
    
    results_orig = []
    results_beta = []
    
    for seed in seeds:
        # Original
        opt_orig = HPOptimizerV5s(bounds=bounds, maximize=False, seed=seed, 
                                  total_budget=args.budget, split_depth_max=8)
        opt_orig.optimize(objective, args.budget)
        results_orig.append(opt_orig.best_y)
        
        # Beta prior only
        opt_beta = HPOptimizerV5s_BetaPrior(bounds=bounds, maximize=False, seed=seed,
                                            total_budget=args.budget, split_depth_max=8)
        opt_beta.optimize(objective, args.budget)
        results_beta.append(opt_beta.best_y)
        
        print(f"Seed {seed}: ORIG={opt_orig.best_y:.6f}, BETA={opt_beta.best_y:.6f}, "
              f"diff={opt_beta.best_y - opt_orig.best_y:+.6f}")
    
    print(f"\nORIGINAL: {np.mean(results_orig):.6f} ± {np.std(results_orig):.6f}")
    print(f"BETA:     {np.mean(results_beta):.6f} ± {np.std(results_beta):.6f}")
    print(f"Stuck ORIG: {sum(1 for r in results_orig if r > args.target_loss)}/{len(seeds)}")
    print(f"Stuck BETA: {sum(1 for r in results_beta if r > args.target_loss)}/{len(seeds)}")


if __name__ == "__main__":
    main()
