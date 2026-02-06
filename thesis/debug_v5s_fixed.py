#!/usr/bin/env python3
"""
Debug script con FIX applicate basate sull'analisi:

FIX APPLICATE:
1. Beta prior per good_ratio: (good + 1) / (trials + 2) invece di good/trials
2. Parent backfill con shuffle invece di prendere i primi
3. Pure random globale 5% delle volte durante exploration
4. Local search integrato nel tree invece di staccato completamente
5. Stagnation-aware exploration: se stagnation > 50, aumenta esplorazione
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import sys
import warnings
warnings.filterwarnings('ignore')


@dataclass(eq=False)
class CubeFixed:
    """Cube con fix applicate"""
    bounds: List[Tuple[float, float]]
    parent: Optional["CubeFixed"] = None
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
        # FIX B: Beta prior (1,1) -> evita che 1 punto domini
        # Formula: (good + alpha) / (trials + alpha + beta) con alpha=beta=1
        return (self.n_good + 1) / (self.n_trials + 2)

    def fit_lgs_model(self, gamma: float, dim: int, rng: np.random.Generator = None) -> None:
        pairs = list(self._tested_pairs)

        # FIX C: Parent backfill con shuffle invece di prendere i primi
        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, "_tested_pairs", [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
                if rng is not None:
                    extra = list(extra)  # Make a copy before shuffling
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
                noise_var = np.average(residuals**2, weights=weights) + 1e-6
                
                # FIX: Floor/ceiling su noise_var
                noise_var = np.clip(noise_var, 1e-4, 10.0)
                
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

    def _to_ranks(self, scores: List[float], higher_is_better: bool) -> np.ndarray:
        scores_arr = np.array(scores)
        n = len(scores_arr)
        if not higher_is_better:
            scores_arr = -scores_arr
        order = np.argsort(scores_arr)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        return ranks / max(1, n - 1)

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
        
        model_var = np.sum((C_norm @ inv_cov) * C_norm, axis=1)
        # FIX: Cap su model_var per evitare esplosioni
        model_var = np.clip(model_var, 0, 10.0)
        total_var = noise_var * (1.0 + model_var)
        sigma = np.sqrt(total_var)
        
        return mu, sigma

    def get_split_axis(self) -> int:
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int) -> List["CubeFixed"]:
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

        child_lo = CubeFixed(bounds=bounds_lo, parent=self)
        child_hi = CubeFixed(bounds=bounds_hi, parent=self)
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


class HPOptimizerV5sFixed:
    """
    HPOptimizerV5s con FIX applicate:
    1. Beta prior per good_ratio
    2. Parent backfill con shuffle
    3. Pure random globale 5%
    4. Local search integrato nel tree (non staccato)
    5. Stagnation-aware exploration
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
        split_depth_max: int = 8,
        split_trials_factor: float = 3.0,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        # Nuovi parametri per le fix
        global_random_prob: float = 0.05,  # FIX 3: probabilità di random globale
        stagnation_threshold: int = 50,     # FIX 5: soglia per aumentare esplorazione
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

        self.root = CubeFixed(bounds=list(bounds))
        self.leaves: List[CubeFixed] = [self.root]

        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.best_y = -np.inf if maximize else np.inf
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0
        
        # Tracking stagnation
        self.stagnation = 0
        self.last_improvement_iter = 0

        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        self.global_widths = np.array([hi - lo for lo, hi in bounds])
        self.last_cube: Optional[CubeFixed] = None

        self._v5s_split_trials_min = split_trials_min
        self._v5s_split_depth_max = split_depth_max
        self._v5s_split_trials_factor = split_trials_factor
        self._v5s_split_trials_offset = split_trials_offset
        self._v5s_novelty_weight = novelty_weight
        
        # Nuovi parametri
        self._global_random_prob = global_random_prob
        self._stagnation_threshold = stagnation_threshold

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)

        # FIX 3: Pure random globale con probabilità global_random_prob
        if self.rng.random() < self._global_random_prob:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            x = self._sample_in_cube(self.last_cube)
            return x

        # FIX 4: Local search integrato nel tree invece di staccato
        # Continua a usare il tree ma con bias verso il best
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        
        # Con probabilità crescente, fai local search puro attorno a best_x
        # Altrimenti continua con il tree
        local_search_prob = 0.5 + 0.4 * progress  # da 0.5 a 0.9
        
        if self.rng.random() < local_search_prob:
            # Local search classico
            x = self._local_search_sample(progress)
            self.last_cube = None
        else:
            # Continua con il tree (esplorazione residua)
            self._update_gamma()
            self._recount_good()
            self.last_cube = self._select_leaf()
            x = self._sample_in_cube(self.last_cube)
        
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        y = y_raw if self.maximize else -y_raw
        
        # Track stagnation
        old_best = self.best_y
        self._update_best(x, y_raw)
        
        if self.best_y != old_best:
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
            if y >= self.gamma:
                cube.n_good += 1

            cube.fit_lgs_model(self.gamma, self.dim, self.rng)

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
        current_quantile = self.gamma_quantile_start - progress * (
            self.gamma_quantile_start - self.gamma_quantile
        )
        self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))

    def _recount_good(self) -> None:
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)

    def _update_best(self, x: np.ndarray, y_raw: float) -> None:
        if self.maximize:
            if y_raw > self.best_y:
                self.best_y = y_raw
                self.best_x = x.copy()
        else:
            if y_raw < self.best_y:
                self.best_y = y_raw
                self.best_x = x.copy()

    def _select_leaf(self) -> CubeFixed:
        if not self.leaves:
            return self.root

        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()  # Ora usa Beta prior
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            
            # FIX 5: Se stagnation alta, aumenta exploration bonus
            if self.stagnation > self._stagnation_threshold:
                exploration *= 2.0  # Raddoppia exploration bonus
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= self.dim + 2:
                    model_bonus = 0.1
            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        
        # FIX 5: Se stagnation alta, usa temperatura più alta (più random)
        temperature = 3.0
        if self.stagnation > self._stagnation_threshold:
            temperature = 1.5  # Più piatto = più random
        
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.leaves), p=probs)
        return self.leaves[int(idx)]

    def _clip_to_cube(self, x: np.ndarray, cube: CubeFixed) -> np.ndarray:
        return np.array([
            np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1])
            for i in range(self.dim)
        ])

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(x[i], self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dim)
        ])

    def _generate_candidates(self, cube: CubeFixed, n: int) -> List[np.ndarray]:
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

    def _sample_with_lgs(self, cube: CubeFixed) -> np.ndarray:
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        candidates = self._generate_candidates(cube, self.n_candidates)
        
        mu, sigma = cube.predict_bayesian(candidates)
        
        beta = self._v5s_novelty_weight * 2.0
        score = mu + beta * sigma
        
        if score.std() > 1e-9:
            score_z = (score - score.mean()) / score.std()
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * 3.0)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _sample_in_cube(self, cube: CubeFixed) -> np.ndarray:
        if self.iteration < 15:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        return self._sample_with_lgs(cube)

    def _local_search_sample(self, progress: float) -> np.ndarray:
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        # FIX: Radius leggermente più grande all'inizio
        radius = 0.15 * (1 - progress) + 0.03
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = self.best_x + noise
        return self._clip_to_bounds(x)

    def _should_split(self, cube: CubeFixed) -> bool:
        if cube.n_trials < self._v5s_split_trials_min:
            return False
        if cube.depth >= self._v5s_split_depth_max:
            return False
        return cube.n_trials >= self._v5s_split_trials_factor * self.dim + self._v5s_split_trials_offset

    def _update_all_models(self) -> None:
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        return self.best_x, self.best_y


# =============================================================================
# TEST COMPARATIVO
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
    
    # Setup HPOBench
    sys.path.insert(0, '/mnt/workspace/HPOBench')
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    
    print("Loading benchmark...")
    bench = TabularBenchmark(model='nn', task_id=args.task_id)
    cs = bench.get_configuration_space()
    hp_names = list(cs.get_hyperparameter_names())
    
    # Get bounds
    bounds = []
    for hp_name in hp_names:
        hp = cs.get_hyperparameter(hp_name)
        if hasattr(hp, 'lower'):
            bounds.append((float(hp.lower), float(hp.upper)))
        elif hasattr(hp, 'sequence'):
            bounds.append((0.0, float(len(hp.sequence) - 1)))
        else:
            bounds.append((0.0, 1.0))
    
    def objective(x):
        config_dict = {}
        for i, hp_name in enumerate(hp_names):
            hp = cs.get_hyperparameter(hp_name)
            if hasattr(hp, 'sequence'):
                idx = int(round(np.clip(x[i], 0, len(hp.sequence) - 1)))
                config_dict[hp_name] = hp.sequence[idx]
            else:
                config_dict[hp_name] = float(np.clip(x[i], hp.lower, hp.upper))
        
        from ConfigSpace import Configuration
        config = Configuration(cs, values=config_dict)
        result = bench.objective_function(config)
        return float(result['function_value'])
    
    print("=" * 80)
    print("CONFRONTO: ORIGINAL vs FIXED")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Budget: {args.budget}")
    print(f"Task ID: {args.task_id}")
    print(f"Target loss (stuck if >): {args.target_loss}")
    print()
    
    # Import original
    sys.path.insert(0, '/mnt/workspace/thesis')
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s
    
    results_original = []
    results_fixed = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        # Original
        print("  Running ORIGINAL...", end=" ", flush=True)
        opt_orig = HPOptimizerV5s(
            bounds=bounds,
            maximize=False,
            seed=seed,
            total_budget=args.budget,
            split_depth_max=8,
        )
        _, best_orig = opt_orig.optimize(objective, args.budget)
        results_original.append(best_orig)
        status_orig = "STUCK" if best_orig > args.target_loss else "OK"
        print(f"loss={best_orig:.6f} [{status_orig}]")
        
        # Fixed
        print("  Running FIXED...", end=" ", flush=True)
        opt_fixed = HPOptimizerV5sFixed(
            bounds=bounds,
            maximize=False,
            seed=seed,
            total_budget=args.budget,
            split_depth_max=8,
            global_random_prob=0.05,
            stagnation_threshold=50,
        )
        _, best_fixed = opt_fixed.optimize(objective, args.budget)
        results_fixed.append(best_fixed)
        status_fixed = "STUCK" if best_fixed > args.target_loss else "OK"
        print(f"loss={best_fixed:.6f} [{status_fixed}]")
        
        # Confronto
        diff = best_fixed - best_orig
        better = "FIXED" if diff < 0 else "ORIGINAL" if diff > 0 else "TIE"
        print(f"  -> Winner: {better} (diff={diff:+.6f})")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    orig_arr = np.array(results_original)
    fixed_arr = np.array(results_fixed)
    
    print(f"\nORIGINAL: {orig_arr.mean():.6f} ± {orig_arr.std():.6f}")
    print(f"  Results: {[f'{r:.6f}' for r in results_original]}")
    print(f"  Stuck (>{args.target_loss}): {sum(1 for r in results_original if r > args.target_loss)}/{len(seeds)}")
    
    print(f"\nFIXED:    {fixed_arr.mean():.6f} ± {fixed_arr.std():.6f}")
    print(f"  Results: {[f'{r:.6f}' for r in results_fixed]}")
    print(f"  Stuck (>{args.target_loss}): {sum(1 for r in results_fixed if r > args.target_loss)}/{len(seeds)}")
    
    # Confronto statistico
    wins_fixed = sum(1 for o, f in zip(results_original, results_fixed) if f < o)
    wins_orig = sum(1 for o, f in zip(results_original, results_fixed) if o < f)
    ties = len(seeds) - wins_fixed - wins_orig
    
    print(f"\nHEAD-TO-HEAD: FIXED wins {wins_fixed}, ORIGINAL wins {wins_orig}, ties {ties}")
    
    if fixed_arr.mean() < orig_arr.mean():
        improvement = (orig_arr.mean() - fixed_arr.mean()) / orig_arr.mean() * 100
        print(f"\n✅ FIXED è MIGLIORE del {improvement:.1f}%")
    else:
        degradation = (fixed_arr.mean() - orig_arr.mean()) / orig_arr.mean() * 100
        print(f"\n❌ FIXED è PEGGIORE del {degradation:.1f}%")


if __name__ == "__main__":
    main()
