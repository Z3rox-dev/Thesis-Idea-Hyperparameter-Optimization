#!/usr/bin/env python3
"""
Debug script per capire perché HPO_v5s si blocca a valori sub-ottimali (es. 0.74 invece di 0.68).
Logga in dettaglio: stato albero, split, sampling, gamma, convergenza.
"""

import sys
import os
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import types
import json

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

# Copia inline di HPOptimizerV5s con hook di debug
from dataclasses import dataclass, field


@dataclass(eq=False)
class Cube:
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    cube_id: int = field(default=0, init=False)  # DEBUG: ID univoco

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
        if self.n_trials == 0:
            return 0.5
        return self.n_good / self.n_trials

    def volume(self) -> float:
        """Volume normalizzato del cubo (prodotto delle width)."""
        return float(np.prod(self._widths()))

    def fit_lgs_model(self, gamma: float, dim: int) -> None:
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
        total_var = noise_var * (1.0 + model_var)
        sigma = np.sqrt(total_var)
        
        return mu, sigma

    def get_split_axis(self) -> int:
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int, next_cube_id: int) -> Tuple[List["Cube"], int]:
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
        child_lo.cube_id = next_cube_id
        child_hi.cube_id = next_cube_id + 1

        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1

        for ch in (child_lo, child_hi):
            ch.fit_lgs_model(gamma, dim)

        return [child_lo, child_hi], next_cube_id + 2


class HPOptimizerV5sDebug:
    """Versione di HPOptimizerV5s con logging dettagliato per debug."""

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
        debug_log: bool = True,
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

        self.root = Cube(bounds=list(bounds))
        self.root.cube_id = 0
        self.next_cube_id = 1
        self.leaves: List[Cube] = [self.root]

        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.y_raw_all: List[float] = []  # DEBUG: valori originali (loss)
        self.best_y = -np.inf if maximize else np.inf
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0

        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        self.global_widths = np.array([hi - lo for lo, hi in bounds])
        self.last_cube: Optional[Cube] = None

        self._v5s_split_trials_min = split_trials_min
        self._v5s_split_depth_max = split_depth_max
        self._v5s_split_trials_factor = split_trials_factor
        self._v5s_split_trials_offset = split_trials_offset
        self._v5s_novelty_weight = novelty_weight

        # DEBUG state
        self.debug_log = debug_log
        self.debug_trace: List[dict] = []
        self.split_events: List[dict] = []
        self.stagnation_counter = 0
        self.last_improvement_iter = 0

    def ask(self) -> np.ndarray:
        self.iteration = len(self.X_all)

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            x = self._sample_in_cube(self.last_cube)
            return x

        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        x = self._local_search_sample(progress)
        self.last_cube = None
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        y = y_raw if self.maximize else -y_raw
        
        old_best = self.best_y
        self._update_best(x, y_raw)
        
        # Track stagnation
        if self.best_y != old_best:
            self.stagnation_counter = 0
            self.last_improvement_iter = self.iteration
        else:
            self.stagnation_counter += 1

        self.X_all.append(x.copy())
        self.y_all.append(y)
        self.y_raw_all.append(y_raw)

        cube_id_used = self.last_cube.cube_id if self.last_cube else -1
        
        split_happened = False
        if self.last_cube is not None:
            cube = self.last_cube
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            if y >= self.gamma:
                cube.n_good += 1

            cube.fit_lgs_model(self.gamma, self.dim)

            if self._should_split(cube):
                children, self.next_cube_id = cube.split(self.gamma, self.dim, self.next_cube_id)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)
                split_happened = True
                
                # Log split event
                self.split_events.append({
                    "iter": self.iteration,
                    "parent_id": cube.cube_id,
                    "parent_depth": cube.depth,
                    "child_ids": [c.cube_id for c in children],
                    "split_axis": cube.get_split_axis(),
                    "parent_n_trials": cube.n_trials,
                    "parent_good_ratio": cube.good_ratio(),
                })

            self.last_cube = None

        # DEBUG trace
        if self.debug_log:
            phase = "exploration" if self.iteration < self.exploration_budget else "local_search"
            trace_entry = {
                "iter": self.iteration,
                "y_raw": float(y_raw),
                "best_y": float(self.best_y),
                "gamma": float(self.gamma),
                "n_leaves": len(self.leaves),
                "max_depth": max(c.depth for c in self.leaves) if self.leaves else 0,
                "cube_id": cube_id_used,
                "phase": phase,
                "split": split_happened,
                "stagnation": self.stagnation_counter,
            }
            self.debug_trace.append(trace_entry)

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

    def _select_leaf(self) -> Cube:
        if not self.leaves:
            return self.root

        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= self.dim + 2:
                    model_bonus = 0.1
            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        probs = np.exp(scores_arr * 3)
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

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        if self.iteration < 15:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        return self._sample_with_lgs(cube)

    def _local_search_sample(self, progress: float) -> np.ndarray:
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        radius = 0.10 * (1 - progress) + 0.02
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = self.best_x + noise
        return self._clip_to_bounds(x)

    def _should_split(self, cube: Cube) -> bool:
        if cube.n_trials < self._v5s_split_trials_min:
            return False
        if cube.depth >= self._v5s_split_depth_max:
            return False
        return cube.n_trials >= self._v5s_split_trials_factor * self.dim + self._v5s_split_trials_offset

    def _update_all_models(self) -> None:
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim)

    def get_tree_summary(self) -> dict:
        """Ritorna un summary dello stato dell'albero."""
        leaf_stats = []
        for leaf in self.leaves:
            best_in_leaf = min((s for _, s in leaf._tested_pairs), default=np.inf) if not self.maximize else max((s for _, s in leaf._tested_pairs), default=-np.inf)
            # Per minimize, s è già negato, quindi il "best" interno è il max
            # ma y_raw era loss, quindi devo ri-negare
            if leaf._tested_pairs:
                raw_scores = [-s for _, s in leaf._tested_pairs]  # ri-converto a loss
                best_raw = min(raw_scores)
            else:
                best_raw = np.inf
            
            leaf_stats.append({
                "cube_id": leaf.cube_id,
                "depth": leaf.depth,
                "n_trials": leaf.n_trials,
                "good_ratio": leaf.good_ratio(),
                "volume": leaf.volume(),
                "best_loss_in_leaf": best_raw,
                "has_model": leaf.lgs_model is not None,
            })
        
        return {
            "n_leaves": len(self.leaves),
            "max_depth": max(c.depth for c in self.leaves) if self.leaves else 0,
            "total_volume": sum(l.volume() for l in self.leaves),
            "leaves": leaf_stats,
        }

    def analyze_stuck_problem(self) -> dict:
        """Analizza perché potrebbe essere bloccato."""
        analysis = {
            "best_y": self.best_y,
            "stagnation_iters": self.stagnation_counter,
            "last_improvement_at": self.last_improvement_iter,
            "current_iter": self.iteration,
            "gamma": self.gamma,
            "phase": "exploration" if self.iteration < self.exploration_budget else "local_search",
        }
        
        # Analizza distribuzione dei punti nei leaf
        if self.leaves:
            trials_per_leaf = [l.n_trials for l in self.leaves]
            analysis["trials_distribution"] = {
                "min": min(trials_per_leaf),
                "max": max(trials_per_leaf),
                "mean": np.mean(trials_per_leaf),
                "std": np.std(trials_per_leaf),
            }
            
            # Concentrazione: quante foglie hanno >50% dei trial?
            total_trials = sum(trials_per_leaf)
            sorted_trials = sorted(trials_per_leaf, reverse=True)
            cumsum = 0
            n_dominant = 0
            for t in sorted_trials:
                cumsum += t
                n_dominant += 1
                if cumsum >= total_trials * 0.5:
                    break
            analysis["n_leaves_with_50pct_trials"] = n_dominant
            
            # Best loss per foglia
            best_per_leaf = []
            for leaf in self.leaves:
                if leaf._tested_pairs:
                    raw_scores = [-s for _, s in leaf._tested_pairs]
                    best_per_leaf.append(min(raw_scores))
            if best_per_leaf:
                analysis["best_loss_across_leaves"] = {
                    "global_best": min(best_per_leaf),
                    "worst_leaf_best": max(best_per_leaf),
                    "mean_leaf_best": np.mean(best_per_leaf),
                }
        
        return analysis


def _enable_fast_hpobench_tabular_lookup(benchmark: TabularBenchmark) -> None:
    import pandas as pd
    df = benchmark.table
    if not hasattr(df, "columns") or "result" not in df.columns:
        return
    idx_cols = [c for c in df.columns if c != "result"]
    indexed = df.set_index(idx_cols, drop=False)
    try:
        indexed = indexed.sort_index()
    except Exception:
        pass
    cache: dict = {}
    max_cache = 200_000

    def _search_dataframe_fast(self, row_dict, _df_unused):
        key = tuple(row_dict[c] for c in idx_cols)
        hit = cache.get(key)
        if hit is not None:
            return hit
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise AssertionError(f"Multiple matches for {row_dict}")
            row = row.iloc[0]
        res = row["result"]
        if len(cache) < max_cache:
            cache[key] = res
        return res

    benchmark._search_dataframe = types.MethodType(_search_dataframe_fast, benchmark)


def run_debug_optimization(benchmark, cs, budget, seed, verbose=True):
    """Esegue HPO_v5s con debug logging."""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    
    optimizer = HPOptimizerV5sDebug(
        bounds=[(0, 1)] * dim,
        seed=seed,
        total_budget=budget,
        maximize=False,
        debug_log=True,
    )
    
    max_fidelity = benchmark.get_max_fidelity()
    
    config_history = []  # Salva le config testate
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        config_history.append({"config": config.copy(), "loss": y, "x": x.tolist()})
        
        return y
    
    # Run optimization con stampe intermedie
    print_every = 50
    for i in range(budget):
        x = optimizer.ask()
        y = objective(x)
        optimizer.tell(x, y)
        
        if verbose and (i + 1) % print_every == 0:
            analysis = optimizer.analyze_stuck_problem()
            print(f"[Iter {i+1:4d}] best={optimizer.best_y:.6f} | "
                  f"stag={analysis['stagnation_iters']:3d} | "
                  f"leaves={analysis.get('trials_distribution', {}).get('mean', 0):.1f}±{analysis.get('trials_distribution', {}).get('std', 0):.1f} in {len(optimizer.leaves)} leaves | "
                  f"gamma={optimizer.gamma:.4f} | phase={analysis['phase']}")
    
    return optimizer, config_history


def main():
    parser = argparse.ArgumentParser(description='Debug HPO_v5s per capire problemi di convergenza')
    parser.add_argument('--budget', type=int, default=600, help='Budget')
    parser.add_argument('--seeds', type=str, default="10,11,12", help='Comma-separated seeds')
    parser.add_argument('--task_id', type=int, default=31, help='Task ID')
    parser.add_argument('--target_loss', type=float, default=0.70, help='Se best > target_loss, considera "stuck"')
    parser.add_argument('--save_trace', action='store_true', help='Salva trace JSON per analisi')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    budget = args.budget
    task_id = args.task_id
    
    print("=" * 80)
    print("DEBUG HPO_v5s - Analisi convergenza")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Budget: {budget}")
    print(f"Task ID: {task_id}")
    print(f"Target loss (stuck if >): {args.target_loss}")
    print()
    
    benchmark = TabularBenchmark(model='nn', task_id=task_id)
    _enable_fast_hpobench_tabular_lookup(benchmark)
    cs = benchmark.get_configuration_space()
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    print(f"Dimensions: {len(hp_names)}")
    print(f"Hyperparameters: {hp_names}")
    print()
    
    results = []
    stuck_cases = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        optimizer, config_history = run_debug_optimization(
            benchmark, cs, budget, seed, verbose=True
        )
        
        final_best = optimizer.best_y
        results.append(final_best)
        
        # Analisi finale
        print(f"\n--- Analisi finale seed {seed} ---")
        analysis = optimizer.analyze_stuck_problem()
        tree_summary = optimizer.get_tree_summary()
        
        print(f"Best loss: {final_best:.6f}")
        print(f"Stagnation iters: {analysis['stagnation_iters']}")
        print(f"Last improvement at iter: {analysis['last_improvement_at']}")
        print(f"N leaves: {tree_summary['n_leaves']}, max depth: {tree_summary['max_depth']}")
        
        if 'best_loss_across_leaves' in analysis:
            bla = analysis['best_loss_across_leaves']
            print(f"Best loss in leaves: global={bla['global_best']:.6f}, worst_leaf={bla['worst_leaf_best']:.6f}, mean={bla['mean_leaf_best']:.6f}")
        
        # Check if stuck
        if final_best > args.target_loss:
            print(f"\n*** STUCK DETECTED: {final_best:.6f} > {args.target_loss} ***")
            stuck_cases.append({
                "seed": seed,
                "final_best": final_best,
                "analysis": analysis,
                "tree_summary": tree_summary,
                "trace": optimizer.debug_trace,
                "split_events": optimizer.split_events,
                "config_history": config_history,
            })
            
            # Diagnosi dettagliata
            print("\nDiagnosi dettagliata:")
            
            # 1. Quando ha smesso di migliorare?
            print(f"  - Smesso di migliorare a iter {analysis['last_improvement_at']} su {budget}")
            
            # 2. In che fase era?
            exploration_budget = int(budget * 0.70)
            if analysis['last_improvement_at'] < exploration_budget:
                print(f"  - Si è bloccato durante EXPLORATION (prima di iter {exploration_budget})")
            else:
                print(f"  - Si è bloccato durante LOCAL SEARCH")
            
            # 3. Distribuzione trial nelle foglie
            if 'trials_distribution' in analysis:
                td = analysis['trials_distribution']
                print(f"  - Trial per foglia: {td['mean']:.1f} ± {td['std']:.1f} (min={td['min']}, max={td['max']})")
                print(f"  - {analysis['n_leaves_with_50pct_trials']} foglie contengono 50% dei trial")
            
            # 4. Split analysis
            if optimizer.split_events:
                print(f"  - {len(optimizer.split_events)} split totali")
                last_split = optimizer.split_events[-1] if optimizer.split_events else None
                if last_split:
                    print(f"  - Ultimo split a iter {last_split['iter']}, depth {last_split['parent_depth']}")
            
            # 5. Dove sta il best?
            best_leaf = None
            for leaf in optimizer.leaves:
                if leaf._tested_pairs:
                    raw_scores = [-s for _, s in leaf._tested_pairs]
                    if min(raw_scores) == final_best:
                        best_leaf = leaf
                        break
            if best_leaf:
                print(f"  - Best trovato in leaf id={best_leaf.cube_id}, depth={best_leaf.depth}, n_trials={best_leaf.n_trials}")
                print(f"    Bounds: {best_leaf.bounds}")
        
        # Save trace if requested
        if args.save_trace:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = f"/mnt/workspace/thesis/tests/debug_trace_seed{seed}_{timestamp}.json"
            with open(trace_path, 'w') as f:
                json.dump({
                    "seed": seed,
                    "final_best": final_best,
                    "analysis": analysis,
                    "tree_summary": {k: v for k, v in tree_summary.items() if k != 'leaves'},
                    "trace": optimizer.debug_trace,
                    "split_events": optimizer.split_events,
                }, f, indent=2, default=str)
            print(f"Trace salvata in: {trace_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Results: {[f'{r:.6f}' for r in results]}")
    print(f"Mean: {np.mean(results):.6f} ± {np.std(results):.6f}")
    print(f"Stuck cases (>{args.target_loss}): {len(stuck_cases)}/{len(seeds)}")
    
    if stuck_cases:
        print("\nStuck seeds:", [sc["seed"] for sc in stuck_cases])


if __name__ == "__main__":
    main()
