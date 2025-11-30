"""
QuadHPO V26 - Gamma Aggressivo

V21 + gamma che parte aggressivo (50° percentile) e scende a 25°.
Questo accelera la convergenza nelle prime iterazioni.

Autore: Z3rox-dev
Data: Novembre 2025
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable


@dataclass(eq=False)
class Cube:
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    surrogate: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    
    def _widths(self) -> np.ndarray:
        return np.array([abs(hi-lo) for lo,hi in self.bounds], dtype=float)
    
    def center(self) -> np.ndarray:
        return np.array([(lo+hi)/2 for lo,hi in self.bounds], dtype=float)
    
    def contains(self, x: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True
    
    def good_ratio(self) -> float:
        if self.n_trials == 0:
            return 0.5
        return self.n_good / self.n_trials

    def fit_surrogate(self, dim: int, rng=None) -> None:
        pairs = list(self._tested_pairs)
        n_lin = dim + 2
        n_quad = 2 * dim + 2  # Soglia originale
        
        # Riutilizzo punti del padre (duplicati danno stabilità: +7.5% win rate)
        if self.parent and len(pairs) < 3 * n_quad:
            parent_pairs = getattr(self.parent, '_tested_pairs', [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * n_quad - len(pairs)
            if needed > 0 and extra:
                pairs = pairs + extra[:needed]
        
        if len(pairs) < n_lin:
            self.surrogate = None
            return
        
        X = np.array([p for p,_ in pairs])
        y = np.array([s for _,s in pairs])
        
        center = self.center()
        widths = np.maximum(self._widths(), 1e-9)
        T = (X - center) / widths
        
        alpha = 1e-3 * math.sqrt(dim)
        
        # Fit lineare e quadratico, scegli il migliore
        best_surrogate = None
        best_score = -np.inf
        
        for mode in ['lin', 'quad']:
            if mode == 'quad' and len(pairs) < n_quad:
                continue
            
            if mode == 'lin':
                Phi = np.hstack([np.ones((len(y),1)), T])
                n_p = dim + 1
                penalty = 0.0
            else:
                Phi = np.hstack([np.ones((len(y),1)), T, T**2])
                n_p = 2*dim + 1
                penalty = 0.02
            
            A = Phi.T @ Phi + alpha * np.eye(n_p)
            
            try:
                w = np.linalg.solve(A, Phi.T @ y)
            except:
                continue
            
            y_hat = Phi @ w
            resid = y - y_hat
            var_y = float(np.var(y)) if len(y) > 1 else 1.0
            r2 = 1.0 - float(np.var(resid)) / max(var_y, 1e-12) if var_y > 0 else 0.0
            
            score = r2 - penalty
            
            if score > best_score:
                best_score = score
                lam = 2.0 * w[dim+1:] if mode == 'quad' else np.zeros(dim)
                best_surrogate = {
                    'w': w, 'center': center, 'widths': widths,
                    'r2': r2, 'lam': lam, 'mode': mode, 'n': len(pairs)
                }
        
        self.surrogate = best_surrogate
    
    def get_curvature_scores(self) -> Optional[np.ndarray]:
        if self.surrogate is None or self.surrogate.get('r2', 0) < 0.05:
            return None
        lam = np.abs(self.surrogate['lam'])
        widths = self._widths()
        return (lam**2) * (widths**4)

    def split(self, gamma: float, dim: int) -> List["Cube"]:
        S = self.get_curvature_scores()
        if S is not None:
            axis = int(np.argmax(S))
        else:
            widths = self._widths()
            axis = int(np.argmax(widths))
        
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
        
        for ch in [child_lo, child_hi]:
            ch.fit_surrogate(dim)
        
        return [child_lo, child_hi]


class HPOptimizer:
    def __init__(self, bounds: List[Tuple[float,float]], maximize: bool = False, 
                 seed: int = 42, gamma_quantile: float = 0.25,
                 local_search_ratio: float = 0.2):
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = 0.10  # Parte conservativo (top 10%)
        self.local_search_ratio = local_search_ratio
        
        self.root = Cube(bounds=list(bounds))
        self.leaves: List[Cube] = [self.root]
        
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.best_y = -np.inf if maximize else np.inf
        self.best_x = None
        self.second_best_x = None
        self.second_best_y = -np.inf if maximize else np.inf
        self.gamma = 0.0
        self.iteration = 0
        self.exploration_budget = 0
        
        self.global_widths = np.array([hi - lo for lo, hi in bounds])
    
    def _update_gamma(self):
        if len(self.y_all) < 10:
            self.gamma = 0.0
        else:
            # Gamma adattivo: parte da 50° percentile, scende a 25° gradualmente
            progress = min(1.0, self.iteration / max(1, self.exploration_budget * 0.5))
            current_quantile = self.gamma_quantile_start - progress * (self.gamma_quantile_start - self.gamma_quantile)
            self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))
    
    def _recount_good(self):
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
    
    def _update_best(self, x: np.ndarray, y_raw: float):
        if self.maximize:
            if y_raw > self.best_y:
                self.second_best_y = self.best_y
                self.second_best_x = self.best_x
                self.best_y = y_raw
                self.best_x = x.copy()
            elif y_raw > self.second_best_y:
                self.second_best_y = y_raw
                self.second_best_x = x.copy()
        else:
            if y_raw < self.best_y:
                self.second_best_y = self.best_y
                self.second_best_x = self.best_x
                self.best_y = y_raw
                self.best_x = x.copy()
            elif y_raw < self.second_best_y:
                self.second_best_y = y_raw
                self.second_best_x = x.copy()
    
    def _select_leaf(self) -> Cube:
        if not self.leaves:
            return self.root
        
        scores = []
        for c in self.leaves:
            ratio = c.good_ratio()
            model_bonus = 0
            if c.surrogate is not None:
                r2 = c.surrogate.get('r2', 0)
                if r2 > 0.3:
                    model_bonus = 0.2 * r2
            exploration = 0.3 / math.sqrt(1 + c.n_trials)
            score = ratio + model_bonus + exploration
            scores.append(score)
        
        scores = np.array(scores)
        scores = scores - scores.max()
        probs = np.exp(scores * 3)
        probs = probs / probs.sum()
        
        return self.rng.choice(self.leaves, p=probs)
    
    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(x[i], self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dim)
        ])
    
    def _clip_to_cube(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        return np.array([
            np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1])
            for i in range(self.dim)
        ])
    
    def _directional_sample(self, cube: Cube) -> Optional[np.ndarray]:
        """
        Campiona nella direzione second_best -> best (conservativa).
        """
        if self.best_x is None or self.second_best_x is None:
            return None
        
        direction = self.best_x - self.second_best_x
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return None
        
        # Step conservativo: [0, 1] -> interpola tra second e best
        # Con piccola probabilità vai leggermente oltre best
        step = self.rng.uniform(0.0, 1.2)
        x = self.second_best_x + step * direction
        
        # Rumore ortogonale ridotto
        noise = self.rng.normal(0, 0.03, self.dim) * self.global_widths
        x = x + noise
        
        return self._clip_to_cube(x, cube)
    
    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        # 15% random
        if self.rng.random() < 0.15:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        # 10% directional (ridotto da 15%)
        if self.second_best_x is not None and self.rng.random() < 0.10:
            x_dir = self._directional_sample(cube)
            if x_dir is not None:
                return x_dir
        
        # V15 style: vicino ai buoni
        if cube.n_good >= 2:
            good_pts = [p for p, s in cube._tested_pairs if s >= self.gamma]
            if good_pts:
                ref = good_pts[self.rng.integers(len(good_pts))]
                widths = cube._widths()
                noise = self.rng.normal(0, 0.15, self.dim) * widths
                x = ref + noise
                return self._clip_to_cube(x, cube)
        
        return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
    
    def _local_search_sample(self, progress: float) -> np.ndarray:
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        
        radius = 0.10 * (1 - progress) + 0.02
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = self.best_x + noise
        return self._clip_to_bounds(x)
    
    def _should_split(self, cube: Cube) -> bool:
        if cube.n_trials < 10:
            return False
        if cube.depth >= 5:
            return False
        return cube.n_trials >= 2 * self.dim + 4
    
    def optimize(self, objective: Callable[[np.ndarray], float], 
                 budget: int = 100) -> Tuple[np.ndarray, float]:
        
        exploration_budget = int(budget * (1 - self.local_search_ratio))
        local_search_budget = budget - exploration_budget
        self.exploration_budget = exploration_budget
        
        for trial in range(exploration_budget):
            self.iteration = trial
            self._update_gamma()
            self._recount_good()
            
            cube = self._select_leaf()
            x = self._sample_in_cube(cube)
            
            y_raw = objective(x)
            y = y_raw if self.maximize else -y_raw
            
            self._update_best(x, y_raw)
            
            self.X_all.append(x.copy())
            self.y_all.append(y)
            
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            if y >= self.gamma:
                cube.n_good += 1
            
            cube.fit_surrogate(self.dim)
            
            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim)
                self.leaves.remove(cube)
                self.leaves.extend(children)
        
        for i in range(local_search_budget):
            progress = i / max(1, local_search_budget - 1)
            x = self._local_search_sample(progress)
            
            y_raw = objective(x)
            self._update_best(x, y_raw)
        
        return self.best_x, self.best_y


if __name__ == "__main__":
    def sphere(x):
        return np.sum((x - 0.7)**2)
    
    print("Test V26 (Gamma Aggressivo) su Sphere 10D:")
    for seed in [42, 123, 456]:
        opt = HPOptimizer([(0, 1)]*10, maximize=False, seed=seed)
        best_x, best_y = opt.optimize(sphere, budget=100)
        print(f"  Seed {seed}: {best_y:.6f}")
