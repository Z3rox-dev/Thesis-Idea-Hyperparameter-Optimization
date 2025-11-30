"""
HPO Density V3: Esplorazione ciclica forzata.

PROBLEMA: Non esploriamo abbastanza i cubi con basso good_ratio,
che contengono l'ottimo!

SOLUZIONE: Ogni N trials, forza esplorazione del cubo meno visitato
invece di selezionare per good_ratio.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from scipy.stats import norm

@dataclass
class Cube:
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    mean_score: float = 0.0
    var_score: float = 0.0
    best_score: float = -np.inf
    M2: float = 0.0
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    surrogate: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    n_good: int = 0
    
    def _widths(self) -> np.ndarray:
        return np.array([abs(hi-lo) for lo,hi in self.bounds], dtype=float)
    
    def center(self) -> np.ndarray:
        return np.array([(lo+hi)/2 for lo,hi in self.bounds], dtype=float)
    
    def volume(self) -> float:
        return float(np.prod(self._widths()))
    
    def contains(self, x: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True
    
    def good_ratio(self) -> float:
        if self.n_trials == 0:
            return 0.5
        return self.n_good / self.n_trials

    def fit_surrogate(self, dim: int) -> None:
        pairs = list(self._tested_pairs)
        n_lin, n_quad = dim+2, 2*dim+2
        
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
        mode = 'quad' if len(pairs) >= n_quad else 'lin'
        
        if mode == 'lin':
            Phi = np.hstack([np.ones((len(y),1)), T])
            n_p = dim+1
        else:
            Phi = np.hstack([np.ones((len(y),1)), T, T**2])
            n_p = 2*dim+1
        
        A = Phi.T @ Phi + alpha * np.eye(n_p)
        
        try:
            w = np.linalg.solve(A, Phi.T @ y)
        except:
            self.surrogate = None
            return
        
        y_hat = Phi @ w
        resid = y - y_hat
        var_y = float(np.var(y)) if len(y)>1 else 1.0
        r2 = 1.0 - float(np.var(resid))/max(var_y,1e-12) if var_y>0 else 0.0
        sigma2 = float(np.sum(resid**2)/max(1,len(y)-n_p))
        lam = 2.0*w[dim+1:] if mode=='quad' else np.zeros(dim)
        
        try:
            A_inv = np.linalg.inv(A)
        except:
            A_inv = None
        
        self.surrogate = {
            'w': w, 'center': center, 'widths': widths,
            'sigma2': sigma2, 'r2': r2, 'lam': lam, 
            'A_inv': A_inv, 'mode': mode, 'n': len(pairs)
        }

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        if self.surrogate is None:
            return 0.0, 1.0
        d = len(x)
        center = self.surrogate['center']
        widths = self.surrogate['widths']
        t = (x - center) / widths
        mode = self.surrogate['mode']
        w = self.surrogate['w']
        if mode == 'lin':
            Phi = np.concatenate([[1.0], t])
        else:
            Phi = np.concatenate([[1.0], t, t**2])
        y_hat = float(w @ Phi)
        sigma2 = self.surrogate['sigma2']
        A_inv = self.surrogate.get('A_inv')
        if A_inv is None:
            return y_hat, math.sqrt(max(sigma2, 1e-12))
        v = max(0.0, float(Phi @ A_inv @ Phi))
        return y_hat, math.sqrt(max(sigma2*(1+v), 1e-12))

    def curvature_scores(self) -> Optional[np.ndarray]:
        if self.surrogate is None or self.surrogate.get('r2', 0) < 0.05:
            return None
        lam = np.abs(self.surrogate['lam'])
        widths = self._widths()
        return (lam**2) * (widths**4)

    def get_split_axis(self) -> int:
        S = self.curvature_scores()
        widths = self._widths()
        if S is not None:
            return int(np.argmax(S))
        else:
            return int(np.argmax(widths))

    def get_split_point(self, axis: int, gamma: float) -> float:
        lo, hi = self.bounds[axis]
        if not self._tested_pairs:
            return (lo + hi) / 2
        good_pts = [p[axis] for p, s in self._tested_pairs if s >= gamma]
        if len(good_pts) >= 3:
            median = float(np.median(good_pts))
            margin = 0.1 * (hi - lo)
            return float(np.clip(median, lo + margin, hi - margin))
        if self.surrogate is not None:
            w = self.surrogate['w']
            d = len(self.bounds)
            mode = self.surrogate['mode']
            if mode == 'quad' and len(w) > d + 1 + axis:
                a = w[d + 1 + axis]
                b = w[1 + axis]
                if abs(a) > 1e-8:
                    center = self.surrogate['center'][axis]
                    width = self.surrogate['widths'][axis]
                    t_opt = -b / (2*a)
                    x_opt = center + t_opt * width
                    margin = 0.15 * (hi - lo)
                    return float(np.clip(x_opt, lo + margin, hi - margin))
        return (lo + hi) / 2

    def split(self, gamma: float) -> List["Cube"]:
        axis = self.get_split_axis()
        cut = self.get_split_point(axis, gamma)
        bounds_lo = list(self.bounds)
        bounds_hi = list(self.bounds)
        bounds_lo[axis] = (self.bounds[axis][0], cut)
        bounds_hi[axis] = (cut, self.bounds[axis][1])
        child_lo = Cube(bounds=bounds_lo, parent=self)
        child_hi = Cube(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1
        for pt, sc in self._tested_pairs:
            if pt[axis] < cut:
                child_lo._tested_pairs.append((pt.copy(), sc))
            else:
                child_hi._tested_pairs.append((pt.copy(), sc))
        for ch in [child_lo, child_hi]:
            if ch._tested_pairs:
                for pt, sc in ch._tested_pairs:
                    ch.n_trials += 1
                    n = ch.n_trials
                    if n == 1:
                        ch.mean_score = sc
                        ch.M2 = 0.0
                    else:
                        delta = sc - ch.mean_score
                        ch.mean_score += delta/n
                        ch.M2 += delta*(sc - ch.mean_score)
                    ch.var_score = ch.M2/(n-1) if n>1 else 0.0
                    if sc > ch.best_score:
                        ch.best_score = sc
                    if sc >= gamma:
                        ch.n_good += 1
                ch.fit_surrogate(len(self.bounds))
        return [child_lo, child_hi]


class HPOptimizer:
    def __init__(self, bounds, maximize=False, seed=42, gamma_quantile=0.25,
                 explore_every=5):  # NUOVO: ogni N trials, esplora
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.explore_every = explore_every
        
        self.root = Cube(bounds=list(bounds))
        self.leaves = [self.root]
        
        self.X_all = []
        self.y_all = []
        self.best_y = -np.inf if maximize else np.inf
        self.best_x = None
        self.total_trials = 0
        self.gamma = 0.0
    
    def _update_gamma(self):
        if len(self.y_all) < 10:
            self.gamma = -np.inf
        else:
            self.gamma = float(np.percentile(self.y_all, 100 * (1 - self.gamma_quantile)))
    
    def _update_good_counts(self):
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
    
    def _select_leaf(self):
        """Selezione con esplorazione ciclica forzata."""
        if not self.leaves:
            return self.root
        
        # ESPLORAZIONE FORZATA: ogni N trials, scegli cubo meno visitato
        if self.total_trials > 0 and self.total_trials % self.explore_every == 0:
            # Scegli cubo con meno trials (proporzionale al volume)
            scores = []
            for c in self.leaves:
                density = c.n_trials / max(c.volume(), 1e-9)
                scores.append(-density)  # Negativo = preferisci bassa densità
            return self.leaves[int(np.argmax(scores))]
        
        # SELEZIONE NORMALE: good_ratio + exploration
        scores = []
        for c in self.leaves:
            n = c.n_trials
            vol = c.volume()
            total_vol = float(np.prod([hi-lo for lo,hi in self.bounds]))
            vol_ratio = vol / max(total_vol, 1e-9)
            
            if n == 0:
                exploration = 2.0
            else:
                exploration = math.sqrt(2 * math.log(self.total_trials + 1) / n)
            
            good_ratio = c.good_ratio()
            score = good_ratio + 0.5 * exploration + 0.1 * vol_ratio
            scores.append(score)
        
        return self.leaves[int(np.argmax(scores))]
    
    def _sample_in_cube(self, cube):
        if cube.n_good >= 3:
            good_pts = [p for p, s in cube._tested_pairs if s >= self.gamma]
            ref = good_pts[self.rng.integers(len(good_pts))]
            widths = cube._widths()
            noise = self.rng.normal(0, 0.15, self.dim) * widths
            x = ref + noise
            x = np.array([
                np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1])
                for i in range(self.dim)
            ])
            return x
        
        if cube.surrogate is not None and cube.n_trials > 3:
            return self._ei_sample(cube)
        
        return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
    
    def _ei_sample(self, cube, n_candidates=50):
        best_ei = -np.inf
        best_x = None
        for _ in range(n_candidates):
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            mu, sigma = cube.predict(x)
            if sigma < 1e-9:
                ei = 0.0
            else:
                best_so_far = max(self.y_all) if self.y_all else -np.inf
                z = (mu - best_so_far) / sigma
                ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
            if ei > best_ei:
                best_ei = ei
                best_x = x
        return best_x if best_x is not None else np.array([
            self.rng.uniform(lo, hi) for lo, hi in cube.bounds
        ])
    
    def _should_split(self, cube):
        if cube.n_trials < 10:
            return False
        if cube.depth >= 5:
            return False
        min_points = 2 * self.dim + 4
        return cube.n_trials >= min_points
    
    def optimize(self, objective, budget=100):
        for trial in range(budget):
            self.total_trials = trial + 1
            self._update_gamma()
            self._update_good_counts()
            
            cube = self._select_leaf()
            x = self._sample_in_cube(cube)
            
            y_raw = objective(x)
            y = y_raw if self.maximize else -y_raw
            
            if self.maximize:
                if y_raw > self.best_y:
                    self.best_y = y_raw
                    self.best_x = x.copy()
            else:
                if y_raw < self.best_y:
                    self.best_y = y_raw
                    self.best_x = x.copy()
            
            self.X_all.append(x.copy())
            self.y_all.append(y)
            
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            n = cube.n_trials
            if n == 1:
                cube.mean_score = y
                cube.M2 = 0.0
            else:
                delta = y - cube.mean_score
                cube.mean_score += delta/n
                cube.M2 += delta*(y - cube.mean_score)
            cube.var_score = cube.M2/(n-1) if n>1 else 0.0
            if y > cube.best_score:
                cube.best_score = y
            if y >= self.gamma:
                cube.n_good += 1
            
            cube.fit_surrogate(self.dim)
            
            if self._should_split(cube):
                children = cube.split(self.gamma)
                self.leaves.remove(cube)
                self.leaves.extend(children)
        
        return self.best_x, self.best_y


if __name__ == "__main__":
    def sphere(x):
        return np.sum((x - 0.7)**2)
    
    opt = HPOptimizer([(0, 1), (0, 1)], maximize=False, seed=42)
    best_x, best_y = opt.optimize(sphere, budget=50)
    print(f"Best: x={best_x}, y={best_y:.6f}")
