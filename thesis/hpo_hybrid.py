"""
HPO HYBRID - Full fusion of hpo_minimal and hpo_main
=====================================================

Struttura basata su hpo_minimal (816 righe) con aggiunta della transizione
a Phase 2 (hpo_main style) dopo depth_threshold.

Phase 1 (depth < threshold): 
- PCA + frame rotato (R, μ)
- EI con trust region
- Gradient steps  
- Quad splits
- UCB per selezione
- Backpropagation
- Pruning sofisticato

Phase 2 (depth >= threshold):
- Frame fisso (eredita da padre)
- Dual surrogate lin+quad (from main)
- Sampling basato su good_ratio
- Binary splits
- Gamma quantile (from main)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import math
import numpy as np

# Global debug logger
_DEBUG_LOG: Optional[Callable[[str], None]] = None

def _log(msg: str) -> None:
    if _DEBUG_LOG is not None:
        _DEBUG_LOG(msg)


@dataclass
class Cube:
    """Cube with full minimal features + phase tracking"""
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    children: List["Cube"] = field(default_factory=list)
    R: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    depth: int = 0
    n_trials: int = 0
    mean_score: float = 0.0
    var_score: float = 0.0
    best_score: float = -np.inf
    M2: float = 0.0
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    surrogate: Optional[dict] = field(default=None, init=False)
    
    # Phase 2 additions
    phase: int = 1  # 1 = minimal style, 2 = main style
    n_good: int = 0  # Count of good points (for Phase 2)

    def _ensure_frame(self) -> None:
        d = len(self.bounds)
        if self.R is None:
            self.R = np.eye(d)
        if self.mu is None:
            if self.parent is not None and self.parent.mu is not None:
                ctr = np.array([(lo+hi)*0.5 for lo,hi in self.bounds])
                self.mu = (self.parent.mu + self.parent.R @ ctr).astype(float)
                self.R = self.parent.R.copy()
            else:
                self.mu = np.full(d, 0.5, dtype=float)

    def to_prime(self, x: np.ndarray) -> np.ndarray:
        self._ensure_frame()
        return (self.R.T @ (x - self.mu)).astype(float)

    def to_original(self, x_prime: np.ndarray) -> np.ndarray:
        self._ensure_frame()
        return (self.mu + self.R @ x_prime).astype(float)

    def _widths(self) -> np.ndarray:
        return np.array([abs(hi-lo) for lo,hi in self.bounds], dtype=float)

    def fit_surrogate(self, dim: int) -> None:
        """Fit surrogate model - works for both phases"""
        pairs = list(self._tested_pairs)
        n_lin, n_quad = dim+2, 2*dim+2
        
        # Data Borrowing from parent
        if self.parent and len(pairs) < 3 * n_quad:
            parent_pairs = getattr(self.parent, '_tested_pairs', [])
            extra = []
            existing_pts = np.array([p for p,_ in pairs]) if pairs else np.empty((0, dim))
            
            for pp in parent_pairs:
                p_loc = pp[0]
                p_prime = self.to_prime(p_loc)
                
                inside = True
                for i, (lo, hi) in enumerate(self.bounds):
                    margin = (hi - lo) * 0.1
                    if not (lo - margin <= p_prime[i] <= hi + margin):
                        inside = False
                        break
                if not inside:
                    continue
                
                if existing_pts.size > 0:
                    dists = np.linalg.norm(existing_pts - p_loc, axis=1)
                    if np.min(dists) < 1e-9:
                        continue
                
                extra.append(pp)
            
            needed = 3 * n_quad - len(pairs)
            if needed > 0 and extra:
                pairs = pairs + extra[:needed]
        
        if len(pairs) < n_lin:
            self.surrogate = None
            return
            
        d = len(self.bounds)
        R, mu, evals, pca_ok = self._pca()
        
        X = np.array([p for p,_ in pairs])
        y = np.array([s for _,s in pairs])
        T = (R.T @ (X.T - mu.reshape(-1,1))).T
        
        widths = np.array([abs(hi-lo) for lo,hi in self.bounds])
        t_std_floor = 0.1 * widths
        t_std = np.maximum(np.std(T, axis=0), t_std_floor)
        Ts = T / t_std
        alpha = 1e-3 * math.sqrt(d)
        mode = 'quad' if len(pairs) >= n_quad else 'lin'
        
        if mode == 'lin':
            Phi = np.hstack([np.ones((len(y),1)), Ts])
            n_p = d+1
        else:
            Phi = np.hstack([np.ones((len(y),1)), Ts, Ts**2])
            n_p = 2*d+1
            
        A = Phi.T @ Phi + alpha * np.eye(n_p)
        
        # Adaptive regularization
        max_cond = 1e8
        for attempt in range(3):
            try:
                cond_A = float(np.linalg.cond(A))
            except:
                cond_A = float('inf')
            
            if cond_A <= max_cond:
                break
            
            boost = cond_A / max_cond
            alpha_new = alpha * boost
            A = Phi.T @ Phi + alpha_new * np.eye(n_p)
        
        try:
            w_s = np.linalg.solve(A, Phi.T @ y)
        except:
            self.surrogate = None
            return
            
        w = np.zeros(2*d+1)
        w[0] = w_s[0]
        w[1:d+1] = w_s[1:d+1] / t_std
        if mode == 'quad':
            w[d+1:] = w_s[d+1:] / (t_std**2)
            
        y_hat = Phi @ w_s
        resid = y - y_hat
        var_y = float(np.var(y)) if len(y)>1 else 1.0
        r2 = 1.0 - float(np.var(resid))/max(var_y,1e-12) if var_y>0 else 0.0
        sigma2 = float(np.sum(resid**2)/max(1,len(y)-n_p))
        lam = 2.0*w[d+1:] if mode=='quad' else np.zeros(d)
        
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            self.surrogate = None
            return
        
        try:
            A_inv = np.linalg.inv(A)
        except:
            A_inv = None
        
        self.surrogate = {
            'w': w, 'R': R, 'mu': mu, 'sigma2': sigma2, 'r2': r2,
            'lam': lam, 'A_inv': A_inv, 't_std': t_std, 'mode': mode, 'n': len(pairs),
            'cond_A': cond_A, 'pca_ok': pca_ok, 'evals': evals,
        }

    def predict(self, x_prime: np.ndarray) -> Tuple[float, float]:
        if self.surrogate is None:
            return 0.0, 1.0
        d = len(x_prime)
        t_std = self.surrogate['t_std']
        xs = x_prime / t_std
        Phi = np.zeros(2*d+1)
        Phi[0] = 1.0
        Phi[1:d+1] = x_prime
        Phi[d+1:] = x_prime**2
        y_hat = float(self.surrogate['w'] @ Phi)
        sigma2 = self.surrogate['sigma2']
        A_inv = self.surrogate.get('A_inv')
        if A_inv is None:
            return y_hat, math.sqrt(max(sigma2, 1e-12))
        if self.surrogate['mode'] == 'lin':
            Phi_s = np.zeros(d+1)
            Phi_s[0] = 1.0
            Phi_s[1:] = xs
        else:
            Phi_s = np.zeros(2*d+1)
            Phi_s[0] = 1.0
            Phi_s[1:d+1] = xs
            Phi_s[d+1:] = xs**2
        v = max(0.0, float(Phi_s @ A_inv @ Phi_s))
        return y_hat, math.sqrt(max(sigma2*(1+v), 1e-12))

    def _pca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        d = len(self.bounds)
        self._ensure_frame()
        if len(self._tested_pairs) < d+2:
            return self.R.copy(), self.mu.copy(), np.ones(d), False
            
        pts = np.array([p for p,_ in self._tested_pairs])
        scs = np.array([s for _,s in self._tested_pairs])
        s = scs - np.median(scs)
        s_std = max(float(np.std(s)), 1e-12)
        s = s / s_std
        
        if float(np.max(np.abs(s))) > 50:
            s = np.clip(s, -50, 50)
        
        w = np.exp(s)
        if np.any(np.isinf(w)) or np.any(np.isnan(w)):
            w = np.ones(len(w))
        
        w = w / np.sum(w)
        w = 0.75*w + 0.25/len(w)
        w = w / np.sum(w)
        mu = np.sum(w[:,None]*pts, axis=0)
        Z = pts - mu
        C = (Z*w[:,None]).T @ Z + 1e-9*np.eye(d)
        
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:,order]
        ratio = evals[0] / max(np.mean(evals[1:]) if d>1 else evals[0], 1e-9)
        ok = ratio >= 1.4
        
        if not ok:
            return self.R.copy(), self.mu.copy(), np.maximum(evals, 1e-9), False
        return evecs, mu, np.maximum(evals, 1e-9), True

    def curvature_scores(self) -> Optional[np.ndarray]:
        s = self.surrogate
        if s is None or s.get('r2', 0) < 0.05:
            return None
        lam = np.abs(s['lam'])
        self._ensure_frame()
        M = s['R'].T @ self.R
        w = self._widths()
        h = np.abs(M) @ w
        return (lam**2) * (h**4)

    def ucb(self, beta: float, global_best: float = -np.inf) -> float:
        if self.n_trials > 0:
            mu = self.best_score if self.best_score > -np.inf else self.mean_score
            var = self.var_score if self.n_trials > 1 else 1.0
        else:
            mu, var = 0.0, 1.0
            
        if self.parent and self.parent.n_trials > 0:
            pw = 1.0/(1.0+self.n_trials)
            parent_best = self.parent.best_score if self.parent.best_score > -np.inf else self.parent.mean_score
            mu = pw*parent_best + (1-pw)*mu
            var = pw*(self.parent.var_score if self.parent.var_score>0 else 1.0) + (1-pw)*var
        
        decay = 1.0/(1.0+0.1*self.depth)
        
        if self.n_trials <= 0:
            explore = beta*decay*math.sqrt(var+1.0)
        else:
            explore = beta*decay*math.sqrt(var/(self.n_trials+1e-8)+1.0)
        
        result = float(mu + explore)
        
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result

    def density_score(self, gamma_threshold: float) -> float:
        """Phase 2 density-based score (from hpo_main)"""
        if self.n_trials == 0:
            return 0.0
        return self.n_good / (self.n_trials + 1)

    def surrogate_sanity(self) -> dict:
        """Check health of surrogate model and PCA
        
        Returns dict with:
        - healthy: bool - overall health
        - r2: float - R² fit quality
        - cond_A: float - condition number of design matrix
        - pca_ok: bool - whether PCA found good directions  
        - pca_ratio: float - ratio of first/mean eigenvalue
        - warnings: list of issues found
        """
        s = self.surrogate
        warnings = []
        
        if s is None:
            return {'healthy': False, 'r2': 0, 'cond_A': float('inf'), 
                    'pca_ok': False, 'pca_ratio': 1.0, 'warnings': ['no_surrogate']}
        
        r2 = s.get('r2', 0)
        cond_A = s.get('cond_A', float('inf'))
        pca_ok = s.get('pca_ok', False)
        evals = s.get('evals', np.array([1.0]))
        
        # Calculate PCA eigenvalue ratio
        if len(evals) > 1 and np.mean(evals[1:]) > 1e-12:
            pca_ratio = evals[0] / np.mean(evals[1:])
        else:
            pca_ratio = 1.0
        
        # Check various health indicators
        if r2 < 0.1:
            warnings.append(f'low_r2={r2:.3f}')
        if cond_A > 1e6:
            warnings.append(f'high_cond={cond_A:.1e}')
        if not pca_ok:
            warnings.append('pca_failed')
        if pca_ratio < 1.5:
            warnings.append(f'low_pca_ratio={pca_ratio:.2f}')
        if pca_ratio > 1e6:
            warnings.append(f'extreme_pca_ratio={pca_ratio:.1e}')
            
        # Check for NaN/Inf in weights
        w = s.get('w', np.array([]))
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            warnings.append('bad_weights')
            
        # Check eigenvalues spread
        if len(evals) > 1:
            eval_spread = evals[0] / (evals[-1] + 1e-12)
            if eval_spread > 1e8:
                warnings.append(f'extreme_eval_spread={eval_spread:.1e}')
        
        healthy = len(warnings) == 0 or (len(warnings) == 1 and 'pca_failed' in warnings and r2 > 0.3)
        
        return {
            'healthy': healthy,
            'r2': r2,
            'cond_A': cond_A,
            'pca_ok': pca_ok,
            'pca_ratio': pca_ratio,
            'warnings': warnings
        }

    def get_debug_info(self, beta: float, gamma_threshold: float) -> dict:
        """Return all debug metrics for this cube"""
        widths = self._widths()
        volume = float(np.prod(widths))
        
        return {
            'depth': self.depth,
            'phase': self.phase,
            'n_trials': self.n_trials,
            'n_good': self.n_good,
            'mean_score': self.mean_score,
            'var_score': self.var_score,
            'best_score': self.best_score,
            'ucb': self.ucb(beta, self.best_score),
            'density': self.density_score(gamma_threshold),
            'volume': volume,
            'widths': widths.tolist(),
            'has_surrogate': self.surrogate is not None,
            'surrogate_r2': self.surrogate.get('r2', 0) if self.surrogate else 0,
            'surrogate_n': self.surrogate.get('n', 0) if self.surrogate else 0,
            'surrogate_cond': self.surrogate.get('cond_A', float('inf')) if self.surrogate else float('inf'),
            'pca_ok': self.surrogate.get('pca_ok', False) if self.surrogate else False,
            'sanity': self.surrogate_sanity(),
        }

    def update(self, score: float, x: np.ndarray, dim: int, gamma_threshold: float = None) -> None:
        self.n_trials += 1
        self._tested_pairs.append((x.copy(), float(score)))
        n = self.n_trials
        
        if n == 1:
            self.mean_score = score
            self.M2 = 0.0
        else:
            delta = score - self.mean_score
            self.mean_score += delta/n
            self.M2 += delta*(score - self.mean_score)
        self.var_score = self.M2/(n-1) if n>1 else 0.0
        
        if score > self.best_score:
            self.best_score = score
            
        # Phase 2: track good points
        if gamma_threshold is not None and score >= gamma_threshold:
            self.n_good += 1
            
        self.fit_surrogate(dim)

    def should_split(self, dim: int, max_depth: int, remaining: int, global_best: float = -np.inf) -> str:
        min_split = 2*dim + 2
        if self.depth >= max_depth:
            return 'none'
        if all(abs(hi-lo) < 1e-6 for lo,hi in self.bounds):
            return 'none'
        if remaining < dim+2:
            return 'none'
        if self.n_trials < min_split:
            return 'none'
        if global_best > -1e12 and self.depth > 0:
            margin = max(abs(global_best), 1.0) * 0.5
            if self.best_score < global_best - margin:
                return 'none'
                
        # Phase 2: only binary splits
        if self.phase == 2:
            return 'binary'
            
        if self.n_trials >= 3*min_split:
            return 'quad' if len(self.bounds) > 1 else 'binary'
            
        S = self.curvature_scores()
        if S is not None and float(np.max(S)) < 1e-8:
            if self.surrogate and np.linalg.norm(self.surrogate['w'][1:dim+1]) < 1e-6:
                return 'none'
                
        if self.surrogate and self.surrogate['n'] >= min_split:
            var_p = self.surrogate['sigma2']
            if var_p < 1e-9:
                return 'none'
            ch = self._sim_split(dim)
            if ch:
                n_tot = sum(c['n'] for c in ch)
                if n_tot > 0:
                    var_post = sum((c['n']/n_tot)*c['var'] for c in ch)
                    gamma = 0.005 * var_p
                    if var_p - var_post < gamma:
                        return 'none'
                        
        return 'quad' if len(self.bounds) > 1 else 'binary'

    def _sim_split(self, dim: int) -> List[dict]:
        if not self._tested_pairs:
            return []
        R, mu, _, ok = self._pca()
        R_use = R if ok else self.R
        mu_use = mu if ok else self.mu
        X = np.array([p for p,_ in self._tested_pairs])
        y = np.array([s for _,s in self._tested_pairs])
        T = (R_use.T @ (X.T - mu_use.reshape(-1,1))).T
        ax = 0
        cut = float(np.median(T[:,ax]))
        left = T[:,ax] < cut
        res = []
        for mask in [left, ~left]:
            idx = np.where(mask)[0]
            res.append({'n': len(idx), 'var': float(np.var(y[idx])) if len(idx)>1 else 1.0})
        return res

    def _quad_cut(self, ax: int, R: np.ndarray, mu: np.ndarray) -> float:
        d = len(self.bounds)
        self._ensure_frame()
        
        widths = self._widths()
        M = R.T @ self.R
        spans = np.abs(M) @ widths
        delta_mu = self.mu - mu
        center_in_R = R.T @ delta_mu
        correct_lo = center_in_R[ax] - spans[ax]/2
        correct_hi = center_in_R[ax] + spans[ax]/2
        
        if len(self._tested_pairs) < 6:
            return 0.5*(correct_lo + correct_hi)
        
        X = np.array([p for p,_ in self._tested_pairs])
        y = np.array([s for _,s in self._tested_pairs])
        T = (R.T @ (X.T - mu.reshape(-1,1))).T
        t = T[:,ax]
        t_std = max(float(np.std(t)), 1e-9)
        ts = t / t_std
        Phi = np.stack([np.ones_like(ts), ts, 0.5*ts*ts], axis=1)
        A = Phi.T @ Phi + 1e-3*np.eye(3)
        try:
            w = np.linalg.solve(A, Phi.T @ y)
            if w[2] > 1e-8:
                t_star = -w[1]/w[2] * t_std
                return float(np.clip(t_star, correct_lo, correct_hi))
        except:
            pass
        return 0.5*(correct_lo + correct_hi)

    def split(self, dim: int, mode: str, depth_threshold: int = 5, gamma_threshold: float = None) -> List["Cube"]:
        d = len(self.bounds)
        self._ensure_frame()
        S = self.curvature_scores()
        R_use, mu_use = self.R, self.mu
        if S is not None and self.surrogate:
            R_use = self.surrogate['R']
            mu_use = self.surrogate['mu']
            
        widths = self._widths()
        if S is not None:
            order = np.argsort(S)[::-1]
            ax_i, ax_j = int(order[0]), int(order[1] if len(order)>1 else order[0])
        else:
            top2 = np.argsort(widths)[-2:]
            ax_i, ax_j = int(top2[-1]), int(top2[-2] if len(top2)>1 else top2[-1])
            
        M = R_use.T @ self.R if self.R is not None else R_use.T
        spans = np.abs(M) @ widths
        delta_mu = (self.mu if self.mu is not None else np.zeros(d)) - mu_use
        center = R_use.T @ delta_mu
        base = [(center[k]-spans[k]/2, center[k]+spans[k]/2) for k in range(d)]
        
        # Ensure base spans have minimum width
        min_span = 1e-4
        for k in range(d):
            lo, hi = base[k]
            if hi - lo < min_span:
                mid = (lo + hi) / 2
                base[k] = (mid - min_span/2, mid + min_span/2)
        
        cut_i = self._quad_cut(ax_i, R_use, mu_use)
        cut_j = self._quad_cut(ax_j, R_use, mu_use) if mode=='quad' else 0.0
        lo_i, hi_i = base[ax_i]
        lo_j, hi_j = base[ax_j] if mode=='quad' else (0,0)
        
        # Ensure cut is not too close to edges (leave at least 10% on each side)
        margin_i = 0.1 * (hi_i - lo_i)
        margin_j = 0.1 * (hi_j - lo_j) if mode == 'quad' else 0
        cut_i = float(np.clip(cut_i, lo_i + margin_i, hi_i - margin_i))
        if mode == 'quad':
            cut_j = float(np.clip(cut_j, lo_j + margin_j, hi_j - margin_j))
            
        if mode == 'binary':
            quads = [(True,), (False,)]
        else:
            quads = [(True,True), (False,True), (True,False), (False,False)]
            
        children = []
        new_depth = self.depth + 1
        
        # DYNAMIC PHASE TRANSITION based on surrogate sanity
        # If parent's surrogate is unhealthy, children should use Phase 2
        parent_sanity = self.surrogate_sanity()
        force_phase2 = False
        
        if not parent_sanity['healthy']:
            force_phase2 = True
            _log(f"  [PHASE] Forcing Phase 2: parent unhealthy, warnings={parent_sanity['warnings']}")
        elif parent_sanity['cond_A'] > 1e6:
            force_phase2 = True
            _log(f"  [PHASE] Forcing Phase 2: parent cond_A={parent_sanity['cond_A']:.1e} > 1e6")
        elif not parent_sanity['pca_ok'] and parent_sanity['r2'] < 0.5:
            force_phase2 = True
            _log(f"  [PHASE] Forcing Phase 2: PCA failed and r2={parent_sanity['r2']:.2f} < 0.5")
        
        child_phase = 2 if (force_phase2 or new_depth >= depth_threshold) else 1
        
        for q in quads:
            nb = list(base)
            nb[ax_i] = (lo_i, cut_i) if q[0] else (cut_i, hi_i)
            if mode == 'quad':
                nb[ax_j] = (lo_j, cut_j) if q[1] else (cut_j, hi_j)
            ctr = np.array([(a+b)*0.5 for a,b in nb])
            wch = np.array([b-a for a,b in nb])
            
            # BUG FIX: Ensure minimum width to prevent degenerate cubes
            min_width = 1e-6
            wch = np.maximum(wch, min_width)
            
            ch = Cube(bounds=[(-w/2, w/2) for w in wch], parent=self)
            ch.R = R_use.copy()
            ch.mu = (mu_use + R_use @ ctr).astype(float)
            ch.depth = new_depth
            ch.phase = child_phase
            children.append(ch)
            
        X = np.array([p for p,_ in self._tested_pairs]) if self._tested_pairs else np.empty((0,d))
        if X.size > 0:
            T = (R_use.T @ (X.T - mu_use.reshape(-1,1))).T
            for idx_pt, (pt, sc) in enumerate(self._tested_pairs):
                t = T[idx_pt]
                if mode == 'binary':
                    ci = 0 if t[ax_i] < cut_i else 1
                else:
                    li = t[ax_i] < cut_i
                    lj = t[ax_j] < cut_j
                    ci = {(True,True):0, (False,True):1, (True,False):2, (False,False):3}[(li,lj)]
                children[ci]._tested_pairs.append((pt.copy(), sc))
                
        for ch in children:
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
                    # CRITICAL: propagate n_good to children
                    if gamma_threshold is not None and sc >= gamma_threshold:
                        ch.n_good += 1
                ch.fit_surrogate(dim)
        
        # Detailed split log
        _log(f"[SPLIT] d={self.depth} {mode} phase{self.phase}->{child_phase} n={len(self._tested_pairs)} -> ch={[len(c._tested_pairs) for c in children]}")
        widths = self._widths()
        _log(f"  Parent bounds widths: {[f'{w:.4f}' for w in widths]}")
        for i, ch in enumerate(children):
            ch_widths = ch._widths()
            _log(f"  Child{i}: n_pts={len(ch._tested_pairs)} n_good={ch.n_good} widths={[f'{w:.4f}' for w in ch_widths]}")
        
        self.children = children
        return children


class HPOptimizer:
    """Hybrid optimizer combining minimal (Phase 1) and main (Phase 2) approaches"""
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        maximize: bool = True,
        seed: Optional[int] = None,
        debug_log: Optional[Callable[[str], None]] = None,
        depth_threshold: int = 5,
        gamma_quantile_start: float = 0.10,
        gamma_quantile_end: float = 0.25,
    ):
        global _DEBUG_LOG
        _DEBUG_LOG = debug_log
        
        self.bounds = list(bounds)
        self.dim = len(bounds)
        self.maximize = maximize
        self.sign = 1.0 if maximize else -1.0
        self.depth_threshold = depth_threshold
        self.gamma_quantile_start = gamma_quantile_start
        self.gamma_quantile_end = gamma_quantile_end
        
        if seed is not None:
            np.random.seed(seed)
            
        lo = np.array([l for l,h in bounds])
        hi = np.array([h for l,h in bounds])
        root_mu = (lo+hi)*0.5
        root_w = hi - lo
        
        self.root = Cube(bounds=[(-w/2, w/2) for w in root_w])
        self.root.R = np.eye(self.dim)
        self.root.mu = root_mu
        self.root.phase = 1
        
        self.leaves: List[Cube] = [self.root]
        self.best_score = -np.inf
        self.best_x: Optional[np.ndarray] = None
        self.trial = 0
        self.budget = 100
        self._preferred: Optional[Cube] = None
        self._all_scores: List[float] = []
        self._last_sample_method: str = 'init'  # Track sampling method
        self._sample_method_counts: dict = {}  # Count sampling methods
        
    def _effective_depth_threshold(self) -> int:
        """Adaptive depth threshold based on budget
        
        REVISED LOGIC: 
        - Low budgets need MORE Phase 1 exploration (higher threshold = stay in P1 longer)
        - High budgets can afford to transition to Phase 2 earlier at deeper levels
        """
        # Base threshold from constructor (default 5)
        base = self.depth_threshold
        
        # With low budget, stay in Phase 1 longer (higher depth threshold)
        if self.budget <= 80:
            return base + 2  # 5+2=7, very deep before Phase 2
        elif self.budget <= 150:
            return base + 1  # 5+1=6
        elif self.budget <= 250:
            return base      # 5
        else:
            return base - 1  # 4, transition earlier for very high budgets

    def _beta(self) -> float:
        return 0.5 * math.sqrt(math.log(max(2, self.trial+1)))

    def _max_depth(self) -> int:
        return int(2*self.dim + 3*math.log2(max(2, self.budget)))

    def _gamma_quantile(self) -> float:
        """Adaptive gamma from main"""
        if len(self._all_scores) < 10:
            return self.gamma_quantile_start
        progress = self.trial / max(1, self.budget)
        return self.gamma_quantile_start + progress * (self.gamma_quantile_end - self.gamma_quantile_start)

    def _gamma_threshold(self) -> float:
        """Compute threshold for 'good' points"""
        if len(self._all_scores) < 5:
            return -np.inf
        q = self._gamma_quantile()
        return float(np.quantile(self._all_scores, 1 - q))  # top q%

    def _in_bounds(self, x: np.ndarray) -> bool:
        for j,(lo,hi) in enumerate(self.bounds):
            if x[j] < lo-1e-9 or x[j] > hi+1e-9:
                return False
        return True

    def _reflect(self, x: np.ndarray) -> np.ndarray:
        """Reflect points that go outside bounds (bounce back)
        
        Instead of clipping (which creates accumulation at borders),
        we reflect the point back into the domain like a ball bouncing off a wall.
        """
        x = x.copy()
        for j, (lo, hi) in enumerate(self.bounds):
            width = hi - lo
            if width < 1e-12:
                x[j] = (lo + hi) / 2
                continue
                
            # Reflect until inside bounds
            max_reflections = 10
            for _ in range(max_reflections):
                if x[j] < lo:
                    x[j] = lo + (lo - x[j])  # Reflect from lower bound
                elif x[j] > hi:
                    x[j] = hi - (x[j] - hi)  # Reflect from upper bound
                else:
                    break
            
            # Final safety clip if still outside after reflections
            x[j] = float(np.clip(x[j], lo, hi))
        return x

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Legacy clip - now uses reflect for better distribution"""
        return self._reflect(x)

    def _trunc_norm(self, mu: float, std: float, lo: float, hi: float) -> float:
        if std <= 1e-12:
            return float(np.clip(mu, lo, hi))
        for _ in range(100):
            v = np.random.normal(mu, std)
            if lo <= v <= hi:
                return float(v)
        return float(np.clip(np.random.normal(mu, std), lo, hi))

    def _sample_phase1(self, cube: Cube) -> np.ndarray:
        """Phase 1 sampling: EI + gradient (from minimal)"""
        d = self.dim
        pairs = cube._tested_pairs
        s = cube.surrogate
        
        # Gradient-based step
        if s and s['n'] >= d+2 and s['r2'] >= 0.5 and np.random.rand() < 0.6:
            w = s['w']
            grad = w[1:d+1]
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm > 0.02:
                widths = cube._widths()
                step_size = 0.10 * widths
                direction = grad / grad_norm
                
                if pairs:
                    best_pt, _ = max(pairs, key=lambda t: t[1])
                    t_center = cube.to_prime(best_pt)
                else:
                    t_center = np.zeros(d)
                
                t_new = t_center + direction * step_size
                
                for j in range(d):
                    lo, hi = cube.bounds[j]
                    t_new[j] = float(np.clip(t_new[j], lo, hi))
                
                x = cube.to_original(t_new)
                if self._in_bounds(x):
                    self._last_sample_method = 'p1_gradient'
                    return self._clip(x)
        
        # Quadratic optimum
        if s and s['n'] >= 2*d+2 and s['r2'] >= 0.85 and s['mode'] == 'quad':
            w = s['w']
            all_neg = all(w[d+1+j] < -1e-9 for j in range(d))
            if all_neg:
                t_opt = np.zeros(d)
                for j in range(d):
                    t_opt[j] = -w[1+j] / (2*w[d+1+j])
                x_opt = s['mu'] + s['R'] @ t_opt
                if self._in_bounds(x_opt):
                    self._last_sample_method = 'p1_quad_opt'
                    return self._clip(x_opt)
        
        # Perturbation around best
        alpha = 0.4 / (1 + 0.05*cube.depth)
        if pairs and np.random.rand() < alpha:
            top_k = min(5, len(pairs))
            best_pts = sorted(pairs, key=lambda t: t[1], reverse=True)[:top_k]
            center, _ = best_pts[np.random.randint(len(best_pts))]
            t_c = cube.to_prime(center)
            t_new = np.zeros(d)
            for j in range(d):
                lo, hi = cube.bounds[j]
                wid = abs(hi-lo)
                t_new[j] = self._trunc_norm(t_c[j], wid*0.1, lo, hi)
            x = cube.to_original(t_new)
            if self._in_bounds(x):
                self._last_sample_method = 'p1_perturb'
                return self._clip(x)
        
        # EI with trust region
        min_pts = d + 2
        if s and s['n'] >= min_pts and s['r2'] >= 0.25:
            R_s, mu_s = s['R'], s['mu']
            cube._ensure_frame()
            T_tr = np.array([(R_s.T @ (p - mu_s)) for p,_ in pairs]) if pairs else np.empty((0,d))
            if T_tr.shape[0] >= d+2:
                C = np.cov(T_tr.T) if T_tr.shape[0] > 1 else np.eye(d)
            else:
                C = np.eye(d)
            C = C + 1e-6*np.eye(d)
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(d)
            
            rho_max2 = 3.0 * d if s['r2'] < 0.6 else 15.0 * d
            local_best = cube.best_score if cube.best_score > -np.inf else 0.0
            M = max(96, 20*d)
            cands = []
            
            if pairs:
                top_k = min(5, len(pairs))
                best_pts = sorted(pairs, key=lambda t: t[1], reverse=True)[:top_k]
                best_ts = [cube.to_prime(p) for p, _ in best_pts]
            else:
                best_ts = [np.zeros(d)]
            
            for _ in range(M):
                if np.random.rand() < 0.7 and best_ts:
                    center = best_ts[np.random.randint(len(best_ts))]
                else:
                    center = np.zeros(d)
                
                u = np.zeros(d)
                for j in range(d):
                    lo, hi = cube.bounds[j]
                    scale = (hi-lo)*0.15 if np.any(center) else (hi-lo)*0.25
                    u[j] = self._trunc_norm(center[j], scale, lo, hi)
                x = cube.to_original(u)
                if not self._in_bounds(x):
                    # Try bounce reflection before discarding
                    x = self._reflect(x)
                    if not self._in_bounds(x):
                        continue
                t = R_s.T @ (x - mu_s)
                rho2 = float(t @ C_inv @ t)
                if rho2 > rho_max2:
                    continue
                y_hat, sigma = cube.predict(t)
                diff = y_hat - local_best
                z = diff / max(sigma, 1e-9)
                
                if z < -30:
                    pdf, cdf = 0.0, 0.0
                elif z > 30:
                    pdf, cdf = 0.0, 1.0
                else:
                    pdf = (1/math.sqrt(2*math.pi)) * math.exp(-0.5*z*z)
                    cdf = 0.5*(1 + math.erf(z/math.sqrt(2)))
                
                ei = diff*cdf + sigma*pdf
                pen = 0.03 if s['r2'] > 0.7 else 0.08
                ei_eff = ei / (1 + pen*rho2)
                cands.append((ei_eff, x))
            
            if cands:
                cands.sort(key=lambda t: t[0], reverse=True)
                self._last_sample_method = 'p1_ei'
                return self._clip(cands[0][1])
        
        # Fallback uniform
        self._last_sample_method = 'p1_uniform'
        return self._sample_uniform(cube)

    def _sample_phase2(self, cube: Cube) -> np.ndarray:
        """Phase 2 sampling: directional + good ratio (from main)"""
        d = self.dim
        pairs = cube._tested_pairs
        
        if len(pairs) < 2:
            return self._sample_uniform(cube)
        
        # Get good ratio
        gamma_thresh = self._gamma_threshold()
        scores = np.array([s for _, s in pairs])
        good_mask = scores >= gamma_thresh
        good_count = np.sum(good_mask)
        good_ratio = good_count / len(pairs) if len(pairs) > 0 else 0
        
        points = np.array([p for p, _ in pairs])
        
        # Directional sampling toward good points
        if good_count >= 2 and np.random.rand() < 0.7:
            good_points = points[good_mask]
            good_center = np.mean(good_points, axis=0)
            best_idx = np.argmax(scores)
            best_point = points[best_idx]
            
            direction = best_point - good_center
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
            else:
                direction = np.random.randn(d)
                direction = direction / np.linalg.norm(direction)
            
            # Convert to local coords
            t_best = cube.to_prime(best_point)
            widths = cube._widths()
            scale = np.mean(widths) * 0.3
            
            # Step along direction
            step = direction * np.random.uniform(0, scale)
            lateral_noise = np.random.randn(d) * scale * 0.1
            
            x = best_point + step + lateral_noise
            x = self._clip(x)
            if self._in_bounds(x):
                self._last_sample_method = 'p2_directional'
                return x
        
        # Local search around best
        if np.random.rand() < 0.2 and self.best_x is not None:
            widths = cube._widths()
            noise = np.random.randn(d) * 0.1 * widths
            x = self.best_x + noise
            x = self._clip(x)
            if self._in_bounds(x):
                self._last_sample_method = 'p2_local_search'
                return x
        
        # Fallback
        self._last_sample_method = 'p2_uniform'
        return self._sample_uniform(cube)

    def _sample(self, cube: Cube) -> np.ndarray:
        """Sample based on cube phase, with dynamic fallback to Phase 2 if surrogate unhealthy"""
        # Check if we should dynamically switch to Phase 2
        use_phase2 = (cube.phase == 2)
        
        if cube.phase == 1:
            sanity = cube.surrogate_sanity()
            # Lower threshold: switch to Phase 2 if surrogate is getting unreliable
            if not sanity['healthy'] or sanity['cond_A'] > 1e6:
                use_phase2 = True
                _log(f"  [DYNAMIC_P2] Cube d={cube.depth} unhealthy, using Phase 2 sampling. "
                     f"cond={sanity['cond_A']:.1e} warnings={sanity['warnings']}")
        
        if use_phase2:
            return self._sample_phase2(cube)
        else:
            return self._sample_phase1(cube)

    def _sample_uniform(self, cube: Cube) -> np.ndarray:
        d = self.dim
        for _ in range(10):
            u = np.zeros(d)
            for j in range(d):
                lo, hi = cube.bounds[j]
                u[j] = np.random.uniform(lo, hi)
            x = cube.to_original(u)
            if self._in_bounds(x):
                return self._clip(x)
        return self._clip(cube.to_original(np.zeros(d)))

    def _select(self) -> Cube:
        """Select cube: UCB for Phase 1, density for Phase 2 cubes"""
        if self._preferred is not None:
            for c in self.leaves:
                if c is self._preferred:
                    pref = self._preferred
                    self._preferred = None
                    return pref
            self._preferred = None
        
        beta = self._beta()
        
        # Separate by phase
        phase1 = [c for c in self.leaves if c.phase == 1]
        phase2 = [c for c in self.leaves if c.phase == 2]
        
        # If we have Phase 2 cubes and passed threshold, prefer them
        progress = self.trial / max(1, self.budget)
        prefer_phase2 = progress > 0.3 and len(phase2) > 0
        
        if prefer_phase2 and np.random.rand() < 0.7:
            # Select from Phase 2 using density
            gamma_thresh = self._gamma_threshold()
            scores = [(c, c.density_score(gamma_thresh) + 0.1*c.ucb(beta, self.best_score)) for c in phase2]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Softmax selection from top 5
            top_k = min(5, len(scores))
            top_scores = np.array([s[1] for s in scores[:top_k]])
            if top_scores.max() > top_scores.min():
                top_scores = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min() + 1e-9)
            probs = np.exp(top_scores / 0.3)
            probs = probs / probs.sum()
            idx = np.random.choice(top_k, p=probs)
            return scores[idx][0]
        
        # Default: UCB selection across all leaves
        return max(self.leaves, key=lambda c: c.ucb(beta, self.best_score))

    def _backprop(self, leaf: Cube, score: float) -> None:
        gamma_thresh = self._gamma_threshold()
        curr = leaf.parent
        while curr:
            curr.n_trials += 1
            n = curr.n_trials
            if n == 1:
                curr.mean_score = score
                curr.M2 = 0.0
            else:
                delta = score - curr.mean_score
                curr.mean_score += delta/n
                curr.M2 += delta*(score - curr.mean_score)
            curr.var_score = curr.M2/(n-1) if n>1 else 0.0
            if score > curr.best_score:
                curr.best_score = score
            # BUG FIX: also propagate n_good to parents
            if gamma_thresh is not None and score >= gamma_thresh:
                curr.n_good += 1
            curr = curr.parent

    def _prune(self) -> None:
        """Prune low-performing cubes"""
        min_leaves = max(4, self.dim)
        if len(self.leaves) <= min_leaves:
            return
            
        progress = min(1.0, self.trial / max(1, self.budget))
        delta = max(0.01, 2.0*(1-progress))
        thresh = self.best_score - delta
        beta = self._beta()
        
        keep = []
        for c in self.leaves:
            # Keep Phase 2 cubes with good performance
            if c.phase == 2 and c.n_trials > 0:
                gamma_thresh = self._gamma_threshold()
                if c.density_score(gamma_thresh) > 0.1 or c.best_score > thresh:
                    keep.append(c)
                    continue
            
            # Phase 1: use UCB
            if c.n_trials == 0 or c.ucb(beta, self.best_score) >= thresh:
                keep.append(c)
                
        if len(keep) < min_leaves:
            ranked = sorted(self.leaves, key=lambda c: c.ucb(beta, self.best_score), reverse=True)
            for c in ranked:
                if not any(c is k for k in keep):
                    keep.append(c)
                if len(keep) >= min_leaves:
                    break
        
        if len(keep) < len(self.leaves):
            _log(f"[PRUNE] {len(self.leaves)} -> {len(keep)} leaves")
        
        self.leaves = keep

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int) -> Tuple[np.ndarray, float]:
        self.budget = budget
        init_phase = max(2*self.dim + 2, 8)
        
        _log(f"[START] B={budget} D={self.dim} init={init_phase} max_d={self._max_depth()} depth_thresh={self._effective_depth_threshold()}")
        
        for i in range(budget):
            self.trial += 1
            
            if i < init_phase:
                cube = self.root
                x = self._sample_uniform(cube)
                self._last_sample_method = 'init_uniform'
            else:
                cube = self._select()
                x = self._sample(cube)
            
            # Count sampling methods
            self._sample_method_counts[self._last_sample_method] = self._sample_method_counts.get(self._last_sample_method, 0) + 1
            
            if not self._in_bounds(x):
                x = self._clip(x)
            
            score_raw = float(objective(x))
            score = self.sign * score_raw
            
            if math.isnan(score) or math.isinf(score):
                continue
            
            self._all_scores.append(score)
            gamma_thresh = self._gamma_threshold()
            
            cube.update(score, x, self.dim, gamma_thresh)
            self._backprop(cube, score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_x = x.copy()
                _log(f"[BEST] t={self.trial} s={self.sign*score:.6f} phase={cube.phase} d={cube.depth}")
            
            # Detailed debug logging every 20 trials
            if self.trial % 20 == 0:
                beta = self._beta()
                _log(f"[DEBUG] t={self.trial} gamma_q={self._gamma_quantile():.3f} gamma_thresh={gamma_thresh:.4f}")
                _log(f"  Leaves: {len(self.leaves)} | Phases: P1={sum(1 for c in self.leaves if c.phase==1)} P2={sum(1 for c in self.leaves if c.phase==2)}")
                
                # Top 3 cubes by UCB with sanity info
                cubes_by_ucb = sorted(self.leaves, key=lambda c: c.ucb(beta, self.best_score), reverse=True)[:3]
                for idx, c in enumerate(cubes_by_ucb):
                    info = c.get_debug_info(beta, gamma_thresh)
                    sanity = info['sanity']
                    _log(f"  TOP{idx+1}_UCB: d={info['depth']} ph={info['phase']} n={info['n_trials']} ng={info['n_good']} "
                         f"ucb={info['ucb']:.4f} dens={info['density']:.3f} best={info['best_score']:.4f} r2={info['surrogate_r2']:.2f}")
                    _log(f"    SANITY: healthy={sanity['healthy']} cond={sanity['cond_A']:.1e} pca_ok={sanity['pca_ok']} "
                         f"pca_ratio={sanity['pca_ratio']:.2f} warnings={sanity['warnings']}")
                
                # Surrogate health across all leaves by depth
                depth_sanity = {}
                for c in self.leaves:
                    d = c.depth
                    san = c.surrogate_sanity()
                    if d not in depth_sanity:
                        depth_sanity[d] = {'total': 0, 'healthy': 0, 'cond_sum': 0, 'r2_sum': 0}
                    depth_sanity[d]['total'] += 1
                    depth_sanity[d]['healthy'] += 1 if san['healthy'] else 0
                    depth_sanity[d]['cond_sum'] += san['cond_A'] if san['cond_A'] < float('inf') else 1e12
                    depth_sanity[d]['r2_sum'] += san['r2']
                
                _log(f"  SANITY BY DEPTH:")
                for d in sorted(depth_sanity.keys()):
                    ds = depth_sanity[d]
                    avg_cond = ds['cond_sum'] / ds['total']
                    avg_r2 = ds['r2_sum'] / ds['total']
                    _log(f"    d={d}: {ds['healthy']}/{ds['total']} healthy, avg_cond={avg_cond:.1e}, avg_r2={avg_r2:.3f}")
                
                # Sample method statistics
                _log(f"  Sample methods: {dict(sorted(self._sample_method_counts.items()))}")
            
            if i >= init_phase:
                mode = cube.should_split(self.dim, self._max_depth(), budget - self.trial, self.best_score)
                if mode != 'none':
                    children = cube.split(self.dim, mode, self._effective_depth_threshold(), gamma_thresh)
                    self.leaves = [c for c in self.leaves if c is not cube] + children
                    self._preferred = max(children, key=lambda c: c.ucb(self._beta()))
                    
                if self.trial % 5 == 0:
                    self._prune()
        
        depths = [c.depth for c in self.leaves]
        phases = [c.phase for c in self.leaves]
        _log(f"[END] L={len(self.leaves)} d=[{min(depths)},{max(depths)}] P1={phases.count(1)} P2={phases.count(2)} best={self.sign*self.best_score:.6f}")
        _log(f"[END] Sample method final stats: {dict(sorted(self._sample_method_counts.items()))}")
        
        return self.best_x, self.sign * self.best_score


# Convenience wrapper
def minimize(
    objective: Callable[[np.ndarray], float],
    bounds,
    budget: int,
    seed: int = 42,
    depth_threshold: int = 5,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """Minimize objective function."""
    debug_log = print if verbose else None
    bounds_list = [(b[0], b[1]) for b in bounds]
    opt = HPOptimizer(bounds_list, maximize=False, seed=seed, debug_log=debug_log, depth_threshold=depth_threshold)
    return opt.optimize(objective, budget)


if __name__ == "__main__":
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
    dim = 5
    bounds = [(-5.0, 10.0)] * dim
    
    print("Testing HybridOptimizer on Rosenbrock...")
    best_x, best_y = minimize(rosenbrock, bounds, budget=200, seed=42, verbose=True)
    print(f"\nBest: {best_y:.6f}")
    print(f"x*: {best_x}")
