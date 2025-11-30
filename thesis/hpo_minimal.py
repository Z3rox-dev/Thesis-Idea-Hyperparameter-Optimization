from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any
import math
import numpy as np

# Global debug logger - set by HPOptimizer
_DEBUG_LOG: Optional[Callable[[str], None]] = None

def _log(msg: str) -> None:
    """Log debug message if logger is set."""
    if _DEBUG_LOG is not None:
        _DEBUG_LOG(msg)


def _rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Calcola l'angolo di rotazione tra due matrici di rotazione (in gradi)."""
    try:
        # R_rel = R2^T @ R1, poi angle = arccos((trace - 1) / 2)
        R_rel = R2.T @ R1
        trace = np.trace(R_rel)
        # Clamp per errori numerici
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))
    except:
        return 0.0


@dataclass
class Cube:
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
        pairs = list(self._tested_pairs)
        n_lin, n_quad = dim+2, 2*dim+2
        
        # Data Borrowing: prendi punti dal parent se ne servono di più
        # Soglia: borrowa se abbiamo meno di 3x il minimo quadratico
        if self.parent and len(pairs) < 3 * n_quad:
            parent_pairs = getattr(self.parent, '_tested_pairs', [])
            extra = []
            existing_pts = np.array([p for p,_ in pairs]) if pairs else np.empty((0, dim))
            
            for pp in parent_pairs:
                p_loc = pp[0]
                p_prime = self.to_prime(p_loc)
                
                # 1. Check bounds CON MARGINE 10%
                inside = True
                for i, (lo, hi) in enumerate(self.bounds):
                    margin = (hi - lo) * 0.1
                    if not (lo - margin <= p_prime[i] <= hi + margin):
                        inside = False
                        break
                if not inside:
                    continue
                
                # 2. Check distanza minima dai punti esistenti (evita duplicati)
                if existing_pts.size > 0:
                    dists = np.linalg.norm(existing_pts - p_loc, axis=1)
                    if np.min(dists) < 1e-9:
                        continue  # punto duplicato
                
                extra.append(pp)
            
            # 3. Limite massimo: prendi solo quanti ne servono per arrivare a 3x n_quad
            needed = 3 * n_quad - len(pairs)
            if needed > 0 and extra:
                pairs = pairs + extra[:needed]
        
        if len(pairs) < n_lin:
            self.surrogate = None
            return
        d = len(self.bounds)
        R_old = self.R.copy() if self.R is not None else np.eye(d)
        R, mu, evals, pca_ok = self._pca()
        
        X = np.array([p for p,_ in pairs])
        y = np.array([s for _,s in pairs])
        T = (R.T @ (X.T - mu.reshape(-1,1))).T
        # Floor proporzionale alla larghezza del cubo
        # Aumentato a 10% per evitare esplosione coefficienti quando
        # i punti sono concentrati su un asse
        widths = np.array([abs(hi-lo) for lo,hi in self.bounds])
        t_std_floor = 0.1 * widths  # 10% della larghezza (era 1%)
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
        
        # Adaptive regularization: aumenta alpha se cond_A troppo alto
        cond_A = float('inf')
        max_cond = 1e8  # soglia per stabilità numerica
        for attempt in range(3):
            try:
                cond_A = float(np.linalg.cond(A))
            except:
                cond_A = float('inf')
            
            if cond_A <= max_cond:
                break
            
            # Aumenta alpha proporzionalmente al problema
            boost = cond_A / max_cond
            alpha_new = alpha * boost
            A = Phi.T @ Phi + alpha_new * np.eye(n_p)
        
        try:
            w_s = np.linalg.solve(A, Phi.T @ y)
        except:
            _log(f"[ERR] d={self.depth} solve fail cond_A={cond_A:.2e}")
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
        
        # DEBUG: Check NaN/Inf
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            _log(f"[CRITICAL] d={self.depth} w NaN/Inf cond_A={cond_A:.2e}")
            self.surrogate = None
            return
        
        max_w = float(np.max(np.abs(w)))
        if max_w > 1e6:
            _log(f"[WARN] d={self.depth} |w|={max_w:.2e} cond_A={cond_A:.2e}")
        
        try:
            A_inv = np.linalg.inv(A)
        except:
            A_inv = None
        
        self.surrogate = {'w':w,'R':R,'mu':mu,'sigma2':sigma2,'r2':r2,'lam':lam,'A_inv':A_inv,'t_std':t_std,'mode':mode,'n':len(pairs)}

    def predict(self, x_prime: np.ndarray) -> Tuple[float,float]:
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
            return y_hat, math.sqrt(max(sigma2,1e-12))
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
            return self.R.copy(), self.mu.copy(), np.maximum(evals,1e-9), False
        return evecs, mu, np.maximum(evals,1e-9), True

    def curvature_scores(self) -> Optional[np.ndarray]:
        s = self.surrogate
        if s is None or s.get('r2',0) < 0.05:
            return None
        lam = np.abs(s['lam'])
        self._ensure_frame()
        M = s['R'].T @ self.R
        w = self._widths()
        h = np.abs(M) @ w
        return (lam**2) * (h**4)

    def ucb(self, beta: float, global_best: float = -np.inf) -> float:
        if self.n_trials > 0:
            # Usa best_score come base per exploitation
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

    def update(self, score: float, x: np.ndarray, dim: int) -> None:
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
        d = len(self.bounds)
        ax = 0
        cut = float(np.median(T[:,ax]))
        left = T[:,ax] < cut
        res = []
        for mask in [left, ~left]:
            idx = np.where(mask)[0]
            res.append({'n':len(idx), 'var':float(np.var(y[idx])) if len(idx)>1 else 1.0})
        return res

    def _quad_cut(self, ax: int, R: np.ndarray, mu: np.ndarray) -> float:
        """Calcola il punto di taglio ottimale per l'asse ax nel frame (R, mu)."""
        d = len(self.bounds)
        self._ensure_frame()
        
        widths = self._widths()
        M = R.T @ self.R  # trasformazione da self.R a R
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

    def split(self, dim: int, mode: str) -> List["Cube"]:
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
        cut_i = self._quad_cut(ax_i, R_use, mu_use)
        cut_j = self._quad_cut(ax_j, R_use, mu_use) if mode=='quad' else 0.0
        lo_i, hi_i = base[ax_i]
        lo_j, hi_j = base[ax_j] if mode=='quad' else (0,0)
        cut_i = float(np.clip(cut_i, lo_i+1e-12, hi_i-1e-12))
        if mode == 'quad':
            cut_j = float(np.clip(cut_j, lo_j+1e-12, hi_j-1e-12))
        if mode == 'binary':
            quads = [(True,), (False,)]
        else:
            quads = [(True,True),(False,True),(True,False),(False,False)]
        children = []
        for q in quads:
            nb = list(base)
            nb[ax_i] = (lo_i, cut_i) if q[0] else (cut_i, hi_i)
            if mode == 'quad':
                nb[ax_j] = (lo_j, cut_j) if q[1] else (cut_j, hi_j)
            ctr = np.array([(a+b)*0.5 for a,b in nb])
            wch = np.array([b-a for a,b in nb])
            ch = Cube(bounds=[(-w/2, w/2) for w in wch], parent=self)
            ch.R = R_use.copy()
            ch.mu = (mu_use + R_use @ ctr).astype(float)
            ch.depth = self.depth + 1
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
                    ci = {(True,True):0,(False,True):1,(True,False):2,(False,False):3}[(li,lj)]
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
                ch.fit_surrogate(dim)
        
        _log(f"[SPLIT] d={self.depth} {mode} n={len(self._tested_pairs)} -> ch={[len(c._tested_pairs) for c in children]}")
        
        self.children = children
        return children


class HPOptimizer:
    def __init__(self, bounds: List[Tuple[float,float]], maximize: bool = True, seed: Optional[int] = None,
                 debug_log: Optional[Callable[[str], None]] = None):
        global _DEBUG_LOG
        _DEBUG_LOG = debug_log
        
        self.bounds = list(bounds)
        self.dim = len(bounds)
        self.maximize = maximize
        self.sign = 1.0 if maximize else -1.0
        if seed is not None:
            np.random.seed(seed)
        lo = np.array([l for l,h in bounds])
        hi = np.array([h for l,h in bounds])
        root_mu = (lo+hi)*0.5
        root_w = hi - lo
        self.root = Cube(bounds=[(-w/2, w/2) for w in root_w])
        self.root.R = np.eye(self.dim)
        self.root.mu = root_mu
        self.leaves: List[Cube] = [self.root]
        self.best_score = -np.inf
        self.best_x: Optional[np.ndarray] = None
        self.trial = 0
        self.budget = 100
        self._preferred: Optional[Cube] = None
        self._debug_log = debug_log

    def _beta(self) -> float:
        return 0.5 * math.sqrt(math.log(max(2, self.trial+1)))

    def _max_depth(self) -> int:
        return int(2*self.dim + 3*math.log2(max(2, self.budget)))

    def _in_bounds(self, x: np.ndarray) -> bool:
        for j,(lo,hi) in enumerate(self.bounds):
            if x[j] < lo-1e-9 or x[j] > hi+1e-9:
                return False
        return True

    def _clip(self, x: np.ndarray) -> np.ndarray:
        for j,(lo,hi) in enumerate(self.bounds):
            x[j] = float(np.clip(x[j], lo, hi))
        return x

    def _trunc_norm(self, mu: float, std: float, lo: float, hi: float) -> float:
        if std <= 1e-12:
            return float(np.clip(mu, lo, hi))
        for _ in range(100):
            v = np.random.normal(mu, std)
            if lo <= v <= hi:
                return float(v)
        return float(np.clip(np.random.normal(mu, std), lo, hi))

    def _sample(self, cube: Cube) -> np.ndarray:
        d = self.dim
        pairs = cube._tested_pairs
        s = cube.surrogate
        sample_method = 'unknown'
        
        # ============================================================
        # Gradient-based step 
        # ============================================================
        if s and s['n'] >= d+2 and s['r2'] >= 0.5 and np.random.rand() < 0.6:
            w = s['w']
            grad = w[1:d+1]  # Coefficienti lineari = gradiente al centro
            grad_norm = np.linalg.norm(grad)
            
            # Se il gradiente è significativo, fai uno step nella direzione
            if grad_norm > 0.02:
                # Normalizza e scala per la larghezza del cubo
                widths = cube._widths()
                # Step size 10% della larghezza
                base_step = 0.10
                step_size = base_step * widths
                direction = grad / grad_norm  # Direzione normalizzata
                
                # Step dal centro del cubo (o dal best point)
                if pairs:
                    best_pt, _ = max(pairs, key=lambda t: t[1])
                    t_center = cube.to_prime(best_pt)
                else:
                    t_center = np.zeros(d)
                
                # Fai step nella direzione del gradiente (maximize -> stesso segno)
                t_new = t_center + direction * step_size
                
                # Clamp ai bounds del cubo
                for j in range(d):
                    lo, hi = cube.bounds[j]
                    t_new[j] = float(np.clip(t_new[j], lo, hi))
                
                x = cube.to_original(t_new)
                if self._in_bounds(x):
                    self._last_sample_method = 'gradient_step'
                    return self._clip(x)
        
        # Ottimo surrogata se curvatura negativa
        if s and s['n'] >= 2*d+2 and s['r2'] >= 0.85 and s['mode'] == 'quad':
            w = s['w']
            all_neg = all(w[d+1+j] < -1e-9 for j in range(d))
            if all_neg:
                t_opt = np.zeros(d)
                for j in range(d):
                    t_opt[j] = -w[1+j] / (2*w[d+1+j])
                x_opt = s['mu'] + s['R'] @ t_opt
                if self._in_bounds(x_opt):
                    for j in range(d):
                        lo, hi = cube.bounds[j]
                        u_j = (s['R'].T @ (x_opt - s['mu']))[j]
                        if lo <= u_j <= hi:
                            continue
                        break
                    else:
                        self._last_sample_method = 'quad_opt'
                        return self._clip(x_opt)
        
        # Perturbazione attorno ai migliori
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
                self._last_sample_method = 'perturb'
                return self._clip(x)
        
        # EI con trust region
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
            
            # Trova i migliori punti per centrare il sampling
            if pairs:
                top_k = min(5, len(pairs))
                best_pts = sorted(pairs, key=lambda t: t[1], reverse=True)[:top_k]
                best_ts = [cube.to_prime(p) for p, _ in best_pts]
            else:
                best_ts = [np.zeros(d)]
            
            for _ in range(M):
                # Centra sampling su uno dei best points (70%), o esplora (30%)
                if np.random.rand() < 0.7 and best_ts:
                    center = best_ts[np.random.randint(len(best_ts))]
                else:
                    center = np.zeros(d)
                
                u = np.zeros(d)
                for j in range(d):
                    lo, hi = cube.bounds[j]
                    # Scala ridotta attorno al best, più ampia per esplorazione
                    scale = (hi-lo)*0.15 if np.any(center) else (hi-lo)*0.25
                    u[j] = self._trunc_norm(center[j], scale, lo, hi)
                x = cube.to_original(u)
                if not self._in_bounds(x):
                    continue
                t = R_s.T @ (x - mu_s)
                rho2 = float(t @ C_inv @ t)
                if rho2 > rho_max2:
                    continue
                y_hat, sigma = cube.predict(t)
                diff = y_hat - local_best
                z = diff / max(sigma, 1e-9)
                
                # Protezione underflow
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
                self._last_sample_method = 'ei'
                self._last_ei_best = cands[0][0]
                return self._clip(cands[0][1])
        
        # Fallback uniform
        self._last_sample_method = 'uniform'
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
        if self._preferred is not None:
            for c in self.leaves:
                if c is self._preferred:
                    pref = self._preferred
                    self._preferred = None
                    return pref
            self._preferred = None
        beta = self._beta()
        return max(self.leaves, key=lambda c: c.ucb(beta, self.best_score))

    def _backprop(self, leaf: Cube, score: float) -> None:
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
            curr = curr.parent

    def _prune(self) -> None:
        min_leaves = max(4, self.dim)
        if len(self.leaves) <= min_leaves:
            return
        progress = min(1.0, self.trial / max(1, self.budget))
        delta = max(0.01, 2.0*(1-progress))
        thresh = self.best_score - delta
        beta = self._beta()
        keep = []
        for c in self.leaves:
            if c.n_trials == 0 or c.ucb(beta, self.best_score) >= thresh:
                keep.append(c)
        if len(keep) < min_leaves:
            ranked = sorted(self.leaves, key=lambda c: c.ucb(beta, self.best_score), reverse=True)
            for c in ranked:
                if not any(c is k for k in keep):
                    keep.append(c)
                if len(keep) >= min_leaves:
                    break
        
        self.leaves = keep

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int) -> Tuple[np.ndarray, float]:
        self.budget = budget
        init_phase = max(2*self.dim + 2, 8)
        self._last_sample_method = 'init'
        self._last_ei_best = 0.0
        
        _log(f"[START] B={budget} D={self.dim} init={init_phase} max_d={self._max_depth()}")
        
        for i in range(budget):
            self.trial += 1
            if i < init_phase:
                cube = self.root
                x = self._sample_uniform(cube)
                self._last_sample_method = 'init'
            else:
                cube = self._select()
                x = self._sample(cube)
            
            if not self._in_bounds(x):
                x = self._clip(x)
            
            score_raw = float(objective(x))
            score = self.sign * score_raw
            
            if math.isnan(score) or math.isinf(score):
                continue
            
            # LOG DEBUG: ogni 10 trial dopo init, logga stato dettagliato
            if i >= init_phase and self.trial % 10 == 0:
                beta = self._beta()
                # Calcola info per ogni foglia
                leaf_details = []
                for c in self.leaves:
                    ucb_val = c.ucb(beta, self.best_score)
                    leaf_details.append((c, c.depth, c.n_trials, ucb_val, c.mean_score, c.var_score, c.best_score))
                leaf_details.sort(key=lambda x: x[3], reverse=True)  # sort by UCB
                
                # Trova il cubo col miglior best_score
                best_leaf = max(self.leaves, key=lambda c: c.best_score)
                best_leaf_ucb = best_leaf.ucb(beta, self.best_score)
                best_leaf_rank = next(i for i, l in enumerate(leaf_details) if l[0] is best_leaf) + 1
                
                gap = self.best_score - score
                _log(f"[DBG] t={self.trial} L={len(self.leaves)} method={self._last_sample_method} gap={gap:.4f}")
                _log(f"  SELECTED: d={cube.depth} n={cube.n_trials} mean={cube.mean_score:.4f} var={cube.var_score:.4f} best={cube.best_score:.4f}")
                _log(f"  BEST_LEAF rank={best_leaf_rank}/{len(self.leaves)}: d={best_leaf.depth} n={best_leaf.n_trials} mean={best_leaf.mean_score:.4f} var={best_leaf.var_score:.4f} best={best_leaf.best_score:.4f} ucb={best_leaf_ucb:.4f}")
                # Top 3 per UCB
                for idx, (c, d, n, ucb, mean, var, best) in enumerate(leaf_details[:3]):
                    marker = "<--BEST" if c is best_leaf else ""
                    _log(f"  TOP{idx+1}: d={d} n={n} ucb={ucb:.4f} mean={mean:.4f} var={var:.4f} best={best:.4f} {marker}")
            
            cube.update(score, x, self.dim)
            self._backprop(cube, score)
            if score > self.best_score:
                self.best_score = score
                self.best_x = x.copy()
                _log(f"[BEST] t={self.trial} s={self.sign*score:.6f} method={self._last_sample_method} cube_d={cube.depth}")
            
            if i >= init_phase:
                mode = cube.should_split(self.dim, self._max_depth(), budget - self.trial, self.best_score)
                if mode != 'none':
                    children = cube.split(self.dim, mode)
                    self.leaves = [c for c in self.leaves if c is not cube] + children
                    self._preferred = max(children, key=lambda c: c.ucb(self._beta()))
                if self.trial % 5 == 0:
                    self._prune()
        
        depths = [c.depth for c in self.leaves]
        _log(f"[END] L={len(self.leaves)} d=[{min(depths)},{max(depths)}] best={self.sign*self.best_score:.6f}")
        
        return self.best_x, self.sign * self.best_score

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
