from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, Dict, Tuple as TypingTuple

import json
import csv
import os
import math
import numpy as np


# Simple ParamSpace placeholder (not used in benchmark, can be extended later)
class ParamSpace:
    """Placeholder for parameter space definition."""
    def __init__(self):
        pass
    
    def denormalize(self, x_norm):
        return x_norm.tolist()


@dataclass
class AdaptiveConfig:
    """Centralized configuration for QuadHPO that adapts to problem scale."""
    # Problem context
    dim: int = 1
    budget: int = 100
    
    # Surrogate
    ridge_alpha: float = 1e-3  # Will be dynamic
    surrogate_min_points: int = 10
    surr_r2_min: float = 0.55
    
    # Tree
    max_depth: int = 6
    min_points_split: int = 10
    min_width: float = 1e-6
    gamma_split: float = 0.02  # Info gain threshold
    
    # Sampling
    ucb_beta: float = 1.0
    pca_mix: float = 0.25
    
    def adapt(self, dim: int, budget: int):
        self.dim = dim
        self.budget = budget
        
        # 1. Surrogate Regularization
        # Logic: Higher dim -> more sparse data -> need more regularization
        # Base 1e-3, scale with sqrt(D)
        self.ridge_alpha = 1e-3 * math.sqrt(dim)
        
        # 2. Min Points
        # Logic: To fit a diagonal quadratic (D+1 params) we need at least D+1 points.
        # We add a small safety margin.
        self.surrogate_min_points = dim + 2
        self.min_points_split = self.surrogate_min_points
        
        # 3. Tree Depth
        # Logic: We need enough depth to cut all dimensions (D) and zoom in (log B).
        # Old: 12 + log2(dim) -> Too shallow for high D.
        # New: Linear in D to cover space, Log in B for precision.
        # Factor 2*D ensures we can cut each dim twice (quadtree style or refinement).
        # Factor 3*log2(B) allows zooming.
        self.max_depth = int(2 * dim + 3 * math.log2(max(2, budget)))
        
        # 4. UCB Exploration
        # Logic: UCB theory suggests scaling with sqrt(log(T)).
        # We scale it down (0.5 factor) because we want faster convergence than pure regret minimization.
        # For Budget=100 -> Beta ~ 1.0. For Budget=1000 -> Beta ~ 1.3.
        self.ucb_beta = 0.5 * math.sqrt(math.log(max(2, budget))) 
        
        # 5. PCA Mixing
        # Logic: In high dim, PCA can overfit. Mix in more uniform sampling.
        # 1/D is a good heuristic.
        self.pca_mix = 1.0 / max(1, dim)

        # 6. Split Threshold (Gamma)
        # Logic: If budget is tight, be picky (high gamma). If budget is loose, explore (low gamma).
        # Base 0.01, scaled by budget density? 
        # For now, let's stick to a robust constant or slight scaling.
        self.gamma_split = 0.01  # Slightly more permissive than 0.02

        # 7. Min Width
        # Logic: Resolution limit. With infinite budget, we want machine epsilon.
        # With finite budget B, we can't reliably distinguish intervals smaller than 1/B (heuristic).
        # We add a safety factor.
        self.min_width = max(1e-8, 1.0 / (100.0 * float(budget)))


@dataclass
class QuadCube:
    # Surrogate model cache (2D quadratic on PC1-PC2)
    surrogate_2d: Optional[Dict[str, Any]] = field(default=None, init=False)
    
    # Debug logger callback (passed from QuadHPO)
    _debug_logger: Optional[Callable[[str], None]] = field(default=None, init=False, repr=False)
    
    #Metodo che adatta una funzione quadratica sulle prime due componenti principali (PC1 e PC2)
    def fit_surrogate(self, min_points: int, ridge_alpha: float = 1e-3) -> None:
        """Fit a quadratic surrogate using local tested pairs.
        
        - If D <= 2: Fits full quadratic on PC1-PC2 (6 params).
        - If D > 2: Fits DIAGONAL quadratic on all D dims (2D+1 params).
          y ~ w0 + sum(w_i * x_i) + sum(v_i * x_i^2)
        """
        
        # --- MODIFICA: Raccolta dati "Allargata" (Data Borrowing) ---
        pairs = getattr(self, "_tested_pairs", [])
        if len(pairs) < min_points * 2 and self.parent is not None:
            parent_pairs = getattr(self.parent, "_tested_pairs", [])
            extra = []
            if len(parent_pairs) > 0:
                # --- FIX: Filter parent points to be within bounds ---
                # Only borrow points that are geometrically relevant
                for pp in parent_pairs:
                    p_loc = pp[0]
                    # Check if inside bounds with margin
                    p_prime = self.to_prime(p_loc)
                    inside = True
                    for i, (lo, hi) in enumerate(self.bounds):
                        margin = (hi - lo) * 0.1 # 10% margin
                        if not (lo - margin <= p_prime[i] <= hi + margin):
                            inside = False
                            break
                    if inside:
                        # Check distance to existing points
                        if len(pairs) > 0:
                            existing_locs = np.array([p[0] for p in pairs])
                            dists = np.linalg.norm(existing_locs - p_loc, axis=1)
                            if np.min(dists) > 1e-9:
                                extra.append(pp)
                        else:
                            extra.append(pp)
                # -----------------------------------------------------
            needed = (min_points * 3) - len(pairs)
            if needed > 0:
                pairs = pairs + extra[:needed]
        # ------------------------------------------------------------

        if len(pairs) < min_points:
            self.surrogate_2d = None
            return
            
        d = len(self.bounds)
        # Use local PCA axes for projection
        R, mu, _, ok = self._principal_axes()
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        T = (R.T @ (X.T - mu.reshape(-1, 1))).T  # shape (n, d)
        
        # --- FIX: Normalize T to avoid over-regularization in small cubes ---
        # Ridge alpha is constant (e.g. 1e-3). If T is small (e.g. 1e-3), T^2 is 1e-6.
        # Phi.T @ Phi becomes very small, dominated by ridge_alpha.
        # We scale T to unit variance for fitting, then unscale weights.
        t_std = np.std(T, axis=0)
        t_std = np.maximum(t_std, 1e-9) # Avoid div by zero
        T_scaled = T / t_std
        # -------------------------------------------------------------------
        
        # DECISION: Full 2D vs Diagonal ND
        use_diag = (d > 2)
        
        if use_diag:
            # Diagonal Quadratic: [1, t1...tD, t1^2...tD^2]
            # Params: 1 + D + D = 2D + 1
            n_params = 2 * d + 1
            Phi = np.zeros((len(y), n_params))
            Phi[:, 0] = 1.0
            Phi[:, 1:d+1] = T_scaled
            Phi[:, d+1:] = T_scaled**2
            
            # Ridge
            A = Phi.T @ Phi + ridge_alpha * np.eye(n_params)
            b = Phi.T @ y
            try:
                w_scaled = np.linalg.solve(A, b)
            except Exception:
                self.surrogate_2d = None
                return
            
            # SANITY CHECK: Check for NaN/Inf in surrogate predictions
            if self._debug_logger:
                test_pred = Phi @ w_scaled
                if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
                    self._debug_logger(f"[CRITICAL] Surrogate predictions contain NaN/Inf! w_scaled={w_scaled}, Phi.shape={Phi.shape}")
            
            # Unscale weights to original frame
            w = np.zeros_like(w_scaled)
            w[0] = w_scaled[0]
            w[1:d+1] = w_scaled[1:d+1] / t_std
            w[d+1:] = w_scaled[d+1:] / (t_std**2)
                
            # Hessian is diagonal: H_ii = 2 * w[d+1+i] (since term is w*t^2, 2nd deriv is 2w)
            # Wait, standard form is 0.5 * a * t^2.
            # If we fit w * t^2, then 0.5 * a = w => a = 2w.
            # Let's stick to w * t^2 for simplicity.
            
            # Eigenvalues of Hessian are just 2 * w[d+1:]
            quad_coeffs = w[d+1:]
            # FIX: Do NOT sort eigenvalues in diagonal case, as they correspond 1:1 to PCA axes
            # Sorting would break the alignment with R columns in _curvature_scores
            lam = 2.0 * quad_coeffs 
            
        else:
            # Full 2D Quadratic (Legacy)
            t1 = T_scaled[:, 0]
            t2 = T_scaled[:, 1] if d > 1 else np.zeros_like(t1)
            Phi = np.stack([
                np.ones_like(t1), t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2
            ], axis=1)
            n_params = 6
            A = Phi.T @ Phi + ridge_alpha * np.eye(6)
            b = Phi.T @ y
            try:
                w_scaled = np.linalg.solve(A, b)
            except Exception:
                self.surrogate_2d = None
                return
            
            # SANITY CHECK: Check for NaN/Inf in surrogate predictions
            if self._debug_logger:
                test_pred = Phi @ w_scaled
                if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
                    self._debug_logger(f"[CRITICAL] Surrogate predictions contain NaN/Inf! w_scaled={w_scaled}, Phi.shape={Phi.shape}")
            
            # Unscale weights
            # w_scaled = [bias, t1, t2, 0.5*t1^2, 0.5*t2^2, t1*t2]
            s1 = t_std[0]
            s2 = t_std[1] if d > 1 else 1.0
            w = np.zeros_like(w_scaled)
            w[0] = w_scaled[0]
            w[1] = w_scaled[1] / s1
            w[2] = w_scaled[2] / s2
            w[3] = w_scaled[3] / (s1**2)
            w[4] = w_scaled[4] / (s2**2)
            w[5] = w_scaled[5] / (s1 * s2)
            
            w_flat = w.reshape(-1)
            H = np.array([[float(w_flat[3]), float(w_flat[5])],
                          [float(w_flat[5]), float(w_flat[4])]], dtype=float)
            lam = np.linalg.eigvalsh(H)[::-1]

        # Common Stats
        try:
            A_inv = np.linalg.inv(A)
        except Exception:
            A_inv = None
            
        # FIX: Use w_scaled for prediction because Phi is scaled
        y_hat = Phi @ w_scaled
        resid = y - y_hat
        G = Phi.T @ Phi
        try:
            if A_inv is None:
                A_inv = np.linalg.inv(A)
            df_eff = float(np.trace(G @ A_inv))
        except Exception:
            df_eff = float(min(Phi.shape[0], Phi.shape[1]))
            
        rss = float(np.sum(resid * resid))
        n_obs = Phi.shape[0]
        denom = max(1.0, n_obs - df_eff)
        sigma2 = float(rss / denom) if n_obs > 1 else 1.0
        var_y = float(np.var(y)) if n_obs > 1 else 0.0
        r2 = 1.0 - (float(np.var(resid)) / max(var_y, 1e-12)) if var_y > 0 else 0.0
        
        self.surrogate_2d = {
            'type': 'diag' if use_diag else 'full2d',
            'w': w,
            'mu': mu,
            'R': R,
            'A_inv': A_inv,
            'sigma2': sigma2,
            'n': n_obs,
            'r2': r2,
            'pca_ok': bool(ok),
            'df_eff': df_eff,
            'lambda': lam,
            't_std': t_std, # Store scaling factors for prediction
        }

    def predict_surrogate(self, x_prime: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std at x_prime (in prime coords)."""
        if self.surrogate_2d is None:
            return 0.0, 1.0
            
        stype = self.surrogate_2d.get('type', 'full2d')
        d = len(x_prime)
        
        # Retrieve scaling factors
        t_std = self.surrogate_2d.get('t_std', np.ones(d))
        
        # Scale input for variance calculation (A_inv is in scaled space)
        x_prime_scaled = x_prime / t_std
        
        if stype == 'diag':
            # Diagonal: [1, t1...tD, t1^2...tD^2]
            # Unscaled Phi for Mean
            Phi = np.zeros(2 * d + 1)
            Phi[0] = 1.0
            Phi[1:d+1] = x_prime
            Phi[d+1:] = x_prime**2
            
            # Scaled Phi for Variance
            Phi_scaled = np.zeros(2 * d + 1)
            Phi_scaled[0] = 1.0
            Phi_scaled[1:d+1] = x_prime_scaled
            Phi_scaled[d+1:] = x_prime_scaled**2
        else:
            # Full 2D
            t1 = x_prime[0]
            t2 = x_prime[1] if len(x_prime) > 1 else 0.0
            # Unscaled Phi for Mean
            Phi = np.array([1.0, t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2])
            
            # Scaled Phi for Variance
            ts1 = x_prime_scaled[0]
            ts2 = x_prime_scaled[1] if len(x_prime_scaled) > 1 else 0.0
            Phi_scaled = np.array([1.0, ts1, ts2, 0.5 * ts1 ** 2, 0.5 * ts2 ** 2, ts1 * ts2])

        y_hat = float(self.surrogate_2d['w'] @ Phi)
        sigma2 = float(self.surrogate_2d.get('sigma2', 1.0))
        A_inv = self.surrogate_2d.get('A_inv', None)
        
        if A_inv is None:
            sigma = float(np.sqrt(max(sigma2, 1e-12)))
            return y_hat, sigma
            
        try:
            # Use Scaled Phi for variance
            v = float(Phi_scaled @ (A_inv @ Phi_scaled))
            v = max(v, 0.0)
        except Exception:
            v = 0.0
        var_mean = sigma2 * v
        sigma = float(np.sqrt(max(var_mean, 1e-12)))
        return y_hat, sigma

    def _curvature_scores(self) -> Optional[np.ndarray]:
        """Compute curvature-driven split scores S_i = |λ_i|² · h_i⁴.
        
        Uses eigenvalues from the Hessian of the 2D quadratic surrogate and
        cube widths measured in the surrogate's PCA frame (not the cube's frame).
        
        Returns:
            np.ndarray of shape (k,) with scores for all available surrogate axes,
            or None if surrogate is not available or doesn't have eigenvalues.
        """
        s = self.surrogate_2d
        if s is None or s.get('lambda') is None or s.get('R') is None or s.get('mu') is None:
            return None
        
        # --- FIX: Check R2 before trusting curvature ---
        # If the model explains nothing (R2 < 0.1), the curvature is likely noise.
        # Fall back to geometric splitting.
        if s.get('r2', 0.0) < 0.05:
            return None
        # -----------------------------------------------
        
        # --- MODIFICA SUBSPACE: Usa tutte le curvature disponibili ---
        lam = np.asarray(s['lambda'], float)
        k = len(lam)
        # -------------------------------------------------------------
        
        # Use absolute curvature to avoid missing informative negative curvature
        lam_abs = np.abs(lam)

        self._ensure_frame()
        R_s = np.asarray(s['R'], float)
        # M = R_s^T @ R  (offset is irrelevant for computing range)
        M = R_s.T @ self.R

        # width_j = hi - lo (handle potentially inverted bounds)
        w = np.array([abs(hi - lo) for (lo, hi) in self.bounds], float)
        if w.size == 0:
            return None

        # h for first k surrogate axes: h_i = Σ |M[i,j]| * width_j
        # Note: h uses full widths (not half-spans). Any constant factor cancels out
        # in axis ranking; only relative magnitudes matter for choosing split axes.
        
        # --- MODIFICA SUBSPACE: Proietta su tutti i k assi ---
        A = np.abs(M[:k, :])         # (k, d)
        h = A @ w                    # (k,)
        # -----------------------------------------------------

        # h is guaranteed >= 0 by construction, but clamp for numerical safety
        h = np.maximum(h, 0.0)

        return (lam_abs**2) * (h**4)
    
    bounds: List[Tuple[float, float]]  # in local (prime) coordinates, typically symmetric around 0
    parent: Optional["QuadCube"] = None
    surrogate_min_points: int = 8

    children: List["QuadCube"] = field(default_factory=list) #Ogni “cella” è un cubetto in un sistema di assi comodo. I cubetti possono avere un genitore e dei figli, come una cartella con sottocartelle

    # local frame
    R: Optional[np.ndarray] = None  # shape (d, d)
    mu: Optional[np.ndarray] = None  # shape (d,) #mu dice dov’è il centro del cubetto nel mondo reale; R dice come sono ruotati gli assi del cubetto rispetto a quelli normali.

    # statistics
    n_trials: int = 0
    scores: List[float] = field(default_factory=list)
    best_score: float = -np.inf
    mean_score: float = 0.0
    var_score: float = 0.0
    M2: float = 0.0
    #Quanti esperimenti ho fatto qui? Com’è andata in media? Qual è stato il migliore? Tengo anche qualche appunto rapido per decidere se continuare o fermarmi presto.
    # region params
    #Manopole per decidere quando una cella è promettente o quando è ora di dividerla/abbandonarla. Tipo regole della casa per l’esplorazione.
    prior_var: float = 1.0
    stale_steps: int = 0
    depth: int = 0
    birth_trial: int = field(default=0, init=False)  # Trial number when this cube was created
    # early stopping removed

    # geometry helpers
    #Se non ho ancora deciso come sono ruotati gli assi, li tengo dritti. Se non so dov’è il centro, provo a mettere qualcosa di sensato (al centro del quadrato standard) finché qualcuno non me lo imposta meglio
    def _ensure_frame(self) -> None:
        d = len(self.bounds)
        if self.R is None:
            self.R = np.eye(d)
        if self.mu is None:
            # center from parent mapping if possible, else origin
            # If bounds are in local coords and typically symmetric around 0, center is mu
            if self.parent is not None and self.parent.mu is not None and self.parent.R is not None:
                # Inherit parent's frame and set center by mapping local prime center to original coords
                ctr_prime = self._center_prime()
                self.mu = (self.parent.mu + (self.parent.R @ ctr_prime)).astype(float)
                self.R = self.parent.R.copy()
            else:
                # default center at midpoint of original [0,1]^d
                self.mu = np.full(d, 0.5, dtype=float)
    #Tolgo il “punto centrale” e ruoto il sistema per vedere il punto con gli “occhiali giusti” (le nuove assi)
    def to_prime(self, x: np.ndarray) -> np.ndarray:
        self._ensure_frame()
        return (self.R.T @ (x - self.mu)).astype(float)
    #Rimetto la rotazione com’era e riaggiungo il centro: torno nel mondo “vero”.
    def to_original(self, x_prime: np.ndarray) -> np.ndarray:
        self._ensure_frame()
        x = (self.mu + self.R @ x_prime).astype(float)
        return x
    #Calcola il punto medio degli intervalli in x' È il centro del cubo nel frame locale
    def _center_prime(self) -> np.ndarray:
        # mid-point in prime coords
        mids = np.array([(lo + hi) * 0.5 for (lo, hi) in self.bounds], dtype=float)
        return mids
    #Calcola le larghezze degli intervalli in x' (distanza tra i limiti) per ogni dimensione. Quanto è largo il cubo su ogni asse?
    def _widths(self) -> np.ndarray:
        # Use absolute width to be robust to any inadvertent (hi, lo) orderings
        return np.array([abs(hi - lo) for (lo, hi) in self.bounds], dtype=float)
    #Pesco a caso un punto dentro il cubetto locale, con la stessa probabilità ovunque.
    def sample_uniform_prime(self) -> np.ndarray:
        d = len(self.bounds)
        point = np.zeros(d, dtype=float)
        for i, (lo, hi) in enumerate(self.bounds):
            lo_i, hi_i = (lo, hi) if hi >= lo else (hi, lo)
            point[i] = np.random.uniform(lo_i, hi_i)
        # Debug: ensure the prime sample is within local bounds
        if hasattr(self, '_debug_assert_bounds') and self._debug_assert_bounds:
            for i, (lo, hi) in enumerate(self.bounds):
                lo_i, hi_i = (lo, hi) if hi >= lo else (hi, lo)
                if point[i] < lo_i - 1e-9 or point[i] > hi_i + 1e-9:
                    raise AssertionError(f"Prime sample fuori bounds asse {i}: {point[i]} not in [{lo_i},{hi_i}]")
        return point
    #Pesco un punto nel cubetto comodo e poi lo traduco nel mondo reale.
    def sample_uniform(self) -> np.ndarray:
        # sample in prime, map to original
        x_prime = self.sample_uniform_prime()
        return self.to_original(x_prime)
    #Segno da qualche parte i punti che ho provato davvero, così non me li dimentico
    def add_tested_point(self, point: np.ndarray) -> None:
        # store original coordinates
        if not hasattr(self, "_tested_points"):
            self._tested_points: List[np.ndarray] = []
        self._tested_points.append(np.array(point, dtype=float))
    #Quando chiedo “che punti hai provato?”, o mi dai la lista oppure mi dici “nessuno”
    @property
    def _points_history(self) -> List[np.ndarray]:
        return getattr(self, "_tested_points", [])

    """A ogni indizio “preliminare” aggiorno una soglia che mi dice se la zona promette. Se ho pochi indizi, mi fido un po’ di quello che pensa il genitore. 
    Aggiorno anche quanto penso che i risultati possano variare. Poi ridisegno la mia collinetta se ho dati a sufficienza."""
    # update_early removed

    def update_final(self, final_score: float, min_points: int = 10, ridge_alpha: float = 1e-3) -> None: #Aggiorna statistiche finali: numero prove, lista degli score, media, varianza
        self.n_trials += 1
        self.scores.append(float(final_score))
        
        # Welford's online algorithm for mean and variance
        n = self.n_trials
        if n == 1:
            self.mean_score = float(final_score)
            self.M2 = 0.0
        else:
            delta = final_score - self.mean_score
            self.mean_score += delta / n
            delta2 = final_score - self.mean_score
            self.M2 += delta * delta2
        
        self.var_score = self.M2 / (n - 1) if n > 1 else 0.0
        
        if final_score > self.best_score:
            self.best_score = float(final_score)
        # Refit surrogate after each update if enough data
        self.fit_surrogate(min_points=min_points, ridge_alpha=ridge_alpha)

    def leaf_score(self, mode: str = 'max') -> float: #Dimmi quanto è “buona” questa zona: preferisci il migliore risultato o la media? Se non ho risultati finali, guardo quelli provvisori; se non ho niente, dico che è pessima 
        if self.scores:
            return float(np.max(self.scores) if mode == 'max' else np.mean(self.scores))
        pairs = getattr(self, "_tested_pairs", [])
        if pairs:
            vals = np.array([s for (_, s) in pairs], dtype=float)
            return float(np.max(vals) if mode == 'max' else np.mean(vals))
        return float('-inf')

    def ucb(self, beta: float = 1.6, eps: float = 1e-8, lambda_geo: float = 0.0, vol_scale: float = 1.0, value_scale: float = 1.0) -> float:
        n_eff = self.n_trials
        if self.n_trials > 0:
            mu = float(self.mean_score)
            var = float(self.var_score) if self.n_trials > 1 else float(self.prior_var)
        else:
            mu = 0.0
            var = float(self.prior_var)
        if self.parent is not None and self.parent.n_trials > 0:
            # Decay parent influence as we gather local data
            # n=0 -> w=1.0 (pure parent)
            # n=1 -> w=0.5
            # n=9 -> w=0.1
            w = 1.0 / (1.0 + float(self.n_trials))
            
            p_mu = float(self.parent.mean_score)
            p_var = float(self.parent.var_score) if self.parent.var_score > 0 else float(self.parent.prior_var)
            
            mu = w * p_mu + (1.0 - w) * mu
            var = w * p_var + (1.0 - w) * var
        
        # --- MODIFICA: Decay di Beta con la profondità ---
        # Più scendo, meno voglio esplorare l'incertezza pura.
        # Voglio fidarmi della media (mu).
        # depth 0 -> factor 1.0
        # depth 5 -> factor ~0.66 (was 0.5)
        depth_decay = 1.0 / (1.0 + 0.1 * self.depth)
        beta_eff = beta * depth_decay
        
        if n_eff <= 0: #Calcola l’UCB base: media + margine d’errore. il margine scende con più dati
            base = float(mu + beta_eff * np.sqrt(var + self.prior_var))
        else: #Più ho provato, più la mia barra d’errore si restringe. Metto sempre un pizzico di incertezza di fondo.
            base = float(mu + beta_eff * np.sqrt(var / (n_eff + eps) + self.prior_var))
        
        
        # Log numerical issues to stderr
        if np.isnan(base) or np.isinf(base) or abs(base) > 1e10:
            import sys
            sys.stderr.write(f"⚠️ UCB base anomaly: {base}, mu={mu}, var={var}, n={n_eff}\n")
        
        if lambda_geo <= 0.0:
            return base
        # Volume nel frame locale (prodotto delle larghezze). Se qualche bound non è valido, width=0.
        vol = 1.0
        for i, (lo, hi) in enumerate(self.bounds):
            width = max(hi - lo, 0.0)
            vol *= width
        
        # Normalize volume
        vol_rel = vol / max(1e-12, vol_scale)
        
        # --- FIX: Scale volume effect by dimension ---
        # Volume decays as r^d. We want bonus to decay as r (side length).
        # So take d-th root of volume ratio.
        d = len(self.bounds)
        vol_factor = pow(vol_rel, 1.0 / max(1, d))
        
        # Bonus di esplorazione attenuato dalla densità dei campioni nella cella.
        # FIX: Use logarithmic decay instead of sqrt for more persistent exploration bonus
        decay = float(np.log(n_eff + 2.0))  # log(2)=0.69 when n=0, log(102)=4.6 when n=100
        
        # Bonus geometrico (lambda_geo) penalizzato ancora di più con la profondità
        geo_decay = 1.0 / (1.0 + 0.1 * self.depth)
        lambda_eff = lambda_geo * geo_decay
        
        # Scale bonus by value_scale (e.g. global sigma) to make it relevant
        bonus = lambda_eff * vol_factor * value_scale / decay
        ucb_val = base + bonus
        
        return base + bonus

    def should_split(self,
                     min_trials: int = 5,
                     min_points: int = 10,
                     max_depth: int = 4,
                     min_width: float = 1e-3,
                     gamma: float = 0.02,
                     global_best: Optional[float] = None,
                     gate_ratio: float = 0.2,
                     remaining_budget: Optional[int] = None,
                     ridge_alpha: float = 1e-3) -> str: #decide se e come dividere il cubo
        #Regole: non spacco troppo in profondità, non spacco briciole, e non spacco se non ho abbastanza prove o il guadagno è risibile
        # Reset block reason at start of check
        self.split_block_reason = None
        
        # Use centralized surrogate_min_points if not overridden
        if min_points is None:
            min_points = self.surrogate_min_points
        
        # --- MODIFICA: Budget Gating ---
        # If we don't have enough budget to initialize the children, don't split.
        # Splitting creates 4 children (usually). Each needs min_points to be useful.
        # If remaining < 4 * min_points, we are just fragmenting the space without time to learn.
        if remaining_budget is not None:
            # Cost to reach "surrogate ready" state in all 4 children
            # We can be slightly lenient: require at least enough to get SOME data (e.g. half min_points)
            # But let's be strict to force exploitation of current leaves.
            # RELAXED: Allow splitting if we can cover at least 1 child fully (or others partially).
            init_cost = min_points
            if remaining_budget < init_cost:
                self.split_block_reason = f"low_budget (remaining={remaining_budget} < cost={init_cost})"
                return 'none'
        # -------------------------------

        # stop per profondità/ampiezza
        if max_depth is not None and self.depth >= max_depth:
            self.split_block_reason = f"max_depth (depth={self.depth} >= {max_depth})"
            return 'none'
        #se hai raggiunto la profondità massima o se tutte le dimensioni sono più strette di min_width, non dividere
        widths = [abs(hi - lo) for (lo, hi) in self.bounds]
        if all(w < min_width for w in widths):
            self.split_block_reason = f"min_width (all widths < {min_width})"
            return 'none'

        # --- MODIFICA: Global Best Gating ---
        # Se la cella è molto peggio del best globale, non spittare.
        # Usiamo un margine relativo basato sul valore assoluto del best.
        if global_best is not None and global_best > -1e12: # check validità
            # Calcola soglia: global_best - margin
            # margin = |global_best| * gate_ratio
            # Se global_best è 0, usa un piccolo epsilon
            margin = max(abs(global_best), 1.0) * gate_ratio
            threshold = global_best - margin
            
            # Se il best score della cella è sotto la soglia, blocca.
            if self.best_score < threshold:
                self.split_block_reason = f"global_gate (best={self.best_score:.4f} < thresh={threshold:.4f}, global={global_best:.4f})"
                return 'none'
        # ----------------------------------------------------

        # --- MODIFICA: Controllo Qualità del Modello (Good Fit Guard) ---
        # Se ho un surrogato valido e fitta bene, non splittare.
        # FIX: Only trust R2 if we have enough points relative to dimensions to avoid overfitting
        # Diagonal model has 2*d + 1 params. We want at least d points to trust it somewhat.
        # d_eff = len(self.bounds)
        # if self.surrogate_2d is not None and self.surrogate_2d.get('n', 0) > max(15, 2 * d_eff):
        #     r2 = self.surrogate_2d.get('r2', 0.0)
        #     # Se il modello spiega bene i dati (> 0.90), NON SPLITTARE.
        #     if r2 > 0.90:
        #         self.split_block_reason = f"good_model_fit (r2={r2:.3f} > 0.90)"
        #         return 'none'
        # ----------------------------------------------------------------

        # evita esplosione precoce
        #Anti-esplosione: se non hai abbastanza trial finali e nemmeno abbastanza punti storici, non dividere. Non ho visto abbastanza: non ha senso spaccare la zona ancora.
        n_trials_now = self.n_trials
        hist_len_now = len(self._points_history)
        if n_trials_now < min_trials and hist_len_now < min_points:
            self.split_block_reason = f"insufficient_data (n_trials={n_trials_now}<{min_trials} and history={hist_len_now}<{min_points})"
            return 'none'
        d = len(self.bounds)
        if d == 1:
            return 'binary'
        # prefer quad if we have enough points for PCA/quadratic
        npts = len(self._points_history)
        #Se ho dati sufficienti, faccio un taglio più fine in quattro; se no, taglio in due per stare sul semplice
        split_type = 'quad' if npts >= min_points else 'binary'

        # Curvature gating: don't split in flat regions
        S = self._curvature_scores()
        if S is not None and float(np.max(S)) < 1e-6:
            self.split_block_reason = f"flat_region (max_curvature={float(np.max(S)):.6f} < 1e-6)"
            return 'none'

        # --- MODIFICA: Force Split (Hoarding Prevention) ---
        # If we have accumulated many points, force a split to refine the search space,
        # even if info gain is negative (due to DoF penalty on children).
        # Lowered threshold to min_points + 1 to encourage depth with limited budget.
        # For min_points=8, this splits at 9 instead of 16.
        if n_trials_now >= min_points + 1:
            self.split_block_reason = None
            return split_type
        # ---------------------------------------------------

        # Info-gain: split solo se riduzione varianza residua > gamma
        # Solo se surrogato disponibile e almeno 2 figli
        #Applica un criterio di information gain basato sulla varianza residua del surrogato: richiede un surrogato con almeno 12 punti
        #Faccio una prova mentale: se taglio, i pezzi nuovi mi riducono abbastanza l’incertezza? Se il guadagno è piccolo, non spacco; se è decente, procedo col tipo di taglio deciso prima.
        if self.surrogate_2d is not None and self.surrogate_2d['n'] >= min_points:
            var_parent = float(self.surrogate_2d['sigma2'])
            # Simula split
            if split_type == 'quad':
                children = self._simulate_split4(ridge_alpha=ridge_alpha)
            else:
                children = self._simulate_split2(ridge_alpha=ridge_alpha)
            if children:
                n_total = sum(ch['n'] for ch in children)
                if n_total > 0:
                    var_post = sum((ch['n']/n_total)*ch['var'] for ch in children)
                    delta = var_parent - var_post
                    if delta < gamma:
                        self.split_block_reason = f"info_gain_too_low (delta={delta:.6f} < gamma={gamma})"
                        return 'none'
        
        self.split_block_reason = None  # Split approved
        return split_type

    def _simulate_split2(self, ridge_alpha: float = 1e-3):
        # Simula split2 e fitta surrogato su ciascun figlio
        d = len(self.bounds)
        widths_parent = self._widths()
        # prende il frame PCA locale (R, mu). Calcola il punto di taglio lungo l’asse ax con _quad_cut_along_axis
        ax = int(np.argmax(widths_parent))
        R, mu, _, ok = self._principal_axes()
        R_use = R if ok else (self.R if self.R is not None else np.eye(d))
        mu_use = mu if ok else (self.mu if self.mu is not None else np.full(d, 0.5))
        # Parent bounds expressed in chosen frame
        # FIX: Account for center shift
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        spans_use = (np.abs(M) @ widths_parent)
        
        p_mu = self.mu if self.mu is not None else np.full(d, 0.5)
        delta_mu = p_mu - mu_use
        center_new = R_use.T @ delta_mu
        
        base_bounds = []
        for k in range(d):
            c = center_new[k]
            w = spans_use[k]
            base_bounds.append((c - w/2.0, c + w/2.0))
        cut = self._quad_cut_along_axis(ax, R_use, mu_use, ridge_alpha=ridge_alpha)
        lo, hi = base_bounds[ax]
        cut = float(np.clip(cut, lo + 1e-12, hi - 1e-12))
        # Assegna punti ai due figli
        pairs = getattr(self, "_tested_pairs", [])
        if not pairs:
            return []
        #recupera i punti testati X e i loro punteggi y, poi li proietta in coordinate prime T
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        T = (R_use.T @ (X.T - mu_use.reshape(-1, 1))).T
        mask_left = T[:, ax] < cut
        children = []
        #assegno ogni punto alla metà sinistra o destra.
        for mask in [mask_left, ~mask_left]:
            idx = np.where(mask)[0]
            #se nella metà ci sono due briciole, non provo a disegnare una parabola: guardo solo quanto ballano i voti, o do un valore standard
            if len(idx) < 4:
                children.append({'n': len(idx), 'var': float(np.var(y[idx])) if len(idx) > 1 else 1.0})
                continue
            #se ci sono abbastanza punti: costruisce le feature quadratiche in (PC1, PC2), fitta una ridge locale nel figlio, calcola i residui e la loro varianza come stima d’incertezza del figlio
            #quando ho abbastanza assaggi in quella metà, ci disegno la mia collinetta locale e vedo quanto i punti se ne discostano
            t1 = T[idx, 0]
            t2 = T[idx, 1] if d > 1 else np.zeros_like(t1)
            
            # --- FIX: Scale features for stability ---
            s1 = float(np.std(t1)) if len(t1) > 1 else 1.0
            if s1 < 1e-9: s1 = 1.0
            t1_s = t1 / s1
            
            s2 = float(np.std(t2)) if len(t2) > 1 else 1.0
            if s2 < 1e-9: s2 = 1.0
            t2_s = t2 / s2
            # -----------------------------------------
            
            Phi = np.stack([
                np.ones_like(t1_s), t1_s, t2_s, 0.5 * t1_s ** 2, 0.5 * t2_s ** 2, t1_s * t2_s
            ], axis=1)
            # ridge_alpha = 1e-3 # Now using self.ridge_alpha
            A = Phi.T @ Phi + ridge_alpha * np.eye(6)
            b = Phi.T @ y[idx]
            try:
                w = np.linalg.solve(A, b)
                y_hat = Phi @ w
                resid = y[idx] - y_hat
                Gc = Phi.T @ Phi
                try:
                    A_inv_c = np.linalg.inv(A)
                    df_eff_c = float(np.trace(Gc @ A_inv_c))
                except Exception:
                    df_eff_c = float(min(Phi.shape[0], Phi.shape[1]))
                rss_c = float(np.sum(resid * resid))
                n_c = Phi.shape[0]
                denom_c = max(1.0, n_c - df_eff_c)
                var = float(rss_c / denom_c) if n_c > 1 else 1.0
            except Exception:
                var = 1.0
            children.append({'n': len(idx), 'var': var})
        return children

    def _simulate_split4(self, ridge_alpha: float = 1e-3):
        # Simula split4 e fitta surrogato su ciascun figlio (PC1/PC2)
        d = len(self.bounds)
        widths_parent = self._widths()
        R, mu, _, ok = self._principal_axes()
        R_use = R if ok else (self.R if self.R is not None else np.eye(d))
        mu_use = mu if ok else (self.mu if self.mu is not None else np.full(d, 0.5))
        # Parent bounds in chosen frame
        # FIX: Account for center shift between parent (self.mu) and new frame (mu_use)
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        spans_use = (np.abs(M) @ widths_parent)
        
        p_mu = self.mu if self.mu is not None else np.full(d, 0.5)
        delta_mu = p_mu - mu_use
        center_new = R_use.T @ delta_mu
        
        base_bounds = []
        for k in range(d):
            c = center_new[k]
            w = spans_use[k]
            base_bounds.append((c - w/2.0, c + w/2.0))
        # se la PCA è ok usa PC1 (0) e PC2 (1, o 0 se 1D) come assi di taglio
        if ok:
            ax_i, ax_j = 0, 1 if d > 1 else 0
        else:
            # altrimenti, prendi i due assi più larghi dai bounds (ordine come implementazione legacy)
            top2 = np.argsort(widths_parent)[-2:]
            ax_i, ax_j = int(top2[0]), int(top2[1])
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R_use, mu_use, ridge_alpha=ridge_alpha)
            cut_j = self._quad_cut_along_axis(ax_j, R_use, mu_use, ridge_alpha=ridge_alpha)
        else:
            lo_i, hi_i = base_bounds[ax_i]
            cut_i = 0.5 * (lo_i + hi_i)
            lo_j, hi_j = base_bounds[ax_j]
            cut_j = 0.5 * (lo_j + hi_j)
        # Clipping nei bounds nel frame scelto
        lo_i, hi_i = base_bounds[ax_i]
        lo_j, hi_j = base_bounds[ax_j]
        cut_i = float(np.clip(cut_i, lo_i + 1e-12, hi_i - 1e-12))
        cut_j = float(np.clip(cut_j, lo_j + 1e-12, hi_j - 1e-12))
        pairs = getattr(self, "_tested_pairs", [])
        if not pairs:
            return []
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        T = (R_use.T @ (X.T - mu_use.reshape(-1, 1))).T
        # definisce i due tagli: sinistra/destra lungo ax_i e sotto/sopra lungo ax_j
        left = T[:, ax_i] < cut_i
        bottom = T[:, ax_j] < cut_j
        children = []
        for mask in [left & bottom, (~left) & bottom, left & (~bottom), (~left) & (~bottom)]:
            idx = np.where(mask)[0]
            if len(idx) < 4:
                children.append({'n': len(idx), 'var': float(np.var(y[idx])) if len(idx) > 1 else 1.0})
                continue
            t1 = T[idx, 0]
            t2 = T[idx, 1] if d > 1 else np.zeros_like(t1)
            
            # --- FIX: Scale features for stability ---
            s1 = float(np.std(t1)) if len(t1) > 1 else 1.0
            if s1 < 1e-9: s1 = 1.0
            t1_s = t1 / s1
            
            s2 = float(np.std(t2)) if len(t2) > 1 else 1.0
            if s2 < 1e-9: s2 = 1.0
            t2_s = t2 / s2
            # -----------------------------------------

            Phi = np.stack([
                np.ones_like(t1_s), t1_s, t2_s, 0.5 * t1_s ** 2, 0.5 * t2_s ** 2, t1_s * t2_s
            ], axis=1)
            # ridge_alpha = 1e-3 # Now using self.ridge_alpha
            A = Phi.T @ Phi + ridge_alpha * np.eye(6)
            b = Phi.T @ y[idx]
            try:
                w = np.linalg.solve(A, b)
                y_hat = Phi @ w
                resid = y[idx] - y_hat
                Gc = Phi.T @ Phi
                try:
                    A_inv_c = np.linalg.inv(A)
                    df_eff_c = float(np.trace(Gc @ A_inv_c))
                except Exception:
                    df_eff_c = float(min(Phi.shape[0], Phi.shape[1]))
                rss_c = float(np.sum(resid * resid))
                n_c = Phi.shape[0]
                denom_c = max(1.0, n_c - df_eff_c)
                var = float(rss_c / denom_c) if n_c > 1 else 1.0
            except Exception:
                var = 1.0
            children.append({'n': len(idx), 'var': var})
        return children

    def _principal_axes(self,
                        q_good: float = 0.3,
                        min_points: int = 10,
                        anisotropy_threshold: float = 1.4,
                        depth_min: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (R, mu, eigvals, ok) where R columns are principal axes.
        Weighted PCA on all tested points to reduce self-referential bias.

        Notes:
        - Uses softmax weights of standardized scores so better points weigh more.
        - Mixes a uniform component so every point has non-zero influence.
        - Applies an anisotropy check; if weak or insufficient data, keep current frame.
        """
        # Use centralized surrogate_min_points if not overridden (Removed)
            
        d = len(self.bounds)
        # default: identity around current center in original space
        # current center in original coords using existing frame and bounds
        self._ensure_frame()
        # If mu is not meaningful yet, approximate by avg of seen points or keep current
        points_pairs = getattr(self, "_tested_pairs", [])
        #Se la cella è troppo giovane (depth < depth_min) o ci sono pochi punti (< min_points), niente PCA: ritorna frame corrente (R, mu) e ok=False
        if self.depth < depth_min or len(points_pairs) < min_points:
            R = self.R.copy() if self.R is not None else np.eye(d)
            mu = self.mu.copy() if self.mu is not None else np.full(d, 0.5, dtype=float)
            eigvals = np.ones(d)
            return R, mu, eigvals, False
        pts = np.array([p for (p, s) in points_pairs], dtype=float)
        scs = np.array([s for (p, s) in points_pairs], dtype=float)

        # Weighted PCA on all points: softmax on standardized scores + uniform mix
        tau = float(getattr(self, 'pca_softmax_tau', 0.6))
        mix = float(getattr(self, 'pca_mix_uniform', 0.25))
        w_floor = float(getattr(self, 'pca_weight_floor', 0.02))

        s = scs.copy()
        s -= float(np.median(s))
        s_std = float(np.std(s)) if np.std(s) > 1e-12 else 1.0
        s /= s_std
        w_soft = np.exp(s / max(tau, 1e-6))
        sum_soft = float(np.sum(w_soft))
        if not np.isfinite(sum_soft) or sum_soft <= 0.0:
            w_soft = np.ones_like(w_soft)
            sum_soft = float(np.sum(w_soft))
        w_soft = w_soft / sum_soft
        w_uniform = np.full_like(w_soft, 1.0 / len(w_soft))
        w = (1.0 - mix) * w_soft + mix * w_uniform
        # apply additional floor per point (fractional mass)
        w = np.maximum(w, (w_floor / max(1, len(w))))
        w = w / float(np.sum(w))

        # weighted mean and covariance
        mu = np.sum(w[:, None] * pts, axis=0)
        Z = pts - mu
        C = (Z * w[:, None]).T @ Z
        C = C / max(1e-12, float(np.sum(w))) + 1e-9 * np.eye(d)

        # eigh returns ascending; we want descending
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        
        # SANITY CHECK: PCA condition number
        if self._debug_logger:
            cond = float(evals[0] / max(evals[-1], 1e-12))
            if cond > 1e6:
                self._debug_logger(f"[WARNING] PCA condition number very high: {cond:.2e} (evals: {evals})")
        
        # SANITY CHECK: PCA reconstruction error
        if self._debug_logger:
            Z_recon = (evecs @ evecs.T) @ Z.T
            recon_err = np.linalg.norm(Z.T - Z_recon, 'fro') / max(np.linalg.norm(Z, 'fro'), 1e-12)
            if recon_err > 0.1:
                self._debug_logger(f"[WARNING] PCA reconstruction error high: {recon_err:.4f}")
        
        # anisotropy check
        ratio = float(evals[0] / max(np.mean(evals[1:]) if d > 1 else evals[0], 1e-9))
        ok = bool(ratio >= anisotropy_threshold)
        if not ok:
            R = self.R.copy() if self.R is not None else np.eye(d)
            mu0 = self.mu.copy() if self.mu is not None else np.full(d, 0.5, dtype=float)
            return R, mu0, np.maximum(evals, 1e-9), False
        return evecs, mu, np.maximum(evals, 1e-9), True

    def _quad_cut_along_axis(self,
                              axis_idx: int,
                              R: np.ndarray,
                              mu: np.ndarray,
                              ridge_alpha: float = 1e-3) -> float:
        """Choose a cut point along given prime axis using 1D quadratic fit over projections.
        Falls back to midpoint if fit is poor or data insufficient.
        Returns t_cut in prime coords, clipped to [lo, hi] for that axis.
        Uses self.ridge_alpha for regularization.
        """
        #Guardiamo i risultati lungo una linea. Disegno una parabola e taglio nel punto dove la curva sembra “scendere di più” (il minimo). Se la parabola non viene bene, taglio a metà o vicino ai punti migliori.
        pairs = getattr(self, "_tested_pairs", [])
        d = len(self.bounds)
        if len(pairs) < 6:
            lo, hi = self.bounds[axis_idx]
            return 0.5 * (lo + hi)
        #Provo a disegnare la parabola. Se i conti non tornano, taglio in mezzo.
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        # project to prime coordinates using provided (R, mu)
        T = (R.T @ (X.T - mu.reshape(-1, 1))).T  # shape (n, d)
        t = T[:, axis_idx]
        
        # --- FIX: Scale t to avoid numerical issues with ridge_alpha ---
        t_std = float(np.std(t))
        if t_std < 1e-9: t_std = 1.0
        t_scaled = t / t_std
        # ---------------------------------------------------------------

        # design matrix for y ~ a + b t + 0.5 c t^2
        Phi = np.stack([np.ones_like(t_scaled), t_scaled, 0.5 * t_scaled * t_scaled], axis=1)
        # ridge solve: (Phi^T Phi + alpha I) w = Phi^T y
        A = Phi.T @ Phi
        A += ridge_alpha * np.eye(3)
        b = Phi.T @ y
        try:
            w = np.linalg.solve(A, b)
            a_hat, b_hat, c_hat = float(w[0]), float(w[1]), float(w[2])
        except Exception:
            lo, hi = self.bounds[axis_idx]
            return 0.5 * (lo + hi)
        # minimum of quadratic: t* = -b/c (if convex: c > 0)
        #Se la parabola ha una valle, taglio nella valle; se no, taglio dove si concentrano i bocconi più buoni.
        if c_hat > 1e-8:
            t_star_scaled = -b_hat / c_hat
            t_star = t_star_scaled * t_std # Unscale
            lo, hi = self.bounds[axis_idx]
            t_cut = float(np.clip(t_star, lo, hi))
            return t_cut
        # fallback: median of good projections
        # pick top 40% by y
        k = max(5, int(0.4 * len(t)))
        idx = np.argsort(y)[::-1][:k]
        t_med = float(np.median(t[idx]))
        lo, hi = self.bounds[axis_idx]
        return float(np.clip(t_med, lo, hi))

    def _axis_range_in_frame(self, axis_idx: int, R: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
        """Restituisce (lo, hi) lungo l'asse axis_idx nel frame (R, mu).
        Strategia pragmatica:
        - Se ho punti testati, uso min/max delle loro proiezioni T[:, axis_idx].
        - Se non ho punti, ricado ai bounds attuali del cubo per quell'asse.
        Nota: i bounds del cubo sono già nel frame "prime" del cubo; qui assumiamo che (R, mu)
        sia il frame corrente per proiezioni coerenti con i tagli simulati.
        """
        pairs = getattr(self, "_tested_pairs", [])
        if pairs:
            X = np.array([p for (p, _s) in pairs], dtype=float)
            T = (R.T @ (X.T - mu.reshape(-1, 1))).T
            t = T[:, axis_idx]
            return float(np.min(t)), float(np.max(t))
        # fallback: usare i bounds locali del cubo per quell'asse
        lo, hi = self.bounds[axis_idx]
        return float(lo), float(hi)

    def split2(self, axis: Optional[int] = None, ridge_alpha: float = 1e-3) -> List["QuadCube"]:
        # binary split along one prime axis (largest width by default)
        d = len(self.bounds)
        self._ensure_frame()
        widths_parent = self._widths()
        # Decide axis: prefer curvature if available, otherwise use PCA frame spans when reliable
        R_loc, mu_loc, _, ok = self._principal_axes()
        R_use = (R_loc if ok else self.R)
        mu_use = (mu_loc if ok else self.mu)
        if R_use is None:
            R_use = np.eye(d)
        if mu_use is None:
            mu_use = np.full(d, 0.5, dtype=float)

        # Use projected spans only to choose axis when PCA is ok
        if ok and R_use is not None and self.R is not None:
            M = R_use.T @ self.R
            spans_use = (np.abs(M) @ widths_parent)  # full widths along R_use axes
        else:
            # Fallback: no transformation, use parent widths directly
            spans_use = widths_parent.copy()
        if axis is not None:
            ax = int(axis)
            # If axis is forced, we assume it refers to the current frame (self.R)
            # So we force R_use to be self.R
            R_use = self.R if self.R is not None else np.eye(d)
            mu_use = self.mu if self.mu is not None else np.full(d, 0.5, dtype=float)
        else:
            S = self._curvature_scores()
            if S is not None:
                # S is computed on Surrogate Axes (R_s)
                # We need to map this back to Prime Axes (self.R) for split2
                # because split2 creates children with self.R
                ax_s = int(np.argmax(S))
                
                # Get Surrogate Frame
                s = self.surrogate_2d
                R_s = np.asarray(s['R'], float)
                
                # Compute projection of Prime axes onto Surrogate axis ax_s
                # M = R_s.T @ R_c
                # We want column j of M such that |M[ax_s, j]| is max
                R_c = self.R if self.R is not None else np.eye(d)
                M = R_s.T @ R_c
                
                # Row ax_s of M represents the surrogate axis in terms of prime axes
                # We pick the prime axis j that contributes most
                ax = int(np.argmax(np.abs(M[ax_s, :])))
                
                # Force usage of Prime Frame for the cut
                R_use = R_c
                mu_use = self.mu if self.mu is not None else np.full(d, 0.5, dtype=float)
                
            elif ok:
                # If PCA is ok, spans_use is in PCA frame.
                # We must map the best PCA axis back to Prime Frame.
                ax_pca = int(np.argmax(spans_use))
                M = R_use.T @ (self.R if self.R is not None else np.eye(d))
                ax = int(np.argmax(np.abs(M[ax_pca, :])))
                
                # Revert to Prime Frame
                R_use = self.R if self.R is not None else np.eye(d)
                mu_use = self.mu if self.mu is not None else np.full(d, 0.5, dtype=float)
            else:
                ax = int(np.argmax(widths_parent))
                R_use = self.R if self.R is not None else np.eye(d)
                mu_use = self.mu if self.mu is not None else np.full(d, 0.5, dtype=float)

        # compute quadratic cut along chosen axis in chosen frame (for position),
        # but apply the split to the parent's prime bounds to preserve exact halving semantics in tests
        cut = self._quad_cut_along_axis(ax, R_use, mu_use, ridge_alpha=ridge_alpha)
        lo, hi = self.bounds[ax]
        cut = float(np.clip(cut, lo + 1e-12, hi - 1e-12))

        # child bounds in parent prime frame before re-centering
        nb_left = list(self.bounds)
        nb_right = list(self.bounds)
        nb_left[ax] = (lo, cut)
        nb_right[ax] = (cut, hi)

        # compute child centers and widths in parent prime frame
        center_left_prime = np.array([(a + b) * 0.5 for (a, b) in nb_left], dtype=float)
        center_right_prime = np.array([(a + b) * 0.5 for (a, b) in nb_right], dtype=float)
        w_left = np.array([b - a for (a, b) in nb_left], dtype=float)
        w_right = np.array([b - a for (a, b) in nb_right], dtype=float)

        # build children: re-center prime boxes around 0 and set mu accordingly
        c1 = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in w_left], parent=self)
        c2 = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in w_right], parent=self)
        for (ch, ctr_prime) in ((c1, center_left_prime), (c2, center_right_prime)):
            # Children keep parent's frame since bounds/centers are defined in parent prime frame
            ch.R = self.R.copy() if self.R is not None else np.eye(d)
            ch.mu = (self.mu + (ch.R @ ctr_prime)).astype(float)
            ch.prior_var = float(self.prior_var)
            ch.depth = self.depth + 1
            ch._tested_points = []
            # Inherit surrogate hyperparameters (removed)
            for attr, default in [
                ('pca_softmax_tau', 0.6),
                ('pca_mix_uniform', 0.25),
                ('pca_weight_floor', 0.02),
            ]:
                setattr(ch, attr, getattr(self, attr, default))

        # redistribute points/pairs according to cut in chosen frame
        points = np.array(self._points_history) if self._points_history else np.empty((0, len(self.bounds)))
        if points.size > 0:
            T = (R_use.T @ (points.T - mu_use.reshape(-1,1))).T
            mask_left = T[:, ax] < cut
            for p, m in zip(points, mask_left):
                (c1._tested_points if m else c2._tested_points).append(np.array(p, dtype=float))
        pairs = getattr(self, "_tested_pairs", [])
        c1._tested_pairs, c2._tested_pairs = [], []
        for (pt, s) in pairs:
            t = float((R_use.T @ (pt - mu_use))[ax])
            (c1._tested_pairs if t < cut else c2._tested_pairs).append((np.array(pt, dtype=float), float(s)))
        
        # Propagate debug logger to children
        c1._debug_logger = self._debug_logger
        c2._debug_logger = self._debug_logger
        
        self.children = [c1, c2]
        return self.children

    def split4(self, ridge_alpha: float = 1e-3) -> List["QuadCube"]:
        # 4-way split using PCA local axes; cut-points from quadratic 1D fits (fallback to midpoints)
        d = len(self.bounds)
        if d == 1:
            return self.split2(axis=0)
        self._ensure_frame()
        widths_parent = self._widths()
        
        # Decide frame: prefer surrogate frame if curvature is available
        S = self._curvature_scores()
        using_surrogate_frame = False
        
        if S is not None and self.surrogate_2d is not None:
            s = self.surrogate_2d
            R_use = np.asarray(s['R'], float)
            mu_use = np.asarray(s['mu'], float)
            ok = True
            using_surrogate_frame = True
        else:
            # principal axes and local center
            R_loc, mu_loc, _, ok = self._principal_axes()
            R_use = (R_loc if ok else self.R)
            mu_use = (mu_loc if ok else self.mu)
            
        if R_use is None:
            R_use = np.eye(d)
        if mu_use is None:
            mu_use = np.full(d, 0.5, dtype=float)
        
        # Parent box expressed in chosen frame
        # KEY FIX: When ok=False, R_use == self.R, so M = I and spans_use = widths_parent
        # This ensures frame consistency: axes chosen and bounds are in the SAME frame
        if ok and R_use is not None and self.R is not None:
            M = R_use.T @ self.R
            A = np.abs(M)
            spans_use = A @ widths_parent
        else:
            # Fallback: no transformation, use parent widths directly
            spans_use = widths_parent.copy()
            
        # FIX: Account for center shift
        p_mu = self.mu if self.mu is not None else np.full(d, 0.5)
        delta_mu = p_mu - mu_use
        center_new = R_use.T @ delta_mu
        
        base_bounds = []
        for k in range(d):
            c = center_new[k]
            w = spans_use[k]
            base_bounds.append((c - w/2.0, c + w/2.0))
        
        # Use curvature-driven split criterion if available
        if S is not None and using_surrogate_frame:
            # Choose top-2 axes by curvature score
            order = np.argsort(S)[::-1]  # descending order
            ax_i = int(order[0])
            ax_j = int(order[1] if len(order) > 1 else order[0])
        else:
            # Fallback: choose two axes consistently
            # If PCA ok: use PC1 (0) and PC2 (1) in PCA frame, then map to parent axes
            # If PCA failed: use widest two axes from parent frame
            if ok:
                # Use spans in the NEW frame (R_use) to pick widest axes
                # argsort is ascending, so top2[1] is the max, top2[0] is second max
                top2 = np.argsort(spans_use)[-2:]
                ax_i, ax_j = int(top2[1]), int(top2[0])
            else:
                # Use widest two axes in parent frame
                top2 = np.argsort(widths_parent)[-2:]
                ax_i, ax_j = int(top2[1]), int(top2[0])
        
        # compute cutpoints: use PCA frame for quad fit
        # --- FIX: Use smart cuts from PCA frame ---
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R_use, mu_use, ridge_alpha=ridge_alpha)
            cut_j = self._quad_cut_along_axis(ax_j, R_use, mu_use, ridge_alpha=ridge_alpha)
        else:
            # Fallback to midpoint in parent frame
            lo_i, hi_i = base_bounds[ax_i]
            cut_i = 0.5 * (lo_i + hi_i)
            lo_j, hi_j = base_bounds[ax_j]
            cut_j = 0.5 * (lo_j + hi_j)
        
        # Clip cuts to bounds
        lo_i, hi_i = base_bounds[ax_i]
        lo_j, hi_j = base_bounds[ax_j]
        cut_i = float(np.clip(cut_i, lo_i + 1e-12, hi_i - 1e-12))
        cut_j = float(np.clip(cut_j, lo_j + 1e-12, hi_j - 1e-12))
        
        # Apply split in PCA FRAME (OBB)
        def make_bounds(quadrant: TypingTuple[bool, bool]) -> List[Tuple[float, float]]:
            nb = list(base_bounds)
            nb[ax_i] = (lo_i, cut_i) if quadrant[0] else (cut_i, hi_i)
            nb[ax_j] = (lo_j, cut_j) if quadrant[1] else (cut_j, hi_j)
            return nb
            
        b_q1 = make_bounds((True, True))
        b_q2 = make_bounds((False, True))
        b_q3 = make_bounds((True, False))
        b_q4 = make_bounds((False, False))
        
        # centers and widths for children in PCA FRAME
        centers_prime = [np.array([(a + b) * 0.5 for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        widths_children = [np.array([b - a for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        
        # instantiate children with recentered prime boxes
        children: List[QuadCube] = []
        for ctr_p, wch in zip(centers_prime, widths_children):
            ch = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in wch], parent=self)
            # --- FIX: Adopt PCA Frame (OBB) ---
            ch.R = R_use.copy()
            # Map child center from PCA frame to Global
            ch.mu = (mu_use + (ch.R @ ctr_p)).astype(float)
            # ----------------------------------
            ch.prior_var = float(self.prior_var)
            # q_threshold removed
            ch.depth = self.depth + 1
            ch._tested_points = []
            # Inherit surrogate hyperparameters (removed)
            # propagate PCA weighting knobs
            for attr, default in [
                ('pca_softmax_tau', 0.6),
                ('pca_mix_uniform', 0.25),
                ('pca_weight_floor', 0.02),
            ]:
                setattr(ch, attr, getattr(self, attr, default))
            children.append(ch)
        # redistribute historical points/pairs to children by quadrant in parent frame
        points = np.array(self._points_history) if self._points_history else np.empty((0, len(self.bounds)))
        if points.size > 0:
            T = (R_use.T @ (points.T - mu_use.reshape(-1, 1))).T
            left = T[:, ax_i] < cut_i
            bottom = T[:, ax_j] < cut_j
            for idx_pt, p in enumerate(points):
                q = (bool(left[idx_pt]), bool(bottom[idx_pt]))
                if q == (True, True):
                    children[0]._tested_points.append(np.array(p, dtype=float))
                elif q == (False, True):
                    children[1]._tested_points.append(np.array(p, dtype=float))
                elif q == (True, False):
                    children[2]._tested_points.append(np.array(p, dtype=float))
                else:
                    children[3]._tested_points.append(np.array(p, dtype=float))
        pairs = getattr(self, "_tested_pairs", [])
        for ch in children:
            ch._tested_pairs = []
        for (pt, s) in pairs:
            t = (R_use.T @ (pt - mu_use))
            q = (bool(t[ax_i] < cut_i), bool(t[ax_j] < cut_j))
            if q == (True, True):
                children[0]._tested_pairs.append((np.array(pt, dtype=float), float(s)))
            elif q == (False, True):
                children[1]._tested_pairs.append((np.array(pt, dtype=float), float(s)))
            elif q == (True, False):
                children[2]._tested_pairs.append((np.array(pt, dtype=float), float(s)))
            else:
                children[3]._tested_pairs.append((np.array(pt, dtype=float), float(s)))
        
        # Propagate debug logger to children
        for ch in children:
            ch._debug_logger = self._debug_logger
        
        self.children = children
        return self.children


class QuadHPO:
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        # Removed all static parameters
        full_epochs: int = 50,
        maximize: bool = True,
        param_space: Optional[ParamSpace] = None,
        rng_seed: Optional[int] = None,
        log_path: Optional[str] = None,
        on_best: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug_log: bool = True,
    ) -> None:
        # ============================================================
        # ADAPTIVE CONFIGURATION (Derived from Problem Context)
        # ============================================================
        self.dim = len(bounds)
        self.config = AdaptiveConfig(dim=self.dim)
        # Initial adaptation with default budget
        self.config.adapt(self.dim, 100)
        
        # Legacy aliases (mapped to config)
        self.min_points = self.config.surrogate_min_points
        self.min_trials = self.config.min_points_split
        self.max_depth = self.config.max_depth
        self.min_width = self.config.min_width
        self.budget = 100
        
        # 6. Pruning & Stagnation
        self.min_leaves = max(2, self.dim)  # Keep at least D leaves active? Or just constant 4.
        self.stale_steps_max = 2 * self.min_points # Allow 2x min points of stagnation
        self.prune_grace_period = max(3, self.min_points // 2)
        
        # ============================================================
        # STANDARD SETUP
        # ============================================================
        self.full_epochs = int(full_epochs)
        self.maximize = bool(maximize)
        self.sign = 1.0 if self.maximize else -1.0
        self.enable_debug_log = debug_log
        
        # PCA/Sampling constants (Robust Defaults)
        self.pca_q_good = 0.3
        self.pca_min_points = self.min_points
        self.anisotropy_threshold = 1.4
        self.depth_min_for_pca = 1
        self.line_search_prob = 0.10
        self.gauss_scale = 0.4
        
        # Surrogate mode: always 'auto'
        self.surr_mode = 'auto'
        self._auto_base = {
            'sigma2_max': 6.0,
            'r2_min': 0.55,
            'ei_min': 0.005,
            'margin': 0.2,
            'rho_max': 1.9,
            'rho_pen': 0.08,
            'elite_frac': 0.55,
            'elite_top_k': 8,
            'candidates': 64,
        }
        
        # ============================================================
        # STATE AND INFRASTRUCTURE
        # ============================================================
        self.root = QuadCube(list(bounds))
        self.leaf_cubes: List[QuadCube] = [self.root]
        self.param_space = param_space
        self.log_path = log_path
        self.on_best = on_best
        self.on_trial = on_trial
        self.best_score_global = -np.inf
        self.best_x_candidate: Optional[List[float]] = None
        self.best_x_real: Optional[List[Any]] = None
        self.trial_id: int = 0
        self._preferred_leaf: Optional[QuadCube] = None
        self.objective_calls: int = 0
        self.obj_calls = self.objective_calls
        
        self.cube_select_mode = 'ucb'
        
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # ============================================================
        # LOGGING
        # ============================================================
        self.total_trials: int = 0
        self.s_final_all: List[float] = []
        self.splits_count: int = 0
        self.split_trials: List[int] = []
        self.split_checks: int = 0
        self.split_reasons: List[str] = []
        self.prunes_count: int = 0
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/mnt/workspace/algo_logs" if os.path.exists("/mnt/workspace/algo_logs") else "."
        self.debug_log_path = os.path.join(log_dir, f"quadhpo_subspace_debug_{timestamp}.log")
        self._init_debug_log()
        
        if self.log_path and not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trial', 'cube_id', 'x_candidate', 'x_real', 's_final', 'best_score_global', 'n_leaves'
                ])
        
        # Root initialization
        d = len(bounds)
        root_lo = np.array([lo for (lo, hi) in bounds], dtype=float)
        root_hi = np.array([hi for (lo, hi) in bounds], dtype=float)
        root_mu = (root_lo + root_hi) * 0.5
        root_w = (root_hi - root_lo)
        self.root.R = np.eye(d)
        self.root.mu = root_mu
        self.root.bounds = [(-wi / 2.0, wi / 2.0) for wi in root_w]
        # self.root.surrogate_min_points = self.min_points (Removed)
        
        self.root.pca_softmax_tau = 0.6
        self.root.pca_mix_uniform = self.config.pca_mix
        self.root.pca_weight_floor = 0.02
        
        # Set debug logger callback on root cube
        self.root._debug_logger = self._debug_log
        self._debug_logger = self._debug_log  # Alias for consistency
        
        # Store global bounds for clipping
        self.bounds = bounds

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)

    def _init_debug_log(self) -> None:
        """Initialize debug log file with header."""
        if not self.enable_debug_log:
            return
        try:
            with open(self.debug_log_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("QuadHPO ADAPTIVE Debug Log\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Configuration (Derived):\n")
                f.write(f"  dim: {self.dim}\n")
                f.write(f"  min_points: {self.min_points}\n")
                f.write(f"  min_trials: {self.min_trials}\n")
                f.write(f"  max_depth: {self.max_depth}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        except Exception as e:
            pass  # Silently fail if can't write debug log
    
    def _debug_log(self, message: str) -> None:
        """Write message to debug log file."""
        if not self.enable_debug_log:
            return
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(message + "\n")
        except Exception:
            pass  # Silently fail if can't write debug log

    def _log(self, row: List[Any]) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass

    def _sample_biased_in(self, cube: QuadCube, alpha: float = 0.4, top_k: int = 5) -> np.ndarray:
        # In auto mode, adapt surrogate knobs to current context before sampling
        if getattr(self, 'surr_mode', 'auto') == 'auto':
            self._auto_tune_surrogate(cube)
        # Acquisition-based sampling using local surrogate if available, fallback to previous logic
        pairs = getattr(cube, "_tested_pairs", [])
        d = len(cube.bounds)
        # exploit around good historical points in original coords
        if pairs and np.random.rand() < alpha:
            pairs_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)
            pick = pairs_sorted[:min(top_k, len(pairs_sorted))]
            center, _ = pick[np.random.randint(len(pick))]
            
            # --- FIX: Scale perturbation to cube size ---
            # Old: x = center + 0.02 * randn (Too big for small cubes!)
            # New: Perturb in prime frame relative to bounds
            t_center = cube.to_prime(np.array(center, dtype=float))
            t_perturb = np.zeros(d)
            for j in range(d):
                lo, hi = cube.bounds[j]
                width = abs(hi - lo)
                # Perturb by 10% of width, clamped to avoid collapse
                scale = max(width * 0.1, 1e-6)
                val = t_center[j] + np.random.normal(0.0, scale)
                # Clip to CUBE bounds (stay in subspace)
                t_perturb[j] = np.clip(val, lo, hi) - t_center[j]
            
            x = cube.to_original(t_center + t_perturb)
            # --------------------------------------------
            
            # Clip to global bounds
            for j in range(d):
                lo, hi = self.bounds[j]
                x[j] = float(np.clip(x[j], lo, hi))
            self._debug_log(f"[Trial {self.trial_id}] Sampling: Biased Random (Scaled)")
            return x

        # Use surrogate-based acquisition if available and reliable enough
        s2d = cube.surrogate_2d
        if s2d is not None:
            cond_n = s2d.get('n', 0) >= self.config.surrogate_min_points
            cond_pca = s2d.get('pca_ok', True)
            cond_sigma = s2d.get('sigma2', 1.0) <= float(self.surr_sigma2_max)
            cond_r2 = s2d.get('r2', 0.0) >= float(self.surr_r2_min)
            
            # RELAXATION: If model is very good (R2 > 0.9), ignore PCA/Sigma constraints
            is_excellent = s2d.get('r2', 0.0) > 0.9
            
            if not (cond_n and (cond_pca or is_excellent) and (cond_sigma or is_excellent) and cond_r2):
                self._debug_log(f"[Trial {self.trial_id}] Surrogate SKIP: n={s2d.get('n',0)}>={self.config.surrogate_min_points}({cond_n}), "
                               f"pca={cond_pca}, sigma2={s2d.get('sigma2',0):.2f}<={self.surr_sigma2_max:.2f}({cond_sigma}), "
                               f"r2={s2d.get('r2',0):.3f}>={self.surr_r2_min:.3f}({cond_r2}) [Exc={is_excellent}]")

        if (
            s2d is not None
            and s2d.get('n', 0) >= self.config.surrogate_min_points
            and s2d.get('r2', 0.0) >= float(self.surr_r2_min)
            and (
                (s2d.get('pca_ok', True) and s2d.get('sigma2', 1.0) <= float(self.surr_sigma2_max))
                or s2d.get('r2', 0.0) > 0.9  # Bypass if excellent
            )
        ):
            stype = s2d.get('type', 'full2d')
            # Evaluate surrogate in its own PCA frame, but sample candidates inside the current cube.
            M = int(self.surr_candidates)
            R_s = s2d['R']
            mu_s = s2d['mu']
            # cube local frame
            cube._ensure_frame()
            R_c = cube.R
            mu_c = cube.mu
            bounds = cube.bounds
            d = len(bounds)
            # Build training projections in surrogate frame for trust region + elite centers
            pairs_np = getattr(cube, "_tested_pairs", [])
            X_tr = np.array([p for (p, s) in pairs_np], dtype=float) if pairs_np else np.empty((0, d))
            y_tr = np.array([s for (p, s) in pairs_np], dtype=float) if pairs_np else np.empty((0,))
            T_tr = ((R_s.T @ (X_tr.T - mu_s.reshape(-1, 1))).T) if X_tr.size > 0 else np.empty((0, d))
            
            # Trust Region Covariance
            # If diag, use full D dims for trust region? Or just 2?
            # Let's use 2 dims for trust region check to be safe/consistent, or D if diag?
            # If diag, we trust all dims.
            tr_dims = d if stype == 'diag' else 2
            T_cov = T_tr[:, :tr_dims] if T_tr.size > 0 else np.empty((0, tr_dims))
            
            if T_cov.shape[0] >= tr_dims + 2:
                C = np.cov(T_cov.T) if T_cov.shape[0] > 1 else np.eye(tr_dims)
            else:
                C = np.eye(tr_dims)
            # regularize
            C = np.asarray(C, dtype=float) + 1e-6 * np.eye(tr_dims)
            try:
                C_inv = np.linalg.inv(C)
            except Exception:
                C_inv = np.linalg.pinv(C)
            rho2_max = float(self.surr_mahalanobis_max ** 2)
            
            # Elite centers
            elite_idx = []
            if X_tr.size > 0:
                order = np.argsort(y_tr)[::-1]
                elite_idx = order[: max(1, min(int(self.surr_elite_top_k), len(order)))]
            elite_centers_t = T_tr[elite_idx, :] if len(elite_idx) > 0 else np.empty((0, d))
            
            candidates_u: List[np.ndarray] = []  # candidates in cube-prime coords
            # trust-region std for sampling inside cube
            sigma2_cur = float(cube.surrogate_2d.get('sigma2', 1.0))
            surr_std_scale = 0.15 if sigma2_cur <= float(self.surr_sigma2_max) * 0.5 else 0.3
            M_elite = int(max(0, min(M, round(M * float(getattr(self, 'surr_elite_frac', 0.0))))))
            M_rand = int(max(0, M - M_elite))
            
            # Random trust-region sampling within cube bounds (in cube-prime coords)
            for _ in range(M_rand):
                u = np.zeros(d, dtype=float)
                for j in range(d):
                    lo, hi = bounds[j]
                    if hi < lo:  # correzione robustezza se bounds invertiti
                        lo, hi = hi, lo
                    half = (hi - lo) * 0.5
                    std = max(0.0, 0.1 * half)
                    if not np.isfinite(std) or std <= 0.0:
                        std = max(1e-12, abs(std))
                    val = np.random.normal(0.0, std)
                    u[j] = float(np.clip(val, lo, hi))
                candidates_u.append(u)
                
            # Elite sampling around top surrogate-frame historical points
            if M_elite > 0 and elite_centers_t.size > 0:
                # derive local scales from cube spans
                stds = []
                for j in range(d):
                    span = (bounds[j][1] - bounds[j][0])
                    s = 0.2 * span
                    if not np.isfinite(s) or s <= 0.0: s = 1e-12
                    stds.append(s)
                
                for _ in range(M_elite):
                    c_idx = int(np.random.randint(elite_centers_t.shape[0]))
                    base_t = elite_centers_t[c_idx].copy()
                    
                    # Perturb
                    t = np.zeros(d, dtype=float)
                    # If full2d, only perturb first 2 dims significantly?
                    # No, perturb all dims to explore!
                    for j in range(d):
                        t[j] = float(base_t[j] + np.random.normal(0.0, stds[j]))
                        
                    # map t (surrogate frame) -> original -> cube-prime u
                    xg = (mu_s + R_s @ t).astype(float)
                    u = (R_c.T @ (xg - mu_c)).astype(float)
                    for j in range(d):
                        lo, hi = bounds[j]
                        u[j] = float(np.clip(u[j], lo, hi))
                    candidates_u.append(u)
                    
            # Deterministic stationary point from surrogate (in surrogate frame) -> map to cube and clip
            try:
                w = np.asarray(cube.surrogate_2d['w'], dtype=float).reshape(-1)
                t_opt = None
                
                if stype == 'diag':
                    # w = [c, b1...bD, a1...aD]
                    b = w[1:d+1]
                    a = w[d+1:]
                    t_opt = np.zeros(d)
                    for i in range(d):
                        if abs(a[i]) > 1e-9:
                            t_opt[i] = -b[i] / (2.0 * a[i])
                        else:
                            t_opt[i] = 0.0
                else:
                    # Full 2D
                    c1, c2 = float(w[1]), float(w[2])
                    c3, c4, c5 = float(w[3]), float(w[4]), float(w[5])
                    H = np.array([[c3, c5], [c5, c4]], dtype=float)
                    g = np.array([c1, c2], dtype=float)
                    if np.all(np.isfinite(H)) and np.linalg.cond(H) < 1e6:
                        t_opt_2d = -np.linalg.solve(H, g)
                        t_opt = np.zeros(d)
                        t_opt[0] = t_opt_2d[0]
                        if d > 1: t_opt[1] = t_opt_2d[1]

                if t_opt is not None:
                    # lift t_opt to original space then back to cube and clip into bounds
                    x_opt = (mu_s + R_s @ t_opt).astype(float)
                    u_opt = (R_c.T @ (x_opt - mu_c)).astype(float)
                    # clip to cube bounds
                    for j in range(d):
                        lo, hi = bounds[j]
                        u_opt[j] = float(np.clip(u_opt[j], lo, hi))
                    candidates_u.append(u_opt)
            except Exception:
                pass
                
            # Small cross grid around origin in surrogate frame -> map to cube and clip
            # ... (omitted for brevity, less critical if we have elite sampling)

            # Evaluate EI for each candidate: map cube-prime -> original -> surrogate-prime
            local_best = float(cube.best_score if hasattr(cube, 'best_score') else -np.inf)
            ei_vals: List[float] = []
            stats_cache: List[TypingTuple[float, float, float, float, float, np.ndarray]] = []  # (yhat, sigma, ei_raw, ei_eff, rho, x)
            any_inside = False
            for u in candidates_u:
                x = (mu_c + R_c @ u).astype(float)
                t = (R_s.T @ (x - mu_s)).astype(float)
                
                # Trust region filter
                t_check = t[:tr_dims]
                try:
                    rho2 = float(t_check.T @ (C_inv @ t_check))
                except Exception:
                    rho2 = float(np.sum(t_check * t_check))
                    
                if not np.isfinite(rho2) or rho2 > rho2_max:
                    # too far from training cloud in surrogate frame, skip
                    continue
                any_inside = True
                y_hat, sigma = cube.predict_surrogate(t)
                sigma = float(max(sigma, 1e-9))
                diff = float(y_hat - local_best)
                z = diff / sigma
                pdf = float((1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * z * z))
                cdf = float(0.5 * (1.0 + math.erf(z / np.sqrt(2.0))))
                ei_raw = diff * cdf + sigma * pdf
                # distance-aware penalization of EI inside trust region
                c_pen = float(getattr(self, 'surr_ei_rho2_penalty', 0.5))
                ei_eff = float(ei_raw / (1.0 + c_pen * max(rho2, 0.0)))
                ei_vals.append(ei_eff)
                stats_cache.append((float(y_hat), float(sigma), float(ei_raw), float(ei_eff), float(np.sqrt(max(rho2, 0.0))), x))
            # If no candidate passed trust-region, fallback
            if not any_inside or not ei_vals:
                self._debug_log(f"[Trial {self.trial_id}] Surrogate REJECTED: No candidates (Inside={any_inside}, EIVals={len(ei_vals)})")
                pass
            if ei_vals:
                best_idx = int(np.argmax(ei_vals))
                y_hat_best, sigma_best, ei_raw_best, ei_best, rho_best, x_best = stats_cache[best_idx]
                # Accept if EI passes threshold and optional margin gate
                margin = float(getattr(self, 'surr_accept_margin_sigma', -1.0))
                ei_min = float(getattr(self, 'surr_ei_min', 0.0))
                yhat_gate_ok = True if margin < 0 else (y_hat_best >= local_best - margin * sigma_best)
                
                if (ei_best >= ei_min) and yhat_gate_ok:
                    self._debug_log(f"[Trial {self.trial_id}] Sampling: Surrogate (EI={ei_best:.4f})")
                    # Clip to global bounds
                    for j in range(d):
                        lo, hi = self.bounds[j]
                        x_best[j] = float(np.clip(x_best[j], lo, hi))
                    return x_best
                else:
                    msg = f"[Trial {self.trial_id}] Surrogate REJECTED: EI={ei_best} Min={ei_min} Gate={yhat_gate_ok}"
                    self._debug_log(msg)
                    # print(msg) # Uncomment for console debug
            # If we reach here without returning from the `if ei_vals:` block above,
            # we fall back to the geometric sampling below.

        # else: fallback to previous geometric sampling
        self._debug_log(f"[Trial {self.trial_id}] Sampling: Geometric Fallback (Surrogate not ready or rejected)")
        
        # If very few points, prefer uniform to avoid center bias early on
        if len(cube._points_history) < 5:
             self._debug_log(f"[Trial {self.trial_id}] Sampling: Uniform (Low History)")
             x = cube.sample_uniform()
             # Clip to global bounds
             for j in range(d):
                 lo, hi = self.bounds[j]
                 x[j] = float(np.clip(x[j], lo, hi))
             return x

        cube._ensure_frame()
        R_loc, mu_loc, eigvals, ok = cube._principal_axes(
            q_good=self.pca_q_good,
            min_points=self.pca_min_points,
            anisotropy_threshold=self.anisotropy_threshold,
            depth_min=self.depth_min_for_pca,
        )
        R_use = R_loc if ok else (cube.R if cube.R is not None else np.eye(d))
        mu_use = mu_loc if ok else (cube.mu if cube.mu is not None else np.full(d, 0.5))
        widths = np.array([hi - lo for (lo, hi) in cube.bounds], dtype=float)
        # line search along first component with some prob
        if np.random.rand() < self.line_search_prob:
            x_prime = np.zeros(d, dtype=float)
            lo1, hi1 = cube.bounds[0]
            # Ensure valid interval ordering
            if hi1 < lo1:
                lo1, hi1 = hi1, lo1
            x_prime[0] = np.random.uniform(lo1, hi1)
            for j in range(1, d):
                loj, hij = cube.bounds[j]
                # Ensure valid interval ordering
                if hij < loj:
                    loj, hij = hij, loj
                span = max(0.0, (hij - loj))
                std = max(0.0, 0.1 * span)
                # Normal with std=0 is fine (degenerate at mean=0), then clip into [loj,hij]
                x_prime[j] = np.clip(np.random.normal(0.0, std), loj, hij)
            x = (mu_use + R_use @ x_prime).astype(float)
            # Clip to global bounds
            for j in range(d):
                lo, hi = self.bounds[j]
                x[j] = float(np.clip(x[j], lo, hi))
            return x
        # gaussian sampling with axis-wise std scaled by eigenvalues
        x_prime = np.zeros(d, dtype=float)
        ev = eigvals if ok else np.ones(d)
        ev = np.maximum(ev, 1e-12)
        scale = ev / float(np.mean(ev))
        for j in range(d):
            lo, hi = cube.bounds[j]
            # Ensure valid interval ordering to avoid negative std
            if hi < lo:
                lo, hi = hi, lo
            half = (hi - lo) * 0.5
            std = max(0.0, self.gauss_scale * half * float(scale[j]))
            # Normal with std=0 is fine (degenerate at mean=0)
            val = np.random.normal(0.0, std)
            x_prime[j] = float(np.clip(val, lo, hi))
        x = (mu_use + R_use @ x_prime).astype(float)
        # Clip to global bounds
        for j in range(d):
            lo, hi = self.bounds[j]
            x[j] = float(np.clip(x[j], lo, hi))
        return x

    def _auto_tune_surrogate(self, cube: QuadCube) -> None:
        """Adjust surrogate knobs adaptively based on fit quality, dim and progress.

        Heuristics:
        - Dimension scaling: more candidates for higher d; keep rho_max moderate.
        - Quality gates: start permissive; tighten if r2↑ and sigma2↓.
        - Margin: small positive to prefer confident mean improvements.
        - Distance penalty: light, increase when sigma2 grows or r2 drops.
        - Elite sampling: moderate by default, reduce if overfitting (very low sigma2 but low improve rate).
        """
        try:
            d = len(cube.bounds)
        except Exception:
            d = 2
        # Defaults from base, scaled by dim
        base = self._auto_base
        # scale candidates with d but cap to keep runtime sane
        cand = int(min(16384, max(4096, base['candidates'] * (1.0 + 0.15 * max(0, d - 4)))))
        rho_max = float(base['rho_max']) if d <= 8 else 2.1
        rho_pen = float(base['rho_pen'])
        r2_min = float(base['r2_min'])
        sigma2_max = float(base['sigma2_max'])
        ei_min = float(base['ei_min'])
        margin = float(base['margin'])
        elite_frac = float(base['elite_frac'])
        elite_top_k = int(base['elite_top_k'])

        # Light progression with total trials: early = more permissive to engage surrogate
        T = int(getattr(self, 'total_trials', 0))
        if T < 15:
            r2_min = min(r2_min, 0.6)
            ei_min = min(ei_min, 0.006)
            margin = max(0.15, margin)
        elif T > 40:
            # Later: demand slightly higher quality
            r2_min = max(r2_min, 0.6)
            ei_min = max(ei_min, 0.005)
            margin = max(0.2, margin)

        # Use local surrogate quality if present to refine gates
        surr = getattr(cube, 'surrogate_2d', None)
        if surr is not None and surr.get('n', 0) >= self.config.surrogate_min_points:
            r2 = float(surr.get('r2', 0.0))
            s2 = float(surr.get('sigma2', 1.0))
            # Tighten when model is good
            if r2 >= 0.7 and s2 <= sigma2_max:
                ei_min = max(0.004, ei_min * 0.8)
                margin = max(0.15, margin * 0.9)
                rho_pen = max(0.05, rho_pen * 0.9)
                rho_max = min(2.0, rho_max)
                elite_frac = min(0.65, elite_frac + 0.05)
                elite_top_k = min(12, max(elite_top_k, 6))
            
            # SUPER TRUST: If model is excellent, allow massive extrapolation
            if r2 > 0.9:
                rho_max = 100.0  # Effectively disable trust region boundary
                rho_pen = 0.001  # Negligible distance penalty
                ei_min = 0.0     # Accept any improvement
                margin = -1.0    # Disable gate (accept even if predicted slightly worse, though unlikely)
                
            # Relax when model is weak
            if r2 < 0.55 or s2 > sigma2_max * 0.75:
                r2_min = 0.5
                ei_min = min(0.008, ei_min * 1.5)
                margin = min(0.3, margin * 1.1)
                rho_pen = min(0.15, rho_pen * 1.2)
                rho_max = min(2.3, rho_max + 0.2)
                elite_frac = max(0.45, elite_frac - 0.05)
                cand = int(min(16384, cand * 1.25))

        # Apply
        self.surr_r2_min = float(r2_min)
        self.surr_sigma2_max = float(sigma2_max)
        self.surr_ei_min = float(ei_min)
        self.surr_accept_margin_sigma = float(margin)
        self.surr_mahalanobis_max = float(rho_max)
        self.surr_ei_rho2_penalty = float(rho_pen)
        self.surr_elite_frac = float(elite_frac)
        self.surr_elite_top_k = int(elite_top_k)
        self.surr_candidates = int(cand)

    def select_cube(self) -> QuadCube:
        """Select next cube to explore using UCB."""
        # Check if we have a preferred leaf from recent split
        if self._preferred_leaf is not None:
            if any(c is self._preferred_leaf for c in self.leaf_cubes):
                leaf = self._preferred_leaf
                self._preferred_leaf = None
                self._debug_log(f"[Select Cube] Using preferred leaf {id(leaf)}")
                return leaf
        
        # Always use UCB (best performer)
        # Calculate dynamic parameters for each cube
        candidates = []
        
        # Calculate root volume for normalization
        root_vol = 1.0
        for lo, hi in self.root.bounds:
            root_vol *= max(hi - lo, 0.0)
        
        # Calculate global scale (sigma) for bonus scaling
        # Use root variance if available, else 1.0
        global_sigma = np.sqrt(self.root.var_score) if self.root.var_score > 0 else 1.0
        
        for c in self.leaf_cubes:
            # --- ADAPTIVE BETA ---
            # Logic: Trust the model if it's good (beta -> 0), explore if it's bad (beta -> 0.5)
            beta_base = self.config.ucb_beta
            r2 = 0.0
            if c.surrogate_2d is not None and c.surrogate_2d.get('n', 0) >= self.min_points:
                r2 = float(c.surrogate_2d.get('r2', 0.0))
            
            # Clamp R2 to [0, 1] for safety
            r2_eff = max(0.0, min(1.0, r2))
            beta_dyn = beta_base * (1.0 - r2_eff)
            
            # --- ADAPTIVE LAMBDA GEO ---
            # Logic: Scale with dimension. D/2 is a reasonable heuristic.
            lambda_geo_dyn = 1.0 + 0.1 * self.dim
            
            val = c.ucb(beta=beta_dyn, lambda_geo=lambda_geo_dyn, vol_scale=root_vol, value_scale=global_sigma)
            candidates.append((c, val, beta_dyn, lambda_geo_dyn, r2_eff))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ALWAYS LOG TOP CANDIDATES FOR DEBUGGING
        self._debug_log(f"\n[Select Cube Trial {self.trial_id}] Top 5 candidates:")
        for i, (c, val, b_dyn, l_dyn, r2) in enumerate(candidates[:5]):
            vol = 1.0
            for lo, hi in c.bounds:
                vol *= max(hi - lo, 0.0)
            n_eff = c.n_trials
            mu = float(c.mean_score) if n_eff > 0 else 0.0
            var = float(c.var_score) if n_eff > 1 else float(c.prior_var)
            decay = float(np.log(n_eff + 2.0))
            
            # Replicate decay logic for logging
            # Match ucb() logic exactly
            d_dim = len(c.bounds)
            vol_rel = vol / max(1e-12, root_vol)
            vol_factor = pow(vol_rel, 1.0 / max(1, d_dim))
            
            geo_decay = 1.0 / (1.0 + 0.1 * c.depth) # Was 0.2 in log, 0.1 in ucb
            lambda_eff = l_dyn * geo_decay
            
            # Scale bonus by value_scale (global_sigma)
            bonus = lambda_eff * vol_factor * global_sigma / decay
            base = val - bonus
            
            self._debug_log(f"  #{i} ID={id(c)%1000} UCB={val:.4f} (Base={base:.4f}, Bonus={bonus:.4f}) "
                           f"mu={mu:.4f}, var={var:.4f}, beta={b_dyn:.3f}, lambda={l_dyn:.3f}, r2={r2:.3f}, vol={vol:.2e}")

        return candidates[0][0]

    def _maybe_denormalize(self, x_candidate: np.ndarray) -> Optional[List[Any]]:
        if self.param_space is None:
            return None
        return self.param_space.denormalize(x_candidate)

    def _call_objective(self, objective_fn: Callable[[np.ndarray, int], Any], x: np.ndarray, epochs: int) -> TypingTuple[float, Optional[Any]]:
        """Call objective function, handling both signatures (with/without epochs)."""
        self.objective_calls += 1
        self.obj_calls = self.objective_calls  # sync alias
        
        # Try calling with epochs parameter first
        try:
            res = objective_fn(x, epochs=epochs)
        except TypeError:
            # Fallback: objective doesn't accept epochs parameter
            res = objective_fn(x)
        
        if isinstance(res, tuple):
            score = float(res[0])
            artifact = res[1] if len(res) > 1 else None
        else:
            score = float(res)
            artifact = None
        return score, artifact

    def _cube_depth(self, cube: QuadCube) -> int:
        d = 0
        cur = cube
        while cur.parent is not None:
            d += 1
            cur = cur.parent
        return d

    def leaf_volumes(self) -> List[float]:
        vols: List[float] = []
        for c in self.leaf_cubes:
            v = 1.0
            for (lo, hi) in c.bounds:
                v *= max(hi - lo, 0.0)
            vols.append(v)
        return vols

    def diagnostics(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        total = max(1, self.total_trials)
    # early diagnostics removed
        res['splits'] = self.splits_count
        res['prunes'] = self.prunes_count
    # split_probes removed
        if self.leaf_cubes:
            res['max_depth'] = max(self._cube_depth(c) for c in self.leaf_cubes)
            res['leaf_volumes'] = self.leaf_volumes()
        else:
            res['max_depth'] = 0
            res['leaf_volumes'] = []
        return res

    def _pick_best_child(self, children: List[QuadCube]) -> QuadCube:
        """Pick best child using UCB metric (always)."""
        # Use default robust params for initial child selection if no history
        beta_def = self.config.ucb_beta
        lambda_def = 1.0 + 0.1 * self.dim
        return max(children, key=lambda c: c.ucb(beta=beta_def, lambda_geo=lambda_def))

    def _backprop_stats(self, leaf: QuadCube, score: float) -> None:
        """Propagate trial result up the tree to update ancestor statistics.
        
        Updates n_trials, mean_score, var_score for all ancestors using Welford's algorithm.
        Does NOT add the point to geometric history or retrain surrogates.
        """
        curr = leaf.parent
        while curr is not None:
            curr.n_trials += 1
            n = curr.n_trials
            
            # Welford's online algorithm for mean and variance
            if n == 1:
                curr.mean_score = score
                curr.M2 = 0.0
            else:
                delta = score - curr.mean_score
                curr.mean_score += delta / n
                delta2 = score - curr.mean_score
                curr.M2 += delta * delta2
            
            curr.var_score = curr.M2 / (n - 1) if n > 1 else 0.0
            
            # Update best score if improved
            if score > curr.best_score:
                curr.best_score = score
                
            curr = curr.parent

    def run_trial(self, cube: QuadCube, objective_fn: Callable[[np.ndarray, int], Any]) -> None:
        self.trial_id += 1
        self.total_trials += 1

        # Recalculate adaptive params for logging
        beta_base = self.config.ucb_beta
        r2 = 0.0
        if cube.surrogate_2d is not None and cube.surrogate_2d.get('n', 0) >= self.min_points:
            r2 = float(cube.surrogate_2d.get('r2', 0.0))
        r2_eff = max(0.0, min(1.0, r2))
        beta_dyn = beta_base * (1.0 - r2_eff)
        lambda_geo_dyn = 1.0 + 0.1 * self.dim

        # DEBUG: Log UCB components for the selected cube
        ucb_val = cube.ucb(beta=beta_dyn, lambda_geo=lambda_geo_dyn)
        # Calculate effective beta for logging
        depth_decay = 1.0 / (1.0 + 0.1 * cube.depth)
        beta_eff = beta_dyn * depth_decay
        self._debug_log(f"[Trial {self.trial_id}] Selected Cube {id(cube)}: depth={cube.depth}, n={cube.n_trials}, "
                       f"mean={cube.mean_score:.4f}, var={cube.var_score:.4f}, ucb={ucb_val:.4f}, beta_eff={beta_eff:.5f}, r2={r2:.3f}")

        x_candidate = self._sample_biased_in(cube, alpha=0.2, top_k=5)
        
        # Check for boundary hits
        hit_bounds = []
        for j in range(self.dim):
            lo, hi = self.bounds[j]
            if abs(x_candidate[j] - lo) < 1e-9:
                hit_bounds.append(f"x{j}=LO")
            elif abs(x_candidate[j] - hi) < 1e-9:
                hit_bounds.append(f"x{j}=HI")
        if hit_bounds:
            self._debug_log(f"[Trial {self.trial_id}] Hit bounds: {', '.join(hit_bounds)} at {x_candidate}")
            # print(f"[Trial {self.trial_id}] Hit bounds: {', '.join(hit_bounds)}")

        cube.add_tested_point(x_candidate)
        x_real = self._maybe_denormalize(x_candidate)
        
        # Objective call: currently default to normalized space for compatibility.
        # If you need real-domain evaluation (e.g., typed/categorical ParamSpace),
        # adapt here to pass x_real when provided.
        x_for_obj = x_candidate

        s_final_raw, artifact = self._call_objective(objective_fn, x_for_obj, epochs=self.full_epochs)
        # Internally use signed score so minimization works seamlessly with UCB/max logic
        s_final = float(self.sign * s_final_raw)
        
        # SANITY CHECK: Log every loss value
        if self._debug_logger:
            self._debug_logger(f"[LOSS] Trial {self.trial_id}: s_raw={s_final_raw:.6f}, s_signed={s_final:.6f}")
        
        if not hasattr(cube, "_tested_pairs"):
            cube._tested_pairs: List[TypingTuple[np.ndarray, float]] = []
        cube._tested_pairs.append((np.array(x_candidate, dtype=float), float(s_final)))
        cube.update_final(float(s_final), min_points=self.config.surrogate_min_points, ridge_alpha=self.config.ridge_alpha)
        # Keep raw scores for external consumers if needed
        self.s_final_all.append(float(s_final_raw))
        
        # --- NEW: Backpropagate statistics to ancestors ---
        self._backprop_stats(cube, float(s_final))
        # --------------------------------------------------
        
        improved = bool(s_final > self.best_score_global)
        if improved:
            self.best_score_global = float(s_final)
            self.best_x_candidate = x_candidate.tolist()
            self.best_x_real = x_real
            if self.on_best is not None:
                try:
                    self.on_best({
                        'trial': self.trial_id,
                        'score': float(s_final_raw),
                        'x_candidate': self.best_x_candidate,
                        'x_real': self.best_x_real,
                        'artifact': artifact,
                    })
                except Exception:
                    pass
    # No per-trial debug prints in production

        cube.stale_steps = 0
        for c in self.leaf_cubes:
            if c is not cube:
                c.stale_steps += 1

        self.split_checks += 1
        
        # --- ADAPTIVE SPLIT LOGIC ---
        # Gamma (Info Gain Threshold): Relative to parent variance
        # If parent is noisy (high var), we need a big drop to justify split.
        # If parent is precise (low var), a small drop is enough.
        # Heuristic: 0.1% of parent variance (was 1%) - VERY AGGRESSIVE
        gamma_dyn = 0.0
        if cube.surrogate_2d is not None:
            sigma2 = float(cube.surrogate_2d.get('sigma2', 1.0))
            gamma_dyn = self.config.gamma_split * sigma2
            
        # Gate Ratio: Dynamic based on budget?
        # Let's keep the relative margin logic: margin = |best| * 0.5
        # This is robust enough.
        gate_ratio_dyn = 0.2

        # DEBUG: Log split decision inputs
        self._debug_log(f"[Split Check Trial {self.trial_id}] Cube {id(cube)}: gamma_dyn={gamma_dyn:.6f}, gate_ratio={gate_ratio_dyn:.2f}, sigma2={cube.surrogate_2d.get('sigma2', 'N/A') if cube.surrogate_2d else 'N/A'}")

        split_type = cube.should_split(
            min_trials=self.min_trials,
            min_points=self.min_points,
            max_depth=self.max_depth,
            min_width=self.min_width,
            gamma=gamma_dyn,
            global_best=self.best_score_global,
            gate_ratio=gate_ratio_dyn,
            remaining_budget=(self.budget - self.trial_id) if self.budget > 0 else None,
            ridge_alpha=self.config.ridge_alpha,
        )
        
        # Track why split didn't happen
        if split_type == 'none' and hasattr(cube, 'split_block_reason') and cube.split_block_reason:
            self.split_reasons.append(f"Trial {self.trial_id}: {cube.split_block_reason}")
            self._debug_log(f"  Split Blocked: {cube.split_block_reason}")
        
        if split_type != 'none':
            children = cube.split4(ridge_alpha=self.config.ridge_alpha) if split_type == 'quad' else cube.split2(ridge_alpha=self.config.ridge_alpha)
            if children:
                self.splits_count += 1
                self.split_trials.append(self.trial_id)  # Record split trial
                
                # Log split event
                self._debug_log(f"\n[SPLIT at Trial {self.trial_id}] Cube {id(cube)} (depth={cube.depth}) split into {len(children)} children")
                self._debug_log(f"  Parent: n_trials={cube.n_trials}, best={cube.best_score:.4f}")
                for i, child in enumerate(children):
                    self._debug_log(f"  Child {i}: depth={child.depth}, inherited_history={len(child._points_history)}")
                
                # Set birth_trial for all children
                for child in children:
                    child.birth_trial = self.trial_id
                # Use identity check to avoid numpy array ambiguity
                for i, c in enumerate(self.leaf_cubes):
                    if c is cube:
                        self.leaf_cubes.pop(i)
                        break
                for child in children:
                    self.leaf_cubes.append(child)
                # split probes removed

                # keep track of the best child to bias next selection
                best_child = self._pick_best_child(children)
                self._preferred_leaf = best_child

        # Prune weak cubes (always enabled)
        prev = len(self.leaf_cubes)
        self.prune_cubes()
        removed = max(0, prev - len(self.leaf_cubes))
        self.prunes_count += removed
        
        # Log pruning if it happened
        if removed > 0:
            self._debug_log(f"\n[PRUNE at Trial {self.trial_id}] Removed {removed} cube(s), {len(self.leaf_cubes)} remain")

        # Log raw s_final and raw-scale best score for readability
        best_score_raw = float(self.sign * self.best_score_global)
        self._log([
            self.trial_id, id(cube), json.dumps(x_candidate.tolist()),
            self._safe_json(x_real) if x_real is not None else '',
            float(s_final_raw), best_score_raw, len(self.leaf_cubes)
        ])
        # Callback per-trial (fine trial)
        if self.on_trial is not None:
            try:
                self.on_trial({
                    'trial': self.trial_id,
                    'x_candidate': x_candidate.tolist(),
                    'x_real': x_real,
                    's_final': s_final,
                    'leaf_id': id(cube),
                    'objective_calls': self.objective_calls,
                })
            except Exception:
                pass
    # No trial-level debug file/prints in production

    def prune_cubes(self) -> None:
        if len(self.leaf_cubes) <= self.min_leaves:
            return
        
        # --- ADAPTIVE PRUNING ---
        # Logic: Time Funnel.
        # Start permissive (delta ~ 1.0), end strict (delta ~ 0.0).
        # progress = t / T_max
        progress = min(1.0, self.trial_id / max(1, self.budget))
        delta_max = 1.0
        delta_dyn = delta_max * ((1.0 - progress) ** 2)
        
        # Ensure we don't go below a tiny margin to avoid float errors
        delta_dyn = max(0.001, delta_dyn)
        
        # Work in signed domain where higher is better regardless of minimize/maximize
        # Sort by UCB (using default robust params for sorting)
        beta_def = self.config.ucb_beta
        lambda_def = 1.0 + 0.1 * self.dim
        ranked = sorted(self.leaf_cubes, key=lambda c: c.ucb(beta=beta_def, lambda_geo=lambda_def), reverse=True)
        
        best_signed = float(self.best_score_global)
        keep_thresh = best_signed - delta_dyn
        
        # DEBUG: Log pruning threshold
        self._debug_log(f"[Prune Check Trial {self.trial_id}] Best={best_signed:.4f}, Delta={delta_dyn:.4f}, Thresh={keep_thresh:.4f}, Progress={progress:.2f}")
        
        keep: List[QuadCube] = []
        for c in ranked:
            # CRITICAL: Never prune cubes with 0 observations
            if c.n_trials == 0:
                keep.append(c)
                continue
            
            # Grace period: don't prune leaves that are too young
            age = self.trial_id - c.birth_trial
            if age < self.prune_grace_period:
                keep.append(c)
                continue
            
            vol = 1.0
            for lo, hi in c.bounds:
                vol *= max(hi - lo, 0.0)
            too_small = vol < 1e-6
            
            # Check against dynamic threshold
            # Use mean score for pruning check (conservative) or UCB?
            # Standard practice: Prune if UCB < Best - Delta
            val = c.ucb(beta=beta_def, lambda_geo=lambda_def)
            ok = (val >= keep_thresh)
            
            if not ok:
                self._debug_log(f"  Pruning Cube {id(c)%1000}: UCB={val:.4f} < Thresh={keep_thresh:.4f}")
            
            if ok and c.stale_steps < self.stale_steps_max and not too_small:
                keep.append(c)
        if len(keep) < self.min_leaves:
            for c in ranked:
                if not any(c is k for k in keep):
                    keep.append(c)
                if len(keep) >= self.min_leaves:
                    break
        self.leaf_cubes = keep

    def optimize(self, objective_fn: Callable[[np.ndarray, int], Any], budget: int) -> None:
        """Run optimization for specified budget (number of trials)."""
        budget = int(budget)
        self.budget = budget
        
        # Adapt configuration to budget
        self.config.adapt(self.dim, budget)
        # Update aliases
        self.min_points = self.config.surrogate_min_points
        self.min_trials = self.config.min_points_split
        self.max_depth = self.config.max_depth
        self.min_width = self.config.min_width
        
        for i in range(budget):
            cube = self.select_cube()
            self.run_trial(cube, objective_fn)
        
        # Console output - just essentials
        print(f"\n[QuadHPO] Run completed: {self.splits_count} splits, {self.prunes_count} prunes, {len(self.leaf_cubes)} leaves")
