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
class QuadCube:
    # Surrogate model cache (2D quadratic on PC1-PC2)
    surrogate_2d: Optional[Dict[str, Any]] = field(default=None, init=False)
    
    # Centralized surrogate hyperparameters
    ridge_alpha: float = field(default=1e-3, init=False)
    surrogate_min_points: int = field(default=8, init=False)

    #Metodo che adatta una funzione quadratica sulle prime due componenti principali (PC1 e PC2)
    def fit_surrogate(self, min_points: Optional[int] = None) -> None: #Prendo i dati che ho già misurato, li guardo da un’angolazione comoda (PCA) e ci disegno sopra una parabola (tipo “colline” e “valli”) che approssima la zona vicina.
        """Fit a 2D quadratic surrogate on PC1-PC2 using local tested pairs.

        Stores ridge solution along with the inverse design matrix to enable
        input-dependent predictive variance at inference time.
        """
        if min_points is None:
            min_points = self.surrogate_min_points
        pairs = getattr(self, "_tested_pairs", [])
        if len(pairs) < min_points:
            self.surrogate_2d = None
            return
        d = len(self.bounds)
        # Use local PCA axes for projection
        R, mu, _, ok = self._principal_axes()  # Calcola pca locale matrice di rotazione verso le nuove coordinate (PC1, PC2, ...)
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)  # Metto in una lista le ricette che ho provato (X) e in un’altra il voto che gli ho dato (y)
        T = (R.T @ (X.T - mu.reshape(-1, 1))).T  # shape (n, d)
        t1 = T[:, 0]
        t2 = T[:, 1] if d > 1 else np.zeros_like(t1)
        # Design matrix: [1, t1, t2, t1^2, t2^2, t1*t2]
        Phi = np.stack([
            np.ones_like(t1), t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2
        ], axis=1)
        A = Phi.T @ Phi + self.ridge_alpha * np.eye(6)
        b = Phi.T @ y
        try:
            w = np.linalg.solve(A, b)  # Trovo i numeri che fanno combaciare la mia parabola con i punti misurati.
        except Exception:
            self.surrogate_2d = None
            return
        # Pre-compute (Phi^T Phi + alpha I)^-1 for predictive variance
        try:
            A_inv = np.linalg.inv(A)  # Mi salvo un calcolo che userò più tardi per sapere “quanto sono sicuro” delle mie previsioni.
        except Exception:
            A_inv = None
        # Residual variance with effective degrees of freedom (ridge)
        y_hat = Phi @ w
        resid = y - y_hat
        G = Phi.T @ Phi  # (6x6)
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
        
        # Extract Hessian H (2x2) from quadratic coefficients w = [c, b1, b2, a11, a22, a12]
        # where f(t1, t2) = c + b1*t1 + b2*t2 + 0.5*a11*t1^2 + 0.5*a22*t2^2 + a12*t1*t2
        # Hessian: H = [[a11, a12], [a12, a22]]
        w_flat = w.reshape(-1)
        H = np.array([[float(w_flat[3]), float(w_flat[5])],
                      [float(w_flat[5]), float(w_flat[4])]], dtype=float)
        # Compute eigenvalues in descending order
        lam = np.linalg.eigvalsh(H)[::-1]  # [λ1, λ2] descending
        
        self.surrogate_2d = {
            'w': w,
            'mu': mu,
            'R': R,
            'A_inv': A_inv,
            'sigma2': sigma2,
            'n': n_obs,
            'r2': r2,
            'pca_ok': bool(ok),
            'df_eff': df_eff,
            'H': H,
            'lambda': lam,
        }

    def predict_surrogate(self, x_prime: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std at x_prime (in prime coords) using 2D quadratic surrogate.

        Uses ridge-regression mean prediction variance: Var[ŷ(x)] ≈ σ² · (φ(x)^T A_inv φ(x)),
        where A_inv = (Φ^T Φ + αI)^-1 saved during fitting and σ² is residual variance.
        This returns the uncertainty on the surrogate mean (not including observation noise).
        If A_inv is unavailable, falls back to a constant σ.
        """
        if self.surrogate_2d is None:
            return 0.0, 1.0
        t1 = x_prime[0]
        t2 = x_prime[1] if len(x_prime) > 1 else 0.0 #Estrae le coordinate locali: PC1 = t1, PC2 = t2
        Phi = np.array([1.0, t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2]) #Creo i sei ingredienti per calcolare l’altezza della mia “collina” nel punto richiesto.
        y_hat = float(self.surrogate_2d['w'] @ Phi) #Predizione media: prodotto scalare
        sigma2 = float(self.surrogate_2d.get('sigma2', 1.0))
        A_inv = self.surrogate_2d.get('A_inv', None) #Recupera la varianza residua del fit e l’inversa regolarizzata per la varianza predittiva. Prendo quanto “rumore” c’era nei dati e la chiave matematica che mi serve per stimare l’incertezza punto per punto
        if A_inv is None:
            sigma = float(np.sqrt(max(sigma2, 1e-12))) # Se non ho l’inversa, torno a una incertezza fissa
            return y_hat, sigma
        # predictive variance term v = φ^T A_inv φ (mean prediction variance only)
        try:
            v = float(Phi @ (A_inv @ Phi))
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
            np.ndarray of shape (2,) with scores [S_1, S_2] for PC1 and PC2,
            or None if surrogate is not available or doesn't have eigenvalues.
        """
        s = self.surrogate_2d
        if s is None or s.get('lambda') is None or s.get('R') is None or s.get('mu') is None:
            return None
        lam = np.asarray(s['lambda'], float)[:2]
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

        # h for first two surrogate axes: h_i = Σ |M[i,j]| * width_j
        # Note: h uses full widths (not half-spans). Any constant factor cancels out
        # in axis ranking; only relative magnitudes matter for choosing split axes.
        A = np.abs(M[:2, :])         # (2, d)
        h = A @ w                    # (2,)

        # h is guaranteed >= 0 by construction, but clamp for numerical safety
        h = np.maximum(h, 0.0)

        return (lam_abs**2) * (h**4)
    
    bounds: List[Tuple[float, float]]  # in local (prime) coordinates, typically symmetric around 0
    parent: Optional["QuadCube"] = None
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
        # Debug assert: punti sempre dentro [0,1]
        if hasattr(self, '_debug_assert_bounds') and self._debug_assert_bounds:
            if np.any(x < -1e-9) or np.any(x > 1.0 + 1e-9):
                raise AssertionError(f"Point fuori bounds dopo mapping prime->original: {x}")
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

    def update_final(self, final_score: float) -> None: #Aggiorna statistiche finali: numero prove, lista degli score, media, varianza
        self.n_trials += 1
        self.scores.append(float(final_score))
        self.mean_score = float(np.mean(self.scores))
        self.var_score = float(np.var(self.scores)) if len(self.scores) > 1 else 0.0
        if final_score > self.best_score:
            self.best_score = float(final_score)
        # Refit surrogate after each update if enough data
        self.fit_surrogate()

    def leaf_score(self, mode: str = 'max') -> float: #Dimmi quanto è “buona” questa zona: preferisci il migliore risultato o la media? Se non ho risultati finali, guardo quelli provvisori; se non ho niente, dico che è pessima 
        if self.scores:
            return float(np.max(self.scores) if mode == 'max' else np.mean(self.scores))
        pairs = getattr(self, "_tested_pairs", [])
        if pairs:
            vals = np.array([s for (_, s) in pairs], dtype=float)
            return float(np.max(vals) if mode == 'max' else np.mean(vals))
        return float('-inf')

    def ucb(self, beta: float = 1.6, eps: float = 1e-8, lambda_geo: float = 0.0) -> float:
        n_eff = self.n_trials
        if self.n_trials > 0:
            mu = float(self.mean_score)
            var = float(self.var_score) if self.n_trials > 1 else float(self.prior_var)
        else:
            mu = 0.0
            var = float(self.prior_var)
        if self.parent is not None and self.parent.n_trials > 0: #Se il genitore ha esperienza, lo ascolto per metà. È come chiedere consiglio a chi ha già visto la zona.
            mu = 0.5 * float(self.parent.mean_score) + 0.5 * mu
            pv = float(self.parent.var_score) if self.parent.var_score > 0 else float(self.parent.prior_var)
            var = 0.5 * pv + 0.5 * var
        if n_eff <= 0: #Calcola l’UCB base: media + margine d’incertezza. il margine scende con più dati
            base = float(mu + beta * np.sqrt(var + self.prior_var))
        else: #Più ho provato, più la mia barra d’errore si restringe. Metto sempre un pizzico di incertezza di fondo.
            base = float(mu + beta * np.sqrt(var / (n_eff + eps) + self.prior_var))
        if lambda_geo <= 0.0:
            return base
        # Volume nel frame locale (prodotto delle larghezze). Se qualche bound non è valido, width=0.
        vol = 1.0
        for (lo, hi) in self.bounds:
            width = max(hi - lo, 0.0)
            vol *= width
        # Bonus di esplorazione attenuato dalla densità dei campioni nella cella.
        bonus = lambda_geo * vol / float(np.sqrt(n_eff + 1.0))
        return base + bonus

    def should_split(self,
                     min_trials: int = 5,
                     min_points: Optional[int] = None,
                     max_depth: Optional[int] = 4,
                     min_width: float = 1e-3,
                     gamma: float = 0.02) -> str: #decide se e come dividere il cubo
        #Regole: non spacco troppo in profondità, non spacco briciole, e non spacco se non ho abbastanza prove o il guadagno è risibile
        # Use centralized surrogate_min_points if not overridden
        if min_points is None:
            min_points = self.surrogate_min_points
        
        # stop per profondità/ampiezza
        if max_depth is not None and self.depth >= max_depth:
            return 'none'
        #se hai raggiunto la profondità massima o se tutte le dimensioni sono più strette di min_width, non dividere
        widths = [abs(hi - lo) for (lo, hi) in self.bounds]
        if all(w < min_width for w in widths):
            return 'none'
        # evita esplosione precoce
        #Anti-esplosione: se non hai abbastanza trial finali e nemmeno abbastanza punti storici, non dividere. Non ho visto abbastanza: non ha senso spaccare la zona ancora.
        if self.n_trials < min_trials and len(self._points_history) < min_points:
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
            return 'none'

        # Info-gain: split solo se riduzione varianza residua > gamma
        # Solo se surrogato disponibile e almeno 2 figli
        #Applica un criterio di information gain basato sulla varianza residua del surrogato: richiede un surrogato con almeno 12 punti
        #Faccio una prova mentale: se taglio, i pezzi nuovi mi riducono abbastanza l’incertezza? Se il guadagno è piccolo, non spacco; se è decente, procedo col tipo di taglio deciso prima.
        if self.surrogate_2d is not None and self.surrogate_2d['n'] >= self.surrogate_min_points:
            var_parent = float(self.surrogate_2d['sigma2'])
            # Simula split
            if split_type == 'quad':
                children = self._simulate_split4()
            else:
                children = self._simulate_split2()
            if children:
                n_total = sum(ch['n'] for ch in children)
                if n_total > 0:
                    var_post = sum((ch['n']/n_total)*ch['var'] for ch in children)
                    delta = var_parent - var_post
                    if delta < gamma:
                        return 'none'
        return split_type

    def _simulate_split2(self):
        # Simula split2 e fitta surrogato su ciascun figlio
        d = len(self.bounds)
        widths_parent = self._widths()
        # prende il frame PCA locale (R, mu). Calcola il punto di taglio lungo l’asse ax con _quad_cut_along_axis
        ax = int(np.argmax(widths_parent))
        R, mu, _, ok = self._principal_axes()
        R_use = R if ok else (self.R if self.R is not None else np.eye(d))
        mu_use = mu if ok else (self.mu if self.mu is not None else np.full(d, 0.5))
        # Parent bounds expressed in chosen frame
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        spans_use = (np.abs(M) @ widths_parent)
        base_bounds = [(-wi/2.0, wi/2.0) for wi in spans_use]
        cut = self._quad_cut_along_axis(ax, R_use, mu_use)
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
            Phi = np.stack([
                np.ones_like(t1), t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2
            ], axis=1)
            # ridge_alpha = 1e-3 # Now using self.ridge_alpha
            A = Phi.T @ Phi + self.ridge_alpha * np.eye(6)
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

    def _simulate_split4(self):
        # Simula split4 e fitta surrogato su ciascun figlio (PC1/PC2)
        d = len(self.bounds)
        widths_parent = self._widths()
        R, mu, _, ok = self._principal_axes()
        R_use = R if ok else (self.R if self.R is not None else np.eye(d))
        mu_use = mu if ok else (self.mu if self.mu is not None else np.full(d, 0.5))
        # Parent bounds in chosen frame
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        spans_use = (np.abs(M) @ widths_parent)
        base_bounds = [(-wi/2.0, wi/2.0) for wi in spans_use]
        # se la PCA è ok usa PC1 (0) e PC2 (1, o 0 se 1D) come assi di taglio
        if ok:
            ax_i, ax_j = 0, 1 if d > 1 else 0
        else:
            # altrimenti, prendi i due assi più larghi dai bounds (ordine come implementazione legacy)
            top2 = np.argsort(widths_parent)[-2:]
            ax_i, ax_j = int(top2[0]), int(top2[1])
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R_use, mu_use)
            cut_j = self._quad_cut_along_axis(ax_j, R_use, mu_use)
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
            Phi = np.stack([
                np.ones_like(t1), t1, t2, 0.5 * t1 ** 2, 0.5 * t2 ** 2, t1 * t2
            ], axis=1)
            # ridge_alpha = 1e-3 # Now using self.ridge_alpha
            A = Phi.T @ Phi + self.ridge_alpha * np.eye(6)
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
                        min_points: Optional[int] = None,
                        anisotropy_threshold: float = 1.4,
                        depth_min: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (R, mu, eigvals, ok) where R columns are principal axes.
        Weighted PCA on all tested points to reduce self-referential bias.

        Notes:
        - Uses softmax weights of standardized scores so better points weigh more.
        - Mixes a uniform component so every point has non-zero influence.
        - Applies an anisotropy check; if weak or insufficient data, keep current frame.
        """
        # Use centralized surrogate_min_points if not overridden
        if min_points is None:
            min_points = self.surrogate_min_points
            
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
                              mu: np.ndarray) -> float:
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
        # design matrix for y ~ a + b t + 0.5 c t^2
        Phi = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
        # ridge solve: (Phi^T Phi + alpha I) w = Phi^T y
        A = Phi.T @ Phi
        A += self.ridge_alpha * np.eye(3)
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
            t_star = -b_hat / c_hat
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

    def split2(self, axis: Optional[int] = None) -> List["QuadCube"]:
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
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        spans_use = (np.abs(M) @ widths_parent)  # full widths along R_use axes
        if axis is not None:
            ax = int(axis)
        else:
            S = self._curvature_scores()
            if S is not None:
                ax = int(np.argmax(S))
            elif ok:
                ax = int(np.argmax(spans_use))
            else:
                ax = int(np.argmax(widths_parent))

        # compute quadratic cut along chosen axis in chosen frame (for position),
        # but apply the split to the parent's prime bounds to preserve exact halving semantics in tests
        cut = self._quad_cut_along_axis(ax, R_use, mu_use)
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
            # Inherit surrogate hyperparameters
            ch.ridge_alpha = self.ridge_alpha
            ch.surrogate_min_points = self.surrogate_min_points
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
        self.children = [c1, c2]
        return self.children

    def split4(self) -> List["QuadCube"]:
        # 4-way split using PCA local axes; cut-points from quadratic 1D fits (fallback to midpoints)
        d = len(self.bounds)
        if d == 1:
            return self.split2(axis=0)
        self._ensure_frame()
        widths_parent = self._widths()
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
        M = (R_use.T @ self.R) if self.R is not None else R_use.T
        A = np.abs(M)
        spans_use = A @ widths_parent
        base_bounds = [(-wi/2.0, wi/2.0) for wi in spans_use]
        
        # Use curvature-driven split criterion if available
        S = self._curvature_scores()
        if S is not None:
            # Choose top-2 axes by curvature score
            order = np.argsort(S)[::-1]  # descending order
            ax_i = int(order[0])
            ax_j = int(order[1] if len(order) > 1 else order[0])
        else:
            # Fallback: choose two axes consistently in R_use frame
            # If PCA ok: use PC1 (0) and PC2 (1)
            # If PCA failed: use widest two axes from spans_use (which equals widths_parent when R_use==self.R)
            if ok:
                ax_i, ax_j = 0, 1 if d > 1 else 0
            else:
                # KEY FIX: Use spans_use (not widths_parent) to stay in R_use frame
                top2 = np.argsort(spans_use)[-2:]
                ax_i, ax_j = int(top2[0]), int(top2[1])
        
        # compute cutpoints
        #Prova a usare PCA locale per decidere i punti di taglio (via _quad_cut_along_axis)
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R_use, mu_use)
            cut_j = self._quad_cut_along_axis(ax_j, R_use, mu_use)
        else:
            lo_i, hi_i = base_bounds[ax_i]; cut_i = 0.5 * (lo_i + hi_i)
            lo_j, hi_j = base_bounds[ax_j]; cut_j = 0.5 * (lo_j + hi_j)
        # clip cuts inside bounds (in chosen frame) - ensure lo_i,etc defined
        lo_i, hi_i = base_bounds[ax_i]
        lo_j, hi_j = base_bounds[ax_j]
        cut_i = float(np.clip(cut_i, lo_i + 1e-12, hi_i - 1e-12))
        cut_j = float(np.clip(cut_j, lo_j + 1e-12, hi_j - 1e-12))
        # child bounds in chosen frame before re-centering
        #Costruisce i quattro bounds (quadranti) nel frame scelto: (sinistra/destra) × (sotto/sopra)
        def make_bounds(quadrant: TypingTuple[bool, bool]) -> List[Tuple[float, float]]:
            bi = (lo_i, cut_i) if quadrant[0] else (cut_i, hi_i)
            bj = (lo_j, cut_j) if quadrant[1] else (cut_j, hi_j)
            nb = list(base_bounds)
            nb[ax_i] = bi
            nb[ax_j] = bj
            return nb
        b_q1 = make_bounds((True, True))
        b_q2 = make_bounds((False, True))
        b_q3 = make_bounds((True, False))
        b_q4 = make_bounds((False, False))
        # centers and widths for children
        #Per ogni quadrante: calcola centro e larghezze nel frame scelto, crea il figlio con bounds ricentrati attorno a 0, e mappa il centro in spazio originale: mu_child = mu_parent + R_use @ center_prime_quadrante
        centers_prime = [np.array([(a + b) * 0.5 for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        widths_children = [np.array([b - a for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        # instantiate children with recentered prime boxes
        children: List[QuadCube] = []
        for ctr_p, wch in zip(centers_prime, widths_children):
            ch = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in wch], parent=self)
            ch.R = R_use.copy()
            # Map child center using the same PCA frame center used for cuts
            ch.mu = (mu_use + (R_use @ ctr_p)).astype(float)
            ch.prior_var = float(self.prior_var)
            # q_threshold removed
            ch.depth = self.depth + 1
            ch._tested_points = []
            # Inherit surrogate hyperparameters
            ch.ridge_alpha = self.ridge_alpha
            ch.surrogate_min_points = self.surrogate_min_points
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
        self.children = children
        return self.children


class QuadHPO:
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        beta: float = 0.05,
        lambda_geo: float = 0.8,
        full_epochs: int = 50,
        maximize: bool = True,
        param_space: Optional[ParamSpace] = None,
        rng_seed: Optional[int] = None,
        log_path: Optional[str] = None,
        on_best: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        # ============================================================
        # CORE PARAMETERS (user-tunable)
        # ============================================================
        self.beta = float(beta)              # Exploitation strength (0.05 = strong)
        self.lambda_geo = float(lambda_geo)  # Geometric shrinkage rate (0.8 recommended)
        self.full_epochs = int(full_epochs)  # Surrogate training epochs (50 recommended)
        self.maximize = bool(maximize)
        self.sign = 1.0 if self.maximize else -1.0
        
        # ============================================================
        # HARDCODED CONSTANTS (tested and optimized)
        # ============================================================
        self.min_trials = 12          # Minimum samples before split
        self.gamma = 0.02             # UCB exploration coefficient
        self.stale_steps_max = 15     # Max iterations without improvement before prune
        self.delta_prune = 0.025      # Prune threshold
        self.max_depth = 1            # Maximum tree depth
        self.min_width = 1e-3         # Minimum cube width
        self.min_points = 8           # Minimum points for surrogate (same as QuadCube.surrogate_min_points)
        self.min_leaves = 5           # Minimum leaf count
        self.prune_grace_period = 3   # Min trials before a leaf can be pruned
        
        # PCA/Sampling constants
        self.pca_q_good = 0.3
        self.pca_min_points = 8       # Use same as surrogate for consistency
        self.anisotropy_threshold = 1.4
        self.depth_min_for_pca = 1
        self.line_search_prob = 0.25
        self.gauss_scale = 0.35
        
        # Surrogate mode: always 'auto' (adaptive)
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
        self.best_x_norm: Optional[List[float]] = None
        self.best_x_real: Optional[List[Any]] = None
        self.trial_id: int = 0
        self._preferred_leaf: Optional[QuadCube] = None
        self.objective_calls: int = 0
        self.obj_calls = self.objective_calls  # alias
        
        # Cube selection: always UCB (best performer)
        self.cube_select_mode = 'ucb'
        
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # ============================================================
        # LOGGING
        # ============================================================
        self.total_trials: int = 0
        self.s_final_all: List[float] = []
        self.splits_count: int = 0
        self.prunes_count: int = 0
        if self.log_path and not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trial', 'cube_id', 'x_norm', 'x_real', 's_final', 'best_score_global', 'n_leaves'
                ])
        
        # ============================================================
        # ROOT CUBE INITIALIZATION (prime coordinate frame)
        # ============================================================
        d = len(bounds)
        root_lo = np.array([lo for (lo, hi) in bounds], dtype=float)
        root_hi = np.array([hi for (lo, hi) in bounds], dtype=float)
        root_mu = (root_lo + root_hi) * 0.5
        root_w = (root_hi - root_lo)
        self.root.R = np.eye(d)
        self.root.mu = root_mu
        self.root.bounds = [(-wi / 2.0, wi / 2.0) for wi in root_w]
        
        # PCA debiasing defaults (hardcoded, tested values)
        self.root.pca_softmax_tau = 0.6
        self.root.pca_mix_uniform = 0.25
        self.root.pca_weight_floor = 0.02

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)

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
            x = np.array(center, dtype=float) + 0.02 * np.random.randn(d)
            x = np.clip(x, 0.0, 1.0)
            return x

        # Use surrogate-based acquisition if available and reliable enough
        if (
            cube.surrogate_2d is not None
            and cube.surrogate_2d.get('n', 0) >= cube.surrogate_min_points
            and cube.surrogate_2d.get('pca_ok', True)
            and cube.surrogate_2d.get('sigma2', 1.0) <= float(self.surr_sigma2_max)
            and cube.surrogate_2d.get('r2', 0.0) >= float(self.surr_r2_min)
        ):
            # Evaluate surrogate in its own PCA frame, but sample candidates inside the current cube.
            M = int(self.surr_candidates)
            R_s = cube.surrogate_2d['R']
            mu_s = cube.surrogate_2d['mu']
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
            # covariance of first 2 dims around 0 (mu_s was computed from good points)
            T2 = T_tr[:, :2] if T_tr.size > 0 else np.empty((0, 2))
            if T2.shape[0] >= 4:
                C = np.cov(T2.T) if T2.shape[0] > 1 else np.eye(2)
            else:
                C = np.eye(2)
            # regularize
            C = np.asarray(C, dtype=float) + 1e-6 * np.eye(2)
            try:
                C_inv = np.linalg.inv(C)
            except Exception:
                C_inv = np.linalg.pinv(C)
            rho2_max = float(self.surr_mahalanobis_max ** 2)
            # Elite centers from top-k historical points (by score)
            elite_idx = []
            if X_tr.size > 0:
                order = np.argsort(y_tr)[::-1]
                elite_idx = order[: max(1, min(int(self.surr_elite_top_k), len(order)))]
            elite_centers_t = T_tr[elite_idx, :2] if len(elite_idx) > 0 else np.empty((0, 2))
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
                    std = self.gauss_scale * surr_std_scale * half
                    if not np.isfinite(std) or std <= 0.0:
                        std = max(1e-12, abs(std))
                    val = np.random.normal(0.0, std)
                    u[j] = float(np.clip(val, lo, hi))
                candidates_u.append(u)
            # Elite sampling around top surrogate-frame historical points
            if M_elite > 0 and elite_centers_t.size > 0:
                # derive local scales from cube spans on first two surrogate axes
                span1 = (bounds[0][1] - bounds[0][0])
                span2 = (bounds[1][1] - bounds[1][0]) if d > 1 else 0.0
                std1 = 0.2 * span1
                std2 = 0.2 * span2
                if not np.isfinite(std1) or std1 <= 0.0:
                    std1 = 1e-12
                if not np.isfinite(std2) or std2 <= 0.0:
                    std2 = 1e-12
                for _ in range(M_elite):
                    c_idx = int(np.random.randint(elite_centers_t.shape[0]))
                    base_t2 = elite_centers_t[c_idx]
                    t = np.zeros(d, dtype=float)
                    t[0] = float(base_t2[0] + np.random.normal(0.0, std1))
                    if d > 1:
                        t[1] = float(base_t2[1] + np.random.normal(0.0, std2))
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
                c1, c2 = float(w[1]), float(w[2])
                c3, c4, c5 = float(w[3]), float(w[4]), float(w[5])
                H = np.array([[c3, c5], [c5, c4]], dtype=float)
                g = np.array([c1, c2], dtype=float)
                if np.all(np.isfinite(H)) and np.linalg.cond(H) < 1e6:
                    t_opt = -np.linalg.solve(H, g)
                    # lift t_opt to original space then back to cube and clip into bounds
                    x_opt = (mu_s + R_s @ np.pad(t_opt, (0, max(0, d - 2)), constant_values=0.0)[:d]).astype(float)
                    u_opt = (R_c.T @ (x_opt - mu_c)).astype(float)
                    # clip to cube bounds
                    for j in range(d):
                        lo, hi = bounds[j]
                        u_opt[j] = float(np.clip(u_opt[j], lo, hi))
                    candidates_u.append(u_opt)
            except Exception:
                pass
            # Small cross grid around origin in surrogate frame -> map to cube and clip
            try:
                # use relative radii from cube widths on first two dims
                t1_span = (bounds[0][1] - bounds[0][0])
                t2_span = (bounds[1][1] - bounds[1][0]) if d > 1 else 0.0
                r1 = 0.2 * t1_span
                r2 = 0.2 * t2_span
                grid_t = [( r1, 0.0), (-r1, 0.0)]
                if d > 1:
                    grid_t += [(0.0,  r2), (0.0, -r2)]
                for (g1, g2) in grid_t:
                    t = np.zeros(d, dtype=float)
                    t[0] = float(g1)
                    if d > 1:
                        t[1] = float(g2)
                    xg = (mu_s + R_s @ t).astype(float)
                    u = (R_c.T @ (xg - mu_c)).astype(float)
                    for j in range(d):
                        lo, hi = bounds[j]
                        u[j] = float(np.clip(u[j], lo, hi))
                    candidates_u.append(u)
            except Exception:
                pass
            # Evaluate EI for each candidate: map cube-prime -> original -> surrogate-prime
            local_best = float(cube.best_score if hasattr(cube, 'best_score') else -np.inf)
            ei_vals: List[float] = []
            stats_cache: List[TypingTuple[float, float, float, float, float, np.ndarray]] = []  # (yhat, sigma, ei_raw, ei_eff, rho, x)
            any_inside = False
            for u in candidates_u:
                x = (mu_c + R_c @ u).astype(float)
                t = (R_s.T @ (x - mu_s)).astype(float)
                # Trust region filter in surrogate frame (2D)
                t2 = t[:2]
                # rho^2 = t2^T C_inv t2 (mu assumed ~0 from PCA center)
                try:
                    rho2 = float(t2.T @ (C_inv @ t2))
                except Exception:
                    rho2 = float(np.sum(t2 * t2))
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
                # fall through to geometric sampling below
                pass
            if ei_vals:
                best_idx = int(np.argmax(ei_vals))
                y_hat_best, sigma_best, ei_raw_best, ei_best, rho_best, x_best = stats_cache[best_idx]
                # Accept if EI passes threshold and optional margin gate
                margin = float(getattr(self, 'surr_accept_margin_sigma', -1.0))
                ei_min = float(getattr(self, 'surr_ei_min', 0.0))
                yhat_gate_ok = True if margin < 0 else (y_hat_best >= local_best - margin * sigma_best)
                if (ei_best >= ei_min) and yhat_gate_ok:
                    return np.clip(x_best, 0.0, 1.0)
            # If we reach here without returning from the `if ei_vals:` block above,
            # we fall back to the geometric sampling below.

        # else: fallback to previous geometric sampling
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
            return np.clip(x, 0.0, 1.0)
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
        return np.clip(x, 0.0, 1.0)

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

        # Use local surrogate quality if present to refine gates
        surr = getattr(cube, 'surrogate_2d', None)
        if surr is not None and surr.get('n', 0) >= cube.surrogate_min_points:
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
            # Relax when model is weak
            if r2 < 0.55 or s2 > sigma2_max * 0.75:
                r2_min = 0.5
                ei_min = min(0.008, ei_min * 1.5)
                margin = min(0.3, margin * 1.1)
                rho_pen = min(0.15, rho_pen * 1.2)
                rho_max = min(2.3, rho_max + 0.2)
                elite_frac = max(0.45, elite_frac - 0.05)
                cand = int(min(16384, cand * 1.25))

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
                return leaf
        
        # Always use UCB (best performer)
        return max(self.leaf_cubes, key=lambda c: c.ucb(beta=self.beta, lambda_geo=self.lambda_geo))

    def _maybe_denormalize(self, x_norm: np.ndarray) -> Optional[List[Any]]:
        if self.param_space is None:
            return None
        return self.param_space.denormalize(x_norm)

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
                v *= float(max(hi - lo, 0.0))
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
        return max(children, key=lambda c: c.ucb(beta=self.beta, lambda_geo=self.lambda_geo))

    def run_trial(self, cube: QuadCube, objective_fn: Callable[[np.ndarray, int], Any]) -> None:
        self.trial_id += 1
        self.total_trials += 1

        x_norm = self._sample_biased_in(cube, alpha=0.4, top_k=5)
        cube.add_tested_point(x_norm)
        x_real = self._maybe_denormalize(x_norm)
        
        # Objective call: currently default to normalized space for compatibility.
        # If you need real-domain evaluation (e.g., typed/categorical ParamSpace),
        # adapt here to pass x_real when provided.
        x_for_obj = x_norm

        s_final_raw, artifact = self._call_objective(objective_fn, x_for_obj, epochs=self.full_epochs)
        # Internally use signed score so minimization works seamlessly with UCB/max logic
        s_final = float(self.sign * s_final_raw)
        if not hasattr(cube, "_tested_pairs"):
            cube._tested_pairs: List[TypingTuple[np.ndarray, float]] = []
        cube._tested_pairs.append((np.array(x_norm, dtype=float), float(s_final)))
        cube.update_final(float(s_final))
        # Keep raw scores for external consumers if needed
        self.s_final_all.append(float(s_final_raw))

        improved = bool(s_final > self.best_score_global)
        if improved:
            self.best_score_global = float(s_final)
            self.best_x_norm = x_norm.tolist()
            self.best_x_real = x_real
            if self.on_best is not None:
                try:
                    self.on_best({
                        'trial': self.trial_id,
                        'score': float(s_final_raw),
                        'x_norm': self.best_x_norm,
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

        split_type = cube.should_split(
            min_trials=self.min_trials,
            min_points=self.min_points,
            max_depth=self.max_depth,
            min_width=self.min_width,
            gamma=self.gamma,
        )
        if split_type != 'none':
            children = cube.split4() if split_type == 'quad' else cube.split2()
            if children:
                self.splits_count += 1
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

        # Log raw s_final and raw-scale best score for readability
        best_score_raw = float(self.sign * self.best_score_global)
        self._log([
            self.trial_id, id(cube), json.dumps(x_norm.tolist()),
            self._safe_json(x_real) if x_real is not None else '',
            float(s_final_raw), best_score_raw, len(self.leaf_cubes)
        ])
        # Callback per-trial (fine trial)
        if self.on_trial is not None:
            try:
                self.on_trial({
                    'trial': self.trial_id,
                    'x_norm': x_norm.tolist(),
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
        # Work in signed domain where higher is better regardless of minimize/maximize
        ranked = sorted(self.leaf_cubes, key=lambda c: c.ucb(beta=self.beta, lambda_geo=self.lambda_geo), reverse=True)
        best_signed = float(self.best_score_global)
        margin = float(self.delta_prune)
        keep_thresh = best_signed - margin
        keep: List[QuadCube] = []
        for c in ranked:
            # Grace period: don't prune leaves that are too young
            age = self.trial_id - c.birth_trial
            if age < self.prune_grace_period:
                keep.append(c)
                continue
            
            vol = 1.0
            for lo, hi in c.bounds:
                vol *= max(hi - lo, 0.0)
            too_small = vol < 1e-6
            val = c.ucb(beta=self.beta, lambda_geo=self.lambda_geo)
            ok = (val >= keep_thresh)
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
        for i in range(budget):
            cube = self.select_cube()
            self.run_trial(cube, objective_fn)
