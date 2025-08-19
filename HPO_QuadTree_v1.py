from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, Dict, Tuple as TypingTuple

import json
import csv
import os
import math
import numpy as np

# Riusa ParamSpace dal modulo esistente per coerenza
from cube_hpo import ParamSpace


@dataclass
class QuadCube:
    # Surrogate model cache (2D quadratic on PC1-PC2)
    #Qui creo una scatola dove terrò il “modellino di prova” già pronto, così non devo rifarlo ogni volta. È come avere un disegno già stampato invece di rifarlo a mano ogni volta
    surrogate_2d: Optional[Dict[str, Any]] = field(default=None, init=False)

    #Metodo che adatta una funzione quadratica sulle prime due componenti principali (PC1 e PC2)
    def fit_surrogate(self, min_points: int = 8) -> None: #Prendo i dati che ho già misurato, li guardo da un’angolazione comoda (PCA) e ci disegno sopra una parabola 3D (tipo “colline” e “valli”) che approssima la zona vicina.
        """Fit a 2D quadratic surrogate on PC1-PC2 using local tested pairs.

        Stores ridge solution along with the inverse design matrix to enable
        input-dependent predictive variance at inference time.
        """
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
        ridge_alpha = 1e-3
        A = Phi.T @ Phi + ridge_alpha * np.eye(6)
        b = Phi.T @ y
        try:
            w = np.linalg.solve(A, b)  # Trovo i numeri che fanno combaciare la mia parabola con i punti misurati.
        except Exception:
            self.surrogate_2d = None
            return
        # Pre-compute (Phi^T Phi + alpha I)^-1 for predictive variance
        try:
            A_inv = np.linalg.inv(A)  # Mi salvo un calcolo magico che userò più tardi per sapere “quanto sono sicuro” delle mie previsioni.
        except Exception:
            A_inv = None
        # Residual variance as uncertainty estimate
        y_hat = Phi @ w
        resid = y - y_hat
        sigma2 = float(np.var(resid)) if len(resid) > 1 else 1.0
        # Fit quality (R^2)
        var_y = float(np.var(y)) if len(y) > 1 else 0.0
        r2 = 1.0 - (float(np.var(resid)) / max(var_y, 1e-12)) if var_y > 0 else 0.0
        self.surrogate_2d = {
            'w': w,
            'mu': mu,
            'R': R,
            'A_inv': A_inv,
            'sigma2': sigma2,
            'n': len(y),
            'r2': r2,
            'pca_ok': bool(ok),
        }

    def predict_surrogate(self, x_prime: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std at x_prime (in prime coords) using 2D quadratic surrogate.

        Uses ridge-regression predictive variance: Var[ŷ(x)] ≈ σ² · (φ(x)^T A_inv φ(x) + 1),
        where A_inv = (Φ^T Φ + αI)^-1 saved during fitting and σ² is residual variance.
        If A_inv is unavailable, falls back to constant σ.
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
        # predictive variance term v = φ^T A_inv φ
        try:
            v = float(Phi @ (A_inv @ Phi)) #Calcolo “quanto è lontano/strano” questo punto rispetto a dove ho dati: più è fuori strada, più aumento l’incertezza
            v = max(v, 0.0)
        except Exception:
            v = 0.0
        # Include +1 to approximate target variance (not just mean prediction variance)
        var_pred = sigma2 * (v + 1.0)
        sigma = float(np.sqrt(max(var_pred, 1e-12))) #Prendo l’incertezza di base e ci sommo quanta ne aggiunge il fatto che il punto è “scomodo”. Alla fine ti do la barra d’errore
        return y_hat, sigma
    bounds: List[Tuple[float, float]]  # in local (prime) coordinates, typically symmetric around 0
    parent: Optional["QuadCube"] = None
    children: List["QuadCube"] = field(default_factory=list) #Ogni “cella” è un cubetto in un sistema di assi comodo. I cubetti possono avere un genitore e dei figli, come una cartella con sottocartelle

    # local frame
    R: Optional[np.ndarray] = None  # shape (d, d)
    mu: Optional[np.ndarray] = None  # shape (d,) #mu dice dov’è il centro del cubetto nel mondo reale; R dice come sono ruotati gli assi del cubetto rispetto a quelli normali.

    # statistics
    n_trials: int = 0
    scores: List[float] = field(default_factory=list)
    scores_early: List[float] = field(default_factory=list)
    best_score: float = -np.inf
    mean_score: float = 0.0
    var_score: float = 0.0
    #Quanti esperimenti ho fatto qui? Com’è andata in media? Qual è stato il migliore? Tengo anche qualche appunto rapido per decidere se continuare o fermarmi presto.
    # region params
    #Manopole per decidere quando una cella è promettente o quando è ora di dividerla/abbandonarla. Tipo regole della casa per l’esplorazione.
    prior_var: float = 1.0
    q_threshold: float = 0.0
    early_quantile_p: float = 0.65
    adaptive_early_quantile: bool = False
    stale_steps: int = 0
    depth: int = 0
    early_below_count: int = 0

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
                # assume this cube was created with proper mu; keep as is
                self.mu = np.zeros(d)  # will be overridden immediately by constructors
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
        return np.array([hi - lo for (lo, hi) in self.bounds], dtype=float)
    #Pesco a caso un punto dentro il cubetto locale, con la stessa probabilità ovunque.
    def sample_uniform_prime(self) -> np.ndarray:
        d = len(self.bounds)
        point = np.zeros(d, dtype=float)
        for i, (lo, hi) in enumerate(self.bounds):
            point[i] = np.random.uniform(lo, hi)
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
    def update_early(self, early_score: float) -> None:
        self.scores_early.append(float(early_score))
        n = len(self.scores_early)
        if self.adaptive_early_quantile:
            p = 0.6 + 0.2 * min(1.0, n / 20.0) #Se adaptive_early_quantile=True, p va da 0.6 a 0.8 fino ad un massimo di 20
        else:
            p = self.early_quantile_p #altrimenti usa valore fisso
        if n >= 3:
            self.q_threshold = float(np.quantile(self.scores_early, p)) # per pochi dati propaga dalla soglia del padre se disponibile
        else:
            if self.parent is not None and self.parent.q_threshold > 0:
                self.q_threshold = 0.5 * float(self.parent.q_threshold) + 0.5 * float(self.q_threshold)
        if self.parent is not None:
            self.prior_var = 0.5 * float(self.parent.prior_var) + 0.5 * float(self.prior_var)
        # Refit surrogate after each update if enough data
        self.fit_surrogate()

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
        if self.scores_early:
            vals = np.array(self.scores_early, dtype=float)
            return float(np.max(vals) if mode == 'max' else np.mean(vals))
        return float('-inf')

    def ucb(self, beta: float = 1.6, eps: float = 1e-8, lambda_geo: float = 0.0) -> float:
        n_e = len(self.scores_early) #numero di prove fatte sul serio
        n_eff = self.n_trials + 0.25 * n_e #numero di trials che valgono un quarto
        if self.n_trials > 0:  #Se ho risultati veri, uso quelli. Se ne ho pochi, guardo quanto ballavano gli assaggi; se non ho quasi niente, mi affido a un’idea di massima
            mu = float(self.mean_score)
            if self.n_trials > 1:
                var = float(self.var_score)
            else:
                var = float(np.var(self.scores_early)) if n_e > 1 else float(self.prior_var)
        else:
            mu = float(np.mean(self.scores_early)) if n_e > 0 else 0.0
            var = float(np.var(self.scores_early)) if n_e > 1 else float(self.prior_var)
        if self.parent is not None and self.parent.n_trials > 0: #Se il genitore ha esperienza, lo ascolto per metà. È come chiedere consiglio a chi ha già visto la zona.
            mu = 0.5 * float(self.parent.mean_score) + 0.5 * mu
            pv = float(self.parent.var_score) if self.parent.var_score > 0 else float(self.parent.prior_var)
            var = 0.5 * pv + 0.5 * var
        if n_eff <= 0: #Calcola l’UCB base: media + margine d’incertezza. il margine scende con più dati
            base = float(mu + beta * np.sqrt(var + self.prior_var))
        else: #Più ho provato, più la mia barra d’errore si restringe. Metto sempre un pizzico di incertezza di fondo.
            base = float(mu + beta * np.sqrt(var / (n_eff + eps) + self.prior_var))
        # geometric exploration bonus: depends on cube diameter in prime coords
        w = self._widths() if hasattr(self, "bounds") else np.array([1.0])
        #Bonus geometrico Incentiva l’esplorazione di celle grandi. Se il cubetto è grosso, vale la pena esplorarlo: gli do un extra per farlo sembrare più appetibile.
        diam = float(np.sqrt(np.sum(np.square(w))))
        return base + lambda_geo * diam

    def should_split(self,
                     min_trials: int = 5,
                     min_points: int = 10,
                     max_depth: Optional[int] = 4,
                     min_width: float = 1e-3,
                     gamma: float = 0.02) -> str: #decide se e come dividere il cubo
        #Regole: non spacco troppo in profondità, non spacco briciole, e non spacco se non ho abbastanza prove o il guadagno è risibile
        # stop per profondità/ampiezza
        if max_depth is not None and self.depth >= max_depth:
            return 'none'
        #se hai raggiunto la profondità massima o se tutte le dimensioni sono più strette di min_width, non dividere
        widths = [hi - lo for (lo, hi) in self.bounds]
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
        split_type = 'quad' if npts >= max(10, min_points) else 'binary'

        # Info-gain: split solo se riduzione varianza residua > gamma
        # Solo se surrogato disponibile e almeno 2 figli
        #Applica un criterio di information gain basato sulla varianza residua del surrogato: richiede un surrogato con almeno 12 punti
        #Faccio una prova mentale: se taglio, i pezzi nuovi mi riducono abbastanza l’incertezza? Se il guadagno è piccolo, non spacco; se è decente, procedo col tipo di taglio deciso prima.
        if self.surrogate_2d is not None and self.surrogate_2d['n'] >= 12:
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
        widths = self._widths()
        # prende il frame PCA locale (R, mu). Calcola il punto di taglio lungo l’asse ax con _quad_cut_along_axis
        ax = int(np.argmax(widths))
        R, mu, _, ok = self._principal_axes()
        cut = self._quad_cut_along_axis(ax, R, mu)
        lo, hi = self.bounds[ax]
        cut = float(np.clip(cut, lo + 1e-12, hi - 1e-12))
        # Assegna punti ai due figli
        pairs = getattr(self, "_tested_pairs", [])
        if not pairs:
            return []
        #recupera i punti testati X e i loro punteggi y, poi li proietta in coordinate prime T
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        T = (R.T @ (X.T - mu.reshape(-1, 1))).T
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
            ridge_alpha = 1e-3
            A = Phi.T @ Phi + ridge_alpha * np.eye(6)
            b = Phi.T @ y[idx]
            try:
                w = np.linalg.solve(A, b)
                y_hat = Phi @ w
                resid = y[idx] - y_hat
                var = float(np.var(resid)) if len(resid) > 1 else 1.0
            except Exception:
                var = 1.0
            children.append({'n': len(idx), 'var': var})
        return children

    def _simulate_split4(self):
        # Simula split4 e fitta surrogato su ciascun figlio (PC1/PC2)
        d = len(self.bounds)
        widths = self._widths()
        R, mu, _, ok = self._principal_axes()
        # se la PCA è ok usa PC1 (0) e PC2 (1, o 0 se 1D) come assi di taglio
        if ok:
            ax_i, ax_j = 0, 1 if d > 1 else 0
        else:
            # altrimenti, prendi i due assi più larghi dai bounds (ordine come implementazione legacy)
            top2 = np.argsort(widths)[-2:]
            ax_i, ax_j = int(top2[0]), int(top2[1])
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R, mu)
            cut_j = self._quad_cut_along_axis(ax_j, R, mu)
        else:
            lo_i, hi_i = self.bounds[ax_i]
            cut_i = 0.5 * (lo_i + hi_i)
            lo_j, hi_j = self.bounds[ax_j]
            cut_j = 0.5 * (lo_j + hi_j)
        # Clipping legacy nei bounds locali del cubo (coerente con versione debug per confronto 1:1)
        lo_i, hi_i = self.bounds[ax_i]
        lo_j, hi_j = self.bounds[ax_j]
        cut_i = float(np.clip(cut_i, lo_i + 1e-12, hi_i - 1e-12))
        cut_j = float(np.clip(cut_j, lo_j + 1e-12, hi_j - 1e-12))
        pairs = getattr(self, "_tested_pairs", [])
        if not pairs:
            return []
        X = np.array([p for (p, s) in pairs], dtype=float)
        y = np.array([s for (p, s) in pairs], dtype=float)
        T = (R.T @ (X.T - mu.reshape(-1, 1))).T
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
            ridge_alpha = 1e-3
            A = Phi.T @ Phi + ridge_alpha * np.eye(6)
            b = Phi.T @ y[idx]
            try:
                w = np.linalg.solve(A, b)
                y_hat = Phi @ w
                resid = y[idx] - y_hat
                var = float(np.var(resid)) if len(resid) > 1 else 1.0
            except Exception:
                var = 1.0
            children.append({'n': len(idx), 'var': var})
        return children

    def _principal_axes(self,
                        q_good: float = 0.3,
                        min_points: int = 10,
                        anisotropy_threshold: float = 1.4,
                        depth_min: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (R, mu, eigvals, ok) where R columns are principal axes, mu is center from top-q% points.
        ok indicates if PCA was applied (enough points and anisotropy high enough and depth >= depth_min).
        """
        """Restituisce:

        R: matrice i cui vettori colonna sono gli assi principali (PCA).

        mu: centro calcolato dai migliori punti (top q% per score).

        eigvals: autovalori (varianze lungo PC).

        ok: True se la PCA è stata applicata (abbastanza punti, anisotropia sufficiente, profondità sufficiente).

        Usa solo i top q_good per stimare la PCA; serve a catturare la direzione “utile” dove i risultati sono migliori.
        Cerchiamo gli “occhiali” migliori per guardare i dati: ruotiamo gli assi verso dove 
        c'è più variazione nei migliori punti. Se la forma non ha direzioni preferite (tutto un po' uguale), non ha senso ruotare gli occhiali: restiamo com'eravamo
        """
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
        # select good points by score (descending)
        #Seleziona i top k per score.
        k = max(5, int(np.ceil(q_good * len(pts))))
        idx = np.argsort(scs)[::-1][:k]
        P = pts[idx]
        #Centro in mu (media dei top) e calcola la covarianza.
        mu = P.mean(axis=0)
        Z = P - mu
        # covariance
        C = np.cov(Z.T) if Z.shape[0] > 1 else np.eye(d)
        # eigh returns ascending; we want descending
        #Decomposizione agli autovalori/vettori: ordina in decrescente (PC1 = direzione di massima varianza).Prendo i punti migliori, li metto “allineati” al centro e scopro da che parte si allargano di più.
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
                              mu: np.ndarray,
                              ridge_alpha: float = 1e-3) -> float:
        """Choose a cut point along given prime axis using 1D quadratic fit over projections.
        Falls back to midpoint if fit is poor or data insufficient.
        Returns t_cut in prime coords, clipped to [lo, hi] for that axis.
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
        widths = self._widths()
        #Taglio dove c’è più spazio (o dove mi dici tu)
        ax = int(np.argmax(widths)) if axis is None else int(axis)
        # compute PCA axes and quadratic cut if possible
        #Prova a usare PCA locale per decidere il punto di taglio (via _quad_cut_along_axis
        R_loc, mu_loc, _, ok = self._principal_axes()
        cut = self._quad_cut_along_axis(ax, R_loc if ok else self.R, mu_loc if ok else self.mu)
        lo, hi = self.bounds[ax]
        cut = float(np.clip(cut, lo + 1e-12, hi - 1e-12))
        # child prime bounds before re-centering
        nb_left = list(self.bounds)
        nb_right = list(self.bounds)
        nb_left[ax] = (lo, cut)
        nb_right[ax] = (cut, hi)
        # compute child centers in prime coords
        #Definisce i nuovi bounds in prime coords per i due figli e ne calcola centri e larghezze.
        center_left_prime = np.array([(a + b) * 0.5 for (a, b) in nb_left], dtype=float)
        center_right_prime = np.array([(a + b) * 0.5 for (a, b) in nb_right], dtype=float)
        # widths for children
        w_left = np.array([b - a for (a, b) in nb_left], dtype=float)
        w_right = np.array([b - a for (a, b) in nb_right], dtype=float)
        # build children: re-center prime boxes around 0 and set mu accordingly
        #Ogni figlio ha il suo “mondo comodo” centrato; poi lo piazzo correttamente nel mondo reale.
        c1 = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in w_left], parent=self,
                      early_quantile_p=self.early_quantile_p, adaptive_early_quantile=self.adaptive_early_quantile)
        c2 = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in w_right], parent=self,
                      early_quantile_p=self.early_quantile_p, adaptive_early_quantile=self.adaptive_early_quantile)
        for (ch, ctr_prime) in ((c1, center_left_prime), (c2, center_right_prime)):
            ch.R = (R_loc if ok else self.R).copy()
            ch.mu = (self.mu + (ch.R @ ctr_prime)).astype(float)
            ch.prior_var = float(self.prior_var)
            ch.q_threshold = float(self.q_threshold)
            ch.depth = self.depth + 1
            ch._tested_points = []
        # redistribute points/pairs
        #Sposto ogni prova fatta finora nel figlio giusto (sinistra o destra) guardando dove cade rispetto al taglio
        points = np.array(self._points_history) if self._points_history else np.empty((0, len(self.bounds)))
        if points.size > 0:
            # assign by parent frame & cut
            T = ( (R_loc if ok else self.R).T @ (points.T - (mu_loc if ok else self.mu).reshape(-1,1)) ).T
            mask_left = T[:, ax] < cut
            for p, m in zip(points, mask_left):
                (c1._tested_points if m else c2._tested_points).append(np.array(p, dtype=float))
        pairs = getattr(self, "_tested_pairs", [])
        c1._tested_pairs, c2._tested_pairs = [], []
        for (pt, s) in pairs:
            t = float(((R_loc if ok else self.R).T @ (pt - (mu_loc if ok else self.mu)))[ax])
            (c1._tested_pairs if t < cut else c2._tested_pairs).append((np.array(pt, dtype=float), float(s)))
        self.children = [c1, c2]
        return self.children

    def split4(self) -> List["QuadCube"]:
        # 4-way split using PCA local axes; cut-points from quadratic 1D fits (fallback to midpoints)
        d = len(self.bounds)
        if d == 1:
            return self.split2(axis=0)
        self._ensure_frame()
        widths = self._widths()
        # principal axes and local center
        R_loc, mu_loc, _, ok = self._principal_axes()
        # choose two axes: first two components if ok, else widest two in prime frame
        #Se la PCA è ok usa PC1 (0) e PC2 (1, o 0 se 1D) come assi di taglio
        if ok:
            ax_i, ax_j = 0, 1 if d > 1 else 0
        else:
            top2 = np.argsort(widths)[-2:]
            ax_i, ax_j = int(top2[0]), int(top2[1])
        # compute cutpoints
        #Prova a usare PCA locale per decidere i punti di taglio (via _quad_cut_along_axis)
        if ok:
            cut_i = self._quad_cut_along_axis(ax_i, R_loc, mu_loc)
            cut_j = self._quad_cut_along_axis(ax_j, R_loc, mu_loc)
        else:
            lo_i, hi_i = self.bounds[ax_i]; cut_i = 0.5 * (lo_i + hi_i)
            lo_j, hi_j = self.bounds[ax_j]; cut_j = 0.5 * (lo_j + hi_j)
        # clip cuts inside bounds
        lo_i, hi_i = self.bounds[ax_i]
        lo_j, hi_j = self.bounds[ax_j]
        cut_i = float(np.clip(cut_i, lo_i + 1e-12, hi_i - 1e-12))
        cut_j = float(np.clip(cut_j, lo_j + 1e-12, hi_j - 1e-12))
        # child prime bounds before re-centering
        #Costruisce i quattro bounds (quadranti) in prime coords: (sinistra/destra) × (sotto/sopra)
        def make_bounds(quadrant: TypingTuple[bool, bool]) -> List[Tuple[float, float]]:
            bi = (lo_i, cut_i) if quadrant[0] else (cut_i, hi_i)
            bj = (lo_j, cut_j) if quadrant[1] else (cut_j, hi_j)
            nb = list(self.bounds)
            nb[ax_i] = bi
            nb[ax_j] = bj
            return nb
        b_q1 = make_bounds((True, True))
        b_q2 = make_bounds((False, True))
        b_q3 = make_bounds((True, False))
        b_q4 = make_bounds((False, False))
        # centers and widths for children
        #Per ogni quadrante: calcola centro e larghezze, crea il figlio con bounds ricentrati attorno a 0, e mappa il centro in spazio originale: mu_child = mu_parent + R @ center_prime_quadrante
        centers_prime = [np.array([(a + b) * 0.5 for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        widths_children = [np.array([b - a for (a, b) in nb], dtype=float) for nb in (b_q1, b_q2, b_q3, b_q4)]
        # instantiate children with recentered prime boxes
        children: List[QuadCube] = []
        for ctr_p, wch in zip(centers_prime, widths_children):
            ch = QuadCube(bounds=[(-wi/2.0, wi/2.0) for wi in wch], parent=self,
                          early_quantile_p=self.early_quantile_p, adaptive_early_quantile=self.adaptive_early_quantile)
            ch.R = (R_loc if ok else self.R).copy()
            ch.mu = (self.mu + (ch.R @ ctr_p)).astype(float)
            ch.prior_var = float(self.prior_var)
            ch.q_threshold = float(self.q_threshold)
            ch.depth = self.depth + 1
            ch._tested_points = []
            children.append(ch)
        # redistribute historical points/pairs to children by quadrant in parent frame
        points = np.array(self._points_history) if self._points_history else np.empty((0, len(self.bounds)))
        if points.size > 0:
            T = ((R_loc if ok else self.R).T @ (points.T - (mu_loc if ok else self.mu).reshape(-1, 1))).T
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
            t = ((R_loc if ok else self.R).T @ (pt - (mu_loc if ok else self.mu)))
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
        min_trials: int = 12,
        gamma: float = 0.02,
        beta: float = 0.5,
        lambda_geo: float = 0.05,
        early_epochs: int = 0,
        full_epochs: int = 50,
        rng_seed: Optional[int] = None,
        param_space: Optional[ParamSpace] = None,
        objective_in_normalized_space: bool = True,
        log_path: Optional[str] = None,
        early_quantile_p: float = 0.65,
        adaptive_early_quantile: bool = False,
        on_best: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
        best_child_metric: str = "leaf_max",
        probes_per_child_on_split: int = 1,
        prefer_best_of_four: bool = True,
        disable_prune: bool = False,
        disable_split_probes: bool = True,
        debug_assert_bounds: bool = True,
    cube_select_mode: str = "ucb",
    maximize: bool = True,
    budget_in_objective_calls: bool = False,
        ) -> None:
            # Core config and state
            self.root = QuadCube(list(bounds))
            self.leaf_cubes: List[QuadCube] = [self.root]
            self.min_trials = int(min_trials)
            self.gamma = float(gamma)
            self.beta = float(beta)
            self.lambda_geo = float(lambda_geo)
            self.early_epochs = int(early_epochs)
            self.full_epochs = int(full_epochs)
            self.param_space = param_space
            self.objective_in_normalized_space = bool(objective_in_normalized_space)
            self.log_path = log_path
            self.on_best = on_best
            self.on_trial = on_trial
            self.best_score_global = -np.inf
            self.best_x_norm: Optional[List[float]] = None
            self.best_x_real: Optional[List[Any]] = None
            self.trial_id: int = 0

            self._preferred_leaf: Optional[QuadCube] = None
            self.stale_steps_max = 15
            self.delta_prune = 0.025
            self.max_depth = 4
            self.min_width = 1e-3
            self.min_points = 10
            self.min_leaves = 5

            self.best_child_metric = str(best_child_metric)
            self.probes_per_child_on_split = int(probes_per_child_on_split)
            self.prefer_best_of_four = bool(prefer_best_of_four)
            self.disable_prune = bool(disable_prune)
            self.disable_split_probes = bool(disable_split_probes)
            self.objective_calls: int = 0
            self._budget_in_objective_calls = bool(budget_in_objective_calls)
            self.maximize = bool(maximize)
            self.sign = 1.0 if self.maximize else -1.0
            # alias richiesto
            self.obj_calls = self.objective_calls
            # Propaga flag di assert ai cube
            self._debug_assert_bounds = bool(debug_assert_bounds)
            self.root._debug_assert_bounds = bool(debug_assert_bounds)

            self.root.early_quantile_p = float(early_quantile_p)
            self.root.adaptive_early_quantile = bool(adaptive_early_quantile)
            if rng_seed is not None:
                np.random.seed(int(rng_seed))

            # Aggregates and logs
            self.total_trials: int = 0
            self.early_stops: int = 0
            self.s_early_all: List[float] = []
            self.s_final_all: List[float] = []
            self.s_early_pass: List[float] = []
            self.splits_count: int = 0
            self.prunes_count: int = 0
            self.split_probes: int = 0
            if self.log_path and not os.path.exists(self.log_path):
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'trial', 'cube_id', 'x_norm', 'x_real', 's_early', 'q_threshold',
                        'early_stop', 's_final', 'best_score_global', 'n_leaves'
                    ])
            # No trial-level debug files in production

            # Early-stop controls (coerenti con la versione equal)
            self.early_min_points_for_stop: int = 5
            self.early_stop_patience: int = 1
            self.early_stop_margin: float = 0.0

            # PCA/Quadratic triggers and sampling controls
            self.pca_q_good: float = 0.3
            self.pca_min_points: int = 7
            self.anisotropy_threshold: float = 1.4
            self.depth_min_for_pca: int = 1
            self.line_search_prob: float = 0.25
            self.gauss_scale: float = 0.35

            # Surrogate acquisition knobs (tunable without behavior change by default)
            self.surr_sigma2_max: float = 12.0   # gate surrogate usage if residual var too high
            self.surr_r2_min: float = 0.0        # optional gate by fit quality (R^2)
            self.surr_candidates: int = 24       # number of random prime-candidates to evaluate
            # Acceptance controls: EI-only by default (margin < 0 disables yhat gate)
            self.surr_accept_margin_sigma: float = -1.0  # if >= 0, require yhat >= local_best - margin*sigma
            self.surr_ei_min: float = 0.0                 # minimal EI to accept
            # Surrogate trust region and elite sampling (in surrogate PCA frame)
            self.surr_mahalanobis_max: float = 2.5        # discard candidates with rho > this (rho^2 Mahalanobis distance)
            self.surr_elite_frac: float = 0.4             # fraction of candidates sampled near top historical points
            self.surr_elite_top_k: int = 5                # top-k points (by score) to form elite centers
            # Penalization of EI by distance in surrogate frame (rho^2 Mahalanobis); ei_eff = ei / (1 + c*rho2)
            self.surr_ei_rho2_penalty: float = 0.5

            # Surrogate mode: 'auto' adapts knobs dynamically, 'manual' uses set values
            self.surr_mode: str = 'auto'
            # Internal: baseline presets for auto mode (may be adjusted per-dimension)
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

            # initialize root frame: identity rotation, center at original cube center; prime bounds symmetric
            d = len(bounds)
            root_lo = np.array([lo for (lo, hi) in bounds], dtype=float)
            root_hi = np.array([hi for (lo, hi) in bounds], dtype=float)
            root_mu = (root_lo + root_hi) * 0.5
            root_w = (root_hi - root_lo)
            self.root.R = np.eye(d)
            self.root.mu = root_mu
            self.root.bounds = [(-wi / 2.0, wi / 2.0) for wi in root_w]

            # Acquisition strategy at cube selection level: 'ucb' (default) or 'thompson'.
            # Accept alias 'tomphson' for convenience.
            try:
                mode_norm = str(cube_select_mode or 'ucb').strip().lower()
                if mode_norm == 'tomphson':
                    mode_norm = 'thompson'
                if mode_norm not in ('ucb', 'thompson'):
                    mode_norm = 'ucb'
                self.cube_select_mode = mode_norm
            except Exception:
                self.cube_select_mode = 'ucb'

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
            and cube.surrogate_2d.get('n', 0) >= 10
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
                    half = (hi - lo) * 0.5
                    std = self.gauss_scale * surr_std_scale * half
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
        if surr is not None and surr.get('n', 0) >= 8:
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
    # no debug prints in production

    def select_cube(self) -> QuadCube:
        # Add Thompson sampling at cube level (optional: mode switch)
        use_thompson = hasattr(self, 'cube_select_mode') and getattr(self, 'cube_select_mode', None) == 'thompson'
        if self.prefer_best_of_four and self._preferred_leaf is not None:
            if any(c is self._preferred_leaf for c in self.leaf_cubes):
                leaf = self._preferred_leaf
                self._preferred_leaf = None
                return leaf
        if use_thompson:
            samples = []
            for c in self.leaf_cubes:
                mu = float(c.mean_score) if c.n_trials > 0 else 0.0
                var = float(c.var_score) if c.n_trials > 1 else float(c.prior_var)
                samples.append(np.random.normal(mu, np.sqrt(max(var, 1e-8))))
            return self.leaf_cubes[int(np.argmax(samples))]
        else:
            ucb_values = [c.ucb(beta=self.beta, lambda_geo=self.lambda_geo) for c in self.leaf_cubes]
            return self.leaf_cubes[int(np.argmax(ucb_values))]

    def _maybe_denormalize(self, x_norm: np.ndarray) -> Optional[List[Any]]:
        if self.param_space is None:
            return None
        return self.param_space.denormalize(x_norm)

    def _call_objective(self, objective_fn: Callable[[np.ndarray, int], Any], x: np.ndarray, epochs: int) -> TypingTuple[float, Optional[Any]]:
        # Incrementa contatore globale chiamate objective (conteggia anche early e final & probes)
        self.objective_calls += 1
        self.obj_calls = self.objective_calls  # sync alias
        res = objective_fn(x, epochs=epochs)
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
                v *= float(hi - lo)
            vols.append(v)
        return vols

    def diagnostics(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        total = max(1, self.total_trials)
        res['early_stop_rate'] = float(self.early_stops) / float(total)
        corr = None
        if len(self.s_early_pass) > 1 and len(self.s_final_all) > 1 and len(self.s_early_pass) == len(self.s_final_all):
            try:
                corr = float(np.corrcoef(np.array(self.s_early_pass), np.array(self.s_final_all))[0, 1])
            except Exception:
                corr = None
        res['corr_early_final'] = corr
        fn_proxy = None
        if self.s_early_pass:
            thr_pass = float(np.median(self.s_early_pass))
            early_stopped = [e for e in self.s_early_all if e not in self.s_early_pass]
            if early_stopped:
                fn = sum(1 for e in early_stopped if e >= thr_pass)
                fn_proxy = fn / float(len(early_stopped))
        res['fn_proxy_rate'] = fn_proxy
        res['splits'] = self.splits_count
        res['prunes'] = self.prunes_count
        res['split_probes'] = self.split_probes
        if self.leaf_cubes:
            res['max_depth'] = max(self._cube_depth(c) for c in self.leaf_cubes)
            res['leaf_volumes'] = self.leaf_volumes()
        else:
            res['max_depth'] = 0
            res['leaf_volumes'] = []
        return res

    def _probe_child_once(self, child: QuadCube, objective_fn: Callable[[np.ndarray, int], Any]) -> None:
        # Sample a probe point in the child and log the target leaf before calling the objective
        x_norm = self._sample_biased_in(child, alpha=0.0)
        child.add_tested_point(x_norm)
        x_real = self._maybe_denormalize(x_norm)
        x_for_obj = x_norm if self.objective_in_normalized_space or x_real is None else np.array(x_real, dtype=float)
        s_early, _ = self._call_objective(objective_fn, x_for_obj, epochs=self.early_epochs)
        if not hasattr(child, "_tested_pairs"):
            child._tested_pairs: List[TypingTuple[np.ndarray, float]] = []
        child._tested_pairs.append((np.array(x_norm, dtype=float), float(s_early)))
        child.update_early(s_early)
        self.split_probes += 1

    def _pick_best_child(self, children: List[QuadCube]) -> QuadCube:
        if self.best_child_metric == "ucb":
            return max(children, key=lambda c: c.ucb(beta=self.beta, lambda_geo=self.lambda_geo))
        if self.best_child_metric == "leaf_mean":
            return max(children, key=lambda c: c.leaf_score(mode='mean'))
        return max(children, key=lambda c: c.leaf_score(mode='max'))

    def run_trial(self, cube: QuadCube, objective_fn: Callable[[np.ndarray, int], Any]) -> None:
        self.trial_id += 1
        self.total_trials += 1

        x_norm = self._sample_biased_in(cube, alpha=0.4, top_k=5)
        cube.add_tested_point(x_norm)
        x_real = self._maybe_denormalize(x_norm)
        x_for_obj = x_norm if self.objective_in_normalized_space or x_real is None else np.array(x_real, dtype=float)

        use_early = self.early_epochs > 0

        s_early = None
        if use_early:
            s_early, _ = self._call_objective(objective_fn, x_for_obj, epochs=self.early_epochs)
            if not hasattr(cube, "_tested_pairs"):
                cube._tested_pairs: List[TypingTuple[np.ndarray, float]] = []
            cube._tested_pairs.append((np.array(x_norm, dtype=float), float(s_early)))
            self.s_early_all.append(s_early)
            cube.update_early(s_early)

        early_stop = False
        if use_early and len(cube.scores_early) >= int(self.early_min_points_for_stop):
            thr = float(cube.q_threshold) - float(self.early_stop_margin)
            if s_early is not None and s_early < thr:
                cube.early_below_count = int(cube.early_below_count) + 1
            else:
                cube.early_below_count = 0
            if cube.early_below_count > int(self.early_stop_patience):
                early_stop = True

        if early_stop:
            self.early_stops += 1
            for c in self.leaf_cubes:
                c.stale_steps += 1
            self._log([
                self.trial_id, id(cube), json.dumps(x_norm.tolist()),
                self._safe_json(x_real) if x_real is not None else '',
                s_early, cube.q_threshold, True, '', self.best_score_global, len(self.leaf_cubes)
            ])
            # Callback per-trial (early stop)
            if self.on_trial is not None:
                try:
                    self.on_trial({
                        'trial': self.trial_id,
                        'x_norm': x_norm.tolist(),
                        'x_real': x_real,
                        's_early': s_early,
                        's_final': None,
                        'early_stop': True,
                        'leaf_id': id(cube),
                        'objective_calls': self.objective_calls,
                    })
                except Exception:
                    pass
            return

        # Capture prev best before final eval
        prev_best = float(self.best_score_global)
        s_final, artifact = self._call_objective(objective_fn, x_for_obj, epochs=self.full_epochs)
        # Se non usiamo early, registriamo comunque il pair con il valore finale (per surrogate/storia)
        if not use_early:
            if not hasattr(cube, "_tested_pairs"):
                cube._tested_pairs: List[TypingTuple[np.ndarray, float]] = []
            cube._tested_pairs.append((np.array(x_norm, dtype=float), float(s_final)))
        cube.update_final(s_final)
        self.s_final_all.append(s_final)
        if s_early is not None:
            self.s_early_pass.append(s_early)

        improved = bool(s_final > self.best_score_global)
        if improved:
            self.best_score_global = s_final
            self.best_x_norm = x_norm.tolist()
            self.best_x_real = x_real
            if self.on_best is not None:
                try:
                    self.on_best({
                        'trial': self.trial_id,
                        'score': s_final,
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
                # Use identity check to avoid numpy array ambiguity
                for i, c in enumerate(self.leaf_cubes):
                    if c is cube:
                        self.leaf_cubes.pop(i)
                        break
                for child in children:
                    self.leaf_cubes.append(child)
                if (not self.disable_split_probes) and self.probes_per_child_on_split > 0:
                    for ch in children:
                        for _ in range(self.probes_per_child_on_split):
                            self._probe_child_once(ch, objective_fn)

                # keep track of the best child to bias next selection
                best_child = self._pick_best_child(children)
                self._preferred_leaf = best_child

        if not self.disable_prune:
            prev = len(self.leaf_cubes)
            self.prune_cubes()
            removed = max(0, prev - len(self.leaf_cubes))
            self.prunes_count += removed

        self._log([
            self.trial_id, id(cube), json.dumps(x_norm.tolist()),
            self._safe_json(x_real) if x_real is not None else '',
            s_early, cube.q_threshold, False, s_final, self.best_score_global, len(self.leaf_cubes)
        ])
        # Callback per-trial (fine trial)
        if self.on_trial is not None:
            try:
                self.on_trial({
                    'trial': self.trial_id,
                    'x_norm': x_norm.tolist(),
                    'x_real': x_real,
                    's_early': s_early,
                    's_final': s_final,
                    'early_stop': False,
                    'leaf_id': id(cube),
                    'objective_calls': self.objective_calls,
                })
            except Exception:
                pass
    # No trial-level debug file/prints in production

    def prune_cubes(self) -> None:
        if len(self.leaf_cubes) <= self.min_leaves:
            return
        ranked = sorted(self.leaf_cubes, key=lambda c: c.ucb(beta=self.beta, lambda_geo=self.lambda_geo), reverse=self.maximize)
        best = float(self.best_score_global)
        margin = float(self.delta_prune)
        if self.maximize:
            keep_thresh = best - margin
        else:
            keep_thresh = best + margin
        keep: List[QuadCube] = []
        for c in ranked:
            vol = 1.0
            for lo, hi in c.bounds:
                vol *= max(hi - lo, 0.0)
            too_small = vol < 1e-6
            val = c.ucb(beta=self.beta, lambda_geo=self.lambda_geo)
            ok = (val >= keep_thresh) if self.maximize else (val <= keep_thresh)
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
        budget = int(budget)
        if self._budget_in_objective_calls:
            # Loop finchè numero chiamate objective < budget
            while self.objective_calls < budget:
                cube = self.select_cube()
                self.run_trial(cube, objective_fn)
        else:
            for i in range(budget):
                cube = self.select_cube()
                self.run_trial(cube, objective_fn)
    # No console summary in production
