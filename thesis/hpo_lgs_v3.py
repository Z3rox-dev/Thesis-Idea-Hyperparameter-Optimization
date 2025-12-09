"""
HPO-LGS v3 - Local Geometry Score NON-PARAMETRICO

Tutti i componenti si auto-calibrano sui dati, nessun peso da tunare.

Idea chiave:
- Ogni componente restituisce un RANK tra i candidati, non uno score assoluto
- Lo score finale è la MEDIA dei RANK normalizzati
- Così ogni componente ha uguale importanza indipendentemente dalla scala

Componenti:
1. DFB (Distance from Bad): rank per distanza dai punti cattivi
2. DTC (Distance to Top): rank per vicinanza ai TOP-k migliori
3. NOV (Novelty): rank per distanza dai punti già visitati
4. GRD (Gradient): rank per allineamento con direzione miglioramento

Autore: Z3rox-dev
Data: Dicembre 2025
"""
from __future__ import annotations
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
    lgs_model: Optional[dict] = field(default=None, init=False)
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

    def fit_lgs_model(self, gamma: float, dim: int) -> None:
        """Costruisce modello geometrico dai dati - MIGLIORATO per paesaggi complessi."""
        pairs = list(self._tested_pairs)
        
        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, '_tested_pairs', [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
                pairs = pairs + extra[:needed]
        
        if len(pairs) < dim + 2:
            self.lgs_model = None
            return
        
        all_pts = np.array([p for p, s in pairs])
        all_scores = np.array([s for p, s in pairs])
        
        # Top-k punti (i migliori) - adattivo
        k = max(5, min(len(pairs) // 4, 15))  # Tra 5 e 15 punti
        top_k_idx = np.argsort(all_scores)[-k:]
        top_k_pts = all_pts[top_k_idx]
        top_k_scores = all_scores[top_k_idx]
        
        # Punti cattivi (sotto gamma)
        bad_mask = all_scores < gamma
        bad_pts = all_pts[bad_mask] if bad_mask.any() else None
        
        # MIGLIORAMENTO: Stima gradiente con focus sui top points
        gradient_dir = None
        gradient_confidence = 0.0
        if len(pairs) >= dim + 3:
            widths = np.maximum(self._widths(), 1e-9)
            center = self.center()
            
            # Prova due approcci e scegli il migliore
            gradients = []
            
            # Approccio 1: Regressione lineare su tutti i punti (robusto)
            X_norm = (all_pts - center) / widths
            y_centered = all_scores - all_scores.mean()
            try:
                XtX = X_norm.T @ X_norm + 0.01 * np.eye(dim)
                Xty = X_norm.T @ y_centered
                grad1 = np.linalg.solve(XtX, Xty)
                grad1_norm = np.linalg.norm(grad1)
                if grad1_norm > 1e-9:
                    gradients.append(grad1 / grad1_norm)
            except:
                pass
            
            # Approccio 2: Weighted regression (più peso ai top points)
            if len(top_k_pts) >= dim + 1:
                try:
                    # Pesi esponenziali: più alto lo score, più peso
                    score_range = top_k_scores.max() - top_k_scores.min()
                    if score_range > 1e-6:  # Check for non-degenerate scores
                        weights = np.exp(2 * (top_k_scores - top_k_scores.min()) / score_range)
                        weights = weights / weights.sum()
                    else:
                        # All scores similar, use uniform weights
                        weights = np.ones(len(top_k_scores)) / len(top_k_scores)
                    
                    X_top_norm = (top_k_pts - center) / widths
                    y_top_centered = top_k_scores - top_k_scores.mean()
                    
                    W_sqrt = np.sqrt(weights)
                    X_weighted = X_top_norm * W_sqrt[:, np.newaxis]
                    y_weighted = y_top_centered * W_sqrt
                    
                    XtX = X_weighted.T @ X_weighted + 0.01 * np.eye(dim)
                    Xty = X_weighted.T @ y_weighted
                    grad2 = np.linalg.solve(XtX, Xty)
                    grad2_norm = np.linalg.norm(grad2)
                    if grad2_norm > 1e-9:
                        gradients.append(grad2 / grad2_norm)
                except:
                    pass
            
            # Approccio 3: Direzione dal centroide bad verso centroide top (semplice)
            if bad_pts is not None and len(bad_pts) > 0 and len(top_k_pts) > 0:
                bad_center = bad_pts.mean(axis=0)
                top_center = top_k_pts.mean(axis=0)
                grad3 = (top_center - bad_center) / widths
                grad3_norm = np.linalg.norm(grad3)
                if grad3_norm > 1e-9:
                    gradients.append(grad3 / grad3_norm)
            
            # Combina i gradienti con media (ensemble)
            if len(gradients) > 0:
                gradient_dir = np.mean(gradients, axis=0)
                gradient_dir = gradient_dir / (np.linalg.norm(gradient_dir) + 1e-9)
                gradient_confidence = len(gradients) / 3.0  # Confidence in [0, 1]
        
        self.lgs_model = {
            'all_pts': all_pts,
            'all_scores': all_scores,
            'top_k_pts': top_k_pts,
            'bad_pts': bad_pts,
            'gradient_dir': gradient_dir,
            'gradient_confidence': gradient_confidence,
            'widths': np.maximum(self._widths(), 1e-9),
            'center': self.center(),
        }
    
    def rank_candidates_lgs(self, candidates: List[np.ndarray]) -> np.ndarray:
        """
        Assegna uno score a ogni candidato basato sulla MEDIA dei RANK.
        Completamente non-parametrico.
        """
        n = len(candidates)
        if n == 0:
            return np.array([])
        
        if self.lgs_model is None:
            return np.ones(n) / n
        
        model = self.lgs_model
        widths = model['widths']
        center = model['center']
        
        # Normalizza candidati
        cands_norm = np.array([(c - center) / widths for c in candidates])
        
        # Collezioniamo i rank per ogni criterio
        all_ranks = []
        
        # ===== 1. DFB: Distance from Bad =====
        # Più lontano dai cattivi = meglio
        if model['bad_pts'] is not None and len(model['bad_pts']) > 0:
            bad_norm = (model['bad_pts'] - center) / widths
            dfb_scores = []
            for c in cands_norm:
                dists = np.linalg.norm(bad_norm - c, axis=1)
                dfb_scores.append(np.min(dists))  # Min distance to any bad
            dfb_ranks = self._to_ranks(dfb_scores, higher_is_better=True)
            all_ranks.append(dfb_ranks)
        
        # ===== 2. DTC: Distance to Top-k =====
        # Più vicino ai migliori = meglio
        if model['top_k_pts'] is not None and len(model['top_k_pts']) > 0:
            top_norm = (model['top_k_pts'] - center) / widths
            dtc_scores = []
            for c in cands_norm:
                dists = np.linalg.norm(top_norm - c, axis=1)
                dtc_scores.append(np.min(dists))  # Min distance to any top
            dtc_ranks = self._to_ranks(dtc_scores, higher_is_better=False)
            all_ranks.append(dtc_ranks)
        
        # ===== 3. NOV: Novelty =====
        # Più lontano da TUTTI i punti visitati = meglio (esplorazione)
        if model['all_pts'] is not None and len(model['all_pts']) > 0:
            all_norm = (model['all_pts'] - center) / widths
            nov_scores = []
            for c in cands_norm:
                dists = np.linalg.norm(all_norm - c, axis=1)
                nov_scores.append(np.min(dists))  # Min distance to any visited
            nov_ranks = self._to_ranks(nov_scores, higher_is_better=True)
            all_ranks.append(nov_ranks)
        
        # ===== 4. GRD: Gradient alignment =====
        # Allineamento con direzione di miglioramento
        if model['gradient_dir'] is not None:
            # Centro dei top-k come riferimento
            top_center = model['top_k_pts'].mean(axis=0)
            top_center_norm = (top_center - center) / widths
            
            grd_scores = []
            for c in cands_norm:
                direction = c - top_center_norm
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-6:  # Increased epsilon for better stability
                    direction = direction / dir_norm
                    alignment = np.dot(direction, model['gradient_dir'])
                    grd_scores.append(alignment)
                else:
                    grd_scores.append(0.0)
            grd_ranks = self._to_ranks(grd_scores, higher_is_better=True)
            all_ranks.append(grd_ranks)
        
        # ===== Combina: media dei rank =====
        if len(all_ranks) == 0:
            return np.ones(n) / n
        
        # Stack e media
        all_ranks = np.array(all_ranks)  # shape: (n_criteria, n_candidates)
        mean_ranks = all_ranks.mean(axis=0)  # Media su tutti i criteri
        
        return mean_ranks
    
    def _to_ranks(self, scores: List[float], higher_is_better: bool) -> np.ndarray:
        """Converte scores in rank normalizzati [0, 1]."""
        scores = np.array(scores)
        n = len(scores)
        
        if not higher_is_better:
            scores = -scores
        
        # Argsort per ottenere i rank (0 = peggiore, n-1 = migliore)
        order = np.argsort(scores)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        
        # Normalizza a [0, 1]
        return ranks / max(1, n - 1)

    def get_split_axis(self) -> int:
        """Sceglie asse di split."""
        if self.lgs_model is not None and self.lgs_model['gradient_dir'] is not None:
            return int(np.argmax(np.abs(self.lgs_model['gradient_dir'])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int) -> List["Cube"]:
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
        
        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1
        
        for ch in [child_lo, child_hi]:
            ch.fit_lgs_model(gamma, dim)
        
        return [child_lo, child_hi]


class HPOptimizer:
    """
    HPO-LGS v3 - Completamente non-parametrico.
    Nessun peso da tunare, tutto si auto-calibra dai dati.
    """
    
    def __init__(self, bounds: List[Tuple[float,float]], maximize: bool = False, 
                 seed: int = 42, gamma_quantile: float = 0.25,
                 local_search_ratio: float = 0.2,
                 n_candidates: int = 40):  # Aumentato da 30 a 40
        
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = 0.10
        self.local_search_ratio = local_search_ratio
        self.n_candidates = n_candidates
        
        self.root = Cube(bounds=list(bounds))
        self.leaves: List[Cube] = [self.root]
        
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []
        self.best_y = -np.inf if maximize else np.inf
        self.best_x = None
        self.gamma = 0.0
        self.iteration = 0
        self.exploration_budget = 0
        
        self.global_widths = np.array([hi - lo for lo, hi in bounds])
    
    def _update_gamma(self):
        if len(self.y_all) < 10:
            self.gamma = 0.0
        else:
            progress = min(1.0, self.iteration / max(1, self.exploration_budget * 0.5))
            current_quantile = self.gamma_quantile_start - progress * (self.gamma_quantile_start - self.gamma_quantile)
            self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))
    
    def _recount_good(self):
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
    
    def _update_best(self, x: np.ndarray, y_raw: float):
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
            
            model_bonus = 0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get('all_pts', []))
                if n_pts >= self.dim + 2:
                    model_bonus = 0.1
            
            score = ratio + exploration + model_bonus
            scores.append(score)
        
        scores = np.array(scores)
        scores = scores - scores.max()
        probs = np.exp(scores * 3)
        probs = probs / probs.sum()
        
        return self.rng.choice(self.leaves, p=probs)
    
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
        """Genera candidati con strategie diverse - MIGLIORATO per funzioni complesse."""
        candidates = []
        widths = cube._widths()
        center = cube.center()
        
        model = cube.lgs_model
        
        for _ in range(n):
            strategy = self.rng.random()
            
            # Strategia 1: Esplorazione vicino ai top-k (aumentata)
            if strategy < 0.25 and model is not None and len(model['top_k_pts']) > 0:
                # Usa più varianza per esplorare meglio
                idx = self.rng.integers(len(model['top_k_pts']))
                scale = 0.15 if self.iteration < self.exploration_budget * 0.5 else 0.08
                x = model['top_k_pts'][idx] + self.rng.normal(0, scale, self.dim) * widths
                
            # Strategia 2: Multi-hop lungo il gradiente (NUOVO)
            elif strategy < 0.40 and model is not None and model['gradient_dir'] is not None:
                # Prova diversi step size lungo il gradiente
                top_center = model['top_k_pts'].mean(axis=0)
                step = self.rng.choice([0.05, 0.15, 0.30, 0.50])  # Multi-scale steps
                direction = model['gradient_dir']
                # Aggiungi perturbazione perpendicolare per esplorare attorno
                perp_noise = self.rng.normal(0, 0.08, self.dim)
                perp_noise = perp_noise - np.dot(perp_noise, direction) * direction  # Ortogonale
                x = top_center + step * direction * widths + perp_noise * widths
                
            # Strategia 3: Esplorazione direzionale tra top punti (NUOVO)
            elif strategy < 0.50 and model is not None and len(model['top_k_pts']) > 1:
                # Interpola/estrapola tra i migliori punti
                idx1, idx2 = self.rng.choice(len(model['top_k_pts']), size=2, replace=False)
                alpha = self.rng.uniform(-0.3, 1.3)  # Permetti extrapolazione
                x = alpha * model['top_k_pts'][idx1] + (1 - alpha) * model['top_k_pts'][idx2]
                x = x + self.rng.normal(0, 0.05, self.dim) * widths
                
            # Strategia 4: Evita zone cattive con bounce (MIGLIORATO)
            elif strategy < 0.65 and model is not None and model['bad_pts'] is not None and len(model['bad_pts']) > 0:
                # Parti dal centro, muoviti lontano dai punti cattivi
                bad_center = model['bad_pts'].mean(axis=0)
                away_direction = center - bad_center
                away_norm = np.linalg.norm(away_direction)
                if away_norm > 1e-9:
                    away_direction = away_direction / away_norm
                    step = self.rng.uniform(0.2, 0.4)
                    x = center + step * away_direction * widths
                    x = x + self.rng.normal(0, 0.1, self.dim) * widths
                else:
                    x = center + self.rng.normal(0, 0.2, self.dim) * widths
                
            # Strategia 5: Cerca vicino al centro
            elif strategy < 0.75:
                # Gaussiano centrato
                x = center + self.rng.normal(0, 0.2, self.dim) * widths
                
            # Strategia 6: Esplorazione uniforme residua
            else:
                # Random uniforme nel cubo
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            
            x = self._clip_to_cube(x, cube)
            candidates.append(x)
        
        return candidates
    
    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        """Genera candidati e seleziona il migliore secondo LGS."""
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        candidates = self._generate_candidates(cube, self.n_candidates)
        
        # Rank non-parametrico
        ranks = cube.rank_candidates_lgs(candidates)
        
        # Softmax sui rank (non sui raw scores)
        ranks = ranks - ranks.max()
        probs = np.exp(ranks * 5)  # Temperature fissa ok, lavoriamo su rank normalizzati
        probs = probs / probs.sum()
        
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        # Early: più random
        if self.iteration < 15:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        return self._sample_with_lgs(cube)
    
    def _local_search_sample(self, progress: float) -> np.ndarray:
        """Local search migliorata con strategie multiple per evitare minimi locali."""
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        
        # Scegli strategia in base al progresso
        strategy = self.rng.random()
        
        # Strategia 1: Raffinamento locale intenso (early)
        if strategy < 0.4 and progress < 0.5:
            radius = 0.08 * (1 - progress) + 0.01
            noise = self.rng.normal(0, radius, self.dim) * self.global_widths
            x = self.best_x + noise
            
        # Strategia 2: Esplorazione radiale multi-scala (NUOVO)
        elif strategy < 0.6:
            # Prova diverse distanze dal best point
            radius = self.rng.choice([0.03, 0.08, 0.15, 0.25]) * (1 - progress * 0.7)
            direction = self.rng.normal(0, 1, self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            x = self.best_x + radius * direction * self.global_widths
            
        # Strategia 3: Jump and descend (per scappare da local minima)
        elif strategy < 0.75 and len(self.X_all) > 10:
            # Salta verso una regione promettente ma non ancora ottimizzata
            top_k = min(10, len(self.y_all))
            top_indices = np.argsort(self.y_all)[-top_k:]
            idx = self.rng.choice(top_indices)
            other_good = self.X_all[idx]
            # Vai tra best e other good point, con rumore
            alpha = self.rng.uniform(0.3, 0.7)
            x = alpha * self.best_x + (1 - alpha) * other_good
            x = x + self.rng.normal(0, 0.05, self.dim) * self.global_widths
            
        # Strategia 4: Pattern search (esplora coordinate systematicamente)
        else:
            x = self.best_x.copy()
            # Perturba 1-3 dimensioni in modo più significativo
            n_dims = min(self.rng.integers(1, 4), self.dim)
            dims = self.rng.choice(self.dim, size=n_dims, replace=False)
            for d in dims:
                x[d] += self.rng.choice([-1, 1]) * self.rng.uniform(0.05, 0.15) * self.global_widths[d]
        
        return self._clip_to_bounds(x)
    
    def _should_split(self, cube: Cube) -> bool:
        if cube.n_trials < 10:
            return False
        if cube.depth >= 5:
            return False
        return cube.n_trials >= 2 * self.dim + 4
    
    def _update_all_models(self):
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim)
    
    def optimize(self, objective: Callable[[np.ndarray], float], 
                 budget: int = 100) -> Tuple[np.ndarray, float]:
        
        exploration_budget = int(budget * (1 - self.local_search_ratio))
        local_search_budget = budget - exploration_budget
        self.exploration_budget = exploration_budget
        
        for trial in range(exploration_budget):
            self.iteration = trial
            self._update_gamma()
            self._recount_good()
            
            if trial % 5 == 0:
                self._update_all_models()
            
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
            
            cube.fit_lgs_model(self.gamma, self.dim)
            
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
    
    def rosenbrock(x):
        total = 0
        for i in range(len(x)-1):
            total += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    print("=" * 60)
    print("  Test HPO-LGS v3 (NON-PARAMETRICO)")
    print("=" * 60)
    
    print("\n📊 Test su Sphere 10D (optimum @ 0.7):")
    for seed in [42, 123, 456]:
        opt = HPOptimizer([(0, 1)]*10, maximize=False, seed=seed)
        best_x, best_y = opt.optimize(sphere, budget=100)
        print(f"  Seed {seed}: {best_y:.6f}")
    
    print("\n📊 Test su Rosenbrock 5D:")
    for seed in [42, 123, 456]:
        opt = HPOptimizer([(0, 2)]*5, maximize=False, seed=seed)
        best_x, best_y = opt.optimize(rosenbrock, budget=150)
        print(f"  Seed {seed}: {best_y:.4f}")
    
    print("\n✅ HPO-LGS v3 funziona!")
