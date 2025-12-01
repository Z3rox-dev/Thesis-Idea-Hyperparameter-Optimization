"""
HPO-LGS - Local Geometry Score Surrogate

Surrogato "esotico" basato SOLO sulla geometria dei punti, zero regressione!
Non modella f(x), ma modella la struttura del paesaggio.

Local Geometry Score (LGS):
    s_LGS(x) = w1 * dist_from_bad(x) + w2 * dist_to_good_center(x) + w3 * anisotropy_alignment(x)

Componenti:
1. Distance from Bad (DFB): Lontananza dai punti peggiori
2. Distance to Good Center (DGC): Vicinanza al centro dei punti buoni
3. Anisotropy Alignment (AA): Allineamento con la direzione PCA dei buoni

Autore: Z3rox-dev
Data: Dicembre 2025
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
        """
        Costruisce il modello LGS basato sulla geometria locale.
        
        Calcola:
        - Centro dei punti "buoni" (above gamma)
        - Cluster dei punti "cattivi" (below gamma)
        - Direzione principale PCA dei buoni
        """
        pairs = list(self._tested_pairs)
        
        # Riutilizzo punti del padre
        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, '_tested_pairs', [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
                pairs = pairs + extra[:needed]
        
        if len(pairs) < dim + 2:
            self.lgs_model = None
            return
        
        # Separa buoni e cattivi
        good_pts = [p for p, s in pairs if s >= gamma]
        bad_pts = [p for p, s in pairs if s < gamma]
        
        if len(good_pts) < 2:
            # Non abbastanza buoni, usa tutti come riferimento
            self.lgs_model = None
            return
        
        X_good = np.array(good_pts)
        good_center = X_good.mean(axis=0)
        
        # Normalizza rispetto al cubo
        widths = np.maximum(self._widths(), 1e-9)
        center = self.center()
        
        # PCA sui buoni per trovare direzione principale
        X_good_centered = X_good - good_center
        X_good_norm = X_good_centered / widths
        
        principal_dir = None
        if len(good_pts) >= dim + 1:
            try:
                # SVD per PCA
                U, S, Vt = np.linalg.svd(X_good_norm, full_matrices=False)
                principal_dir = Vt[0]  # Prima componente principale
                # Scala con gli eigenvalues per dare importanza
                variance_explained = (S[0] ** 2) / (np.sum(S ** 2) + 1e-12)
            except:
                principal_dir = None
                variance_explained = 0.0
        else:
            variance_explained = 0.0
        
        # Punti cattivi per il calcolo delle distanze
        bad_pts_array = np.array(bad_pts) if bad_pts else None
        
        self.lgs_model = {
            'good_center': good_center,
            'bad_pts': bad_pts_array,
            'principal_dir': principal_dir,
            'variance_explained': variance_explained,
            'widths': widths,
            'cube_center': center,
            'n_good': len(good_pts),
            'n_bad': len(bad_pts)
        }
    
    def score_candidate_lgs(self, x: np.ndarray, w1: float = 1.0, w2: float = 1.0, w3: float = 0.5) -> float:
        """
        Calcola il Local Geometry Score per un candidato x.
        
        s_LGS(x) = w1 * dist_from_bad + w2 * closeness_to_good + w3 * alignment
        
        Maggiore è meglio!
        """
        if self.lgs_model is None:
            return 0.0
        
        model = self.lgs_model
        widths = model['widths']
        
        # Normalizza x
        x_norm = (x - model['cube_center']) / widths
        good_center_norm = (model['good_center'] - model['cube_center']) / widths
        
        score = 0.0
        
        # 1. Distance from Bad (DFB) - più lontano dai cattivi = meglio
        if model['bad_pts'] is not None and len(model['bad_pts']) > 0:
            bad_norm = (model['bad_pts'] - model['cube_center']) / widths
            distances_to_bad = np.linalg.norm(bad_norm - x_norm, axis=1)
            min_dist_to_bad = np.min(distances_to_bad)
            # Normalizza [0, sqrt(dim)] -> [0, 1]
            dfb = min_dist_to_bad / np.sqrt(len(widths))
            score += w1 * dfb
        else:
            # Nessun punto cattivo, bonus massimo
            score += w1 * 1.0
        
        # 2. Closeness to Good Center (DGC) - più vicino al centro dei buoni = meglio
        dist_to_good = np.linalg.norm(x_norm - good_center_norm)
        # Inverti: vogliamo che vicino = alto score
        # Usa esponenziale per smooth decay
        dgc = np.exp(-dist_to_good)
        score += w2 * dgc
        
        # 3. Anisotropy Alignment (AA) - allineamento con direzione principale
        if model['principal_dir'] is not None and model['variance_explained'] > 0.3:
            # Vettore da good_center a x
            direction = x_norm - good_center_norm
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-9:
                direction = direction / dir_norm
                # Proiezione sulla direzione principale (valore assoluto)
                alignment = abs(np.dot(direction, model['principal_dir']))
                # Pesa con la varianza spiegata
                aa = alignment * model['variance_explained']
                score += w3 * aa
        
        return score

    def get_split_axis_lgs(self) -> int:
        """
        Sceglie l'asse di split basandosi sulla geometria.
        Usa la direzione principale PCA se disponibile.
        """
        if self.lgs_model is not None and self.lgs_model['principal_dir'] is not None:
            # Split lungo la direzione di massima variazione (denormalizzata)
            principal = np.abs(self.lgs_model['principal_dir'])
            return int(np.argmax(principal))
        else:
            # Fallback: asse più largo
            return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int) -> List["Cube"]:
        axis = self.get_split_axis_lgs()
        
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
    HPO con Local Geometry Score (LGS) come surrogato.
    
    Invece di fare regressione quadratica, usa la geometria locale:
    - Distanza dai punti cattivi
    - Vicinanza al centro dei punti buoni  
    - Allineamento con la direzione principale PCA
    """
    
    def __init__(self, bounds: List[Tuple[float,float]], maximize: bool = False, 
                 seed: int = 42, gamma_quantile: float = 0.25,
                 local_search_ratio: float = 0.2,
                 # Pesi LGS
                 w_dfb: float = 1.0,   # Distance from Bad
                 w_dgc: float = 1.5,   # Distance to Good Center  
                 w_aa: float = 0.5):   # Anisotropy Alignment
        
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = 0.10
        self.local_search_ratio = local_search_ratio
        
        # Pesi LGS
        self.w_dfb = w_dfb
        self.w_dgc = w_dgc
        self.w_aa = w_aa
        
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
            
            # Bonus per modello LGS valido
            model_bonus = 0
            if c.lgs_model is not None:
                n_good = c.lgs_model.get('n_good', 0)
                var_exp = c.lgs_model.get('variance_explained', 0)
                if n_good >= 3:
                    model_bonus = 0.15 * min(1.0, n_good / 10)
                if var_exp > 0.5:
                    model_bonus += 0.1 * var_exp
            
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
    
    def _sample_with_lgs(self, cube: Cube, n_candidates: int = 20) -> np.ndarray:
        """
        Genera candidati e seleziona quello con il miglior LGS score.
        """
        if cube.lgs_model is None or cube.n_trials < 5:
            # Fallback a random
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        # Genera candidati
        candidates = []
        for _ in range(n_candidates):
            # Mix di strategie per generare candidati
            strategy = self.rng.random()
            
            if strategy < 0.4:
                # Vicino al centro dei buoni
                good_center = cube.lgs_model['good_center']
                widths = cube._widths()
                noise = self.rng.normal(0, 0.2, self.dim) * widths
                x = good_center + noise
            elif strategy < 0.7:
                # Lungo la direzione principale
                if cube.lgs_model['principal_dir'] is not None:
                    good_center = cube.lgs_model['good_center']
                    principal = cube.lgs_model['principal_dir']
                    widths = cube._widths()
                    # Step lungo la direzione principale
                    step = self.rng.uniform(-0.5, 0.5)
                    x = good_center + step * principal * widths
                    # Piccolo rumore ortogonale
                    noise = self.rng.normal(0, 0.1, self.dim) * widths
                    x = x + noise
                else:
                    x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            else:
                # Random nel cubo
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            
            x = self._clip_to_cube(x, cube)
            candidates.append(x)
        
        # Valuta tutti i candidati con LGS
        scores = [cube.score_candidate_lgs(x, self.w_dfb, self.w_dgc, self.w_aa) 
                  for x in candidates]
        
        # Selezione softmax (non solo argmax per mantenere esplorazione)
        scores = np.array(scores)
        scores = scores - scores.max()
        probs = np.exp(scores * 5)  # Temperature = 0.2
        probs = probs / probs.sum()
        
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def _directional_sample(self, cube: Cube) -> Optional[np.ndarray]:
        """
        Campiona nella direzione second_best -> best.
        """
        if self.best_x is None or self.second_best_x is None:
            return None
        
        direction = self.best_x - self.second_best_x
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return None
        
        step = self.rng.uniform(0.0, 1.2)
        x = self.second_best_x + step * direction
        
        noise = self.rng.normal(0, 0.03, self.dim) * self.global_widths
        x = x + noise
        
        return self._clip_to_cube(x, cube)
    
    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        # 10% random puro
        if self.rng.random() < 0.10:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        # 10% directional
        if self.second_best_x is not None and self.rng.random() < 0.10:
            x_dir = self._directional_sample(cube)
            if x_dir is not None:
                return x_dir
        
        # 80% LGS-guided sampling
        return self._sample_with_lgs(cube)
    
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
    
    def _update_all_lgs_models(self):
        """Aggiorna i modelli LGS di tutte le foglie."""
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
            
            # Aggiorna modelli LGS periodicamente
            if trial % 5 == 0:
                self._update_all_lgs_models()
            
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
            
            # Aggiorna LGS del cubo corrente
            cube.fit_lgs_model(self.gamma, self.dim)
            
            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim)
                self.leaves.remove(cube)
                self.leaves.extend(children)
        
        # Local search finale
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
    print("  Test HPO-LGS (Local Geometry Score Surrogate)")
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
    
    print("\n✅ HPO-LGS funziona!")
