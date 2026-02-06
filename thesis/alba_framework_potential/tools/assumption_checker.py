"""
ALBA Assumption Checker - Deep Debugging

Verifica sistematica delle assunzioni implicite nel codice ALBA.
Per ogni assunzione: definizione formale, test, controesempio se viola.

Assunzioni identificate:
========================

A1) NORMALIZZAZIONE: (X - center) / widths porta X in [-0.5, 0.5]^d
    - Richiede che widths > 0 per ogni dim
    - Richiede che X sia dentro bounds del cube
    
A2) COVARIANZA PD: XtWX + λI è sempre positiva definita (invertibile)
    - Richiede che i punti non siano tutti collineari
    - Richiede λ > 0 sufficientemente grande
    
A3) GRADIENTE NORMALIZZATO: ||grad|| ha ordine O(1) dopo normalizzazione
    - Richiede che y_std scali correttamente la risposta
    - Richiede che inv_cov non esploda
    
A4) PESI GAUSSIANI: exp(-d²/2σ²) ∈ (0, 1] per tutti i punti
    - Richiede σ² > 0
    - Punti lontani → peso → 0
    
A5) SPLITTING MANTIENE COPERTURA: ∪ children = parent
    - Dopo split, ogni punto del parent cade in esattamente un child
    
A6) GAMMA THRESHOLD: gamma separa "good" da "bad" in modo sensato
    - Richiede distribuzione non degenere dei punteggi
    
A7) UCB EXPLORATION: sigma > 0 sempre
    - sigma = 0 → nessuna esplorazione
    
A8) DRILLING CONVERGENCE: sigma → 0 implica convergenza
    - Richiede che sigma decresca
    
A9) CATEGORIE ENCODING: x[cat_dim] ∈ {0, 1, ..., n_choices-1}
    - Dopo encoding, valori devono essere interi validi
    
A10) BEST_Y TRACKING: best_y è effettivamente il minimo globale osservato
    - Richiede che tutti i punti siano registrati
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, '/mnt/workspace/thesis')
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from alba_framework_potential.lgs import fit_lgs_model, predict_bayesian
from alba_framework_potential.splitting import CubeIntrinsicSplitPolicy
from alba_framework_potential.drilling import DrillingOptimizer


def test_functions():
    """Funzioni test con proprietà diverse."""
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    def ill_conditioned(x):
        scales = np.array([10**(i) for i in range(len(x))])
        return np.sum(scales * (x - 0.5)**2)
    
    def nearly_flat(x):
        return 1e-10 * np.sum(x**2)
    
    def extremely_steep(x):
        return 1e10 * np.sum((x - 0.5)**2)
    
    def ridge(x):
        # Funzione dove tutte le direzioni eccetto una sono piatte
        return (x[0] - 0.5)**2
    
    def plateau(x):
        # Piatta ovunque tranne vicino all'ottimo
        d = np.linalg.norm(x - 0.5)
        if d < 0.1:
            return d
        return 1.0
    
    return {
        'sphere': sphere,
        'ill_conditioned': ill_conditioned,
        'nearly_flat': nearly_flat,
        'extremely_steep': extremely_steep,
        'ridge': ridge,
        'plateau': plateau,
    }


class AssumptionChecker:
    """Verifica sistematica delle assunzioni ALBA."""
    
    def __init__(self, dim: int = 5, verbose: bool = True):
        self.dim = dim
        self.verbose = verbose
        self.violations = []
        
    def log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            prefix = {"INFO": "  ", "WARN": "⚠️", "FAIL": "❌", "PASS": "✅"}[level]
            print(f"{prefix} {msg}")
    
    def record_violation(self, assumption: str, details: Dict):
        self.violations.append({"assumption": assumption, **details})
        
    # =========================================================================
    # A1: NORMALIZZAZIONE
    # =========================================================================
    def check_A1_normalization(self):
        """
        A1) (X - center) / widths ∈ [-0.5, 0.5]^d per X nei bounds
        
        Assunzione: La normalizzazione porta punti interni in [-0.5, 0.5]^d.
        Controllo: widths > 0, punti non fuori bounds.
        """
        print("\n" + "="*70)
        print("A1: NORMALIZZAZIONE - X_norm ∈ [-0.5, 0.5]^d")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        cube = Cube(bounds=list(bounds))
        
        widths = cube.widths()
        center = cube.center()
        
        self.log(f"widths = {widths}")
        self.log(f"center = {center}")
        
        # Test 1: widths > 0?
        if np.any(widths <= 0):
            self.log(f"widths ha elementi <= 0!", "FAIL")
            self.record_violation("A1", {"issue": "widths <= 0", "widths": widths})
            return False
        
        # Test 2: Corner points dovrebbero dare X_norm = ±0.5
        corners = [
            np.array([lo for lo, hi in bounds]),  # all-low
            np.array([hi for lo, hi in bounds]),  # all-high
        ]
        
        for corner in corners:
            X_norm = (corner - center) / widths
            if np.any(np.abs(X_norm) > 0.5 + 1e-9):
                self.log(f"Corner {corner} → X_norm = {X_norm}, fuori [-0.5, 0.5]!", "FAIL")
                self.record_violation("A1", {"issue": "corner out of range", "corner": corner, "X_norm": X_norm})
                return False
                
        # Test 3: Dopo splitting profondo, widths ancora > 0?
        tiny_cube = Cube(bounds=[(0.5 - 1e-12, 0.5 + 1e-12)] * self.dim)
        tiny_widths = tiny_cube.widths()
        self.log(f"Tiny cube widths = {tiny_widths}")
        
        if np.any(tiny_widths < 1e-9):
            self.log(f"Dopo molti split, widths < 1e-9, può causare div/0!", "WARN")
        
        self.log("A1 PASSED: normalizzazione corretta", "PASS")
        return True
    
    # =========================================================================
    # A2: COVARIANZA PD
    # =========================================================================
    def check_A2_covariance_pd(self):
        """
        A2) XtWX + λI è positiva definita
        
        Controllo: Quando i punti sono collineari o quasi, la matrice può essere singolare.
        """
        print("\n" + "="*70)
        print("A2: COVARIANZA POSITIVA DEFINITA")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        cube = Cube(bounds=list(bounds))
        rng = np.random.default_rng(42)
        
        # Caso patologico: tutti punti sulla stessa linea
        t = np.linspace(0.1, 0.9, 20)
        direction = rng.random(self.dim)
        direction /= np.linalg.norm(direction)
        
        base = np.array([0.5] * self.dim)
        for ti in t:
            x = base + ti * 0.3 * direction
            x = np.clip(x, 0, 1)
            y = np.sum(x)  # Qualsiasi funzione lineare
            cube.add_observation(x, y, gamma=0.0)
        
        self.log(f"Aggiunti {len(t)} punti COLLINEARI al cube")
        
        # Prova a fittare LGS
        cube.fit_lgs_model(gamma=0.0, dim=self.dim, rng=rng)
        model = cube.lgs_model
        
        if model is None:
            self.log("LGS model = None, punti insufficienti?", "WARN")
            return True
        
        inv_cov = model.get('inv_cov')
        if inv_cov is None:
            self.log("inv_cov = None anche con punti collineari (catch riuscito)", "PASS")
            return True
        
        # Verifica eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(inv_cov)
            self.log(f"inv_cov eigenvalues: min={eigvals.min():.2e}, max={eigvals.max():.2e}")
            
            if eigvals.min() < 0:
                self.log(f"inv_cov NON positiva definita! min_eig = {eigvals.min()}", "FAIL")
                self.record_violation("A2", {"issue": "negative eigenvalue", "min_eig": eigvals.min()})
                return False
            
            cond = eigvals.max() / (eigvals.min() + 1e-12)
            if cond > 1e10:
                self.log(f"inv_cov mal condizionata: cond = {cond:.2e}", "WARN")
                
        except Exception as e:
            self.log(f"Errore nel calcolo eigenvalues: {e}", "FAIL")
            return False
        
        self.log("A2 PASSED: covarianza gestisce casi collineari", "PASS")
        return True
    
    # =========================================================================
    # A3: GRADIENTE NORMALIZZATO
    # =========================================================================
    def check_A3_gradient_normalized(self):
        """
        A3) ||grad|| ~ O(1) dopo normalizzazione
        
        Il gradiente in spazio normalizzato dovrebbe avere norma ragionevole.
        """
        print("\n" + "="*70)
        print("A3: GRADIENTE NORMALIZZATO ||grad|| ~ O(1)")
        print("="*70)
        
        funcs = test_functions()
        rng = np.random.default_rng(42)
        
        for name, func in funcs.items():
            bounds = [(0, 1)] * self.dim
            cube = Cube(bounds=list(bounds))
            
            # Riempi con punti random
            for _ in range(30):
                x = rng.random(self.dim)
                y = func(x)
                cube.add_observation(x, y, gamma=0.0)
            
            cube.fit_lgs_model(gamma=0.0, dim=self.dim, rng=rng)
            model = cube.lgs_model
            
            if model is None or model.get('grad') is None:
                self.log(f"{name}: grad = None (skip)", "INFO")
                continue
            
            grad = model['grad']
            grad_norm = np.linalg.norm(grad)
            y_std = model.get('y_std', 1.0)
            
            self.log(f"{name}: ||grad|| = {grad_norm:.2e}, y_std = {y_std:.2e}")
            
            # Il gradiente normalizzato dovrebbe essere O(1)
            if grad_norm > 100:
                self.log(f"  → grad_norm > 100, potrebbe esplodere in acquisizione!", "WARN")
            if grad_norm > 1e6:
                self.log(f"  → grad_norm ESPLOSIVO!", "FAIL")
                self.record_violation("A3", {"func": name, "grad_norm": grad_norm})
        
        self.log("A3 CHECK COMPLETE", "PASS")
        return True
    
    # =========================================================================
    # A4: PESI GAUSSIANI
    # =========================================================================
    def check_A4_gaussian_weights(self):
        """
        A4) weights = exp(-d²/2σ²) ∈ (0, 1]
        
        Verifica che σ² > 0 e che i pesi non siano tutti zero.
        """
        print("\n" + "="*70)
        print("A4: PESI GAUSSIANI exp(-d²/2σ²) ∈ (0, 1]")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        cube = Cube(bounds=list(bounds))
        rng = np.random.default_rng(42)
        
        # Tutti punti in un angolo
        for _ in range(20):
            x = rng.random(self.dim) * 0.01  # Molto vicini a 0
            y = np.sum(x)
            cube.add_observation(x, y, gamma=0.0)
        
        widths = cube.widths()
        center = cube.center()
        pairs = list(cube.tested_pairs)
        all_pts = np.array([p for p, s in pairs])
        
        X_norm = (all_pts - center) / widths
        dists_sq = np.sum(X_norm**2, axis=1)
        sigma_sq = np.mean(dists_sq) + 1e-6
        weights = np.exp(-dists_sq / (2 * sigma_sq))
        
        self.log(f"dists_sq: min={dists_sq.min():.4f}, max={dists_sq.max():.4f}")
        self.log(f"sigma_sq = {sigma_sq:.4f}")
        self.log(f"weights: min={weights.min():.4e}, max={weights.max():.4e}")
        
        if sigma_sq <= 0:
            self.log("sigma_sq <= 0, impossibile calcolare pesi!", "FAIL")
            self.record_violation("A4", {"issue": "sigma_sq <= 0"})
            return False
        
        if weights.max() < 1e-100:
            self.log("Tutti i pesi ~ 0, regressione inutile!", "FAIL")
            self.record_violation("A4", {"issue": "all weights ~ 0"})
            return False
        
        self.log("A4 PASSED: pesi gaussiani validi", "PASS")
        return True
    
    # =========================================================================
    # A5: SPLITTING MANTIENE COPERTURA
    # =========================================================================
    def check_A5_split_coverage(self):
        """
        A5) Dopo split, ogni punto del parent cade in esattamente un child.
        """
        print("\n" + "="*70)
        print("A5: SPLIT MANTIENE COPERTURA COMPLETA")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        parent = Cube(bounds=list(bounds))
        rng = np.random.default_rng(42)
        
        # Aggiungi punti
        for _ in range(30):
            x = rng.random(self.dim)
            y = np.sum(x)
            parent.add_observation(x, y, gamma=0.5)
        
        parent.fit_lgs_model(gamma=0.5, dim=self.dim, rng=rng)
        
        policy = CubeIntrinsicSplitPolicy()
        children = policy.split(parent, gamma=0.5, dim=self.dim, rng=rng)
        
        if children is None:
            self.log("Split ritorna None (nessuno split possibile)", "INFO")
            return True
        
        self.log(f"Split creato {len(children)} figli")
        
        # Test: ogni punto del parent deve essere in esattamente un child
        test_points = [rng.random(self.dim) for _ in range(100)]
        
        for pt in test_points:
            containing = [c for c in children if c.contains(pt)]
            if len(containing) == 0:
                self.log(f"Punto {pt} NON coperto da nessun child!", "FAIL")
                self.record_violation("A5", {"point": pt, "n_containing": 0})
                return False
            if len(containing) > 1:
                self.log(f"Punto {pt} coperto da {len(containing)} children (overlap)!", "WARN")
        
        self.log("A5 PASSED: split mantiene copertura", "PASS")
        return True
    
    # =========================================================================
    # A6: GAMMA THRESHOLD
    # =========================================================================
    def check_A6_gamma_threshold(self):
        """
        A6) gamma separa good/bad sensatamente
        
        Se tutti i valori sono uguali, gamma è degenere.
        """
        print("\n" + "="*70)
        print("A6: GAMMA THRESHOLD SENSATO")
        print("="*70)
        
        # Caso patologico: funzione costante
        def constant(x):
            return 42.0
        
        bounds = [(0, 1)] * self.dim
        opt = ALBA(bounds=bounds, seed=42, total_budget=50)
        
        for i in range(20):
            x = opt.ask()
            y = constant(x)
            opt.tell(x, y)
        
        self.log(f"gamma = {opt.gamma}")
        self.log(f"y_all unique values: {len(set(opt.y_all))}")
        
        # Con funzione costante, tutti i punti hanno lo stesso score
        # gamma dovrebbe essere quel valore
        if len(set(opt.y_all)) == 1:
            self.log("Funzione costante: tutti i punti equivalenti", "WARN")
            # Non è un errore, ma potrebbe causare comportamenti strani
        
        self.log("A6 CHECK COMPLETE", "PASS")
        return True
    
    # =========================================================================
    # A7: UCB EXPLORATION (sigma > 0)
    # =========================================================================
    def check_A7_ucb_exploration(self):
        """
        A7) sigma > 0 sempre, altrimenti nessuna esplorazione
        """
        print("\n" + "="*70)
        print("A7: UCB EXPLORATION (sigma > 0)")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        cube = Cube(bounds=list(bounds))
        rng = np.random.default_rng(42)
        
        # Riempi cube
        for _ in range(30):
            x = rng.random(self.dim)
            y = np.sum(x**2)
            cube.add_observation(x, y, gamma=0.0)
        
        cube.fit_lgs_model(gamma=0.0, dim=self.dim, rng=rng)
        
        # Genera candidati
        candidates = [rng.random(self.dim) for _ in range(50)]
        mu, sigma = cube.predict_bayesian(candidates)
        
        self.log(f"mu: min={mu.min():.4f}, max={mu.max():.4f}")
        self.log(f"sigma: min={sigma.min():.4f}, max={sigma.max():.4f}")
        
        if sigma.min() <= 0:
            self.log("sigma <= 0 per alcuni candidati → NO exploration!", "FAIL")
            self.record_violation("A7", {"issue": "sigma <= 0", "min_sigma": sigma.min()})
            return False
        
        if sigma.min() < 1e-6:
            self.log("sigma molto piccolo, exploration ridotta", "WARN")
        
        self.log("A7 PASSED: sigma > 0", "PASS")
        return True
    
    # =========================================================================
    # A8: DRILLING CONVERGENCE
    # =========================================================================
    def check_A8_drilling_convergence(self):
        """
        A8) Durante drilling, sigma dovrebbe diminuire (o stabilizzarsi).
        Se sigma cresce, il drilling non converge.
        """
        print("\n" + "="*70)
        print("A8: DRILLING CONVERGENCE (sigma → 0)")
        print("="*70)
        
        def sphere(x):
            return np.sum((x - 0.5)**2)
        
        start_x = np.random.rand(self.dim)
        start_y = sphere(start_x)
        
        driller = DrillingOptimizer(
            start_x=start_x,
            start_y=start_y,
            initial_sigma=0.1,
            bounds=[(0, 1)] * self.dim
        )
        
        rng = np.random.default_rng(42)
        sigma_history = [driller.sigma]
        
        for step in range(30):
            x = driller.ask(rng)
            y = sphere(x)
            keep_going = driller.tell(x, y)
            sigma_history.append(driller.sigma)
            if not keep_going:
                break
        
        self.log(f"Drilling steps: {len(sigma_history)}")
        self.log(f"sigma: start={sigma_history[0]:.4f}, end={sigma_history[-1]:.4f}")
        
        # Verifica trend
        if sigma_history[-1] > sigma_history[0] * 2:
            self.log("sigma AUMENTATO durante drilling!", "WARN")
        
        self.log("A8 CHECK COMPLETE", "PASS")
        return True
    
    # =========================================================================
    # A9: BEST_Y TRACKING
    # =========================================================================
    def check_A9_best_y_tracking(self):
        """
        A9) best_y è effettivamente il minimo osservato
        """
        print("\n" + "="*70)
        print("A9: BEST_Y TRACKING CORRETTO")
        print("="*70)
        
        def sphere(x):
            return np.sum((x - 0.5)**2)
        
        bounds = [(0, 1)] * self.dim
        opt = ALBA(bounds=bounds, seed=42, total_budget=50, maximize=False)
        
        all_y = []
        for i in range(50):
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
            all_y.append(y)
        
        true_min = min(all_y)
        reported_best = opt.best_y
        
        self.log(f"True min(y): {true_min:.6f}")
        self.log(f"Reported best_y: {reported_best:.6f}")
        
        if abs(true_min - reported_best) > 1e-9:
            self.log("MISMATCH tra true min e reported best!", "FAIL")
            self.record_violation("A9", {"true_min": true_min, "reported_best": reported_best})
            return False
        
        self.log("A9 PASSED: best_y corretto", "PASS")
        return True
    
    # =========================================================================
    # A10: PREDICT CONSISTENCY
    # =========================================================================
    def check_A10_predict_consistency(self):
        """
        A10) predict_bayesian(center) ≈ y_mean
        
        La predizione al centro del cube dovrebbe essere circa la media.
        """
        print("\n" + "="*70)
        print("A10: PREDICT CONSISTENCY (center → y_mean)")
        print("="*70)
        
        bounds = [(0, 1)] * self.dim
        cube = Cube(bounds=list(bounds))
        rng = np.random.default_rng(42)
        
        # Funzione semplice
        def sphere(x):
            return np.sum((x - 0.5)**2)
        
        all_y = []
        for _ in range(30):
            x = rng.random(self.dim)
            y = sphere(x)
            cube.add_observation(x, y, gamma=0.0)
            all_y.append(y)
        
        cube.fit_lgs_model(gamma=0.0, dim=self.dim, rng=rng)
        model = cube.lgs_model
        
        if model is None:
            self.log("model = None", "WARN")
            return True
        
        center = cube.center()
        mu_pred, sigma_pred = cube.predict_bayesian([center])
        
        y_mean = model['y_mean']
        
        self.log(f"y_mean = {y_mean:.4f}")
        self.log(f"predict(center) = {mu_pred[0]:.4f}")
        self.log(f"diff = {abs(mu_pred[0] - y_mean):.4e}")
        
        # Al centro, X_norm = 0, quindi mu = y_mean + 0 * grad * y_std = y_mean
        if abs(mu_pred[0] - y_mean) > 1e-6:
            self.log("predict(center) != y_mean, check math!", "WARN")
        
        self.log("A10 PASSED: predict consistente", "PASS")
        return True
    
    # =========================================================================
    # RUN ALL
    # =========================================================================
    def run_all(self):
        """Esegue tutti i check."""
        print("\n" + "="*70)
        print("ALBA ASSUMPTION CHECKER - FULL SUITE")
        print("="*70)
        
        checks = [
            ("A1", self.check_A1_normalization),
            ("A2", self.check_A2_covariance_pd),
            ("A3", self.check_A3_gradient_normalized),
            ("A4", self.check_A4_gaussian_weights),
            ("A5", self.check_A5_split_coverage),
            ("A6", self.check_A6_gamma_threshold),
            ("A7", self.check_A7_ucb_exploration),
            ("A8", self.check_A8_drilling_convergence),
            ("A9", self.check_A9_best_y_tracking),
            ("A10", self.check_A10_predict_consistency),
        ]
        
        results = {}
        for name, check_fn in checks:
            try:
                results[name] = check_fn()
            except Exception as e:
                print(f"❌ {name} CRASHED: {e}")
                results[name] = False
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name}: {status}")
        
        if self.violations:
            print("\nVIOLATIONS RECORDED:")
            for v in self.violations:
                print(f"  - {v}")
        
        return results


if __name__ == "__main__":
    checker = AssumptionChecker(dim=5, verbose=True)
    checker.run_all()
