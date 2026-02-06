#!/usr/bin/env python3
"""
Test varianti di costruzione dei delta per ECFS.

Problema attuale: delta = X_elite - anchor, ma gli elite sono spesso PEGGIORI del best.
Quindi μ = mean(delta) punta verso punti peggiori.

Varianti da testare:
A) Delta "Better-than-anchor": solo punti migliori dell'anchor
B) Delta "Improvement": passi che hanno portato miglioramenti (X[t] - X[t-1] quando y[t] < y[t-1])
C) Delta "To-best": direzione verso il best (best - X)
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from ecfs import ECFSConfig

np.set_printoptions(precision=3, suppress=True)


# =============================================================================
# Funzioni Test
# =============================================================================

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    d = len(x)
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


# =============================================================================
# ECFS Varianti
# =============================================================================

class ECFS_Variant:
    """ECFS con diverse strategie per costruire i delta."""
    
    def __init__(self, bounds, seed=0, delta_strategy="elite", 
                 gamma=0.2, step_scale=1.0, reg=1e-6, alpha_shrink=0.1):
        bounds_arr = np.asarray(bounds, dtype=float)
        self.lower = bounds_arr[:, 0]
        self.upper = bounds_arr[:, 1]
        self._range = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        self.d = bounds_arr.shape[0]
        
        self.delta_strategy = delta_strategy  # "elite", "better", "improvement", "to_best"
        self.gamma = gamma
        self.step_scale = step_scale
        self.reg = reg
        self.alpha_shrink = alpha_shrink
        
        self.rng = np.random.default_rng(seed)
        
        self.X_hist = np.empty((0, self.d), dtype=float)
        self.y_hist = np.empty((0,), dtype=float)
        
        self.best_x = None
        self.best_y = float("inf")
    
    def _normalize(self, x):
        return (x - self.lower) / self._range
    
    def _denormalize(self, xn):
        return self.lower + xn * self._range
    
    def ask(self):
        n = len(self.y_hist)
        n_min = max(10, 2 * self.d)
        
        # Exploration
        if n < n_min or self.rng.random() < 0.05:
            return self._denormalize(self.rng.random(self.d))
        
        # Get anchor
        if self.best_x is not None:
            anchor = self.best_x.copy()
        else:
            return self._denormalize(self.rng.random(self.d))
        
        anchor_Xn = self._normalize(anchor)
        anchor_y = self.best_y
        
        # Build deltas based on strategy
        deltas = self._build_deltas(anchor_Xn, anchor_y)
        
        if len(deltas) < 2:
            # Fallback
            sigma = 0.15
            Xn = anchor_Xn + self.step_scale * self.rng.normal(0, sigma, self.d)
            return self._denormalize(np.clip(Xn, 0, 1))
        
        # Fit Gaussian
        mu = deltas.mean(axis=0)
        centered = deltas - deltas.mean(axis=0)
        cov = (centered.T @ centered) / max(len(deltas) - 1, 1)
        cov = cov + self.reg * np.eye(self.d)
        
        # Shrinkage
        tr = np.trace(cov)
        target = (tr / self.d) * np.eye(self.d)
        cov = (1 - self.alpha_shrink) * cov + self.alpha_shrink * target
        
        # Sample
        try:
            L = np.linalg.cholesky(cov)
        except:
            L = np.eye(self.d) * 0.1
        
        z = self.rng.normal(0, 1, self.d)
        delta = mu + L @ z
        
        Xn_new = anchor_Xn + self.step_scale * delta
        Xn_new = np.clip(Xn_new, 0, 1)
        
        return self._denormalize(Xn_new)
    
    def _build_deltas(self, anchor_Xn, anchor_y):
        """Costruisce i delta in base alla strategia."""
        
        X_Xn = np.array([self._normalize(x) for x in self.X_hist])
        
        if self.delta_strategy == "elite":
            # Originale: delta = X_elite - anchor
            tau = np.quantile(self.y_hist, self.gamma)
            elite_mask = self.y_hist <= tau
            X_elite_Xn = X_Xn[elite_mask]
            
            # Escludi anchor stesso
            dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            X_elite_Xn = X_elite_Xn[keep]
            
            return X_elite_Xn - anchor_Xn
        
        elif self.delta_strategy == "better":
            # VARIANTE A: Solo punti MIGLIORI dell'anchor
            better_mask = self.y_hist < anchor_y
            X_better_Xn = X_Xn[better_mask]
            
            if len(X_better_Xn) == 0:
                # Nessun punto migliore, fallback a elite
                tau = np.quantile(self.y_hist, self.gamma)
                elite_mask = self.y_hist <= tau
                X_better_Xn = X_Xn[elite_mask]
            
            dists = np.linalg.norm(X_better_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            X_better_Xn = X_better_Xn[keep]
            
            return X_better_Xn - anchor_Xn
        
        elif self.delta_strategy == "improvement":
            # VARIANTE B: Passi che hanno portato miglioramenti
            # delta = X[t] - X[t-1] quando y[t] < y[t-1]
            deltas = []
            for t in range(1, len(self.y_hist)):
                if self.y_hist[t] < self.y_hist[t-1]:
                    delta = X_Xn[t] - X_Xn[t-1]
                    deltas.append(delta)
            
            if len(deltas) < 2:
                # Fallback
                tau = np.quantile(self.y_hist, self.gamma)
                elite_mask = self.y_hist <= tau
                X_elite_Xn = X_Xn[elite_mask]
                dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
                keep = dists > 1e-9
                return X_elite_Xn[keep] - anchor_Xn
            
            return np.array(deltas)
        
        elif self.delta_strategy == "to_best":
            # VARIANTE C: Direzione verso il best (invertita!)
            # delta = best - X (per tutti i punti)
            best_Xn = self._normalize(self.best_x)
            
            # Usa tutti i punti non-best
            dists = np.linalg.norm(X_Xn - best_Xn, axis=1)
            keep = dists > 1e-9
            X_others_Xn = X_Xn[keep]
            
            # Delta = direzione DA ogni punto VERSO il best
            return best_Xn - X_others_Xn
        
        elif self.delta_strategy == "mu_zero":
            # Baseline: μ=0, solo covarianza
            tau = np.quantile(self.y_hist, self.gamma)
            elite_mask = self.y_hist <= tau
            X_elite_Xn = X_Xn[elite_mask]
            
            dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            X_elite_Xn = X_elite_Xn[keep]
            
            deltas = X_elite_Xn - anchor_Xn
            # Ritorna delta con media artificialmente a 0
            return deltas - deltas.mean(axis=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.delta_strategy}")
    
    def tell(self, x, y):
        x = np.asarray(x, dtype=float)
        self.X_hist = np.vstack([self.X_hist, x[None, :]])
        self.y_hist = np.concatenate([self.y_hist, [float(y)]])
        
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()


# =============================================================================
# Test
# =============================================================================

def run_test(fn, fn_name, bounds, d, budget, n_seeds):
    """Testa tutte le varianti su una funzione."""
    
    strategies = ["elite", "better", "improvement", "to_best", "mu_zero"]
    
    results = {}
    for strat in strategies:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS_Variant(bounds, seed=seed, delta_strategy=strat)
            for _ in range(budget):
                x = opt.ask()
                y = fn(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        results[strat] = (np.mean(scores), np.std(scores))
    
    print(f"\n  {fn_name}:")
    print(f"  " + "-" * 55)
    
    best_strat = min(results.keys(), key=lambda k: results[k][0])
    
    for strat in strategies:
        mean, std = results[strat]
        marker = " ✓ BEST" if strat == best_strat else ""
        
        # Descrizione
        desc = {
            "elite": "X_elite - anchor (originale)",
            "better": "X_better - anchor (solo migliori)",
            "improvement": "X[t]-X[t-1] se y[t]<y[t-1]",
            "to_best": "best - X (verso il best)",
            "mu_zero": "μ=0 (solo covarianza)",
        }
        
        print(f"    {strat:12s}: {mean:10.4f} ± {std:7.4f}  [{desc[strat]}]{marker}")
    
    return results


def main():
    print("=" * 70)
    print("TEST VARIANTI COSTRUZIONE DELTA")
    print("=" * 70)
    
    d = 10
    budget = 300
    n_seeds = 10
    
    functions = [
        ("Sphere", sphere, [(-5, 5)] * d),
        ("Rosenbrock", rosenbrock, [(-5, 10)] * d),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * d),
    ]
    
    all_results = {}
    for fn_name, fn, bounds in functions:
        all_results[fn_name] = run_test(fn, fn_name, bounds, d, budget, n_seeds)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Quale strategia vince su ogni funzione?")
    print("=" * 70)
    
    for fn_name, results in all_results.items():
        best = min(results.keys(), key=lambda k: results[k][0])
        print(f"  {fn_name:12s}: {best}")
    
    # Analisi μ
    print("\n" + "=" * 70)
    print("ANALISI: μ funziona con delta corretti?")
    print("=" * 70)
    
    for fn_name, results in all_results.items():
        # Confronta "better" (usa μ) vs "mu_zero"
        better_mean = results["better"][0]
        muzero_mean = results["mu_zero"][0]
        
        if better_mean < muzero_mean:
            verdict = f"μ AIUTA! ({better_mean:.2f} < {muzero_mean:.2f})"
        else:
            verdict = f"μ non aiuta ({better_mean:.2f} >= {muzero_mean:.2f})"
        
        print(f"  {fn_name:12s}: {verdict}")


if __name__ == "__main__":
    main()
