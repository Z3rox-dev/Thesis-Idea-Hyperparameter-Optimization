#!/usr/bin/env python3
"""
Test varianti ECFS - versione corretta.

Il problema: se anchor = best, non esistono punti "better than anchor".
Soluzione: testare con anchor diversificato (random elite, non solo best).
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

np.set_printoptions(precision=3, suppress=True)


def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    d = len(x)
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


class ECFS_V2:
    """ECFS con costruzione delta corretta."""
    
    def __init__(self, bounds, seed=0, delta_strategy="elite", 
                 gamma=0.2, step_scale=1.0, reg=1e-6, alpha_shrink=0.1,
                 anchor_mode="best"):
        bounds_arr = np.asarray(bounds, dtype=float)
        self.lower = bounds_arr[:, 0]
        self.upper = bounds_arr[:, 1]
        self._range = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        self.d = bounds_arr.shape[0]
        
        self.delta_strategy = delta_strategy
        self.anchor_mode = anchor_mode  # "best" o "random_elite"
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
    
    def _get_anchor(self):
        """Scegli anchor in base al mode."""
        if self.anchor_mode == "best":
            return self.best_x.copy(), self.best_y
        elif self.anchor_mode == "random_elite":
            tau = np.quantile(self.y_hist, self.gamma)
            elite_idx = np.where(self.y_hist <= tau)[0]
            if len(elite_idx) == 0:
                return self.best_x.copy(), self.best_y
            idx = self.rng.choice(elite_idx)
            return self.X_hist[idx].copy(), self.y_hist[idx]
        else:
            return self.best_x.copy(), self.best_y
    
    def ask(self):
        n = len(self.y_hist)
        n_min = max(10, 2 * self.d)
        
        if n < n_min or self.rng.random() < 0.05:
            return self._denormalize(self.rng.random(self.d))
        
        if self.best_x is None:
            return self._denormalize(self.rng.random(self.d))
        
        anchor, anchor_y = self._get_anchor()
        anchor_Xn = self._normalize(anchor)
        
        deltas, mu_from_deltas = self._build_deltas(anchor_Xn, anchor_y)
        
        if len(deltas) < 2:
            sigma = 0.15
            Xn = anchor_Xn + self.step_scale * self.rng.normal(0, sigma, self.d)
            return self._denormalize(np.clip(Xn, 0, 1))
        
        # Usa μ dai delta o forza a zero
        if self.delta_strategy == "mu_zero":
            mu = np.zeros(self.d)
        else:
            mu = mu_from_deltas
        
        # Covarianza
        centered = deltas - deltas.mean(axis=0)
        cov = (centered.T @ centered) / max(len(deltas) - 1, 1)
        cov = cov + self.reg * np.eye(self.d)
        
        tr = np.trace(cov)
        target = (tr / self.d) * np.eye(self.d)
        cov = (1 - self.alpha_shrink) * cov + self.alpha_shrink * target
        
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
        """Costruisce i delta e ritorna (deltas, mu)."""
        
        X_Xn = np.array([self._normalize(x) for x in self.X_hist])
        
        if self.delta_strategy in ["elite", "mu_zero"]:
            tau = np.quantile(self.y_hist, self.gamma)
            elite_mask = self.y_hist <= tau
            X_elite_Xn = X_Xn[elite_mask]
            
            dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            X_elite_Xn = X_elite_Xn[keep]
            
            deltas = X_elite_Xn - anchor_Xn
            mu = deltas.mean(axis=0) if len(deltas) > 0 else np.zeros(self.d)
            return deltas, mu
        
        elif self.delta_strategy == "better":
            # Solo punti STRETTAMENTE migliori dell'anchor
            better_mask = self.y_hist < anchor_y - 1e-9
            
            if better_mask.sum() >= 3:
                X_better_Xn = X_Xn[better_mask]
            else:
                # Fallback: top-k punti (k=20% dei dati)
                k = max(3, int(len(self.y_hist) * self.gamma))
                top_k_idx = np.argsort(self.y_hist)[:k]
                X_better_Xn = X_Xn[top_k_idx]
            
            dists = np.linalg.norm(X_better_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            X_better_Xn = X_better_Xn[keep]
            
            deltas = X_better_Xn - anchor_Xn
            mu = deltas.mean(axis=0) if len(deltas) > 0 else np.zeros(self.d)
            return deltas, mu
        
        elif self.delta_strategy == "gradient":
            # Stima gradiente locale: direzione media verso punti migliori
            # pesata per quanto sono migliori
            tau = np.quantile(self.y_hist, 0.5)  # usa 50% come soglia
            better_mask = self.y_hist < anchor_y
            
            if better_mask.sum() < 2:
                # Fallback
                elite_mask = self.y_hist <= np.quantile(self.y_hist, self.gamma)
                X_elite_Xn = X_Xn[elite_mask]
                dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
                keep = dists > 1e-9
                deltas = X_elite_Xn[keep] - anchor_Xn
                return deltas, deltas.mean(axis=0) if len(deltas) > 0 else np.zeros(self.d)
            
            X_better_Xn = X_Xn[better_mask]
            y_better = self.y_hist[better_mask]
            
            # Delta normalizzati
            deltas_raw = X_better_Xn - anchor_Xn
            norms = np.linalg.norm(deltas_raw, axis=1, keepdims=True)
            deltas_unit = deltas_raw / (norms + 1e-9)
            
            # Pesi: quanto è migliore (inversamente proporzionale a y)
            improvement = anchor_y - y_better  # positivo se migliore
            weights = improvement / (improvement.sum() + 1e-9)
            
            # Gradiente stimato = media pesata delle direzioni
            gradient = (deltas_unit * weights[:, None]).sum(axis=0)
            
            # I delta sono nella direzione del gradiente, con varianza dagli elite
            tau = np.quantile(self.y_hist, self.gamma)
            elite_mask = self.y_hist <= tau
            X_elite_Xn = X_Xn[elite_mask]
            dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            deltas = X_elite_Xn[keep] - anchor_Xn
            
            # μ = gradiente normalizzato con scala dalla varianza dei delta
            if len(deltas) > 0:
                scale = np.std(deltas)
                mu = gradient * scale * 0.5  # scala ridotta per non dominare
            else:
                mu = np.zeros(self.d)
            
            return deltas, mu
        
        elif self.delta_strategy == "momentum":
            # Accumula "momentum" dalle ultime transizioni migliorative
            momentum = np.zeros(self.d)
            count = 0
            decay = 0.9
            
            for t in range(len(self.y_hist) - 1, 0, -1):
                if self.y_hist[t] < self.y_hist[t-1]:
                    step = X_Xn[t] - X_Xn[t-1]
                    momentum = momentum * decay + step
                    count += 1
                    if count >= 20:
                        break
            
            # Normalizza
            if np.linalg.norm(momentum) > 1e-9:
                momentum = momentum / np.linalg.norm(momentum)
            
            # Covarianza dagli elite
            tau = np.quantile(self.y_hist, self.gamma)
            elite_mask = self.y_hist <= tau
            X_elite_Xn = X_Xn[elite_mask]
            dists = np.linalg.norm(X_elite_Xn - anchor_Xn, axis=1)
            keep = dists > 1e-9
            deltas = X_elite_Xn[keep] - anchor_Xn
            
            if len(deltas) > 0:
                scale = np.std(deltas) * 0.5
                mu = momentum * scale
            else:
                mu = np.zeros(self.d)
            
            return deltas, mu
        
        else:
            raise ValueError(f"Unknown: {self.delta_strategy}")
    
    def tell(self, x, y):
        x = np.asarray(x, dtype=float)
        self.X_hist = np.vstack([self.X_hist, x[None, :]])
        self.y_hist = np.concatenate([self.y_hist, [float(y)]])
        
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()


def run_test(fn, fn_name, bounds, d, budget, n_seeds):
    """Testa varianti."""
    
    configs = [
        ("elite (orig)", "elite", "best"),
        ("mu_zero", "mu_zero", "best"),
        ("better+rnd_anchor", "better", "random_elite"),
        ("gradient", "gradient", "best"),
        ("momentum", "momentum", "best"),
    ]
    
    results = {}
    for name, strat, anchor in configs:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS_V2(bounds, seed=seed, delta_strategy=strat, anchor_mode=anchor)
            for _ in range(budget):
                x = opt.ask()
                y = fn(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        results[name] = (np.mean(scores), np.std(scores))
    
    print(f"\n  {fn_name}:")
    print(f"  " + "-" * 60)
    
    best_name = min(results.keys(), key=lambda k: results[k][0])
    
    for name in [c[0] for c in configs]:
        mean, std = results[name]
        marker = " ✓ BEST" if name == best_name else ""
        print(f"    {name:20s}: {mean:10.4f} ± {std:8.4f}{marker}")
    
    return results


def main():
    print("=" * 70)
    print("TEST VARIANTI DELTA - V2 (con anchor diversificato)")
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
    
    print("\n" + "=" * 70)
    print("CONCLUSIONE: μ può funzionare?")
    print("=" * 70)
    
    for fn_name, results in all_results.items():
        best = min(results.keys(), key=lambda k: results[k][0])
        uses_mu = best not in ["mu_zero"]
        print(f"  {fn_name:12s}: {best:20s} → μ {'AIUTA' if uses_mu else 'non serve'}")


if __name__ == "__main__":
    main()
