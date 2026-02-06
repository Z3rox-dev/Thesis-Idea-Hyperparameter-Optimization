"""
Confronto diretto BUGGY vs FIXED per le funzioni dove esplodeva il gradiente.

Questo script crea due implementazioni di LGS:
1. BUGGY: grad *= y_std (causa esplosione)  
2. FIXED: grad in spazio normalizzato (de-norm solo in predict)

E le confronta sulle stesse funzioni.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================================
# IMPLEMENTAZIONE BUGGY (originale)
# ============================================================================

def fit_lgs_BUGGY(all_pts, all_scores, widths, center, dim):
    """Versione BUGGY: grad * y_std"""
    gradient_dir = None
    grad = None
    inv_cov = None
    y_mean = 0.0
    y_std = 1.0
    noise_var = 1.0
    
    if len(all_pts) >= dim + 3:
        X_norm = (all_pts - center) / widths
        y_mean = all_scores.mean()
        y_std = all_scores.std() + 1e-6
        y_centered = (all_scores - y_mean) / y_std

        dists_sq = np.sum(X_norm**2, axis=1)
        sigma_sq = np.mean(dists_sq) + 1e-6
        weights = np.exp(-dists_sq / (2 * sigma_sq))
        rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (all_scores.ptp() + 1e-9)
        weights = weights * rank_weights
        W = np.diag(weights)

        lambda_base = 0.1 * (1 + dim / max(len(all_pts) - dim, 1))
        XtWX = X_norm.T @ W @ X_norm
        XtWX_reg = XtWX + lambda_base * np.eye(dim)
        inv_cov = np.linalg.inv(XtWX_reg)
        
        grad = inv_cov @ (X_norm.T @ W @ y_centered)
        grad = grad * y_std  # BUG! Esplode quando y_std >> 1

        y_pred = X_norm @ grad / y_std
        residuals = y_centered - y_pred
        noise_var = np.clip(np.average(residuals**2, weights=weights) * (y_std**2) + 1e-6, 1e-4, 10.0)

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            gradient_dir = grad / grad_norm

    return {"grad": grad, "inv_cov": inv_cov, "y_mean": y_mean, "noise_var": noise_var,
            "widths": widths, "center": center, "gradient_dir": gradient_dir}


def predict_BUGGY(model, candidates):
    """Versione BUGGY: mu = y_mean + C_norm @ grad (grad già scalato)"""
    if model is None or model.get("inv_cov") is None:
        return np.zeros(len(candidates)), np.ones(len(candidates))

    C_norm = (np.array(candidates) - model["center"]) / model["widths"]
    mu = model["y_mean"] + C_norm @ model["grad"]  # BUG: grad è O(y_std)!

    model_var = np.clip(np.sum((C_norm @ model["inv_cov"]) * C_norm, axis=1), 0, 10.0)
    total_var = model["noise_var"] * (1.0 + model_var)
    sigma = np.sqrt(total_var)

    return mu, sigma


# ============================================================================
# IMPLEMENTAZIONE FIXED (corretta)
# ============================================================================

def fit_lgs_FIXED(all_pts, all_scores, widths, center, dim):
    """Versione FIXED: grad in spazio normalizzato"""
    gradient_dir = None
    grad = None
    inv_cov = None
    y_mean = 0.0
    y_std = 1.0
    noise_var = 1.0
    
    if len(all_pts) >= dim + 3:
        X_norm = (all_pts - center) / widths
        y_mean = all_scores.mean()
        y_std = all_scores.std() + 1e-6
        y_centered = (all_scores - y_mean) / y_std

        dists_sq = np.sum(X_norm**2, axis=1)
        sigma_sq = np.mean(dists_sq) + 1e-6
        weights = np.exp(-dists_sq / (2 * sigma_sq))
        rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (all_scores.ptp() + 1e-9)
        weights = weights * rank_weights
        W = np.diag(weights)

        lambda_base = 0.1 * (1 + dim / max(len(all_pts) - dim, 1))
        XtWX = X_norm.T @ W @ X_norm
        XtWX_reg = XtWX + lambda_base * np.eye(dim)
        inv_cov = np.linalg.inv(XtWX_reg)
        
        # FIXED: grad resta in spazio normalizzato (NO * y_std)
        grad = inv_cov @ (X_norm.T @ W @ y_centered)

        y_pred = X_norm @ grad
        residuals = y_centered - y_pred
        noise_var = np.clip(np.average(residuals**2, weights=weights) + 1e-6, 1e-4, 10.0)

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            gradient_dir = grad / grad_norm

    return {"grad": grad, "inv_cov": inv_cov, "y_mean": y_mean, "y_std": y_std,
            "noise_var": noise_var, "widths": widths, "center": center, 
            "gradient_dir": gradient_dir}


def predict_FIXED(model, candidates):
    """Versione FIXED: de-normalizza solo alla fine"""
    if model is None or model.get("inv_cov") is None:
        return np.zeros(len(candidates)), np.ones(len(candidates))

    y_std = model.get("y_std", 1.0)
    C_norm = (np.array(candidates) - model["center"]) / model["widths"]
    
    # FIXED: predizione in spazio normalizzato, poi de-normalizza
    mu_norm = C_norm @ model["grad"]
    mu = model["y_mean"] + mu_norm * y_std

    model_var = np.clip(np.sum((C_norm @ model["inv_cov"]) * C_norm, axis=1), 0, 10.0)
    total_var = model["noise_var"] * (1.0 + model_var)
    sigma = np.sqrt(total_var) * y_std

    return mu, sigma


# ============================================================================
# MINI-OTTIMIZZATORE per confronto
# ============================================================================

class MiniOptimizer:
    """Ottimizzatore minimale che usa solo LGS per campionare."""
    
    def __init__(self, dim, bounds, fit_fn, predict_fn, seed=42):
        self.dim = dim
        self.bounds = bounds
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.rng = np.random.default_rng(seed)
        self.history_x = []
        self.history_y = []
        self.best_x = None
        self.best_y = float('inf')
    
    def ask(self):
        # Prima fase: random exploration
        if len(self.history_x) < 2 * self.dim:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        
        # Fit LGS
        all_pts = np.array(self.history_x)
        all_scores = np.array(self.history_y)
        widths = np.array([hi - lo for lo, hi in self.bounds])
        center = np.array([(lo + hi) / 2 for lo, hi in self.bounds])
        
        model = self.fit_fn(all_pts, all_scores, widths, center, self.dim)
        
        # Generate candidates
        n_cand = 20
        candidates = []
        for _ in range(n_cand):
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            candidates.append(x)
        
        # Select best (minimize = lowest predicted mu)
        mu, sigma = self.predict_fn(model, candidates)
        
        # UCB-style: prefer low mu with high uncertainty
        acq = mu - 1.5 * sigma
        best_idx = np.argmin(acq)
        
        return candidates[best_idx]
    
    def tell(self, x, y):
        self.history_x.append(x)
        self.history_y.append(y)
        if y < self.best_y:
            self.best_y = y
            self.best_x = x


# ============================================================================
# BENCHMARK
# ============================================================================

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def ill_conditioned(x):
    scales = np.array([10**i for i in range(len(x))])
    return np.sum(scales * x**2)


if __name__ == "__main__":
    DIM = 5
    BUDGET = 150
    N_SEEDS = 10
    
    funcs = [
        ("Rosenbrock", rosenbrock),
        ("IllConditioned", ill_conditioned),
    ]
    
    results = {"BUGGY": {}, "FIXED": {}}
    
    for version, fit_fn, predict_fn in [
        ("BUGGY", fit_lgs_BUGGY, predict_BUGGY),
        ("FIXED", fit_lgs_FIXED, predict_FIXED),
    ]:
        print(f"\n{'='*60}")
        print(f"  {version} VERSION")
        print(f"{'='*60}")
        
        for func_name, func in funcs:
            bests = []
            for seed in range(N_SEEDS):
                opt = MiniOptimizer(
                    dim=DIM,
                    bounds=[(0.0, 1.0)] * DIM,
                    fit_fn=fit_fn,
                    predict_fn=predict_fn,
                    seed=seed
                )
                
                for _ in range(BUDGET):
                    x = opt.ask()
                    y = func(x)
                    opt.tell(x, y)
                
                bests.append(opt.best_y)
            
            mean_best = np.mean(bests)
            std_best = np.std(bests)
            results[version][func_name] = {"mean": mean_best, "std": std_best}
            print(f"  {func_name}: {mean_best:.4f} ± {std_best:.4f}")
    
    # Confronto finale
    print("\n" + "="*70)
    print("  CONFRONTO FINALE: BUGGY vs FIXED")
    print("="*70)
    print(f"{'Funzione':<20} {'BUGGY':<18} {'FIXED':<18} {'Miglioramento':<15}")
    print("-"*70)
    
    for func_name in results["BUGGY"]:
        buggy = results["BUGGY"][func_name]["mean"]
        fixed = results["FIXED"][func_name]["mean"]
        improvement = (buggy - fixed) / buggy * 100
        
        buggy_str = f"{buggy:.4f}"
        fixed_str = f"{fixed:.4f}"
        
        if fixed < buggy:
            print(f"{func_name:<20} {buggy_str:<18} {fixed_str:<18} ✅ {improvement:.1f}% meglio")
        elif fixed > buggy:
            print(f"{func_name:<20} {buggy_str:<18} {fixed_str:<18} ❌ {-improvement:.1f}% peggio")
        else:
            print(f"{func_name:<20} {buggy_str:<18} {fixed_str:<18} = uguale")
