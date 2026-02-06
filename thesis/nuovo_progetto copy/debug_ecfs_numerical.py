#!/usr/bin/env python3
"""
Debug numerico ECFS - Verifica empirica delle ipotesi sui bug.

Ipotesi da verificare:
1. LogPDF inconsistente con Sample (chol vs chol.T)
2. Log-ratio scoring non discrimina (std ~ 0)
3. Covarianza mal condizionata (cond >> 1e8)
4. Step scale sbagliato
5. Elite selection non informativa
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from ecfs import ECFS, ECFSConfig, _GaussianModel

np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# TEST 1: Verifica consistenza LogPDF vs Sample
# =============================================================================
def test_logpdf_sample_consistency():
    """
    Se sample e logpdf sono consistenti, campioni dalla distribuzione 
    dovrebbero avere logpdf medio vicino al valore teorico.
    
    Per N(0, Sigma) in d dimensioni:
    E[logpdf] = -d/2 * (1 + log(2*pi)) - 0.5 * log(det(Sigma))
    
    Per Sigma = I: E[logpdf] = -d/2 * (1 + log(2*pi))
    """
    print("=" * 60)
    print("TEST 1: Consistenza LogPDF vs Sample")
    print("=" * 60)
    
    for d in [2, 5, 10, 20]:
        # Crea covarianza casuale PD
        A = np.random.randn(d, d)
        cov = A @ A.T + 0.1 * np.eye(d)
        mu = np.random.randn(d)
        
        # Crea modello
        chol = np.linalg.cholesky(cov)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        model = _GaussianModel(mu=mu, chol=chol, logdet=logdet)
        
        # Campiona
        rng = np.random.default_rng(42)
        samples = model.sample(rng, 10000)
        
        # Calcola logpdf
        logp = model.logpdf(samples)
        
        # Valore teorico per samples dalla distribuzione
        # E[(x-mu)^T Sigma^-1 (x-mu)] = d (trace of identity)
        # Quindi E[logpdf] = -0.5 * (d + logdet + d*log(2*pi))
        expected_logp = -0.5 * (d + logdet + d * np.log(2 * np.pi))
        
        actual_mean = logp.mean()
        actual_std = logp.std()
        
        diff = abs(actual_mean - expected_logp)
        status = "✓ OK" if diff < 0.5 else "✗ PROBLEMA"
        
        print(f"  d={d:2d}: E[logp]={actual_mean:8.2f}, teorico={expected_logp:8.2f}, "
              f"diff={diff:6.2f}, std={actual_std:5.2f} {status}")
    
    print()


# =============================================================================
# TEST 2: Verifica che il log-ratio discrimini
# =============================================================================
def test_logratio_discrimination():
    """
    Il log-ratio score(Δ) = logp_E(Δ) - logp_N(Δ) dovrebbe:
    - Avere std > 0 (discriminare)
    - Dare score più alto a delta che vengono da elite
    """
    print("=" * 60)
    print("TEST 2: Discriminazione Log-Ratio")
    print("=" * 60)
    
    d = 10
    n_samples = 1000
    rng = np.random.default_rng(42)
    
    # Crea due distribuzioni diverse
    # Elite: covarianza stretta, centrata su [1,0,0,...]
    mu_E = np.zeros(d)
    mu_E[0] = 1.0
    cov_E = 0.1 * np.eye(d)
    chol_E = np.linalg.cholesky(cov_E)
    logdet_E = 2.0 * np.sum(np.log(np.diag(chol_E)))
    model_E = _GaussianModel(mu=mu_E, chol=chol_E, logdet=logdet_E)
    
    # Non-elite: covarianza larga, centrata su origine
    mu_N = np.zeros(d)
    cov_N = 1.0 * np.eye(d)
    chol_N = np.linalg.cholesky(cov_N)
    logdet_N = 2.0 * np.sum(np.log(np.diag(chol_N)))
    model_N = _GaussianModel(mu=mu_N, chol=chol_N, logdet=logdet_N)
    
    # Campiona da entrambe
    samples_E = model_E.sample(rng, n_samples)
    samples_N = model_N.sample(rng, n_samples)
    
    # Calcola log-ratio per campioni da E
    logp_E_on_E = model_E.logpdf(samples_E)
    logp_N_on_E = model_N.logpdf(samples_E)
    score_E = logp_E_on_E - logp_N_on_E
    
    # Calcola log-ratio per campioni da N
    logp_E_on_N = model_E.logpdf(samples_N)
    logp_N_on_N = model_N.logpdf(samples_N)
    score_N = logp_E_on_N - logp_N_on_N
    
    print(f"  Campioni da Elite:     score mean={score_E.mean():7.2f}, std={score_E.std():6.2f}")
    print(f"  Campioni da Non-Elite: score mean={score_N.mean():7.2f}, std={score_N.std():6.2f}")
    print(f"  Differenza medie: {score_E.mean() - score_N.mean():.2f}")
    
    # Il log-ratio dovrebbe essere più alto per campioni da E
    if score_E.mean() > score_N.mean() + 1.0:
        print("  ✓ Log-ratio discrimina correttamente")
    else:
        print("  ✗ PROBLEMA: Log-ratio non discrimina abbastanza")
    
    print()


# =============================================================================
# TEST 3: Condizione covarianza durante ottimizzazione reale
# =============================================================================
def test_covariance_condition_during_optimization():
    """
    Esegue ECFS su Sphere e traccia il condition number delle covarianze.
    """
    print("=" * 60)
    print("TEST 3: Condition Number durante Ottimizzazione")
    print("=" * 60)
    
    def sphere(x):
        return float(np.sum(x**2))
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    # Modifica ECFS per logging
    class ECFSWithLogging(ECFS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_cond_E = []
            self.log_cond_N = []
            self.log_score_std = []
            self.log_n_elite = []
        
        def ask(self):
            x = super().ask()
            return x
    
    # Esegui
    opt = ECFSWithLogging(bounds, seed=42)
    
    cond_E_list = []
    cond_N_list = []
    score_std_list = []
    n_elite_list = []
    
    for i in range(200):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        
        # Ogni 20 iterazioni, ispeziona lo stato interno
        if i > 0 and i % 20 == 0:
            n = len(opt.y_hist)
            if n > 10:
                tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
                n_elite = int(np.sum(opt.y_hist <= tau))
                n_elite_list.append(n_elite)
    
    print(f"  Budget: 200, dim: {d}")
    print(f"  Best found: {opt.best_y:.6f}")
    print(f"  N elite nel tempo: {n_elite_list}")
    print()


# =============================================================================
# TEST 4: Verifica che i delta siano informativi
# =============================================================================
def test_delta_informativeness():
    """
    I delta (X_elite - anchor) dovrebbero puntare verso regioni migliori.
    Su Sphere, dovrebbero puntare verso l'origine.
    """
    print("=" * 60)
    print("TEST 4: Informatività dei Delta")
    print("=" * 60)
    
    def sphere(x):
        return float(np.sum(x**2))
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    # Simula storia
    rng = np.random.default_rng(42)
    n_points = 100
    
    # Genera punti random
    X_hist = rng.uniform(-5, 5, (n_points, d))
    y_hist = np.array([sphere(x) for x in X_hist])
    
    # Trova elite e non-elite
    tau = np.quantile(y_hist, 0.2)
    elite_mask = y_hist <= tau
    nonelite_mask = y_hist > tau
    
    X_elite = X_hist[elite_mask]
    X_nonelite = X_hist[nonelite_mask]
    
    print(f"  N points: {n_points}, N elite: {elite_mask.sum()}, N nonelite: {nonelite_mask.sum()}")
    
    # Anchor = best point
    best_idx = np.argmin(y_hist)
    anchor = X_hist[best_idx]
    anchor_norm = np.linalg.norm(anchor)
    
    print(f"  Anchor (best): norm = {anchor_norm:.4f}, y = {y_hist[best_idx]:.4f}")
    
    # Calcola delta verso elite
    deltas_E = X_elite - anchor
    deltas_N = X_nonelite - anchor
    
    # Su Sphere, un buon delta dovrebbe puntare verso origine
    # Cioè: delta · (-anchor) > 0 (nella direzione opposta all'anchor)
    
    # Cosine similarity con direzione verso origine
    direction_to_origin = -anchor / (np.linalg.norm(anchor) + 1e-9)
    
    cos_E = np.array([np.dot(d, direction_to_origin) / (np.linalg.norm(d) + 1e-9) for d in deltas_E])
    cos_N = np.array([np.dot(d, direction_to_origin) / (np.linalg.norm(d) + 1e-9) for d in deltas_N])
    
    print(f"  Cosine(delta_E, verso_origine): mean={cos_E.mean():.3f}, std={cos_E.std():.3f}")
    print(f"  Cosine(delta_N, verso_origine): mean={cos_N.mean():.3f}, std={cos_N.std():.3f}")
    
    if cos_E.mean() > cos_N.mean():
        print("  ✓ Delta elite puntano più verso origine (corretto)")
    else:
        print("  ✗ PROBLEMA: Delta elite NON puntano verso origine meglio di non-elite")
    
    # Verifica: la media dei delta elite punta verso origine?
    mean_delta_E = deltas_E.mean(axis=0)
    cos_mean = np.dot(mean_delta_E, direction_to_origin) / (np.linalg.norm(mean_delta_E) + 1e-9)
    print(f"  Cosine(mean_delta_E, verso_origine): {cos_mean:.3f}")
    
    print()


# =============================================================================
# TEST 5: Verifica step scale effect
# =============================================================================
def test_step_scale_effect():
    """
    Testa diversi step_scale e vede l'impatto.
    """
    print("=" * 60)
    print("TEST 5: Effetto Step Scale")
    print("=" * 60)
    
    def sphere(x):
        return float(np.sum(x**2))
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    budget = 200
    
    results = []
    
    for step_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        opt = ECFS(bounds, step_scale=step_scale, seed=42)
        
        for _ in range(budget):
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
        
        results.append((step_scale, opt.best_y))
        print(f"  step_scale={step_scale:.1f}: best_y={opt.best_y:.4f}")
    
    best_ss, best_val = min(results, key=lambda x: x[1])
    print(f"  Migliore step_scale: {best_ss} con best_y={best_val:.4f}")
    print()


# =============================================================================
# TEST 6: Ispezione interna durante ask()
# =============================================================================
def test_internal_state_inspection():
    """
    Modifica temporaneamente ECFS per stampare stato interno.
    """
    print("=" * 60)
    print("TEST 6: Ispezione Stato Interno")
    print("=" * 60)
    
    def sphere(x):
        return float(np.sum(x**2))
    
    d = 5  # Basso per leggibilità
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    # Warmup
    for i in range(30):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    print(f"  Dopo 30 iterazioni:")
    print(f"    n_obs: {len(opt.y_hist)}")
    print(f"    best_y: {opt.best_y:.4f}")
    print(f"    best_x norm: {np.linalg.norm(opt.best_x):.4f}")
    
    # Calcola stato manualmente
    tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
    n_elite = int(np.sum(opt.y_hist <= tau))
    print(f"    tau (elite threshold): {tau:.4f}")
    print(f"    n_elite: {n_elite}")
    
    # Ora ispeziona cosa succede in un ask()
    print("\n  Prossimo ask():")
    
    # Normalizza - ECFS usa Xn_hist (già normalizzato) e lower/upper
    Xn = opt.Xn_hist  # già normalizzato
    anchor_Xn = (opt.best_x - opt.lower) / (opt._range + 1e-12)
    
    elite_mask = np.array(opt.y_hist) <= tau
    Xn_elite = Xn[elite_mask]
    Xn_nonelite = Xn[~elite_mask]
    
    print(f"    anchor_Xn: {anchor_Xn}")
    print(f"    n_elite vicini: {len(Xn_elite)}")
    
    # Calcola delta
    deltas_E = Xn_elite - anchor_Xn
    print(f"    deltas_E shape: {deltas_E.shape}")
    print(f"    deltas_E mean: {deltas_E.mean(axis=0)}")
    print(f"    deltas_E std per dim: {deltas_E.std(axis=0)}")
    
    # Covarianza
    if len(deltas_E) >= 3:
        cov_E = np.cov(deltas_E, rowvar=False)
        if cov_E.ndim == 0:
            cov_E = np.array([[cov_E]])
        cond_E = np.linalg.cond(cov_E)
        print(f"    cov_E condition number: {cond_E:.2e}")
        print(f"    cov_E diagonal: {np.diag(cov_E)}")
    
    print()


# =============================================================================
# TEST 7: Confronto diretto ECFS variants
# =============================================================================
def test_ablation():
    """
    Testa le varianti ablation per vedere quale componente aiuta/fa male.
    """
    print("=" * 60)
    print("TEST 7: Ablation (use_ratio, diag_cov, mu_zero)")
    print("=" * 60)
    
    def sphere(x):
        return float(np.sum(x**2))
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    budget = 200
    n_seeds = 3
    
    configs = [
        ("Default", {}),
        ("use_ratio=False", {"use_ratio": False}),
        ("diag_cov=True", {"diag_cov": True}),
        ("mu_zero=True", {"mu_zero": True}),
        ("diag+no_ratio", {"use_ratio": False, "diag_cov": True}),
    ]
    
    for name, kwargs in configs:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS(bounds, seed=seed, **kwargs)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {name:20s}: {mean_score:.4f} ± {std_score:.4f}")
    
    print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ECFS NUMERICAL DEBUG SESSION")
    print("=" * 60 + "\n")
    
    test_logpdf_sample_consistency()
    test_logratio_discrimination()
    test_covariance_condition_during_optimization()
    test_delta_informativeness()
    test_step_scale_effect()
    test_internal_state_inspection()
    test_ablation()
    
    print("=" * 60)
    print("DEBUG SESSION COMPLETE")
    print("=" * 60)
