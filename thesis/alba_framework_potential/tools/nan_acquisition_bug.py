#!/usr/bin/env python3
"""
BUG ANALYSIS: NaN handling in acquisition.py
=============================================

Problema: quando mu/sigma contengono NaN o tutti i valori sono NaN,
l'algoritmo non fallisce ma può selezionare indici casuali.

Questo può portare a comportamenti imprevedibili.
"""

import numpy as np


def simulate_nan_behavior():
    """Simula il comportamento con NaN."""
    
    print("=" * 70)
    print("ANALYSIS: NaN in acquisition.py")
    print("=" * 70)
    
    # Caso 1: Alcuni NaN
    print("\n1. Alcuni mu sono NaN:")
    mu = np.array([1.0, float('nan'), 3.0, 4.0])
    sigma = np.array([0.1, 0.1, 0.1, 0.1])
    
    score = mu + 2.0 * sigma
    print(f"  score = mu + 2*sigma = {score}")
    
    score_mean = score.mean()
    score_std = score.std()
    print(f"  score.mean() = {score_mean}")
    print(f"  score.std() = {score_std}")
    
    # La std è > 1e-9 (è NaN ma non viene controllato)
    if score_std > 1e-9:
        score_z = (score - score_mean) / score_std
    else:
        score_z = np.zeros_like(score)
    
    print(f"  score_z = {score_z}")
    
    probs = np.exp(score_z * 3.0)
    probs = probs / probs.sum()
    print(f"  probs = {probs}")
    print(f"  probs.sum() = {probs.sum()}")
    
    # Caso 2: Tutti NaN
    print("\n2. Tutti mu sono NaN:")
    mu_all_nan = np.array([float('nan')] * 4)
    sigma_all_nan = np.array([float('nan')] * 4)
    
    score2 = mu_all_nan + 2.0 * sigma_all_nan
    print(f"  score = {score2}")
    
    score_std2 = score2.std()
    print(f"  score.std() = {score_std2}")
    print(f"  score.std() > 1e-9 = {score_std2 > 1e-9}")  # False! NaN > x = False
    
    # Quindi score_z diventa zeros!
    score_z2 = np.zeros_like(score2)
    print(f"  score_z (zeros) = {score_z2}")
    
    probs2 = np.exp(score_z2 * 3.0)
    probs2 = probs2 / probs2.sum()
    print(f"  probs = {probs2}")
    
    # Questo funziona! Uniforme su tutti gli indici
    print("\n  → Con tutti NaN, seleziona uniformemente (casualità)")
    print("  → Non è un crash, ma potrebbe nascondere problemi")


def propose_fix():
    print("\n" + "=" * 70)
    print("PROPOSTA FIX")
    print("=" * 70)
    
    print("""
Ci sono due approcci:

1. FAIL-FAST: Solleva un'eccezione se ci sono NaN
   Pro: Rende evidente il problema
   Contro: Può crashare il sistema

2. SANITIZE: Sostituisci NaN con valori neutri
   Pro: Robusto
   Contro: Nasconde problemi

Per ALBA, suggeriamo l'approccio 2 (sanitize) perché:
- I NaN possono venire da modelli LGS non ancora fittati
- È meglio continuare l'ottimizzazione che crashare

Fix proposta:
""")
    
    fix_code = '''
def select(self, mu, sigma, rng, novelty_weight):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # BUG FIX: Handle NaN/Inf in mu and sigma
    # Replace NaN with median of valid values, or 0 if all NaN
    # This prevents silent selection of invalid candidates
    valid_mu = np.isfinite(mu)
    valid_sigma = np.isfinite(sigma)
    
    if not valid_mu.all():
        if valid_mu.any():
            mu = np.where(valid_mu, mu, np.median(mu[valid_mu]))
        else:
            mu = np.zeros_like(mu)  # All NaN → uniform selection
    
    if not valid_sigma.all():
        if valid_sigma.any():
            sigma = np.where(valid_sigma, sigma, np.median(sigma[valid_sigma]))
        else:
            sigma = np.ones_like(sigma)  # All NaN → equal uncertainty
    
    # Rest of the function...
'''
    print(fix_code)


if __name__ == "__main__":
    simulate_nan_behavior()
    propose_fix()
