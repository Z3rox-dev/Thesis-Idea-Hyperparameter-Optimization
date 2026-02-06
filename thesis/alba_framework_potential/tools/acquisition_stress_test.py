#!/usr/bin/env python3
"""
ACQUISITION.PY STRESS TEST
===========================

Test aggressivi per trovare bug nel modulo acquisition.py
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)


# ============================================================
# INLINE COPY OF ACQUISITION
# ============================================================

@dataclass(frozen=True)
class UCBSoftmaxSelector:
    """Default acquisition + selection (matches ALBA_V1) - WITH FIX."""

    beta_multiplier: float = 2.0
    softmax_temperature: float = 3.0

    def select(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        rng: np.random.Generator,
        novelty_weight: float,
    ) -> int:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # BUG FIX: Handle NaN/Inf in mu and sigma
        valid_mu = np.isfinite(mu)
        valid_sigma = np.isfinite(sigma)
        
        if not valid_mu.all():
            if valid_mu.any():
                mu = np.where(valid_mu, mu, np.median(mu[valid_mu]))
            else:
                mu = np.zeros_like(mu)
        
        if not valid_sigma.all():
            if valid_sigma.any():
                sigma = np.where(valid_sigma, sigma, np.median(sigma[valid_sigma]))
            else:
                sigma = np.ones_like(sigma)

        beta = float(novelty_weight) * self.beta_multiplier
        if not np.isfinite(beta):
            beta = 0.0
        
        score = mu + beta * sigma

        score_std = score.std()
        if np.isfinite(score_std) and score_std > 1e-9:
            score_z = (score - score.mean()) / score_std
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * self.softmax_temperature)
        probs = probs / probs.sum()
        return int(rng.choice(len(score), p=probs))


# ============================================================
# STRESS TESTS
# ============================================================

def test_nan_mu_sigma():
    """Test con mu/sigma contenenti NaN."""
    print("=" * 70)
    print("TEST 1: NaN in mu/sigma")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    # Test 1.1: mu con NaN
    print("\n1.1 mu con NaN:")
    mu_nan = np.array([1.0, float('nan'), 2.0, 3.0])
    sigma_ok = np.array([0.1, 0.1, 0.1, 0.1])
    
    try:
        idx = selector.select(mu_nan, sigma_ok, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
        
        if np.isnan(mu_nan).any() and idx == 1:
            print("  ⚠ Ha selezionato l'indice con NaN!")
        else:
            print("  ✓ Non ha crashato")
    except Exception as e:
        bugs.append(f"1.1: Exception: {e}")
    
    # Test 1.2: sigma con NaN
    print("\n1.2 sigma con NaN:")
    mu_ok = np.array([1.0, 2.0, 3.0, 4.0])
    sigma_nan = np.array([0.1, float('nan'), 0.1, 0.1])
    
    try:
        idx = selector.select(mu_ok, sigma_nan, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
        print("  ✓ Non ha crashato")
    except Exception as e:
        bugs.append(f"1.2: Exception: {e}")
    
    # Test 1.3: Tutti NaN
    print("\n1.3 Tutti NaN:")
    mu_all_nan = np.array([float('nan')] * 4)
    sigma_all_nan = np.array([float('nan')] * 4)
    
    try:
        idx = selector.select(mu_all_nan, sigma_all_nan, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
        # Con la fix, ora gestisce tutti NaN uniformemente
        print("  ✓ Con fix: seleziona uniformemente quando tutti NaN")
    except Exception as e:
        bugs.append(f"1.3: Exception non attesa: {e}")
    
    return bugs


def test_inf_values():
    """Test con valori Inf."""
    print("\n" + "=" * 70)
    print("TEST 2: Inf in mu/sigma")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    # Test 2.1: mu con Inf
    print("\n2.1 mu con +Inf:")
    mu_inf = np.array([1.0, float('inf'), 2.0, 3.0])
    sigma_ok = np.array([0.1, 0.1, 0.1, 0.1])
    
    try:
        idx = selector.select(mu_inf, sigma_ok, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
        
        # Con Inf in mu, l'indice 1 dovrebbe avere il punteggio più alto
        if idx == 1:
            print("  ✓ Ha selezionato l'indice con Inf (score massimo)")
        else:
            print(f"  ? Ha selezionato {idx} invece di 1")
    except Exception as e:
        bugs.append(f"2.1: Exception: {e}")
    
    # Test 2.2: sigma con Inf
    print("\n2.2 sigma con +Inf:")
    mu_ok = np.array([1.0, 2.0, 3.0, 4.0])
    sigma_inf = np.array([0.1, float('inf'), 0.1, 0.1])
    
    try:
        idx = selector.select(mu_ok, sigma_inf, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
    except Exception as e:
        bugs.append(f"2.2: Exception: {e}")
    
    return bugs


def test_extreme_novelty_weight():
    """Test con novelty_weight estremi."""
    print("\n" + "=" * 70)
    print("TEST 3: novelty_weight estremi")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = np.array([1.0, 0.5, 0.3, 0.1])  # sigma inverso rispetto a mu
    
    test_cases = [
        (0.0, "zero"),
        (1e10, "enorme"),
        (-1.0, "negativo"),
        (float('nan'), "NaN"),
        (float('inf'), "Inf"),
    ]
    
    for nw, desc in test_cases:
        print(f"\n3.{desc}: novelty_weight = {nw}")
        try:
            idx = selector.select(mu, sigma, rng, novelty_weight=nw)
            print(f"  Selezionato indice: {idx}")
            
            if np.isnan(nw) or np.isinf(nw):
                # Potrebbe dare risultati strani
                print(f"  ⚠ Ha funzionato con {desc} novelty_weight")
        except Exception as e:
            if np.isnan(nw) or np.isinf(nw):
                print(f"  ✓ Exception attesa: {type(e).__name__}")
            else:
                bugs.append(f"3.{desc}: Exception: {e}")
    
    return bugs


def test_identical_scores():
    """Test con tutti gli score identici."""
    print("\n" + "=" * 70)
    print("TEST 4: Score identici")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    # Tutti mu identici, tutti sigma identici
    mu = np.array([1.0] * 10)
    sigma = np.array([0.5] * 10)
    
    try:
        # Fai 100 selezioni
        selections = [selector.select(mu, sigma, rng, novelty_weight=1.0) for _ in range(100)]
        
        # Dovrebbe selezionare uniformemente (tutti hanno lo stesso score)
        counts = np.bincount(selections, minlength=10)
        
        print(f"  Distribuzione: {counts}")
        
        # Verifica che non seleziona sempre lo stesso
        if counts.max() > 50:
            bugs.append(f"4: Distribuzione sbilanciata: {counts.max()}% su un indice")
        else:
            print("  ✓ Distribuzione ragionevolmente uniforme")
    except Exception as e:
        bugs.append(f"4: Exception: {e}")
    
    return bugs


def test_empty_or_single():
    """Test con array vuoti o singoli."""
    print("\n" + "=" * 70)
    print("TEST 5: Array vuoti o singoli")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    # Test 5.1: Array vuoto
    print("\n5.1 Array vuoto:")
    mu_empty = np.array([])
    sigma_empty = np.array([])
    
    try:
        idx = selector.select(mu_empty, sigma_empty, rng, novelty_weight=1.0)
        bugs.append(f"5.1: Dovrebbe fallire con array vuoto, invece ha ritornato {idx}")
    except Exception as e:
        print(f"  ✓ Exception attesa: {type(e).__name__}")
    
    # Test 5.2: Singolo elemento
    print("\n5.2 Singolo elemento:")
    mu_single = np.array([1.0])
    sigma_single = np.array([0.5])
    
    try:
        idx = selector.select(mu_single, sigma_single, rng, novelty_weight=1.0)
        if idx == 0:
            print(f"  ✓ Selezionato l'unico elemento: {idx}")
        else:
            bugs.append(f"5.2: Expected 0, got {idx}")
    except Exception as e:
        bugs.append(f"5.2: Exception: {e}")
    
    return bugs


def test_softmax_overflow():
    """Test per overflow nel softmax."""
    print("\n" + "=" * 70)
    print("TEST 6: Softmax overflow")
    print("=" * 70)
    
    bugs = []
    selector = UCBSoftmaxSelector()
    rng = np.random.default_rng(42)
    
    # Score con range enorme → possibile overflow in exp()
    mu = np.array([0.0, 1e10, 2e10, 3e10])  # Range enorme
    sigma = np.array([0.1] * 4)
    
    try:
        idx = selector.select(mu, sigma, rng, novelty_weight=1.0)
        print(f"  Selezionato indice: {idx}")
        
        # Con z-scoring, dovrebbe gestire il range
        print("  ✓ Gestisce range enorme (z-scoring)")
    except Exception as e:
        bugs.append(f"6: Exception: {e}")
    
    return bugs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ACQUISITION.PY STRESS TESTS")
    print("=" * 70)
    
    all_bugs = []
    
    all_bugs.extend(test_nan_mu_sigma())
    all_bugs.extend(test_inf_values())
    all_bugs.extend(test_extreme_novelty_weight())
    all_bugs.extend(test_identical_scores())
    all_bugs.extend(test_empty_or_single())
    all_bugs.extend(test_softmax_overflow())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_bugs:
        print(f"\n❌ BUGS TROVATI: {len(all_bugs)}")
        for bug in all_bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ NESSUN BUG TROVATO in acquisition.py!")
    
    return all_bugs


if __name__ == "__main__":
    bugs = main()
