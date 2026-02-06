#!/usr/bin/env python3
"""
BUG ANALYSIS: NaN in Scores → NaN in Gradient
==============================================

Bug trovato: quando uno score è NaN, il gradiente diventa NaN.
"""

import numpy as np


def simulate_bug():
    """Simula il bug nel fit_lgs_model."""
    
    dim = 2
    
    # Dati con un NaN
    pts = [
        (np.array([0.1, 0.1]), 1.0),
        (np.array([0.2, 0.2]), float('nan')),  # NaN!
        (np.array([0.3, 0.3]), 2.0),
        (np.array([0.4, 0.4]), 3.0),
        (np.array([0.5, 0.5]), 4.0),
    ]
    
    all_pts = np.array([p for p, s in pts])
    all_scores = np.array([s for p, s in pts])
    
    print("=" * 70)
    print("BUG ANALYSIS: NaN in Scores")
    print("=" * 70)
    
    print(f"\nScores: {all_scores}")
    print(f"Contains NaN: {np.any(np.isnan(all_scores))}")
    
    # Step 1: Normalizzazione y
    y_mean = all_scores.mean()  # Diventa NaN!
    y_std = all_scores.std() + 1e-6  # Diventa NaN!
    y_centered = (all_scores - y_mean) / y_std  # Tutto NaN!
    
    print(f"\ny_mean: {y_mean}")
    print(f"y_std: {y_std}")
    print(f"y_centered: {y_centered}")
    
    # Da qui, tutto è NaN
    print("\n→ y_mean, y_std, y_centered sono tutti NaN!")
    print("→ Il gradiente calcolato sarà NaN!")


def propose_fix():
    print("\n" + "=" * 70)
    print("PROPOSTA FIX")
    print("=" * 70)
    
    fix_code = '''
# In lgs.py, dopo aver estratto all_pts e all_scores:

# BUG FIX: Remove NaN/Inf scores
valid_mask = np.isfinite(all_scores)
if not valid_mask.all():
    # Filter out invalid points
    all_pts = all_pts[valid_mask]
    all_scores = all_scores[valid_mask]
    
    # Re-check if we have enough points
    if len(all_scores) < dim + 2:
        return None
    
    # Update k for top_k
    k = max(3, len(all_scores) // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]
'''
    print(fix_code)


if __name__ == "__main__":
    simulate_bug()
    propose_fix()
