#!/usr/bin/env python3
"""
BUG ANALYSIS: NaN in Potentials
================================

Problema trovato: quando good_ratio() ritorna NaN, i potentials diventano tutti NaN.

Root cause: manca validazione/sanitizzazione dei good_ratio values.
"""

import sys
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
import os
os.chdir(parent_dir)


class MockCube:
    def __init__(self, bounds, lgs_model=None, good_ratio_val=0.5):
        self.bounds = bounds
        self.lgs_model = lgs_model
        self._good_ratio = good_ratio_val
    
    def center(self):
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])
    
    def widths(self):
        return np.array([hi - lo for lo, hi in self.bounds])
    
    def good_ratio(self):
        return self._good_ratio


from coherence import compute_coherence_scores


def test_nan_good_ratio():
    """Test che il NaN nel good_ratio causa potentials NaN."""
    
    print("=" * 70)
    print("BUG: NaN in good_ratio → NaN in potentials")
    print("=" * 70)
    
    grad = np.array([1.0, 0.0])
    
    # Test cases
    test_cases = [
        ("Tutti validi", [0.5, 0.6, 0.7, 0.4, 0.5]),
        ("Un NaN", [0.5, 0.6, float('nan'), 0.4, 0.5]),
        ("Tutti NaN", [float('nan')] * 5),
        ("Inf positivo", [0.5, 0.6, float('inf'), 0.4, 0.5]),
        ("Inf negativo", [0.5, 0.6, float('-inf'), 0.4, 0.5]),
        ("Fuori bounds (>1)", [0.5, 0.6, 1.5, 0.4, 0.5]),
        ("Fuori bounds (<0)", [0.5, 0.6, -0.3, 0.4, 0.5]),
    ]
    
    for name, ratios in test_cases:
        print(f"\n{name}:")
        print(f"  good_ratios: {ratios}")
        
        leaves = []
        for i, ratio in enumerate(ratios):
            leaves.append(MockCube(
                [(i*0.2, (i+1)*0.2), (0, 1)],
                lgs_model={"grad": grad},
                good_ratio_val=ratio
            ))
        
        scores, potentials, coh, _, _ = compute_coherence_scores(leaves)
        
        pot_values = list(potentials.values())
        has_nan = any(np.isnan(v) for v in pot_values)
        has_inf = any(np.isinf(v) for v in pot_values)
        has_oob = any(v < 0 or v > 1 for v in pot_values if not np.isnan(v) and not np.isinf(v))
        
        status = "✓ OK" if not (has_nan or has_inf or has_oob) else "❌ BUG"
        
        print(f"  potentials: {pot_values}")
        print(f"  global_coherence: {coh}")
        print(f"  {status}")
        
        if has_nan:
            print("    → Contiene NaN!")
        if has_inf:
            print("    → Contiene Inf!")
        if has_oob:
            print("    → Contiene valori fuori [0,1]!")


def propose_fix():
    """Propone una fix per il bug."""
    
    print("\n" + "=" * 70)
    print("PROPOSTA FIX")
    print("=" * 70)
    
    fix_code = '''
# Alla linea 379 in coherence.py, dopo:
#   leaf_good_ratios = np.array([leaves[i].good_ratio() for i in range(n)])
# Aggiungere:

# Sanitize good_ratios: replace NaN/Inf with median, clip to [0, 1]
leaf_good_ratios = np.array([leaves[i].good_ratio() for i in range(n)])

# Handle NaN and Inf
valid_mask = np.isfinite(leaf_good_ratios)
if valid_mask.any():
    median_ratio = np.median(leaf_good_ratios[valid_mask])
    leaf_good_ratios = np.where(valid_mask, leaf_good_ratios, median_ratio)
else:
    # All are NaN/Inf - use default 0.5
    leaf_good_ratios = np.full(n, 0.5)

# Clip to valid range [0, 1]
leaf_good_ratios = np.clip(leaf_good_ratios, 0.0, 1.0)
'''
    print(fix_code)


if __name__ == "__main__":
    test_nan_good_ratio()
    propose_fix()
