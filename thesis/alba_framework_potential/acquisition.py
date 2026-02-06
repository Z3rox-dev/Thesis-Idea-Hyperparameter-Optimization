"""ALBA Framework - Acquisition / Selection

This module defines how ALBA selects one candidate among many using the
surrogate mean (mu) and uncertainty (sigma).

Default implementation matches ALBA_V1:
- UCB score = mu + beta * sigma, where beta = novelty_weight * 2
- softmax over z-scored UCB with temperature 3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AcquisitionSelector(Protocol):
    def select(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        rng: np.random.Generator,
        novelty_weight: float,
    ) -> int:
        ...


@dataclass(frozen=True)
class UCBSoftmaxSelector:
    """Default acquisition + selection (matches ALBA_V1)."""

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
        # Candidates with invalid predictions get neutral scores
        valid_mu = np.isfinite(mu)
        valid_sigma = np.isfinite(sigma)
        
        if not valid_mu.all():
            if valid_mu.any():
                mu = np.where(valid_mu, mu, np.median(mu[valid_mu]))
            else:
                mu = np.zeros_like(mu)  # All NaN → will be uniform
        
        if not valid_sigma.all():
            if valid_sigma.any():
                sigma = np.where(valid_sigma, sigma, np.median(sigma[valid_sigma]))
            else:
                sigma = np.ones_like(sigma)  # All NaN → equal uncertainty

        beta = float(novelty_weight) * self.beta_multiplier
        # Handle NaN/Inf novelty_weight
        if not np.isfinite(beta):
            beta = 0.0  # Fall back to pure exploitation
        
        score = mu + beta * sigma

        score_std = score.std()
        if np.isfinite(score_std) and score_std > 1e-9:
            score_z = (score - score.mean()) / score_std
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * self.softmax_temperature)
        probs = probs / probs.sum()
        return int(rng.choice(len(score), p=probs))
