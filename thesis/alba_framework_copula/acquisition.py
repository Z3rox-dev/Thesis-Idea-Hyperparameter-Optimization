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

        beta = float(novelty_weight) * self.beta_multiplier
        score = mu + beta * sigma

        if score.std() > 1e-9:
            score_z = (score - score.mean()) / score.std()
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * self.softmax_temperature)
        probs = probs / probs.sum()
        return int(rng.choice(len(score), p=probs))
