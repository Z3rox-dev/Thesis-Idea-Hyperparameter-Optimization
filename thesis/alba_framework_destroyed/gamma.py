"""ALBA Framework - Gamma Scheduler

Gamma ($\\gamma$) is the dynamic threshold used to label points as "good".

This module defines an interchangeable interface for gamma scheduling.
The default implementation matches the ALBA_V1 behavior (quantile annealing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class GammaScheduler(Protocol):
    """Strategy interface for gamma scheduling."""

    def compute(
        self,
        y_all: Sequence[float],
        iteration: int,
        exploration_budget: int,
    ) -> float:
        """Compute current gamma threshold."""


@dataclass(frozen=True)
class QuantileAnnealedGammaScheduler:
    """Default gamma schedule (matches ALBA_V1).

    - If fewer than 10 observations, gamma = 0.0
    - Otherwise gamma is the (1 - q) percentile of y_all
    - q anneals linearly from gamma_quantile_start to gamma_quantile
      over the first ~50% of the exploration budget.
    """

    gamma_quantile: float = 0.20
    gamma_quantile_start: float = 0.15

    def compute(
        self,
        y_all: Sequence[float],
        iteration: int,
        exploration_budget: int,
    ) -> float:
        if len(y_all) < 10:
            return 0.0

        progress = min(1.0, iteration / max(1, exploration_budget * 0.5))
        current_quantile = self.gamma_quantile_start - progress * (
            self.gamma_quantile_start - self.gamma_quantile
        )
        return float(np.percentile(np.asarray(y_all, dtype=float), 100 * (1 - current_quantile)))
