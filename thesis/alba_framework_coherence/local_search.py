"""ALBA Framework - Local Search

This module defines how ALBA samples around the incumbent best solution
in the local-search phase.

Default implementation matches ALBA_V1: Gaussian perturbation with a
radius that shrinks as progress increases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Optional, Tuple

import numpy as np


class LocalSearchSampler(Protocol):
    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussianLocalSearchSampler:
    """Default local-search sampler (matches ALBA_V1)."""

    radius_start: float = 0.15
    radius_end: float = 0.03

    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        progress = float(np.clip(progress, 0.0, 1.0))
        radius = self.radius_start * (1 - progress) + self.radius_end
        radius = max(radius, 1e-6)  # Ensure non-negative
        noise = rng.normal(0, radius, len(bounds)) * global_widths
        x = best_x + noise

        return np.array([np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(bounds))], dtype=float)
