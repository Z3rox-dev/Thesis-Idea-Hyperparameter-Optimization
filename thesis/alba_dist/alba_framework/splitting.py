"""ALBA Framework - Splitting Policies

This module defines two independent aspects of splitting:
1) When to split a cube (split decision)
2) How to split a cube (split execution)

Default behavior matches ALBA_V1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from .cube import Cube


class SplitDecider(Protocol):
    def should_split(self, cube: Cube, dim: int) -> bool:
        ...


class SplitPolicy(Protocol):
    def split(self, cube: Cube, gamma: float, dim: int, rng: np.random.Generator) -> List[Cube]:
        ...


@dataclass(frozen=True)
class ThresholdSplitDecider:
    """Default split decision (matches ALBA_V1)."""

    split_trials_min: int = 15
    split_depth_max: int = 16
    split_trials_factor: float = 3.0
    split_trials_offset: int = 6

    def should_split(self, cube: Cube, dim: int) -> bool:
        if cube.n_trials < self.split_trials_min:
            return False
        if cube.depth >= self.split_depth_max:
            return False
        return cube.n_trials >= self.split_trials_factor * dim + self.split_trials_offset


@dataclass(frozen=True)
class CubeIntrinsicSplitPolicy:
    """Default split execution: delegates to Cube.split() (matches ALBA_V1)."""

    def split(self, cube: Cube, gamma: float, dim: int, rng: np.random.Generator) -> List[Cube]:
        return cube.split(gamma, dim, rng)
