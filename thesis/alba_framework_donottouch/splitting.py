"""ALBA Framework - Splitting Policies

This module defines two independent aspects of splitting:
1) When to split a cube (split decision)
2) How to split a cube (split execution)

Default policies include ALBA_V1-compatible and adaptive variants.
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
class AdaptiveSplitDecider:
    """Adaptive split decision using volume, good ratio, and model quality."""

    split_trials_min: int = 15
    split_depth_max: int = 16
    split_trials_factor: float = 3.0
    split_trials_offset: int = 6
    min_volume: float = 1e-8
    min_good_ratio: float = 0.02
    min_good_points: int = 1
    model_quality_min: float = 0.15
    fallback_multiplier: float = 2.5

    def should_split(self, cube: Cube, dim: int) -> bool:
        if cube.n_trials < self.split_trials_min:
            return False
        if cube.depth >= self.split_depth_max:
            return False
        if cube.volume() <= self.min_volume:
            return False

        threshold = self.split_trials_factor * dim + self.split_trials_offset
        if cube.n_trials < threshold:
            return False

        if cube.n_good >= self.min_good_points and cube.good_ratio() >= self.min_good_ratio:
            return True

        model = cube.lgs_model or {}
        quality = float(model.get("quality", 0.0))
        n_pts = len(model.get("all_pts", []))
        if quality >= self.model_quality_min and n_pts >= dim + 2:
            return True

        return cube.n_trials >= threshold * self.fallback_multiplier


@dataclass(frozen=True)
class CubeIntrinsicSplitPolicy:
    """Default split execution: delegates to Cube.split() (matches ALBA_V1)."""

    def split(self, cube: Cube, gamma: float, dim: int, rng: np.random.Generator) -> List[Cube]:
        return cube.split(gamma, dim, rng)
