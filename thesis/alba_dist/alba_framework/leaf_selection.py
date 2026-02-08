"""ALBA Framework - Leaf Selection

This module defines how ALBA selects a leaf Cube to sample from during
exploration/local-search tree-guided steps.

Default implementation matches ALBA_V1: a softmax over a UCB-like score
based on good_ratio, exploration bonus, and a small model bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Any

import numpy as np

from .cube import Cube


class LeafSelector(Protocol):
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        ...


@dataclass
class UCBSoftmaxLeafSelector:
    """Default leaf selection policy (matches ALBA_V1)."""

    base_exploration: float = 0.3
    stagnation_exploration_multiplier: float = 2.0
    model_bonus: float = 0.1
    model_bonus_min_points_offset: int = 2
    temperature_normal: float = 3.0
    temperature_stagnating: float = 1.5

    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")

        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            # FIX Finding 30: Handle NaN in good_ratio
            if not np.isfinite(ratio):
                ratio = 0.5  # Neutral default
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        
        # FIX Finding 30: Handle all NaN scores
        if not np.any(np.isfinite(scores_arr)):
            # All scores are NaN/Inf, select uniformly
            return leaves[rng.integers(len(leaves))]
        
        # Replace any remaining NaN with min score
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        
        scores_arr = scores_arr - scores_arr.max()

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]


@dataclass
class ThompsonSamplingLeafSelector(UCBSoftmaxLeafSelector):
    """
    Thompson Sampling leaf selector.
    
    Instead of using deterministic good_ratio = (n_good + 1) / (n_trials + 2),
    samples from Beta(n_good + 1, n_bad + 1) posterior.
    
    This naturally balances exploration/exploitation:
    - Leaves with few samples have high posterior variance → more exploration
    - Leaves with many samples have tight posteriors → exploitation
    
    Empirically won against Base on:
    - Synthetic (10-3 head-to-head)
    - ParamNet (3-1)  
    - YAHPO XGBoost (tie 2-2)
    """
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> Cube:
        """Select leaf using Thompson Sampling from Beta posterior."""
        if len(leaves) == 1:
            return leaves[0]

        scores = []
        for c in leaves:
            # Thompson Sampling: sample from Beta posterior
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            sample = rng.beta(alpha, beta_param)
            
            # Handle edge cases
            if not np.isfinite(sample):
                sample = 0.5

            # Keep exploration bonus (Thompson already explores naturally, but helps early)
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            # Model bonus for leaves with good local models
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            scores.append(sample + exploration + model_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        
        # Handle all NaN scores
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        
        # Softmax selection with temperature
        scores_arr = scores_arr - scores_arr.max()

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]


@dataclass
class PotentialAwareLeafSelector(UCBSoftmaxLeafSelector):
    """
    Leaf selector that incorporates Global Potential Field from CoherenceTracker.
    
    Potential u in [0, 1] (lower is better).
    Bonus = weight * (1.0 - u).
    """
    
    potential_weight: float = 0.5
    tracker: Any = None  # Injected CoherenceTracker

    def set_tracker(self, tracker: Any):
        self.tracker = tracker

    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")

        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            # FIX Finding 30: Handle NaN in good_ratio
            if not np.isfinite(ratio):
                ratio = 0.5
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            potential_bonus = 0.0
            if self.tracker is not None:
                # Potential is in [0, 1], lower is better. Convert to bonus.
                u = self.tracker.get_potential(c, leaves)
                # FIX Finding 30: Handle NaN potential
                if not np.isfinite(u):
                    u = 0.5
                
                # STABILITY FIX: Scale potential weight by global coherence
                # When global_coherence is low (~0.5), the potential field is unreliable
                # and should not strongly influence leaf selection.
                # When global_coherence is high (~0.8+), trust the potential more.
                global_coh = self.tracker.global_coherence
                # Only use potential when coherence is significantly above baseline (0.5)
                # Scale: coherence 0.5 -> weight 0, coherence 0.8 -> weight 1.0
                coherence_scale = max(0.0, min(1.0, (global_coh - 0.5) * 3.33))
                effective_weight = self.potential_weight * coherence_scale
                
                potential_bonus = effective_weight * (1.0 - u)

            scores.append(ratio + exploration + model_bonus + potential_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        
        # FIX Finding 30: Handle all NaN scores
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        
        scores_arr = scores_arr - scores_arr.max()

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]
