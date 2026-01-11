"""
ALBA Framework - Gravity-Aware Leaf Selection

Extends UCBSoftmaxLeafSelector with cube-level gravitational bias.
This integrates the "ChatGPT per-cube gravity" idea into leaf selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .cube import Cube

if TYPE_CHECKING:
    from .cube_gravity import CubeGravity


@dataclass
class GravityLeafSelector:
    """
    Leaf selector with gravitational field integration.
    
    Combines standard UCB-like score with:
    - Potential-based attraction (lower loss cubes attract)
    - Visit-based repulsion (avoid over-explored cubes)
    """
    
    # Standard UCB parameters
    base_exploration: float = 0.3
    stagnation_exploration_multiplier: float = 2.0
    model_bonus: float = 0.1
    model_bonus_min_points_offset: int = 2
    temperature_normal: float = 3.0
    temperature_stagnating: float = 1.5
    
    # Gravity parameters
    gravity_weight: float = 0.2  # Weight for gravity component
    
    # Reference to gravity system (set externally)
    gravity: Optional['CubeGravity'] = field(default=None, repr=False)
    
    def select(
        self, 
        leaves: List[Cube], 
        dim: int, 
        stagnating: bool, 
        rng: np.random.Generator
    ) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        scores = []
        gravity_scores = []
        
        for c in leaves:
            # Standard UCB score
            ratio = c.good_ratio()
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            base_score = ratio + exploration + model_bonus
            scores.append(base_score)
            
            # Gravity score
            if self.gravity is not None:
                g_score = self.gravity.get_gravity_score(c, leaves)
                gravity_scores.append(g_score)
            else:
                gravity_scores.append(0.0)
        
        # Combine scores
        scores_arr = np.asarray(scores, dtype=float)
        gravity_arr = np.asarray(gravity_scores, dtype=float)
        
        # Normalize gravity scores to similar scale as base scores
        if gravity_arr.std() > 0:
            gravity_arr = (gravity_arr - gravity_arr.mean()) / gravity_arr.std()
            gravity_arr = gravity_arr * scores_arr.std() if scores_arr.std() > 0 else gravity_arr
        
        # Combined score
        combined = scores_arr + self.gravity_weight * gravity_arr
        combined = combined - combined.max()
        
        # Softmax selection
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(combined * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]
