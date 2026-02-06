"""
ALBA Framework - Optimizer Module

This module implements the main ALBA optimizer class, orchestrating:
- Adaptive cube partitioning of the search space
- Local Gradient Surrogate (LGS) models for each region
- UCB-style acquisition with exploration bonuses
- Two-phase optimization: exploration + local search refinement
- Categorical handling with curiosity-driven sampling

ALBA (Adaptive Local Bayesian Algorithm) is designed for efficient
hyperparameter optimization in mixed continuous-categorical spaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .acquisition import AcquisitionSelector, UCBSoftmaxSelector
from .candidates import CandidateGenerator, MixtureCandidateGenerator
from .cube import Cube
from .cube_gravity import CubeGravity
from .free_geometry import FreeGeometryEstimator
from .categorical import CategoricalSampler
from .gamma import GammaScheduler, QuantileAnnealedGammaScheduler
from .leaf_selection import LeafSelector, UCBSoftmaxLeafSelector
from .local_search import GaussianLocalSearchSampler, LLRGradientLocalSearchSampler, LocalSearchSampler
from .param_space import ParamSpaceHandler
from .splitting import (
    CubeIntrinsicSplitPolicy,
    SplitDecider,
    SplitPolicy,
    ThresholdSplitDecider,
)

warnings.filterwarnings("ignore")


@dataclass
class _CatKeyStats:
    n: int = 0
    ema_y: float = 0.0
    best_y: float = float("-inf")


class ALBA:
    """
    ALBA (Adaptive Local Bayesian Allocator) for Hyperparameter Optimization.

    ALBA combines adaptive space partitioning with local surrogate models to
    efficiently optimize both continuous and categorical hyperparameters.

    Parameters
    ----------
    bounds : Optional[List[Tuple[float, float]]]
        Search space bounds for each dimension. Required if param_space is not provided.
    param_space : Optional[Dict[str, Any]]
        Typed parameter space specification. If provided, ALBA handles
        encoding/decoding automatically.
    param_order : Optional[List[str]]
        Optional explicit ordering of parameters when using param_space.
    maximize : bool
        Whether to maximize (True) or minimize (False) the objective.
    seed : int
        Random seed for reproducibility.
    gamma_quantile : float
        Final quantile threshold for "good" points.
    gamma_quantile_start : float
        Initial quantile threshold (more selective at start).
    local_search_ratio : float
        Fraction of budget for local search phase.
    n_candidates : int
        Number of candidates to generate per iteration.
    split_trials_min : int
        Minimum trials before cube can split.
    split_depth_max : int
        Maximum tree depth.
    split_trials_factor : float
        Factor for split threshold: factor * dim + offset.
    split_trials_offset : int
        Offset for split threshold.
    novelty_weight : float
        Weight for exploration in UCB acquisition.
    total_budget : int
        Total optimization budget.
    global_random_prob : float
        Probability of pure random sampling for diversity.
    stagnation_threshold : int
        Iterations without improvement before increasing exploration.
    categorical_dims : Optional[List[Tuple[int, int]]]
        List of (dim_idx, n_choices) for categorical dimensions.
        Only needed when using bounds directly.

    Examples
    --------
    Using param_space (recommended):

    >>> param_space = {
    ...     'learning_rate': (1e-4, 1e-1, 'log'),
    ...     'hidden_size': (32, 512, 'int'),
    ...     'activation': ['relu', 'tanh', 'gelu'],
    ... }
    >>> opt = ALBA(param_space=param_space, seed=42)
    >>> for _ in range(100):
    ...     config = opt.ask()  # Returns dict
    ...     score = evaluate(config)
    ...     opt.tell(config, score)

    Using bounds directly:

    >>> bounds = [(0, 1), (0, 1), (0, 1)]
    >>> categorical_dims = [(2, 3)]  # Third dim has 3 categories
    >>> opt = ALBA(bounds=bounds, categorical_dims=categorical_dims)
    >>> for _ in range(100):
    ...     x = opt.ask()  # Returns np.ndarray
    ...     score = evaluate(x)
    ...     opt.tell(x, score)
    """

    def __init__(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        param_space: Optional[Dict[str, Any]] = None,
        param_order: Optional[List[str]] = None,
        maximize: bool = False,
        seed: int = 42,
        # Strategy components (optional). If not provided, defaults match ALBA_V1.
        gamma_scheduler: Optional[GammaScheduler] = None,
        leaf_selector: Optional[LeafSelector] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        acquisition_selector: Optional[AcquisitionSelector] = None,
        split_decider: Optional[SplitDecider] = None,
        split_policy: Optional[SplitPolicy] = None,
        local_search_sampler: Optional[LocalSearchSampler] = None,
        gamma_quantile: float = 0.20,
        gamma_quantile_start: float = 0.15,
        local_search_ratio: float = 0.30,
        n_candidates: int = 25,
        split_trials_min: int = 15,
        split_depth_max: int = 16,
        split_trials_factor: float = 3.0,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        global_random_prob: float = 0.05,
        stagnation_threshold: int = 50,
        categorical_dims: Optional[List[Tuple[int, int]]] = None,
        cube_gravity: bool = False,
        cube_gravity_drift: float = 0.3,
        cube_gravity_mode: str = "potential",
        geo_drift: bool = False,
        llr_gradient: bool = False,
        llr_gradient_weight: float = 0.7,
    ) -> None:
        # Initialize param space handler if using param_space mode
        self._param_space_handler: Optional[ParamSpaceHandler] = None
        self._param_space_mode = param_space is not None

        if self._param_space_mode:
            self._param_space_handler = ParamSpaceHandler(param_space, param_order)
            bounds = self._param_space_handler.get_bounds()
            categorical_dims = self._param_space_handler.categorical_dims

        if bounds is None:
            raise TypeError("ALBA requires either bounds=... or param_space=...")

        # Core attributes
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)

        # Gamma (good threshold) parameters
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = gamma_quantile_start
        self.gamma = 0.0

        # Budget allocation
        self.total_budget = total_budget
        self.local_search_ratio = local_search_ratio
        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        # Sampling parameters
        self.n_candidates = n_candidates
        self._novelty_weight = novelty_weight
        self._global_random_prob = global_random_prob

        # Split parameters
        self._split_trials_min = split_trials_min
        self._split_depth_max = split_depth_max
        self._split_trials_factor = split_trials_factor
        self._split_trials_offset = split_trials_offset

        # Stagnation tracking
        self._stagnation_threshold = stagnation_threshold
        self.stagnation = 0
        self.last_improvement_iter = 0

        # Initialize cube tree
        self.root = Cube(bounds=list(bounds))
        self.leaves: List[Cube] = [self.root]
        self._last_cube: Optional[Cube] = None

        # Observation history
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []  # Always stored as "higher is better"
        self.best_y_internal = -np.inf
        self.best_x: Optional[np.ndarray] = None

        # Global geometry
        self._global_widths = np.array([hi - lo for lo, hi in bounds])

        # Initialize categorical sampler
        self._cat_sampler = CategoricalSampler(
            categorical_dims=categorical_dims or [],
            elite_size=10,
            curiosity_bonus=0.3,
            crossover_rate=0.15,
        )
        try:
            self.root.categorical_dims = tuple(int(i) for i, _ in self._cat_sampler.categorical_dims)
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Categorical "GravJump" state.
        #
        # Idea: treat each categorical combination (cat_key) as a discrete
        # planet with learned quality; we optimize continuous variables inside
        # a chosen key, and only "jump" keys when enough evidence accumulates.
        # -----------------------------------------------------------------
        self._cat_active_key: Optional[Tuple[int, ...]] = None
        self._cat_jump_momentum: float = 0.0
        self._cat_key_stats: Dict[Tuple[int, ...], _CatKeyStats] = {}

        # Strategy components (defaults reproduce the current baseline behavior)
        self._gamma_scheduler: GammaScheduler = (
            gamma_scheduler
            if gamma_scheduler is not None
            else QuantileAnnealedGammaScheduler(
                gamma_quantile=self.gamma_quantile,
                gamma_quantile_start=self.gamma_quantile_start,
            )
        )
        self._leaf_selector: LeafSelector = (
            leaf_selector if leaf_selector is not None else UCBSoftmaxLeafSelector()
        )
        self._candidate_generator: CandidateGenerator = (
            candidate_generator if candidate_generator is not None else MixtureCandidateGenerator()
        )
        self._acquisition_selector: AcquisitionSelector = (
            acquisition_selector if acquisition_selector is not None else UCBSoftmaxSelector()
        )
        self._split_decider: SplitDecider = (
            split_decider
            if split_decider is not None
            else ThresholdSplitDecider(
                split_trials_min=self._split_trials_min,
                split_depth_max=self._split_depth_max,
                split_trials_factor=self._split_trials_factor,
                split_trials_offset=self._split_trials_offset,
            )
        )
        self._split_policy: SplitPolicy = (
            split_policy if split_policy is not None else CubeIntrinsicSplitPolicy()
        )
        
        # Local search sampler: LLR gradient or Gaussian
        if local_search_sampler is not None:
            self._local_search_sampler: LocalSearchSampler = local_search_sampler
        elif llr_gradient:
            self._local_search_sampler = LLRGradientLocalSearchSampler(
                gradient_weight=llr_gradient_weight
            )
        else:
            self._local_search_sampler = GaussianLocalSearchSampler()

        # Cube gravity for physics-based drift
        self._use_cube_gravity = cube_gravity
        self._cube_gravity_drift = cube_gravity_drift
        self._cube_gravity: Optional[CubeGravity] = None
        if self._use_cube_gravity:
            self._cube_gravity = CubeGravity(drift_mode=cube_gravity_mode)

        # Free geometry estimator for drift modulation
        self._use_geo_drift = geo_drift
        self._free_geometry: Optional[FreeGeometryEstimator] = None
        if self._use_geo_drift:
            self._free_geometry = FreeGeometryEstimator(n_dims=self.dim)

        # Gravity velocity state per-leaf (used only when cube_gravity=True).
        # This is an additive offset used in local search, with temporal momentum.
        self._gravity_velocity: Dict[int, np.ndarray] = {}

        # Iteration counter
        self.iteration = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def best_y(self) -> float:
        """Best objective value found (in original scale)."""
        return self._to_raw(self.best_y_internal)

    @property
    def n_observations(self) -> int:
        """Number of observations so far."""
        return len(self.X_all)

    @property
    def is_stagnating(self) -> bool:
        """Check if optimization is currently stagnating."""
        return self.stagnation > self._stagnation_threshold

    # -------------------------------------------------------------------------
    # Internal score conversion
    # -------------------------------------------------------------------------

    def _to_internal(self, y_raw: float) -> float:
        """Convert raw objective value to internal score (higher is better)."""
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        """Convert internal score back to raw objective value."""
        return y_internal if self.maximize else -y_internal

    # -------------------------------------------------------------------------
    # Ask interface
    # -------------------------------------------------------------------------

    def ask(self) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Suggest the next point to evaluate.

        Returns
        -------
        Union[np.ndarray, Dict[str, Any]]
            - If constructed with bounds=..., returns a normalized np.ndarray.
            - If constructed with param_space=..., returns a dict config.
        """
        x = self.ask_array()
        if self._param_space_mode:
            return self.decode(x)
        return x

    def ask_array(self) -> np.ndarray:
        """
        Suggest next point in the internal normalized space ([0,1]^d).

        Returns
        -------
        np.ndarray
            Next point to evaluate in normalized coordinates.
        """
        self.iteration = len(self.X_all)

        # Global random for diversity
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self._last_cube = self._find_containing_leaf(x)
            return x

        if self.iteration < self.exploration_budget:
            # Exploration phase
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self._last_cube = self._select_leaf()
            return self._sample_in_cube(self._last_cube)

        # Local search phase with tree integration
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress

        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self._last_cube = self._find_containing_leaf(x)
            # Apply categorical sampling using Thompson Sampling + Elite Crossover
            if self._last_cube is not None and self._cat_sampler.has_categoricals:
                x = self._cat_sampler.sample(x, self._last_cube, self.rng, self.is_stagnating)
                x = self._clip_to_cube(x, self._last_cube)
        else:
            self._update_gamma()
            self._recount_good()
            self._last_cube = self._select_leaf()
            x = self._sample_in_cube(self._last_cube)

        return x

    # -------------------------------------------------------------------------
    # Tell interface
    # -------------------------------------------------------------------------

    def tell(self, x: Union[np.ndarray, Dict[str, Any]], y_raw: float) -> None:
        """
        Report the objective value for a point.

        Parameters
        ----------
        x : Union[np.ndarray, Dict[str, Any]]
            The evaluated point. Can be dict (if using param_space) or array.
        y_raw : float
            The objective value (in original scale).
        """
        if self._param_space_mode and isinstance(x, dict):
            x = self.encode(x)
        x = np.asarray(x, dtype=float)
        y = self._to_internal(y_raw)

        # Update best
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
            if self._cube_gravity is not None:
                try:
                    leaf = self._find_containing_leaf(self.best_x)
                    leaf_id = int(id(leaf))
                    self._gravity_velocity[leaf_id] = np.zeros(self.dim, dtype=float)
                except Exception:
                    pass
        else:
            self.stagnation += 1

        # Record observation
        self.X_all.append(x.copy())
        self.y_all.append(y)

        # Update local search sampler history (for LLR gradient estimation)
        if hasattr(self._local_search_sampler, 'update_history'):
            self._local_search_sampler.update_history(x, y)

        # Update categorical tracking
        self._cat_sampler.record_observation(x, y)
        self._update_cat_key_stats(x, y)

        # Update free geometry estimator if enabled
        if self._free_geometry is not None:
            # Normalize x to [0,1]^d
            x_norm = (x - np.array([b[0] for b in self.bounds])) / (
                np.array([b[1] - b[0] for b in self.bounds])
            )
            self._free_geometry.update(x_norm, y_raw)

        # Update cube
        if self._last_cube is not None:
            cube = self._last_cube
            cube.add_observation(x, y, self.gamma)
            cube.update_warp(x, y)

            # Update categorical stats in cube
            for dim_idx, n_choices in self._cat_sampler.categorical_dims:
                val_idx = self._cat_sampler.discretize(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                is_good = y >= self.gamma
                cube.cat_stats[dim_idx][val_idx] = (
                    n_g + (1 if is_good else 0),
                    n_t + 1,
                )

            cube.fit_lgs_model(self.gamma, self.dim, self.rng)

            # Update cube gravity if enabled
            if self._cube_gravity is not None:
                self._cube_gravity.update_cube(cube, y_raw)

            # Check for split
            if self._should_split(cube):
                children = self._split_policy.split(cube, self.gamma, self.dim, self.rng)
                for child in children:
                    self._cat_sampler.recompute_cube_cat_stats(child, self.gamma)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)

            self._last_cube = None

    # -------------------------------------------------------------------------
    # Encode / Decode (param_space mode)
    # -------------------------------------------------------------------------

    def encode(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Convert typed config dict to internal normalized vector.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        np.ndarray
            Normalized vector in [0, 1]^dim.

        Raises
        ------
        RuntimeError
            If not in param_space mode.
        """
        if self._param_space_handler is None:
            raise RuntimeError("encode() requires param_space mode")
        return self._param_space_handler.encode(config)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Convert internal normalized vector to typed config dict.

        Parameters
        ----------
        x : np.ndarray
            Normalized vector in [0, 1]^dim.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        RuntimeError
            If not in param_space mode.
        """
        if self._param_space_handler is None:
            raise RuntimeError("decode() requires param_space mode")
        return self._param_space_handler.decode(x)

    # -------------------------------------------------------------------------
    # Gamma (threshold) management
    # -------------------------------------------------------------------------

    def _update_gamma(self) -> None:
        """Update the gamma threshold based on current observations."""
        self.gamma = float(
            self._gamma_scheduler.compute(self.y_all, self.iteration, self.exploration_budget)
        )

    def _recount_good(self) -> None:
        """Recount good points in all leaves with current gamma."""
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf.tested_pairs if s >= self.gamma)
            self._cat_sampler.recompute_cube_cat_stats(leaf, self.gamma)

    # -------------------------------------------------------------------------
    # Leaf selection
    # -------------------------------------------------------------------------

    def _select_leaf(self) -> Cube:
        """
        Select a leaf cube for the next sample using UCB-like scoring.

        Returns
        -------
        Cube
            Selected leaf cube.
        """
        if not self.leaves:
            return self.root
        return self._leaf_selector.select(self.leaves, self.dim, self.is_stagnating, self.rng)

    # -------------------------------------------------------------------------
    # Categorical "GravJump" (discrete gravity) helpers
    # -------------------------------------------------------------------------

    def _cat_global_baseline(self) -> Tuple[float, float]:
        """Return (mean, std) of internal scores for scale-safe key comparisons."""
        if len(self.y_all) >= 2:
            y = np.asarray(self.y_all, dtype=float)
            mu = float(np.mean(y))
            std = float(np.std(y))
            span = float(np.max(y) - np.min(y))
            # Robust floor to avoid tiny scales early (prevents runaway z-scores).
            sigma = float(max(std, 0.25 * span, 1e-3))
            return mu, sigma
        if len(self.y_all) == 1:
            return float(self.y_all[0]), 1.0
        return 0.0, 1.0

    def _cat_allowed_values_in_cube(self, cube: Cube) -> List[List[int]]:
        """Allowed categorical indices per categorical dim (respecting cube bounds when possible)."""
        allowed: List[List[int]] = []
        for dim_idx, n_ch in self._cat_sampler.categorical_dims:
            dim_idx_i = int(dim_idx)
            n_ch_i = int(n_ch)
            lo, hi = cube.bounds[dim_idx_i]
            vals: List[int] = []
            for i in range(n_ch_i):
                v = float(self._cat_sampler.to_continuous(i, n_ch_i))
                if (float(lo) - 1e-12) <= v <= (float(hi) + 1e-12):
                    vals.append(int(i))
            if not vals:
                vals = list(range(n_ch_i))
            allowed.append(vals)
        return allowed

    def _cat_sanitize_key_for_cube(self, cat_key: Tuple[int, ...], cube: Cube) -> Tuple[int, ...]:
        """Snap cat_key components to values allowed in this cube (closest in encoded value)."""
        allowed = self._cat_allowed_values_in_cube(cube)
        if not allowed:
            return tuple()

        dims = self._cat_sampler.categorical_dims
        out: List[int] = []
        for i, vals in enumerate(allowed):
            dim_idx, n_ch = dims[i]
            n_ch_i = int(n_ch)
            cur = int(cat_key[i]) if i < len(cat_key) else int(vals[0])
            if cur in vals:
                out.append(cur)
                continue
            cur_v = float(self._cat_sampler.to_continuous(cur, n_ch_i))
            best = min(vals, key=lambda j: abs(float(self._cat_sampler.to_continuous(int(j), n_ch_i)) - cur_v))
            out.append(int(best))
        return tuple(out)

    def _cat_force_neighbor(
        self,
        current_key: Tuple[int, ...],
        neighbor_key: Tuple[int, ...],
        *,
        y_mean: float,
        y_std: float,
    ) -> float:
        """
        Compute a dimensionless "force" to move from current_key -> neighbor_key.

        - Uses z-scored EMA improvement for scale invariance.
        - Adds a small curiosity term that decays with visits to promote early exploration.
        """
        cur_stats = self._cat_key_stats.get(current_key)
        nbr_stats = self._cat_key_stats.get(neighbor_key)

        # For key-level decisions we care more about "does this key have potential?"
        # than the average. Use best_y (optimistic) to avoid punishing a key
        # because a few random continuous samples were bad.
        cur_val = float(y_mean) if cur_stats is None or cur_stats.n <= 0 else float(cur_stats.best_y)
        nbr_val = float(y_mean) if nbr_stats is None or nbr_stats.n <= 0 else float(nbr_stats.best_y)
        nbr_n = 0 if nbr_stats is None else int(nbr_stats.n)

        delta_z = float((nbr_val - cur_val) / max(float(y_std), 1e-12))
        conf = float(nbr_n) / float(nbr_n + 5.0)  # confidence ramp (no external knob)
        curiosity = float(self._cat_sampler.curiosity_bonus) / float(np.sqrt(1.0 + float(nbr_n)))
        return conf * delta_z + curiosity

    def _choose_cat_key_gravjump(self, cube: Cube) -> Optional[Tuple[int, ...]]:
        """Select (and possibly update) the active categorical key for this ask()."""
        if not self._cat_sampler.has_categoricals:
            return None

        allowed = self._cat_allowed_values_in_cube(cube)
        if not allowed:
            return None

        if self._cat_active_key is None or len(self._cat_active_key) != len(allowed):
            self._cat_active_key = tuple(int(self.rng.choice(vals)) for vals in allowed)
            self._cat_jump_momentum = 0.0
            return self._cat_active_key

        current_key = self._cat_sanitize_key_for_cube(self._cat_active_key, cube)
        if current_key != self._cat_active_key:
            self._cat_active_key = current_key
            self._cat_jump_momentum = 0.0

        y_mean, y_std = self._cat_global_baseline()

        # Enumerate Hamming-1 neighbors within this cube.
        best_key = current_key
        best_force = float("-inf")
        current_list = list(current_key)
        for i, vals in enumerate(allowed):
            cur_val = int(current_list[i])
            for v in vals:
                vv = int(v)
                if vv == cur_val:
                    continue
                cand = list(current_list)
                cand[i] = vv
                k2 = tuple(cand)
                f = self._cat_force_neighbor(current_key, k2, y_mean=y_mean, y_std=y_std)
                if f > best_force:
                    best_force = f
                    best_key = k2

        # Stick-slip via momentum + friction: requires sustained positive force to switch keys.
        rho = 0.90
        stagn = float(np.clip(float(self.stagnation) / float(max(1, self._stagnation_threshold)), 0.0, 1.0))
        friction = 0.15 - 0.10 * stagn  # 0.15 when improving, down to 0.05 when stagnating
        progress = float(np.clip(float(self.iteration) / float(max(1, self.total_budget - 1)), 0.0, 1.0))
        threshold = 0.2 + 0.8 * progress  # explore keys early, stick more later

        self._cat_jump_momentum = max(
            0.0,
            float(rho) * float(self._cat_jump_momentum) + float(best_force) - float(friction),
        )
        if best_key != current_key and self._cat_jump_momentum >= threshold:
            self._cat_active_key = best_key
            self._cat_jump_momentum = 0.0

        return self._cat_active_key

    def _update_cat_key_stats(self, x: np.ndarray, y_internal: float) -> None:
        if not self._cat_sampler.has_categoricals:
            return
        try:
            key = self._cat_sampler.get_cat_key(np.asarray(x, dtype=float))
        except Exception:
            return

        st = self._cat_key_stats.get(key)
        if st is None:
            self._cat_key_stats[key] = _CatKeyStats(n=1, ema_y=float(y_internal), best_y=float(y_internal))
            return

        st.n = int(st.n) + 1
        alpha = 0.20
        st.ema_y = float((1.0 - alpha) * float(st.ema_y) + alpha * float(y_internal))
        st.best_y = float(max(float(st.best_y), float(y_internal)))

    # -------------------------------------------------------------------------
    # Sampling within cubes
    # -------------------------------------------------------------------------

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        """
        Sample a point within the given cube.

        Parameters
        ----------
        cube : Cube
            The cube to sample from.

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        if self.iteration < 15:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)
        else:
            x = self._sample_with_lgs(cube)

        # Apply categorical sampling using Thompson Sampling + Elite Crossover (from experimental)
        x = self._cat_sampler.sample(x, cube, self.rng, self.is_stagnating)

        # Re-clip to cube bounds (categorical sampling may have moved outside)
        return self._clip_to_cube(x, cube)

    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        """
        Sample using the LGS model with UCB acquisition.

        Parameters
        ----------
        cube : Cube
            The cube to sample from.

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

        candidates = self._generate_candidates(cube, self.n_candidates)
        mu, sigma = cube.predict_bayesian(candidates)

        idx = self._acquisition_selector.select(mu, sigma, self.rng, self._novelty_weight)
        return candidates[int(idx)]

    def _generate_candidates(self, cube: Cube, n: int) -> List[np.ndarray]:
        """
        Generate candidate points within a cube.

        Uses multiple strategies:
        - Perturbation of top-k points
        - Gradient-guided sampling
        - Center-based sampling
        - Pure random

        Parameters
        ----------
        cube : Cube
            The cube to sample from.
        n : int
            Number of candidates to generate.

        Returns
        -------
        List[np.ndarray]
            List of candidate points.
        """
        return self._candidate_generator.generate(cube, self.dim, self.rng, n)

    def _local_search_sample(self, progress: float) -> np.ndarray:
        """
        Sample near the best point for local search refinement.

        Parameters
        ----------
        progress : float
            Progress through local search phase in [0, 1].

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        x_base = self._local_search_sampler.sample(
            self.best_x,
            list(self.bounds),
            self._global_widths,
            float(progress),
            self.rng,
        )

        w_leaf = None
        leaf_id = None
        if self.best_x is not None and self.leaves:
            try:
                leaf = self._find_containing_leaf(self.best_x)
                w_leaf = leaf.get_warp_multipliers()
                leaf_id = int(id(leaf))
            except Exception:
                w_leaf = None
                leaf_id = None
            if w_leaf is not None and getattr(w_leaf, "shape", None) == (self.dim,):
                step = x_base - self.best_x
                x_base = self.best_x + step / np.asarray(w_leaf, dtype=float)
                x_base = self._clip_to_bounds(x_base)

        # Relative local-search move (LLR/Gaussian) expressed as an offset from best_x.
        eps = np.zeros(self.dim, dtype=float)
        if self.best_x is not None:
            eps = np.asarray(x_base, dtype=float) - np.asarray(self.best_x, dtype=float)
        
        # Apply cube gravity with temporal momentum (velocity integrates acceleration).
        if self._cube_gravity is None or self.best_x is None or leaf_id is None:
            return np.asarray(x_base, dtype=float)

        v = self._gravity_velocity.get(int(leaf_id), np.zeros(self.dim, dtype=float))
        if getattr(v, "shape", None) != (self.dim,):
            v = np.zeros(self.dim, dtype=float)

        # Acceleration at the local-search point (unnormalized, so magnitude can grow near attractors).
        acc = self._cube_gravity.get_drift_vector(np.asarray(x_base, dtype=float), self.leaves, normalize=False)
        if self._free_geometry is not None:
            acc = self._free_geometry.modulate_drift(acc)

        acc = np.asarray(acc, dtype=float)
        acc_norm = float(np.linalg.norm(acc))
        if np.isfinite(acc_norm) and acc_norm > 1e-12:
            acc_dir = acc / acc_norm
            acc_mag = float(min(3.0, np.log1p(acc_norm)))  # compress + cap
        else:
            acc_dir = np.zeros(self.dim, dtype=float)
            acc_mag = 0.0

        radius_start = float(getattr(self._local_search_sampler, "radius_start", 0.15))
        radius_end = float(getattr(self._local_search_sampler, "radius_end", 0.03))
        ls_radius = radius_start * (1.0 - float(progress)) + radius_end

        stagn = float(
            np.clip(float(self.stagnation) / float(max(1, self._stagnation_threshold)), 0.0, 1.0)
        )
        rho = 0.75 + 0.20 * stagn
        scale = self._cube_gravity_drift * (1.0 - 0.5 * float(progress))
        avg_width = float(np.mean(self._global_widths))
        base_step = 0.05 * avg_width

        dv = scale * base_step * acc_mag * acc_dir
        if w_leaf is not None and getattr(w_leaf, "shape", None) == (self.dim,):
            dv = dv / np.asarray(w_leaf, dtype=float)
        dv = dv * (0.25 + 0.75 * stagn)

        v = float(rho) * np.asarray(v, dtype=float) + dv

        # Keep gravity offset within the local-search trust region.
        v_cap = np.asarray(self._global_widths, dtype=float) * float(ls_radius)
        v = np.minimum(np.maximum(v, -v_cap), v_cap)

        self._gravity_velocity[int(leaf_id)] = v

        x = np.asarray(x_base, dtype=float) + v
        return self._clip_to_bounds(x)

    # -------------------------------------------------------------------------
    # Clipping helpers
    # -------------------------------------------------------------------------

    def _clip_to_cube(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        """Clip point to cube bounds."""
        return np.array(
            [np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(self.dim)]
        )

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip point to global bounds."""
        return np.array(
            [np.clip(x[i], self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
        )

    # -------------------------------------------------------------------------
    # Cube tree management
    # -------------------------------------------------------------------------

    def _find_containing_leaf(self, x: np.ndarray) -> Cube:
        """
        Find the leaf cube that contains point x.

        Parameters
        ----------
        x : np.ndarray
            Point to locate.

        Returns
        -------
        Cube
            Containing leaf cube.
        """
        for leaf in self.leaves:
            if leaf.contains(x):
                return leaf
        # Fallback: return leaf with closest center
        min_dist = float("inf")
        closest = self.leaves[0] if self.leaves else self.root
        for leaf in self.leaves:
            dist = np.linalg.norm(x - leaf.center())
            if dist < min_dist:
                min_dist = dist
                closest = leaf
        return closest

    def _should_split(self, cube: Cube) -> bool:
        """Check if a cube should be split."""
        return bool(self._split_decider.should_split(cube, self.dim))

    def _update_all_models(self) -> None:
        """Update LGS models for all leaf cubes."""
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    # -------------------------------------------------------------------------
    # High-level optimization interface
    # -------------------------------------------------------------------------

    def optimize(
        self,
        objective: Callable[[Union[np.ndarray, Dict[str, Any]]], float],
        budget: int = 100,
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], float]:
        """
        Run optimization loop.

        Parameters
        ----------
        objective : Callable
            Function to optimize. Takes array or dict (matching ask() output).
        budget : int
            Number of function evaluations.

        Returns
        -------
        best_x : Union[np.ndarray, Dict[str, Any]]
            Best configuration found.
        best_y : float
            Best objective value found.
        """
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        if self._param_space_mode and self.best_x is not None:
            return self.decode(self.best_x), self.best_y
        return self.best_x, self.best_y

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current optimization statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with optimization statistics.
        """
        return {
            "n_observations": self.n_observations,
            "n_leaves": len(self.leaves),
            "max_depth": max((l.depth for l in self.leaves), default=0),
            "best_y": self.best_y,
            "gamma": self.gamma,
            "stagnation": self.stagnation,
            "is_stagnating": self.is_stagnating,
            "iteration": self.iteration,
            "phase": "exploration" if self.iteration < self.exploration_budget else "local_search",
        }

    def __repr__(self) -> str:
        mode = "param_space" if self._param_space_mode else "bounds"
        return (
            f"ALBA(dim={self.dim}, mode={mode}, n_obs={self.n_observations}, "
            f"n_leaves={len(self.leaves)}, best_y={self.best_y:.4f})"
        )
