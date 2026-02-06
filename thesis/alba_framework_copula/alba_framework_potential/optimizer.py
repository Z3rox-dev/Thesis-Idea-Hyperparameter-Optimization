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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .acquisition import AcquisitionSelector, UCBSoftmaxSelector
from .candidates import CandidateGenerator, MixtureCandidateGenerator
from .coherence import CoherenceTracker
from .cube import Cube
from .categorical import CategoricalSampler
from .gamma import GammaScheduler, QuantileAnnealedGammaScheduler
from .leaf_selection import LeafSelector, UCBSoftmaxLeafSelector, PotentialAwareLeafSelector
from .local_search import GaussianLocalSearchSampler, LocalSearchSampler, CovarianceLocalSearchSampler
from .drilling import DrillingOptimizer
from .param_space import ParamSpaceHandler
from .splitting import (
    CubeIntrinsicSplitPolicy,
    SplitDecider,
    SplitPolicy,
    ThresholdSplitDecider,
)

warnings.filterwarnings("ignore")


class ALBA:
    """
    ALBA (Adaptive Local Bayesian Allocator) for Hyperparameter Optimization.
    """
    # ... (skipping unchanged parts)
    
    def __init__(
        self,
        # ... (skipping unchanged args)
        coherence_update_interval: int = 5,
        use_potential_field: bool = True,
    ) -> None:
        # ... (skipping unchanged init parts)

        # Coherence tracking for exploit/explore gating
        self._use_coherence_gating = use_coherence_gating
        self._use_potential_field = use_potential_field
        self._coherence_tracker: Optional[CoherenceTracker] = None
        if use_coherence_gating or use_potential_field:
            self._coherence_tracker = CoherenceTracker(
                categorical_dims=categorical_dims or [],
                k_neighbors=6,
                update_interval=coherence_update_interval,
                min_leaves_for_coherence=5,
            )

        # ... (skipping)

        # Strategy components
        # Use PotentialAwareLeafSelector by default if potential field is enabled
        self._leaf_selector: LeafSelector = (
            leaf_selector 
            if leaf_selector is not None 
            else (PotentialAwareLeafSelector() if self._use_potential_field else UCBSoftmaxLeafSelector())
        )
        
        # Inject tracker if selector supports it
        if hasattr(self._leaf_selector, "set_tracker") and self._coherence_tracker is not None:
             self._leaf_selector.set_tracker(self._coherence_tracker)

        # ... (rest unchanged)
warnings.filterwarnings("ignore")


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
    use_coherence_gating : bool
        Whether to use geometric coherence for exploit/explore gating.
    coherence_update_interval : int
        Iterations between coherence cache updates.

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
        split_depth_max: Optional[int] = None,
        split_trials_factor: Optional[float] = None,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        global_random_prob: float = 0.05,
        stagnation_threshold: int = 50,
        categorical_dims: Optional[List[Tuple[int, int]]] = None,
        use_coherence_gating: bool = True,
        coherence_update_interval: int = 5,
        use_potential_field: bool = True,
        use_drilling: bool = False,
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
        
        # Drilling Flag
        self.drilling_enabled = use_drilling

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

        # Split parameters - Apply Scaling Laws if defaults are used
        if split_depth_max is None:
            # Law 4.1: Inverse Depth-Dimensionality
            # 2D -> 16+, 10D -> 8, 20D -> 4
            self._split_depth_max = max(4, int(40 / self.dim))
        else:
            self._split_depth_max = split_depth_max

        if split_trials_factor is None:
            # Law 4.3: Density-Budget Scaling
            # High dim requires higher density (factor) to fit robust LGS
            base_factor = 6.0 if self.dim > 10 else 3.0
            
            # Scale with budget (logarithmic) to force higher density on long runs
            # effectively delaying splits to accumulate more data
            budget_scale = np.log(1 + self.total_budget / 500.0)
            self._split_trials_factor = base_factor * max(1.0, budget_scale)
        else:
            self._split_trials_factor = split_trials_factor
            
        self._split_trials_min = split_trials_min
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

        # Coherence tracking for exploit/explore gating
        self._use_coherence_gating = use_coherence_gating
        self._use_potential_field = use_potential_field
        self._coherence_tracker: Optional[CoherenceTracker] = None
        if use_coherence_gating or use_potential_field:
            self._coherence_tracker = CoherenceTracker(
                categorical_dims=categorical_dims or [],
                k_neighbors=6,
                update_interval=coherence_update_interval,
                min_leaves_for_coherence=5,
            )

        # Initialize categorical sampler
        self._cat_sampler = CategoricalSampler(
            categorical_dims=categorical_dims or [],
            elite_size=10,
            curiosity_bonus=0.3,
            crossover_rate=0.15,
        )

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
            leaf_selector 
            if leaf_selector is not None 
            else UCBSoftmaxLeafSelector()
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
        self._local_search_sampler: LocalSearchSampler = (
            local_search_sampler if local_search_sampler is not None else GaussianLocalSearchSampler()
        )
        
        # Drilling Strategy (Iterative Local Refinement)
        self.driller: Optional[DrillingOptimizer] = None
        # self.drilling_enabled set via init arg

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
        # DRILLING CHECK
        if self.driller is not None:
            # Drilling is active!
            x = self.driller.ask(self.rng)
            if x is not None:
                return x

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
                # Update coherence cache
                if self._coherence_tracker is not None:
                    self._coherence_tracker.update(self.leaves, self.iteration)

            self._last_cube = self._select_leaf()
            return self._sample_in_cube(self._last_cube)

        # Local search phase with tree integration
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress

        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self._last_cube = self._find_containing_leaf(x)
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

        # ---------------------------------------------------------------------
        # DRILLING LOGIC (Iterative Refinement)
        # ---------------------------------------------------------------------
        if self.driller is not None:
            # We are in drilling mode. Update the driller.
            # Driller minimizes cost. ALBA maximizes y.
            # So pass -y as cost.
            continue_drilling = self.driller.tell(x, -y)
            if not continue_drilling:
                self.driller = None  # Stop drilling
        
        # Standard Record Keeping & Triggering
        
        # Update best
        new_best_found = False
        if y > self.best_y_internal: # Internal maximization (fitness)
            # Check if this is a genuine improvement over global best
            # best_y_internal increases.
            # Drilling uses COST (minimization). 
            # We need to be careful about signs.
            # ALBA converts everything to maximize internally. 
            # If y > best_y_internal, it is a new best.
            
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
            new_best_found = True
        else:
            self.stagnation += 1

        # TRIGGER DRILLING
        # If we found a new best (and aren't already drilling), start drilling!
        if self.drilling_enabled and new_best_found and self.driller is None:
            # Convert internal fitness (y) back to cost for Driller?
            # Or use fitness? Drilling is usually minimization.
            # Let's check maximize arg.
            # If maximize=False, then `y` passed to tell is `-cost`.
            # So maximizing `y` minimizes cost.
            # Driller likely expects minimization (lower is better).
            # So we pass -y to Driller.
            
            drilling_cost = -y if not self.maximize else -y # Usually CMA minimizes. 
            # Wait, if maximize=True, we want to MAXIMIZE.
            # CMA minimizes. So we pass -y.
            # If maximize=False, y is already flipped (-cost). So -y is cost.
            # Correct: cost = -y always for minimization target.
            
            # Initialize Driller at x
            self.driller = DrillingOptimizer(
                start_x=x,
                start_y=-y, # Minimization target
                initial_sigma=0.05, # Start small, we are refining
                bounds=list(self.bounds)
            )

        # Record observation
        self.X_all.append(x.copy())
        self.y_all.append(y)

        # Update categorical tracking
        self._cat_sampler.record_observation(x, y)

        # Update cube
        if self._last_cube is not None:
            cube = self._last_cube
            cube.add_observation(x, y, self.gamma)

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

            # Check for split
            if self._should_split(cube):
                children = self._split_policy.split(cube, self.gamma, self.dim, self.rng)
                for child in children:
                    self._cat_sampler.recompute_cube_cat_stats(child, self.gamma)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)
                    # Force coherence update after structural change
                    if self._coherence_tracker is not None:
                        self._coherence_tracker.update(self.leaves, self.iteration, force=True)

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
    # Sampling within cubes
    # -------------------------------------------------------------------------

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        """
        Sample a point within the given cube.
        
        Uses potential-guided sampling when enabled:
        - Low potential (promising region): aggressive gradient following
        - High potential (unpromising region): more exploration
        
        The potential modulates the probability of exploitation vs exploration.

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
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        else:
            # Get potential for this cube (0 = best, 1 = worst)
            potential = 0.5  # default neutral
            if self._coherence_tracker is not None and self._use_potential_field:
                potential = self._coherence_tracker.get_potential(cube, self.leaves)
            
            # Also check coherence for basic gating
            coherence_ok = True
            if self._coherence_tracker is not None and self._use_coherence_gating:
                coherence_ok = self._coherence_tracker.should_exploit(cube, self.leaves)
            
            # Potential-modulated exploitation probability
            # Low potential (0.0) → 95% exploit, High potential (1.0) → 30% exploit
            # But if coherence is bad, cap at 50%
            base_exploit_prob = 0.95 - 0.65 * potential  # Range: [0.30, 0.95]
            if not coherence_ok:
                base_exploit_prob = min(base_exploit_prob, 0.50)
            
            if self.rng.random() < base_exploit_prob:
                # Trust gradient - use LGS-based acquisition
                x = self._sample_with_lgs(cube)
            else:
                # Explore mode
                x = self._sample_explore_mode(cube)

        # Apply categorical sampling
        x = self._cat_sampler.sample(x, cube, self.rng, self.is_stagnating)

        # Re-clip to cube bounds
        x = self._clip_to_cube(x, cube)
        return x

    def _sample_explore_mode(self, cube: Cube) -> np.ndarray:
        """
        Sample in exploration mode when coherence is low.
        
        Uses a mix of:
        - Pure random sampling
        - Center-based sampling with jitter
        - Best point perturbation (local exploration)
        
        Parameters
        ----------
        cube : Cube
            The cube to sample from.
            
        Returns
        -------
        np.ndarray
            Sampled point.
        """
        strategy = self.rng.random()
        
        if strategy < 0.4:
            # Pure random within cube bounds
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        elif strategy < 0.7:
            # Center-based with jitter
            center = cube.center()
            widths = cube.widths()
            jitter = self.rng.uniform(-0.3, 0.3, size=len(center)) * widths
            x = center + jitter
        else:
            # Perturb best point in cube (if available)
            if cube.best_x is not None:
                widths = cube.widths()
                perturbation = self.rng.normal(0, 0.15, size=len(cube.best_x)) * widths
                x = cube.best_x + perturbation
            else:
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        
        return x

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
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

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
        x = self._local_search_sampler.sample(
            self.best_x,
            list(self.bounds),
            self._global_widths,
            float(progress),
            self.rng,
            X_history=self.X_all,
            y_history=self.y_all,
        )
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
        stats = {
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
        
        # Add coherence statistics if available
        if self._coherence_tracker is not None:
            coherence_stats = self._coherence_tracker.get_statistics()
            stats["coherence"] = coherence_stats
        
        return stats

    def __repr__(self) -> str:
        mode = "param_space" if self._param_space_mode else "bounds"
        return (
            f"ALBA(dim={self.dim}, mode={mode}, n_obs={self.n_observations}, "
            f"n_leaves={len(self.leaves)}, best_y={self.best_y:.4f})"
        )
