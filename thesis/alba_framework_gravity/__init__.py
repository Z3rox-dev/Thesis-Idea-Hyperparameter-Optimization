"""
ALBA Framework
==============

Adaptive Local Bayesian Algorithm for Hyperparameter Optimization.

ALBA is a novel optimization algorithm combining:
- Adaptive cube partitioning of the search space
- Local Gradient Surrogate (LGS) models within each region
- UCB-style acquisition with exploration bonuses
- Two-phase optimization: exploration + local search refinement
- Curiosity-driven categorical sampling with elite crossover

Installation
------------
The framework is self-contained and requires only numpy.

Quick Start
-----------
Using typed parameter space (recommended):

    >>> from alba_framework import ALBA
    >>> param_space = {
    ...     'learning_rate': (1e-4, 1e-1, 'log'),
    ...     'hidden_size': (32, 512, 'int'),
    ...     'activation': ['relu', 'tanh', 'gelu'],
    ... }
    >>> opt = ALBA(param_space=param_space, maximize=False, seed=42)
    >>> for _ in range(100):
    ...     config = opt.ask()  # Returns dict
    ...     loss = train_model(**config)
    ...     opt.tell(config, loss)
    >>> best_config, best_loss = opt.decode(opt.best_x), opt.best_y

Using bounds directly:

    >>> from alba_framework import ALBA
    >>> bounds = [(0, 1), (0, 1), (0, 1)]
    >>> opt = ALBA(bounds=bounds, maximize=True)
    >>> for _ in range(100):
    ...     x = opt.ask()  # Returns np.ndarray
    ...     y = objective(x)
    ...     opt.tell(x, y)

Using the optimize() helper:

    >>> from alba_framework import ALBA
    >>> opt = ALBA(param_space=param_space)
    >>> best_x, best_y = opt.optimize(objective_fn, budget=200)

Modules
-------
- optimizer: Main ALBA class
- cube: Cube class for space partitioning
- param_space: Parameter space handling and encoding
- categorical: Categorical sampling strategies

Author
------
Thesis implementation.

Version
-------
1.0.0
"""

__version__ = "1.0.0"
__author__ = "Thesis Implementation"

from .optimizer import ALBA
from .cube import Cube
from .param_space import ParamSpaceHandler
from .categorical import CategoricalSampler
from .lgs import fit_lgs_model, predict_bayesian
from .gamma import GammaScheduler, QuantileAnnealedGammaScheduler
from .leaf_selection import LeafSelector, UCBSoftmaxLeafSelector
from .candidates import CandidateGenerator, MixtureCandidateGenerator
from .acquisition import AcquisitionSelector, UCBSoftmaxSelector
from .splitting import (
    SplitDecider,
    SplitPolicy,
    ThresholdSplitDecider,
    CubeIntrinsicSplitPolicy,
)
from .local_search import LocalSearchSampler, GaussianLocalSearchSampler, LLRGradientLocalSearchSampler

__all__ = [
    "ALBA",
    "Cube",
    "ParamSpaceHandler",
    "CategoricalSampler",
    "fit_lgs_model",
    "predict_bayesian",
    "GammaScheduler",
    "QuantileAnnealedGammaScheduler",
    "LeafSelector",
    "UCBSoftmaxLeafSelector",
    "CandidateGenerator",
    "MixtureCandidateGenerator",
    "AcquisitionSelector",
    "UCBSoftmaxSelector",
    "SplitDecider",
    "SplitPolicy",
    "ThresholdSplitDecider",
    "CubeIntrinsicSplitPolicy",
    "LocalSearchSampler",
    "GaussianLocalSearchSampler",
    "LLRGradientLocalSearchSampler",
    "__version__",
]
