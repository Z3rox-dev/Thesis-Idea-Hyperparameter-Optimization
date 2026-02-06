"""Local Optuna sampler forks for experimentation.

This package exists to let you modify sampler behavior without patching the
installed Optuna distribution.
"""

from .forked_tpe import LocalGravityTPESampler, LocalGravityConfig

__all__ = ["LocalGravityTPESampler", "LocalGravityConfig"]
