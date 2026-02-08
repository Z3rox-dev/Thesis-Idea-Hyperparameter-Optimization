# ALBA Framework

**ALBA** (Adaptive Local Bayesian Algorithm) for Hyperparameter Optimization.

See `INSTALL.md` for detailed instructions.

## Quick Install

```bash
pip install .
```

This installs ALBA with all dependencies (numpy, scipy, optuna, nevergrad).

## Quick Demo

```bash
python -m alba_framework.examples.quick_demo
```

## Usage

```python
from alba_framework import ALBA

# Continuous optimization
opt = ALBA(bounds=[(-5, 5)] * 5, maximize=False, seed=42, total_budget=400)
best_x, best_y = opt.optimize(my_objective, budget=400)

# Mixed continuous + categorical
param_space = {
    "learning_rate": (1e-4, 1e-1),
    "dropout": (0.0, 0.5),
    "activation": ["relu", "tanh", "gelu"],
}
opt = ALBA(param_space=param_space, maximize=False, total_budget=200)
best_config, best_loss = opt.optimize(train_fn, budget=200)
```
