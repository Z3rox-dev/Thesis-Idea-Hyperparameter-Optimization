# ALBA Framework — Installation & Usage Guide

## Prerequisites

- **Python >= 3.9** (tested with 3.9, 3.10, 3.11, 3.12)
- `pip` (included in any modern Python installation)

> No CUDA, GPU, or external datasets needed: the framework is self-contained.

---

## Option 1 — Install from wheel (recommended)

If you have **`alba_framework-1.0.0-py3-none-any.whl`** or **`alba_framework-1.0.0.tar.gz`**:

```bash
pip install alba_framework-1.0.0-py3-none-any.whl
# or
pip install alba_framework-1.0.0.tar.gz
```

All dependencies (numpy, scipy, optuna, nevergrad) are installed automatically.

---

## Option 2 — Install from source folder

From the `alba_dist/` directory:

```bash
pip install .
```

---

## Option 3 — Direct import (no install)

```bash
pip install numpy scipy optuna nevergrad
```

Then point your Python path:

```python
import sys
sys.path.insert(0, "/path/to/alba_dist")

from alba_framework import ALBA
```

---

## Running the Demo

```bash
# After installation (Option 1 or 2):
python -m alba_framework.examples.quick_demo

# Or directly:
python alba_framework/examples/quick_demo.py
```

The demo compares ALBA vs Optuna (TPE), Random Search, and CMA-ES (nevergrad)
on 10 continuous and 3 mixed continuous+categorical benchmarks.

---

## Quick Usage

### Minimize a function with continuous bounds

```python
from alba_framework import ALBA

bounds = [(-5, 5)] * 10  # 10 dimensions

opt = ALBA(bounds=bounds, maximize=False, seed=42, total_budget=300)
best_x, best_y = opt.optimize(my_objective, budget=300)
print(f"Best: {best_y:.6f}")
```

### Mixed continuous + categorical hyperparameter space

```python
from alba_framework import ALBA

param_space = {
    "learning_rate": (1e-4, 1e-1),
    "hidden_size":   (32.0, 512.0),
    "activation":    ["relu", "tanh", "gelu"],
    "dropout":       (0.0, 0.5),
}

opt = ALBA(param_space=param_space, maximize=False, seed=42, total_budget=200)

for i in range(200):
    config = opt.ask()       # -> dict {"learning_rate": 0.003, ...}
    loss = train_model(**config)
    opt.tell(config, loss)

best_config, best_loss = opt.decode(opt.best_x), opt.best_y
print(f"Best config: {best_config}")
print(f"Best loss:   {best_loss:.6f}")
```

### Ask/tell loop (more control)

```python
opt = ALBA(bounds=bounds, maximize=False, total_budget=200)

for i in range(200):
    x = opt.ask()         # np.ndarray
    y = my_function(x)
    opt.tell(x, y)

print(f"Best: {opt.best_y:.6f}")
print(f"Stats: {opt.get_statistics()}")
```

---

## Package Structure

```
alba_framework/
├── __init__.py          # Exports ALBA and components
├── optimizer.py         # Main ALBA class
├── cube.py              # Adaptive space partitioning
├── param_space.py       # Mixed space handling (continuous, categorical)
├── categorical.py       # Categorical sampling
├── lgs.py               # Local Gradient Surrogate
├── gamma.py             # Dynamic gamma threshold
├── leaf_selection.py    # Leaf selection (Thompson Sampling, UCB)
├── candidates.py        # Candidate generation
├── acquisition.py       # UCB acquisition function
├── splitting.py         # Split policies
├── local_search.py      # Local search (Gaussian, Covariance)
├── coherence.py         # Geometric gradient coherence
├── examples/
│   └── quick_demo.py    # Ready-to-run demo with baselines
└── pyproject.toml       # Metadata for pip install
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'alba_framework'` | Make sure you ran `pip install .` from the `alba_dist/` folder, or use `sys.path.insert()` |
| `ImportError: numpy` | `pip install numpy scipy` |
| `ImportError: optuna` | `pip install optuna nevergrad` |
| Python < 3.9 | Upgrade Python (3.9+ required) |
