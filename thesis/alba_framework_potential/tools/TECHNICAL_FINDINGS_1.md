# ALBA-Potential: Technical Findings

## What is the Potential Field?

The potential field assigns a scalar value to each leaf cube, representing how "promising" that region is for containing the global minimum.

**Mathematical Basis**:
Given local gradient estimates `g_l` from each leaf's LGS model, we solve for a global potential `u` such that:
```
u_m - u_l ≈ g_l^T (c_m - c_l)   (for neighboring leaves l, m)
```
This is solved via weighted least-squares, creating a consistent "energy landscape" over the search space.

**Intuition**: Think of it like a topographic map where:
- **Low potential (0)** = likely near the minimum (exploit aggressively)
- **High potential (1)** = likely far from minimum (explore more)

The potential modulates the exploitation probability: `P(exploit) = 0.95 - 0.65 * potential`

---

## Overview
This document captures key technical findings from the development of the potential field extension for ALBA.

---

## Finding 1: High-Dimensional Graph Sparsity

**Problem**: At high dimensions (20D+), the potential field becomes uninformative.

| Dimension | #Leaves | Valid Edge Ratio | Potential Variance |
|-----------|---------|------------------|-------------------|
| 10D | 13 | 0.65 | 0.006 |
| 20D | 7 | **0.16** | **0.001** |
| 30D | 5 | 0.05 | 0.0003 |

**Root Cause**: 
- Fewer cubes are created at high-D due to the curse of dimensionality
- With only 7 leaves and k=6 neighbors, the kNN graph becomes nearly complete but with few valid edges (leaves needing valid LGS models)
- Potential variance drops below threshold → falls back to `good_ratio` proxy

**Mitigation**: When `variance < 0.001`, use `1.0 - good_ratio` as potential instead of neutral 0.5.

---

## Finding 2: Potential Field Noise at Low Dimensions

**Problem**: At 10D, the potential field is noisy (oscillates between updates).

| Dimension | Mean Δ Potential | Max Δ Potential |
|-----------|-----------------|-----------------|
| 10D | 0.021 | **0.29** |
| 20D | 0.000 | 0.000 |

**Root Cause**:
- With more leaves (13), the kNN graph changes frequently as new points are added
- Each update recalculates the least-squares potential from scratch
- Small changes in gradient estimates cause ~30% swings in potential values

**Counter-intuitive Result**: Update interval=1 performs **worse** than interval=5, because interval=5 provides implicit temporal smoothing.

**Future Work**: Consider exponential moving average (EMA) for potential smoothing.

---

## Finding 4: Optimal Tree Depth vs Dimensionality

**Problem**: The "optimal" max tree depth is not constant but depends inversely on dimensionality.

| Dimension | Depth 4 | Depth 8 | Depth 16+ | Optimal |
|-----------|---------|---------|-----------|---------|
| 2D (Sphere) | 0.0047 | **0.0035** | 0.0054 | **8** |
| 10D (Sphere) | 2.56 | **1.76** | 1.76 | **8** |
| 10D (Ackley) | **12.22** | 12.55 | 12.55 | **4** |
| 20D (Sphere) | **11.19** | 13.74 | 13.74 | **4** |

**Insight**: 
- **Low Dimensions (2D-10D)**: Deeper trees (Depth 8) allow finer partitioning, which is beneficial because local density is high enough for LGS models.
- **High Dimensions (20D+)**: Shallower trees (Depth 4) are superior. Deeper splits create leaves with too few points relative to the dimensionality (`N < dim + 2`), making LGS gradients unstable. For 20D, keeping leaves larger (Depth 4) maintains model stability.

**Conclusion**: Depth should be adapted to dimensionality. A rule of thumb like `max_depth ≈ 40 / dim` (bounded) might be appropriate.

---

## Finding 5: LGS Distortion and The "Endless Split" Paradox

**Hypothesis**: Increasing the budget should allow deeper trees to work well by providing more data points per leaf.

**Result**: **False**.

| Depth | Budget | Points/Leaf | Gradient Quality (CosSim) | Best Score |
|-------|--------|-------------|---------------------------|------------|
| 16 | 400 | 44.4 | -0.15 | 15.02 |
| 16 | 1600 | 41.0 | -0.21 | 5.11 |
| 16 | 3200 | 45.1 | -0.37 | 3.37 |

**Insight**:
- As budget increases, the algorithm simply splits more leaves.
- The **local density (points per leaf) remains constant** (~40-50 points) regardless of the total budget.
- Therefore, the gradient estimation quality (distortion) does NOT improve with budget if `split_depth_max` allows further splitting.

**Calibration**: To improve gradient quality, we must force higher local density. This requires:
1.  Limiting `split_depth_max` (as seen in Finding 4)
2.  Or increasing `stagnation_threshold` (points before split)

---

## Finding 6: Optimal Leaf Density (Split Factor)

**Problem**: The default split threshold (`3.0 * dim + 6`) is too aggressive for high dimensions, creating sparse leaves before LGS converges.

**Experiment**: Varying `split_trials_factor` on Sphere 20D (Budget 400).

| Factor | Required Points | Mean Score | Pts/Leaf |
|--------|-----------------|------------|----------|
| 3.0 (Def) | 66 | 13.74 | 46 |
| **6.0** | **126** | **11.31** | **80** |
| 10.0 | 206 | 12.88 | 133 |

**Insight**:
- Increasing density/factor improves performance significantly (similar to limiting depth).
- **Factor 6.0 is optimal**: it enforces ~80 points/leaf, allowing LGS to fit 20D gradients reliably.
- Too high (10.0+) delays splitting too much for the given budget.

**Conclusion**: For >10D problems, setting `split_trials_factor=6.0` is recommended over the default 3.0.

---

## Finding 7: Density-Budget Scaling (Stability)

**Hypothesis**: High budget improves performance more effectively if coupled with high split factor (forced density).

**Experiment**: Sphere 20D, comparing Factor 3.0 vs 10.0 at Budget 400 vs 1600.

| Scenario | Factor | Budget | Mean Y | Std Y | Insight |
|---|---|---|---|---|---|
| Base | 3.0 | 400 | 13.74 | 2.47 | Poor |
| High Budget | 3.0 | 1600 | 5.24 | 1.88 | Good, but unstable |
| High Factor | 10.0 | 400 | 12.87 | 3.64 | Too slow for low budget |
| **Scaled** | **10.0** | **1600** | **4.66** | **0.48** | **Best & Most Stable** |

**Conclusion**:
- Increasing budget alone improves mean performance but leaves high variance (Std 1.88) because leaves remain sparse.
- Increasing budget AND split factor delivers the best performance AND stability (Std 0.48).
- **Recommendation**: For expensive/long runs, scale `split_trials_factor` proportionally to budget to ensure LGS model quality.

---

---

## Finding 8: The Gradient Noise-Robustness Paradox

**Problem**: A forensic audit revealed that local gradient estimates (LGS) are surprisingly poor in high dimensions, yet the potential field works well.

**Experiment**: Sphere 10D, Budget 1000.
- **Metric 1 (Fidelity)**: Cosine Similarity between LGS gradient and True gradient.
- **Metric 2 (Utility)**: Directional Accuracy (does moving to a lower-potential neighbor actually decrease cost?).

| Metric | Result | Target | Interpretation |
|--------|--------|--------|----------------|
| Gradient Fidelity | **-0.44** | -1.0 | **Poor**. LGS is noisy and barely better than random (-0.0) at pointing downhill. |
| Potential Accuracy | **77%** | 100% | **Good**. The field correctly identifies descent directions 3 out of 4 times. |

**Insight**: 
- Individual LGS models are weak learners (suffering from N ≈ D sparsity).
- However, the **Potential Field solving step (Poisson equation)** acts as a powerful global regularizer. It integrates conflicting noisy local signals into a smooth global topology, effectively "denoising" the landscape.

**Conclusion**: ALBA-Potential works not because it estimates exact gradients, but because it recovers the **topological flow** through consensus of many weak estimators.

---

## Finding 9: The Conditioning/Locality Trade-Off (Whitening Experiment)

**Hypothesis**: Standardizing input features (Whitening) inside each leaf should fix the poor performance on anisotropic functions like Ellipsoid.

**Result**: **Trade-off confirmed**.
- **Ellipsoid**: Whitening improved gradient quality significantly (-0.14 -> -0.56) and guidance accuracy (59% -> 65%).
- **Rastrigin**: Whitening degraded guidance accuracy (70% -> 57%) by amplifying noise in local minima.

**Insight**:
- **Geometry vs Function**: Splitting produces leaves that are geometrically isotropic (condition number < 50), even if the function is anisotropic. Thus, geometric condition checks cannot reliability trigger whitening.
- **Regularization**: Standard LGS uses Ridge Regularization which penalizes high slopes. Anisotropic functions require high slopes. Whitening effectively lowers the penalty, fixing the bias for convex functions but increasing variance for noisy functions.
- **Decision**: We prioritize **Robustness on Multimodal Landscapes** (Rastrigin) over speed on simple Anisotropic Valleys (Ellipsoid). Therefore, Whitening is **disabled** by default.

---

## Finding 3: Tested Improvements (Rejected)

| Modification | Result | Reason for Rejection |
|-------------|--------|---------------------|
| Residual-based uncertainty | ❌ | High variance on simple functions |
| Success propagation on graph | ❌ | No improvement, slight regression |
| Tiered gating (Q60/Q80) | ❌ | Catastrophic on Rosenbrock 20D (-73%) |
| Update interval=1 | ❌ | Worse due to noise amplification |
| EMA smoothing (α=0.3) | ❌ | Improves Ackley, worsens Sphere |

---

## Current Configuration (Validated)

```python
# coherence.py fixes:
- Potential inverted for minimization (-u instead of u)
- Re-anchored to best-performing leaf  
- Sigmoid normalization (prevents extreme values)
- Fallback to good_ratio when variance < 0.001
- Normalized kNN distances (equal dimension contribution)

# optimizer.py defaults:
- coherence_update_interval = 5 (temporal smoothing)
- empirical_bonus = good_ratio * 2.0
```

**Benchmark Result**: ALBA-Potential wins 7/8 comparisons vs ALBA-Coherence.

---

## Finding 10: The Hybridization Breakthrough (Partitioning + Covariance)

**Problem**: While ALBA excels at multimodal exploration (Rastrigin), it struggles significantly on **correlated valleys** (Rosenbrock, Ellipsoid) compared to CMA-ES.
- **Rosenbrock 20D**: CMA ~20, ALBA ~35,000.
- **Root Cause**: ALBA's partitioning creates axis-aligned boxes. Its local search uses isotropic Gaussian noise ($\mathcal{N}(0, I)$). Neither mechanism can efficiently traverse a diagonal valley $x_2 \approx x_1^2$.

**Hypothesis**: Integrating a simplified **Covariance Matrix Adaptation (CMA)** mechanism into the *Local Search* phase—while keeping the Global Partitioning for exploration—should capture the best of both worlds.

**Implementation**: `CovarianceLocalSearchSampler`.
- Instead of sampling from a sphere, we sample from $\mathcal{N}(x_{best}, \Sigma)$, where $\Sigma$ is the empirical covariance of the top $k$ points found so far.

**Experiment**: Run 15 (20D, Budget 500).

| Function | Standard ALBA | **Hybrid ALBA** (Covariance) | Improvement | CMA-ES (Baseline) |
|----------|---------------|------------------------------|-------------|-------------------|
| **Rosenbrock** | 34,960 | **3,330** | **10.5x** | 19.18 |
| **Ellipsoid** | 334,641 | **37,840** | **8.8x** | 222.47 |

**Insight**:
- The global partitioning successfully locates the promising region (the valley).
- The covariance sampler successfully "learns the shape" of the valley, enabling efficient descent down the diagonal.
- This closes the performance gap with CMA by an order of magnitude.

**Conclusion**: **Hybrid Partitioning + Covariance** is the superior architecture for general-purpose black-box optimization. It combines:
1.  **Global Robustness**: Partitioning prevents getting stuck in bad local optima (where CMA sometimes fails).
2.  **Local Efficiency**: Covariance adaptation handles ill-conditioned/correlated curvature.

---

## Session: January 30, 2026 - Intensive Numerical Debugging

### Overview
Deep numerical debugging session focused on finding bugs by analyzing exploding results across 22 test functions. The approach was: dump numbers, trace computations, identify anomalies, and distinguish real bugs from expected stochastic behavior.

---

## Finding 11: LGS Gradient Explosion Bug

**File**: `lgs.py`  
**Severity**: Critical  
**Discovered via**: IllConditioned function showing gradient norms of ~1,000,000

### Symptoms
- `grad_norm` reaching 10^6 on ill-conditioned functions
- Poor optimization on functions with heterogeneous scaling
- IllConditioned function: mean result 9.91 (should be ~0)

### Root Cause
In `fit_lgs_model()`, the gradient was computed in normalized space then incorrectly scaled:

```python
# Bug: gradient explosion
grad = inv_cov @ (X_norm.T @ W @ y_centered)  # grad in normalized space
grad *= y_std  # EXPLOSION when y_std >> 1 (e.g., 2742 on IllConditioned)
```

On IllConditioned with scales [1, 10, 100, 1000, 10000], `y_std ≈ 2742`. Multiplying the normalized gradient by this caused explosion to ~10^6.

### Fix
Keep gradient in normalized space. Store `y_std` in model dict and denormalize only at prediction time:

```python
# In fit_lgs_model:
# grad stays in normalized space (no *= y_std)
model['y_std'] = y_std  # Save for later denormalization

# In predict_bayesian:
mu = y_mean + mu_normalized * y_std  # Denormalize prediction
sigma = np.sqrt(var_normalized) * y_std  # Denormalize uncertainty
```

### Result
- IllConditioned: 9.91 → 0.37 (**96.2% improvement**)

---

## Finding 12: Drilling Points Lost Bug

**File**: `optimizer.py`  
**Severity**: Critical  
**Discovered via**: BentCigar showing 232x gap between seeds, with worst seed having n_leaves=1 forever

### Symptoms
- BentCigar seed 9: best_y = 12,658 (should be ~0)
- n_leaves stayed at 1 for all 150 iterations
- Only 3 out of 60 points were being used for LGS fitting

### Root Cause
In the drilling path of `ask()`, the method returned early without setting `_last_cube`:

```python
if self.driller is not None:
    x = self.driller.ask(self.rng)
    self._last_cube = self._find_containing_leaf(x)  # THIS LINE WAS MISSING!
    return x
```

Without `_last_cube` set, `tell()` skipped `cube.add_observation()`:

```python
if self._last_cube is not None:  # False during drilling!
    cube = self._last_cube
    cube.add_observation(x, y, self.gamma)  # Never executed
```

This meant 57 out of 60 drilling points were completely lost - no cube updates, no LGS fitting, no splitting.

### Fix
Added `self._last_cube = self._find_containing_leaf(x)` before return in the drilling block.

### Result
- BentCigar: 795 → 0.20 (**99.97% improvement**)

---

## Finding 13: Drilling Budget Monopoly Bug

**Files**: `optimizer.py`, `drilling.py`  
**Severity**: High  
**Discovered via**: DifferentPowers seed 9 showing drilling=150/150 iterations (100% of budget!)

### Symptoms
- Some seeds spent 100% of budget in drilling mode
- DifferentPowers seed 9: drilling for all 150 iterations, best_y = 0.09
- No exploration phase, no cube splitting, no LGS model fitting

### Root Cause
In `drilling.py`, every success granted +10 steps:

```python
# ADAPTIVE BUDGET: Earn more steps for success!
if self.max_steps < self.step_cap:
    max_steps += 10  # Too aggressive!
```

Combined with:
1. Drilling started immediately on first "new best" (iteration 0!)
2. No global cap on total drilling iterations
3. Smooth functions kept producing small improvements → infinite drilling

### Fix (3 parts)

**1. drilling.py** - Reduce success bonus:
```python
max_steps += 2  # Reduced from +10
```

**2. optimizer.py** - Add warmup period:
```python
self.drilling_warmup = 20  # Min iterations before drilling can start
```

**3. optimizer.py** - Add global cap:
```python
self.drilling_budget_max = int(total_budget * 0.4)  # Max 40% for drilling
self.drilling_budget_used = 0  # Track usage

# In tell():
if self.drilling_budget_used >= self.drilling_budget_max:
    self.driller = None  # Force stop
```

### Result
- DifferentPowers seed 9: 0.09 → 2.14e-08 (**drilling capped at 60 iterations**)

---

## Finding 14: Seed Variability on Boundary Optima (Non-Bug)

**Status**: Expected behavior, not a bug  
**Functions affected**: Levy, Ackley, DifferentPowers

### Observation
Some functions showed extreme gaps (10^30) between best and worst seeds. Example:
- Levy: seed 1 finds optimum exactly (10^-32), seed 0 stops at 10^-4

### Analysis

**Case Study: Ackley seed 2 (failed)**

The optimum is at [0,0,0,0,0] (corner of search space).

| Dimension | Min value seen | First time < 0.05 |
|-----------|---------------|-------------------|
| 0 | 0.0000 | iter 9 |
| 1 | 0.0000 | iter 12 |
| 2 | **0.0098** | iter 11 |
| 3 | 0.0000 | iter 19 |
| 4 | 0.0000 | iter 19 |

Final best: `[0, 0, 0.907, 0, 0]` with y = 1.65

**Key insight**: The algorithm DID find points with low x[2] (down to 0.0098), but when x[2] was low, other dimensions were high. Example at iter 120:
- x = [0.19, 0.87, **0.0098**, 0.72, 0.80]
- y = 3.52 (worse than best!)

The best point [0, 0, 0.907, 0, 0] has 4 dimensions at 0 but dimension 2 stuck at 0.907 because that combination was never sampled.

### Correlation Analysis

| Function | Correlation (early_dist vs final_y) | Interpretation |
|----------|-------------------------------------|----------------|
| Levy | -0.374 | Weak/negative - algorithm corrects |
| Ackley | 0.404 | Moderate - some luck factor |
| Sphere | -0.279 | Weak/negative - algorithm corrects |

**Conclusion**: The correlation is moderate to low, meaning ALBA generally recovers from poor starts. The 1-in-10 failure rate on boundary optima is expected for stochastic optimization with limited budget.

### Possible Mitigations (not bug fixes)
1. Latin Hypercube Sampling for initial points
2. Force corner exploration
3. Increase `global_random_prob`
4. Restart on prolonged stagnation

---

## Summary Table (Jan 30, 2026 Session)

| Bug | Before | After | Improvement |
|-----|--------|-------|-------------|
| LGS Gradient Explosion | 9.91 | 0.37 | 96.2% |
| Drilling Points Lost | 795 | 0.20 | 99.97% |
| Drilling Budget Monopoly | 150 iters | 60 iters (capped) | 60% budget saved |

---

## Debugging Methodology

1. **Create bug_hunter.py** - automated testing across 22 functions, 5 seeds each
2. **Flag anomalies**: extreme gaps between seeds, NaN/Inf, warnings
3. **Deep trace**: dump numerical values at each step for worst seed
4. **Compare good vs bad**: find the divergence point
5. **Root cause analysis**: trace back to exact line of code
6. **Fix without bias**: no clipping, no masking - fix the actual math
7. **Verify**: run same tests to confirm improvement

---

## Session: January 30, 2026 - Deep Assumption Verification

### Overview
Systematic verification of implicit assumptions in ALBA codebase. For each assumption: formal definition, test, counter-example if violated.

---

## Finding 15: Implicit Assumptions Verified

Ten implicit assumptions were tested systematically. Results:

| Assumption | Description | Status |
|------------|-------------|--------|
| A1 | Normalization: X_norm ∈ [-0.5, 0.5]^d | ✅ PASS (with warning for tiny cubes) |
| A2 | Covariance PD: XtWX + λI invertible | ✅ PASS |
| A3 | Gradient normalized: ||grad|| ~ O(1) | ✅ PASS |
| A4 | Gaussian weights: exp(-d²/2σ²) ∈ (0, 1] | ✅ PASS |
| A5 | Split coverage: children cover parent | ✅ PASS |
| A6 | Gamma threshold: sensible separation | ✅ PASS |
| A7 | UCB exploration: sigma > 0 | ✅ PASS |
| A8 | Drilling convergence: sigma decreases | ✅ PASS |
| A9 | Best_y tracking: matches true minimum | ✅ PASS |
| A10 | Predict consistency: center → y_mean | ✅ PASS |

### A1 Warning: Tiny Cube Division Risk
After many splits, `widths < 1e-9`. The code uses `np.maximum(widths, 1e-9)` as protection, but this is worth monitoring.

---

## Finding 16: Rank Weights Semantics Verification

### Concern
The rank_weights formula appeared to give higher weight to worse points:
```python
rank_weights = 1.0 + 0.5 * (scores - scores.min()) / ptp
```

### Analysis
- **In raw space**: Higher score = worse for minimization → gets higher weight (wrong!)
- **In ALBA**: Scores are **negated** in `_to_internal()` when `maximize=False`
- **Result**: Higher internal score = better for optimization → gets higher weight (correct!)

### Verification
```python
original_scores = [0.1, 0.5, 1.0, 2.0, 5.0]  # 0.1 is best
internal_scores = [-0.1, -0.5, -1.0, -2.0, -5.0]  # -0.1 is highest = best
rank_weights = [1.500, 1.459, 1.408, 1.306, 1.000]
# ✅ Best point (0.1) has highest weight (1.5)
```

**Conclusion**: The formula is correct because ALBA internally maximizes negated scores.

---

## Finding 17: LGS Gradient Alignment on Multimodal Functions

### Observation
On Rastrigin (highly multimodal), 4/9 leaves had gradients pointing **opposite** to the global optimum:

| Leaf Center | Alignment | Status |
|-------------|-----------|--------|
| [0.5, 0.5, 0.87] | 0.925 | ✅ Good |
| [0.5, 0.06, 0.35] | 0.329 | ⚠️ Weak |
| [0.5, 0.17, 0.35] | **-0.353** | ❌ Wrong |
| [0.5, 0.5, 0.71] | **-0.091** | ❌ Wrong |
| [0.5, 0.5, 0.72] | 0.844 | ✅ Good |
| [0.5, 0.27, 0.31] | 0.357 | ⚠️ Weak |
| [0.5, 0.27, 0.66] | **-0.446** | ❌ Wrong |
| [0.5, 0.36, 0.35] | **-0.286** | ❌ Wrong |
| [0.5, 0.70, 0.35] | 0.759 | ✅ Good |

### Root Cause
This is **not a bug** - it's a fundamental limitation of local gradient estimation:
- LGS fits a local linear model to points in each cube
- On multimodal functions, the local gradient points toward the **local** minimum
- This may be opposite to the **global** optimum direction

### Mitigation Strategies (already in ALBA)
1. **Random exploration**: `global_random_prob = 0.05` for diversity
2. **Space partitioning**: Splitting creates multiple local models
3. **UCB exploration**: sigma term explores uncertain regions
4. **Coherence tracker**: Detects when local gradients disagree

### Recommendation
For highly multimodal functions, consider:
- Increasing `global_random_prob` to 0.1-0.2
- Reducing trust in gradient direction (`novelty_weight` higher)
- Using restarts after prolonged stagnation

---

## Finding 18: Leaves Without LGS Model

### Observation
Some leaves have `lgs_model = None`, causing `predict_bayesian` to return `(mu=0, sigma=1)`.

### Analysis
```
Leaf 22: n_trials=2 (< 5 = dim+2), NO MODEL
Leaf 28: n_trials=2 (< 5 = dim+2), NO MODEL
```

The minimum requirement for LGS fitting is `dim + 2` points (for linear regression with regularization). New leaves created by splitting often have fewer points.

### Impact
When `model = None`:
- `predict_bayesian` returns `(0, 1)` for all candidates
- UCB = 0 + 0.4 * 1 = 0.4 (constant)
- Selection is essentially **random** in these cubes

### Current Mitigation
Parent backfill: `if len(pairs) < 3*dim: pairs += parent_pairs[:needed]`

### Potential Improvement
Consider fallback to parent's LGS model instead of returning constant values:
```python
if model is None and cube.parent is not None:
    model = cube.parent.lgs_model
```

---

## Finding 19: LGS Gradient Mathematical Correctness

### Test Setup
Quadratic function with known gradient:
```
f(x) = (x - c)^T A (x - c)
c = [0.3, 0.6, 0.4]
A = diag([1, 4, 9])
∇f(x) = 2A(x - c)
```

### Results
At center [0.5, 0.5, 0.5]:
- True gradient (internal): [-0.4, 0.8, -1.8]
- LGS gradient: [-0.017, 1.013, -1.585]
- **Alignment: 0.9706** ✅

### Conclusion
For unimodal/convex functions, LGS gradient is highly accurate. The weighted regression correctly approximates the true gradient direction.

---

## Tools Created

| Tool | Purpose |
|------|---------|
| `tools/assumption_checker.py` | Systematic verification of 10 implicit assumptions |
| `tools/deep_trace.py` | Step-by-step numerical tracing of LGS fit |
| `tools/e2e_trace.py` | End-to-end verification from user input to selection |
| `tools/counter_examples.py` | Edge cases that stress test assumptions |
| `tools/deep_investigation.py` | Detailed analysis of problematic cases |

---

## Finding 27: GammaScheduler NaN propagation from y_all

**Date**: 2025-01-13
**Severity**: HIGH (corrupts core algorithm state)
**Validation Status**: VALIDATED ✅

### Problem
`QuantileAnnealedGammaScheduler.compute()` uses `np.percentile()` which returns NaN if input contains NaN:

```python
# Pre-fix: If any y_raw is NaN, gamma becomes NaN
y_all = [1.0, 2.0, np.nan, 3.0, 4.0]
gamma = np.percentile(y_all, 80)  # NaN!
```

When gamma is NaN, all "good point" calculations break:
- `y >= gamma` → always False when gamma is NaN
- No points are marked as "good"
- LGS models cannot fit properly
- The entire optimization degrades

### How NaN enters y_all
- User calls `tell(x, y_raw)` where `y_raw = NaN` (e.g., failed evaluation)
- `y_all.append(y)` stores the NaN
- Next `_update_gamma()` call corrupts gamma

### Fix Applied
```python
# In QuantileAnnealedGammaScheduler.compute():
y_arr = np.asarray(y_all, dtype=float)
finite_mask = np.isfinite(y_arr)
if np.sum(finite_mask) < 10:
    return 0.0  # Not enough finite values
y_finite = y_arr[finite_mask]
return float(np.percentile(y_finite, 100 * (1 - current_quantile)))
```

### Verification
```python
opt = ALBA(bounds=[(0, 1), (0, 1)], total_budget=50)
for i in range(15):
    x = opt.ask()
    y = np.nan if i % 3 == 0 else -np.sum((x - 0.5)**2)
    opt.tell(x, y)

print(opt.gamma)  # Valid float, not NaN
```

---

## Finding 26: DrillingOptimizer NaN Propagation (LOW PRIORITY)

**Date**: 2025-01-13
**Severity**: LOW (defense in depth)
**Validation Status**: VALIDATED ✅

### Problem
`DrillingOptimizer` does not sanitize `start_x` or `start_y` in `__init__`, causing NaN to propagate through `ask()`:

```python
# Pre-fix: NaN in start_x propagates
driller = DrillingOptimizer(np.array([0.5, np.nan, 0.5]), start_y=1.0)
x = driller.ask(rng)  # x contains NaN
```

### Validation Analysis
**Is this a real-world bug?**
- In ALBA context: NO - `start_x` comes from `ask()` → `CandidateGenerator` which is already fixed (Finding 23)
- Standalone use: YES - if someone uses `DrillingOptimizer` directly with invalid input

**Verdict**: Defense in depth fix - not critical for ALBA but improves API robustness

### Fix Applied
```python
# In __init__:
start_x = np.array(start_x, dtype=float)
if not np.all(np.isfinite(start_x)):
    if bounds is not None:
        centers = np.array([(lo + hi) / 2 for lo, hi in bounds])
    else:
        centers = np.full(self.dim, 0.5)
    invalid_mask = ~np.isfinite(start_x)
    start_x[invalid_mask] = centers[invalid_mask]

self.best_y = float(start_y) if np.isfinite(start_y) else 0.0
```

### Verification
```python
driller = DrillingOptimizer(np.array([0.5, np.nan, 0.5]), start_y=1.0, bounds=[(0,1)]*3)
print(driller.mu)  # [0.5, 0.5, 0.5] - NaN replaced with center
x = driller.ask(rng)  # No NaN
```

---

1. **Score transformation**: `y_internal = -y_raw` when `maximize=False`
2. **X normalization**: `X_norm = (X - center) / widths` always in [-0.5, 0.5]^d
3. **Gradient space**: LGS gradient computed in normalized space, no y_std scaling
4. **Weight correctness**: Best internal scores get highest rank_weights
5. **Split coverage**: Children completely cover parent bounds
6. **Best tracking**: `best_y` always matches `min(y_raw)` observed


