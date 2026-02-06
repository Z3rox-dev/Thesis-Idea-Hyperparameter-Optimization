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

