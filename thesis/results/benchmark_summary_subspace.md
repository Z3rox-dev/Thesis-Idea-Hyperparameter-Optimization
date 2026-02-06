# Benchmark Comparison: Backprop vs Subspace (10D)

**Date:** 2025-11-23
**Dimension:** 10
**Budget:** 100 evaluations
**Trials:** 5 per function

## Summary of Results

| Function | Backprop Mean | Subspace Mean | Winner | Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Sphere** | 0.0004 | 0.0005 | Backprop | 0.0001 |
| **Rosenbrock** | 46.42 | 51.71 | Backprop | 5.29 |
| **Rastrigin** | 64.59 | 57.67 | **Subspace** | 6.93 |
| **Griewank** | 0.628 | 0.534 | **Subspace** | 0.095 |
| **Levy** | 4.80 | 4.91 | Backprop | 0.10 |
| **Michalewicz** | -3.96 | -4.01 | **Subspace** | 0.05 |
| **Alpine1** | 11.80 | 13.05 | Backprop | 1.25 |
| **Zakharov** | 48989 | 137197 | Backprop | Huge |

## Analysis

1.  **Subspace Advantage:** The Subspace variant (scanning all curvature axes) performed significantly better on **Rastrigin** and **Griewank**. These are highly multimodal functions where important curvature information might be distributed across many dimensions, not just the top 2 PCA components.
2.  **Backprop Advantage:** The standard Backprop variant (top 2 PCA axes) performed better on **Rosenbrock**, **Sphere**, and **Zakharov**. These functions might have a strong primary direction of curvature where focusing on the top components is more efficient.
3.  **Trade-off:** The Subspace variant adds exploration capability by considering more splitting directions, which helps avoid local optima in complex landscapes (Rastrigin), but might dilute the search focus in simpler or valley-shaped landscapes (Rosenbrock).

## Conclusion

The "Subspace" hypothesis (Active Subspaces) holds merit for complex, multimodal functions in higher dimensions (10D). It successfully identified better splitting directions for Rastrigin and Griewank. However, it is not a universal improvement and can be less efficient for functions with strong primary valleys.
