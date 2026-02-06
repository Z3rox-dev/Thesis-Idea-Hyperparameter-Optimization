import numpy as np
import unittest

from hpo_curvature import QuadCube

class TestSurrogatePredictiveVariance(unittest.TestCase):
    def test_predictive_variance_nonnegative_and_scales(self):
        # Build a simple quadratic surface with noise
        rng = np.random.default_rng(0)
        cube = QuadCube(bounds=[(-1,1),(-1,1)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        X = rng.normal(size=(40,2)) * 0.5
        def f(x):
            return 1.0 + 0.2*x[0] - 0.3*x[1] + 0.5*x[0]*x[0] + 0.25*x[1]*x[1] + 0.1*x[0]*x[1]
        y = np.array([f(xi) + 0.05*rng.normal() for xi in X], dtype=float)
        cube._tested_pairs = [(xi.astype(float), float(yi)) for xi, yi in zip(X, y)]
        cube.fit_surrogate(min_points=8)
        self.assertIsNotNone(cube.surrogate_2d)
        # Predictive std at origin vs far point: variance term v = phi^T Ainv phi grows with |t|
        y0, s0 = cube.predict_surrogate(np.array([0.0,0.0]))
        y1, s1 = cube.predict_surrogate(np.array([1.5,1.5]))
        self.assertTrue(np.isfinite(s0) and np.isfinite(s1))
        self.assertGreaterEqual(s1, s0 * 0.9, f"Expected larger or similar std far from center, got s0={s0}, s1={s1}")
        self.assertGreater(s0, 0.0)

if __name__ == '__main__':
    unittest.main()
