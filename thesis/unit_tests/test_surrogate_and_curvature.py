import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestSurrogateAndCurvature(unittest.TestCase):
    def build_quadratic_pairs(self, center, H, n=40, noise=0.0):
        # y = -0.5 * (x-center)^T H (x-center)
        d = H.shape[0]
        rng = np.random.default_rng(0)
        X = []
        Y = []
        for _ in range(n):
            u = rng.uniform(-1.0, 1.0, size=d)
            x = center + u
            y = -0.5 * float(u.T @ (H @ u))
            if noise > 0:
                y += rng.normal(0, noise)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def test_fit_surrogate_quality_and_curvature(self):
        d = 2
        cube = QuadCube(bounds=[(-1.5, 1.5), (-1.5, 1.5)])
        cube.R = np.eye(d)
        cube.mu = np.zeros(d)
        # Positive definite H so quadratic has a maximum at center when using negative sign
        H = np.array([[3.0, 0.5],[0.5, 1.5]], dtype=float)
        X, y = self.build_quadratic_pairs(center=np.zeros(d), H=H, n=64, noise=0.0)
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), float(val)) for x, val in zip(X, y)]
        cube.fit_surrogate(min_points=8)
        self.assertIsNotNone(cube.surrogate_2d)
        s = cube.surrogate_2d
        # R^2 reasonably high for noise-free quadratic
        self.assertGreaterEqual(s.get('r2', 0.0), 0.95)
        # Eigenvalues should be finite and mostly positive (since we fitted -0.5 * H)
        lam = s.get('lambda')
        self.assertIsNotNone(lam)
        self.assertTrue(np.all(np.isfinite(lam)))
        # Curvature scores should be > 0 and scale with width^4
        S1 = cube._curvature_scores()
        self.assertIsNotNone(S1)
        # Enlarge bounds -> larger scores
        cube2 = QuadCube(bounds=[(-3.0, 3.0), (-3.0, 3.0)])
        cube2.R = cube.R.copy(); cube2.mu = cube.mu.copy()
        cube2._tested_points = cube._tested_points
        cube2._tested_pairs = cube._tested_pairs
        cube2.fit_surrogate(min_points=8)
        S2 = cube2._curvature_scores()
        self.assertIsNotNone(S2)
        self.assertTrue(np.all(S2 > S1))

    def test_predict_surrogate_positive_sigma(self):
        d = 2
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        cube.R = np.eye(d)
        cube.mu = np.zeros(d)
        H = np.array([[2.0, 0.0],[0.0, 1.0]], dtype=float)
        X, y = self.build_quadratic_pairs(center=np.zeros(d), H=H, n=40, noise=0.0)
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), float(val)) for x, val in zip(X, y)]
        cube.fit_surrogate(min_points=8)
        # Predict at a few points
        for t in [np.array([0.0, 0.0]), np.array([0.5, -0.3]), np.array([-0.9, 0.8])]:
            # Map t in surrogate frame (assume PCA near identity)
            yhat, sigma = cube.predict_surrogate(np.array([t[0], t[1]]))
            self.assertTrue(np.isfinite(yhat))
            self.assertTrue(np.isfinite(sigma))
            self.assertGreaterEqual(sigma, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
