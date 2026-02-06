import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestNumericConsistency(unittest.TestCase):
    def test_surrogate_coefficients_recovery(self):
        # True quadratic in 2D surrogate frame
        # f(t) = c + b1 t1 + b2 t2 + 0.5 a11 t1^2 + 0.5 a22 t2^2 + a12 t1 t2
        c_true = 0.7
        b1_true, b2_true = 0.3, -0.2
        a11_true, a22_true, a12_true = 2.5, 1.2, 0.4
        d = 2
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        cube.R = np.eye(d); cube.mu = np.zeros(d)
        rng = np.random.default_rng(0)
        T = rng.uniform(-1.0, 1.0, size=(400, d))
        y = (
            c_true
            + b1_true * T[:, 0]
            + b2_true * T[:, 1]
            + 0.5 * a11_true * T[:, 0] ** 2
            + 0.5 * a22_true * T[:, 1] ** 2
            + a12_true * T[:, 0] * T[:, 1]
        )
        X = (cube.mu.reshape(-1, 1) + cube.R @ T.T).T
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), float(val)) for x, val in zip(X, y)]
        cube.fit_surrogate(min_points=8)
        self.assertIsNotNone(cube.surrogate_2d)
        w = np.asarray(cube.surrogate_2d['w']).reshape(-1)
        # Map w to parameters: [c, b1, b2, a11, a22, a12]
        c_hat, b1_hat, b2_hat, a11_hat, a22_hat, a12_hat = w
        # Allow small ridge bias; tolerances tuned
        self.assertAlmostEqual(c_hat, c_true, delta=0.02)
        self.assertAlmostEqual(b1_hat, b1_true, delta=0.03)
        self.assertAlmostEqual(b2_hat, b2_true, delta=0.03)
        self.assertAlmostEqual(a11_hat, a11_true, delta=0.05)
        self.assertAlmostEqual(a22_hat, a22_true, delta=0.05)
        self.assertAlmostEqual(a12_hat, a12_true, delta=0.05)

    def test_curvature_k4_scaling(self):
        # Use the previously learned surrogate to compute curvature scores.
        d = 2
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        cube.R = np.eye(d); cube.mu = np.zeros(d)
        rng = np.random.default_rng(1)
        T = rng.uniform(-1.0, 1.0, size=(200, d))
        # Simple convex quadratic around 0
        y = -0.5 * (3.0 * T[:, 0] ** 2 + 1.0 * T[:, 1] ** 2)
        X = T.copy()
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), float(val)) for x, val in zip(X, y)]
        cube.fit_surrogate(min_points=8)
        S1 = cube._curvature_scores()
        self.assertIsNotNone(S1)
        # Scale bounds by k and expect S scales by k^4
        k = 2.0
        cube2 = QuadCube(bounds=[(-k, k), (-k, k)])
        cube2.R = cube.R.copy(); cube2.mu = cube.mu.copy()
        cube2._tested_points = cube._tested_points
        cube2._tested_pairs = cube._tested_pairs
        cube2.fit_surrogate(min_points=8)
        S2 = cube2._curvature_scores()
        self.assertIsNotNone(S2)
        ratio = S2 / S1
        self.assertTrue(np.allclose(ratio, (k ** 4) * np.ones_like(ratio), rtol=0.1, atol=0.1))

    def test_pca_principal_axes_alignment(self):
        # Create anisotropic Gaussian with known covariance Q diag(e) Q^T
        d = 3
        rng = np.random.default_rng(2)
        # Random orthogonal via QR
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        evals_true = np.array([5.0, 2.0, 0.5])
        C = Q @ np.diag(evals_true) @ Q.T
        X = rng.multivariate_normal(mean=np.zeros(d), cov=C, size=500)
        cube = QuadCube(bounds=[(-3.0, 3.0)] * d)
        cube.R = np.eye(d); cube.mu = np.zeros(d)
        cube.depth = 2
        # Scores uniform
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), 0.0) for x in X]
        R_est, mu_est, evals_est, ok = cube._principal_axes(min_points=10)
        self.assertTrue(ok)
        # First principal axis aligns with first column of Q up to sign
        v_true = Q[:, 0]
        v_est = R_est[:, 0]
        cosang = abs(float(v_true @ v_est) / (np.linalg.norm(v_true) * np.linalg.norm(v_est)))
        self.assertGreater(cosang, 0.98)

    def test_quad_cut_along_axis_recovers_minimum(self):
        # 1D quadratic along axis 0 with min at t* = -b/c inside bounds
        d = 2
        cube = QuadCube(bounds=[(-2.0, 2.0), (-1.0, 1.0)])
        cube.R = np.eye(d); cube.mu = np.zeros(d)
        rng = np.random.default_rng(3)
        t = rng.uniform(-2.0, 2.0, size=300)
        s = rng.uniform(-1.0, 1.0, size=300)
        b, c = 0.7, 1.4
        y = 0.5 + b * t + 0.5 * c * t ** 2 + 0.1 * rng.normal(size=t.shape)
        X = np.stack([t, s], axis=1)
        cube._tested_points = [x.copy() for x in X]
        cube._tested_pairs = [(x.copy(), float(val)) for x, val in zip(X, y)]
        R_loc, mu_loc, _, ok = cube._principal_axes()
        t_star = -b / c
        cut = cube._quad_cut_along_axis(0, R_loc if ok else cube.R, mu_loc if ok else cube.mu)
        self.assertAlmostEqual(cut, t_star, delta=0.15)

    def test_ucb_numeric(self):
        # Construct cube with known mean/var and compare ucb to manual formula
        c = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        c.R = np.eye(2); c.mu = np.zeros(2)
        c.prior_var = 0.25
        # Two scores to get non-zero var
        c.scores = [1.0, 3.0]
        c.n_trials = len(c.scores)
        c.mean_score = float(np.mean(c.scores))
        c.var_score = float(np.var(c.scores))
        beta = 0.2
        # No parent
        n_eff = c.n_trials
        mu = c.mean_score
        var = c.var_score
        base_expected = float(mu + beta * np.sqrt(var / (n_eff + 1e-8) + c.prior_var))
        u = c.ucb(beta=beta, lambda_geo=0.0)
        self.assertAlmostEqual(u, base_expected, places=6)
        # With parent influence
        p = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        p.scores = [2.0, 2.0, 4.0]
        p.n_trials = len(p.scores)
        p.mean_score = float(np.mean(p.scores))
        p.var_score = float(np.var(p.scores))
        c.parent = p
        # recompute expected with parent blending
        mu2 = 0.5 * p.mean_score + 0.5 * mu
        pv = p.var_score if p.var_score > 0 else p.prior_var
        var2 = 0.5 * pv + 0.5 * var
        base_expected2 = float(mu2 + beta * np.sqrt(var2 / (n_eff + 1e-8) + c.prior_var))
        u2 = c.ucb(beta=beta, lambda_geo=0.0)
        self.assertAlmostEqual(u2, base_expected2, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
