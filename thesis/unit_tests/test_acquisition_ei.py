import unittest
import numpy as np

from hpo_curvature import QuadHPO


class TestAcquisitionEI(unittest.TestCase):
    def test_surrogate_ei_prefers_stationary_point(self):
        d = 2
        hpo = QuadHPO(bounds=[(0.0, 1.0)] * d, rng_seed=0)
        cube = hpo.root
        # Disable auto tuning to keep fixed knobs
        hpo._auto_tune_surrogate = lambda cube: None
        hpo.surr_candidates = 2048
        hpo.surr_sigma2_max = 10.0
        hpo.surr_r2_min = 0.0
        hpo.surr_ei_min = 0.0
        hpo.surr_accept_margin_sigma = -1.0
        hpo.surr_mahalanobis_max = 3.0
        hpo.surr_ei_rho2_penalty = 0.0
        hpo.surr_elite_frac = 0.0
        hpo.surr_elite_top_k = 0
        # Root frame: R=I, mu ~ 0.5, bounds prime ~ [-0.5, 0.5]
        cube._ensure_frame()
        R = np.eye(d)
        # Surrogate center must match cube center so that training projections are around t, not 0.5+t
        mu = cube.mu.copy()
        # Target stationary point in surrogate frame (prime): t0 within [-0.5,0.5]^2
        t0 = np.array([0.1, -0.2])
        # Quadratic H positive definite
        H = np.array([[8.0, 0.0],[0.0, 4.0]], dtype=float)
        g = -H @ t0  # ensure stationary at t0
        # Build surrogate weights w = [c, b1, b2, a11, a22, a12]
        c0 = 0.0
        w = np.array([c0, g[0], g[1], H[0,0], H[1,1], H[0,1]], dtype=float)
        cube.surrogate_2d = {
            'w': w,
            'mu': mu,
            'R': R,
            'A_inv': np.eye(6),
            'sigma2': 0.05,
            'n': 20,
            'r2': 0.9,
            'pca_ok': True,
            'df_eff': 6.0,
            'H': H,
            'lambda': np.linalg.eigvalsh(H)[::-1],
        }
        # Provide some training pairs around t0 in original space: x = mu_c + R_c @ u ; here mu_c=root.mu ~0.5
        # Create a small cloud in surrogate frame around t0, map to original and attach (score values unused for EI ranking)
        mu_c = cube.mu.copy()
        R_c = cube.R.copy()
        pts = []
        scores = []
        rng = np.random.default_rng(0)
        for _ in range(30):
            t = t0 + 0.05 * rng.standard_normal(d)
            u = t.copy()  # since surrogate frame equals cube frame here
            x = (mu_c + R_c @ u).astype(float)
            pts.append(x)
            scores.append(float(-0.5 * (t @ (H @ t))))
        cube._tested_points = [p.copy() for p in pts]
        cube._tested_pairs = list(zip([p.copy() for p in pts], scores))
        # Sample with surrogate acquisition only
        x_sampled = hpo._sample_biased_in(cube, alpha=0.0, top_k=0)
        # Map sampled x back to surrogate frame t
        t_sampled = (R.T @ (x_sampled - mu_c)).astype(float)
        # Expect close to t0
        self.assertLess(np.linalg.norm(t_sampled - t0), 0.35)


if __name__ == '__main__':
    unittest.main(verbosity=2)
