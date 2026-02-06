import numpy as np
import unittest

from hpo_curvature import QuadCube

class TestCurvatureScoresConsistency(unittest.TestCase):
    def test_scores_scale_with_eigenvalues_and_widths(self):
        cube = QuadCube(bounds=[(-2,2),(-1,1)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        # Fake surrogate with known Hessian eigenvalues
        cube.surrogate_2d = {
            'w': np.zeros(6),
            'mu': np.zeros(2),
            'R': np.eye(2),
            'A_inv': np.eye(6),
            'sigma2': 0.1,
            'n': 20,
            'r2': 0.9,
            'pca_ok': True,
            'df_eff': 6.0,
            'H': np.diag([5.0, 2.0]),
            'lambda': np.array([5.0, 2.0])
        }
        s = cube._curvature_scores()
        self.assertIsNotNone(s)
        # Expected scaling: |lambda|^2 * h^4; width along axis 0 bigger so score[0] >> score[1]
        self.assertGreater(s[0], s[1]*2.0, f"Curvature score along axis 0 should dominate. Got {s}")

if __name__ == '__main__':
    unittest.main()
