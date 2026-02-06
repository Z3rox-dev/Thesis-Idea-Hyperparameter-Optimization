import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestSplit4CurvatureRotated(unittest.TestCase):
    def test_split4_curvature_axes_with_rotated_frame(self):
        # 3D, symmetric bounds; PCA frame is a rotation (swap axes 0<->2)
        cube = QuadCube(bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)])
        R_rot = np.array([[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0]], dtype=float)  # swaps x<->z
        cube.R = np.eye(3)
        cube.mu = np.zeros(3)
        cube.depth = 2
        # Ensure we have some points so _principal_axes path is taken; we'll monkeypatch it
        cube._tested_points = [np.zeros(3) for _ in range(12)]
        cube._tested_pairs = [(np.zeros(3), 0.0) for _ in range(12)]
        # Force surrogate curvature with lambda preferring PC1,PC2 and PCA ok, with R=R_rot
        cube.surrogate_2d = {
            'w': np.zeros(6),
            'mu': np.zeros(3),
            'R': R_rot,
            'A_inv': np.eye(6),
            'sigma2': 0.1,
            'n': 12,
            'r2': 0.9,
            'pca_ok': True,
            'df_eff': 6.0,
            'H': np.array([[4.0, 0.0],[0.0, 3.0]]),
            'lambda': np.array([4.0, 3.0]),
        }
        # Make _principal_axes return the same rotated frame for consistency
        cube._principal_axes = lambda: (R_rot, np.zeros(3), np.array([4.0, 3.0, 1.0]), True)
        children = cube.split4()
        self.assertEqual(len(children), 4)
        # Check child centers in the surrogate (rotated) frame are at +/- width/4 along PC1,PC2 and ~0 on PC3
        parent_mu = cube.mu
        for ch in children:
            delta = ch.mu - parent_mu
            t = R_rot.T @ delta
            # Width per axis (all equal 4.0) so center shift magnitude should be ~1.0 (= width/4)
            self.assertTrue(np.isclose(abs(t[0]), 1.0, atol=1e-8))
            self.assertTrue(np.isclose(abs(t[1]), 1.0, atol=1e-8))
            self.assertTrue(np.isclose(t[2], 0.0, atol=1e-8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
