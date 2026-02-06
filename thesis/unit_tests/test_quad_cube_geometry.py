import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestQuadCubeGeometry(unittest.TestCase):
    def test_to_prime_to_original_inverse(self):
        d = 3
        cube = QuadCube(bounds=[(-1.0, 1.0)] * d)
        # Set explicit frame
        cube.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cube.mu = np.array([0.2, 0.4, 0.6], dtype=float)
        # Random point in original coords
        x = np.array([0.9, 0.1, 0.7], dtype=float)
        x_prime = cube.to_prime(x)
        x_back = cube.to_original(x_prime)
        np.testing.assert_allclose(x, x_back, atol=1e-12)

    def test_sample_uniform_prime_respects_bounds_even_if_inverted(self):
        # Include an inverted bound and ensure sampling respects the numeric interval
        cube = QuadCube(bounds=[(1.0, -1.0), (0.0, 2.0)])
        # Stabilize frame to avoid surprises
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        for _ in range(200):
            u = cube.sample_uniform_prime()
            self.assertGreaterEqual(u[0], -1.0)
            self.assertLessEqual(u[0], 1.0)
            self.assertGreaterEqual(u[1], 0.0)
            self.assertLessEqual(u[1], 2.0)

    def test_widths_are_absolute(self):
        cube = QuadCube(bounds=[(2.0, -1.0), (-3.0, 5.0)])
        w = cube._widths()
        np.testing.assert_allclose(w, np.array([3.0, 8.0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
