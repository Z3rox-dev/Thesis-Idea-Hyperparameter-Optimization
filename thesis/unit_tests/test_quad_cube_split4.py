import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestQuadCubeSplit4(unittest.TestCase):
    def test_split4_children_and_redistribution(self):
        # 2D symmetric bounds so midpoint cuts are at 0
        cube = QuadCube(bounds=[(-2.0, 2.0), (-2.0, 2.0)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        cube.depth = 2
        # Add four points at corners with dummy scores
        pts = [np.array([-1.0, -1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])]
        scores = [1.0, 2.0, 3.0, 4.0]
        cube._tested_points = [p.copy() for p in pts]
        cube._tested_pairs = list(zip([p.copy() for p in pts], scores))
        # Make PCA trivial and reliable; ensure curvature path doesn't interfere
        cube._principal_axes = lambda: (np.eye(2), np.zeros(2), np.ones(2), True)
        cube._curvature_scores = lambda: None

        children = cube.split4()
        self.assertEqual(len(children), 4)
        # Each child should have widths halved along both axes
        pw = np.array([abs(hi - lo) for (lo, hi) in cube.bounds], dtype=float)
        for ch in children:
            cw = np.array([abs(hi - lo) for (lo, hi) in ch.bounds], dtype=float)
            ratios = np.divide(cw, pw, out=np.ones_like(cw), where=pw > 0)
            np.testing.assert_allclose(ratios, np.array([0.5, 0.5]), atol=1e-8)
        # Redistribution: check each child got exactly one point
        self.assertListEqual(sorted(len(ch._tested_points) for ch in children), [1, 1, 1, 1])
        self.assertListEqual(sorted(len(ch._tested_pairs) for ch in children), [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
