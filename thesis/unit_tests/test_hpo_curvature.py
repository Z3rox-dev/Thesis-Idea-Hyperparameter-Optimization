import types
import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestQuadCubeAxisSelection(unittest.TestCase):
    def make_cube(self, bounds):
        c = QuadCube(bounds=list(bounds))
        d = len(bounds)
        # Set an explicit frame to avoid defaults changing during _ensure_frame
        c.R = np.eye(d)
        c.mu = np.zeros(d, dtype=float)
        # Minimal history so split can proceed; keep <6 to force midpoint cut path
        c._tested_points = []
        c._tested_pairs = []
        return c

    def chosen_axis_after_split2(self, parent, children):
        pw = np.array([abs(hi - lo) for (lo, hi) in parent.bounds], dtype=float)
        cw = np.array([abs(hi - lo) for (lo, hi) in children[0].bounds], dtype=float)
        # The chosen axis is the one whose child width is half of the parent (others unchanged)
        ratios = np.divide(cw, pw, out=np.ones_like(cw), where=pw>0)
        # Tolerate floating rounding
        diffs = np.abs(ratios - 0.5)
        ax = int(np.argmin(diffs))
        self.assertTrue(np.isclose(ratios[ax], 0.5, atol=1e-6), f"No axis halved: ratios={ratios}")
        return ax

    def test_split2_curvature_prefers_pc1(self):
        # 3D cube, equal widths; force curvature to favor axis 0 (PC1)
        cube = self.make_cube([(-2, 2), (-2, 2), (-2, 2)])
        cube.depth = 2
        cube._curvature_scores = types.MethodType(lambda self: np.array([10.0, 1.0]), cube)
        # Patch principal axes to return a stable frame
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), True), cube)
        children = cube.split2()
        ax = self.chosen_axis_after_split2(cube, children)
        self.assertEqual(ax, 0, f"Expected split along PC1 (0), got {ax}")

    def test_split2_curvature_prefers_pc2(self):
        # 3D cube, equal widths; force curvature to favor axis 1 (PC2)
        cube = self.make_cube([(-2, 2), (-2, 2), (-2, 2)])
        cube.depth = 2
        cube._curvature_scores = types.MethodType(lambda self: np.array([1.0, 10.0]), cube)
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), True), cube)
        children = cube.split2()
        ax = self.chosen_axis_after_split2(cube, children)
        self.assertEqual(ax, 1, f"Expected split along PC2 (1), got {ax}")

    def test_split2_pca_projected_width_when_curvature_none(self):
        # Anisotropic parent widths; set frames so PC2 sees largest projected span
        # Parent prime widths: [4, 2, 2]
        cube = self.make_cube([(-2, 2), (-1, 1), (-1, 1)])
        cube.depth = 2
        # Set cube frame (R) as a permutation swapping axes 0 and 1
        cube.R = np.array([[0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]], dtype=float)
        cube.mu = np.zeros(3, dtype=float)
        # Curvature disabled
        cube._curvature_scores = types.MethodType(lambda self: None, cube)
        # PCA returns identity frame as "reliable"
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), True), cube)
        children = cube.split2()
        ax = self.chosen_axis_after_split2(cube, children)
        # Projected span h = |I^T @ R| @ w = |R| @ [4,2,2] => [2,4,2] => argmax = 1
        self.assertEqual(ax, 1, f"Expected split along projected widest axis PC2 (1), got {ax}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
