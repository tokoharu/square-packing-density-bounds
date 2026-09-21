"""Meaningful checks: independent geometry, LP dual sign, incremental consistency."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import shapely
from scipy.optimize import linprog
from geometry import Geometry
from master import Master
from pricing import scores, propose


class EngineTests(unittest.TestCase):
    def test_geometry_against_geos(self):
        rng = np.random.default_rng(726)
        lo = rng.uniform(.001, 2.3, (12, 2))
        rects = np.c_[lo, lo+rng.uniform(.001, .4, (12, 2))]
        model = Geometry(5.45, .9977, rects)
        poses = rng.random((50, 3)); poses[:2, 2] = [0, 1]
        numeric = model.matrix(poses)
        boxes = shapely.box(*model.full.T)
        expected = []
        local = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*model.B/2
        for p in poses:
            angle = p[2]*np.pi/4; c, s = np.cos(angle), np.sin(angle)
            center = model.L/2+p[:2]*(model.L-model.B*(c+s))/2
            vertices = local@np.array([[c,s],[-s,c]])+center
            a = shapely.area(shapely.intersection(shapely.Polygon(vertices), boxes))
            expected.append((a/model.areas).reshape(-1, 8).mean(axis=1))
        np.testing.assert_allclose(numeric, expected, atol=2e-12, rtol=1e-10)

    def test_incremental_master_matches_cold_and_dual_score(self):
        rng = np.random.default_rng(26)
        m = Master(5.45, .9977, [[.001,.001,5.449,5.449]])
        m.add_rows(rng.random((70, 3)))
        s0 = m.solve()
        rects = [[.8,.9,1.,1.7],[1.3,1.6,1.8,2.1],[.5,2.3,1.,2.7]]
        pricing = scores(m.L, m.B, rects, m.poses, s0.dual)
        direct = s0.dual@Geometry(m.L, m.B, rects).matrix(m.poses)
        np.testing.assert_allclose(pricing, direct, atol=1e-12)
        m.add_columns(rects); s1 = m.solve()
        self.assertLessEqual(s1.mass, s0.mass+1e-7)
        m.add_rows(rng.random((30, 3))); s2 = m.solve()
        self.assertGreaterEqual(s2.mass, s1.mass-1e-7)
        cold = linprog(np.ones(len(m.rectangles)), A_ub=-m.A,
                       b_ub=-np.full(len(m.poses), m.rhs), bounds=(0,None), method='highs-ds')
        self.assertTrue(cold.success)
        self.assertAlmostEqual(s2.mass, cold.fun, places=7)
        self.assertGreaterEqual(s2.dual.min(), 0)
        self.assertLess(s2.dual_violation, 1e-7)
        self.assertLess(abs(s2.duality_gap), 1e-6)
        np.testing.assert_allclose(m.A.toarray(), Geometry(m.L,m.B,m.rectangles).matrix(m.poses), atol=1e-13)
        self.assertEqual(m.add_rows(m.poses), 0)
        self.assertEqual(m.add_columns(m.rectangles), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'basis.json'; m.save_basis(path)
            restored = Master(m.L,m.B,m.rectangles,m.rhs)
            restored.add_rows(m.poses); restored.load_basis(path)
            self.assertAlmostEqual(restored.solve().mass,s2.mass,places=7)
            restored.add_rows([[.333,.444,.555]])
            with self.assertRaises(ValueError):
                restored.load_basis(path)

    def test_free_position_pricing_finds_improvement(self):
        m = Master(5.45, .9977, [[.001,.001,5.449,5.449]])
        m.add_rows([[1,1,0]])
        s = m.solve()
        candidates, _ = propose(m, s, samples_power=4, local_starts=0,
                                widths=(.25,), max_columns=2)
        self.assertTrue(candidates)
        self.assertGreater(candidates[0]['score'], 1.001)
        m.add_columns([x['rectangle'] for x in candidates])
        self.assertLess(m.solve().mass, s.mass-1)

    def test_invalid_geometry_and_infeasible_support(self):
        with self.assertRaises(ValueError):
            Geometry(5.45,.9977,[[1,1,1,2]])
        m = Master(5.45,.9977,[[2.5,2.5,2.6,2.6]])
        m.add_rows([[1,1,0]])
        with self.assertRaisesRegex(RuntimeError, 'not optimal'):
            m.solve()


if __name__ == '__main__':
    unittest.main()
