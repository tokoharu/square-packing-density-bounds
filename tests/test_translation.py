import unittest
import numpy as np
from master import Master
from translation import propose_translations


class TranslationTests(unittest.TestCase):
    def test_size_is_unchanged_and_stays_in_bounds(self):
        m = Master(5.45, .9977, [[.001, .001, 5.449, 5.449], [.5, .5, 1., .8]])
        m.add_rows([[1, 1, 0], [-1, -1, 0], [0, 0, .5]])
        s = m.solve()
        selected, ranking = propose_translations(m, s, max_parents=4)
        self.assertTrue(ranking)
        for rec in ranking:
            parent = np.asarray(rec['parent_rectangle'])
            moved = np.asarray(rec['rectangle'])
            np.testing.assert_allclose(moved[2:]-moved[:2], parent[2:]-parent[:2], atol=1e-9)
            self.assertTrue(np.all(moved >= -1e-9) and np.all(moved <= m.L+1e-9))
        for rec in selected:
            self.assertGreater(rec['advantage'], 1e-6)

    def test_selected_candidates_actually_improve_the_lp(self):
        m = Master(5.45, .9977, [[.001, .001, 5.449, 5.449]])
        m.add_rows([[1, 1, 0]])
        s = m.solve()
        m.add_columns([[.1, .1, .35, .35]])
        s = m.solve()
        selected, _ = propose_translations(m, s, max_parents=4)
        self.assertTrue(selected)
        before = s.mass
        m.add_columns([r['rectangle'] for r in selected])
        after = m.solve()
        self.assertLessEqual(after.mass, before+1e-7)

    def test_full_span_rectangle_has_nowhere_to_move(self):
        m = Master(5.45, .9977, [[.001, .001, 5.449, 5.449]])
        m.add_rows([[1, 1, 0]])
        s = m.solve()
        selected, ranking = propose_translations(m, s, max_parents=4)
        self.assertEqual(ranking, [])
        self.assertEqual(selected, [])

    def test_invalid_budget_rejected(self):
        m = Master(5.45, .9977, [[.001, .001, 5.449, 5.449]])
        m.add_rows([[1, 1, 0]])
        s = m.solve()
        with self.assertRaises(ValueError):
            propose_translations(m, s, max_parents=0)
        with self.assertRaises(ValueError):
            propose_translations(m, s, local_starts=0)


if __name__ == '__main__':
    unittest.main()
