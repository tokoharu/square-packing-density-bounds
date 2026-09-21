import unittest
import numpy as np
from geometry import Geometry
from refinement import bisect_long, propose_splits


class RefinementTests(unittest.TestCase):
    def test_area_weighted_children_reproduce_parent(self):
        rng=np.random.default_rng(627)
        poses=rng.random((100,3));poses=np.vstack([poses,[[0,0,0],[1,1,1]]])
        for r in [[.9,.5,1.,2.],[1.,1.9,2.5,1.95],[.3,.7,2.4,1.1]]:
            children,fractions,axis=bisect_long(r)
            original=Geometry(5.451,.9977,[r]).matrix(poses)[:,0]
            refined=Geometry(5.451,.9977,children).matrix(poses)@fractions
            np.testing.assert_allclose(original,refined,rtol=2e-11,atol=2e-12)
            self.assertAlmostEqual(float(fractions.sum()),1)
            self.assertEqual(axis,int(np.argmax(np.asarray(r[2:])-r[:2])))

    def test_short_dimension_is_unchanged(self):
        r=np.array([.9,.5,1.,2.]);children,_,_=bisect_long(r)
        np.testing.assert_array_equal(children[:,[0,2]],np.tile(r[[0,2]],(2,1)))
        self.assertEqual(children[0,3],children[1,1])

    def test_forced_axis_overrides_the_default_long_choice(self):
        rng=np.random.default_rng(627)
        poses=rng.random((100,3));poses=np.vstack([poses,[[0,0,0],[1,1,1]]])
        r=[.9,.5,1.,2.]  # lengths (x,y) = (0.1, 1.5): long side is y (axis 1)
        children,fractions,axis=bisect_long(r,axis=0)
        self.assertEqual(axis,0)
        original=Geometry(5.451,.9977,[r]).matrix(poses)[:,0]
        refined=Geometry(5.451,.9977,children).matrix(poses)@fractions
        np.testing.assert_allclose(original,refined,rtol=2e-11,atol=2e-12)
        np.testing.assert_array_equal(children[:,[1,3]],np.tile(np.asarray(r)[[1,3]],(2,1)))

    def test_forced_axis_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            bisect_long([.9,.5,1.,2.],axis=2)

    def test_propose_splits_rejects_unknown_split_axis(self):
        with self.assertRaises(ValueError):
            propose_splits(None,None,split_axis='sideways')


if __name__=='__main__':unittest.main()
