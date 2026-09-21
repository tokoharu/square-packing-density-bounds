import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from geometry import Geometry
from global_separation import net_pose, separate_global
from master import Master
from certify import prepare
from plot_solution import density_grid
from fast_geometry import overlap


class GlobalTests(unittest.TestCase):
    def test_last_angle_reflection(self):
        L,B=5.45,.9977;r=200;t=83*r/40000
        c,s=(1-t*t)/(1+t*t),2*t/(1+t*t)
        extent=(L-B*(c+s))/2;cx,cy=L/2+.7*extent,L/2+.2*extent
        model=Geometry(L,B,[[.9,1.4,1.,2.],[1.8,2.,2.,2.5]])
        pose=net_pose(r,cx,cy,L,B)
        expected=[]
        for k in range(2):
            expected.append(sum(overlap(model.full[8*k+j],cx,cy,c,s,B)/model.areas[8*k+j]/8 for j in range(8)))
        np.testing.assert_allclose(model.matrix(pose)[0],expected,atol=1e-12)

    def test_screen_budget_is_unresolved(self):
        m=Master(5.45,.9977,[[.001,.001,5.449,5.449]])
        m.add_rows(Geometry(m.L,m.B,m.rectangles).axis_poses())
        sol=m.solve()
        with patch('global_separation.verify_angle',return_value=(-1,1,0,1.,np.zeros(3),1,0.)):
            poses,report=separate_global(m,sol,budget=1,max_rows=1)
        self.assertEqual(report['status'],'UNRESOLVED')
        self.assertEqual(len(report['unknown_angles']),200)
        self.assertFalse(report['globally_verified'])
        self.assertEqual(len(poses),0)

    def test_exact_weights_and_plot_mass(self):
        data={'L':5.45,'B':.9977,'rectangles':[[.9,1.,1.,2.],[2.,2.,2.5,2.5]],'weights':[.123456789,2.75]}
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'candidate.json';source.write_text(json.dumps(data))
            meta=prepare(source,Path(tmp)/'certificate')
            self.assertFalse(meta['rescaled'])
            self.assertEqual(meta['mass_exact'],'2873456789/1000000000')
        edges,f=density_grid(data)
        self.assertAlmostEqual(float(np.diff(edges)@f@np.diff(edges)),sum(data['weights']),places=8)


if __name__=='__main__':unittest.main()
