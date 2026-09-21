import json
import math
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from point_export import convert, cdf, rescale_factor, run_validator, WDEN

class PointExportTests(unittest.TestCase):
    def data(self):
        return dict(L='4',B='9977/10000',rectangles=[['1','1','2','2']],weights=['3'])
    def test_cdf_triangle(self):
        for x,w in [(F(0),F(1,8)),(F(1,2),F(1,2)),(F(1),F(7,8))]:
            self.assertEqual(cdf(x,F(0),F(1),F(1,2)),w)
    def test_mass_symmetry_and_spacing(self):
        for eps in (F(0),F(1,20)):
            cert,meta=convert(self.data(),F(3,10),eps,rescale=False)
            atoms={(F(x),F(y)):F(w) for x,y,w in cert['atoms']}
            self.assertEqual(sum(atoms.values()),3)
            self.assertFalse(meta['rescaled'])
            self.assertLessEqual(F(meta['actual_spacing']),F(3,10))
            for (x,y),w in atoms.items():
                self.assertEqual(atoms[y,x],w)
                self.assertEqual(atoms[4-x,y],w)
    def test_overlapping_orbit_duplicates_add(self):
        d=self.data();d['rectangles']=[['1','1','3','3']]
        c,_=convert(d,F(1),F(0),rescale=False)
        self.assertEqual(len(c['atoms']),4)
        self.assertTrue(all(F(w)==F(3,4) for x,y,w in c['atoms']))
    def test_rescale_boosts_every_weight_uniformly_below_n(self):
        raw,_=convert(self.data(),F(3,10),rescale=False)
        raw_atoms={(F(x),F(y)):F(w) for x,y,w in raw['atoms']}
        cert,meta=convert(self.data(),F(3,10),n=10)
        atoms={(F(x),F(y)):F(w) for x,y,w in cert['atoms']}
        self.assertTrue(meta['rescaled'])
        factor=F(meta['rescale_factor'])
        self.assertGreater(factor,1)
        self.assertEqual(set(atoms),set(raw_atoms))
        for key,w in raw_atoms.items():
            self.assertEqual(atoms[key],w*factor)
        total=sum(atoms.values())
        self.assertEqual(total,F(meta['mass_exact']))
        self.assertLess(total,10)
        checked=sum(F(math.ceil(float(w)*WDEN),WDEN) for w in atoms.values())
        self.assertLess(checked,10)
        self.assertEqual(checked,F(meta['validator_rounded_mass_exact']))
    def test_rescale_to_exact_target(self):
        cert,meta=convert(self.data(),F(3,10),rescale_to=F(5))
        self.assertEqual(sum(F(w) for x,y,w in cert['atoms']),5)
        self.assertEqual(meta['rescale_factor'],str(F(5,3)))
    def test_rescale_to_below_current_mass_rejected(self):
        with self.assertRaises(ValueError):
            convert(self.data(),F(3,10),rescale_to=F(1))
    def test_rescale_factor_has_no_headroom_left(self):
        self.assertEqual(rescale_factor(F(26),0,26),F(1))
    def test_bad_inputs(self):
        for spacing in (0,-1):
            with self.assertRaises(ValueError): convert(self.data(),spacing)
        with self.assertRaises(ValueError): convert(self.data(),F(1,1000),max_cells=100)
        d=self.data();d['weights']=[-1]
        with self.assertRaises(ValueError): convert(d,F(1,4))
    def test_original_exit_zero_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'points.json';v=Path(tmp)/'checker.py'
            p.write_text(json.dumps({'B':'9977/10000'}))
            v.write_text("print('NOT VERIFIED')\n")
            self.assertEqual(run_validator(p,v,10)['status'],'NOT_VERIFIED')
if __name__=='__main__': unittest.main()
