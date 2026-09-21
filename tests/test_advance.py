import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
from fractions import Fraction as F
import numpy as np
from advance import transfer, load_incumbent, main
from geometry import Geometry


class AdvanceTests(unittest.TestCase):
    def test_constructive_fallback_and_transfer(self):
        parent={'L':5.45,'B':.9977,'rectangles':[[.9,1.4,1.,2.]],'weights':[2.]}
        initial=transfer(parent,5.451)
        self.assertEqual(initial['L'],5.451)
        np.testing.assert_allclose(initial['rectangles'][0],np.asarray(parent['rectangles'][0])*5.451/5.45)
        np.testing.assert_allclose(initial['rectangles'][1],parent['rectangles'][0])
        self.assertGreaterEqual(F(initial['transfer']['fallback_coverage_lower_bound']),F('1.001'))
        ids=initial['transfer']['fallback_index']
        model=Geometry(initial['L'],initial['B'],[initial['rectangles'][ids]])
        rng=np.random.default_rng(51);poses=rng.random((100,3))
        poses=np.vstack([poses,[[1,1,0],[1,1,1],[0,0,0]]])
        self.assertGreaterEqual(float((model.matrix(poses)*initial['weights'][ids]).min()),1.001)
        self.assertFalse(initial['globally_verified'])
        with self.assertRaises(ValueError):transfer(parent,5.45)

    def test_unverified_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fake.json'
            path.write_text(json.dumps({'L':5.45,'B':.9977,'weights':[1.],
                                        'rectangles':[[1.,1.,2.,2.]],'globally_verified':False}))
            with self.assertRaises(ValueError):load_incumbent(path,26)

    def test_failed_trial_preserves_best_and_resume_uses_state(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'run'
            argv=['advance.py','--out',str(out),'--step','0.001','--attempts','1']
            failure=subprocess.CalledProcessError(1,['engine.py'])
            with patch('sys.argv',argv),patch('advance.subprocess.run',side_effect=failure):main()
            state=json.loads((out/'advance_state.json').read_text())
            self.assertEqual(state['best_L'],5.45)
            self.assertEqual(F(state['next_step']),F('0.0005'))
            with patch('sys.argv',argv+['--resume']),patch('advance.subprocess.run',side_effect=failure):main()
            state=json.loads((out/'advance_state.json').read_text())
            self.assertEqual(state['best_L'],5.45)
            self.assertEqual(state['attempts'][1]['L'],5.4505)
            self.assertEqual(F(state['next_step']),F('0.00025'))


if __name__=='__main__':unittest.main()
