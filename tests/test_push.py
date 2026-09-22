import json
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from push import certified, make_poses, rung_tag


class PushTests(unittest.TestCase):
    def test_rung_tag_never_contains_a_path_separator(self):
        # L arrives as a Fraction, so str() would render 5.7125 as "457/80";
        # using that in a directory name silently nests the run one level down.
        for L in (F('5.71') + F('0.0025'), F(1, 3), F('7.38'), F(29)):
            tag = rung_tag(L)
            self.assertNotIn('/', tag)
            self.assertNotIn('.', tag)
        self.assertEqual(rung_tag(F('5.71') + F('0.0025')), '5_7125')
        self.assertEqual(rung_tag(F('7.38')), '7_38')

    def test_certified_requires_both_flag_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertFalse(certified(d))  # nothing there at all

            (d / 'certified_candidate.json').write_text(json.dumps(
                {'globally_verified': False}))
            self.assertFalse(certified(d))  # flag says no

            (d / 'certified_candidate.json').write_text(json.dumps(
                {'globally_verified': True}))
            self.assertFalse(certified(d))  # no summary to corroborate it

            (d / 'verification_summary.json').write_text(json.dumps(
                {'status': 'INCOMPLETE'}))
            self.assertFalse(certified(d))  # summary disagrees

            (d / 'verification_summary.json').write_text(json.dumps(
                {'status': 'VERIFIED'}))
            self.assertTrue(certified(d))

    def test_generated_poses_are_in_the_normalised_pose_box(self):
        # geometry.matrix indexes (u, v, angle) with xy in [-1,1] and angle in
        # [0,1] and rejects anything outside, so that is what has to come out.
        L, B = 5.71, 0.9977
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'poses.npz'
            count = make_poses(L, B, out, angle_step=40, grid=4)
            poses = np.load(out)['poses']
        self.assertEqual(len(poses), count)
        self.assertEqual(poses.shape[1], 3)
        self.assertLessEqual(np.abs(poses[:, :2]).max(), 1.0)
        self.assertGreaterEqual(poses[:, 2].min(), 0.0)
        self.assertLessEqual(poses[:, 2].max(), 1.0)
        # a grid over the admissible centres must reach both extremes, and the
        # negative half is exactly what net_pose's clip would have thrown away
        self.assertAlmostEqual(poses[:, 0].min(), -1.0, places=9)
        self.assertAlmostEqual(poses[:, 0].max(), 1.0, places=9)
        self.assertTrue((poses[:, :2] < 0).any())
        # the angle coordinate covers the net from 0 up to pi/4
        self.assertAlmostEqual(poses[:, 2].min(), 0.0, places=9)
        self.assertGreater(poses[:, 2].max(), 0.9)


if __name__ == '__main__':
    unittest.main()
