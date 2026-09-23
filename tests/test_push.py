import json
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import push
from push import certified, make_poses, next_step, rung_tag, why


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


class WhyTests(unittest.TestCase):
    def test_reports_the_exception_from_a_traceback(self):
        tail = ('some progress output\n'
                'Traceback (most recent call last):\n'
                '  File "engine.py", line 1, in <module>\n'
                'ValueError: Pose=(normalized cx,cy,theta); xy in [-1,1]\n')
        self.assertEqual(
            why(tail), 'ValueError: Pose=(normalized cx,cy,theta); xy in [-1,1]')

    def test_ignores_trailing_blank_lines(self):
        self.assertEqual(why('first\nlast line\n\n   \n'), 'last line')

    def test_empty_output_is_not_an_error(self):
        self.assertEqual(why(''), '')
        self.assertEqual(why('\n  \n'), '')


class LogTests(unittest.TestCase):
    def test_records_go_to_the_file_not_only_stdout(self):
        # the run record for one n=29 rung was lost because the caller did not
        # redirect stdout; the file must not depend on that
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'push.jsonl'
            old = push.LOGFILE
            push.LOGFILE = path
            try:
                push.log(step='rung', L=5.72, status='ENGINE_FAILED')
                push.log(step='rung', L=5.7175, status='ACCEPTED')
            finally:
                push.LOGFILE = old
            rows = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows],
                         ['ENGINE_FAILED', 'ACCEPTED'])
        self.assertTrue(all('t' in r for r in rows))

    def test_logging_without_a_file_still_works(self):
        old = push.LOGFILE
        push.LOGFILE = None
        try:
            push.log(step='launch')          # must not raise
        finally:
            push.LOGFILE = old


class NextStepTests(unittest.TestCase):
    INIT = F(1, 400)

    def run_ladder(self, outcomes, widen_after=2):
        """Replay a sequence of rung outcomes; return the step after each."""
        step, streak, seen = self.INIT, 0, []
        for ok in outcomes:
            step, streak = next_step(step, self.INIT, streak, ok, widen_after)
            seen.append(step)
        return seen

    def test_failure_halves_the_step(self):
        self.assertEqual(self.run_ladder([False]), [F(1, 800)])
        self.assertEqual(self.run_ladder([False, False]), [F(1, 800), F(1, 1600)])

    def test_two_clean_rungs_widen_it_back(self):
        # this is the ratchet the n=29 run got stuck in: 1/1600 twice, and the
        # step still narrowed because nothing ever widened it
        self.assertEqual(self.run_ladder([False, False, True, True]),
                         [F(1, 800), F(1, 1600), F(1, 1600), F(1, 800)])

    def test_widening_never_exceeds_the_starting_step(self):
        after = self.run_ladder([False, True, True, True, True, True, True])
        self.assertEqual(after[-1], self.INIT)
        self.assertTrue(all(s <= self.INIT for s in after))

    def test_a_failure_resets_the_streak(self):
        # one clean rung, then a failure, then one more clean rung: the single
        # rung on each side of the failure must not add up to a widen
        self.assertEqual(self.run_ladder([False, True, False, True]),
                         [F(1, 800), F(1, 800), F(1, 1600), F(1, 1600)])

    def test_a_run_that_never_fails_keeps_the_initial_step(self):
        self.assertEqual(self.run_ladder([True] * 5), [self.INIT] * 5)

    def test_the_step_never_narrows_below_the_floor(self):
        floor = F(1, 3200)
        step, streak = self.INIT, 0
        for _ in range(10):
            step, streak = next_step(step, self.INIT, streak, False, 2, floor)
        self.assertEqual(step, floor)

    def test_scattered_failures_do_not_accumulate_into_a_stop(self):
        # the worry this replaces: with termination keyed to the total number of
        # failures, a ladder that keeps climbing still dies.  Interleave one
        # failure with two clean rungs, twelve times, and the step must stay put.
        floor = F(1, 12800)
        step, streak = self.INIT, 0
        consecutive, stopped = 0, False
        for i in range(36):
            ok = (i % 3) != 0
            consecutive = 0 if ok else consecutive + 1
            if consecutive >= 4:
                stopped = True
            step, streak = next_step(step, self.INIT, streak, ok, 2, floor)
        self.assertFalse(stopped)
        self.assertGreaterEqual(step, F(1, 800))

    def test_four_failures_in_a_row_is_what_stops_it(self):
        consecutive = 0
        for ok in (False, True, False, False, False, False):
            consecutive = 0 if ok else consecutive + 1
        self.assertGreaterEqual(consecutive, 4)


if __name__ == '__main__':
    unittest.main()
