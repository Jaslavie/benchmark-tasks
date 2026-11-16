"""Boundary, history, input and native-physics checks for the submission scorer."""
import unittest
from task import Scorer


def row(step, **changes):
    sample = dict(step=step, time_s=step/20, lid_deg=0., loop_deg=0., upright_dot=1.,
                  bottle_z_m=.8005, wrist_force_n=0., numerical_warning=False)
    sample.update(changes)
    return sample


class ScoringTests(unittest.TestCase):
    def test_partial_and_full_credit(self):
        for lid, loop, expected in [(0, 0, 0), (0, 150, .25), (15, 150, .25),
                                    (15.001, 150, .5), (60, 150, .5), (60.001, 150, 1),
                                    (90, 149.99, 1), (0, 149.99, 0)]:
            with self.subTest(lid=lid, loop=loop):
                s = Scorer()
                for step in range(1, 8):
                    self.assertEqual(s.update(row(step, lid_deg=lid, loop_deg=loop)), 0)
                self.assertEqual(s.update(row(8, lid_deg=lid, loop_deg=loop)), expected)

    def test_hold_is_consecutive_and_score_is_current(self):
        s = Scorer()
        for step in range(1, 8):
            s.update(row(step, lid_deg=90, loop_deg=180))
        s.update(row(8))
        self.assertEqual(s.update(row(9, lid_deg=90, loop_deg=180)), 0)
        for step in range(10, 17):
            s.update(row(step, lid_deg=90, loop_deg=180))
        self.assertEqual(s.score, 1)
        self.assertEqual(s.update(row(17)), 0)

    def test_tipping_invalidates_even_after_recovery(self):
        s = Scorer()
        s.update(row(1, upright_dot=.9))
        for step in range(2, 15):
            s.update(row(step, lid_deg=90, loop_deg=180))
        self.assertEqual(s.score, 0)
        self.assertEqual(s.failure, "bottle_tipped")

    def test_force_requires_four_consecutive_ticks(self):
        s = Scorer()
        for step in range(1, 4):
            s.update(row(step, wrist_force_n=101))
        s.update(row(4, wrist_force_n=100))
        self.assertIsNone(s.failure)
        for step in range(5, 9):
            s.update(row(step, wrist_force_n=101))
        self.assertEqual(s.failure, "wrist_force_limit")

    def test_bad_traces_rejected(self):
        for bad in [row(2), row(1, time_s=1), row(1, lid_deg=float("nan")),
                    row(1, numerical_warning=0), row(1, wrist_force_n=-1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Scorer().update(bad)

    def test_numerical_warning_and_fall_veto_credit(self):
        for changes in [dict(numerical_warning=True), dict(bottle_z_m=.78)]:
            s = Scorer()
            s.update(row(1, **changes))
            for step in range(2, 10):
                s.update(row(step, lid_deg=90, loop_deg=180))
            self.assertEqual(s.score, 0)


class NativeTests(unittest.TestCase):
    def test_model_and_idle(self):
        import numpy as np
        from detent_bottle.environment import BottleEnv
        e = BottleEnv(max_steps=40)
        try:
            self.assertEqual(e.model.nu, 18)
            self.assertEqual(e.model.neq, 0)
            self.assertEqual(e.model.npair, 294)
            for aid in range(e.model.nu):
                joint = e.model.joint(int(e.model.actuator_trnid[aid, 0])).name
                self.assertTrue(joint.startswith(("left_", "right_")), joint)
            with self.assertRaises(ValueError):
                e.action([float("nan")] * 14)
            with self.assertRaises(ValueError):
                e.target(0, [0, 0, .1], [0, 0, 0], .05)
            for _ in range(40):
                e.step()
            self.assertEqual(e.scorer.score, 0)
            self.assertEqual(e.reason, "time_limit")
            force = e.data.qfrc_applied.copy()
            force[e.catch_d] = 0
            self.assertFalse(np.any(force))
            self.assertFalse(np.any(e.data.xfrc_applied))
            self.assertLess(abs(e.mechanism()["lid_deg"]), 5)
            e.data.joint('lid_hinge').qpos[0] = float('nan')
            invalid = e.measurements()
            invalid.update(step=1, time_s=.05)
            scorer = Scorer()
            self.assertEqual(scorer.update(invalid), 0)
            self.assertEqual(scorer.failure, 'numerical_instability')
        finally:
            e.close()


if __name__ == "__main__":
    unittest.main()
