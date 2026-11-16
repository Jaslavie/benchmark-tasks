import math
import unittest
from detent_bottle.support import table_support
from detent_bottle.scoring import Scorer


class SupportTests(unittest.TestCase):
    def test_resting_and_tilted_base(self):
        self.assertTrue(table_support((.2,0,.7475),1)['bottle_on_table'])
        tilt=math.radians(20)
        self.assertTrue(table_support((.2,0,.7475+.036*math.sin(tilt)),math.cos(tilt))['bottle_on_table'])

    def test_airborne_off_edge_and_invalid_geometry(self):
        for pos,dot in [((.2,0,.85),1),((1.,0,.7475),1),((.2,.7,.7475),1),((.2,0,.70),1),((float('nan'),0,.7475),1)]:
            with self.subTest(pos=pos):self.assertFalse(table_support(pos,dot)['bottle_on_table'])

    def test_airborne_opening_is_permanently_invalid(self):
        s=Scorer()
        for step in range(1,10):
            s.update(dict(step=step,time_s=step/20,lid_deg=90,loop_deg=180,upright_dot=1,
                bottle_z_m=.8,wrist_force_n=7,numerical_warning=False,bottle_on_table=step>1))
        self.assertEqual(s.score,0)
        self.assertEqual(s.failure,'bottle_left_table')

    def test_table_clearance_boundary(self):
        self.assertTrue(table_support((.2,0,.7575),1)["bottle_on_table"])
        self.assertFalse(table_support((.2,0,.757501),1)["bottle_on_table"])

    def test_support_flag_must_be_boolean(self):
        with self.assertRaises(ValueError):
            Scorer().update(dict(step=1,time_s=.05,lid_deg=0,loop_deg=0,upright_dot=1,
                bottle_z_m=.8,wrist_force_n=7,numerical_warning=False,bottle_on_table=1))


if __name__=='__main__':unittest.main()
