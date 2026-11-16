"""CPU checks for the experimental YAM adapter, run in Arena's Python venv."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from yam_arena import ik,task_file
from yam_arena.constants import READY
from yam_arena.station import BASES


class ContractTests(unittest.TestCase):
    def test_task_loads_without_starting_isaac(self):
        task=task_file.load(Path(__file__).with_name("task.py"))
        self.assertEqual(task.names,("bottle",))
        self.assertEqual(task.free_objects,("bottle",))
        self.assertEqual(task.scene,"station")
        self.assertNotIn("isaacsim",sys.modules)
        self.assertNotIn("isaaclab.sim",sys.modules)

    def test_missing_force_measurement_cannot_award_opening_credit(self):
        import bottle_runtime
        from detent_bottle.scoring import Scorer
        previous=bottle_runtime._SCORER
        ready=bottle_runtime._FORCE_READY
        bottle_runtime._FORCE_READY=False
        try:
            bottle_runtime._SCORER=Scorer()
            for step in range(1,9):
                bottle_runtime._SCORER.update(dict(step=step,time_s=step/20,
                    lid_deg=90,loop_deg=180,upright_dot=1.,bottle_z_m=.80,
                    wrist_force_n=0.,numerical_warning=False))
            self.assertEqual(bottle_runtime._SCORER.score,1.)
            self.assertEqual(bottle_runtime.current_score(),0.)
            task=task_file.load(Path(__file__).with_name("task.py"))
            self.assertFalse(task.success({}))
            self.assertEqual(task.score({}),0.)
        finally:
            bottle_runtime._SCORER=previous
            bottle_runtime._FORCE_READY=ready

    def test_live_force_gate_allows_valid_held_opening(self):
        import bottle_runtime as runtime
        from detent_bottle.scoring import Scorer
        previous,ready=runtime._SCORER,runtime._FORCE_READY
        try:
            runtime._SCORER=Scorer();runtime._FORCE_READY=True
            for step in range(1,9):
                runtime._SCORER.update(dict(step=step,time_s=step/20,lid_deg=90.,loop_deg=180.,
                    upright_dot=1.,bottle_z_m=.8,wrist_force_n=7.,numerical_warning=False))
            self.assertEqual(runtime.current_score(),1.)
            runtime._FORCE_READY=False
            self.assertEqual(runtime.current_score(),0.)
        finally:
            runtime._SCORER,runtime._FORCE_READY=previous,ready

    def test_contact_waypoints_fit_the_yam_joint_limits(self):
        from motion import plan
        for arm,waypoints in plan().items():
            q=np.array(READY)
            for w in waypoints:
                q,p,r=ik.solve(w.pos,w.rotation,q,base_pos=BASES[arm],iters=200)
                with self.subTest(arm=arm,label=w.label):
                    self.assertLess(p,.005)
                    self.assertLess(r,math.radians(1))
                    self.assertTrue(np.all(q>=ik.LOWER))
                    self.assertTrue(np.all(q<=ik.UPPER))

    def test_tipping_terminates_the_immutable_step_result(self):
        import bottle_runtime as runtime
        from detent_bottle.scoring import Scorer
        from inspect_robots.types import StepResult
        class FakeEmb:
            task=SimpleNamespace(objects=[runtime.Bottle(name="bottle",pos=(0,0,.8),mass=.35)])
            def reset(self): pass
            def _task_step(self): return StepResult(observation=None,info={"score":0.})
        runtime.patch_embodiment(SimpleNamespace(YamArenaEmbodiment=FakeEmb))
        previous=runtime._SCORER
        runtime._SCORER=Scorer()
        row=dict(step=1,time_s=.05,lid_deg=90.,loop_deg=180.,upright_dot=.8,
                 bottle_z_m=.8,bottle_on_table=True,wrist_force_n=0.,wrist_force_measured=True,numerical_warning=False)
        try:
            with patch.object(runtime,"measure",return_value=row):
                result=FakeEmb()._task_step()
            self.assertTrue(result.terminated)
            self.assertEqual(result.termination_reason,"bottle_tipped")
            self.assertEqual(result.info["geometry_score_diagnostic"],0.)
        finally:
            runtime._SCORER=previous

    def test_missing_table_assessment_fails_closed(self):
        import bottle_runtime as runtime
        from detent_bottle.scoring import Scorer
        from inspect_robots.types import StepResult
        class FakeEmb:
            task=SimpleNamespace(objects=[runtime.Bottle(name="bottle",pos=(0,0,.8),mass=.35)])
            def reset(self): pass
            def _task_step(self): return StepResult(observation=None,info={"score":0.})
        runtime.patch_embodiment(SimpleNamespace(YamArenaEmbodiment=FakeEmb))
        previous,ready=runtime._SCORER,runtime._FORCE_READY
        runtime._SCORER=Scorer()
        row=dict(step=1,time_s=.05,lid_deg=90.,loop_deg=180.,upright_dot=1.,
                 bottle_z_m=.8,wrist_force_n=0.,wrist_force_measured=True,numerical_warning=False)
        try:
            with patch.object(runtime,"measure",return_value=row):
                result=FakeEmb()._task_step()
            self.assertTrue(result.terminated)
            self.assertEqual(result.termination_reason,"bottle_left_table")
            self.assertEqual(runtime.current_score(),0.)
        finally:
            runtime._SCORER,runtime._FORCE_READY=previous,ready

    def test_non_20hz_is_rejected_before_simulator_start(self):
        import bottle_runtime as runtime
        class FakeEmb:
            task=SimpleNamespace(objects=[runtime.Bottle(name="bottle",pos=(0,0,.8),mass=.35)])
            info=SimpleNamespace(control_hz=10.)
            def reset(self): raise AssertionError("wrong control rate must fail first")
            def _task_step(self): pass
        runtime.patch_embodiment(SimpleNamespace(YamArenaEmbodiment=FakeEmb))
        with self.assertRaisesRegex(ValueError,"control_hz=20"):
            FakeEmb().reset()

    def test_robot_nan_invalidates_measurements_even_if_bottle_is_finite(self):
        import torch
        import bottle_runtime as runtime
        class SceneDict(dict):
            sensors={}
            env_origins=torch.zeros(1,3)
        def robot():
            return SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,8),joint_vel=torch.zeros(1,8)))
        bottle=SimpleNamespace(joint_names=["lid_hinge","button_hinge","catch_hinge"],
            data=SimpleNamespace(root_link_pose_w=torch.tensor([[.2,0.,.7475,0.,0.,0.,1.]]),
                joint_pos=torch.zeros(1,3),joint_vel=torch.zeros(1,3)))
        scene=SceneDict(bottle=bottle,left_robot=robot(),right_robot=robot())
        emb=SimpleNamespace(_base=SimpleNamespace(scene=scene))
        self.assertFalse(runtime.measure(emb,step=0)["numerical_warning"])
        import json
        json.dumps(runtime.measure(emb,step=0))  # Live numpy scalars must serialize.
        scene["right_robot"].data.joint_pos[0,0]=float("nan")
        self.assertTrue(runtime.measure(emb,step=0)["numerical_warning"])


if __name__=="__main__": unittest.main()
