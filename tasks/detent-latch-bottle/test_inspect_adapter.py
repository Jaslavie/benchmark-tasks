"""Real Inspect Robots integration, without a network policy or model call."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("inspect_robots"), "Install the optional Inspect Robots dependency")
class AdapterTests(unittest.TestCase):
    def test_compatibility_observations_and_rollout(self):
        import numpy as np
        from inspect_robots import eval
        from inspect_robots.compat import check_compatibility
        from inspect_robots.task import Task
        from detent_bottle.inspect_adapter import make_task, make_embodiment, make_demo_policy
        task = make_task()
        robot = make_embodiment(64)
        policy = make_demo_policy()
        try:
            self.assertTrue(check_compatibility(policy, robot, task).ok)
            observation = robot.reset(task.scenes[0])
            self.assertEqual(set(observation.state),
                             {"eef_state", "joint_pos", "joint_vel", "gripper_width", "wrist_wrench"})
            self.assertEqual(dict(observation.extra), {})
            self.assertEqual(set(observation.images), {"overview", "closeup", "front"})
            self.assertTrue(all(im.shape == (64, 64, 3) for im in observation.images.values()))
            qpos = robot.env.data.qpos.copy()
            simtime = robot.env.data.time
            robot.observe()
            np.testing.assert_array_equal(robot.env.data.qpos, qpos)
            self.assertEqual(robot.env.data.time, simtime)
            policy.reset(task.scenes[0])
            result = robot.step(policy.act(observation).actions[0])
            self.assertEqual(result.info["measurements"]["step"], 1)
            self.assertNotIn("measurements", result.observation.extra)
            short = Task(name="adapter-smoke", scenes=task.scenes, scorer=task.scorer, max_steps=3)
            with tempfile.TemporaryDirectory() as output:
                log, = eval(short, make_demo_policy(), robot, log_dir=output, store_frames=True)
                self.assertEqual(log.status, "success")
                self.assertTrue(list(Path(output).rglob("*.json")))
                self.assertTrue(all(value == 0 for value in log.results.metrics.values()))
        finally:
            robot.close()


if __name__ == "__main__":
    unittest.main()
