"""Embodiment ``yam_weld_fitup``: the stock bimanual YAM station with two bevelled steel coupons on a welding table.

Robot, cameras, 14-D joint action and observations are the stock adapter's (``yam_arena``), unchanged. This module
adds the two coupons as loose rigid bodies (custom geometry, which the task-file format can't express) and scores the
fit-up from their measured poses every control step. There are no joints.

Scoring (``design.grade``): the step's credit is 1.0 for the complete fit-up, 0.5 or 0.25 for partial ones. The trial
ends with success once the complete fit-up has held for 1 s; until then ``info["score"]`` reports the complete
fit-up as 0.9, so a trial that runs out of time just as the coupons settle doesn't get full marks (the official
``task_score`` takes the last step's score when success hasn't fired).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import design as d
from .mechanism import MechanismEmbodiment

TASK_FILE = Path(__file__).resolve().parents[1] / "weld_fitup_task.py"
HOLD_S = 1.0
COMPLETE_SCORE = 0.9


class WeldFitupEmbodiment(MechanismEmbodiment):
    ENV_NAME = "yam_weld_fitup"
    TASK_FILE = TASK_FILE

    def builders(self):
        from .assets import BUILDERS

        return BUILDERS

    def layout(self):
        return d.start_poses()

    def make_joints(self, js, origin):
        pass  # loose parts only

    def on_reset(self):
        self._held = 0

    def measure(self):
        p = self.poses()
        still = {}
        for n in d.COUPONS:
            v = self.velocity(n)
            still[n] = bool(np.linalg.norm(v[:3]) < d.STILL_MPS and np.linalg.norm(v[3:]) < d.STILL_RADPS)
        credit, reason = d.grade(p, still)
        self._held = self._held + 1 if credit >= 1.0 else 0
        success = self._held >= max(1, round(HOLD_S * self.info.control_hz))
        score = min(credit, COMPLETE_SCORE)
        info = {"score": score, "credit": credit, "reason": reason, "success": success,
                "still": [still[n] for n in d.COUPONS]}
        for n in d.COUPONS:
            info[f"{n}_pose"] = p[n]
            info[f"{n}_heading_deg"] = d.heading_deg(p[n])
            info[f"{n}_tilt_deg"] = d.tilt_deg(p[n])
        if d.faces(p["coupon_a"], p["coupon_b"]) and d.faces(p["coupon_b"], p["coupon_a"]):
            j = d.joint(p["coupon_a"], p["coupon_b"])
            info.update({k: v * 1000.0 if k != "parallel_deg" else v for k, v in j.items()})
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            a, b = p["coupon_a"], p["coupon_b"]
            extra = (f" open {info['open_min']:.2f}-{info['open_max']:.2f} mm stagger {info['stagger']:+.2f} mm"
                     if "open_min" in info else "")
            print(f"[fitup] t={self._t:6.2f} A=({a[0] * 100:5.2f},{a[1] * 100:5.2f}) {d.heading_deg(a):+7.2f}deg "
                  f"B=({b[0] * 100:5.2f},{b[1] * 100:5.2f}) {d.heading_deg(b):+7.2f}deg"
                  f"{extra} credit={credit:.2f} {reason}", flush=True)
        return info, success
