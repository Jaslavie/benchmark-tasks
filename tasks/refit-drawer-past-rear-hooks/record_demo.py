"""Play the task file's scripted demo() in the yam_arena embodiment and record it.

    python record_demo.py task.py out_dir [--max-steps 1200] [--no-record]

The same plan the ``yam_scripted`` policy plays (open loop, from the fixed start layout), stepped through the same
embodiment, so the score printed is the live ``score()``/``success()`` of the task file. Writes one H.264 MP4 per
camera, setup.png (the first top-camera frame) and result.json. DEMO_VARIANT picks the demo's variant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int, record: bool, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena import station
    from yam_arena.constants import READY
    from yam_arena.demo import ArmPlan
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=320, cam_height=176)
    task = emb.task.module
    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    plans = {arm: ArmPlan(list(wps), station.BASES[arm]) for arm, wps in task.demo().items()}
    end = max((p.duration for p in plans.values()), default=0.0)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, success, info = 0, False, {}
    while steps < max_steps:
        t = (steps + 1) / CONTROL_HZ
        per_arm = [plans[a].action7(t)[0] if a in plans else np.r_[READY, 1.0] for a in ("left", "right")]
        res = emb.step(Action(data=pack(*per_arm)))
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 20 == 0:
            from yam_arena.scene import object_states

            task._update(object_states(emb._env, emb.task.names))
            st = task._S.get("status", {})
            labels = {a: p.target(t)[3] for a, p in plans.items()}
            print(f"step {steps} score {info.get('score')} {labels} height_mm {1000 * st['height']:.1f} "
                  f"tilt {st['tilt']:.2f} pitch {st['pitch']:.2f} yaw {st['yaw']:.2f} opened_mm {1000 * st['opened']:.1f} "
                  f"y_mm {1000 * st.get('y', 0):.1f} box {np.round(1000 * st['box_local'], 0)} on_slides {st['on_slides']} "
                  f"clear {st['clear']} closed_free {st['closed_free']} lost {st['lost']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if t > end + hold_s:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {a: round(p.worst_ik_mm, 2) for a, p in plans.items()}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
