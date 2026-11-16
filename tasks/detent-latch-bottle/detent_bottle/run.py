"""Run or exactly replay a trial; model inference is a separate Inspect Robots command."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .environment import BottleEnv
from .reference import opening_sequence
from .scoring import MAX_STEPS, score_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("demo", "idle"), default="demo")
    parser.add_argument("--actions", type=Path, help="Replay actions.jsonl instead of a controller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--record", action="store_true", help="Write setup.png and H.264 demo.mp4")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error("--max-steps must be in 1..2400")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = BottleEnv(max_steps=args.max_steps)
    views = writer = None
    poses = [env.data.qpos.copy()]
    source_actions = None
    try:
        if args.record:
            import imageio.v2 as imageio
            from PIL import Image
            from .rendering import Views
            views = Views(env)
            image = views.composite()
            Image.fromarray(image).save(args.output / "setup.png")
            writer = imageio.get_writer(str(args.output / "demo.mp4"), fps=10,
                                       codec="libx264", pixelformat="yuv420p", quality=8,
                                       ffmpeg_params=["-movflags", "+faststart", "-threads", "2"])
            writer.append_data(image)
        if args.actions:
            source_actions = [json.loads(s)["action"] for s in args.actions.read_text().splitlines()]
            stages = [("action replay", None, None, None, None, len(source_actions))]
        elif args.controller == "idle":
            stages = [("idle diagnostic (not a model run)", None, None, None, None, args.max_steps)]
        else:
            stages = opening_sequence()
        with (args.output / "actions.jsonl").open("w") as actions, (args.output / "measurements.jsonl").open("w") as measurements:
            for stage, arm, position, rpy, gap, ticks in stages:
                if arm is not None:
                    env.target(arm, position, rpy, gap)
                for _ in range(ticks):
                    if env.reason:
                        break
                    action = source_actions[env.steps] if source_actions is not None else env.targets_vector()
                    env.action(action)
                    actions.write(json.dumps(dict(step=env.steps, action=np.asarray(action).tolist())) + "\n")
                    measurements.write(json.dumps(env.measurements()) + "\n")
                    poses.append(env.data.qpos.copy())
                    if writer and env.steps % 2 == 0:
                        writer.append_data(views.composite())
                print(json.dumps(dict(stage=stage, step=env.steps, score=env.scorer.score,
                                      mechanism=env.mechanism(), reason=env.reason)), flush=True)
                if env.reason:
                    break
        if writer and env.steps % 2:
            writer.append_data(views.composite())
        if views:
            Image.fromarray(views.composite()).save(args.output / "final.png")
        result = env.scorer.result()
        result.update(termination_reason=env.reason or "controller_finished",
                      controller="action replay" if args.actions else "scripted known-geometry demo" if args.controller == "demo" else "idle diagnostic; not a model run",
                      wall_seconds=round(time.perf_counter() - started, 3),
                      control_hz=20, physics_timestep_s=float(env.model.opt.timestep),
                      final_mechanism=env.mechanism(),
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))})
        rescored = score_trace(args.output / "measurements.jsonl")
        assert rescored["score"] == result["score"]
        np.savez_compressed(args.output / "trajectory.npz", qpos=poses)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if args.controller == "demo" and not args.actions and not result["success"]:
            raise SystemExit("The scripted controller did not reach full score")
    finally:
        if writer:
            writer.close()
        if views:
            views.close()
        env.close()


if __name__ == "__main__":
    main()
