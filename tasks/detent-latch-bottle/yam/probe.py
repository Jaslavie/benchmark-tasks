"""Run a bounded diagnostic on the real YAM/PhysX stack; no model API calls."""
import argparse
import json
import os
from pathlib import Path
import time
import shutil
import hashlib
import subprocess
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--steps",type=int,default=100)
parser.add_argument("--mode",choices=["idle","demo","force"],default="idle")
args=parser.parse_args()
if not 1<=args.steps<=2400:
    parser.error("--steps must be between 1 and 2400")
if args.mode=="force" and args.steps<50:
    parser.error("force calibration needs at least 50 steps (60 recommended)")
args.out.mkdir(parents=True,exist_ok=False)
source=args.out/"source"
source.mkdir()
for path in Path(__file__).parent.glob("*.py"):
    shutil.copy2(path,source/path.name)
(args.out/"source-sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source.glob("*.py")},indent=2))
(args.out/"native-source-sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path(__file__).resolve().parents[1]/"detent_bottle").glob("*.py")},indent=2))
os.environ["BOTTLE_TRACE"]=str(args.out/"measurements.jsonl")
os.environ["BOTTLE_MODEL_REPORT"]=str(args.out/"model.json")

from yam_arena.inspect_embodiment import yam_arena_embodiment
from inspect_robots import Scene
from inspect_robots.types import Action
from PIL import Image

emb=yam_arena_embodiment(task=str(Path(__file__).with_name("task.py")),object_jitter=0,
    control_hz=20,cam_width=320,cam_height=176)
t0=time.monotonic()
try:
    obs=emb.reset(Scene(id="bottle-idle",instruction="Open the bottle and keep it upright on the table."),seed=42)
    import bottle_runtime
    target=np.array(obs.state["joint_pos"],copy=True)
    for name,frame in obs.images.items(): Image.fromarray(frame).save(args.out/f"initial-{name}.png")
    initial=bottle_runtime.measure(emb,step=0)
    if args.mode=="demo":
        from yam_arena.scripted import YamScriptedPolicy
        policy=YamScriptedPolicy()
        policy.bind(emb.info)
        policy.reset(Scene(id="bottle-demo",instruction="Open the bottle and keep it upright on the table."))
    frames={name:[] for name in obs.images}
    count=0
    force_samples={side:[] for side in ("left","right")}
    for i in range(args.steps):
        if args.mode=="force" and i in (10,40):
            import torch
            for side,load in (("left",10.),("right",20.)):
                robot=emb._base.scene[side+"_robot"]
                body=robot.find_bodies("gripper")[0]
                force=torch.zeros((1,1,3),device=emb._base.device)
                if i==10: force[:,:,2]=-load
                robot.permanent_wrench_composer.reset()
                robot.permanent_wrench_composer.add_forces_and_torques(
                    force,torch.zeros_like(force),body_ids=body,is_global=True)
        action=policy.act(obs).actions[0] if args.mode=="demo" else Action(data=target)
        result=emb.step(action); count=i+1;obs=result.observation
        for name,frame in obs.images.items(): frames[name].append(frame.copy())
        if args.mode=="force":
            measured=bottle_runtime.measure(emb,step=i+1)["wrist_forces_n"]
            for side in force_samples: force_samples[side].append(measured[side])
        if i%20==0:
            print(json.dumps(dict(tick=i+1,measurement=bottle_runtime.measure(emb,step=i+1),info=result.info)),flush=True)
            for name,frame in obs.images.items(): Image.fromarray(frame).save(args.out/f"latest-{name}.png")
        if result.terminated: break
    for name,frame in result.observation.images.items(): Image.fromarray(frame).save(args.out/f"final-{name}.png")
    for name,images in frames.items():
        shape=images[0].shape
        subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
            "-s",f"{shape[1]}x{shape[0]}","-r","20","-i","pipe:0","-an",
            "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",
            str(args.out/f"{name}.mp4")],input=np.stack(images).tobytes(),check=True)
    summary={"kind":args.mode+" diagnostic, not a model evaluation or full-score demonstration",
        "diagnostic_completed":True,
        "physics_dt_s":emb._base.physics_dt,"control_dt_s":emb._base.step_dt,
        "initial":initial,"final":bottle_runtime.measure(emb,step=count),"steps":count,
        "info":result.info,"terminated":result.terminated,"reason":result.termination_reason,
        "elapsed_seconds":time.monotonic()-t0,"camera_std":{k:float(v.std()) for k,v in result.observation.images.items()}}
    if args.mode=="demo":
        from yam_arena.runtime import tcp_world
        summary["final_tcp_world"]={side:tcp_world(emb._base.scene[side+"_robot"]).tolist()
            for side in ("left","right")}
        summary["worst_plan_ik"]={arm:{"position_mm":p.worst_ik_mm,"rotation_deg":p.worst_ik_deg}
            for arm,p in policy._plans.items()}
    if args.mode=="force":
        calibration={}
        for side,load in (("left",10.),("right",20.)):
            samples=force_samples[side]
            baseline=float(np.mean(samples[5:10]))
            loaded=float(np.mean(samples[35:40]))
            recovered=float(np.mean(samples[-5:]))
            calibration[side]={"applied_world_downward_force_n":load,
                "baseline_n":baseline,"loaded_n":loaded,"recovered_n":recovered,
                "increment_n":loaded-baseline,
                "passed":abs(loaded-baseline-load)<1.0 and abs(recovered-baseline)<1.0}
        summary["force_calibration"]={"per_arm":calibration,
            "passed":all(c["passed"] for c in calibration.values())}
    (args.out/"result.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
except BaseException as exc:
    (args.out/"error.json").write_text(json.dumps({"type":type(exc).__name__,"message":str(exc)},indent=2))
    raise
finally:
    emb.close()
