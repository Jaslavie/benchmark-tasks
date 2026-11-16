"""Robot-contact audit: press rocker while leaving safety loop raised.

This changes only the scripted robot trajectory, in this process. It does not
write bottle poses/joints, task physics, or the installed SDK. No model calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'yam'))
    import motion
    original=motion.plan
    def press_only(top=.7475):
        plans=original(top)
        plans['right']=[plans['right'][0],*plans['right'][-5:]]
        return plans
    motion.plan=press_only
    files=[root/'yam'/name for name in ('task.py','bottle_runtime.py','bottle_usd.py','motion.py')]
    files += [root/'detent_bottle'/name for name in ('scoring.py','bottle_model.py')]
    (out/'registration.json').write_text(json.dumps(dict(
        kind='negative robot-contact audit; no model calls',
        hypothesis='Raised safety loop blocks pressing the lower rocker from opening the lid',
        policy='yam_scripted with loop-folding waypoints omitted',seed=0,max_steps=320,
        expected_score=0,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
    os.environ['BOTTLE_TRACE']=str(out/'measurements.jsonl')
    sys.argv=['inspect-robots','run','--embodiment','yam_arena','--policy','yam_scripted',
        '-E','task='+str(root/'yam/task.py'),'-E','object_jitter=0','-E','control_hz=20',
        '-E','cam_width=320','-E','cam_height=176','--instruction',
        'Open the bottle and keep it upright on the table.','--scorer','task_score',
        '--max-steps','320','--seed','0','--store-frames','--log-dir',str(out)]
    from inspect_robots.cli import main as cli_main
    cli_main()


if __name__=='__main__':main()
