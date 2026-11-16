"""The score lines on hand-built states, with no simulator: python3 check_score.py. Runs `judge` from task.py, read from its source."""

import ast
import pathlib
import sys

src = pathlib.Path(__file__).with_name("task.py").read_text()
np = __import__("numpy")
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and any(n.id in ("GOOD_MISS", "FAIR_MISS", "RISE_MIN", "BOW_MAX", "REEL_R", "REEL_H", "SUCCESS_HOLD_S", "HOLD_STEPS") for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)):
        exec(ast.get_source_segment(src, node), ns)
    if isinstance(node, ast.FunctionDef) and node.name in ("judge", "release_outcome", "_on_reel"):
        exec("from __future__ import annotations\n" + ast.get_source_segment(src, node), ns)
judge, release_outcome = ns["judge"], ns["release_outcome"]
assert ns["HOLD_STEPS"] == round(ns["SUCCESS_HOLD_S"] * 20), "HOLD_STEPS must be SUCCESS_HOLD_S at 20 Hz"

# (snaps, miss at the near mark in metres, bowed, reel parked, fingers clear, still[, steps the full state has held]), wanted (full, score)
CASES = {
    "one snap, 2 mm off, parked, clear, still for 1 s": ((1, 0.002, False, True, True, True, 20), (True, 1.0)),
    "one snap, 2 mm off, parked, clear, still for 3 s": ((1, 0.002, False, True, True, True, 60), (True, 1.0)),
    "one snap, 2 mm off, parked, clear, still for 0.95 s": ((1, 0.002, False, True, True, True, 19), (True, 0.75)),
    "one snap, 2 mm off, reel still held": ((1, 0.002, False, False, False, True), (False, 0.75)),
    "one snap, 8 mm off, parked, clear": ((1, 0.008, False, True, True, True), (False, 0.5)),
    "one snap, 2 cm off": ((1, 0.02, False, True, True, True), (False, 0.0)),
    "two snaps": ((2, 0.002, False, True, True, True), (False, 0.0)),
    "no snap: line lowered gently, or released slack": ((0, 1.0, False, True, True, True), (False, 0.0)),
    "lifted sideways: a bowed print": ((1, 0.002, True, True, True, True), (False, 0.0)),
}
bad = 0
for name, (args, want) in CASES.items():
    full, score = judge(*args)
    good = (full, score) == want
    bad += not good
    print(f"{'ok ' if good else 'BAD'} {name}: full {full}, score {score}")
for name, args, want in (("taut, 6 cm up, straight", (True, 0.06, 0.002), "snap"), ("slack, 6 cm up: a bowed line that stays", (False, 0.06, 0.002), "bowed"), ("taut, 6 cm up, 2 cm sideways", (True, 0.06, 0.02), "bowed"), ("lowered gently, 1 cm up", (True, 0.01, 0.0), "none")):
    got = release_outcome(*args)
    bad += got != want
    print(f"{'ok ' if got == want else 'BAD'} {name}: {got}")


def fingers_at(x, y, z, half):
    """Fingertip points as `fingertips` lists them: two rows of six, a finger each side."""
    pts = [(x + sx * (half - 0.0), y + dy, z + dz) for sx in (-1.0, 1.0) for dy in (-0.018, 0.0, 0.018) for dz in (0.015, -0.015)]
    return np.array(pts)


reel = np.array([0.0, 0.0, 0.0])
_on_reel = ns["_on_reel"]
for name, f, want in (
    ("closed on the reel", fingers_at(0.0, 0.0, 0.025, 0.04), True),
    ("closed gripper hovering over the reel", fingers_at(0.0, 0.0, 0.12, 0.04), False),
    ("shut fingers beside the reel's middle", fingers_at(0.0, 0.0, 0.025, 0.01), False),
    ("wide open round the reel", fingers_at(0.0, 0.0, 0.025, 0.09), False),
):
    got = _on_reel(f, reel)
    bad += got != want
    print(f"{'ok ' if got == want else 'BAD'} {name}: {got}")
print("all good" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
