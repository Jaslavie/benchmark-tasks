"""Every score line of task.py on hand-built states, with no simulator. Run it with the harness's Python, from this folder: python check_scorer.py"""

from __future__ import annotations

import numpy as np

import mechkit as m
import task as T

POS = np.array([*T.PIVOT, T.TOP])
FAR = np.full((24, 3), 2.0)
DT = 0.05  # a control step at 20 Hz


def run(path, seed=None, tips=FAR, pos=POS, tilt=0.0):
    m._SEED[0] = seed
    t = T.Trial()
    for k, bend in enumerate(path):
        t.update(float(bend), pos, tilt, tips, k * DT, DT)
    return t


def pull(*peaks, rest=30):
    """To each peak and back to rest, in steps, then this many steps at rest: 30 is 1.5 s."""
    out = [0.0]
    for p in peaks:
        out += list(np.linspace(out[-1], p, 30)) + list(np.linspace(p, 0.0, 30))
    return out + [0.0] * rest


CASES = [
    ("arms wait", [0.0] * 20, None, 0.0, False),
    ("drawn: pulls to 90 and relaxes: sets at 78", pull(90.0), None, 0.25, False),
    ("drawn: pulls to 102: sets at 90: full", pull(102.0), None, 1.0, True),
    ("drawn: in two pulls, 85 then 102: full", pull(85.0, 102.0), None, 1.0, True),
    ("drawn: 100, sets at 88: full", pull(100.0), None, 1.0, True),
    ("drawn: 102, at rest for 0.5 s only: in the band, but not yet still for 1 s", pull(102.0, rest=10), None, 0.0, True),
    ("drawn: 106, at rest for 0.5 s only: not yet still for 1 s", pull(106.0, rest=10), None, 0.0, False),
    ("drawn: 106, sets at 94: over the band", pull(106.0), None, 0.5, False),
    ("drawn: 109, sets at 97: past 96", pull(109.0), None, 0.0, False),
    ("annealed: fixed overbend for drawn, 102: sets at 98: past 96", pull(102.0), 1, 0.0, False),
    ("annealed: pulls to 94: sets at 90: full", pull(94.0), 1, 1.0, True),
    ("drawn: fixed overbend for annealed, 94: sets at 82", pull(94.0), None, 0.5, False),
    ("a fixed 98: drawn sets at 86, under the band", pull(98.0), None, 0.5, False),
    ("a fixed 98: annealed sets at 94, over the band", pull(98.0), 1, 0.5, False),
    ("under 70 scores nothing", pull(70.0), None, 0.0, False),
    ("never relaxed: lever held at 102", list(np.linspace(0, 102, 40)) + [102.0] * 20, None, 0.0, False),
]


def main() -> None:
    bad = 0
    for name, path, seed, want, want_full in CASES:
        t = run(path, seed)
        ok = t.full == want_full and (want is None or abs(t.score - want) < 1e-6)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: score {t.score:.3f}, full {t.full}, set {t.set_angle:.1f}, tube {t.tube:.1f}")
    near = np.tile(np.array([[0.0, T.GRIP_AT, T.BAR_Z + 0.02]]), (24, 1))
    t = run(pull(102.0), tips=near)
    ok = not t.full and t.score == 0.0
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} a gripper 2 cm above the form handle: score {t.score:.3f}, full {t.full}")
    near = np.tile(np.array([[0.0, -T.GRIP_AT, T.BAR_Z + 0.02]]), (24, 1))
    t = run(pull(102.0), tips=near)
    ok = not t.full and t.score == 0.0
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} a gripper 2 cm above the lever at rest: score {t.score:.3f}, full {t.full}")
    t = run(pull(102.0), pos=np.array([1.2, 0.0, T.TOP]))
    ok = not t.full and t.score == 0.0
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} off the bench: full {t.full}")
    t = run(pull(102.0), tilt=45.0)
    ok = not t.full and t.score == 0.0
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} tipped over: full {t.full}")
    print("all as expected" if not bad else f"{bad} NOT as expected")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
