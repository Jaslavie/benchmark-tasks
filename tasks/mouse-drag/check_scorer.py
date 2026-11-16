"""Every score line of task.py on hand-built states, with no simulator. Run it with the harness's Python, from this folder: python check_scorer.py"""

from __future__ import annotations

import math

import numpy as np

import mechkit as m
import task as T

DT = 0.05  # a control step at 20 Hz
FAR = np.full((24, 3), 2.0)  # every finger far from everything
Z0 = T.TOP + T.MAT_T


def yaw_rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


class Sim:
    """Drives a Trial as the simulator would: the mouse slides in its own axes, the button goes in and out, the fingers sit where they are put."""

    def __init__(self, seed=1):
        m._SEED[0] = seed
        self.t = T.Trial()
        lay = self.t.lay
        self.pos = np.array([*lay["mouse"], Z0])
        self.rot = yaw_rot(lay["yaw_deg"])
        self.pressed = 0.0
        self.tips = FAR
        self.step()

    def step(self, n: int = 1):
        for _ in range(n):
            self.t.update(self.pos, self.rot, self.pressed, self.tips, DT)

    def cursor_to(self, goal_px, steps: int = 20):
        """Slide the mouse so that the cursor ends on `goal_px`, over `steps` steps."""
        err = np.asarray(goal_px, float) - self.t.cursor
        move = self.rot @ np.array([-err[1], -err[0], 0.0]) / (T.GAIN * 1000.0)
        for _ in range(steps):
            self.pos = self.pos + move / steps
            self.step()

    def press(self):
        self.pressed = 0.0005
        self.tips = np.tile(self.pos + np.array([0.03, 0.015, 0.03]), (24, 1))
        self.step(2)

    def release(self, away: bool = True):
        self.pressed = 0.0
        self.step()
        if away:
            self.tips = FAR

    def drag(self, colour_or_px, steps: int = 20):
        s = self
        s.cursor_to(s.t.file)
        s.press()
        goal = s.t.folders[colour_or_px] if isinstance(colour_or_px, str) else colour_or_px
        s.cursor_to(goal, steps)
        s.release()


def full(seed=1):
    s = Sim(seed)
    s.drag(T.TARGET)
    s.step(25)
    return s


def check(name, s, want):
    got = s.t.score
    print(f"{'ok ' if got == want else 'BAD'} {got:.2f} (want {want:.2f})  {name}  | {s.t.failed or ''}")
    return got == want


def cases():
    ok = True
    s = Sim()
    s.step(40)
    ok &= check("nothing done", s, 0.0)
    for seed in (1, 2, 3, 7, 11):
        ok &= check(f"seed {seed}: file dragged into the green folder, hands off, held 1 s", full(seed), 1.0)
    s = Sim()
    s.drag(T.TARGET)
    s.step(10)
    ok &= check("in the green folder, held only 0.5 s", s, 0.5)
    s = Sim()
    s.cursor_to(s.t.file)
    s.press()
    s.cursor_to(s.t.folders[T.TARGET])
    s.pressed = 0.0
    s.step(30)
    ok &= check("in the green folder, the finger left on the mouse", s, 0.5)
    s = Sim()
    wrong = [c for c in T.COLOURS if c != T.TARGET][0]
    s.drag(wrong)
    s.step(25)
    ok &= check(f"dropped into the {wrong} folder", s, 0.0)
    s = Sim()
    s.drag(wrong)
    s.cursor_to(s.t.folders[T.TARGET])
    s.step(25)
    ok &= check("dropped into a wrong folder, then nothing more can be done", s, 0.0)
    s = Sim()
    start, goal = s.t.file.copy(), s.t.folders[T.TARGET]
    near = goal + np.array([0.0, 140.0])  # just below the folder: not on it
    s.drag(near)
    s.step(25)
    ok &= check("dropped just short of the green folder (more than half way)", s, 0.25)
    s = Sim()
    s.cursor_to(s.t.file)
    s.press()
    s.cursor_to(start + 0.7 * (goal - start))
    s.step(25)
    ok &= check("dragged 70% of the way, the button still held at the end", s, 0.0)
    s = Sim()
    s.cursor_to(s.t.folders["blue" if T.TARGET != "blue" else "yellow"])
    s.press()
    s.release()
    s.drag(T.TARGET)
    s.step(25)
    ok &= check("a single click on another folder, then the file into the green one", s, 1.0)
    s = Sim()
    s.drag(start + 0.3 * (goal - start))
    s.step(25)
    ok &= check("dropped a third of the way", s, 0.0)
    s = Sim()
    s.drag(start + 0.6 * (goal - start))
    s.drag(T.TARGET)
    s.step(25)
    ok &= check("dropped short, then dragged again into the green folder", s, 1.0)
    s = Sim()
    s.cursor_to(s.t.file + np.array([0.0, 0.0]))
    s.drag(s.t.folders[T.TARGET] + np.array([T.FOLDER_BOX[1] + 4.0, 0.0]))
    s.step(25)
    ok &= check("dropped 4 px right of the folder's edge", s, s.t.score if s.t.score in (0.0, 0.25) else -1)
    ok &= s.t.file_in is None
    s = Sim()
    s.cursor_to(s.t.folders[T.TARGET])
    s.press()
    s.cursor_to(s.t.folders[T.TARGET] + np.array([200.0, 100.0]))
    s.release()
    s.drag(T.TARGET)
    s.step(25)
    ok &= check("a folder dragged first, then the file into it", s, 0.0)
    s = Sim()
    s.cursor_to(s.t.folders[T.TARGET])
    for _ in range(2):
        s.press()
        s.release(away=False)
    s.tips = FAR
    s.drag(T.TARGET)
    s.step(25)
    ok &= check("the green folder double-clicked, then the file into it", s, 0.0)
    s = Sim()
    s.cursor_to(s.t.folders[T.TARGET])
    s.press()
    s.release(away=False)
    s.step(12)
    s.press()
    s.release()
    s.drag(T.TARGET)
    s.step(25)
    ok &= check("the green folder clicked twice, 0.7 s apart: no double click", s, 1.0)
    s = Sim()
    s.drag(T.TARGET)
    s.pos = s.pos + np.array([0.0, 0.2, 0.0])
    s.step(25)
    ok &= check("in the green folder, then the mouse knocked off the mat", s, 0.0)
    s = Sim()
    s.drag(T.TARGET)
    s.tips = np.tile(np.array([0.0, 0.0, T.TOP]) + T.screen_point(960, 540, 0.002), (24, 1))
    s.step(2)
    s.tips = FAR
    s.step(25)
    ok &= check("in the green folder, the screen touched", s, 0.0)
    s = Sim()
    s.cursor_to(s.t.file)
    s.press()
    s.pos = s.pos + np.array([0.0, 0.0, 0.003])  # lifted 3 mm: the sensor loses the mat
    s.cursor_to(s.t.folders[T.TARGET])
    s.pos[2] = Z0
    s.release()
    s.step(25)
    ok &= check("dragged with the mouse lifted: the cursor does not move", s, 0.0)
    s = Sim()
    s.cursor_to(s.t.file)
    s.press()
    s.release()
    s.step(25)
    ok &= check("the file clicked, not dragged", s, 0.0)
    return ok


def geometry():
    ok = True
    for px, py in ((0, 0), (1919, 1079), (960, 540), (123.4, 777.7)):
        back = T.to_pixels(T.screen_point(px, py))
        ok &= bool(np.allclose(back, (px, py), atol=1e-6))
    lo = T.screen_point(0, 1079)
    hi = T.screen_point(0, 0)
    print(f"screen: bottom edge {lo[0]:.3f} m out, {lo[2] * 100:.1f} cm up; top edge {hi[0]:.3f} m out, {hi[2] * 100:.1f} cm up; width {T.PX_W * T.PITCH * 100:.1f} cm")
    print(f"gain: {T.GAIN} px/mm; folder drop area {T.FOLDER_BOX[1] - T.FOLDER_BOX[0]:.0f} px wide = +/- {(T.FOLDER_BOX[1] - T.FOLDER_BOX[0]) / 2 / T.GAIN:.2f} mm of mouse travel")
    for seed in range(1, 6):
        lay = T.layout_for(seed)
        print(f"seed {seed}: order {lay['order']}, file {np.round(lay['file'])}, cursor {np.round(lay['cursor'])}, mouse {np.round(lay['mouse'], 3)} yaw {lay['yaw_deg']:.1f}")
    left = T._button_shell(1.0)
    top = left.v[: len(left.v) // 2]
    print(f"left button: z {top[:, 2].min() * 1000:.1f} to {top[:, 2].max() * 1000:.1f} mm, y {top[:, 1].min() * 1000:.1f} to {top[:, 1].max() * 1000:.1f} mm; press point z {T.dome_z(*T.PRESS_AT) * 1000:.1f} mm")
    return ok


if __name__ == "__main__":
    g = geometry()
    c = cases()
    print("ALL OK" if g and c else "SOME BAD")
