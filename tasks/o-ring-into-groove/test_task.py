"""Offline tests of the O-ring task's scorer (no simulator). Run: /data/IsaacLab-Arena/.venv/bin/python test_task.py

The ring model runs as in the sim (ringkit.update each step); the grippers and the holder are hand-made states.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import ringkit as rk  # noqa: E402
import yamkit as kit  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

T = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.py")).module
KEY = T.KEY
R_HOME = np.array([0.20, -0.30, T.TOP + 0.25])
L_HOME = np.array([0.20, 0.30, T.TOP + 0.25])
FULL = 2 * abs(FINGER_OPEN)
END = T.TOP + T.END_H
HEXZ = T.TOP + T.HEX_TOP


def down_quat(yaw=0.0, tilt=0.0, about="x"):
    """A straight-down grip (fingers closing along world y at yaw 0), tilted `tilt` deg about world x (the line
    between the fingertips tilts: one tip higher than the other) or world y (about that line: both tips level)."""
    q = (0.70711, 0.70711, 0.0, 0.0)
    q = kit._quat_mul((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), q)
    h = math.radians(tilt) / 2
    ax = (math.sin(h), 0.0, 0.0) if about == "x" else (0.0, math.sin(h), 0.0)
    return kit._quat_mul((*ax, math.cos(h)), q)


def st(pos, quat=(0.0, 0.0, 0.0, 1.0)):
    p = np.asarray(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.asarray(quat, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p,
                       high=p)


class Sim:
    """A trial: the right gripper's tool point at `tcp`, its pads `gap` apart, pointing straight down (or tilted)."""

    def __init__(self, about="x"):
        kit._EPISODE["count"] += 1
        self.tcp = R_HOME.copy()
        self.gap = FULL
        self.tilt = 0.0
        self.about = about
        self.scores = []

    def objects(self):
        kit._OPENING["right_gripper"] = self.gap / FULL
        kit._OPENING["left_gripper"] = 1.0
        return {"holder": st((T.SX, T.SY, T.TOP)), "right_gripper": st(self.tcp, down_quat(0.0, self.tilt, self.about)),
                "left_gripper": st(L_HOME, down_quat())}

    def step(self, n=1):
        for _ in range(n):
            o = self.objects()
            rk.update(o)
            T.success(o)
            self.scores.append(T.score(o))
        return self

    def move(self, to, steps=20, gap=None, tilt=None):
        a, b = self.tcp.copy(), np.asarray(to, float)
        g0, g1 = self.gap, (self.gap if gap is None else gap)
        t0, t1 = self.tilt, (self.tilt if tilt is None else tilt)
        for k in range(1, steps + 1):
            f = k / steps
            self.tcp, self.gap, self.tilt = a + f * (b - a), g0 + f * (g1 - g0), t0 + f * (t1 - t0)
            self.step()
        return self

    def tips_at(self, z, xy=(T.SX, T.SY)):
        return np.array([xy[0], xy[1], z + T.TIP])

    @property
    def o(self):
        return self.objects()

    @property
    def last(self):
        return self.scores[-1]


def into_ring(s):
    s.move(s.tips_at(END + 0.06), gap=0.008)
    s.move(s.tips_at(END + 0.001))
    assert rk.engaged(s.o, KEY), "fingertips in the ring take hold"
    return s


def demo_like(gap=0.057, home=True, tilt_out=0.0, about="x", tilt_steps=10):
    s = into_ring(Sim(about))
    s.move(s.tips_at(END + 0.001), gap=gap, steps=30)
    s.move(s.tips_at(HEXZ + 0.0015), steps=40)
    if tilt_out:
        s.move(s.tcp, tilt=tilt_out, steps=tilt_steps)
    s.move(s.tips_at(END + 0.02), steps=30)
    s.move(s.tips_at(END + 0.08), steps=10)
    if home:
        s.move(R_HOME, steps=30)
        s.step(10)
    return s


def test_do_nothing():
    s = Sim().step(60)
    assert max(s.scores) == 0.0
    assert not rk.seated(s.o, KEY) and rk.groove(s.o, KEY) is None


def test_demo_full():
    s = demo_like()
    o = s.o
    assert rk.seated(o, KEY) and not rk.engaged(o, KEY) and not rk.cocked(o, KEY)
    assert abs(rk.position(o, KEY) - T.GROOVE_AT) < 1e-9  # cross-check: the ring's position is the groove's middle
    assert abs(rk.inner_d(o, KEY) - T.GROOVE_D) < 1e-9
    assert abs(rk.max_d(o, KEY) / T.FREE_ID - 1.287) < 0.002, rk.max_d(o, KEY)  # stretched 28.7%, under 50%
    assert rk.twist_deg(o, KEY) == 0.0
    assert s.last == 1.0, s.last


def test_not_home():
    s = demo_like(home=False)
    s.step(5)
    assert s.last == 1.0, s.last  # review, Nov 12: where the arms end is not scored
    assert not hasattr(T, "REQUIRE_ARMS_HOME") and not hasattr(T, "_arms_home")


def test_still_held_at_groove():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=30)
    s.move(s.tips_at(HEXZ + 0.0015), steps=40)
    s.step(20)
    assert rk.engaged(s.o, KEY) and not rk.seated(s.o, KEY)
    assert s.last == 0.5 and max(s.scores) == 0.5, "down at the groove but still held open: 0.5, never 1.0"


def test_seated_with_a_hand_on_it():
    """Andres, Nov 13: a ring seated right but with a gripper still touching it (its own shape within 2 mm) scores
    0.75, not handed off; the same hand lifted clear scores 1.0. Contact, not nearness: a hand 3 mm over the ring or
    one down by the hex face 2.5 mm outside it does not count."""
    s = demo_like(home=False)
    assert s.last == 1.0
    xy = (T.SX, T.SY - 0.0245)  # closed fingertips over the seated ring's section, beside the thread
    s.move(s.tips_at(END + 0.01, xy), gap=0.0, steps=20)
    ring_top = HEXZ + (T.GROOVE_W + T.RING_CS) / 2
    s.move(s.tips_at(ring_top + 0.001, xy), steps=10)  # 1 mm over its top: touching
    s.step(5)
    assert rk.seated(s.o, KEY) and rk.twist_deg(s.o, KEY) == 0.0 and s.last == 0.75, s.last
    s.move(s.tips_at(ring_top - 0.001, xy), steps=10)  # resting on it, 1 mm in: still seated, still 0.75
    s.step(5)
    assert rk.seated(s.o, KEY) and s.last == 0.75, s.last
    s.move(s.tips_at(ring_top + 0.003, xy), steps=10)  # lifted 3 mm clear: not touching
    s.step(5)
    assert s.last == 1.0, s.last
    s.move(s.tips_at(HEXZ + 0.0005, (T.SX, T.SY - 0.034)), steps=20)  # down by the hex face, tips 2.5 mm outside it
    s.step(5)
    assert s.last == 1.0, s.last
    s.move(R_HOME, steps=30)
    s.step(10)
    assert s.last == 1.0, s.last


def test_overstretch_latched():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=FULL, steps=30)  # opened all the way: 78% stretch
    assert rk.overspread(s.o, KEY)
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=20)  # back to a sane stretch, then a perfect fit
    s.move(s.tips_at(HEXZ + 0.0015), steps=40)
    s.move(s.tips_at(END + 0.08), steps=40)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert not rk.seated(s.o, KEY) and max(s.scores) == 0.0, "an over-stretch is never forgiven (it sits loose)"


def test_just_under_limit():
    s = demo_like(gap=0.073)  # 49%
    assert not rk.overspread(s.o, KEY) and s.last == 1.0, (rk.max_d(s.o, KEY), s.last)


def test_too_little_stretch_cannot_pass():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.030, steps=20)  # fingers still over the end face: the ring stays loose
    assert rk.inner_d(s.o, KEY) == T.FREE_ID
    s.move(s.tips_at(END + 0.06), steps=20)
    assert not rk.engaged(s.o, KEY) and abs(rk.position(s.o, KEY) + T.RING_CS / 2) < 1e-9


def test_rolled_on_with_a_fingertip():
    for steps in (60, 5):  # slowly, or fast (4 mm a step)
        s = Sim()
        xy = (T.SX, T.SY - 0.0245)  # fingers closed, their tips on the ring's top beside the thread
        s.move(s.tips_at(END + 0.03, xy), gap=0.0, steps=20)
        s.move(s.tips_at(END + 0.0025, xy), steps=10)
        s.move(s.tips_at(HEXZ + 0.0005, xy), steps=steps)  # pushed down the thread: it rolls
        assert rk.twist_deg(s.o, KEY) > 300, rk.twist_deg(s.o, KEY)
        s.move(s.tips_at(END + 0.06, xy), steps=20)
        s.move(R_HOME, steps=30)
        s.step(10)
        assert rk.groove(s.o, KEY) == 0 and rk.seated(s.o, KEY), "it does end up in the groove"
        assert max(s.scores) == 0.0, "but twisted: 0"


def test_rolled_part_way():
    s = Sim()
    xy = (T.SX, T.SY - 0.0245)
    s.move(s.tips_at(END + 0.0025, xy), gap=0.0, steps=20)
    s.move(s.tips_at(END - 0.004, xy), steps=10)  # rolled 4 mm onto the thread
    s.move(s.tips_at(END + 0.03, xy), steps=10)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert rk.twist_deg(s.o, KEY) > 100 and not rk.seated(s.o, KEY)
    assert max(s.scores) == 0.0


def test_tilted_release_cocked():
    for tilt, steps in ((5.0, 10), (8.0, 10), (12.0, 10), (20.0, 10), (8.0, 1), (8.0, 40)):
        s = demo_like(tilt_out=tilt, tilt_steps=steps)  # one fingertip lifted: it leaves the ring tilted
        assert rk.cocked(s.o, KEY) and not rk.seated(s.o, KEY), (tilt, steps, rk.max_tilt_deg(s.o, KEY))
        assert s.last == 0.5 and max(s.scores) == 0.5, (tilt, steps, s.last)  # at the groove, one side on the thread


def test_small_tilt_ok():
    s = demo_like(tilt_out=2.0)
    assert rk.seated(s.o, KEY) and s.last == 1.0
    assert abs(rk.max_tilt_deg(s.o, KEY) - 2.0) < 0.05, "tilt is the fingertips' height difference over their gap"


def test_gripper_angle_alone_does_not_tilt_it():
    s = demo_like(tilt_out=8.0, about="y")  # tilted about the line between the tips: both still at one height
    assert rk.max_tilt_deg(s.o, KEY) < 0.5 and rk.seated(s.o, KEY) and s.last == 1.0


def test_poking_a_seated_ring_does_not_roll_it():
    s = demo_like(home=False)
    assert rk.seated(s.o, KEY) and rk.twist_deg(s.o, KEY) == 0.0
    xy = (T.SX, T.SY - 0.0245)  # closed fingertips down beside the thread, onto and into the seated ring
    s.move(s.tips_at(END + 0.01, xy), gap=0.0, steps=20)
    for z in (HEXZ + 0.003, HEXZ + 0.0015, HEXZ + 0.0003):
        s.move(s.tips_at(z, xy), steps=10)
        s.step(20)
        assert rk.max_roll_deg(s.o, KEY) == 0.0, (z, rk.max_roll_deg(s.o, KEY))
    s.move(s.tips_at(END + 0.06, xy), steps=20)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert rk.seated(s.o, KEY) and rk.max_roll_deg(s.o, KEY) == 0.0 and s.last == 1.0


def test_tip_beside_the_section_does_not_push():
    s = Sim()
    xy = (T.SX, T.SY - 0.0245)
    s.move(s.tips_at(END + 0.0025, (T.SX, T.SY - 0.06)), gap=0.0, steps=20)  # level with the ring, off to one side
    s.move(s.tips_at(END + 0.0025, xy), steps=10)  # slid in sideways into its section: not on its face
    s.move(s.tips_at(END + 0.0005, xy), steps=10)
    assert abs(rk.position(s.o, KEY) + T.RING_CS / 2) < 1e-9 and rk.max_roll_deg(s.o, KEY) == 0.0


def test_let_go_on_thread():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=30)
    s.move(s.tips_at(END - 0.010), steps=20)  # halfway down the thread
    s.move(s.tips_at(END + 0.06), steps=20)
    s.move(R_HOME, steps=30)
    s.step(10)
    o = s.o
    assert not rk.engaged(o, KEY) and rk.groove(o, KEY) is None and not rk.seated(o, KEY)
    assert abs(rk.inner_d(o, KEY) - T.THREAD_D) < 1e-9, "it grips the thread"
    assert s.last == 0.25, s.last  # safe progress: stretched over and carried 8.5 mm down the thread


def test_cocked_then_respread_and_seated():
    """Judged by where the ring ends: a ring let go tilted (one side on the thread) and then stretched again, squared
    and let go square ends seated, 1.0."""
    s = demo_like(tilt_out=8.0, home=False)
    assert rk.cocked(s.o, KEY) and s.last == 0.5
    s.move(s.tips_at(END + 0.02), gap=0.008, tilt=0.0, steps=20)
    s.move(s.tips_at(HEXZ + 0.0015), steps=20)
    s.move(s.tips_at(HEXZ + 0.0015), gap=0.057, steps=20)
    assert not rk.cocked(s.o, KEY)
    s.move(s.tips_at(END + 0.02), steps=30)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert rk.seated(s.o, KEY) and s.last == 1.0, s.last


def _high_side_xy(sign=1.0, rad=None):
    """(x, y) over a cocked ring's high side (sign=-1: its low, deeper side), at the middle of its section."""
    stt = rk.state(KEY)
    u = np.array([0.0, 0.0, -1.0])
    hi = -np.cross(u, np.asarray(stt.tilt_axis, float))
    hi = sign * hi / np.linalg.norm(hi)
    rad = T.THREAD_D / 2 + T.RING_CS / 2 if rad is None else rad
    return (T.SX + rad * hi[0], T.SY + rad * hi[1])


def _press(s, xy, depth, steps=10):
    """Closed fingertips from above down onto (xy) to `depth` m past the high side's near face, then up and away."""
    hn = rk.FingerTips.high_near(T.RING, rk.state(KEY))
    s.move(s.tips_at(END + 0.02, xy), gap=0.0, steps=20)
    s.move(s.tips_at(END - hn - depth, xy), steps=steps)
    s.step(5)
    return s


def test_cocked_then_pressed_into_groove():
    """Review, Nov 12: judged by the ring's final condition. A ring let go 8 degrees off square lands with its high side
    on the thread (0.5); a fingertip pressing that side down 1.5 mm pushes it over the last crest into the groove,
    rolling it 57 degrees (1.5 mm for its 3.0 mm section), under the 90-degree limit: seated, 1.0."""
    for fast in (False, True):
        s = demo_like(tilt_out=8.0, home=False)
        assert rk.cocked(s.o, KEY) and s.last == 0.5
        _press(s, _high_side_xy(), T.RING_CS / 2 + 0.0003, steps=2 if fast else 20)
        assert not rk.cocked(s.o, KEY) and abs(rk.position(s.o, KEY) - T.GROOVE_AT) < 1e-9
        assert abs(rk.twist_deg(s.o, KEY) - math.degrees(1.0)) < 1e-6, rk.twist_deg(s.o, KEY)
        s.move(s.tips_at(END + 0.06), steps=20)
        s.move(R_HOME, steps=30)
        s.step(10)
        assert rk.seated(s.o, KEY) and s.last == 1.0, (fast, s.last)


def test_press_short_or_on_the_low_side_leaves_it_cocked():
    s = demo_like(tilt_out=8.0, home=False)
    _press(s, _high_side_xy(), T.RING_CS / 2 - 0.0005)  # only 1.0 mm down: it springs back up onto the land
    assert rk.cocked(s.o, KEY)
    s.move(s.tips_at(END + 0.03), steps=10)
    _press(s, _high_side_xy(-1.0), 0.004)  # on the side already in the groove: nothing to push over
    assert rk.cocked(s.o, KEY)
    _press(s, _high_side_xy(1.0, rad=0.034), 0.004)  # 34 mm out, past the ring's outside: misses it
    assert rk.cocked(s.o, KEY)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert s.last == 0.5 and max(s.scores) == 0.5


def test_tip_already_down_by_the_ring_does_not_press():
    """A fingertip that slid in beside the high side below its near face, not down onto it, does not press it."""
    s = demo_like(tilt_out=8.0, home=False)
    hn = rk.FingerTips.high_near(T.RING, rk.state(KEY))
    far = _high_side_xy(rad=0.040)
    s.move(s.tips_at(END - hn - 0.001, far), gap=0.0, steps=20)  # level with the ring's section, 4 cm out
    s.move(s.tips_at(END - hn - 0.001, _high_side_xy()), steps=10)  # slid in over the section from the side
    s.move(s.tips_at(END - hn - 0.0025, _high_side_xy()), steps=10)
    assert rk.cocked(s.o, KEY)


def released_on_thread(above):
    """Stretched over and carried down, then let go square with its middle `above` m short of the groove's middle:
    it closes onto the thread there."""
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=30)
    s.move(s.tips_at(END - (T.GROOVE_AT - above + T.RING_CS / 2)), steps=40)
    s.move(s.tips_at(END + 0.02), steps=30)
    assert rk.groove(s.o, KEY) is None and not rk.engaged(s.o, KEY)
    return s


def test_twist_from_0_to_90_degrees():
    """The ring is scored on its twist as it ends: let go on the thread and pushed the rest of the way down with a
    fingertip, it rolls about 38 degrees per mm until it snaps into the groove (within 1.8 mm of its middle). Up to
    90 degrees it scores 1.0; past 90, 0."""
    xy = (T.SX, T.SY - 0.0245)
    for above_mm, lo, hi, want in ((1.9, 0, 10, 1.0), (3.0, 40, 52, 1.0), (3.9, 75, 87, 1.0), (4.4, 92, 108, 0.0)):
        s = released_on_thread(above_mm / 1000)
        s.move(s.tips_at(END + 0.03, xy), gap=0.0, steps=20)
        s.move(s.tips_at(HEXZ + 0.0005, xy), steps=150)  # about 0.2 mm a step
        tw = rk.twist_deg(s.o, KEY)
        assert rk.groove(s.o, KEY) == 0 and lo <= tw <= hi, (above_mm, tw)
        s.move(s.tips_at(END + 0.06, xy), steps=20)
        s.move(R_HOME, steps=30)
        s.step(10)
        assert rk.seated(s.o, KEY) and s.last == want, (above_mm, tw, s.last)
    for tw in (0.0, 45.0, 89.9, 90.0, 90.1):  # the limit itself, read straight off the ring's state
        s = demo_like(home=False)
        rk.state(KEY).roll = tw
        s.step()
        assert s.last == (1.0 if tw <= T.TWIST_LIMIT else 0.0), (tw, s.last)


def test_partial_needs_real_progress():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=30)
    s.move(s.tips_at(END - 0.0035), steps=10)  # 2 mm down the thread only
    s.move(s.tips_at(END + 0.06), steps=20)
    s.move(R_HOME, steps=30)
    s.step(10)
    assert rk.position(s.o, KEY) < T.ON_THREAD and s.last == 0.0, (rk.position(s.o, KEY), s.last)


def test_partial_is_where_the_ring_is_now():
    """Mock review, Nov 12: the 0.25 reads where the ring is at the end, not the deepest it ever reached. The ring
    model only lets fingertips push the ring down, so a ring worked back up the thread (by hand, in a real trial) is
    set straight in its state: carried 8.5 mm down, then 2 mm down at the end, it scores 0."""
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=30)
    s.move(s.tips_at(END - 0.010), steps=20)
    s.move(s.tips_at(END + 0.06), steps=20)
    s.move(R_HOME, steps=30)
    assert rk.position(s.o, KEY) >= T.ON_THREAD and s.last == 0.25, (rk.position(s.o, KEY), s.last)
    rk.state(KEY).s = 0.002
    s.step()
    assert rk.position(s.o, KEY) < T.ON_THREAD and s.last == 0.0, (rk.position(s.o, KEY), s.last)


def test_partial_lost_to_overstretch_or_twist():
    s = into_ring(Sim())
    s.move(s.tips_at(END + 0.001), gap=FULL, steps=30)  # over-stretched
    s.move(s.tips_at(END + 0.001), gap=0.057, steps=20)
    s.move(s.tips_at(HEXZ + 0.0015), steps=40)
    s.step(10)
    assert s.last == 0.0, "held at the groove after an over-stretch: 0"
    s = Sim()
    xy = (T.SX, T.SY - 0.0245)
    s.move(s.tips_at(END + 0.0025, xy), gap=0.0, steps=20)
    s.move(s.tips_at(END - 0.009, xy), steps=20)  # rolled 9 mm down: twisted
    s.move(R_HOME, steps=30)
    s.step(10)
    assert rk.position(s.o, KEY) >= T.ON_THREAD and s.last == 0.0, "pushed down unstretched: twisted, 0"


def test_pad_gap_at_50_percent():
    """The text's 73.7 mm: the pad gap at which the loop round the fingertips and the thread reaches 65.5 mm."""
    def at(gap):
        s = into_ring(Sim())
        s.move(s.tips_at(END + 0.001), gap=gap, steps=20)
        s.move(s.tips_at(END - 0.006), steps=10)  # both fingers outside the thread: the loop includes it
        return rk.max_d(s.o, KEY), rk.overspread(s.o, KEY)
    d, over = at(0.0735)
    assert not over and d < T.LIMIT_ID, d
    d, over = at(0.0740)
    assert over, d


def test_hover_and_touch():
    s = Sim()
    s.move(s.tips_at(END + 0.005), gap=0.008, steps=20)  # hovering just above the ring
    s.move(s.tips_at(END + 0.005, (T.SX + 0.04, T.SY)), steps=20)  # beside the plug
    s.move(s.tips_at(HEXZ + 0.002, (T.SX + 0.04, T.SY)), gap=FULL, steps=20)  # down beside it, open
    s.move(R_HOME, steps=20)
    s.step(10)
    assert max(s.scores) == 0.0 and not rk.overspread(s.o, KEY)


def test_episode_resets_history():
    s = demo_like()
    assert s.last == 1.0
    s2 = Sim().step(5)  # a scene reset: the ring is back on the end face, history fresh
    assert not rk.seated(s2.o, KEY) and max(s2.scores) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all pass")
