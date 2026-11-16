"""Offline checks for hang-stemware-rack (NumPy and pytest only; no Isaac Sim).

Run from this folder with the yam-isaac-lab-arena package importable (installed, as on the Brev VM, or a clone
named by YAM_ARENA):

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

import yam_arena  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.analysis import world_hulls  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import forward_kinematics, rpy_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0  # the rate the README's runs use
ARM_URDF = Path(yam_arena.__file__).resolve().parent / "assets" / "yam" / "yam_arm.urdf"  # links with collision meshes


# -- helpers -----------------------------------------------------------------------------------------------------------


def _stl(path):
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:300]:
        return np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data), float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0)


@lru_cache(maxsize=1)
def collision_points():
    """Every link's collision-mesh vertices in its own frame, from yam_arm.urdf."""
    root = ET.parse(ARM_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array(o.get("xyz").split(), float)
            rpy = np.array(o.get("rpy").split(), float)
            mesh = ARM_URDF.parent / col.find("geometry/mesh").get("filename")
            pts.append(_stl(mesh) @ rpy_matrix(rpy).T + xyz)
        if pts:
            out[link.get("name")] = np.vstack(pts)
    return out


def _poses(q6, finger, side):
    joints = {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)}
    joints["joint7"] = joints["joint8"] = float(finger)
    fk = forward_kinematics(ARM_URDF, joints)
    base = np.eye(4)
    base[:3, 3] = station.BASES[side]
    return {k: base @ v for k, v in fk.items() if k != "base"}


def links_in_world(q6, finger, side):
    poses = _poses(q6, finger, side)
    return {
        name: p @ poses[name][:3, :3].T + poses[name][:3, 3]
        for name, p in collision_points().items()
        if name in poses
    }


@lru_cache(maxsize=None)
def played(side):
    """Every control step of one arm's plan at 20 Hz: (t, joints, finger opening in m, label)."""
    plan = TASK.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    out = []
    for k in range(round(arm.duration * HZ)):
        t = (k + 1) / HZ
        action, label = arm.action7(t)
        out.append((t, action[:6].copy(), FINGER_OPEN * action[6], label))
    return arm, out


def world_pieces(compound, centre_world, rot=np.eye(3)):
    """(lo, hi) world AABBs of an axis-aligned compound's pieces (rot must be identity or a pure yaw of 0)."""
    out = []
    for p in compound.pieces:
        c = np.asarray(centre_world) + rot @ np.asarray(p.offset)
        h = np.abs(rot) @ (np.asarray(p.size) / 2)
        out.append((c - h, c + h))
    return out


def box_gap(a, b):
    """Separation of two AABBs (m); negative = overlap depth along the least-overlapping axis."""
    (alo, ahi), (blo, bhi) = a, b
    sep = np.maximum(alo - bhi, blo - ahi)
    return float(sep.max()) if np.any(sep > 0) else float(sep.max())


def points_to_box(pts, lo, hi):
    """Distance from each point to an AABB (0 inside)."""
    return np.linalg.norm(pts - np.clip(pts, lo, hi), axis=1)


def step_at(side, label, last=True):
    _, steps = played(side)
    hits = [s for s in steps if s[3] == label]
    assert hits, label
    return hits[-1] if last else hits[0]


def finger_points(side, step):
    _, q, f, _ = step
    w = links_in_world(q, f, side)
    return np.vstack([w["tip_left"], w["tip_right"]])


def arm_points(side, step):
    _, q, f, _ = step
    return np.vstack(list(links_in_world(q, f, side).values()))


def glass_state(foot, tilt=0.0, vel=(0.0, 0.0, 0.0)):
    """ObjectState of a glass whose foot underside centre is at ``foot`` (world), tilted ``tilt`` deg about x."""
    from yam_arena.urdf import axis_angle_matrix

    r = axis_angle_matrix(np.array([1.0, 0, 0]), math.radians(tilt))
    c = np.asarray(foot, float) - r @ np.array([0.0, 0.0, M.FOOT_Z])
    half = np.abs(r) @ (np.asarray(M.GLASS_SIZE) / 2)
    return ObjectState(c, np.asarray(ik.matrix_to_quat_xyzw(r), float), np.asarray(vel, float), np.zeros(3),
                       c - half, c + half)


def hung_at(y, dx=0.0, dz=0.0, **kw):
    return glass_state((M.XS + dx, y, TOP + M.H + dz), **kw)


def on_mat(name):
    x, y = M.SPAWN[name]
    return glass_state((x, y, TOP + M.MAT_T + M.GL / 2 + M.FOOT_Z))


def goal_objects():
    return {n: hung_at(M.HANG_Y[n]) for n in M.GLASS_NAMES}


def rack_pieces():
    return world_pieces(M._rack_compound(), (M.RACK_C[0], M.RACK_C[1], TOP + M.RACK_C[2]))


def post_pieces():
    return world_pieces(M._posts_compound(M.POSTS_C), (M.POSTS_C[0], M.POSTS_C[1], TOP + M.POSTS_C[2]))


GLASS_PARTS = {name: np.array(p) for name, p, _ in M.rx.read_obj_parts(str(HERE / "glass.obj"))}


def glass_parts_world(foot):
    """Each glass part's vertices for a glass hanging straight with its foot underside centre at ``foot``."""
    c = np.asarray(foot, float) - np.array([0.0, 0.0, M.FOOT_Z])
    return {k: v + c for k, v in GLASS_PARTS.items()}


# -- 1. the file loads ------------------------------------------------------------------------------------------------


def test_the_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert TASK.free_objects == M.GLASS_NAMES
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"right"}
    assert "yam_arena.scene" not in sys.modules
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name


# -- 2. geometry -------------------------------------------------------------------------------------------------------


def test_real_rack_profile_and_a_glass_that_fits_it():
    assert (M.SLOT, M.OPEN, M.PROF) == (0.025, 0.089, 0.038)  # 1 in channel, 3 1/2 in opening, 1 1/2 in tall
    assert M.H == pytest.approx(0.22)
    assert M.SLOT < 2 * M.FOOT_R < M.OPEN and 2 * M.FOOT_R <= 0.083  # foot rides on the lips; racks take <= 83 mm bases
    stem = GLASS_PARTS["stem"]
    assert 2 * np.abs(stem[:, 0]).max() == pytest.approx(0.009, abs=1e-4) and 0.009 < M.SLOT


def test_the_glass_mesh_matches_its_box_and_foot_height():
    pts = np.vstack(list(GLASS_PARTS.values()))
    assert set(GLASS_PARTS) == {"bowl", "stem", "foot"}
    assert np.allclose(pts.max(0) - pts.min(0), M.GLASS_SIZE, atol=2e-4)
    assert np.allclose((pts.max(0) + pts.min(0)) / 2, 0.0, atol=2e-4)
    assert GLASS_PARTS["foot"][:, 2].min() == pytest.approx(M.FOOT_Z)
    assert np.ptp(GLASS_PARTS["stem"][:, 2]) == pytest.approx(0.080)


def test_a_hung_glass_rests_on_both_lips_and_clears_the_rest_of_the_rack_and_the_posts():
    lips = rack_pieces()[:2]
    others = rack_pieces()[2:] + post_pieces()
    for y in M.HANG_Y.values():
        parts = glass_parts_world((M.XS, y, TOP + M.H))
        foot = parts["foot"]
        for lo, hi in lips:  # the foot underside lies on each lip's top over at least 20 mm
            over = (foot[:, 0] > lo[0]) & (foot[:, 0] < hi[0]) & (np.abs(foot[:, 2] - hi[2]) < 1e-6)
            assert over.any() and np.ptp(foot[over, 0]) >= 0.020
        allpts = np.vstack(list(parts.values()))
        for lo, hi in others:
            assert points_to_box(allpts, lo, hi).min() >= 0.004
        for lo, hi in lips:  # the stem and bowl pass between the lips
            below = np.vstack([parts["stem"], parts["bowl"]])
            assert points_to_box(below, lo, hi).min() >= 0.003
    a, b = (M.HANG_Y[n] for n in M.GLASS_NAMES)
    assert a - b >= M.GLASS_SIZE[0] + 0.005  # the hung bowls do not touch
    assert M.YC - M.WALL - (a + M.FOOT_R) >= 0.005  # A's foot short of the closed end


def test_the_glasses_stand_on_the_mat_apart_and_clear_of_the_rack():
    for n in M.GLASS_NAMES:
        x, y = M.SPAWN[n]
        assert abs(x - M.MAT.pos[0]) + M.GLASS_SIZE[0] / 2 <= M.MAT.size[0] / 2
        assert abs(y - M.MAT.pos[1]) + M.GLASS_SIZE[1] / 2 <= M.MAT.size[1] / 2
        assert y + M.GLASS_SIZE[1] / 2 < M.YE  # outside the rack's open end
    (xa, _), (xb, _) = M.SPAWN.values()
    assert xb - xa - M.GLASS_SIZE[0] >= 0.004


# -- 3. scorer ---------------------------------------------------------------------------------------------------------


def test_goal_scores_one_and_succeeds():
    objs = goal_objects()
    assert M.score(objs) == 1.0 and M.success(objs)
    assert M.distance(objs) == pytest.approx(0.0, abs=1e-9)


def test_one_glass_hung_scores_half():
    objs = goal_objects()
    objs["glass_b"] = on_mat("glass_b")
    assert M.score(objs) == 0.5 and not M.success(objs)
    objs = {n: on_mat(n) for n in M.GLASS_NAMES}
    assert M.score(objs) == 0.0


@pytest.mark.parametrize("kw", [
    {"dz": 0.004},  # held up off the lips
    {"dz": -0.010},  # foot under the lips (fallen through or never above them)
    {"dx": 0.009},  # stem against the lip's edge, foot half off
    {"tilt": 6.0},  # not hanging straight
])
def test_a_glass_not_hanging_on_the_lips_does_not_count(kw):
    objs = goal_objects()
    objs["glass_b"] = hung_at(M.HANG_Y["glass_b"], **kw)
    assert M.score(objs) == 0.5


def test_a_foot_sticking_out_of_the_open_end_does_not_count():
    objs = goal_objects()
    objs["glass_b"] = hung_at(M.YE + M.FOOT_R - 0.002)
    assert M.score(objs) == 0.5
    objs["glass_b"] = hung_at(M.YE + M.FOOT_R + 0.001)
    assert M.score(objs) == 1.0


def test_a_moving_glass_scores_zero():
    objs = goal_objects()
    objs["glass_a"] = hung_at(M.HANG_Y["glass_a"], vel=(0.0, 0.05, 0.0))
    assert M.score(objs) == 0.0


def test_a_tcp_within_3_cm_of_a_glass_scores_zero(monkeypatch):
    objs = goal_objects()
    stem = np.array([M.XS, M.HANG_Y["glass_b"], TOP + M.H - 0.06])
    far = np.array([0.15, 0.30, TOP + 0.25])
    monkeypatch.setattr(M, "tcp_points", lambda: [far, stem - np.array([0.05, 0.0, 0.0])])
    assert M.score(objs) == 0.0
    monkeypatch.setattr(M, "tcp_points", lambda: [far, stem - np.array([0.10, 0.0, 0.0])])
    assert M.score(objs) == 1.0


def test_a_knocked_over_glass_scores_zero_and_stays_zero_until_the_next_episode(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(M, "episode_t", lambda: clock["t"])
    monkeypatch.setitem(M._KNOCK, "t", None)
    monkeypatch.setitem(M._KNOCK, "knocked", False)
    objs = {n: on_mat(n) for n in M.GLASS_NAMES}
    assert M.score(objs) == 0.0 and not M._KNOCK["knocked"]
    clock["t"] = 4.0
    objs["glass_b"] = glass_state((0.53, -0.30, TOP + 0.05), tilt=80.0)  # on its side on the mat
    assert M.score(objs) == 0.0 and M._KNOCK["knocked"]
    clock["t"] = 30.0
    assert M.score(goal_objects()) == 0.0 and not M.success(goal_objects())  # both hung later: still latched
    clock["t"] = 0.05  # reset: the clock goes back
    assert M.score(goal_objects()) == 1.0


def test_the_carry_tilt_does_not_trip_the_knock_latch():
    assert M.KNOCK_TILT_DEG >= 3 * 15.0  # the GPU spike's carries tilted at most 15 deg in the hand


# -- 4. demo -----------------------------------------------------------------------------------------------------------


def test_the_plan_is_reachable_within_the_joint_limits_and_the_run_limit():
    arm, steps = played("right")
    joints = np.array([s[1] for s in steps])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
    limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
    assert arm.duration * HZ <= limit - 100  # leaves 5 s for settling and the 1 s success hold


def test_each_close_lasts_at_least_1_s_on_the_stem():
    closes = [w for w in TASK.demo()["right"] if w.label.endswith(": close")]
    assert len(closes) == 2
    assert all(w.duration >= 1.0 and w.finger == FINGER_CLOSED for w in closes)


@pytest.mark.parametrize("name", M.GLASS_NAMES)
def test_the_open_pads_straddle_the_stem_and_close_on_it(name):
    x, y = M.SPAWN[name]
    z = TOP + M.MAT_T + M.GL / 2 + M.FOOT_Z  # foot underside on the mat
    tips = finger_points("right", step_at("right", f"{name}: in around the stem"))
    level = (tips[:, 2] > z - 0.080) & (tips[:, 2] < z)
    assert level.any()
    r = np.hypot(tips[level, 0] - x, tips[level, 1] - y)
    assert r.min() >= 0.0045 + 0.002  # no pad point inside the 9 mm stem
    ys = tips[level, 1] - y
    assert ys.min() < -0.0045 and ys.max() > 0.0045  # one finger each side


LOADED = ("lift, foot over the lips' height", "to the open end", "slide along the channel", "lower the foot onto the lips")
LOADED_SAG = 0.012  # the hand carrying a glass rode 14.5-19 mm under its target on the GPU; the plan aims high for it


def test_no_arm_link_enters_the_rack_the_posts_or_the_mat():
    """Arm links stay 3 mm clear of the fixed parts. While a glass is carried the hand is taken 12 mm under its
    planned pose (the smallest sag measured, less a margin): unloaded, the planned carry would touch the lips."""
    _, steps = played("right")
    boxes = rack_pieces() + post_pieces() + [(np.asarray(M.MAT.pos) - np.asarray(M.MAT.size) / 2,
                                              np.asarray(M.MAT.pos) + np.asarray(M.MAT.size) / 2)]
    worst = np.inf
    for st in steps[::2]:
        pts = arm_points("right", st)
        if st[3].split(": ")[-1] in LOADED:
            pts = pts - np.array([0.0, 0.0, LOADED_SAG])
        for lo, hi in boxes:
            worst = min(worst, float(points_to_box(pts, lo, hi).min()))
    assert worst >= 0.003, worst


def test_task_json_and_readme_have_every_required_field():
    meta = json.loads((HERE / "task.json").read_text())
    for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                "fidelity", "reset_notes", "scores"):
        assert meta.get(key), key
    assert meta["category"] == "Impact on jobs"
    assert len(meta["fidelity"]) <= 500
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run: ")
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
