"""On the station table: a 24-inch monitor at the back, showing a desktop with a row of four coloured folders and a red file, and a wireless mouse on a cloth mat in front of it, on the right.
The robot moves the file into the green folder with the mouse: the cursor follows the mouse's sensor as a real one does, and the left button is a sprung shell that clicks at 0.6 N.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1200 --seed 1 --instruction "<INSTRUCTION below>"
The demo:  python demokit.py task.py OUT --seed 1
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mechkit as m  # noqa: E402  (hooks yam_arena so that the parts below spawn from USD)
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
INSTRUCTION = "Using the mouse, drag the red file into the green folder on the screen, then let go of the mouse."
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

# -- the monitor and its desktop -------------------------------------------------------------
# A 24-inch 1920 x 1080 panel (531 x 299 mm), leaned back 20 degrees on its stand, its picture's bottom edge at x 57 cm and 4.5 cm above the table.
PX_W, PX_H = 1920, 1080
PITCH = 0.5313 / PX_W  # metres per screen pixel
MON_X, MON_Z, TILT = 0.57, 0.045, math.radians(20.0)
R_AX = np.array([0.0, -1.0, 0.0])  # the screen's right, as seen from in front of it
U_AX = np.array([math.sin(TILT), 0.0, math.cos(TILT)])  # its up
N_AX = np.array([-math.cos(TILT), 0.0, math.sin(TILT)])  # out of its face, towards the viewer
CENTRE = np.array([MON_X, 0.0, MON_Z]) + 0.5 * PX_H * PITCH * U_AX
BEZEL, CHIN, PANEL_D = 0.009, 0.020, 0.018

GAIN = 31.5  # cursor pixels per millimetre of mouse travel: a 1,600 cpi sensor at a 0.5 pointer multiplier, acceleration off
LIFT_MM = 1.5  # the sensor stops tracking when lifted further than this off the mat
DRAG_PX = 6.0  # the cursor must move this far with the button held before a drag starts, as on a real desktop
DOUBLE_S, DOUBLE_PX = 0.5, 10.0  # two clicks this close in time and place are a double click

COLOURS = ("blue", "green", "yellow", "purple")
TARGET = "green"
SLOTS_X, ROW_Y = (420.0, 780.0, 1140.0, 1500.0), 250.0  # the folders' places, screen pixels from the top left
FOLDER_BOX = (-75.0, 75.0, -68.0, 60.0)  # a folder's drop area about its centre: left, right, top, bottom (pixels, y down)
FILE_BOX = (-55.0, 55.0, -70.0, 70.0)

# -- the mouse and its mat --------------------------------------------------------------------
MA, MB, MH = 0.058, 0.031, 0.036  # half length, half width, and the height that sets the top (m): a 116 x 62 x 38 mm mouse
MOUSE_MASS = 0.085
SENSOR = np.array([0.012, 0.0, 0.0])  # under the mouse, in its frame
BTN_X0, BTN_X1, BTN_Y0 = 0.004, 0.054, 0.0025  # the left button's shell, from the wheel's slot to the nose
BTN_TRAVEL, BTN_SPRING = 0.0006, 1500.0  # 0.6 mm to the stop; 0.6 N at the click
CLICK_AT, RELEASE_AT = 0.0004, 0.0003  # metres pressed: the switch closes at 0.4 mm and opens again above 0.3 mm
MAT_C, MAT_SIZE, MAT_T = np.array([0.34, -0.13]), np.array([0.24, 0.26]), 0.003
MOUSE_START = np.array([0.34, -0.13])
MAT_FRICTION = (0.15, 0.12)  # the mouse's PTFE feet on the cloth mat; the mat's "min" combine makes this the friction of everything on it
SHELL_FRICTION = (0.6, 0.5)  # a fingertip on the shell's matte plastic
CLEAR = 0.01  # fingers this far from the mouse count as off it
TIP_DEG = 30.0  # tilted further than this, the mouse is knocked over

LOOKS = {
    "housing": m.Look(color=m.rgb(0.10, 0.10, 0.11), roughness=0.5),
    "stand": m.Look(color=m.rgb(0.16, 0.16, 0.17), metallic=0.3, roughness=0.45),
    "wallpaper": m.Look(color=m.rgb(0.12, 0.20, 0.30), roughness=0.3, emissive=m.rgb(0.10, 0.17, 0.26)),
    "dock": m.Look(color=m.rgb(0.06, 0.08, 0.11), roughness=0.3, emissive=m.rgb(0.05, 0.07, 0.10)),
    "highlight": m.Look(color=m.rgb(0.45, 0.55, 0.70), roughness=0.3, emissive=m.rgb(0.38, 0.48, 0.62)),
    "window": m.Look(color=m.rgb(0.95, 0.95, 0.95), roughness=0.3, emissive=m.rgb(0.85, 0.85, 0.85)),
    "titlebar": m.Look(color=m.rgb(0.75, 0.76, 0.78), roughness=0.3, emissive=m.rgb(0.62, 0.63, 0.65)),
    "file": m.Look(color=m.rgb(0.90, 0.12, 0.10), roughness=0.3, emissive=m.rgb(0.80, 0.08, 0.06)),
    "fold": m.Look(color=m.rgb(0.98, 0.70, 0.66), roughness=0.3, emissive=m.rgb(0.88, 0.60, 0.56)),
    "cursor": m.Look(color=m.rgb(1.0, 1.0, 1.0), roughness=0.3, emissive=m.rgb(0.95, 0.95, 0.95)),
    "outline": m.Look(color=m.rgb(0.0, 0.0, 0.0), roughness=0.3),
    "mouse": m.Look(color=m.rgb(0.20, 0.20, 0.22), roughness=0.45),
    "button": m.Look(color=m.rgb(0.25, 0.25, 0.27), roughness=0.4),
    "wheel": m.Look(color=m.rgb(0.08, 0.08, 0.08), roughness=0.8),
    "mat": m.Look(color=m.rgb(0.16, 0.18, 0.22), roughness=0.95),
}
FOLDER_RGB = {"blue": (0.25, 0.55, 0.98), "green": (0.20, 0.78, 0.30), "yellow": (0.98, 0.80, 0.12), "purple": (0.66, 0.36, 0.90)}
for _c, (_r, _g, _b) in FOLDER_RGB.items():
    LOOKS[f"{_c}_back"] = m.Look(color=m.rgb(0.8 * _r, 0.8 * _g, 0.8 * _b), roughness=0.3, emissive=m.rgb(0.7 * _r, 0.7 * _g, 0.7 * _b))
    LOOKS[f"{_c}_front"] = m.Look(color=m.rgb(_r, _g, _b), roughness=0.3, emissive=m.rgb(0.9 * _r, 0.9 * _g, 0.9 * _b))


def screen_point(px: float, py: float, d: float = 0.0) -> np.ndarray:
    """A point on the monitor's face, in the monitor part's frame: screen pixels from the top left, and `d` metres out of the face."""
    return CENTRE + (px - PX_W / 2) * PITCH * R_AX + (PX_H / 2 - py) * PITCH * U_AX + d * N_AX


def to_pixels(p: np.ndarray) -> tuple[float, float]:
    """The inverse of `screen_point`, for a point on the face."""
    q = np.asarray(p, float) - CENTRE
    return float(q @ R_AX / PITCH + PX_W / 2), float(PX_H / 2 - q @ U_AX / PITCH)


def _on_face(mesh: m.Mesh, origin=np.zeros(3)) -> m.Mesh:
    """A mesh drawn in the screen's own axes (x right, y up, z out of the face), turned onto the leaned monitor."""
    v = mesh.v[:, :1] * R_AX + mesh.v[:, 1:2] * U_AX + mesh.v[:, 2:3] * N_AX + origin
    return m.Mesh(v, mesh.f)


def _flat(outline_px, d: float) -> m.Mesh:
    """A flat shape on the face, from an outline in pixels (x right, y up), 0.1 mm thick at `d` out of the face, about the anchor it is moved by."""
    pts = np.asarray(outline_px, float) * PITCH
    return _on_face(m.prism(pts, d, d + 0.0001))


def _rect(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# The pictures on the screen, each about its anchor, in pixels with y up. Layers, out of the face: wallpaper, highlight, folders, the file, a window, the cursor.
D_WALL, D_DOCK, D_HIGH, D_BACK, D_FRONT, D_FILE, D_FOLD, D_WIN, D_BAR, D_OUT, D_CUR = 0.0003, 0.0005, 0.0007, 0.0010, 0.0012, 0.0016, 0.0018, 0.0022, 0.0024, 0.0028, 0.0030
FOLDER_BACK = [(-75, -60), (75, -60), (75, 55), (-12, 55), (-20, 68), (-75, 68)]
FOLDER_FRONT = _rect(-75, 75, -60, 40)
HIGHLIGHT = _rect(-92, 92, -80, 84)
FILE = [(-55, -70), (55, -70), (55, 40), (25, 70), (-55, 70)]
FOLD = [(25, 70), (25, 40), (55, 40)]
CURSOR_SCALE = 1.6  # the accessibility setting's large pointer: about 145 px tall
CURSOR_HEAD = [(x * CURSOR_SCALE, y * CURSOR_SCALE) for x, y in [(0, 0), (0, -76), (54, -54)]]
CURSOR_TAIL = [(x * CURSOR_SCALE, y * CURSOR_SCALE) for x, y in [(12, -56), (24, -52), (38, -86), (26, -91)]]


def _grow(outline, by: float):
    """An outline pushed out from its centre by about `by` pixels: the cursor's black edge."""
    p = np.asarray(outline, float)
    c = p.mean(axis=0)
    d = p - c
    return c + d * (1.0 + by / np.linalg.norm(d, axis=1, keepdims=True))


def _monitor():
    w, h = PX_W * PITCH, PX_H * PITCH
    bottom = np.array([MON_X, 0.0, MON_Z])
    panel_c = bottom + (h / 2 + (BEZEL - CHIN) / 2) * U_AX - PANEL_D / 2 * N_AX
    panel = _on_face(m.box((w + 2 * BEZEL, h + BEZEL + CHIN, PANEL_D)), panel_c)
    hump_c = CENTRE - 0.03 * N_AX - 0.02 * U_AX
    hump = _on_face(m.box((0.30, 0.18, 0.025)), hump_c)
    foot_x = float(hump_c[0])
    neck = m.box((0.025, 0.06, float(hump_c[2]) - 0.012), (foot_x + 0.01, 0.0, 0.012 + (float(hump_c[2]) - 0.012) / 2))
    foot = m.box((0.17, 0.24, 0.012), (foot_x, 0.0, 0.006))
    wall = _on_face(m.box((w, h, 0.0001), (0.0, 0.0, D_WALL)), CENTRE)
    dock = _on_face(m.box((w * 0.5, 70 * PITCH, 0.0001), (0.0, -h / 2 + 50 * PITCH, D_DOCK)), CENTRE)
    parts = [("panel", panel, "housing"), ("hump", hump, "stand"), ("neck", neck, "stand"), ("foot", foot, "stand"), ("wallpaper", wall, "wallpaper"), ("dock", dock, "dock")]
    for c in COLOURS:
        parts += [(f"high_{c}", _flat(HIGHLIGHT, D_HIGH), "highlight", True), (f"{c}_back", _flat(FOLDER_BACK, D_BACK), f"{c}_back"), (f"{c}_front", _flat(FOLDER_FRONT, D_FRONT), f"{c}_front")]
    parts += [("high_file", _flat([(x * 1.2, y * 1.15) for x, y in FILE], D_HIGH), "highlight", True), ("file", _flat(FILE, D_FILE), "file"), ("fold", _flat(FOLD, D_FOLD), "fold")]
    parts += [("window", _flat(_rect(-480, 480, -300, 300), D_WIN), "window", True), ("titlebar", _flat(_rect(-480, 480, 300, 345), D_BAR), "titlebar", True)]
    parts += [("cursor_out_h", _flat(_grow(CURSOR_HEAD, 10), D_OUT), "outline"), ("cursor_out_t", _flat(_grow(CURSOR_TAIL, 8), D_OUT), "outline"),
              ("cursor_h", _flat(CURSOR_HEAD, D_CUR), "cursor"), ("cursor_t", _flat(CURSOR_TAIL, D_CUR), "cursor")]
    cols = [m.col_hull("panel_c", panel), m.col_hull("hump_c", hump), m.col_box("neck_c", (0.025, 0.06, float(hump_c[2]) - 0.012), (foot_x + 0.01, 0.0, 0.012 + (float(hump_c[2]) - 0.012) / 2)),
            m.col_box("foot_c", (0.17, 0.24, 0.012), (foot_x, 0.0, 0.006))]
    return m.fixed("monitor", (0.0, 0.0, TOP), parts, cols, LOOKS)


MONITOR_BOXES = None  # (centre, axes, half sizes) of the monitor's solid parts, for "touched": filled in below


def _monitor_boxes():
    w, h = PX_W * PITCH, PX_H * PITCH
    bottom = np.array([MON_X, 0.0, MON_Z])
    panel_c = bottom + (h / 2 + (BEZEL - CHIN) / 2) * U_AX - PANEL_D / 2 * N_AX
    hump_c = CENTRE - 0.03 * N_AX - 0.02 * U_AX
    axes = np.stack([R_AX, U_AX, N_AX])
    eye = np.eye(3)
    foot_x = float(hump_c[0])
    neck_h = float(hump_c[2]) - 0.012
    return [(panel_c, axes, np.array([w / 2 + BEZEL, (h + BEZEL + CHIN) / 2, PANEL_D / 2])), (hump_c, axes, np.array([0.15, 0.09, 0.0125])),
            (np.array([foot_x + 0.01, 0.0, 0.012 + neck_h / 2]), eye, np.array([0.0125, 0.03, neck_h / 2])), (np.array([foot_x, 0.0, 0.006]), eye, np.array([0.085, 0.12, 0.006]))]


MONITOR_BOXES = _monitor_boxes()


def touches_monitor(tips_local: np.ndarray, margin: float = 0.003) -> bool:
    """Whether any finger point (in the monitor's frame: the station's, less the table height) is within `margin` of the monitor's solid parts."""
    for c, axes, half in MONITOR_BOXES:
        q = (tips_local - c) @ axes.T
        if np.any(np.all(np.abs(q) <= half + margin, axis=1)):
            return True
    return False


# -- the mouse ----------------------------------------------------------------------------------


EX, EY = 3.0, 5.0  # the shell's plan and profile: rounded along its length, nearly flat across the buttons and steep at the sides, as real mice are


def dome_z(x, y):
    """The mouse's top above its sole: highest under the palm, falling gently over the buttons to the nose."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = np.abs(x / MA) ** EX + np.abs(y / MB) ** EY
    return MH * np.sqrt(np.clip(1.0 - u, 0.0, 1.0)) * (1.0 - 0.22 * x / MA)


def rim(t: float) -> tuple[float, float]:
    """The sole's outline at angle parameter `t`: a superellipse, on which `dome_z` falls to zero."""
    c, s_ = math.cos(t), math.sin(t)
    return MA * math.copysign(abs(c) ** (2.0 / EX), c), MB * math.copysign(abs(s_) ** (2.0 / EY), s_)


def _dome(rings: int = 14, segs: int = 40, drop: float = 0.0, keep=None) -> m.Mesh:
    """The mouse's shell, as a closed mesh with a flat sole: an apex, rings out to the rim at the sole, and the sole. `drop` lowers the top (not below 2 mm), and `keep(x)` keeps a slice of its
    points with their feet on the sole: a point cloud for a hull collider."""
    th = np.linspace(0.0, 2 * np.pi, segs, endpoint=False)
    rho = np.linspace(1.0 / rings, 1.0, rings)
    pts = []
    for r in rho:
        for t in th:
            x, y = rim(t)
            x, y = r * x, r * y
            pts.append((x, y, max(float(dome_z(x, y)) - drop, 0.002 if drop else 0.0)))
    pts = np.array(pts)
    if keep is not None:
        sel = pts[keep(pts[:, 0])]
        return m.Mesh(np.vstack([sel, np.c_[sel[:, :2], np.zeros(len(sel))]]), np.array([[0, 1, 2]]))
    n = len(pts)
    apex, sole = n, n + 1
    v = np.vstack([pts, [[0.0, 0.0, float(dome_z(0.0, 0.0))]], [[0.0, 0.0, 0.0]]])
    faces = []
    for j in range(segs):
        k = (j + 1) % segs
        faces.append((apex, j, k))  # the top cap, facing up
        faces.append((sole, n - segs + k, n - segs + j))  # the sole, facing down
    for i in range(rings - 1):
        for j in range(segs):
            k = (j + 1) % segs
            a, b, c, d = i * segs + j, (i + 1) * segs + j, (i + 1) * segs + k, i * segs + k
            faces += [(a, b, c), (a, c, d)]
    return m.Mesh(v, np.array(faces))


def _hull_points(points: np.ndarray) -> m.Mesh:
    """A point cloud for `col_hull`, which takes the convex hull: three points make a dummy face."""
    return m.Mesh(points, np.array([[0, 1, 2]]))


def _button_shell(side: float, rows: int = 10, cols: int = 6) -> m.Mesh:
    """A button's shell: the top of the shell over x from BTN_X0 to BTN_X1, from the middle slot to near the edge, 2.5 mm thick, 0.3 mm proud of the body."""
    xs = np.linspace(BTN_X0, BTN_X1, rows)
    top, bot = [], []
    for x in xs:
        ymax = 0.85 * MB * max(1.0 - abs(x / MA) ** EX, 0.0) ** (1.0 / EY)
        for y in np.linspace(BTN_Y0, max(ymax, BTN_Y0 + 0.002), cols):
            z = float(dome_z(x, y)) + 0.0003
            top.append((x, side * y, z))
            bot.append((x, side * y, z - 0.0025))
    top, bot = np.array(top), np.array(bot)
    n = len(top)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, d = i * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1, (i + 1) * cols + j
            if side > 0:
                faces += [(a, c, b), (a, d, c), (n + a, n + b, n + c), (n + a, n + c, n + d)]
            else:
                faces += [(a, b, c), (a, c, d), (n + a, n + c, n + b), (n + a, n + d, n + c)]
    ring = [i * cols for i in range(rows)] + [(rows - 1) * cols + j for j in range(1, cols)] + [i * cols + cols - 1 for i in range(rows - 2, -1, -1)] + [j for j in range(cols - 2, 0, -1)]
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        faces += [(a, b, n + b), (a, n + b, n + a)] if side < 0 else [(b, a, n + a), (b, n + a, n + b)]
    return m.Mesh(np.vstack([top, bot]), np.array(faces))


def _mouse():
    body = _dome()
    wheel = m.cylinder(0.0095, 0.007, (0.030, 0.0, float(dome_z(0.030, 0.0)) - 0.0062), axis="y", segments=24)
    right = _button_shell(-1.0)
    left = _button_shell(1.0)
    rear = _dome(keep=lambda x: x <= BTN_X0)
    front = _dome(drop=0.006, keep=lambda x: x >= BTN_X0 - 0.002)
    frame_parts = [("body", body, "mouse"), ("wheel", wheel, "wheel"), ("right_button", right, "button")]
    frame_cols = [m.col_hull("rear_c", rear), m.col_hull("front_c", front), m.col_hull("right_c", _hull_points(right.v))]
    button = m.slider("lbtn", "Z", -BTN_TRAVEL, 0.0, [("shell", left, "button")], [m.col_hull("lbtn_c", _hull_points(left.v))], at=(0.03, 0.015, 0.03),
                      spring=BTN_SPRING, damping=0.5, friction=0.0, mass=0.004, armature=1e-5, effort=50.0)
    inertia = (MOUSE_MASS / 12 * ((2 * MB) ** 2 + MH ** 2), MOUSE_MASS / 12 * ((2 * MA) ** 2 + MH ** 2), MOUSE_MASS / 12 * ((2 * MA) ** 2 + (2 * MB) ** 2))
    return m.mechanism("mouse", (*MOUSE_START, TOP + MAT_T), [button], LOOKS, frame_parts=frame_parts, frame_colliders=frame_cols, friction=SHELL_FRICTION,
                       free_mass=MOUSE_MASS, frame_com=(0.0, 0.0, 0.014), frame_inertia=inertia)


def _mat():
    sx, sy = MAT_SIZE / 2
    r = 0.012
    outline = []
    for cx, cy, a0 in ((sx - r, sy - r, 0), (-sx + r, sy - r, 90), (-sx + r, -sy + r, 180), (sx - r, -sy + r, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, 5)):
            outline.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    mat = m.prism(outline, 0.0, MAT_T)
    return m.fixed("mat", (*MAT_C, TOP), [("cloth", mat, "mat")], [m.col_box("mat_c", (*MAT_SIZE, MAT_T), (0.0, 0.0, MAT_T / 2))], LOOKS, friction=MAT_FRICTION, combine="min")


OBJECTS = [_monitor(), _mat(), _mouse()]


# -- the desktop's behaviour ----------------------------------------------------------------------


def layout_for(seed: int | None) -> dict:
    """Where everything starts, by seed: the folders' order in the row, the file, the cursor, and the mouse on its mat."""
    rng = np.random.default_rng(0 if seed is None else int(seed))
    order = [COLOURS[i] for i in rng.permutation(len(COLOURS))]
    file = np.array([rng.uniform(300.0, 1620.0), rng.uniform(720.0, 860.0)])
    while True:
        cursor = np.array([rng.uniform(250.0, 1670.0), rng.uniform(420.0, 940.0)])
        if np.linalg.norm(cursor - file) > 200.0:
            break
    mouse = MOUSE_START + rng.uniform(-0.02, 0.02, 2)
    yaw = float(rng.uniform(-8.0, 8.0))
    return {"order": order, "folders": {c: np.array([SLOTS_X[i], ROW_Y]) for i, c in enumerate(order)}, "file": file, "cursor": cursor, "mouse": mouse, "yaw_deg": yaw}


def inside(point, centre, box) -> bool:
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    return bool(box[0] <= dx <= box[1] and box[2] <= dy <= box[3])


def task_score(failed: bool, in_target: bool, end_held: bool, progress: float) -> float:
    """The scoring text, line by line. `progress`: how much nearer the file's centre rests to the green folder's centre than it started, in a straight line, as a share of the
    start distance; 0 while the file is still being dragged."""
    if failed:
        return 0.0
    if in_target:
        return 1.0 if end_held else 0.5
    return 0.25 if progress >= 0.5 else 0.0


class Trial:
    """The computer's side: the cursor, the button's clicks, the file's drag and drop; and the trial's latches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, env=None) -> None:
        self.lay = layout_for(m.run_seed())
        self.cursor = self.lay["cursor"].copy()
        self.file = self.lay["file"].copy()
        self.folders = {c: p.copy() for c, p in self.lay["folders"].items()}
        self.file_in: str | None = None
        self.down = False
        self.grab: str | None = None  # what the press landed on: "file", a colour, or None
        self.grab_at = np.zeros(2)
        self.grab_off = np.zeros(2)
        self.dragging = False
        self.last_click = (-9.0, np.zeros(2))
        self.opened: str | None = None
        self.failed: str | None = None
        self.sensor_prev: np.ndarray | None = None
        self.t = 0.0
        self.held = 0.0
        self.ready = self.full = False
        self.score = 0.0
        self.report: dict = {}
        self.drawn: dict = {}
        if env is not None:
            self._place_mouse(env)
            self.draw()

    # the pure part, which the offline check drives with hand-built states

    def fail(self, why: str) -> None:
        if self.failed is None:
            self.failed = f"{why} at {self.t:.2f} s"
            print(f"[mouse] {self.failed}", flush=True)

    def over(self, point) -> str | None:
        """The icon under a screen point: the file, or a folder's colour."""
        if self.file_in is None and inside(point, self.file, FILE_BOX):
            return "file"
        for c, p in self.folders.items():
            if inside(point, p, FOLDER_BOX):
                return c
        return None

    def update(self, mouse_pos, mouse_rot, pressed_m: float, tips, dt: float) -> None:
        """One control step. `mouse_pos`, `mouse_rot`: the mouse's origin (station frame) and rotation; `pressed_m`: how far its left button is in; `tips`: finger points (station frame)."""
        self.t += dt
        mouse_pos, mouse_rot = np.asarray(mouse_pos, float), np.asarray(mouse_rot, float)
        sensor = mouse_pos + mouse_rot @ SENSOR
        lift_mm = (float(sensor[2]) - (TOP + MAT_T)) * 1000.0
        if self.sensor_prev is not None and lift_mm <= LIFT_MM:
            d = mouse_rot.T @ (sensor - self.sensor_prev)  # in the mouse's own axes: forward moves the cursor up, left moves it left
            self.cursor = np.clip(self.cursor + GAIN * 1000.0 * np.array([-d[1], -d[0]]), [0.0, 0.0], [PX_W - 1.0, PX_H - 1.0])
        self.sensor_prev = sensor
        was_down = self.down
        self.down = pressed_m >= CLICK_AT or (was_down and pressed_m > RELEASE_AT)
        if self.down and not was_down:
            self._press()
        elif was_down and not self.down:
            self._release()
        elif self.down and self.grab is not None:
            if not self.dragging and np.linalg.norm(self.cursor - self.grab_at) >= DRAG_PX:
                self.dragging = True
                if self.grab != "file":
                    self.fail(f"the {self.grab} folder dragged")
            if self.dragging:
                if self.grab == "file":
                    self.file = self.cursor + self.grab_off
                else:
                    self.folders[self.grab] = self.cursor + self.grab_off
        off_mat = np.any(np.abs(mouse_pos[:2] - MAT_C) > MAT_SIZE / 2)
        tilt = math.degrees(math.acos(float(np.clip(mouse_rot[2, 2], -1.0, 1.0))))
        if off_mat or tilt > TIP_DEG:
            self.fail("the mouse knocked off its mat" if off_mat else f"the mouse tipped {tilt:.0f} degrees")
        tips = np.asarray(tips, float)
        if touches_monitor(tips - np.array([0.0, 0.0, TOP])):
            self.fail("the monitor touched")
        local = (tips - mouse_pos) @ mouse_rot
        on_mouse = bool(np.any((np.abs(local[:, 0]) <= MA + CLEAR) & (np.abs(local[:, 1]) <= MB + CLEAR) & (local[:, 2] <= MH + CLEAR)))
        in_target = self.file_in == TARGET
        self.ready = bool(in_target and not self.down and not on_mouse and not off_mat and tilt <= TIP_DEG)
        self.held = self.held + dt if self.ready else 0.0
        self.full = bool(self.ready and self.failed is None)
        start = np.linalg.norm(self.lay["file"] - self.folders[TARGET])
        now = 0.0 if in_target else float(np.linalg.norm(self.file - self.folders[TARGET]))
        let_go = not (self.grab == "file" and self.dragging)  # a file still being dragged has not been put anywhere
        progress = 1.0 - now / max(start, 1.0) if let_go else 0.0
        self.score = task_score(self.failed is not None, in_target, self.held >= SUCCESS_HOLD_S - 1e-9, progress)
        self.report = {"t": round(self.t, 2), "cursor": [round(float(v)) for v in self.cursor], "file": None if self.file_in else [round(float(v)) for v in self.file], "file_in": self.file_in,
                       "down": self.down, "pressed_mm": round(pressed_m * 1000, 3), "lift_mm": round(lift_mm, 2), "dragging": self.dragging, "on_mouse": on_mouse, "tilt_deg": round(tilt, 1),
                       "failed": self.failed, "progress": round(progress, 3), "ready": self.ready, "held_s": round(self.held, 2), "score": self.score}

    def _press(self) -> None:
        under = self.over(self.cursor)
        t0, p0 = self.last_click
        if under is not None and self.t - t0 <= DOUBLE_S and np.linalg.norm(self.cursor - p0) <= DOUBLE_PX:
            self.opened = under
            self.fail(f"the {'file' if under == 'file' else under + ' folder'} opened by a double click")
        self.last_click = (self.t, self.cursor.copy())
        self.grab, self.grab_at, self.dragging = under, self.cursor.copy(), False
        if under == "file":
            self.grab_off = self.file - self.cursor
        elif under is not None:
            self.grab_off = self.folders[under] - self.cursor

    def _release(self) -> None:
        if self.grab == "file" and self.dragging:
            for c, p in self.folders.items():
                if inside(self.cursor, p, FOLDER_BOX):
                    self.file_in = c
                    print(f"[mouse] the file dropped into the {c} folder at {self.t:.2f} s", flush=True)
                    if c != TARGET:
                        self.fail(f"the file dropped into the {c} folder")
                    break
            else:
                self.file = np.clip(self.file, [60.0, 75.0], [PX_W - 60.0, PX_H - 75.0])
        self.grab, self.dragging = None, False

    def hover(self) -> str | None:
        """The folder the dragged file is over: its highlight shows."""
        if self.grab == "file" and self.dragging:
            for c, p in self.folders.items():
                if inside(self.cursor, p, FOLDER_BOX):
                    return c
        return None

    # the simulator's part

    def _place_mouse(self, env) -> None:
        import torch

        base = env.unwrapped
        asset = base.scene["mouse"]
        pose = asset.data.root_link_pose_w.torch.clone()
        h = math.radians(self.lay["yaw_deg"]) / 2
        origin = base.scene.env_origins[0]
        pose[:, 0], pose[:, 1], pose[:, 2] = origin[0] + float(self.lay["mouse"][0]), origin[1] + float(self.lay["mouse"][1]), origin[2] + TOP + MAT_T
        pose[:, 3:7] = torch.tensor([0.0, 0.0, math.sin(h), math.cos(h)], device=pose.device)
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=asset.data.root_vel_w.torch * 0.0)
        self.sensor_prev = None

    @staticmethod
    def _move(name: str, at: np.ndarray) -> None:
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom

        prim = sim_utils.get_current_stage().GetPrimAtPath(f"{m.ENV_PATH}/monitor/visual/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"no prim at monitor/visual/{name}")
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        op = ops[0] if ops else xf.AddTranslateOp()
        op.Set(Gf.Vec3d(float(at[0]), float(at[1]), float(at[2])))

    def draw(self) -> None:
        """Put the pictures where the computer would draw them; only what changed."""
        want = {"cursor": tuple(np.round(self.cursor, 2)), "file": None if self.file_in else tuple(np.round(self.file, 2)), "hover": self.hover(), "selected": self.grab == "file" and self.down,
                "opened": self.opened, **{c: tuple(np.round(p, 2)) for c, p in self.folders.items()}}
        old = self.drawn
        if old.get("cursor") != want["cursor"]:
            p = screen_point(*self.cursor)
            for n in ("cursor_out_h", "cursor_out_t", "cursor_h", "cursor_t"):
                self._move(n, p)
        if old.get("file", "x") != want["file"]:
            for n in ("file", "fold", "high_file"):
                m.set_visible(f"{m.ENV_PATH}/monitor/visual/{n}", want["file"] is not None and (n != "high_file" or want["selected"]))
            if want["file"] is not None:
                for n in ("file", "fold", "high_file"):
                    self._move(n, screen_point(*self.file))
        m.set_visible(f"{m.ENV_PATH}/monitor/visual/high_file", want["file"] is not None and want["selected"])
        for c in COLOURS:
            if old.get(c) != want[c]:
                for n in (f"high_{c}", f"{c}_back", f"{c}_front"):
                    self._move(n, screen_point(*self.folders[c]))
            m.set_visible(f"{m.ENV_PATH}/monitor/visual/high_{c}", want["hover"] == c)
        if old.get("opened") != want["opened"] and want["opened"] is not None:
            for n in ("window", "titlebar"):
                self._move(n, screen_point(PX_W / 2, PX_H / 2 + 40))
                m.set_visible(f"{m.ENV_PATH}/monitor/visual/{n}", True)
        if want["opened"] is None:
            for n in ("window", "titlebar"):
                m.set_visible(f"{m.ENV_PATH}/monitor/visual/{n}", False)
        self.drawn = want

    def step(self, env) -> None:
        s = m.state(env, "mouse")
        self.update(s.pos, s.rotation(), -s.joint("lbtn"), m.fingertips(env), float(env.unwrapped.step_dt))
        self.draw()
        if int(round(self.t * 20)) % 10 == 0:
            d = os.environ.get("MOUSE_SCORE_DIR", "mouse_scores")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{os.environ.get('TASK_DEMO', 'run')}.json"), "w") as f:
                json.dump(self.report, f, indent=0)


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)
m.settle_after_reset(6)  # the seeded mouse and the drawn desktop show from the first picture on


def success(objects) -> bool:
    return TRIAL.full


def score(objects) -> float:
    """See `task_score`."""
    return TRIAL.score


# -- the demo ------------------------------------------------------------------------------------

PRESS_AT = np.array([0.012, 0.012])  # on the left button, where its top is nearly level (10 and 6 degrees), in the mouse's frame
PALM_AT = np.array([-0.016, 0.0])  # the top of the palm hump, where the shell is level: pushed to move the mouse without clicking
TIP_BELOW_TCP = 0.012  # the closed fingertips reach this far below the harness's TCP (measured on the palm in a first run)
PALM_DEPTH, PRESS_DEPTH = 0.004, 0.0025  # how far the fingertips are commanded into the shell: the arm's stiffness turns this into the push


GRID_STEP = 8  # screen pixels between the points the demo samples in the camera's picture of the screen
WHITE_SAMPLES = (6, 400)  # how many samples the pointer's white covers, about 90; the arms' white covers are larger, or smaller where they only clip the screen


def _grid():
    xs, ys = np.arange(GRID_STEP / 2, PX_W, GRID_STEP), np.arange(GRID_STEP / 2, PX_H, GRID_STEP)
    gx, gy = np.meshgrid(xs, ys)
    world = np.array([screen_point(x, y, 0.004) for x, y in zip(gx.ravel(), gy.ravel())]) + np.array([0.0, 0.0, TOP])
    return gx, gy, world


def _components(mask: np.ndarray) -> list:
    """Groups of touching True cells in a 2-D mask, each a list of (row, col)."""
    seen = np.zeros_like(mask, bool)
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(cells)
    return out


class ScreenReader:
    """Reads the monitor through the top camera, as a person looking at it would: it samples the camera's picture where each screen pixel of a grid shows, and finds the pointer
    (a small white arrow: its tip is the white's upper-left extreme), the red file and the green folder by colour. Nothing is read from the scene's state."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.gx, self.gy, self.world = _grid()
        self.pred = None  # where the pointer should be, from the last sighting and the mouse's motion since

    def look(self) -> dict:
        rgb = self.ctx.see(self.world).reshape(*self.gx.shape, 3)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        white = (rgb.min(axis=-1) > 190) & (rgb.max(axis=-1) - rgb.min(axis=-1) < 40)
        red = (r > 170) & (g < 100) & (b < 100)
        green = (g > 180) & (g - r > 80) & (g - b > 60)

        def centre(mask):
            if mask.sum() < 4:
                return None
            return np.array([self.gx[mask].mean(), self.gy[mask].mean()])

        best, best_d = None, None
        for cells in _components(white):
            if not WHITE_SAMPLES[0] <= len(cells) <= WHITE_SAMPLES[1]:
                continue
            rows, cols = np.array(cells).T
            px, py = self.gx[rows, cols], self.gy[rows, cols]
            k = int(np.argmin(px + py))
            tip = np.array([px[k], py[k]])
            d = 0.0 if self.pred is None else float(np.linalg.norm(tip - self.pred))
            if best is None or d < best_d or (self.pred is None and len(cells) > best[1]):
                best, best_d = (tip, len(cells)), d
        cursor = None
        if best is not None and (self.pred is None or best_d < 250.0):
            cursor = best[0]
        elif self.pred is not None:
            cursor = self.pred.copy()  # hidden behind an arm: dead reckoning from the mouse's motion
            print(f"[mouse] pointer not seen; taking it at {np.round(cursor)} from the mouse's motion", flush=True)
        if cursor is not None:
            self.pred = cursor.copy()
        return {"cursor": cursor, "file": centre(red), "green": centre(green)}


def demo_script(ctx):
    """One arm. A scripted reference: it reads the screen through the top camera (ScreenReader), and the mouse's pose, which is in view, from the simulator. A policy gets only the cameras.
    It never reads the seed, the button's joint or the scorer. It pushes the palm to bring the pointer onto the file, presses the left button, drags by the button, checking that the file comes
    along, until the pointer is on the green folder, and lifts off. `TASK_DEMO`: full; still (no motion, for stills)."""
    import demokit as d
    from yam_arena.constants import FINGER_CLOSED

    r = ctx.right
    if ctx.mode == "still":
        yield from r.wait(1.0)
        return
    eye = ScreenReader(ctx)

    def mouse():
        s = ctx.state("mouse")
        return s.pos, s.rotation()

    def point(local_xy, above: float) -> np.ndarray:
        """Where the TCP goes for the fingertips to stand `above` the shell at `local_xy` (negative: pressed in)."""
        pos, rot = mouse()
        x, y = local_xy
        return pos + rot @ np.array([x, y, float(dome_z(x, y)) + above]) + np.array([0.0, 0.0, TIP_BELOW_TCP])

    def yaw_deg() -> float:
        _, rot = mouse()
        return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))

    def push(err_px, label: str):
        """Move the finger, and the mouse under it, by what should carry the pointer `err_px`; then tell the reader where the pointer should now be."""
        pos0, rot = mouse()
        move_m = np.array([-err_px[1], -err_px[0], 0.0]) / (GAIN * 1000.0)  # pointer right is the mouse's -y, pointer down is its -x
        step = rot @ move_m
        step[2] = 0.0
        dist = float(np.linalg.norm(step))
        yield from r.move(r.at() + step, t=max(0.6, dist / 0.03), label=label)
        yield from r.wait(0.25)
        pos1, rot1 = mouse()
        moved = rot1.T @ (pos1 - pos0)
        if eye.pred is not None:
            eye.pred = np.clip(eye.pred + GAIN * 1000.0 * np.array([-moved[1], -moved[0]]), 0.0, [PX_W - 1.0, PX_H - 1.0])

    def slide(goal_px, label: str, tol_px: float, rounds: int = 6, first=None):
        """Push until the pointer is within `tol_px` of `goal_px`. `first`, if given, caps the first push, in pixels."""
        for k in range(rounds):
            seen = eye.look()
            if seen["cursor"] is None:
                print(f"[mouse] {label} {k}: no pointer seen", flush=True)
                yield from r.wait(0.25)
                continue
            err = np.asarray(goal_px, float) - seen["cursor"]
            if np.linalg.norm(err) <= tol_px:
                return True
            if k == 0 and first is not None and np.linalg.norm(err) > first:
                err = err * first / np.linalg.norm(err)
            if np.linalg.norm(err) < 100.0:
                err = err * 1.6  # a short push is mostly taken up by the finger's give on the shell
            print(f"[mouse] {label} {k}: pointer {np.round(seen['cursor'])}, goal {np.round(goal_px)}, push {np.round(err)} px", flush=True)
            yield from push(err, f"{label} {k}")
        return bool(np.linalg.norm(np.asarray(goal_px, float) - eye.look()["cursor"]) <= tol_px)

    g = d.down(yaw_deg())
    seen = eye.look()
    goal = seen["green"]
    print(f"[mouse] the camera sees: pointer {seen['cursor']}, file {seen['file']}, green folder {goal}", flush=True)
    yield from r.move(point(PALM_AT, 0.04), grip=g, finger=FINGER_CLOSED, t=2.5, label="over the palm")
    yield from r.servo(point(PALM_AT, 0.04), timeout=1.0, label="over the palm")
    for attempt in range(3):
        # 1. push the palm, button up, to put the pointer on the file
        file = eye.look()["file"]
        yield from r.move(point(PALM_AT, 0.02), t=0.8, label="over the palm")
        yield from r.move(point(PALM_AT, -PALM_DEPTH), t=0.8, label="onto the palm")
        yield from slide(file, "pointer to the file", tol_px=40.0)
        yield from r.move(r.at() + np.array([0.0, 0.0, 0.02]), t=0.5, label="off the palm")
        # 2. press the left button and hold it, and start the drag: the file must come along with the pointer
        yield from r.move(point(PRESS_AT, 0.015), t=0.8, label="over the button")
        yield from r.move(point(PRESS_AT, -PRESS_DEPTH), t=0.6, label="press")
        yield from r.wait(0.3)
        before = eye.look()
        yield from push(np.clip(goal - before["cursor"], -150.0, 150.0), "start the drag")
        after = eye.look()
        moved_p = after["cursor"] - before["cursor"]
        moved_f = None if after["file"] is None or before["file"] is None else after["file"] - before["file"]
        if moved_f is not None and np.linalg.norm(moved_p) > 30.0 and np.linalg.norm(moved_f - moved_p) < 0.4 * np.linalg.norm(moved_p):
            break
        print(f"[mouse] attempt {attempt}: the file did not follow the pointer (pointer moved {np.round(moved_p)}, file {moved_f}); up, and again", flush=True)
        yield from r.move(r.at() + np.array([0.0, 0.0, 0.02]), t=0.5, label="up")
    # 3. drag on to the green folder, by the button
    yield from slide(goal, "drag", tol_px=45.0)
    # 4. lift off: the button lets go and the file drops
    yield from r.move(r.at() + np.array([0.0, 0.0, 0.04]), t=0.8, label="lift off")
    yield from r.move(r.at() + np.array([-0.08, -0.05, 0.06]), t=1.2, label="away")
