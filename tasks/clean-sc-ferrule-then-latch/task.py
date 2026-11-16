"""On the station table: an SC/UPC fibre patch cord (blue connector, yellow 3 mm cord), a cassette end-face cleaner
and a short patch-panel section with one SC adapter.

- The cord's far end is held in a black cord clip on the table (x 16 cm, y -12 cm; it runs on to the switch). The
  blue SC/UPC connector lies loose on the table near x 44 cm, y -8 cm, pointing away from the robot. The connector
  is true size: a 9 x 7.4 mm housing 25 mm long with a white 2.5 mm ceramic ferrule protruding 2 mm from its front,
  a latch notch on each side, and a black boot.
- A grey cassette cleaner (125 x 85 x 35 mm, 150 g, the size of a CLETOP-S) stands loose on its long edge on the
  left (centre x 30 cm, y +25 cm), turned 60 degrees so its front face looks back towards the robot and to the
  left. Near the top of that face, towards one end, is a window of white cleaning tape (30 x 10 mm, from 52 to
  82 mm above the table) with an arrow beside it pointing down, the wiping direction (top to bottom).
- On the right is a 1U patch-panel section (120 mm wide) standing on two posts. Its face is at x 42 cm and looks at
  the robot. One SC adapter (beige, 27.4 mm long) is mounted in it, port centre at y -20 cm, 6 cm above the
  table. A second blue connector is already latched into the back of the adapter. The adapter's entry has a 1 mm
  45-degree chamfer and 0.25 mm of clearance per side around the housing. Two spring-loaded detent pins in its
  side walls snap into the connector's side notches when it is pushed fully home.

Goal: clean the blue connector's ferrule end face on the cleaner's tape, then latch the connector into the panel's
SC adapter and let go, keeping the cleaned end face off everything but the adapter's sleeve.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score \\
           --instruction "<INSTRUCTION below>"
Test-only demo variants (env var SC_DEMO): correct (default) | nothing | dirty | touch | short | hover | hold |
clean_only | no_latch.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cablekit  # noqa: E402,F401  (hooks yam_arena so the Cable below is simulated)
import numpy as np  # noqa: E402
from cablekit import Plug, cable  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"

INSTRUCTION = "Clean the blue fibre connector on the cleaner, then latch it into the panel's SC adapter."

SC_BLUE = (0.10, 0.33, 0.85)
YELLOW = (0.95, 0.80, 0.08)
CERAMIC = (0.97, 0.96, 0.92)
BLACK = (0.06, 0.06, 0.07)
CASSETTE_GREY = (0.32, 0.34, 0.37)
TAPE = (0.98, 0.98, 0.97)
BEIGE = (0.86, 0.80, 0.66)
STEEL = (0.62, 0.64, 0.67)
DARK = (0.20, 0.21, 0.23)

# -- SC/UPC connector (IEC 61754-4 plug; 9 x 7.4 mm housing 25 mm long, 2.5 mm ferrule) -------------------------------
# Along the connector axis, s = 0 at the ferrule tip, negative towards the cord:
#   ferrule  s -2 .. 0     2.5 mm (collider 2.5 mm square, drawn round)
#   front    s -6 .. -2    9.0 x 7.4 mm
#   notch    s -9 .. -6    7.8 x 7.4 mm (a 0.6 mm latch notch in each side)
#   grip     s -25 .. -9   9.0 x 7.4 mm (the cable kit's end plug body)
#   boot     s -35 .. -25  6 x 6 mm collider (drawn tapering on to s -50)
HW, HH = 0.0090, 0.0074  # housing width (y in the plug frame) and height (z)
FER_D, FER_L = 0.0025, 0.0020
GRIP_L = 0.016
PLUG_TIP = GRIP_L / 2 + 0.003 + 0.004 + FER_L  # plug-body centre -> ferrule tip (17 mm)
NOTCH_W = 0.0078
CORD_R = 0.0015  # 3 mm patch cord
CLIP = Plug(size=(0.020, 0.012, 0.010), mass=0.010, color=BLACK)  # the cord clip (anchored)
A0 = np.array([0.16, -0.12])  # cord clip
CORD_YAW = 0.15
CORD_LEN = 0.28
CORD = cable("fibre", start=(A0[0], A0[1], TOP + CLIP.size[2] / 2 + 0.0005), yaw=CORD_YAW, length=CORD_LEN,
             radius=CORD_R, anchor_start=True, start_plug=CLIP,
             end_plug=Plug(size=(GRIP_L, HW, HH), mass=0.008, color=SC_BLUE), color=YELLOW,
             cable_mass=0.008, bend_stiffness=0.004, bend_damping=0.0005)
# NB: the kit lays the cord at the start plug's height; the 7.4 mm connector settles onto the table at the start.


def _plug_parts() -> list[tuple[str, tuple, tuple, str, bool]]:
    """(name, centre in the plug-body frame, size, look, collides) of the parts added to the end plug."""
    g = GRIP_L / 2
    return [
        ("notch", (g + 0.0015, 0.0, 0.0), (0.003, NOTCH_W, HH), "blue", True),
        ("front", (g + 0.005, 0.0, 0.0), (0.004, HW, HH), "blue", True),
        ("ferrule", (g + 0.007 + FER_L / 2, 0.0, 0.0), (FER_L, FER_D, FER_D), "ceramic", True),
        ("boot", (-g - 0.005, 0.0, 0.0), (0.010, 0.006, 0.006), "black", True),
        ("boot_tail", (-g - 0.0175, 0.0, 0.0), (0.015, 0.0045, 0.0045), "black", False),
        ("key", (0.0, 0.0, HH / 2 + 0.00045), (0.012, 0.0018, 0.0009), "blue", False),
    ]


# -- cassette cleaner (CLETOP-S size: 125 x 85 x 35 mm, 150 g), standing on its long edge ------------------------------
CAS = (0.035, 0.125, 0.085)  # x (thickness), y (length), z (height)
CAS_YAW = math.radians(-60.0)  # turned so its window looks at the right arm, 60 deg to the right of straight back
CAS_XY0 = (0.30, 0.25)  # start (left; out of the right arm's reach)
CAS_XY1 = (0.30, 0.0)  # where the demo's left arm holds it (any place works)
CASSETTE = Box("cleaner", pos=(CAS_XY0[0], CAS_XY0[1], TOP + CAS[2] / 2 + 0.0005), size=CAS, color=CASSETTE_GREY,
               mass=0.15, yaw=CAS_YAW)
# tape window in the cassette frame (x out of the back, y along its length, z up; origin at its centre)
WIN_Y = (-0.055, -0.045)  # 10 mm wide, near the cleaner's right-hand end
WIN_Z = (0.052 - CAS[2] / 2, 0.082 - CAS[2] / 2)  # 30 mm tall, 52 to 82 mm up; the arrow points down
WIN_LEN = WIN_Z[1] - WIN_Z[0]
ARROW = np.array([0.0, 0.0, -1.0])  # wiping direction in the cleaner's frame (top to bottom)

# -- 1U patch-panel section with one SC simplex adapter (Hirose HSC-A3 type: body 27.4 mm long, 12.8 x 9.4 mm) --------
PORT_Y, PORT_Z = -0.20, TOP + 0.060
AF = 0.415  # adapter front face (x); the connector goes in along +x
PANEL_X = AF + 0.0048  # panel front face: the adapter stands 3 mm + its 1.8 mm flange proud of the panel
PANEL_T = 0.0016
CH_W, CH_H = HW + 0.0005, HH + 0.0005  # channel: 0.25 mm per side round the housing
CHAMF = 0.001  # 1 mm x 45 deg lead-in
AD_W, AD_H, AD_L = 0.0128, 0.0094, 0.0274
STOP_D = 0.0115  # housing front stops here (depth from the adapter face): ferrule tip at 13.5 mm
MID_D = AD_L / 2  # sleeve midpoint: the far connector's ferrule tip (13.7 mm)
SLEEVE_ID = FER_D + 0.0007  # rigid bore: 0.35 mm per side (a real split sleeve grips the ferrule)
PIN_R, PIN_H = 0.0010, 0.0050
PIN_D = 0.006  # pin centre depth: the notch (s -9..-6) sits over it when the housing is home
PIN_REST = CH_W / 2 - 0.00055 + PIN_R  # pin centre, lateral, at rest: protrudes 0.55 mm into the channel
PIN_TRAVEL = 0.0015
PIN_K, PIN_PRE = 500.0, 0.002  # N/m; drive target 2 mm past the rest stop (1 N preload per pin)
PINS = [Cylinder(f"detent_{n}", pos=(AF + PIN_D, PORT_Y + sg * PIN_REST, PORT_Z), radius=PIN_R, height=PIN_H,
                 color=(0.75, 0.75, 0.78), mass=0.003) for n, sg in (("l", 1), ("r", -1))]


def adapter_boxes() -> list[tuple[str, tuple, tuple, tuple, str]]:
    """Static colliders of the adapter and panel: (name, centre, size, rot (w, x, y, z), look). World frame."""
    out = []
    xy, z = PORT_Y, PORT_Z
    wall_s = (AD_W - CH_W) / 2  # side wall thickness 1.65 mm
    wall_t = (AD_H - CH_H) / 2  # top/bottom wall 0.75 mm
    q0 = (1.0, 0.0, 0.0, 0.0)
    d0, d1 = CHAMF, STOP_D
    # top / bottom walls (behind the chamfer)
    for n, sg in (("top", 1), ("bot", -1)):
        out.append((f"ad_{n}", (AF + (d0 + d1) / 2, xy, z + sg * (CH_H + wall_t) / 2), (d1 - d0, AD_W, wall_t), q0,
                    "beige"))
    # side walls, split round the detent-pin slots (d 4.5 .. 7.5 mm)
    for n, sg in (("l", 1), ("r", -1)):
        yc = xy + sg * (CH_W + wall_s) / 2
        for k, (a, b) in enumerate(((d0, PIN_D - 0.0015), (PIN_D + 0.0015, d1))):
            out.append((f"ad_side_{n}{k}", (AF + (a + b) / 2, yc, z), (b - a, wall_s, CH_H), q0, "beige"))
        # slot backs (outside the pin's travel)
        out.append((f"ad_slotback_{n}", (AF + PIN_D, xy + sg * (PIN_REST + PIN_TRAVEL + PIN_R + 0.0004), z),
                    (0.003, 0.0006, CH_H), q0, "beige"))
    # 45-degree lead-in plates (1 mm deep) round the mouth
    t = 0.0008
    ln = CHAMF * math.sqrt(2)
    for n, sg in (("l", 1), ("r", -1)):  # rotate about z
        ang = -sg * math.pi / 4
        h = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
        off = CH_W / 2 + CHAMF / 2 + t / 2 * math.sqrt(0.5)
        cx = AF + CHAMF / 2 + t / 2 * math.sqrt(0.5)
        out.append((f"ad_ch_{n}", (cx, xy + sg * off, z), (ln, t, CH_H + 2 * CHAMF), h, "beige"))
    for n, sg in (("top", 1), ("bot", -1)):  # rotate about y
        ang = sg * math.pi / 4
        h = (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)
        off = CH_H / 2 + CHAMF / 2 + t / 2 * math.sqrt(0.5)
        cx = AF + CHAMF / 2 + t / 2 * math.sqrt(0.5)
        out.append((f"ad_ch_{n}", (cx, xy, z + sg * off), (ln, CH_W + 2 * CHAMF, t), h, "beige"))
    # front rim (the adapter's face outside the chamfer)
    out.append(("ad_rim_l", (AF + 0.0005, xy + (CH_W / 2 + CHAMF + AD_W / 2) / 2 + 0.0, z),
                (0.001, AD_W / 2 - CH_W / 2 - CHAMF, AD_H), q0, "beige"))
    out.append(("ad_rim_r", (AF + 0.0005, xy - (CH_W / 2 + CHAMF + AD_W / 2) / 2, z),
                (0.001, AD_W / 2 - CH_W / 2 - CHAMF, AD_H), q0, "beige"))
    # stop wall with the sleeve hole (4 pieces round a SLEEVE_ID square)
    sw = 0.001
    hw = SLEEVE_ID / 2
    out.append(("ad_stop_l", (AF + d1 + sw / 2, xy + (hw + CH_W / 2) / 2, z), (sw, CH_W / 2 - hw, CH_H), q0, "beige"))
    out.append(("ad_stop_r", (AF + d1 + sw / 2, xy - (hw + CH_W / 2) / 2, z), (sw, CH_W / 2 - hw, CH_H), q0, "beige"))
    out.append(("ad_stop_t", (AF + d1 + sw / 2, xy, z + (hw + CH_H / 2) / 2), (sw, 2 * hw, CH_H / 2 - hw), q0, "beige"))
    out.append(("ad_stop_b", (AF + d1 + sw / 2, xy, z - (hw + CH_H / 2) / 2), (sw, 2 * hw, CH_H / 2 - hw), q0, "beige"))
    # sleeve bore (rigid, square) from the stop wall to the midpoint, then the far connector's ferrule
    st = 0.0006
    sl0, sl1 = d1 + sw, MID_D + 0.002
    for n, dy, dz, sz in (("l", hw + st / 2, 0, (st, 2 * hw + 2 * st)), ("r", -hw - st / 2, 0, (st, 2 * hw + 2 * st)),
                          ("t", 0, hw + st / 2, (2 * hw, st)), ("b", 0, -hw - st / 2, (2 * hw, st))):
        out.append((f"sleeve_{n}", (AF + (sl0 + sl1) / 2, xy + dy, z + dz), (sl1 - sl0, *sz), q0, "ceramic"))
    out.append(("far_ferrule", (AF + MID_D + 0.001, xy, z), (0.002, FER_D, FER_D), q0, "ceramic"))
    # flange (22 mm tall, 1.8 mm thick) on the panel face, and the panel (1U: 44.5 mm) round the 13 x 9.5 cutout
    fw, fh, fx = AD_W + 0.004, 0.022, PANEL_X - 0.0009  # flange round the adapter body (12.8 x 9.4 hole)
    for n, sg in (("l", 1), ("r", -1)):
        out.append((f"ad_flange_{n}", (fx, xy + sg * (AD_W / 2 + fw / 2) / 2, z), (0.0018, fw / 2 - AD_W / 2, fh), q0,
                    "beige"))
    for n, sg in (("t", 1), ("b", -1)):
        out.append((f"ad_flange_{n}", (fx, xy, z + sg * (AD_H / 2 + fh / 2) / 2), (0.0018, AD_W, fh / 2 - AD_H / 2), q0,
                    "beige"))
    pw, ph = 0.120, 0.0445
    cw, chh = 0.013, 0.0095
    px = PANEL_X + PANEL_T / 2
    out.append(("panel_l", (px, xy + (cw / 2 + pw / 2) / 2, z), (PANEL_T, pw / 2 - cw / 2, ph), q0, "steel"))
    out.append(("panel_r", (px, xy - (cw / 2 + pw / 2) / 2, z), (PANEL_T, pw / 2 - cw / 2, ph), q0, "steel"))
    out.append(("panel_t", (px, xy, z + (chh / 2 + ph / 2) / 2), (PANEL_T, cw, ph / 2 - chh / 2), q0, "steel"))
    out.append(("panel_b", (px, xy, z - (chh / 2 + ph / 2) / 2), (PANEL_T, cw, ph / 2 - chh / 2), q0, "steel"))
    # posts (rack-rail stubs) from the table to the panel top, behind the panel's ends
    for n, sg in (("l", 1), ("r", -1)):
        out.append((f"post_{n}", (px + PANEL_T / 2 + 0.008, xy + sg * (pw / 2 - 0.008), (TOP + z + ph / 2) / 2),
                    (0.016, 0.016, z + ph / 2 - TOP), q0, "dark"))
    return out


def adapter_visuals() -> list[tuple[str, tuple, tuple, str]]:
    """Visual-only boxes: the adapter's rear half behind the panel and the far connector + cord."""
    xy, z = PORT_Y, PORT_Z
    out = [("ad_rear", (AF + (STOP_D + 0.001 + AD_L) / 2, xy, z), (AD_L - STOP_D - 0.001, AD_W, AD_H), "beige")]
    far0 = AF + AD_L  # the far connector's housing sticks out of the adapter's back by 12 mm, then boot and cord
    out += [("far_housing", (far0 + 0.006, xy, z), (0.012, HW, HH), "blue"),
            ("far_boot", (far0 + 0.012 + 0.012, xy, z), (0.024, 0.005, 0.005), "black"),
            ("far_cord", (far0 + 0.036 + 0.03, xy, z), (0.06, 0.003, 0.003), "yellow")]
    return out


OBJECTS = [CASSETTE, *PINS, CORD]

# -- spawn hook: connector parts, adapter + panel colliders, detent pins' springs, cleaner details ---------------------
_LOOKS = {"blue": SC_BLUE, "ceramic": CERAMIC, "black": BLACK, "beige": BEIGE, "steel": STEEL, "dark": DARK,
          "yellow": YELLOW, "tape": TAPE, "grey": CASSETTE_GREY}


def _install_spawn_hook() -> None:
    orig = cablekit._spawn_cable
    if getattr(orig, "_sc_hook", False):
        return

    def spawn(prim_path, cfg, *a, **k):
        prim = orig(prim_path, cfg, *a, **k)
        if getattr(cfg, "cable_name", "") != CORD.name:
            return prim
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import bind_physics_material, bind_visual_material
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

        from yam_arena.station import linear_rgb

        stage = get_current_stage()
        parent = str(prim_path).rsplit("/", 1)[0]
        w = UsdGeom.Xformable(stage.GetPrimAtPath(parent)).ComputeLocalToWorldTransform(0).ExtractTranslation()
        W = np.array([w[0], w[1], w[2]], dtype=float)
        looks = f"{prim_path}/Looks/sc"
        for n, col in _LOOKS.items():
            lk = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(col), roughness=0.35 if n != "tape" else 0.9)
            lk.func(f"{looks}/{n}", lk)
        slick = f"{prim_path}/Looks/sc_slick"
        sm = sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2)
        sm.func(slick, sm)

        def cube(path, c, size, look, collide, rot=(1.0, 0.0, 0.0, 0.0), offset=0.001):
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            g.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, c)))
            if rot != (1.0, 0.0, 0.0, 0.0):
                g.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, rot)))
            g.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            if collide:
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                cp = PhysxSchema.PhysxCollisionAPI.Apply(g.GetPrim())
                cp.CreateContactOffsetAttr(offset)
                cp.CreateRestOffsetAttr(0.0)
                bind_physics_material(path, slick, stage=stage)
            bind_visual_material(path, f"{looks}/{look}", stage=stage)
            return g

        # connector parts (children of the end-plug body)
        plug = f"{prim_path}/end_plug"
        for name, c, size, look, col in _plug_parts():
            if name == "ferrule":  # collider square, drawn round
                g = cube(f"{plug}/{name}", c, size, look, True, offset=0.0005)
                g.CreatePurposeAttr("guide")
                cy = UsdGeom.Cylinder.Define(stage, f"{plug}/{name}_vis")
                cy.CreateAxisAttr("X")
                cy.CreateRadiusAttr(FER_D / 2)
                cy.CreateHeightAttr(FER_L)
                cy.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
                bind_visual_material(str(cy.GetPath()), f"{looks}/ceramic", stage=stage)
            else:
                cube(f"{plug}/{name}", c, size, look, col, offset=0.0005)
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(plug)).CreateThresholdAttr(0.0)

        # adapter + panel: static colliders (env-local positions -> world by the env origin)
        root = f"{parent}/sc_panel"
        UsdGeom.Xform.Define(stage, root)
        for name, c, size, rot, look in adapter_boxes():
            cube(f"{root}/{name}", W + np.asarray(c), size, look, True, rot=rot)
        for name, c, size, look in adapter_visuals():
            cube(f"{root}/{name}", W + np.asarray(c), size, look, False)

        # detent pins: prismatic joints (world -> pin) along y with a preloaded spring drive pushing inwards
        for pin, sg in zip(PINS, (1, -1)):
            pp = f"{parent}/{pin.name}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/{pin.name}_slide")
            j.CreateBody1Rel().SetTargets([Sdf.Path(pp)])
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, W + np.asarray(pin.pos))))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0 if sg > 0 else -PIN_TRAVEL)
            j.CreateUpperLimitAttr(PIN_TRAVEL if sg > 0 else 0.0)
            j.CreateExcludeFromArticulationAttr(True)
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(PIN_K)
            d.CreateDampingAttr(0.5)
            d.CreateTargetPositionAttr(-sg * PIN_PRE)
            for p in stage.Traverse():
                ps = p.GetPath().pathString
                if ps.startswith(pp + "/") and p.HasAPI(UsdPhysics.CollisionAPI):
                    bind_physics_material(ps, slick, stage=stage)
            fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pp))  # pins sit in their wall slots
            for name, *_ in adapter_boxes():
                if name.startswith("ad_side") or name.startswith("ad_slotback") or name in ("ad_top", "ad_bot"):
                    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(f"{root}/{name}"))

        # cleaner: tape window, arrow, shutter frame and lever (visual only, children of its body)
        cas = f"{parent}/{CASSETTE.name}"
        if stage.GetPrimAtPath(cas).IsValid():
            fx = -CAS[0] / 2 - 0.0002
            wy, wz = (WIN_Y[0] + WIN_Y[1]) / 2, (WIN_Z[0] + WIN_Z[1]) / 2
            ww, wh = WIN_Y[1] - WIN_Y[0], WIN_Z[1] - WIN_Z[0]
            cube(f"{cas}/tape", (fx, wy, wz), (0.0004, ww, wh), "tape", False)
            for n, dy in (("l", ww / 2 + 0.002), ("r", -ww / 2 - 0.002)):  # shutter frame either side
                cube(f"{cas}/frame_{n}", (fx - 0.0002, wy + dy, wz), (0.0004, 0.004, wh + 0.004), "black", False)
            ax_y = WIN_Y[1] + 0.010  # arrow pointing down, beside the window
            cube(f"{cas}/arrow_0", (fx - 0.0002, ax_y, wz + 0.002), (0.0004, 0.0012, 0.020), "black", False)
            for i, dy in enumerate((0.002, -0.002, 0.004, -0.004)):
                cube(f"{cas}/arrow_{i + 1}", (fx - 0.0002, ax_y + dy, wz - 0.008 + abs(dy)), (0.0004, 0.0012, 0.0012),
                     "black", False)
            cube(f"{cas}/lever", (fx + 0.004, CAS[1] / 2 - 0.03, CAS[2] / 2 + 0.003), (0.020, 0.030, 0.006), "black",
                 False)
            cube(f"{cas}/label", (fx, 0.02, -0.01), (0.0004, 0.05, 0.03), "blue", False)
        else:
            print(f"[sc] cleaner prim {cas} not found when the cord spawned; no tape visuals", flush=True)
        return prim

    spawn._sc_hook = True
    cablekit._spawn_cable = spawn


_install_spawn_hook()

# -- per-physics-step tracking --------------------------------------------------------------------------------------
TILT_MAX = math.radians(10.0)  # connector axis within 10 deg of the tape normal: end face flat on the tape
STROKE_MIN = 0.021  # m: end-face centre travels this far along the arrow (window 30 mm, face 2.5 mm, 3 mm margins)
BACKTRACK = 0.0015  # m: moving back against the arrow by more than this breaks the stroke (no scrubbing)
CONTACT_GAP_S = 0.05  # s: tape-contact gaps shorter than this don't split a stroke
WIN_MARGIN = 0.0003  # end-face rim may be this close to the window edge
ENTER_D = 0.0005  # ferrule tip this far past the adapter face = it has entered the adapter
FER_ZONE_S = -0.00185  # contact point on the ferrule protrusion (s > this, r < FER_ZONE_R); the housing face is at
# s = -2 mm and is not part of the ferrule
FER_ZONE_R = 0.0023  # the 2.5 mm square collider's corners are 1.77 mm out
TABLE_TOUCH = 0.0005
ALLOWED = ("sleeve_", "far_ferrule", "ad_stop_")  # the sleeve, its mouth in the stop wall, the far ferrule
TOUCH_SEP = 0.0001  # m: a contact point this close (or penetrating) is a touch
_RIM = [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)]

_TR: dict = {"env": None, "art": None, "view": None, "err": None, "polls": 0, "armed": False}


def _reset_tracker() -> None:
    _TR.update(armed=True, clean=False, dirty_insert=False, touched_after=False, touch_what=None, stroke=None,
               best=0.0, gap=0, wipes=0, entered=False, tape_contacts=0, pins_pushed=False, contacts_seen=0,
               broken=False, max_depth=-1.0, pin_max=[0.0, 0.0])


def _quat_rot_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray) and hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    try:
        import warp as wp

        if isinstance(x, wp.array):
            return wp.to_torch(x).detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(x)


def _origin() -> np.ndarray:
    base = _TR["env"].unwrapped
    o = base.scene.env_origins
    o = getattr(o, "torch", o)
    return _np(o[0]).astype(np.float64)


def _plug_frame():
    """Live connector pose from PhysX: (ferrule tip, axis a (out through the end face), width dir, height dir)."""
    env = _TR["env"]
    if env is None:
        return None
    if _TR["art"] is None:
        base = env.unwrapped
        art = base.scene[CORD.name]
        names = list(art.body_names)
        _TR.update(art=art, plug_i=names.index("end_plug"), origin=_origin())
    arr = _np(_TR["art"].root_physx_view.get_link_transforms())
    arr = arr.reshape(-1, arr.shape[-1])
    c = arr[_TR["plug_i"], :3].astype(np.float64) - _TR["origin"]
    rot = _quat_rot_xyzw(arr[_TR["plug_i"], 3:7])
    a = rot[:, 0]
    return c + a * PLUG_TIP, a, rot[:, 1], rot[:, 2]


def _cassette_frame():
    """Live cleaner pose: (centre, rotation) from PhysX."""
    base = _TR["env"].unwrapped
    obj = base.scene[CASSETTE.name]
    t = _np(obj.root_physx_view.get_transforms()).reshape(-1)[:7].astype(np.float64)
    return t[:3] - _TR["origin"], _quat_rot_xyzw(t[3:7])


def _enable_contact_processing() -> None:
    try:
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # pragma: no cover
        pass


def _make_view(mgr) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    sv = mgr.get_physics_sim_view()
    if sv is None:
        return
    stage = get_current_stage()
    plug, filters = None, []
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p:
            continue
        if p.endswith(f"/{CORD.name}/end_plug"):
            plug = p
        elif f"/{CORD.name}/" in p:
            continue
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            # a moving body's contacts are reported against its rigid-body prim, not its collider prims
            q = prim
            while q and q.IsValid() and not q.HasAPI(UsdPhysics.RigidBodyAPI):
                q = q.GetParent()
            body = q.GetPath().pathString if q and q.IsValid() and q.HasAPI(UsdPhysics.RigidBodyAPI) else p
            if body not in filters:
                filters.append(body)
    if plug is None or not filters:
        raise RuntimeError(f"sc contact monitor: plug {plug}, {len(filters)} colliders")
    view = sv.create_rigid_contact_view([plug], filter_patterns=[filters], max_contact_data_count=4096)
    _TR.update(view=view, filters=filters, filter_names=[f.rsplit("/", 1)[-1] for f in filters])
    _TR["cas_idx"] = {i for i, f in enumerate(filters) if f"/{CASSETTE.name}/" in f + "/"}
    print(f"[sc] contact monitor: connector x {len(filters)} colliders "
          f"({len(_TR['cas_idx'])} on the cleaner)", flush=True)


def _ferrule_contacts(tip, a):
    """[(filter name, filter index, point)] for contacts at the ferrule (separation <= 0.1 mm) this substep."""
    view = _TR["view"]
    out = []
    _, pts, _, sep, counts, starts = view.get_contact_data(dt=_TR["dt"])
    counts = _np(counts).reshape(-1)
    if not counts.any():
        return out
    starts = _np(starts).reshape(-1)
    pts = _np(pts).reshape(-1, 3)
    sep = _np(sep).reshape(-1)
    o = _TR["origin"]
    for j in np.nonzero(counts)[0]:
        s0 = int(starts[j])
        for k in range(s0, s0 + int(counts[j])):
            if sep[k] > TOUCH_SEP:
                continue
            p = pts[k].astype(np.float64) - o
            v = p - tip
            sa = float(v @ a)
            r = float(np.linalg.norm(v - sa * a))
            if sa > FER_ZONE_S and r < FER_ZONE_R:
                out.append((_TR["filter_names"][j], int(j), p))
    return out


def _in_window(cc, cr, p, margin=WIN_MARGIN) -> bool:
    q = cr.T @ (p - cc)
    return (WIN_Y[0] + margin <= q[1] <= WIN_Y[1] - margin and WIN_Z[0] + margin <= q[2] <= WIN_Z[1] - margin
            and q[0] < -CAS[0] / 2 + 0.0008)


def _adapter_depth(tip, a) -> tuple[float, bool]:
    """(depth of the ferrule tip past the adapter face, lined up with the channel)."""
    d = float(tip[0] - AF)
    ok = abs(tip[1] - PORT_Y) < CH_W / 2 and abs(tip[2] - PORT_Z) < CH_H / 2 and a[0] > 0.95
    return d, ok


def _track(dt: float) -> None:
    if not _TR.get("armed") or _TR["env"] is None:
        return
    try:
        fr = _plug_frame()
        if fr is None:
            return
        tip, a, wv, hv = fr
        if not np.isfinite(tip).all():
            return
        _TR["dt"] = dt
        if _TR["view"] is None:
            _make_view(_TR["mgr"])
        contacts = _ferrule_contacts(tip, a) if _TR["view"] is not None else []
        cc, cr = _cassette_frame()
    except Exception as e:  # pragma: no cover
        if _TR["err"] is None:
            _TR["err"] = repr(e)
            print(f"[sc] tracker error: {e!r}", flush=True)
        return
    _TR["polls"] += 1
    # 1) what the ferrule touches this substep
    on_tape, dirty, dirty_p = False, None, None
    for name, j, p in contacts:
        _TR["contacts_seen"] += 1
        if any(name.startswith(x) for x in ALLOWED):
            continue
        if j in _TR["cas_idx"] and _in_window(cc, cr, p, margin=-0.0005):
            on_tape = True
            continue
        dirty = name
        dirty_p = cr.T @ (p - cc) if j in _TR["cas_idx"] else p
    face = [tip + 0.5 * FER_D * (u * wv + v * hv) for u, v in _RIM]
    if min(p[2] for p in face) < TOP + TABLE_TOUCH:
        dirty = "table"
    if dirty is not None:
        if _TR["clean"] and not _TR["touched_after"]:
            _TR["touched_after"], _TR["touch_what"] = True, dirty
            where = np.round(np.asarray(dirty_p) * 1000, 1).tolist() if dirty != "table" else None
            print(f"[sc] cleaned end face touched {dirty} at {where} mm (t {_TR['polls']} polls): score 0", flush=True)
        elif not _TR["clean"]:
            _TR["touch_what"] = dirty
    # 2) wiping: end face flat on the tape, in contact, moving one way along the arrow
    n_in = cr[:, 0]  # into the cleaner's front face
    flat = float(a @ n_in) >= math.cos(TILT_MAX)
    inside = all(_in_window(cc, cr, p) for p in face) and _in_window(cc, cr, tip)
    if on_tape:
        _TR["tape_contacts"] += 1
    if on_tape and flat and inside:
        u = float(ARROW @ (cr.T @ (tip - cc)))  # progress along the arrow (down the window)
        st = _TR["stroke"]
        if st is None:
            _TR["stroke"] = st = {"u0": u, "umax": u}
            _TR["broken"] = False
        st["umax"] = max(st["umax"], u)
        if u < st["umax"] - BACKTRACK and not _TR["broken"]:
            _TR["broken"] = True
            print("[sc] stroke reversed (scrubbing): it does not count", flush=True)
        _TR["gap"] = 0
        length = st["umax"] - st["u0"]
        _TR["best"] = max(_TR["best"], length)
        if length >= STROKE_MIN and not _TR["broken"] and not _TR["clean"] and dirty is None:
            _TR["clean"] = True
            _TR["wipes"] += 1
            print(f"[sc] valid one-way stroke on the tape ({length * 1000:.1f} mm): end face clean", flush=True)
    else:
        _TR["gap"] += 1
        if _TR["gap"] * dt > CONTACT_GAP_S:
            _TR["stroke"] = None
    # pins (diagnostic: how far each detent pin has been pushed out)
    try:
        base = _TR["env"].unwrapped
        for i, pin in enumerate(PINS):
            t = _np(base.scene[pin.name].root_physx_view.get_transforms()).reshape(-1)
            _TR["pin_max"][i] = max(_TR["pin_max"][i], abs(float(t[1] - _TR["origin"][1]) - pin.pos[1]))
    except Exception:  # noqa: BLE001
        pass
    # 3) entering the adapter
    d, lined = _adapter_depth(tip, a)
    _TR["max_depth"] = max(_TR["max_depth"], d if lined else -1.0)
    if lined and d > ENTER_D:
        if not _TR["entered"]:
            _TR["entered"] = True
            if not _TR["clean"]:
                _TR["dirty_insert"] = True
                print("[sc] connector entered the adapter before a valid clean: score 0", flush=True)


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_sc_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _TR["mgr"] = cls
        try:
            dt = float(cls.get_physics_dt())
        except Exception:  # noqa: BLE001
            dt = 1.0 / 240.0
        _track(dt)

    step._sc_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, so not at task import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


def _install_hooks() -> None:
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MgrFinder())
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover  (offline)
        return
    cls = ie.YamArenaEmbodiment
    if not getattr(cls.reset, "_sc_hooked", False):
        base_reset = cls.reset

        def reset(self, *a, **k):
            _TR["armed"] = False
            out = base_reset(self, *a, **k)
            _TR.update(env=self._env, art=None)
            _reset_tracker()
            return out

        reset._sc_hooked = True
        cls.reset = reset
    if not getattr(cls._task_step, "_sc_hooked", False):
        base = cls._task_step

        def _task_step(self, *a, **k):
            if _TR["env"] is not self._env:
                _TR.update(env=self._env, art=None)
            return base(self, *a, **k)

        _task_step._sc_hooked = True
        cls._task_step = _task_step


_install_hooks()

# -- scoring -------------------------------------------------------------------------------------------------------
LATCH_D = (STOP_D - 0.0012, STOP_D + 0.0003)  # housing front depth window with the notch over the pins
PIN_SEATED = 0.00015  # m: pin within this of its rest stop = snapped back into the notch
RELEASE_DIST = 0.035  # m: no gripper TCP this close to the connector's grip at the end


def _conn(objects) -> tuple[np.ndarray, np.ndarray]:
    """(ferrule tip, axis) from the cable state (end-plug centre and the last cord point)."""
    st = objects[CORD.name]
    p = np.asarray(st.end_plug, dtype=float)
    a = p - np.asarray(st.points[-1], dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return p + a * PLUG_TIP, a


def pin_offsets(objects) -> list[float]:
    """How far (m) each detent pin is pushed out of its rest stop."""
    return [abs(float(objects[p.name].pos[1]) - p.pos[1]) for p in PINS]


def latched(objects) -> bool:
    """Both detent pins sit back at rest inside the connector's side notches: housing front within 1.2 mm of the
    stop and lined up with the channel."""
    tip, a = _conn(objects)
    if not np.isfinite(tip).all():
        return False
    front = float(tip[0] - AF) - FER_L
    lined = abs(tip[1] - PORT_Y) < 0.0006 and abs(tip[2] - PORT_Z) < 0.0006 and a[0] > math.cos(math.radians(4))
    return bool(lined and LATCH_D[0] <= front <= LATCH_D[1] and max(pin_offsets(objects)) < PIN_SEATED)


def entered(objects) -> bool:
    tip, a = _conn(objects)
    d, ok = _adapter_depth(tip, a)
    return bool(ok and d > ENTER_D)


def grippers() -> list[np.ndarray]:
    """World positions of both grippers' TCP; [] if unknown (offline)."""
    env = _TR["env"]
    if env is None:
        return []
    base = env.unwrapped
    origin = _origin()
    out = []
    for key in ("left_robot", "right_robot"):
        art = base.scene[key]
        names = list(art.body_names)
        pos = _np(getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0]).astype(np.float64) - origin
        g = pos[names.index("gripper")]
        mid = (pos[names.index("tip_left")] + pos[names.index("tip_right")]) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        out.append(g + ax * 0.134)
    return out


def released(objects) -> bool:
    tip, a = _conn(objects)
    grip = tip - a * 0.027
    return all(float(np.linalg.norm(grip - g)) >= RELEASE_DIST for g in grippers())


def clean() -> bool:
    return bool(_TR.get("clean"))


def score(objects) -> float:
    """0 if the connector entered the adapter before a valid clean, or the cleaned ferrule touched anything but the
    tape, the adapter's sleeve or the far ferrule (latched for the trial), or no valid clean. Otherwise: 1.0 latched
    and let go; 0.75 latched, still held; 0.5 in the adapter, not latched; 0.25 cleaned, not in the adapter."""
    if _TR.get("dirty_insert") or _TR.get("touched_after") or not clean():
        return 0.0
    if latched(objects):
        return 1.0 if released(objects) else 0.75
    return 0.5 if entered(objects) else 0.25


def success(objects) -> bool:
    return bool(score(objects) == 1.0 and np.linalg.norm(objects[CORD.name].lin_vel) < 0.02)


def distance(objects) -> float:
    """1 m until clean (or after a latched 0); then how far (m) the housing front still is from the stop."""
    if _TR.get("dirty_insert") or _TR.get("touched_after") or not clean():
        return 1.0
    tip, a = _conn(objects)
    return float(max(0.0, STOP_D + FER_L - (tip[0] - AF)) + np.linalg.norm(tip[1:] - (PORT_Y, PORT_Z)))


def progress(objects) -> dict:
    tip, a = _conn(objects)
    return {"clean": clean(), "best_stroke_mm": round(_TR.get("best", 0.0) * 1000, 1), "wipes": _TR.get("wipes"),
            "tape_contacts": _TR.get("tape_contacts"), "dirty_insert": _TR.get("dirty_insert"),
            "touched_after": _TR.get("touched_after"), "touch_what": _TR.get("touch_what"),
            "entered": entered(objects), "latched": latched(objects), "released": released(objects),
            "tip_depth_mm": round(float(tip[0] - AF) * 1000, 2),
            "tip_off_mm": [round(float(tip[1] - PORT_Y) * 1000, 2), round(float(tip[2] - PORT_Z) * 1000, 2)],
            "pins_mm": [round(v * 1000, 3) for v in pin_offsets(objects)],
            "pins_max_mm": [round(v * 1000, 3) for v in _TR.get("pin_max", [0, 0])], "polls": _TR.get("polls"),
            "err": _TR.get("err")}


# The scripted demo lives in demo.py (cabledemo.py loads it); it reads the simulator's poses (see README).
