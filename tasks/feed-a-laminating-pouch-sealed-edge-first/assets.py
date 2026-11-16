"""Geometry of the laminator, the box of pouches and the loaded pouch, and the USD files built from them (into assets/,
at load time). Pure NumPy. Sizes in metres; the station frame has x forward, y to the robot's left, z up from the table.

Laminator (one fixed body whose origin is the table top under the arms' midpoint): an A4 desktop pouch laminator lying
along x in the middle of the table, its entry face toward the right arm (-y), so pouches pass through it toward the
robot's left (+y). A tunnel 8 mm high runs through it from the entry slot to the exit slot in its back face; the rollers'
nip line crosses the tunnel 2.5 cm in.

Pouch (one rigid body; origin at its centre, +y toward its sealed edge): a jumbo card pouch, 75 x 105 mm, of two clear
films welded along one short edge, holding a playing-card-size flash card pushed against the seal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- laminator ----------------------------------------------------------------------------------------------------
LAM_X = (0.18, 0.52)  # its length, along x
FRONT_Y, BACK_Y = -0.07, 0.04  # entry face (facing -y) and back face
IN = 1.0  # the feed direction along y, from the entry face into the laminator
LAM_H = 0.07
SLOT_X = (0.2325, 0.4675)  # 23.5 cm wide entry
SLOT_Z = (0.035, 0.043)  # the tunnel's floor and roof, 8 mm apart
FLARE_D, FLARE = 0.006, 0.002  # the lips flare over 6 mm to a 12 mm mouth at the face
NIP_Y = FRONT_Y + IN * 0.025  # the rollers' nip line
ROLLER_R = 0.015
SLOT_CX = 0.5 * (SLOT_X[0] + SLOT_X[1])

# -- box of pouches -----------------------------------------------------------------------------------------------
BOX = (0.11, 0.08, 0.032)
BOX_XY = (0.35, -0.255)
BOX_MASS = 0.30

# -- pouch --------------------------------------------------------------------------------------------------------
POUCH = (0.075, 0.105, 0.0025)  # x (width), y (length, +y toward the seal), z: the rigid stand-in
POUCH_MASS = 0.008
SEAL_W = 0.004
CARD = (0.063, 0.088, 0.0008)
FILM_T = 0.0003  # each film's visual thickness
CARD_Y = POUCH[1] / 2 - SEAL_W - CARD[1] / 2  # the card's centre: against the seal
OPEN_MARGIN = POUCH[1] - SEAL_W - CARD[1]  # 13 mm of clear film at the open end


def _lam() -> str:
    looks = {
        "housing": uk.Look(texture="plastic.png", color=(0.50, 0.51, 0.52), roughness=0.45, tile=0.05),
        "top_panel": uk.Look(texture="plastic.png", color=(0.075, 0.08, 0.09), roughness=0.4, tile=0.05),
        "panel": uk.Look(color=(0.055, 0.058, 0.064), roughness=0.5),
        "bezel": uk.Look(color=(0.11, 0.115, 0.12), roughness=0.55),
        "inside": uk.Look(color=(0.012, 0.012, 0.012), roughness=0.9),
        "roller": uk.Look(color=(0.02, 0.02, 0.02), roughness=0.75),
        "lamp": uk.Look(color=(0.05, 0.6, 0.12), roughness=0.3, emissive=(0.1, 2.2, 0.35)),
        "mark": uk.Look(color=(0.85, 0.86, 0.87), roughness=0.5),
        "foot": uk.Look(color=(0.03, 0.03, 0.03), roughness=0.9),
    }
    x0, x1 = LAM_X
    sx0, sx1 = SLOT_X
    zf, zr = SLOT_Z
    L = x1 - x0
    D = abs(BACK_Y - FRONT_Y)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (FRONT_Y + BACK_Y)
    # housing in flush pieces round the tunnel, so the slot is a real opening with the rollers inside
    low = uk.box((L, D, zf - 0.0005), (cx, cy, (zf - 0.0005) / 2), round_xy=0.012)
    top = uk.box((L, D, LAM_H - 0.006 - zr - 0.0005), (cx, cy, (zr + 0.0005 + LAM_H - 0.006) / 2), round_xy=0.012)
    cap = uk.box((L, D, 0.012), (cx, cy, LAM_H - 0.006), bevel=0.006, round_xy=0.012)
    ends = [uk.box((sx0 - x0, D, zr - zf + 0.001), (0.5 * (x0 + sx0), cy, 0.5 * (zf + zr)), round_xy=0.004),
            uk.box((x1 - sx1, D, zr - zf + 0.001), (0.5 * (sx1 + x1), cy, 0.5 * (zf + zr)), round_xy=0.004)]
    # dark bezels above and below the entry and exit slots, and the black inside of the tunnel
    bez_w = sx1 - sx0 + 0.016
    bezels = []
    for y, s in ((FRONT_Y, -IN), (BACK_Y, IN)):
        yc = y + s * 0.0008
        bezels += [uk.box((bez_w, 0.0016, 0.006), (SLOT_CX, yc, zf - FLARE - 0.003)),
                   uk.box((bez_w, 0.0016, 0.006), (SLOT_CX, yc, zr + FLARE + 0.003))]
    inside = [uk.box((sx1 - sx0, D - 0.001, 0.0006), (SLOT_CX, cy, zf - 0.0003)),
              uk.box((sx1 - sx0, D - 0.001, 0.0006), (SLOT_CX, cy, zr + 0.0003))]
    # the rollers: the upper one shows through the slot, its underside on the pouch's top as it rests on the floor
    upper = uk.cylinder(ROLLER_R, sx1 - sx0 - 0.002, (SLOT_CX, NIP_Y, zf + POUCH[2] + ROLLER_R), axis="x", segments=32)
    lower = uk.cylinder(ROLLER_R, sx1 - sx0 - 0.002, (SLOT_CX, NIP_Y, zf - ROLLER_R - 0.0004), axis="x", segments=32)
    # a dark control strip along the entry edge carries the ready lamp and the power switch
    controls = uk.box((L - 0.05, 0.03, 0.0012), (cx, FRONT_Y + IN * 0.024, LAM_H + 0.0004), round_xy=0.004)
    lamp = uk.cylinder(0.006, 0.0016, (0.47, FRONT_Y + IN * 0.022, LAM_H + 0.0012), segments=24)
    power = uk.box((0.016, 0.010, 0.002), (0.488, FRONT_Y + IN * 0.022, LAM_H + 0.0014), round_xy=0.002)
    tri = [[SLOT_CX - 0.004, FRONT_Y + IN * 0.017], [SLOT_CX, FRONT_Y + IN * 0.010], [SLOT_CX + 0.004, FRONT_Y + IN * 0.017]]
    mark = uk.prism(np.array(tri if IN > 0 else tri[::-1]), LAM_H, LAM_H + 0.0016)  # points at the entry
    feet = [uk.box((0.03, 0.012, 0.002), (x, y, 0.001)) for x in (x0 + 0.03, x1 - 0.03) for y in (FRONT_Y + IN * 0.012, BACK_Y - IN * 0.012)]
    parts = [("low", low, "housing"), ("top", top, "housing"), ("cap", cap, "housing"), ("controls", controls, "top_panel"), ("end_l", ends[0], "housing"), ("end_r", ends[1], "housing"),
             ("upper_roller", upper, "roller"), ("lower_roller", lower, "roller"), ("lamp", lamp, "lamp"),
             ("power", power, "panel"), ("mark", mark, "mark")]
    parts += [(f"bezel_{k}", m, "bezel") for k, m in enumerate(bezels)]
    parts += [(f"inside_{k}", m, "inside") for k, m in enumerate(inside)]
    parts += [(f"foot_{k}", m, "foot") for k, m in enumerate(feet)]
    yi = FRONT_Y + IN * FLARE_D  # where the flare meets the tunnel
    cols = [
        uk.collider("col_floor", (L, abs(BACK_Y - yi), zf), (cx, 0.5 * (yi + BACK_Y), zf / 2), "Laminator"),
        uk.collider("col_roof", (L, abs(BACK_Y - yi), LAM_H - zr), (cx, 0.5 * (yi + BACK_Y), 0.5 * (zr + LAM_H)), "Laminator"),
        uk.collider("col_end_l", (sx0 - x0, D, zr - zf), (0.5 * (x0 + sx0), cy, 0.5 * (zf + zr)), "Laminator"),
        uk.collider("col_end_r", (x1 - sx1, D, zr - zf), (0.5 * (sx1 + x1), cy, 0.5 * (zf + zr)), "Laminator"),
        uk.hull_collider("col_lip_low", [(x, y, z) for x in (x0, x1) for y, z in ((yi, 0.0), (FRONT_Y, 0.0), (yi, zf), (FRONT_Y, zf - FLARE))],
                         "Laminator"),
        uk.hull_collider("col_lip_high", [(x, y, z) for x in (x0, x1) for y, z in ((yi, LAM_H), (FRONT_Y, LAM_H), (yi, zr), (FRONT_Y, zr + FLARE))],
                         "Laminator"),
    ]
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "laminator.usda"), "Laminator", parts, looks, cols, mass=1.5, friction=(0.4, 0.35))


def _box() -> str:
    looks = {
        "lid": uk.Look(texture="box_top.png", color=(1, 1, 1), roughness=0.7, origin=(-BOX[0] / 2, -BOX[1] / 2, 0.0),
                       tile_uv=(BOX[0], BOX[1])),
        "card": uk.Look(color=(0.013, 0.05, 0.2), roughness=0.75),
        "band": uk.Look(color=(0.84, 0.87, 0.9), roughness=0.7),
    }
    sx, sy, sz = BOX
    body = uk.box((sx, sy, sz - 0.0004), (0, 0, -0.0002), round_xy=0.0015)
    lid = uk.box((sx, sy, 0.0004), (0, 0, sz / 2 - 0.0002), round_xy=0.0015)
    band = uk.box((sx + 0.0006, sy + 0.0006, 0.007), (0, 0, -0.004), round_xy=0.0018)
    parts = [("body", body, "card"), ("lid", lid, "lid"), ("band", band, "band")]
    cols = [uk.collider("col_box", BOX, (0, 0, 0), "PouchBox")]
    return uk.write_usda(str(OUT / "pouch_box.usda"), "PouchBox", parts, looks, cols, mass=BOX_MASS, friction=(0.5, 0.45))


def _pouch() -> str:
    looks = {
        "film": uk.Look(color=(0.55, 0.60, 0.62), roughness=0.04, opacity=0.10),
        "film_edge": uk.Look(color=(0.85, 0.90, 0.93), roughness=0.1, opacity=0.6),
        "seal": uk.Look(color=(0.95, 0.96, 0.97), roughness=0.65, opacity=0.93),
        "card": uk.Look(texture="card.png", color=(1, 1, 1), roughness=0.55, origin=(-CARD[0] / 2, CARD_Y - CARD[1] / 2, 0.0),
                        tile_uv=(CARD[0], CARD[1])),
        "card_edge": uk.Look(color=(0.92, 0.92, 0.9), roughness=0.6),
    }
    w, l, t = POUCH
    half = t / 2
    films = [uk.box((w, l - SEAL_W, FILM_T), (0, -SEAL_W / 2, z)) for z in (-half + FILM_T / 2, half - FILM_T / 2)]
    # the open end: two loose film edges with air between them, a thin bright line on each
    lips = [uk.box((w, 0.0008, FILM_T * 1.4), (0, -l / 2 + 0.0004, z)) for z in (-half + FILM_T * 0.7, half - FILM_T * 0.7)]
    # the sealed end: the two films welded into one frosted seam, a little thicker, its outer edge rounded
    seam = uk.box((w, SEAL_W, t), (0, l / 2 - SEAL_W / 2, 0), bevel=0.0009)
    card_top = uk.box((CARD[0], CARD[1], 0.0002), (0, CARD_Y, CARD[2] / 2 - 0.0001))
    card_body = uk.box((CARD[0], CARD[1], CARD[2] - 0.0002), (0, CARD_Y, -0.0001))
    parts = [("film_low", films[0], "film"), ("film_high", films[1], "film"), ("lip_low", lips[0], "film_edge"),
             ("lip_high", lips[1], "film_edge"), ("seam", seam, "seal"), ("card_face", card_top, "card"),
             ("card_body", card_body, "card_edge")]
    cols = [uk.collider("col_pouch", POUCH, (0, 0, 0), "Pouch")]
    return uk.write_usda(str(OUT / "pouch.usda"), "Pouch", parts, looks, cols, mass=POUCH_MASS, friction=(0.45, 0.4))


def build() -> None:
    _lam()
    _box()
    _pouch()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")), "open margin", round(OPEN_MARGIN, 4), "card y", round(CARD_Y, 4))
