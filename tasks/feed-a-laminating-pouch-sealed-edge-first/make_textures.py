"""Draws the task's textures as image files in textures/ (our own artwork, no text or logos). Run once with any Python
that has Pillow and NumPy: python make_textures.py. The task loads the image files; it does not need Pillow itself.

- card.png: the picture flash card, 63 x 88 mm (playing-card size): a red apple on white inside a green border. usdkit's
  box projection maps u along the pouch's x (its width) and v along its y (its length, +y toward the sealed edge), with
  v = 1 at the image's top row; the picture is drawn upside down so that its top faces the pouch's open end.
- box_top.png: the lid of the box of card pouches, 110 x 80 mm: dark blue card with a printed picture of a clear pouch
  holding a card (the box's white band runs round its sides).
- plastic.png: a faint mottled grain for the laminator's moulded plastic housing (tiled every 5 cm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"


def _apple(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    red, dark, leaf, stem = (204, 32, 36), (150, 18, 24), (64, 150, 52), (96, 62, 30)
    # two overlapping lobes and a rounded bottom make the apple's outline
    for dx in (-0.42 * r, 0.42 * r):
        d.ellipse([cx + dx - 0.68 * r, cy - 0.78 * r, cx + dx + 0.68 * r, cy + 0.72 * r], fill=red)
    d.ellipse([cx - 0.85 * r, cy - 0.45 * r, cx + 0.85 * r, cy + 0.95 * r], fill=red)
    d.ellipse([cx - 0.14 * r, cy - 0.80 * r, cx + 0.14 * r, cy - 0.58 * r], fill=dark)  # the dip at the stem
    d.ellipse([cx + 0.30 * r, cy - 0.45 * r, cx + 0.62 * r, cy - 0.05 * r], fill=(236, 120, 120))  # highlight
    d.line([cx, cy - 0.66 * r, cx + 0.10 * r, cy - 1.10 * r], fill=stem, width=max(3, int(0.12 * r)))
    d.ellipse([cx + 0.12 * r, cy - 1.18 * r, cx + 0.72 * r, cy - 0.86 * r], fill=leaf)


def card() -> None:
    w, h = 504, 704  # 8 px per mm
    im = Image.new("RGB", (w, h), (250, 250, 246))
    d = ImageDraw.Draw(im)
    b = 32  # 4 mm green border
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=40, fill=(46, 140, 72))
    d.rounded_rectangle([b, b, w - 1 - b, h - 1 - b], radius=24, fill=(250, 250, 246))
    _apple(d, w / 2, h * 0.47, 150)
    d.rounded_rectangle([w * 0.2, h * 0.80, w * 0.8, h * 0.86], radius=12, fill=(46, 140, 72))  # a green stripe below
    im = im.rotate(180)
    im.save(OUT / "card.png")


def box_top() -> None:
    w, h = 704, 512  # 110 x 80 mm, 6.4 px per mm
    im = Image.new("RGB", (w, h), (30, 64, 124))  # dark blue card, as many pouch boxes are
    d = ImageDraw.Draw(im)
    # the printed pouch: a clear rectangle with a sheen, holding a pale card, drawn at an angle
    pouch = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pouch)
    pd.rounded_rectangle([0, 0, 299, 399], radius=18, fill=(214, 228, 238, 255), outline=(150, 170, 186, 255), width=4)
    pd.rounded_rectangle([26, 70, 273, 372], radius=12, fill=(255, 255, 255, 255), outline=(190, 200, 210, 255), width=3)
    pd.rectangle([0, 0, 299, 22], fill=(196, 210, 222, 255))  # its sealed edge
    pd.polygon([(40, 380), (120, 380), (260, 90), (180, 90)], fill=(255, 255, 255, 90))  # sheen
    pouch = pouch.rotate(-12, expand=True, resample=Image.BICUBIC).resize((190, 230))
    im.paste(pouch, (int(w * 0.36), int(h * 0.12)), pouch)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    im.save(OUT / "box_top.png")


def plastic() -> None:
    rng = np.random.default_rng(3)
    n = 256
    noise = rng.normal(0.0, 1.0, (n, n))
    img = Image.fromarray(np.uint8(np.clip(128 + 18 * noise, 0, 255)), "L").filter(ImageFilter.GaussianBlur(2.2))
    a = np.asarray(img, float)
    a = 236 + (a - a.mean()) * 0.6  # a light grey that the material's colour tints
    Image.fromarray(np.uint8(np.clip(a, 0, 255)), "L").convert("RGB").save(OUT / "plastic.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    card()
    box_top()
    plastic()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
