"""Icons painted at runtime: signed-distance shapes on a 16-unit grid, a PNG encoder and a cache. FINAL_SPEC §4.8."""
from __future__ import annotations

import base64
import math
import struct
import tkinter as tk
import zlib
from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np

from yfs.ui.theme import LOGO_COLORS, TOKENS as T

DIM_ALPHA = 0.45
_OME_PAGE, _OME_FOLD = "#5E8FC4", "#9DBDE0"           # the stitched-image page (icon-only shades)


def encode_png(rgba: "np.ndarray") -> bytes:
    """8-bit RGBA PNG from a (h, w, 4) uint8 array."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _rgb(color: str) -> "np.ndarray":
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


class Canvas16:
    """16x16 design units -> `size` px. Signed-distance primitives; coverage = clip(0.5 - d*s, 0, 1)."""

    def __init__(self, size: int, w: float = 16, h: float = 16) -> None:
        self.s = size / 16.0
        self.W, self.H = max(1, round(w * self.s)), max(1, round(h * self.s))
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.x, self.y = (xx + 0.5) / self.s, (yy + 0.5) / self.s
        self.c = np.zeros((self.H, self.W, 3), np.float32)
        self.a = np.zeros((self.H, self.W), np.float32)

    def rrect(self, x0: float, y0: float, x1: float, y1: float, r: float) -> "np.ndarray":
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        r = min(r, hx, hy)
        qx, qy = np.abs(self.x - cx) - (hx - r), np.abs(self.y - cy) - (hy - r)
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r

    def circle(self, cx: float, cy: float, r: float) -> "np.ndarray":
        return np.hypot(self.x - cx, self.y - cy) - r

    def seg(self, ax: float, ay: float, bx: float, by: float, w: float) -> "np.ndarray":
        pax, pay, bax, bay = self.x - ax, self.y - ay, bx - ax, by - ay
        t = np.clip((pax * bax + pay * bay) / max(bax * bax + bay * bay, 1e-9), 0, 1)
        return np.hypot(pax - bax * t, pay - bay * t) - w / 2

    def line(self, pts: Sequence[Tuple[float, float]], w: float) -> "np.ndarray":
        return np.minimum.reduce([self.seg(*pts[i], *pts[i + 1], w) for i in range(len(pts) - 1)])

    def ring(self, d: "np.ndarray", w: float) -> "np.ndarray":
        return np.abs(d) - w / 2

    def poly(self, pts: Sequence[Tuple[float, float]], inset: float = 0.0) -> "np.ndarray":
        """Convex polygon (clockwise on screen) as the max of its half-planes."""
        def hp(p: Tuple[float, float], q: Tuple[float, float]) -> "np.ndarray":
            nx, ny = q[1] - p[1], -(q[0] - p[0])
            return ((self.x - p[0]) * nx + (self.y - p[1]) * ny) / math.hypot(nx, ny)
        return np.maximum.reduce([hp(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]) - inset

    def paint(self, d: "np.ndarray", color: str, alpha: float = 1.0) -> None:
        cov = np.clip(0.5 - d * self.s, 0, 1) * alpha
        self.c = self.c * (1 - cov[..., None]) + _rgb(color) * cov[..., None]
        self.a = self.a * (1 - cov) + cov

    def erase(self, d: "np.ndarray") -> None:
        self.a *= 1 - np.clip(0.5 - d * self.s, 0, 1)

    def rgba(self, alpha: float = 1.0) -> "np.ndarray":
        return np.ascontiguousarray(np.dstack([np.clip(self.c, 0, 255), np.clip(self.a * 255 * alpha, 0, 255)])
                                    .round().astype(np.uint8))


# ---------------------------------------------------------------- recipe helpers
def _folder(c: Canvas16) -> None:
    c.paint(c.rrect(1.5, 2.5, 7.8, 7.0, 1.3), T["folder_tab"])
    c.paint(c.rrect(1.5, 4.2, 14.5, 13.5, 1.6), T["folder"])


def _badge(c: Canvas16, color: str, r: float = 3.7, cx: float = 12.0, cy: float = 11.8) -> None:
    c.erase(c.circle(cx, cy, r + 1.0))
    c.paint(c.circle(cx, cy, r), color)


def _cross(c: Canvas16, cx: float, cy: float, h: float, w: float) -> "np.ndarray":
    return np.minimum(c.seg(cx - h, cy - h, cx + h, cy + h, w), c.seg(cx - h, cy + h, cx + h, cy - h, w))


def _plus(c: Canvas16, cx: float, cy: float, h: float, w: float) -> "np.ndarray":
    return np.minimum(c.seg(cx - h, cy, cx + h, cy, w), c.seg(cx, cy - h, cx, cy + h, w))


def _page(c: Canvas16, body: str, fold: str) -> None:
    c.paint(np.maximum(c.rrect(3.0, 1.5, 13.0, 14.5, 1.2), (c.x - c.y - 8.0) / math.sqrt(2)), body)
    c.paint(c.poly([(9.5, 1.5), (13.0, 5.0), (9.5, 5.0)]), fold)


def _grid(c: Canvas16, x0: float, y0: float, cell: float, gap: float, n: int, color: str, r: float = 0.4) -> None:
    for i in range(n):
        for j in range(n):
            x, y = x0 + i * (cell + gap), y0 + j * (cell + gap)
            c.paint(c.rrect(x, y, x + cell, y + cell, r), color)


def _pyramid(c: Canvas16) -> None:
    for k, w in enumerate((2.6, 4.4, 6.2)):              # three bars, narrow to wide: the pyramid levels
        c.paint(c.rrect(7.6 - w / 2, 6.0 + k * 2.6, 7.6 + w / 2, 7.6 + k * 2.6, 0.4), "#FFFFFF")


def _arrow(c: Canvas16, tip: Tuple[float, float], tail: Tuple[float, float], head: float, color: str) -> None:
    (tx, ty), (ax, ay) = tip, tail
    n = math.hypot(ax - tx, ay - ty)
    ux, uy, k = (ax - tx) / n, (ay - ty) / n, head / math.sqrt(2)      # wings at 45 degrees, `head` long
    wing = [(tx + k * (ux - uy), ty + k * (uy + ux)), (tx, ty), (tx + k * (ux + uy), ty + k * (uy - ux))]
    c.paint(np.minimum(c.seg(tx + 0.3 * ux, ty + 0.3 * uy, ax, ay, 1.7), c.line(wing, 1.7)), color)


def _refresh(c: Canvas16) -> None:
    a1, a2 = math.radians(-75), math.radians(-15)            # gap at the top right; the arc runs clockwise into a1
    gap = c.poly([(8, 8), (8 + 9 * math.cos(a1), 8 + 9 * math.sin(a1)), (8 + 9 * math.cos(a2), 8 + 9 * math.sin(a2))])
    c.paint(np.maximum(c.ring(c.circle(8, 8, 5.0), 1.7), -gap), T["icon_ink"])
    ex, ey = 8 + 5 * math.cos(a1), 8 + 5 * math.sin(a1)
    dx, dy, rx, ry = -math.sin(a1), math.cos(a1), math.cos(a1), math.sin(a1)  # clockwise tangent, radial
    c.paint(c.poly([(ex + 3.0 * dx, ey + 3.0 * dy), (ex - 2.6 * rx, ey - 2.6 * ry), (ex + 2.6 * rx, ey + 2.6 * ry)]),
            T["icon_ink"])


def _spin(frame: int) -> Callable[[Canvas16], None]:
    def paint(c: Canvas16) -> None:
        for i in range(8):
            a = math.radians(-90 + 45 * i)
            c.paint(c.circle(8 + 5.2 * math.cos(a), 8 + 5.2 * math.sin(a), 1.45), T["text_muted"],
                    1.0 - 0.11 * ((frame - i) % 8))
    return paint


def _logo(c: Canvas16) -> None:
    for k, col in enumerate((0, 1, 2, 3, 0, 4, 1, 5, 0)):
        x, y = 0.6 + (k % 3) * 5.2, 0.6 + (k // 3) * 5.2
        c.paint(c.rrect(x, y, x + 4.4, y + 4.4, 1.0), LOGO_COLORS[col], 0.95)


_I, _P, _W = T["icon_ink"], T["place_ink"], "#FFFFFF"
RECIPES: Dict[str, Callable[[Canvas16], None]] = {
    # folders
    "folder": _folder,
    "folder_tiles": lambda c: (_folder(c), _badge(c, T["accent"], r=4.0), _grid(c, 9.75, 9.55, 1.9, 0.7, 2, _W)),
    "folder_slides": lambda c: (_folder(c), _badge(c, T["accent"], r=3.0)),
    "folder_error": lambda c: (_folder(c), _badge(c, T["error"]), c.paint(_cross(c, 12.0, 11.8, 1.5, 1.2), _W)),
    "folder_new": lambda c: (_folder(c), _badge(c, T["accent"]), c.paint(_plus(c, 12.0, 11.8, 2.0, 1.3), _W)),
    # files
    "file": lambda c: _page(c, T["border_strong"], T["folder"]),
    "file_ome": lambda c: (_page(c, _OME_PAGE, _OME_FOLD), _pyramid(c)),
    "file_tiles": lambda c: _grid(c, 1.9, 1.9, 3.5, 0.8, 3, T["accent"], r=0.7),
    # places
    "home": lambda c: (c.paint(np.minimum(c.poly([(8, 1.6), (14.8, 8.3), (1.2, 8.3)]),
                                          c.rrect(3.0, 7.0, 13.0, 14.4, 0.8)), _P),
                       c.erase(c.rrect(6.6, 9.8, 9.4, 14.8, 0.5))),
    "desktop": lambda c: (c.paint(c.rrect(1.5, 2.5, 14.5, 11.2, 1.4), _P), c.erase(c.rrect(2.9, 3.9, 13.1, 9.8, 0.5)),
                          c.paint(np.minimum(c.seg(8, 11, 8, 13.4, 1.6), c.seg(5.0, 13.8, 11.0, 13.8, 1.4)), _P)),
    "documents": lambda c: (_page(c, _P, T["border_strong"]),
                            c.paint(np.minimum.reduce([c.seg(5.4, y, 10.6, y, 1.1) for y in (7.4, 9.7, 12.0)]), _W)),
    "downloads": lambda c: (_arrow(c, (8, 10.4), (8, 1.8), 3.6, _P),
                            c.paint(c.line([(2.2, 10.0), (2.2, 13.6), (13.8, 13.6), (13.8, 10.0)], 1.6), _P)),
    "drive": lambda c: (c.paint(c.rrect(1.5, 4.0, 14.5, 12.0, 2.0), _P), c.paint(c.seg(4.2, 8, 8, 8, 1.1), T["folder"]),
                        c.paint(c.circle(11.6, 8.0, 1.1), T["ok_line"])),
    "drive_net": lambda c: (c.paint(np.minimum.reduce([c.rrect(1.5, 2.5, 14.5, 9.0, 2.0), c.seg(8, 9, 8, 12.6, 1.4),
                                                       c.seg(2, 13.3, 14, 13.3, 1.4)]), _P),
                            c.paint(c.circle(11.6, 5.75, 1.0), T["info_line"])),
    "drive_cloud": lambda c: c.paint(np.minimum.reduce([c.circle(5.6, 9.3, 3.2), c.circle(9.8, 7.4, 4.0),
                                                        c.rrect(2.2, 8.8, 14.2, 12.9, 2.0)]), _P),
    "recent": lambda c: (c.paint(c.ring(c.circle(8, 8, 5.9), 1.5), _P),
                         c.paint(c.line([(8, 4.6), (8, 8.3), (10.6, 9.8)], 1.4), _P)),
    "bookmark": lambda c: c.paint(np.maximum(c.rrect(4.0, 1.5, 12.0, 14.5, 1.0),
                                             -c.poly([(8, 10.2), (12.8, 15.6), (3.2, 15.6)])), _P),
    # health dots (bottom right; badged() composites them onto place icons)
    "dot_ok": lambda c: c.paint(c.circle(12.5, 12.5, 3.0), T["accent"]),
    "dot_slow": lambda c: c.paint(c.circle(12.5, 12.5, 3.0), T["warn_icon"]),
    "dot_bad": lambda c: c.paint(c.circle(12.5, 12.5, 3.0), T["error"]),
    "dot_off": lambda c: c.paint(c.ring(c.circle(12.5, 12.5, 2.4), 1.3), T["text_faint"]),
    # navigation
    "back": lambda c: _arrow(c, (3.0, 8), (13.0, 8), 4.4, _I),
    "fwd": lambda c: _arrow(c, (13.0, 8), (3.0, 8), 4.4, _I),
    "up": lambda c: _arrow(c, (8, 3.0), (8, 13.0), 4.4, _I),
    "refresh": _refresh,
    "chev_right": lambda c: c.paint(c.line([(6.2, 4.2), (10, 8), (6.2, 11.8)], 1.6), _I),
    "chev_down": lambda c: c.paint(c.line([(4.2, 6.2), (8, 10), (11.8, 6.2)], 1.6), _I),
    "search": lambda c: c.paint(np.minimum(c.ring(c.circle(6.8, 6.8, 4.3), 1.7), c.seg(10, 10, 13.6, 13.6, 2.2)), _I),
    "close": lambda c: c.paint(_cross(c, 8, 8, 4.0, 1.7), _I),
    "sort_up": lambda c: c.paint(c.poly([(8, 5.2), (11.6, 10.4), (4.4, 10.4)]), T["text_muted"]),
    "sort_down": lambda c: c.paint(c.poly([(4.4, 5.6), (11.6, 5.6), (8, 10.8)]), T["text_muted"]),
    # status
    "ready": lambda c: c.paint(c.ring(c.circle(8, 8, 5.7), 1.7), T["accent"]),
    "wait": lambda c: (c.paint(c.ring(c.circle(8, 8, 5.7), 1.4), T["text_faint"]),
                       c.paint(c.line([(8, 5.0), (8, 8.3), (10.3, 9.6)], 1.3), T["text_faint"])),
    "run": lambda c: (c.paint(c.circle(8, 8, 6.8), T["focus"]), c.paint(c.circle(8, 8, 2.6), _W)),
    "ok": lambda c: (c.paint(c.circle(8, 8, 6.8), T["ok"]),
                     c.paint(c.line([(4.8, 8.2), (7.1, 10.5), (11.3, 5.9)], 1.7), _W)),
    "warn": lambda c: (c.paint(c.poly([(8, 1.6), (15, 14), (1, 14)], 0.3), T["warn_icon"]),
                       c.paint(np.minimum(c.seg(8, 6.2, 8, 9.7, 1.6), c.circle(8, 11.8, 0.95)), T["text"])),
    "error": lambda c: (c.paint(c.circle(8, 8, 6.8), T["error"]), c.paint(_cross(c, 8, 8, 2.4, 1.6), _W)),
    "info": lambda c: (c.paint(c.circle(8, 8, 6.8), T["info"]),
                       c.paint(np.minimum(c.seg(8, 7.2, 8, 11.6, 1.7), c.circle(8, 4.8, 1.0)), _W)),
    "skip": lambda c: (c.paint(c.circle(8, 8, 6.4), T["border_strong"]), c.paint(c.seg(5.2, 8, 10.8, 8, 1.6), _W)),
    # tick boxes
    "cb_on": lambda c: (c.paint(c.rrect(1.5, 1.5, 14.5, 14.5, 3.0), T["accent"]),
                        c.paint(c.line([(4.6, 8.2), (7, 10.6), (11.6, 5.6)], 1.8), _W)),
    "cb_off": lambda c: (c.paint(c.rrect(1.5, 1.5, 14.5, 14.5, 3.0), T["border_strong"]),
                         c.paint(c.rrect(2.7, 2.7, 13.3, 13.3, 2.0), _W)),
    "cb_mixed": lambda c: (c.paint(c.rrect(1.5, 1.5, 14.5, 14.5, 3.0), T["accent"]),
                           c.paint(c.seg(4.8, 8, 11.2, 8, 1.8), _W)),
}
RECIPES.update({f"spin{i}": _spin(i) for i in range(8)})
RECIPES["logo"] = _logo
ICON_NAMES: Tuple[str, ...] = tuple(RECIPES)


def _blank(h: int, w: int) -> "np.ndarray":
    return np.zeros((h, w, 4), np.uint8)


class IconSet:
    """PhotoImages of the recipes at one icon size, cached per (name, size, variant). The cache owns the images."""

    def __init__(self, root: tk.Misc, size: int) -> None:
        self.root, self.size = root, int(size)
        self.gap = max(2, self.size // 4)                 # pair(): [left][gap][right][gap+1]
        self.pad = max(1, round(4 * self.size / 16))      # tree(): px(4) at this icon size
        self._cache: Dict[Hashable, tk.PhotoImage] = {}

    def _paint(self, name: str, size: Optional[int] = None, alpha: float = 1.0) -> "np.ndarray":
        c = Canvas16(size or self.size)
        RECIPES[name](c)
        return c.rgba(alpha)

    def _image(self, key: Hashable, make: Callable[[], "np.ndarray"]) -> tk.PhotoImage:
        img = self._cache.get(key)
        if img is None:
            img = self._cache[key] = tk.PhotoImage(master=self.root, data=base64.b64encode(encode_png(make())),
                                                   format="png")
        return img

    def get(self, name: str, dim: bool = False) -> tk.PhotoImage:
        return self._image((name, self.size, "dim" if dim else ""),
                           lambda: self._paint(name, alpha=DIM_ALPHA if dim else 1.0))

    def tree(self, name: str, dim: bool = False) -> tk.PhotoImage:
        """+px(4) transparent right padding (Treeview #0)."""
        return self._image((name, self.size, "tree-dim" if dim else "tree"), lambda: np.concatenate(
            [self._paint(name, alpha=DIM_ALPHA if dim else 1.0), _blank(self.size, self.pad)], axis=1))

    def pair(self, left: Optional[str], right: Optional[str]) -> tk.PhotoImage:
        """[left][gap][right][gap+1]; a None side is a transparent slot of the same size (alignment)."""
        s, g = self.size, self.gap
        side = (lambda n: self._paint(n) if n else _blank(s, s))
        return self._image((f"{left}+{right}", s, "pair"), lambda: np.concatenate(
            [side(left), _blank(s, g), side(right), _blank(s, g + 1)], axis=1))

    def badged(self, name: str, dot: Optional[str]) -> tk.PhotoImage:
        """dot: ok|slow|bad|off|None, composited at the bottom right with a transparent halo."""
        if not dot:
            return self.get(name)

        def make() -> "np.ndarray":
            c = Canvas16(self.size)
            RECIPES[name](c)
            c.erase(c.circle(12.5, 12.5, 4.2))
            RECIPES["dot_" + dot](c)
            return c.rgba()
        return self._image((name, self.size, "dot_" + dot), make)

    def logo(self, size: int) -> tk.PhotoImage:
        return self._image(("logo", int(size), ""), lambda: self._paint("logo", int(size)))

    def spinner(self, frame: int) -> tk.PhotoImage:
        return self.get(f"spin{frame % 8}")

    def app_icons(self) -> Tuple[tk.PhotoImage, ...]:
        """16, 32, 64 for iconphoto."""
        return tuple(self.logo(n) for n in (16, 32, 64))
