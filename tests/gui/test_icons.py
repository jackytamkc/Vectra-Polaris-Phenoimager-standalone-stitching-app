"""Runtime icons (FINAL_SPEC §4.8, acceptance D2): every recipe at 16/20/24/32 px, composites, cache, PNG encoder."""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from yfs.ui.icons import DIM_ALPHA, ICON_NAMES, RECIPES, Canvas16, IconSet, encode_png
from yfs.ui.theme import TOKENS

REPO = Path(__file__).resolve().parents[2]
SPEC_NAMES = {"folder", "folder_tiles", "folder_slides", "folder_error", "folder_new", "file", "file_ome", "file_tiles",
              "home", "desktop", "documents", "downloads", "drive", "drive_net", "drive_cloud", "recent", "bookmark",
              "dot_ok", "dot_slow", "dot_bad", "dot_off", "back", "fwd", "up", "refresh", "chev_right", "chev_down",
              "search", "close", "sort_up", "sort_down", "ready", "wait", "run", "ok", "warn", "error", "info", "skip",
              "cb_on", "cb_off", "cb_mixed", "logo"} | {f"spin{i}" for i in range(8)}


def pixel(root, img, x: int, y: int) -> tuple:
    """(r, g, b) of a PhotoImage pixel as Tk decoded it."""
    value = root.tk.call(img, "get", x, y)
    return tuple(int(v) for v in (value if isinstance(value, tuple) else str(value).split()))


def rgba(name: str, size: int) -> np.ndarray:
    c = Canvas16(size)
    RECIPES[name](c)
    return c.rgba()


def test_icon_names_are_exactly_the_spec_set():
    assert set(ICON_NAMES) == SPEC_NAMES and len(ICON_NAMES) == len(SPEC_NAMES)


def test_every_icon_renders_at_every_size_in_under_250_ms(tk_root):
    started = time.perf_counter()
    for size in (16, 20, 24, 32):
        icons = IconSet(tk_root, size)
        for name in ICON_NAMES:
            img = icons.get(name)
            assert (img.width(), img.height()) == (size, size), (name, size)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.25, f"{len(ICON_NAMES) * 4} icons took {elapsed * 1000:.0f} ms"


@pytest.mark.parametrize("size", [16, 20, 24, 32])
def test_composites_have_the_right_sizes(tk_root, size):
    icons = IconSet(tk_root, size)
    gap = max(2, size // 4)
    for left, right in (("cb_on", "ok"), ("cb_off", None), (None, "info"), ("chev_down", "cb_mixed"), (None, None)):
        img = icons.pair(left, right)
        assert (img.width(), img.height()) == (2 * size + 2 * gap + 1, size)
    for dot in ("ok", "slow", "bad", "off"):
        assert (icons.badged("drive_net", dot).width(), icons.badged("drive_net", dot).height()) == (size, size)
    assert icons.badged("home", None) is icons.get("home")
    tree = icons.tree("folder")
    assert (tree.width(), tree.height()) == (size + max(1, round(4 * size / 16)), size)
    assert icons.tree("folder", dim=True) is not tree
    assert [icons.logo(n).width() for n in (28, 40, 48, 64)] == [28, 40, 48, 64]
    assert [(i.width(), i.height()) for i in icons.app_icons()] == [(16, 16), (32, 32), (64, 64)]


def test_cache_spinner_and_dim_variants(tk_root, icons):
    assert icons.get("ok") is icons.get("ok") and icons.get("ok", dim=True) is not icons.get("ok")
    assert icons.pair("cb_on", "ok") is icons.pair("cb_on", "ok")
    frames = [icons.spinner(i) for i in range(8)]
    assert len({str(f) for f in frames}) == 8 and icons.spinner(9) is frames[1]
    full = Canvas16(16)
    RECIPES["ok"](full)
    assert full.rgba(DIM_ALPHA)[..., 3].max() <= round(255 * DIM_ALPHA) + 1 < full.rgba()[..., 3].max()


def test_pixels_alpha_and_composite_slots(tk_root, icons):
    a = rgba("cb_on", 16)
    assert a[0, 0, 3] == 0 and a[3, 3, 3] == 255                          # transparent corner, opaque fill
    assert "#%02X%02X%02X" % tuple(a[3, 3, :3]) == TOKENS["accent"]
    img = icons.get("cb_on")                                              # Tk decoded colour and alpha
    assert pixel(tk_root, img, 3, 3) == tuple(a[3, 3, :3])
    assert tk_root.tk.getboolean(tk_root.tk.call(img, "transparency", "get", 0, 0))
    pair = icons.pair(None, "ok")                                         # a None slot is fully transparent
    assert all(tk_root.tk.getboolean(tk_root.tk.call(pair, "transparency", "get", x, 8)) for x in range(16))
    assert not tk_root.tk.getboolean(tk_root.tk.call(pair, "transparency", "get", 16 + icons.gap + 8, 8))


def test_badged_puts_the_health_dot_bottom_right(tk_root):
    icons = IconSet(tk_root, 32)
    img = icons.badged("drive_net", "bad")
    assert "#%02X%02X%02X" % pixel(tk_root, img, 25, 25) == TOKENS["error"]
    assert "#%02X%02X%02X" % pixel(tk_root, icons.badged("drive_net", "ok"), 25, 25) == TOKENS["accent"]


def test_encode_png_round_trip():
    arr = np.random.default_rng(1).integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    png = encode_png(arr)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        chunks[kind] = data
        pos += 12 + length
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (7, 5, 8, 6, 0, 0, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [raw[i * (1 + 7 * 4):(i + 1) * (1 + 7 * 4)] for i in range(5)]
    assert all(r[0] == 0 for r in rows)
    assert np.array_equal(np.frombuffer(b"".join(r[1:] for r in rows), np.uint8).reshape(5, 7, 4), arr)


def test_no_image_files_are_shipped():
    exts = {".png", ".gif", ".ico", ".icns", ".jpg", ".jpeg", ".bmp", ".svg", ".xbm", ".xpm", ".ppm", ".webp"}
    found = [p for top in (REPO / "yfs", REPO / "tests") for p in top.rglob("*")
             if p.suffix.lower() in exts and "_shots" not in p.parts]
    found += [p for p in REPO.iterdir() if p.suffix.lower() in exts]
    assert not found, found
