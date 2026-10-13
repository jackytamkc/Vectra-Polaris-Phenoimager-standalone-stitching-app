"""Shared widgets (FINAL_SPEC §2.1.2, §5.5.13) and the app skeleton (UiBus, --smoke-test; §5.5.10, D6 smoke part)."""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from yfs.core import events as ev
from yfs.core.model import (Candidate, DirSummary, Estimate, FsError, OutputIndex, OutputInfo, Rollup, TileGroup,
                            TileHeader)
from yfs.ui.app import UiBus
from yfs.ui.widgets import PathField, SlideFacts, TileMap, Tooltip, center_over, safe_grab

REPO = Path(__file__).resolve().parents[2]
GRID = tuple((x * 900.0, y * 700.0) for y in range(3) for x in range(3))          # 3 x 3 tile centres in µm
NAMES = tuple(f"SlideA1_Scan1_[{int(x)},{int(y)}]_component_data.tif" for x, y in GRID)


def texts(widget) -> list:
    """Texts of every label below `widget`, depth first."""
    out = []
    for child in widget.winfo_children():
        if isinstance(child, (ttk.Label, ttk.Button)) and str(child.cget("text")):
            out.append(str(child.cget("text")))
        out += texts(child)
    return out


def no_bgerror_dialog(root) -> bool:
    return not int(root.tk.call("winfo", "exists", ".bgerrorDialog"))


# ------------------------------------------------------------------------------------------------ PathField
@pytest.fixture
def field(ui_ctx, pump):
    calls = SimpleNamespace(commits=[], browses=0)

    def browse():
        calls.browses += 1
    f = PathField(ui_ctx.root, ui_ctx, placeholder="Folder with your …_component_data.tif tiles",
                  recents=lambda: [f"/recent/{i}" for i in range(12)], on_commit=calls.commits.append, on_browse=browse)
    f.pack(fill="x", padx=10, pady=10)
    other = ttk.Entry(ui_ctx.root)
    other.pack()
    pump(ui_ctx.root, 50)
    yield f, calls, other


def type_into(f: PathField, text: str) -> None:
    f.combo.delete(0, "end")
    f.combo.insert(0, text)


def test_set_never_commits_unless_asked(field, pump):
    f, calls, _ = field
    f.set("/data/project/slide_folder")
    pump(f, 50)
    assert f.get() == "/data/project/slide_folder" and calls.commits == []
    f.set("/data/other", commit=True)
    assert calls.commits == ["/data/other"]


def test_enter_commits_the_cleaned_text_even_if_unchanged(field, pump):
    f, calls, _ = field
    f.combo.focus_force()
    pump(f, 50)
    type_into(f, '  "/data/pasted folder"\n')
    f.combo.event_generate("<Return>")
    assert calls.commits == ["/data/pasted folder"] and f.get() == "/data/pasted folder"
    f.combo.event_generate("<Return>")                  # Enter is explicit: commit again (rescan)
    assert calls.commits == ["/data/pasted folder"] * 2


def test_focus_out_commits_only_a_change(field, pump):
    f, calls, other = field
    f.set("/data/a")
    f.combo.focus_force()
    pump(f, 50)
    other.focus_force()
    pump(f, 50)
    assert calls.commits == []                          # unchanged
    f.combo.focus_force()
    pump(f, 50)
    type_into(f, "/data/b")
    other.focus_force()
    pump(f, 50)
    assert calls.commits == ["/data/b"]


def test_emptied_field_reverts_to_the_current_folder(field, pump):
    f, calls, other = field
    f.set("/data/a")
    f.combo.focus_force()
    pump(f, 50)
    type_into(f, "   ")
    other.focus_force()
    pump(f, 50)
    assert f.get() == "/data/a" and calls.commits == []


@pytest.mark.parametrize("live", [False, True])
def test_pasted_paths_are_cleaned_by_places(ui_ctx, pump, live):
    """A pasted share path becomes the folder where the share is mounted. With an FsService, its live mount table
    counts (UD2): ctx.mounts may still be the table from before the share was mounted."""
    places = pytest.importorskip("yfs.core.places")
    mounts = places.MountTable.from_mountinfo_text(
        "36 25 0:55 / /mnt/labshare rw,relatime shared:1 - cifs //fileserver.example.ac.uk/share rw,vers=3.0\n")
    stale = places.MountTable.from_mountinfo_text("")
    ctx = (dataclasses.replace(ui_ctx, mounts=stale, fs=SimpleNamespace(mounts=mounts)) if live
           else dataclasses.replace(ui_ctx, mounts=mounts))
    commits = []
    f = PathField(ui_ctx.root, ctx, placeholder="x", recents=list, on_commit=commits.append, on_browse=lambda: None)
    f.pack()
    f.combo.focus_force()
    pump(f, 50)
    type_into(f, '  "\\\\fileserver.example.ac.uk\\share\\ProjectA"  ')
    f.combo.event_generate("<Return>")
    assert commits == ["/mnt/labshare/ProjectA"] and f.get() == "/mnt/labshare/ProjectA"


def test_combobox_selection_commits(field, pump):
    f, calls, _ = field
    f.combo.tk.eval(str(f.combo.cget("postcommand")))  # what opening the dropdown does
    assert list(f.combo.cget("values")) == [f"/recent/{i}" for i in range(8)]
    f.combo.current(3)
    f.combo.event_generate("<<ComboboxSelected>>")
    assert calls.commits == ["/recent/3"]


def test_typing_an_absolute_path_commits_after_600_ms(field, pump):
    f, calls, _ = field
    f.combo.focus_force()
    pump(f, 50)
    type_into(f, "relative/folder")
    f.combo.event_generate("<KeyRelease>", keysym="r")
    pump(f, 800)
    assert calls.commits == []                          # not an absolute path: wait for Enter
    path = os.path.abspath("typed_folder")
    type_into(f, path)
    f.combo.event_generate("<KeyRelease>", keysym="r")
    pump(f, 300)
    assert calls.commits == []                          # still within the 600 ms
    pump(f, 600)
    assert calls.commits == [path]


def test_placeholder_shows_only_when_empty_and_unfocused(field, pump):
    f, _, other = field
    assert f._placeholder.winfo_manager() == "place"
    assert "component_data" in str(f._placeholder.cget("text"))
    f.combo.focus_force()
    pump(f, 50)
    assert f._placeholder.winfo_manager() == ""
    other.focus_force()
    pump(f, 50)
    assert f._placeholder.winfo_manager() == "place"
    f.set("/data/a")
    assert f._placeholder.winfo_manager() == ""


def test_status_row_icon_text_actions_and_browse(field, pump):
    f, calls, _ = field
    hits = []
    f.set_status("ok", "6 slides found (1,102 tiles) · 42 folders in 1.4 s", "Surface.Ok.TLabel",
                  [("Rescan", lambda: hits.append("rescan")), ("Search deeper", lambda: hits.append("deeper"))])
    assert str(f._text.cget("style")) == "Surface.Ok.TLabel" and str(f._icon.cget("image"))
    buttons = [w for w in f.status.winfo_children() if isinstance(w, ttk.Button)]
    assert [str(b.cget("text")) for b in buttons] == ["Rescan", "Search deeper"]
    buttons[1].invoke()
    f.browse.invoke()
    assert hits == ["deeper"] and calls.browses == 1
    f.set_status(None, "A local disk is fastest.")
    assert not [w for w in f.status.winfo_children() if isinstance(w, ttk.Button)]
    assert f._icon.winfo_manager() == ""


def test_spinner_animates_and_stops(field, pump):
    f, _, _ = field
    f.set_status("spinner", "Checking…")
    first = str(f._icon.cget("image"))
    pump(f, 400)
    assert str(f._icon.cget("image")) != first and f._spin
    f.set_status("info", "done")
    assert f._spin is None


def test_set_enabled_locks_field_browse_and_actions(field):
    f, _, _ = field
    f.set_status("error", "Can't find this folder.", "Surface.Error.TLabel", [("Go there", lambda: None)])
    f.set_enabled(False)
    action = [w for w in f.status.winfo_children() if isinstance(w, ttk.Button)][0]
    assert f.combo.instate(["disabled"]) and f.browse.instate(["disabled"]) and action.instate(["disabled"])
    f.set_enabled(True)
    assert not f.combo.instate(["disabled"]) and not f.browse.instate(["disabled"])


def test_long_status_wraps_inside_the_field(field, pump):
    f, _, _ = field
    f.master.geometry("520x200")
    f.set_status("info", "Your last batch (10 Oct, 21:30) stopped after 4 of 6 slides — the rest are ticked "
                         "below. " * 3, actions=[("Rescan", lambda: None)])
    pump(f, 150)
    assert int(f._text.cget("wraplength")) <= f.status.winfo_width()
    assert f.status.winfo_reqwidth() <= f.winfo_width() + 2


def test_destroy_cancels_pending_timers(field, pump):
    f, _, _ = field
    f.set_status("spinner", "Searching")
    f.combo.event_generate("<KeyRelease>", keysym="a")
    root = f.master
    f.destroy()
    pump(root, 800)
    assert no_bgerror_dialog(root)


def test_core_font_text_is_made_display_safe(ui_ctx):
    f = PathField(ui_ctx.root, ui_ctx, placeholder="Tiles … here", recents=list, on_commit=print,
                  on_browse=lambda: None)
    f.set_status(None, "Searching… — 中")
    shown = str(f._text.cget("text"))
    assert shown == ui_ctx.theme.text("Searching… — 中")
    assert str(f.browse.cget("text")) == "Browse" + ui_ctx.theme.G["ell"]


# ------------------------------------------------------------------------------------------------ TileMap
def rects(canvas: tk.Canvas) -> list:
    return [canvas.coords(i) for i in canvas.find_all() if canvas.type(i) == "rectangle"]


def test_tilemap_draws_one_rectangle_per_tile_plus_the_dashed_outline(tk_root, theme, pump):
    tm = TileMap(tk_root, width=260, height=200)
    tm.pack()
    pump(tk_root, 50)
    tm.show(GRID, None, None)                           # tile size unknown: median step (900 x 700 µm)
    items = rects(tm)
    assert len(items) == 10
    dashed = [i for i in tm.find_all() if tm.itemcget(i, "dash")]
    assert len(dashed) == 1
    tiles = items[:9]
    w, h = tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1]
    assert w / h == pytest.approx(900 / 700, rel=0.02)
    xs = [c for r in items for c in (r[0], r[2])]
    assert min(xs) >= theme.px(8) - 3 and max(xs) <= 260 - theme.px(8) + 3
    assert tiles[1][0] == pytest.approx(tiles[0][2], abs=0.01)            # neighbours touch: step == tile size
    tm.show(GRID, 1200.0, 700.0)                        # header tile size: overlapping tiles
    tiles = rects(tm)[:9]
    assert tiles[1][0] < tiles[0][2]
    tm.show((), None, None)
    assert not tm.find_all()


def test_tilemap_single_tile_and_redraw_on_resize(tk_root, pump):
    tm = TileMap(tk_root, width=200, height=150)
    tm.pack(fill="both", expand=True)
    tm.show([(5.0, 5.0)], None, None)
    pump(tk_root, 50)
    assert len(rects(tm)) == 2
    tm.show(GRID, None, None)
    tk_root.geometry("400x300")
    pump(tk_root, 100)
    assert len(rects(tm)) == 10 and max(r[2] for r in rects(tm)) > 200


# ------------------------------------------------------------------------------------------------ SlideFacts
def summary(**kw) -> DirSummary:
    base = dict(path="/data/ProjectA/SlideA1_Scan1", n_entries=12, truncated=False, subdirs=(),
                groups=(TileGroup("SlideA1_Scan1", "component_data", NAMES, GRID),), outputs=(), wsi=(),
                raw_scan=False, inform_export=False, other_tiffs=0, leftovers=(), notable_files=(), other_files=0)
    base.update(kw)
    return DirSummary(**base)


def test_show_folder_states(ui_ctx):
    facts = SlideFacts(ui_ctx.root, ui_ctx)
    facts.show_folder(None, None, None)
    assert texts(facts.facts) == [ui_ctx.theme.text("Checking what's inside…")]
    facts.show_folder(summary(), None, None)
    shown = texts(facts.facts)
    assert shown[0] == "SlideA1_Scan1" and ui_ctx.theme.text("Slide tiles — 9 tiles") in shown
    assert len(rects(facts.map)) == 10 and "SlideA1_Scan1" in str(facts.caption.cget("text"))
    seg = TileGroup("S1_Scan1", "binary_seg_maps", NAMES[:2], GRID[:2])
    groups = (TileGroup("S1_Scan1", "component_data", NAMES, GRID),
              TileGroup("S2_Scan1", "component_data", NAMES[:4], GRID[:4]), seg)
    two = summary(path="/data/inform_export", inform_export=True, outputs=("old.ome.tif",), wsi=("a_stitch.tif",),
                  other_tiffs=3, subdirs=("x", "y"), groups=groups)
    facts.show_folder(two, None, None)
    shown = texts(facts.facts)
    assert ui_ctx.theme.text("2 slides in one folder — 13 tiles") in shown and "inForm batch export" in shown
    assert {"1 stitched image (.ome.tif)", "3 other TIFF images", "2 folders"} <= set(shown)
    parent = summary(path="/data/ProjectA", groups=(), subdirs=("A", "B"))
    facts.show_folder(parent, Rollup(probed=40, slide_folders=40, tiles=5880, capped=True), None)
    assert "40+ slide folders inside (5,880+ tiles)" in texts(facts.facts) and not facts.map.find_all()
    assert str(facts.caption.cget("text")) == ""
    facts.show_folder(summary(groups=(), error=FsError("no_access", "You don't have permission.")), None, None)
    assert "You don't have permission." in texts(facts.facts)
    facts.show_folder(summary(path="/data/empty", groups=(), n_entries=0), None, None)
    assert "Empty folder" in texts(facts.facts)


def test_preview_takes_the_callers_title_counts_right_and_hides_an_empty_map(ui_ctx, pump):
    """BR-25: the browser's name for the folder ('Home', not the user name), '1 slide folder', and no empty white map
    box for folders without tiles; a wide view (Slide details) moves its facts to the left edge instead."""
    t = ui_ctx.theme.text
    facts = SlideFacts(ui_ctx.root, ui_ctx)
    facts.pack(anchor="nw")
    home = summary(path="/home/user", groups=(), subdirs=("Desktop", "LabShare"))
    facts.show_folder(home, Rollup(probed=2, slide_folders=1, tiles=21, capped=False), None, title="Home")
    pump(ui_ctx.root, 50)
    assert texts(facts.facts)[:2] == ["Home", "1 slide folder inside (21 tiles)"]
    assert not facts.drawing.winfo_ismapped()
    facts.show_folder(summary(groups=(TileGroup("SlideA1_Scan1", "component_data", NAMES[:1], GRID[:1]),)), None, None)
    pump(ui_ctx.root, 50)
    assert texts(facts.facts)[:2] == ["SlideA1_Scan1", t("Slide tiles — 1 tile")] and facts.map.winfo_ismapped()
    facts.show_folder(None, None, None, title="Home")
    pump(ui_ctx.root, 50)
    assert texts(facts.facts) == ["Home", t("Checking what's inside…")] and not facts.drawing.winfo_ismapped()
    wide = SlideFacts(ui_ctx.root, ui_ctx, wide=True, surface="Surface")
    wide.pack(anchor="nw")
    legacy = plan_row(header=None, est=None, flags=[])
    legacy.cand = Candidate("k", "/d/WSIs", "WSIs", "WSIs", "", "legacy", "legacy_all", ("a.tif",), (), False, False, 1)
    wide.show_row(legacy)
    pump(ui_ctx.root, 50)
    assert not wide.drawing.winfo_ismapped() and wide.facts.winfo_x() == 0 and "1 tile" in texts(wide.facts)
    wide.show_row(plan_row())
    pump(ui_ctx.root, 50)
    assert wide.drawing.winfo_ismapped() and wide.facts.winfo_x() > wide.map.winfo_width()


def test_show_folder_lists_free_space_and_outputs_of_the_save_folder(ui_ctx):
    facts = SlideFacts(ui_ctx.root, ui_ctx)
    infos = {f"s{i}.ome.tif": OutputInfo(f"S{i}.ome.tif", (i + 1) * 1024 ** 3, 0) for i in range(10)}
    idx = OutputIndex("/data/out/", True, frozenset(infos), infos, {}, (), int(1.3 * 1024 ** 4), "/data", "local",
                      False)
    facts.show_folder(summary(path="/data/out", groups=(), outputs=tuple(infos)), None, idx)
    shown = texts(facts.facts)
    assert "Free space: 1.3 TB" in shown and "S0.ome.tif · 1.0 GB" in shown
    assert sum(1 for t in shown if t.startswith("S") and ".ome.tif ·" in t) == 8
    assert ui_ctx.theme.text("…and 2 more") in shown
    facts.show_folder(summary(path="/data/elsewhere", groups=()), None, idx)
    assert not [t for t in texts(facts.facts) if t.startswith("Free space")]


def plan_row(**kw) -> SimpleNamespace:
    cand = Candidate(key="k", folder="/data/ProjectA/SlideA1_Scan1", rel_folder="ProjectA/SlideA1_Scan1",
                     folder_name="SlideA1_Scan1", prefix="SR01_Scan1", suffix="component_data", kind="component",
                     tile_names=NAMES, coords=GRID, multi_slide_folder=False, inform_export=False, depth=2)
    header = TileHeader(file="x.tif", shape=(5, 700, 900), dtype="float32", n_channels=5, tile_h=700, tile_w=900,
                        um_per_px_x=0.5, um_per_px_y=0.5, resolution_found=True, channels_extracted=("DAPI",) * 5,
                        channel_problem="duplicate", channel_colors=("#0000ff", "#00ff00", None, "#ff0000", "#ffffff"),
                        first_image_bytes=12_600_000)
    est = Estimate(canvas_w=4500, canvas_h=2800, n_channels=5, itemsize=4, raw_bytes=252_000_000, out_typical=1,
                   out_worst=2, coverage=0.45, memory_mode="ram")
    base = dict(row_id="r1", cand=cand, header=header, header_error=None, est=est, output_stem="SlideA1_Scan1", existing=None,
                channel_override=None, flags=[SimpleNamespace(code="channels", severity="warn",
                                                              text="Check channel names (DAPI ×5)")])
    base.update(kw)
    return SimpleNamespace(**base)


def test_show_row_facts_channels_and_actions(ui_ctx):
    opened, fixes = [], []
    facts = SlideFacts(ui_ctx.root, ui_ctx, wide=True, surface="Surface", on_open=opened.append,
                       on_fix_channels=fixes.append)
    facts.show_row(plan_row())
    shown = texts(facts.facts)
    t = ui_ctx.theme.text
    assert shown[0] == "SlideA1_Scan1" and "Folder: ProjectA/SlideA1_Scan1" in shown
    assert "Tiles: SR01_Scan1_[x,y]_component_data.tif" in shown and "9 tiles · 45% of the image covered" in shown
    assert "Tile 900 × 700 px · float32 · 0.500 µm/px" in shown
    assert "Output: SlideA1_Scan1.ome.tif" in shown and "Check channel names (DAPI ×5)" in shown
    assert [s for s in shown if s[:1].isdigit() and "DAPI" in s] == [f"{i}  DAPI" for i in range(5)]
    dots = [w for w in facts.facts.winfo_children() for w in w.winfo_children() if isinstance(w, tk.Canvas)]
    assert len(dots) == 5 and dots[0].itemcget(dots[0].find_all()[0], "fill") == "#0000ff"
    buttons = {str(b.cget("text")): b for b in facts.facts.winfo_children() if isinstance(b, ttk.Button)}
    buttons["Open folder"].invoke()
    buttons[t("Fix channel names…")].invoke()
    assert opened == ["/data/ProjectA/SlideA1_Scan1"] and fixes == [["r1"]]
    tiles = rects(facts.map)[:9]                         # header tile size: 900 x 0.5 µm = 450 µm wide tiles
    assert (tiles[0][2] - tiles[0][0]) < (tiles[1][0] - tiles[0][0]) + 0.01


def test_show_row_before_the_header_and_after_a_header_error(ui_ctx):
    facts = SlideFacts(ui_ctx.root, ui_ctx, wide=True, surface="Bg", on_fix_channels=lambda rows: None)
    facts.show_row(plan_row(header=None, est=None, flags=[]))
    shown = texts(facts.facts)
    t = ui_ctx.theme.text
    assert t("Tile details appear once the first tile has been read…") in shown and "9 tiles" in shown
    assert not any("Fix channel" in s or s == "Open folder" for s in shown)
    facts.show_row(plan_row(header=None, est=None, flags=[], header_error="Permission denied"))
    assert "Can't read the first tile (Permission denied)" in texts(facts.facts)
    legacy = plan_row(header=None, est=None, flags=[], channel_override=("A", "B"))
    legacy.cand = Candidate("k", "/d/WSIs", "WSIs", "WSIs", "", "legacy", "legacy_all", (), (), False, False, 1)
    facts.show_row(legacy)
    assert "Every TIFF in this folder" in texts(facts.facts) and not facts.map.find_all()


def test_channel_override_names_and_missing_names(ui_ctx):
    facts = SlideFacts(ui_ctx.root, ui_ctx, surface="Surface")
    row = plan_row(channel_override=("DAPI", "Opal 520", "Opal 570", "Opal 690", "AF"))
    facts.show_row(row)
    assert "1  Opal 520" in texts(facts.facts)
    row = plan_row()
    row.header = TileHeader("x", (3, 7, 9), "uint16", 3, 7, 9, 0.5, 0.5, True, (), "missing", (), 100)
    facts.show_row(row)
    assert [s for s in texts(facts.facts) if s.startswith(("0  ", "1  ", "2  "))] == [
        "0  Channel 0", "1  Channel 1", "2  Channel 2"]


# --------------------------------------------------------------------------------------------- modal helpers, tooltip
def test_center_over_and_safe_grab(tk_root, pump):
    tk_root.geometry("800x600+100+50")
    pump(tk_root, 50)
    win = tk.Toplevel(tk_root)
    win.geometry("300x200")
    center_over(win, tk_root)
    pump(tk_root, 50)
    assert abs(win.winfo_rootx() + 150 - (tk_root.winfo_rootx() + 400)) <= 2
    assert abs(win.winfo_rooty() + 100 - (tk_root.winfo_rooty() + 300)) <= 2
    real, attempts = win.grab_set, []

    def flaky_grab():
        attempts.append(1)
        if len(attempts) < 3:
            raise tk.TclError("grab failed: window not viewable")
        real()
    win.grab_set = flaky_grab
    safe_grab(win)
    pump(tk_root, 300)
    assert len(attempts) == 3 and win.grab_current() is win
    win.grab_release()
    gone = tk.Toplevel(tk_root)
    gone.grab_set = lambda: (_ for _ in ()).throw(tk.TclError("not viewable"))
    safe_grab(gone)
    gone.destroy()                                      # destroyed before the retry fires: nothing happens
    pump(tk_root, 200)
    assert no_bgerror_dialog(tk_root)


def test_tooltip_shows_after_the_delay_and_hides(tk_root, pump):
    button = ttk.Button(tk_root, text="Recent")
    button.pack()
    text = ["/data/labshare/ProjectA — 6 slides"]
    tip = Tooltip(button, lambda: text[0], delay_ms=100)
    pump(tk_root, 50)
    button.event_generate("<Enter>")
    pump(tk_root, 250)
    assert tip._tip is not None and tip._tip.winfo_viewable()
    label = tip._tip.winfo_children()[0]
    assert str(label.cget("text")) == tk_root._yfs_theme.text(text[0])
    button.event_generate("<Leave>")
    assert tip._tip is None
    text[0] = ""
    button.event_generate("<Enter>")
    pump(tk_root, 250)
    assert tip._tip is None                             # nothing to say: no tooltip


def test_a_tooltip_near_the_bottom_of_the_screen_opens_above_its_widget(tk_root, pump):
    """V-U2b-2: the run bar sits at the bottom of a maximised window, and its tooltips (the compact banner's problems,
    a cut issue line) must open where they can be read, not below the screen's edge."""
    button = ttk.Button(tk_root, text="Finished: 3 stitched, 3 failed.")
    button.pack()
    pump(tk_root, 50)
    tk_root.geometry(f"+0+{tk_root.winfo_screenheight() - tk_root.winfo_reqheight() - 4}")
    tip = Tooltip(button, lambda: "\n".join(f"PC-0{i}_Scan1 failed: see the log for details." for i in range(5)),
                  delay_ms=30)
    pump(tk_root, 100)
    button.event_generate("<Enter>")
    pump(tk_root, 150)
    assert tip._tip is not None and tip._tip.winfo_viewable()
    top, bottom = tip._tip.winfo_rooty(), tip._tip.winfo_rooty() + tip._tip.winfo_height()
    assert 0 <= top and bottom <= button.winfo_rooty()                   # above it, all of it on the screen


# ------------------------------------------------------------------------------------------------ app skeleton: UiBus
def drain(bus: UiBus, root, ms: int = 200, until=None) -> None:
    """Run the event loop for `ms` (or until `until()` is true)."""
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end and not (until and until()):
        root.update()
        time.sleep(0.005)


def test_uibus_delivers_in_order_by_exact_type_and_unsubscribes(tk_root):
    bus = UiBus(tk_root)
    got = []
    off = bus.subscribe(ev.LogLine, lambda e: got.append(("log", e.text)))
    bus.subscribe(ev.ScanNote, lambda e: got.append(("note", e.scan_id)))
    bus.start()
    for i in range(3):
        bus.post(ev.LogLine(0.0, 20, "", str(i), "info"))
        bus.post(ev.ScanNote(i, None))
    bus.post(object())                                   # nobody listens: ignored
    drain(bus, tk_root)
    assert got == [("log", "0"), ("note", 0), ("log", "1"), ("note", 1), ("log", "2"), ("note", 2)]
    off()
    bus.post(ev.LogLine(0.0, 20, "", "late", "info"))
    drain(bus, tk_root, 120)
    assert got[-1] == ("note", 2)
    bus.stop()


def test_uibus_coalesces_per_key_within_a_tick_and_keeps_order(tk_root):
    bus = UiBus(tk_root)
    got = []
    bus.subscribe(ev.RowProgress, lambda e: got.append(("p", e.row_id, e.current)))
    bus.subscribe(ev.LogLine, lambda e: got.append(("log", e.text)))
    bus.subscribe(ev.ScanProgress, lambda e: got.append(("scan", e.dirs_done)))
    for i in range(50):
        bus.post(ev.RowProgress(1, "a", "stitching", i, 50, 0.0))
        bus.post(ev.RowProgress(1, "b", "stitching", i, 50, 0.0))
        bus.post(ev.ScanProgress(7, i, 0, "", 0.0))
    bus.post(ev.LogLine(0.0, 20, "", "between", "info"))
    bus.post(ev.RowProgress(1, "a", "writing", 0, 0, 0.0))
    bus.start()
    drain(bus, tk_root)
    assert got == [("p", "b", 49), ("scan", 49), ("log", "between"), ("p", "a", 0)]
    bus.stop()


def test_uibus_respects_the_budget_and_runs_tick_hooks_once_per_tick(tk_root):
    bus = UiBus(tk_root, interval_ms=20, budget_ms=30)
    got, ticks = [], []

    def slow(e):
        time.sleep(0.012)
        got.append(e.text)
    bus.subscribe(ev.LogLine, slow)
    bus.add_tick_hook(lambda: ticks.append(len(got)))
    for i in range(12):
        bus.post(ev.LogLine(0.0, 20, "", str(i), "info"))
    bus.start()
    drain(bus, tk_root, 2000, until=lambda: got)        # the first tick ...
    assert 0 < len(got) < 12                             # ... cannot deliver everything within its 30 ms
    drain(bus, tk_root, 600)
    assert got == [str(i) for i in range(12)]
    assert len(ticks) >= 3 and ticks == sorted(set(ticks)) and ticks[-1] == 12
    n = len(ticks)
    drain(bus, tk_root, 100)
    assert len(ticks) == n                               # no events, no hook calls
    assert len(bus.lateness) > 3 and min(bus.lateness) >= 0.0
    bus.stop()


def test_uibus_survives_a_failing_handler_and_threads(tk_root):
    bus = UiBus(tk_root)
    reported, got = [], []
    tk_root.report_callback_exception = lambda *exc: reported.append(exc[1])
    bus.subscribe(ev.LogLine, lambda e: 1 / 0)
    bus.subscribe(ev.LogLine, lambda e: got.append(e.text))
    workers = [threading.Thread(target=lambda k=k: [bus.post(ev.LogLine(0.0, 20, "", f"{k}-{i}", "info"))
                                                      for i in range(250)]) for k in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    bus.start()
    drain(bus, tk_root, 5000, until=lambda: len(got) == 1000)
    assert len(got) == 1000 and len(reported) == 1000 and isinstance(reported[0], ZeroDivisionError)
    assert [g for g in got if g.startswith("2-")] == [f"2-{i}" for i in range(250)]
    bus.stop()
    bus.post(ev.LogLine(0.0, 20, "", "after stop", "info"))
    drain(bus, tk_root, 150)
    assert got[-1] != "after stop"


# ------------------------------------------------------------------------------------------------ app skeleton: smoke
def test_main_smoke_test_exits_cleanly(xdisplay, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), YFS_CONFIG_DIR=str(tmp_path))   # no settings or logs in the profile
    code = ("import sys; from yfs.ui.app import main; "
            "sys.exit(main(object, ['--smoke-test', '--classic', '--scale', '1.25', '--unknown-flag']))")
    res = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=60)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == "SMOKE OK 2.0.0"
