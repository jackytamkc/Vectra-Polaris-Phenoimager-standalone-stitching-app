"""The main window under Xvfb with the real FsService, discovery, plan, preflight and runner on synthetic trees
(FINAL_SPEC §2.1, §2.4, §2.5, §5.5.10; acceptance D3-D6). FakeEngine stands in for the engine except in the
byte-identity run, which uses the real engine at NUM_THREADS = 1 and compares with v1."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.conftest import REPO
from tests.fakes import ENOSPC_ERROR, FakeEngine
from yfs import APP_NAME, __version__
from yfs.core.fsservice import FsService
from yfs.core.model import OutputIndex, OutputInfo
from yfs.core.places import MountTable
from yfs.core.settings import Settings
from yfs.ui import dialogs
from yfs.ui.app import MainWindow, restore_geometry

ROOT_FS = "1 0 8:1 / / rw - ext4 /dev/sda1 rw"


def until(root, pred, timeout: float = 20.0, what: str = "condition"):
    """Run the Tk loop until pred() is truthy (returned) or fail."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        value = pred()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def pump(root, ms: int) -> None:
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


def dialog(root, cls):
    return next((w for w in root.winfo_children() if isinstance(w, cls) and w.winfo_exists()), None)


def label_texts(widget) -> str:
    """The text of every ttk label inside `widget`, at any depth (a sheet's warnings may sit in a scrolled frame)."""
    texts = [str(w.cget("text")) if w.winfo_class() == "TLabel" else label_texts(w) for w in widget.winfo_children()]
    return " ".join(t for t in texts if t)


def ready(win) -> bool:
    """Scan finished, every slide header read and the output index applied."""
    plan = win.plan()
    return (win.state == "READY" and plan is not None and not win._index_pending and
            all(r.header is not None or r.header_error for r in plan.rows()
                if r.section != "note" and r.cand.kind != "legacy_all"))


def row_of(win, name: str):
    """The plan row of the slide folder `name` (None while the search hasn't found it)."""
    plan = win.plan()
    return next((r for r in plan.rows() if r.section != "note" and r.cand.folder_name == name), None) if plan else None


def status(field) -> str:
    """A step's status line as shown."""
    return str(field._text.cget("text"))


def buttons(field):
    return [str(b.cget("text")) for b in field._actions]


def make_slides(folder, names, synth_tree, tiles: int = 12):
    """Slide folders whose tiles are links to one synthetic tile: their headers read like real ones, and 12 tiles
    make them ticked by default."""
    tile = sorted((synth_tree / "ProjectA" / "SlideA1_Scan1").iterdir())[0]
    for name in names:
        (folder / name).mkdir(parents=True)
        for i in range(tiles):
            (folder / name / f"{name}_[{5000 + 150 * i},15064]_component_data.tif").symlink_to(tile)
    return folder


def old_output(path, data=None):
    """An output file made an hour ago (so it is checked, not trusted as just written): a small valid TIFF, or
    `data`."""
    import numpy as np
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        tifffile.imwrite(path, np.zeros((64, 64), np.uint8))
    else:
        path.write_bytes(data)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    return path


@pytest.fixture
def share(tmp_path, synth_tree):
    """tmp/share, which tests declare a network share (share_mount), holding ProjectC: two slides of 12 tiles."""
    make_slides(tmp_path / "share" / "ProjectC", ["Slide1_Scan1", "Slide2_Scan1"], synth_tree)
    return tmp_path / "share"


@pytest.fixture
def gates():
    """gates() makes an Event for blocking lane calls (see hang); all of them are set when the test ends, so no
    worker thread stays stuck."""
    made = []

    def new() -> threading.Event:
        made.append(threading.Event())
        return made[-1]
    yield new
    for gate in made:
        gate.set()


def hang(monkeypatch, owner, name: str, gate: threading.Event, when=lambda *args: True):
    """Calls of owner.name block until `gate` is set, while when(*args) is true: a place that stopped answering."""
    real = getattr(owner, name)

    def blocked(*args, **kw):
        if when(*args):
            gate.wait(60)
        return real(*args, **kw)
    monkeypatch.setattr(owner, name, blocked)


@pytest.fixture(autouse=True)
def _no_stale_tk_garbage():
    """Earlier tests' worker threads (FS lanes, discovery, places) keep their destroyed Tk roots reachable for a
    moment. The runner calls gc.collect() on its own thread, which must not be the one that frees them (Tcl objects
    die on the main thread only): wait for those threads, then collect here."""
    end = time.monotonic() + 5
    while time.monotonic() < end and any(t.name.startswith(("yfs-fs-", "yfs-discovery", "yfs-runner", "yfs-places"))
                                         for t in threading.enumerate()):
        time.sleep(0.02)
    gc.collect()
    yield


def share_mount(path) -> str:
    """A mountinfo line that makes `path` a network share: a lane of its own, with network deadlines."""
    return f"2 1 0:60 / {path} rw - cifs //fileserver.example.org/labshare rw"


@pytest.fixture
def make_window(ui_ctx, cfg_dir, tmp_path, monkeypatch):
    """MainWindow(ctx) factory with a real FsService and settings in a temporary folder (and a private HOME).
    mounts: extra mountinfo lines (share_mount) besides the root file system; MountTable.load() reads this table
    too (the window reloads it after each lookup, UD2), not the computer's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    made = []

    def make(engine=None, input=None, output=None, settings=None, restore=True, os_api=None, mounts=()):
        mounts = MountTable.from_mountinfo_text("\n".join((ROOT_FS,) + tuple(mounts)))
        monkeypatch.setattr(MountTable, "load", classmethod(lambda cls, timeout_s=1.0: mounts))
        fs = FsService(ui_ctx.bus.post, mounts, os_api=os_api)
        ctx = dataclasses.replace(ui_ctx, fs=fs, mounts=mounts, settings=settings or Settings.load(),
                                  engine_cls=engine or FakeEngine.configure())
        win = MainWindow(ctx, None if input is None else str(input), None if output is None else str(output),
                         restore=restore)
        made.append((win, fs))
        return win
    yield make
    for win, fs in made:
        if win.runner is not None:
            win.runner.stop_now()
            win.runner.join(10)
        if win.discovery is not None:
            win.discovery.stop()
        fs.shutdown()
    for w in ui_ctx.root.winfo_children():
        w.destroy()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ D3: flows
def test_welcome_then_choose_scan_ready_sheet_run_stop_banner_new_batch(make_window, synth_tree, tmp_path):
    out = tmp_path / "out"
    Engine = FakeEngine.configure(durations={"stitch": 0.4}, steps={"stitch": 4})
    win = make_window(Engine)
    root = win.root
    # welcome (first run): no input, the plan card shows the welcome, the run bar asks for the tiles
    assert win.state == "EMPTY" and win.table._welcome is not None
    assert str(root.title()) == APP_NAME and str(win.footer.cget("text")) == "Ready"
    assert win.runbar.button.instate(["disabled"])
    # choose: output then input (as through the folder browser), scanning, ready
    win.set_output(str(out))
    win.set_input(str(synth_tree))
    until(root, lambda: win.state in ("SCANNING", "READY"), what="scanning")
    until(root, lambda: ready(win), what="ready")
    status = str(win.input_field._text.cget("text"))                     # synth/locked can't be listed (UD5)
    assert status == "Found 10 slides · 1 folder couldn't be searched"
    assert [str(b.cget("text")) for b in win.input_field._actions] == ["Show", "Rescan", "Search deeper"]
    assert "doesn't exist yet" in str(win.output_field._text.cget("text"))
    plan = win.plan()
    sus = row_of(win, "SusSlide_Scan1")
    assert sus.header.channel_problem == "duplicate"
    # R26: Fix channel names... only on a row with a channel problem
    labels = lambda rid: [win.table.menu.entrycget(i, "label") for i in range(win.table.fill_menu(rid))]  # noqa
    assert win.ctx.theme.text("Fix channel names…") in labels(sus.row_id)
    assert not any("Fix channel" in label for label in labels(row_of(win, "SlideA1_Scan1").row_id))
    # every synthetic slide has fewer than 10 tiles: tick them all, then Stitch -> the sheet (channels warning)
    plan.select("all_new")
    win._plan_edited(full=True)
    assert win.runbar.button.instate(["!disabled"])
    win.runbar.button.invoke()
    sheet = until(root, lambda: dialog(root, dialogs.ConfirmStartDialog), what="the sheet")
    texts = label_texts(sheet.body)
    assert "SusSlide_Scan1" in texts and "channel" in texts
    assert "1 folder couldn't be searched" in texts                       # UD5: the locked folder, before Start
    sheet._start()                                                       # generic names (preselected)
    until(root, lambda: win.state == "RUNNING", what="running")
    assert row_of(win, "SusSlide_Scan1").generic_accepted
    until(root, lambda: "%" in str(root.title()) and win._started >= 1, what="title")
    assert str(win.footer.cget("text")).startswith("Stitching ")
    # the stop dialog: finish the current slide, then stop
    win.runbar.stop_button.invoke()
    stop = until(root, lambda: dialog(root, dialogs.StopDialog), what="stop dialog")
    assert str(stop.title()) == "Stop stitching?"
    stop._close("after_current")
    assert win.state == "STOPPING"
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    assert win.runbar.mode == "finished" and win.runbar.banner is not None
    title = str(root.title())                                            # as the banner says: "Stopped at ..."
    assert title.startswith(win.ctx.theme.text("Stopped — ")) and title.endswith(APP_NAME), title
    assert str(win.footer.cget("text")) == "Stopped"
    done = [r for r in plan.rows() if r.status in ("done", "done_generic")]
    assert done and all((out / (r.output_stem + ".ome.tif")).is_file() for r in done)
    assert win.settings.get("last_run")["state"] == "stopped"
    # Start a new batch: rescan; stitched slides are now under Already stitched
    win.runbar.dismiss()
    win.rescan()
    until(root, lambda: ready(win) and win.plan() is not plan, what="rescan")
    until(root, lambda: all(win.plan().row(r.row_id).section == "existing" for r in done), what="existing")


def test_a_failed_slide_shows_in_the_banner_and_retry_failed_ticks_it(make_window, synth_tree, tmp_path):
    Engine = FakeEngine.configure(fail="write", last_error=ENOSPC_ERROR,
                                  fail_if=lambda eng: eng.image_name == "SlideA2_Scan1")
    win = make_window(Engine, synth_tree / "ProjectA", tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win._plan_edited(full=True)
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    rows = {r.cand.folder_name: r.status for r in win.plan().rows() if r.section != "note"}
    assert rows == {"SlideA1_Scan1": "done", "SlideA2_Scan1": "failed"}
    assert "1 failed" in str(root.title())
    banner = " ".join(str(w.cget("text")) for w in win.runbar.banner.winfo_children() if w.winfo_class() == "TLabel")
    assert "SlideA2_Scan1 failed: The disk is full." in banner
    retry = next(b for b in win.runbar.banner.winfo_children()[0].winfo_children()
                 if b.winfo_class() == "TButton" and str(b.cget("text")) == "Retry failed")
    retry.invoke()                                                       # the sheet (recent folders) comes first
    until(root, lambda: dialog(root, dialogs.ConfirmStartDialog), what="the sheet")._start()
    until(root, lambda: win.state == "FINISHED" and win._counts == {"failed": 1}, 30, what="retry")
    assert row_of(win, "SlideA1_Scan1").status == "done"                 # not stitched again


def test_closing_during_a_run_offers_quit_and_exits_after_the_slide_stops(make_window, synth_tree, tmp_path,
                                                                           monkeypatch):
    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(sys, "stdout", None)                             # PL-2: a windowed Windows build
    monkeypatch.setattr(sys, "stderr", None)
    Engine = FakeEngine.configure(durations={"stitch": 20}, steps={"stitch": 200})
    win = make_window(Engine, synth_tree / "ProjectA", tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win.start(confirm=False)
    until(root, lambda: win.runner is not None and win.runner.current_row, what="a running slide")
    win.close()                                                          # the window's close button
    quit_dlg = until(root, lambda: dialog(root, dialogs.StopDialog), what="quit dialog")
    assert str(quit_dlg.title()) == f"Quit {APP_NAME}?"
    quit_dlg._close("now")
    until(root, lambda: exits, 10, what="os._exit")
    assert exits == [0]
    saved = json.loads(Path(win.settings.path).read_text(encoding="utf-8"))
    assert saved["last_run"]["state"] == "stopped" and saved["last_run"]["planned"] == 2


def test_quit_anyway_from_the_stopping_state_exits_at_once(make_window, synth_tree, tmp_path, monkeypatch):
    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(sys, "stdout", None)                             # PL-2: a windowed Windows build
    monkeypatch.setattr(sys, "stderr", None)
    win = make_window(FakeEngine.configure(durations={"write": 20}, steps={"write": 100}), synth_tree / "ProjectA",
                      tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win.start(confirm=False)
    until(root, lambda: win._phase == "writing", what="the level-0 write")
    win.request_stop("now")
    until(root, lambda: win.runbar.quit_button.winfo_manager(), what="[Quit anyway]")
    assert "finishing the current write step" in str(win.runbar.status.cget("text"))
    errors = []                                                          # PL-2: nothing may keep it open
    monkeypatch.setattr(win.settings, "save_now", lambda timeout: 1 / 0)
    monkeypatch.setattr(root, "report_callback_exception", lambda *exc: errors.append(exc[0]))
    win.runbar.quit_button.invoke()
    until(root, lambda: exits, 2, what="bounded background shutdown")
    assert exits == [0] and errors == []


def test_missing_and_pasted_file_paths(make_window, synth_tree, tmp_path):
    win = make_window()
    root = win.root
    win.set_input(str(synth_tree / "ProjectA" / "nope" / "deeper"))
    until(root, lambda: "Can't find" in str(win.input_field._text.cget("text")), what="missing")
    assert str(synth_tree / "ProjectA") in str(win.input_field._text.cget("text")) and win.state == "EMPTY"
    go = win.input_field._actions[0]
    assert str(go.cget("text")) == "Go there"
    tile = sorted((synth_tree / "ProjectA" / "SlideA1_Scan1").iterdir())[0]
    win.set_input(f'"{tile}"')                                           # pasted with quotes: its folder is used
    until(root, lambda: win.state == "READY", what="ready")
    assert win.source == str(tile.parent) and win.input_field.get() == str(tile.parent)
    assert "Using the folder of the pasted file." in str(win.input_field._text.cget("text"))
    assert [r.cand.folder_name for r in win.plan().rows() if r.section != "note"] == ["SlideA1_Scan1"]


def test_no_slides_found_offers_search_deeper(make_window, synth_tree):
    win = make_window(input=synth_tree / "unrelated")
    root = win.root
    until(root, lambda: win.state == "READY", what="ready")
    status = win.input_field
    assert str(status._text.cget("text")).startswith("No slides found within 3 levels. Found: 1 folder of other")
    assert [str(b.cget("text")) for b in status._actions] == ["Search deeper", win.ctx.theme.text(
        "Choose another folder…")]
    for depth in (6, 9):                                                 # DeepSlide is 10 levels down
        status._actions[0].invoke()
        until(root, lambda: win.state == "READY" and win._depth() == depth, what=f"depth {depth}")
        assert str(status._text.cget("text")).startswith(f"No slides found within {depth} levels.")
    status._actions[0].invoke()
    until(root, lambda: win.state == "READY" and win._depth() == 12, what="depth 12")
    assert [r.cand.folder_name for r in win.plan().rows("new")] == ["DeepSlide_Scan1"]


def test_output_folder_states_and_leftovers(make_window, synth_tree, tmp_path):
    out = tmp_path / "out"
    pdir = out / ".stitcher-in-progress"
    pdir.mkdir(parents=True)
    old = pdir / "Orphan_Scan1.ome.tif"
    old.write_bytes(b"x" * 2048)
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    win = make_window(input=synth_tree / "ProjectA", output=out)
    root = win.root
    until(root, lambda: ready(win), what="ready")
    text = lambda: str(win.output_field._text.cget("text"))                # noqa: E731
    assert text() == "1 leftover file (2 KB) from an unfinished run."
    win.output_field._actions[0].invoke()
    confirm = until(root, lambda: dialog(root, dialogs._Dialog), what="confirmation")
    items = confirm.listing.winfo_children()[0]
    assert "Orphan_Scan1.ome.tif" in str(items.winfo_children()[0].cget("text"))
    confirm._close(True)
    until(root, lambda: not old.exists() and text().startswith("Local disk"), what="deleted")
    # a slide folder as the output: holds tiles
    win.set_output(str(synth_tree / "ProjectA" / "SlideA2_Scan1"))
    until(root, lambda: "holds tiles" in text(), what="holds tiles")
    win.output_field._actions[0].invoke()                                # Use a "stitched" subfolder
    until(root, lambda: not win._output_resolving, what="resolved output")
    assert win.out_dir == str(synth_tree / "ProjectA" / "SlideA2_Scan1" / "stitched")
    until(root, lambda: "doesn't exist yet" in text(), what="new folder")


def test_output_chips_when_the_output_is_empty(make_window, synth_tree):
    win = make_window(input=synth_tree / "ProjectA")
    root = win.root
    until(root, lambda: win.state == "READY", what="ready")
    win._status2_from_index()
    chips = [str(b.cget("text")) for b in win.output_field._actions]
    assert chips == ['New "stitched" folder next to the tiles', "Save next to the 1 stitched file already in ProjectA"]
    win.output_field._actions[1].invoke()
    until(root, lambda: not win._output_resolving, what="resolved output")
    assert win.out_dir == str(synth_tree / "ProjectA")


def test_rescan_keeps_the_users_ticks_and_names(make_window, synth_tree, tmp_path):
    win = make_window(input=synth_tree / "ProjectA", output=tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    a1 = row_of(win, "SlideA1_Scan1").row_id
    win.plan().toggle([a1], True)
    assert win.plan().rename_output(a1, "Renamed") is None
    old = win.plan()
    root.event_generate("<F5>")
    until(root, lambda: win.plan() is not old and ready(win), what="rescan")
    r = win.plan().row(a1)
    assert r.checked and r.output_stem == "Renamed"


def test_resume_line_after_a_run_that_did_not_finish(make_window, synth_tree, tmp_path, cfg_dir):
    s = Settings.load()
    s.set("paths.input", str(synth_tree / "ProjectA"))
    s.set("last_run", {"state": "running", "started": "2026-10-10T21:30:00", "ended": "",
                       "input": str(synth_tree / "ProjectA"), "output": str(tmp_path / "o"),
                       "planned": 6, "done": 4, "failed": 0})
    assert s.save_now()
    win = make_window(settings=Settings.load())                          # R23: the last input is searched again
    root = win.root
    until(root, lambda: win.state == "READY", what="auto-rescan")
    assert ("Your last batch (10 Oct, 21:30) stopped after 4 of 6 slides " in
            win.ctx.theme.text(str(win.input_field._text.cget("text"))).replace(" - ", " — "))


def test_cloud_input_is_not_rescanned_at_launch(make_window, tmp_path, cfg_dir, ui_ctx):
    cloud = tmp_path / "clouddrive"
    cloud.mkdir()
    s = Settings.load()
    s.set("paths.input", str(cloud))
    assert s.save_now()
    mounts = MountTable.from_mountinfo_text(f"{ROOT_FS}\n2 1 0:50 / {cloud} rw - fuse.rclone remote: rw")
    fs = FsService(ui_ctx.bus.post, mounts)
    ctx = dataclasses.replace(ui_ctx, fs=fs, mounts=mounts, settings=Settings.load(), engine_cls=FakeEngine)
    win = MainWindow(ctx)
    pump(win.root, 600)
    assert win.state == "EMPTY" and win.input_field.get() == str(cloud)
    assert str(win.input_field._text.cget("text")).startswith("Press Rescan (F5)")
    fs.shutdown()


def test_settings_reset_note_and_background_error_banner(make_window, cfg_dir):
    (cfg_dir / "settings.json").write_text("{broken", encoding="utf-8")
    win = make_window(settings=Settings.load())
    assert "Settings were reset (the file was damaged)." in str(win.input_field._text.cget("text"))
    from yfs.core.events import BackgroundError
    win.ctx.bus.post(BackgroundError("thread x", "Traceback: boom"))
    until(win.root, lambda: win.error_banner.winfo_manager(), what="error banner")
    win._copy_error()
    assert win.root.clipboard_get() == "Traceback: boom"


def test_keyboard_shortcuts_open_their_targets(make_window, synth_tree, monkeypatch):
    calls = []
    win = make_window()
    for name in ("browse_input", "browse_output", "open_settings", "info"):
        monkeypatch.setattr(win, name, lambda *a, n=name: calls.append(n))
    win._bind_keys()
    root = win.root
    root.focus_force()
    win.table.tree.focus_set()
    pump(root, 50)
    for seq in ("<Control-o>", "<Control-O>", "<Control-comma>", "<F1>", "<Control-j>"):
        win.table.tree.event_generate(seq)
    pump(root, 50)
    assert calls == ["browse_input", "browse_output", "open_settings", "info"]
    assert win.drawer.is_open


def test_ctrl_enter_stitches_from_the_plan_and_the_fields(make_window, synth_tree, tmp_path):
    """The table's own <Return> (Slide details) and the fields' (commit) must not swallow Ctrl+Enter."""
    win = make_window(FakeEngine.configure(durations={"stitch": 2}), synth_tree / "ProjectA", tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win._plan_edited(full=True)
    win.table.tree.focus_set()
    win.table.tree.focus(win.table.tree.get_children(win.table.tree.get_children("")[0])[0])
    pump(root, 50)
    win.table.tree.event_generate("<Control-Return>")
    until(root, lambda: win.state == "RUNNING" or dialog(root, dialogs.ConfirmStartDialog), what="start")
    assert dialog(root, dialogs.SlideDetailsDialog) is None


def test_search_progress_keeps_its_stop_button(make_window):
    """A refresh with the same actions keeps the buttons (a click is never lost) and the spinner turning."""
    win = make_window()
    stop = lambda: None                                                  # noqa: E731
    win._status1("spinner", "Searching… 1 folder", [("Stop searching", stop)])
    button, spin = win.input_field._actions[0], win.input_field._spin
    win._status1("spinner", "Searching… 2 folders", [("Stop searching", stop)])
    assert win.input_field._actions[0] is button and win.input_field._spin == spin
    assert str(win.input_field._text.cget("text")) == win.ctx.theme.text("Searching… 2 folders")


def test_browse_opens_next_to_the_folder_in_the_field(make_window, tmp_path, monkeypatch):
    """R25 through the main window: the parent of the field's folder, with that folder selected (also for '~/')."""
    seen = []
    monkeypatch.setattr("yfs.ui.app.FolderBrowser.open", lambda master, ctx, **kw: seen.append(kw))
    win = make_window()
    (tmp_path / "home" / "lab" / "ProjectC").mkdir(parents=True)
    win.input_field.combo.set("~/lab/ProjectC")             # an edit, not a literal filesystem selection
    win.browse_input()
    win.output_field.set(str(tmp_path / "out"))
    win.browse_output()
    assert [(kw["mode"], kw["initial"]) for kw in seen] == [
        ("input", (str(tmp_path / "home" / "lab"), "ProjectC")), ("output", (str(tmp_path), "out"))]
    seen[0]["on_done"](str(tmp_path / "home" / "lab" / "ProjectC"))     # choosing commits the field
    until(win.root, lambda: win.state == "READY", what="ready")
    assert win.source == str(tmp_path / "home" / "lab" / "ProjectC")


# ------------------------------------------------------------------------------------------------ D3: real engine
@pytest.mark.slow
def test_real_engine_run_matches_v1_byte_for_byte(make_window, synth_tree, tmp_path, legacy_engine, monkeypatch):
    import stitcher_RAM
    monkeypatch.setattr(stitcher_RAM, "NUM_THREADS", 1)
    monkeypatch.setattr(legacy_engine, "NUM_THREADS", 1)                 # v1 pastes overlaps in timing order else
    out = tmp_path / "out"
    win = make_window(stitcher_RAM.StitchingEngine, synth_tree / "ProjectA", out)
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED", 120, what="finished")
    assert {r.status for r in win.plan().rows() if r.section != "note"} == {"done"}
    assert not (out / ".stitcher-in-progress").exists()
    for name in ("SlideA1_Scan1", "SlideA2_Scan1"):
        ref = tmp_path / "v1" / f"{name}.ome.tif"
        ref.parent.mkdir(exist_ok=True)
        eng = legacy_engine.StitchingEngine(str(synth_tree / "ProjectA" / name), str(ref))
        assert eng.scan_metadata()
        canvas = eng.stitch(use_ram=True)
        eng.write_to_disk(canvas, True)
        del canvas
        assert sha256(out / f"{name}.ome.tif") == sha256(ref), name
    path = str(out / "SlideA1_Scan1.ome.tif")
    st = os.stat(path)
    assert win.settings.ledger_has(path, st.st_size, st.st_mtime_ns)    # OutputWritten -> the ledger


# ------------------------------------------------------------------------------------------------ D4: layout
def _inside(widget, root) -> bool:
    x, y = widget.winfo_rootx() - root.winfo_rootx(), widget.winfo_rooty() - root.winfo_rooty()
    return (widget.winfo_ismapped() and x >= 0 and y >= 0 and x + widget.winfo_width() <= root.winfo_width()
            and y + widget.winfo_height() <= root.winfo_height())


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_layout_at_the_minimum_size_and_the_default_size(tk_root, make_window, synth_tree, tmp_path):
    win = make_window(input=synth_tree, output=tmp_path / "out")
    root, px = win.root, win.ctx.theme.px
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win._plan_edited(full=True)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    for w, h, compact in ((px(760), px(560), True), (min(px(1040), int(sw * .9)), min(px(740), int(sh * .9)), None)):
        root.geometry(f"{w}x{h}+0+0")
        until(root, lambda: root.winfo_width() == w and root.winfo_height() == h, what="resize")
        pump(root, 300)
        if compact is not None:
            assert win._compact is compact
        assert _inside(win.runbar.button, root), (w, h)                  # the Stitch button is never clipped
        assert win.runbar.winfo_reqwidth() <= root.winfo_width()
        assert all(card.winfo_reqwidth() <= root.winfo_width() for card in win.cards)
        tree = win.table.tree
        heading = tree.winfo_height() - (tree.bbox(tree.get_children("")[0]) or (0, 0, 0, 0))[1]
        assert heading // win.ctx.theme.rowheight >= 4, (w, h)            # at least four plan rows
        assert _inside(win.footer, root)


# ------------------------------------------------------------------------------------------------ D5
_AUDIT = {"guard": (), "hits": []}


def _audit(event, args):
    guard = _AUDIT["guard"]
    if guard and event in ("open", "os.scandir", "os.listdir", "os.mkdir", "os.remove", "os.rename") and args \
            and threading.current_thread() is threading.main_thread():
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(guard):
            _AUDIT["hits"].append((event, os.fsdecode(path)))


sys.addaudithook(_audit)                      # audit hooks can't be removed: it is inert while guard is empty


def test_main_thread_stays_responsive_and_never_touches_the_tree(make_window, synth_tree, tmp_path, monkeypatch):
    """D5: during a whole session (scan, headers, index, verification, run, re-index) the pump latency stays
    below 100 ms at the 99th percentile and the Tk thread never opens, lists or stats a path of the trees."""
    work = tmp_path / "work"
    shutil.copytree(synth_tree / "ProjectA", work / "ProjectA", symlinks=True)
    hits = []
    real_stat, real_lstat = os.stat, os.lstat

    def watch(name, fn):
        def wrapped(path, *a, **kw):
            if threading.current_thread() is threading.main_thread() and str(path).startswith(str(work)):
                hits.append((name, str(path)))
            return fn(path, *a, **kw)
        return wrapped
    monkeypatch.setattr(os, "stat", watch("os.stat", real_stat))
    monkeypatch.setattr(os, "lstat", watch("os.lstat", real_lstat))
    _AUDIT["guard"], _AUDIT["hits"] = (str(work),), hits
    try:
        win = make_window(FakeEngine.configure(durations={"stitch": 0.5}, steps={"stitch": 10}),
                          work / "ProjectA", work / "out")
        root = win.root
        win.ctx.bus.lateness.clear()
        until(root, lambda: ready(win), what="ready")
        win.plan().select("all_new")
        win._plan_edited(full=True)
        win.start(confirm=False)
        until(root, lambda: win.state == "FINISHED", 30, what="finished")
        win.rescan()
        until(root, lambda: ready(win), what="rescan")
    finally:
        _AUDIT["guard"] = ()
    assert hits == []
    late = sorted(win.ctx.bus.lateness)
    assert late and late[int(len(late) * 0.99) - 1] < 0.1, late[-5:]


# ------------------------------------------------------------------------------------------------ D6
def test_smoke_test_and_diag_through_the_entry_point(xdisplay, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), YFS_CONFIG_DIR=str(tmp_path / "cfg"))
    res = subprocess.run([sys.executable, "stitcher_RAM.py", "--smoke-test", "--scale", "1.5"], cwd=str(REPO),
                         env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == f"SMOKE OK {__version__}"
    res = subprocess.run([sys.executable, "stitcher_RAM.py", "--diag"], cwd=str(REPO), env=env,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    lines = dict(line.split(": ", 1) for line in res.stdout.splitlines()[1:] if ": " in line)
    assert lines["Version"] == __version__ and lines["Settings"] == str(tmp_path / "cfg" / "settings.json")
    assert lines["Tk"].split()[0].startswith("8.6") and lines["Frozen"] == "no"
    if shutil.which("zenity"):
        version = subprocess.run(["zenity", "--version"], capture_output=True, text=True).stdout.split()[0]
        assert lines["zenity"] == version


def test_diag_without_a_display_still_exits_0(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.update(PYTHONPATH=str(REPO), YFS_CONFIG_DIR=str(tmp_path / "cfg"))
    res = subprocess.run([sys.executable, "-c", "import sys; from yfs.ui.app import main; sys.exit(main(object, "
                          "['--diag']))"], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "Tk: unavailable" in res.stdout and f"Version: {__version__}" in res.stdout


# ------------------------------------------------------------------------------------------------ pure helpers
def test_restore_geometry_rejects_mostly_off_screen_windows():
    screen, default, minimum = (1600, 1000), (1040, 740), (760, 560)
    assert restore_geometry("1100x700+100+50", screen, default, minimum) == "1100x700+100+50"
    assert restore_geometry("1100x700+1200+50", screen, default, minimum) == "1040x740+280+130"   # 2/3 off-screen
    assert restore_geometry("1100x700+-200+50", screen, default, minimum) == "1100x700+-200+50"  # 18 % off
    assert restore_geometry("", screen, default, minimum) == "1040x740+280+130"
    assert restore_geometry("400x300+0+0", screen, default, minimum) == "760x560+0+0"            # minimum size
    assert restore_geometry("3000x2000+0+0", screen, default, minimum) == "1600x1000+0+0"        # screen size


def test_output_index_status_texts(make_window, tmp_path):
    win = make_window()
    win.out_dir = str(tmp_path)
    base = dict(out_dir=str(tmp_path), exists=True, names=frozenset(), outputs={}, partials={}, leftovers=(),
                free_bytes=1300 * 1024 ** 3, lane="/", lane_class="local", is_system_volume=False)
    cases = [(dict(), "Local disk · 1.3 TB free"),
             (dict(outputs={"a.ome.tif": OutputInfo("a.ome.tif", 1, 1)}, names=frozenset({"a.ome.tif"})),
              "Local disk · 1.3 TB free · 1 stitched image already here"),
             (dict(lane_class="network", free_bytes=1200 * 1024 ** 3),
              "Network drive · ~1.2 TB free · saving over the network is slower"),
             (dict(lane_class="cloud"), "Cloud folder: stitched files will upload in the background."),
             (dict(exists=False), "This folder doesn't exist yet — it will be created.")]
    for change, expected in cases:
        win.out_index = OutputIndex(**dict(base, **change))
        win._status2_from_index()
        assert str(win.output_field._text.cget("text")) == win.ctx.theme.text(expected)
