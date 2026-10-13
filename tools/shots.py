"""Scripted screenshots of the real app under Xvfb (FINAL_SPEC §7.2), with synthetic data only.

    python tools/shots.py [--python INTERP] [--scale 1.0] [--out tests/_shots/<build>-<scale>] [--only NAME ...]

--python re-runs this script with another interpreter (e.g. the Xft Tk build). DISPLAY is used when set; otherwise
a private Xvfb (2400x1700) is started. A synthetic lab tree with neutral names is generated in a temporary folder
(HOME points into it, so paths show as ~/...): realistic tile headers with zero pixels (sizes look real, files are
small), a sparse "existing" output and a tiny slide for one real-engine run. MainWindow is driven through its
public methods with FakeEngine (tests/fakes.py); the last scenario stitches with the real engine.
The live workflow is followed by deterministic review snapshots for every main-window state, completion outcome,
browser mode (including empty, cloud, hung and failed folders), dialog variant and minimum-size layout. These
snapshots use the same UI methods as incoming events; no extra stitching or destructive actions are started.
SHOTS below is the complete capture manifest. A missing requested PNG makes the script fail.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SHOTS = ("main_welcome", "issues", "main_scanning", "main_ready_warn", "main_min_760x560", "sheet", "channels",
         "main_running", "stop", "stop_quitting", "main_done", "main_done_760x560", "log_open", "browser_input",
         "browser_hung", "browser_output", "browser_pathedit", "details", "settings", "browser_temp", "about", "how",
         "main_done_real", "main_welcome_min", "main_empty", "main_checking", "main_paused", "main_running_min",
         "main_stopping", "main_quitting", "main_stopping_write", "main_done_clean", "main_done_gaps",
         "main_done_finalize", "main_done_finalize_min", "main_stopped", "main_output_unavailable", "sheet_many",
         "sheet_many_bottom", "issues_many", "new_folder", "new_folder_error", "settings_read_only", "stop_closing",
         "stop_stopping_now", "stop_finished", "stop_long_name", "confirm_delete", "confirm_finalize",
         "issues_many_bottom", "plan_notes", "plan_rename", "plan_select", "confirm_delete_many",
         "confirm_delete_many_bottom") + tuple(
             f"browser_{mode}_{state}" for mode in ("input", "output", "temp")
             for state in ("cloud", "empty", "error", "hung", "min") if (mode, state) != ("input", "hung"))
DAY = time.mktime((2026, 9, 3, 14, 0, 0, 0, 0, -1))          # every file and folder of the tree is "from 3 Sep"


def _xvfb() -> "subprocess.Popen":
    for n in range(150, 200):
        if not os.path.exists(f"/tmp/.X11-unix/X{n}") and not os.path.exists(f"/tmp/.X{n}-lock"):
            proc = subprocess.Popen(["Xvfb", f":{n}", "-screen", "0", "2400x1700x24", "-nolisten", "tcp"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                if os.path.exists(f"/tmp/.X11-unix/X{n}"):
                    os.environ["DISPLAY"] = f":{n}"
                    return proc
                time.sleep(0.05)
            proc.kill()
    raise SystemExit("no free X display and DISPLAY is not set")


# ---------------------------------------------------------------------------------------------------- the tree
def make_tree(home: Path) -> None:
    """A lab share with six slides (one with repeated channel names, one tiny copy in 'stamps'), raw scans, other
    TIFFs and an inForm marker; a local data disk with an earlier stitched file; a tiny pilot slide."""
    import numpy as np
    import tifffile
    sys.path.insert(0, str(REPO / "tests" / "fixtures"))
    import make_synth
    names = ["DAPI", "Opal 520", "Opal 570", "Opal 690", "Autofluorescence"]
    res, zero = 20065, np.zeros((1396, 1860), np.float32)          # px per cm, as the scanner writes it

    def tile(path: Path, channels, centre_um, data=zero) -> None:
        """XPosition/YPosition: the top-left corner in cm (the engine places tiles by them)."""
        h, w = data.shape
        corner = [(round(c * res / 10000) - half, res) for c, half in zip(centre_um, (w // 2, h // 2))]
        tags = [(286, 5, 1, corner[0], True), (287, 5, 1, corner[1], True)]
        with tifffile.TiffWriter(str(path)) as tif:
            for name in channels:
                tif.write(data, photometric="minisblack", compression="lzw", resolution=(res, res),
                          resolutionunit="CENTIMETER", description=make_synth.pe_xml(name), extratags=tags,
                          metadata=None)

    def slide(folder: Path, prefix: str, nx: int, ny: int, channels=names, holes=()) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        cx, cy = (nx - 1) / 2, (ny - 1) / 2
        first = None
        for j in range(ny):
            for i in range(nx):
                if ((i - cx) / (cx + 0.6)) ** 2 + ((j - cy) / (cy + 0.6)) ** 2 > 1 or (i, j) in holes:
                    continue                                   # an oval tissue outline, like a real scan
                centre = (8000 + i * 890, 30000 + j * 666)
                path = folder / f"{prefix}_[{centre[0]},{centre[1]}]_component_data.tif"
                if first is None:                              # the rest are copies: only FakeEngine stitches them
                    tile(path, channels, centre)
                    first = path
                else:
                    shutil.copyfile(first, path)
    share = home / "LabShare" / "ProjectC"
    for k, (nx, ny) in enumerate([(7, 6), (6, 5), (8, 6), (7, 7), (9, 7), (6, 6)], 1):
        slide(share / f"PC-0{k}_Scan1", f"PC-0{k}_Scan1", nx, ny, ["DAPI"] * 5 if k == 4 else names,
              holes={(2, 2)} if k == 3 else ())
    slide(share / "stamps" / "PC-02_Scan1", "PC-02_Scan1", 2, 2)
    for k in range(1, 9):
        raw = share / "round1_original" / f"PC-0{k}" / "Scan1"
        raw.mkdir(parents=True)
        (raw / f"PC-0{k}_Scan1.qptiff").write_bytes(b"")
    (share / "MultiplexOptim").mkdir()
    for k in range(4):
        tifffile.imwrite(str(share / "MultiplexOptim" / f"optim_{k}.tif"), np.zeros((64, 64), np.uint8))
    for name in ("Batch.log", "notes.txt"):
        (share / name).write_text("x\n")
    (share / "PC-01_Scan1_stitch.tif").write_bytes(b"")
    out = home / "Data" / "Stitched"
    out.mkdir(parents=True)
    with open(out / "PC-05_Scan1.ome.tif", "wb") as f:            # an earlier result: sparse, 17.7 GB
        f.truncate(int(17.7 * 1024 ** 3))
    pilot = home / "LabShare" / "Pilot" / "PL-01_Scan1"           # tiny tiles for the real-engine run
    pilot.mkdir(parents=True)
    for j in range(3):
        for i in range(4):
            gx, gy = 10000 + i * 60, 30000 + j * 45
            make_synth.write_tile(str(pilot / f"PL-01_Scan1_[{(gx + 32) // 2},{(gy + 24) // 2}]_component_data.tif"),
                                  gx, gy, w=64, h=48)
    for d in ("Desktop", "Documents", "Downloads"):
        (home / d).mkdir()
    (home / ".config").mkdir()
    (home / ".config" / "user-dirs.dirs").write_text('XDG_DESKTOP_DIR="$HOME/Desktop"\nXDG_DOCUMENTS_DIR="$HOME/'
                                                     'Documents"\nXDG_DOWNLOAD_DIR="$HOME/Downloads"\n')
    for folder in ("Empty", "Restricted", "Cloud/ProjectD", "Cloud/ProjectE"):
        (home / folder).mkdir(parents=True)
    for dirpath, dirs, files in os.walk(home):
        for name in files + dirs:
            os.utime(os.path.join(dirpath, name), (DAY, DAY))


# ---------------------------------------------------------------------------------------------------- the run
def run(out_dir: Path, scale: float, only) -> int:
    work = Path(tempfile.mkdtemp(prefix="yfs-shots-"))
    home = work / "home" / "user"
    os.environ.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), YFS_CONFIG_DIR=str(work / "cfg"))
    sys.path[:0] = [str(REPO), str(REPO / "tests" / "gui")]
    make_tree(home)
    import tkinter as tk
    import stitcher_RAM
    from shotlib import Steps, grab_widget
    from tests.fake_fs import FakeOs
    from tests.fakes import ENOSPC_ERROR, FakeEngine
    from yfs.core import fsservice, osutil
    from yfs.core.fsservice import FsService
    from yfs.core.events import ScanPaused
    from yfs.core.model import Issue, RowOutcome, RunSummary
    from yfs.core.places import MountTable
    from yfs.core.settings import Settings, log_dir
    from yfs.ui import dialogs
    from yfs.ui.app import AppContext, MainWindow, UiBus, _ConfirmDialog
    from yfs.ui.browser import FolderBrowser
    from yfs.ui.icons import IconSet
    from yfs.ui.runpanel import ProgressModel
    from yfs.ui.theme import install_indicators, install_theme

    share, data = home / "LabShare", home / "Data"
    mounts = MountTable.from_mountinfo_text("\n".join([
        "1 0 8:1 / / rw - ext4 /dev/sda1 rw",
        f"2 1 0:40 / {share} rw - cifs //fileserver.example.ac.uk/labshare rw",
        f"3 1 8:17 / {data} rw - ext4 /dev/sdb1 rw",
        f"4 1 0:41 / {home / 'Cloud'} rw - fuse.rclone remote: rw"]))
    # the app reloads the mount table (UD2): a screenshot must never show this computer's own mounts
    MountTable.load = classmethod(lambda cls, timeout_s=1.0: mounts)
    settings = Settings.load()
    existing = data / "Stitched" / "PC-05_Scan1.ome.tif"
    st = existing.stat()
    settings.ledger_add(str(existing), st.st_size, st.st_mtime_ns)    # checked once already: no I/O for it
    settings.set("recent.inputs", [{"path": str(share / "ProjectC"), "slides": 6}])
    settings.set("recent.outputs", [{"path": str(data / "Stitched")}])
    root = tk.Tk(className="YourFavouriteStitcher")
    theme = install_theme(root, scale)
    icons = IconSet(root, theme.icon_px)
    install_indicators(root, icons)
    root.iconphoto(True, *icons.app_icons())
    bus = UiBus(root)
    bus.start()
    listener = osutil.setup_logging(log_dir(), bus.post)
    fake = FakeOs()
    fs = FsService(bus.post, mounts, os_api=fake)

    def bad(eng):
        return eng.image_name == "PC-03_Scan1"

    def step(eng, phase, i):
        if phase == "stitch" and i == 5 and bad(eng):
            logging.getLogger("stitcher").error("❌ Failed to read tile PC-03_Scan1_[9780,31332]_component_data.tif: "
                                                "[Errno 5] Input/output error")
    engine = FakeEngine.configure(durations={"scan": 0.6, "stitch": 4.0, "write": 2.5}, steps={"stitch": 40,
                                  "write": 20}, fail="write", last_error=ENOSPC_ERROR, fail_if=bad, on_step=step)
    ctx = AppContext(root, theme, icons, bus, fs, mounts, settings, engine, lambda: None, frozenset, lambda: False)
    win = MainWindow(ctx, restore=False)
    px = theme.px
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    full = (min(px(1040), int(sw * .9)), min(px(740), int(sh * .9)))
    root.geometry(f"{full[0]}x{full[1]}+0+0")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = Steps(root, delay_ms=300, quit_at_end=False)       # runs inside mainloop(), like the app
    state = {}

    def shot(name: str, widget=None) -> None:
        if not only or name in only:
            grab_widget(widget or root, str(out_dir / f"{name}.png"))

    def top(cls):
        return next((w for w in root.winfo_children() if isinstance(w, cls) and w.winfo_exists()), None)

    def ready() -> bool:
        plan = win.plan()
        return win.state == "READY" and not win._index_pending and plan is not None and all(
            r.header is not None or r.header_error for r in plan.rows()
            if r.section != "note" and r.cand.kind != "legacy_all")

    def add(fn, wait=None, timeout_ms=30000):
        steps.add(fn)
        if wait is not None:
            steps.wait_until(wait, timeout_ms, 100, getattr(fn, "__name__", "step"))

    def scan():
        state["gate"] = fake.block(share / "ProjectC" / "PC-05_Scan1")      # one folder answers slowly
        win.set_output("~/Data/Stitched")
        win.set_input("~/LabShare/ProjectC")

    def select_flagged():
        state["review_plan"] = win.plan().rebuild()
        sus = next(r for r in win.plan().rows("new") if r.cand.folder_name == "PC-04_Scan1")
        state["sus"] = sus.row_id
        win.table.tree.selection_set(win.table._iid[sus.row_id])
        win.table.tree.focus(win.table._iid[sus.row_id])

    def extras(on: bool) -> None:                                    # the existing file and the small copy
        plan = win.plan()
        plan.toggle([r.row_id for r in plan.rows() if r.cand.folder_name == "PC-05_Scan1" or
                     r.cand.rel_folder.startswith("stamps")], on)
        win._plan_edited(full=True)

    def sheet():
        shot("sheet", top(dialogs.ConfirmStartDialog))
        top(dialogs.ConfirmStartDialog)._close("back")
        win.fix_channels([state["sus"]])
        for var, name in zip(top(dialogs.ChannelNamesDialog).vars, ("DAPI", "Opal 520", "Opal 570", "Opal 690",
                                                                     "Autofluorescence")):
            var.set(name)

    def channels():
        shot("channels", top(dialogs.ChannelNamesDialog))
        top(dialogs.ChannelNamesDialog)._save()
        extras(False)
        win.start(confirm=False)

    def backdate():                                   # the ETA needs a minute of history: back-date the model
        now = time.time()
        win.model._t0, win.model._sample = now - 150, (now - 20, win.model._done() * 0.8)
        return 1300

    def stop():
        win.runbar.stop_button.invoke()
        return 400

    def quitting():                                   # closing again after 'Finish …, then quit' (MW-19)
        shot("stop", top(dialogs.StopDialog))
        top(dialogs.StopDialog)._close("keep")
        dialogs.StopDialog.open(root, ctx, "PC-03_Scan1", 240.0, 2, True, lambda _choice: None, quitting_after=True)

    def more_issues():                                # '+1 more…' in the welcome run bar (MW-11)
        link = next((w for w in win.runbar.issue.winfo_children() if str(w.cget("text")).startswith("+")), None)
        if link is not None:
            link.invoke()

    def issues():
        dlg = top(dialogs.IssuesDialog)
        if dlg is not None:
            shot("issues", dlg)
            dlg._cancel()

    def browser(mode: str, initial, wait: int = 1500):
        def open_it():
            state["b"] = FolderBrowser.open(root, ctx, mode=mode, initial=initial, on_done=lambda p: None)
            width, height = min(px(1080), sw - px(40)), min(px(680), sh - px(40))
            state["b"].geometry(f"{width}x{height}+{px(20)}+{px(20)}")
            return wait
        return open_it

    def browser_shot(name: str):
        def take():
            shot(name, state["b"])
            state["b"].cancel()
        return take

    def hang():
        fsservice.DEADLINE["list_browser"]["network"] = 4
        fsservice.DEADLINE["readlink"]["network"] = 1
        fsservice.DEADLINE["locate"]["network"] = 1
        (share / "Archive").mkdir(exist_ok=True)
        state["gate"] = fake.block(share / "Archive")

    def path_edit():
        state["b"].edit_path("~/LabShare/ProjectC/PC-0")
        state["b"].after(300, lambda: state["b"]._lb_move(1))
        return 900

    def details():
        win.details(next(r.row_id for r in win.plan().rows("new") if r.cand.folder_name == "PC-03_Scan1"))

    def temp_browser():                               # Settings › Browse… for the temporary folder (BR-20)
        shot("settings", top(dialogs.SettingsDialog))
        top(dialogs.SettingsDialog)._browse_temp()
        return 1500

    def temp_browser_shot():
        settings_dialog = top(dialogs.SettingsDialog)
        b = next(w for w in settings_dialog.winfo_children() if isinstance(w, FolderBrowser))
        shot("browser_temp", b)
        b.cancel()
        settings_dialog._close(set())

    def info(page: str):
        def show():
            neutral = {"Settings": "~/.config/yourfavouritestitcher/settings.json",
                       "Logs": "~/.local/state/yourfavouritestitcher/logs"}
            real = state.setdefault("diagnostics", dialogs.diagnostics)
            dialogs.diagnostics = lambda c: [(k, neutral.get(k, v)) for k, v in real(c)]
            win.info(page)
        return show

    def real_run():                                   # the real engine, on the tiny pilot slide
        stitcher_RAM.NUM_THREADS = 1
        ctx.engine_cls = stitcher_RAM.StitchingEngine
        win.set_output("~/Data/Stitched/pilot")
        win.set_input("~/LabShare/Pilot")

    def real_start():
        win.plan().select("all_new")
        win._plan_edited(full=True)
        win.start(confirm=False)

    def review_plan():
        win.runner, win.model = None, None
        win._plan = state["review_plan"].rebuild()
        win.source, win.out_dir = str(share / "ProjectC"), str(data / "Stitched")
        win.input_field.set("~/LabShare/ProjectC")
        win.output_field.set("~/Data/Stitched")
        win.out_index = win.plan()._index
        win.runbar.dismiss()
        win._set_state("READY")
        win.table.bind_plan(win.plan())
        win._status1("ok", "Found 7 slides in 12 folders")
        win._status2_from_index()
        win._refresh_idle()

    def review_empty(checking=False):
        win.runbar.dismiss()
        win._no_input()
        win.input_field.set("~/LabShare/ProjectC" if checking else "")
        win._status1("wait" if checking else None, "Checking the folder…" if checking else
                     "The search looks 3 folder levels deep and shows you everything before stitching.")
        if checking:
            win._set_state("CHECKING")
        win._refresh_idle()

    def review_running():
        review_plan()
        plan = win.plan()
        win._n_jobs, win._started = plan.begin_run(), 1
        job = plan.claim_next(win.out_dir, None)
        win.runner = SimpleNamespace(current_row=job.row_id)
        win.model = ProgressModel({r.row_id: 1 for r in plan.pending()})
        win.model.on_progress(job.row_id, "stitching", 18, 40, time.time())
        plan.set_run_text(job.row_id, "Stitching tiles 18/40")
        win._phase, win._quit_after, win._stop_mode = "stitching", False, None
        win._set_state("RUNNING")
        win._enable_folders(False)
        win.input_field.set_status("info", "You can change folders after this batch.")
        win.table.refresh()
        win._update_run_bar()

    def review_stopping(mode="after_current", quitting=False):
        win._stop_mode, win._quit_after = mode, quitting
        if mode == "now":
            win._phase = "writing"
        win._set_state("STOPPING")
        win._show_stopping()
        win._update_run_bar()

    def review_finished(kind):
        review_plan()
        rows = [r for r in win.plan().rows("new") if r.checkable][:4]
        outcomes = []
        special = {"gaps": ("done_gaps", "Done with gaps: 2 tiles couldn't be read"),
                   "finalize": ("finalize", "Finished image kept: close the file using it, then Finalize"),
                   "stopped": ("cancelled", "Stopped: this slide can be redone"),
                   "unavailable": ("failed", "Failed: the save folder is no longer available")}
        for i, row in enumerate(rows):
            status, message = special.get(kind, ("done", "Done · 1.2 GB · 2 min")) if i == 0 else (
                "done", "Done · 1.2 GB · 2 min")
            outcome = RowOutcome(status, message, output_path=str(data / "Stitched" / (row.output_stem + ".ome.tif")),
                                 failed_tiles=2 if status == "done_gaps" else 0)
            win.plan().finish(row.row_id, outcome)
            outcomes.append((row.row_id, outcome))
        reason = {"stopped": "user", "unavailable": "output_unavailable"}.get(kind)
        summary = RunSummary(DAY, DAY + 480, dict(Counter(o.status for _, o in outcomes)), tuple(outcomes),
                             reason == "user", reason, str(data / "Stitched" / "summary.csv"))
        win._set_state("FINISHED")
        win._enable_folders(True)
        win.table.refresh()
        win._show_result(summary, new=True)

    def many_warnings():
        warnings = [Issue("warn", "replace", "2 existing stitched images will be replaced: PC-01 and PC-02.")]
        warnings += [Issue("warn", "recent_change", f"Project {i + 1}: tiles changed recently. Wait until export "
                           "finishes before stitching this batch.") for i in range(18)]
        warnings.append(Issue("warn", "channels", "PC-04 has repeated channel names."))
        dialogs.ConfirmStartDialog.open(root, ctx, warnings, [state["sus"]], lambda _choice: None)

    def dialog_shot(name, cls):
        shot(name, top(cls))
        top(cls)._cancel()

    def browser_review(mode, kind):
        folder = home / {"cloud": "Cloud", "empty": "Empty", "error": "Restricted",
                         "hung": "Empty", "min": "LabShare/ProjectC"}[kind]
        if kind == "error":
            fake.fail(folder, PermissionError(13, "Permission denied"))
        elif kind == "hung":
            hang()
            folder = share / "Archive"
        browser(mode, (str(folder), None))()
        if kind == "min":
            state["b"].geometry(f"{px(720)}x{px(480)}+0+0")

    def browser_settled():
        b = state["b"]
        return b._ov_kind in ("error", "unresponsive") or (b._listing is not None and not b._loading and
                 not any(key in b._jobs for key in ("layout", "preview", "probe", "facts")))

    def browser_review_shot(mode, kind):
        b = state["b"]
        shot(f"browser_{mode}_{kind}", b)
        b.cancel()
        if kind == "hung":
            state["gate"].set()
            fake.clear(share / "Archive")
        fake.clear(home / "Restricted")

    add(lambda: shot("main_welcome"))
    add(lambda: root.geometry(f"{px(760)}x{px(560)}+0+0") or 300)
    add(lambda: shot("main_welcome_min"))
    add(lambda: root.geometry(f"{full[0]}x{full[1]}+0+0") or 300)
    add(more_issues)
    add(issues)
    add(scan, lambda: win.plan() is not None and len(win.plan().rows("new")) >= 3)
    add(lambda: 1500)
    add(lambda: shot("main_scanning"))
    add(lambda: state["gate"].set(), ready)
    add(select_flagged)
    add(lambda: shot("main_ready_warn"))
    add(lambda: root.geometry(f"{px(760)}x{px(560)}+0+0") or 500)
    add(lambda: shot("main_min_760x560"))
    add(lambda: root.geometry(f"{full[0]}x{full[1]}+0+0") or 500)
    add(lambda: (extras(True), win.start()), lambda: top(dialogs.ConfirmStartDialog))
    add(sheet)
    add(channels, lambda: win.state == "RUNNING" and win._started >= 2 and win._phase == "stitching")
    add(backdate)
    add(lambda: shot("main_running"))
    add(stop)
    add(quitting)
    add(lambda: (shot("stop_quitting", top(dialogs.StopDialog)), top(dialogs.StopDialog).dismiss()),
        lambda: win.state == "FINISHED", 90000)
    add(lambda: shot("main_done"))
    add(lambda: root.geometry(f"{px(760)}x{px(560)}+0+0") or 500)
    add(lambda: shot("main_done_760x560"))
    add(lambda: root.geometry(f"{full[0]}x{full[1]}+0+0") or 500)
    add(lambda: win.drawer.set_open(True))
    add(lambda: (shot("log_open"), win.drawer.set_open(False), win.runbar.dismiss()))
    for mode, initial, name, before in (("input", (str(share), "ProjectC"), "browser_input", None),
                                        ("input", (str(share / "Archive"), None), "browser_hung", hang),
                                        ("output", (str(share / "ProjectC"), "PC-02_Scan1"), "browser_output", None),
                                        ("input", (str(share / "ProjectC"), None), "browser_pathedit", None)):
        if before is not None:
            add(before)
        add(browser(mode, initial, 5500 if name == "browser_hung" else 1500))
        if name == "browser_pathedit":
            add(path_edit)
        add(browser_shot(name))
        if name == "browser_hung":                    # the share answers again (its late listing arrives)
            add(lambda: state["gate"].set(), lambda: fs.health(str(share)) != "unresponsive")
    add(details)
    add(lambda: (shot("details", top(dialogs.SlideDetailsDialog)), top(dialogs.SlideDetailsDialog)._close(None),
                 win.open_settings()))
    add(temp_browser)
    add(temp_browser_shot)
    for page in ("about", "how"):
        add(info(page))
        add(lambda p=page: shot(p, top(dialogs.InfoDialog)))
    add(lambda: top(dialogs.InfoDialog)._close(None))
    if not only or "main_done_real" in only:
        add(real_run, ready)
        add(real_start, lambda: win.state == "FINISHED", 120000)
        add(lambda: shot("main_done_real"))
    add(lambda: review_empty(checking=True))
    add(lambda: shot("main_checking"))
    add(review_plan)
    add(lambda: win._set_state("SCANNING"))
    add(lambda: win._on_paused(ScanPaused(win._scan_id, "budget", 800, 60)))
    add(win._refresh_idle)
    add(lambda: shot("main_paused"))
    add(review_plan)
    add(lambda: win.table.reveal([r.row_id for r in win.plan().rows("note")], "excluded"))
    add(lambda: shot("plan_notes"))
    add(lambda: win.table.rename(state["sus"]))
    add(lambda: shot("plan_rename"))
    add(win.table._close_editor)
    add(lambda: win.table.select_menu.post(win.table.select_button.winfo_rootx(),
                                          win.table.select_button.winfo_rooty()))
    add(lambda: shot("plan_select", win.table.select_menu))
    add(win.table.select_menu.unpost)
    add(review_running)
    add(lambda: root.geometry(f"{px(760)}x{px(560)}+0+0") or 300)
    add(lambda: shot("main_running_min"))
    add(lambda: root.geometry(f"{full[0]}x{full[1]}+0+0") or 300)
    add(lambda: review_stopping())
    add(lambda: shot("main_stopping"))
    add(lambda: review_stopping(quitting=True))
    add(lambda: shot("main_quitting"))
    add(lambda: review_stopping(mode="now"))
    add(lambda: shot("main_stopping_write"))
    for kind in ("clean", "gaps", "finalize", "stopped", "unavailable"):
        add(lambda k=kind: review_finished(k))
        name = {"stopped": "main_stopped", "unavailable": "main_output_unavailable"}.get(kind, f"main_done_{kind}")
        add(lambda n=name: shot(n))
        if kind == "finalize":
            add(lambda: root.geometry(f"{px(760)}x{px(560)}+0+0") or 300)
            add(lambda: shot("main_done_finalize_min"))
            add(lambda: root.geometry(f"{full[0]}x{full[1]}+0+0") or 300)
            add(win.close)
            add(lambda: dialog_shot("confirm_finalize", _ConfirmDialog))
    add(many_warnings)
    add(lambda: shot("sheet_many", top(dialogs.ConfirmStartDialog)))
    add(lambda: top(dialogs.ConfirmStartDialog).listing.yview_moveto(1))
    add(lambda: dialog_shot("sheet_many_bottom", dialogs.ConfirmStartDialog))
    add(lambda: dialogs.IssuesDialog.open(root, ctx, [
        ("warn", f"Project {i + 1}: tiles changed recently. Check that the export has finished before stitching.",
         (("Show", "show"),)) for i in range(16)], lambda _choice: None))
    add(lambda: shot("issues_many", top(dialogs.IssuesDialog)))
    add(lambda: top(dialogs.IssuesDialog).listing.yview_moveto(1))
    add(lambda: dialog_shot("issues_many_bottom", dialogs.IssuesDialog))
    add(lambda: dialogs.NewFolderDialog.ask(root, ctx, frozenset({"stitched"}), lambda _name: None))
    add(lambda: shot("new_folder", top(dialogs.NewFolderDialog)))
    add(lambda: (top(dialogs.NewFolderDialog).var.set("stitched"), top(dialogs.NewFolderDialog)._ok()))
    add(lambda: dialog_shot("new_folder_error", dialogs.NewFolderDialog))
    add(lambda: dialogs.SettingsDialog.open(root, ctx, True, lambda _keys: None))
    add(lambda: dialog_shot("settings_read_only", dialogs.SettingsDialog))
    for kind in ("closing", "stopping_now", "finished"):
        add(lambda k=kind: dialogs.StopDialog.open(root, ctx, "PC-03_Scan1", 240, 2, True,
                                                  lambda _choice: None, stopping_now=k == "stopping_now"))
        if kind == "finished":
            add(lambda: top(dialogs.StopDialog).finished())
        add(lambda k=kind: dialog_shot(f"stop_{k}", dialogs.StopDialog))
    add(lambda: dialogs.StopDialog.open(root, ctx, "ProjectC_" + "Extended_sample_name_" * 5 + "Scan1", 240,
                                        2, True, lambda _choice: None))
    add(lambda: dialog_shot("stop_long_name", dialogs.StopDialog))
    add(lambda: _ConfirmDialog(root, ctx, "Delete leftover files?", "These files were left behind by an "
                               "unfinished run:\nPC-01_Scan1.tmp.raw (2.4 GB)\n\nFiles changed in the last "
                               "2 minutes are kept (another stitcher may be using them).", "Delete them", lambda _ok: None))
    add(lambda: dialog_shot("confirm_delete", _ConfirmDialog))
    names = ["ProjectC_" + "Extended_sample_name_" * 9 + f"{i}.tmp.raw (2.4 GB)" for i in range(10)]
    cleanup = ("These files were left behind by an unfinished run:\n" + "\n".join(names) +
               "\n\nFiles changed in the last 2 minutes are kept (another stitcher may be using them).")
    add(lambda: _ConfirmDialog(root, ctx, "Delete leftover files?", cleanup, "Delete them", lambda _ok: None))
    add(lambda: shot("confirm_delete_many", top(_ConfirmDialog)))
    add(lambda: top(_ConfirmDialog).listing.yview_moveto(1))
    add(lambda: dialog_shot("confirm_delete_many_bottom", _ConfirmDialog))
    for mode in ("input", "output", "temp"):
        for kind in ("cloud", "empty", "error", "hung", "min"):
            if (mode, kind) == ("input", "hung"):
                continue
            add(lambda m=mode, k=kind: browser_review(m, k), browser_settled)
            add(lambda: 400)
            add(lambda m=mode, k=kind: browser_review_shot(m, k))
    add(lambda: (settings.set("ui.welcome_dismissed", True), review_empty()))
    add(lambda: shot("main_empty"))
    add(root.quit)
    steps.run()
    root.mainloop()
    errors = list(steps.errors)
    errors += [f"missing screenshot: {name}" for name in (only or SHOTS) if not (out_dir / f"{name}.png").is_file()]
    if (not only or "main_done_real" in only) and not (data / "Stitched" / "pilot" / "PL-01_Scan1.ome.tif").is_file():
        errors.append("the real engine wrote no output")
    bus.stop()
    root.destroy()
    listener.stop()
    fs.shutdown()
    shutil.rmtree(work, ignore_errors=True)
    for e in errors:
        print("ERROR", e, flush=True)
    print(f"shots: {out_dir} ({len(errors)} problem(s))", flush=True)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", help="run this script with another interpreter")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", help="default: tests/_shots/<py version>-<scale>")
    ap.add_argument("--only", nargs="*", default=[], choices=SHOTS)
    args = ap.parse_args()
    out = Path(args.out or REPO / "tests" / "_shots" / f"py{sys.version_info[0]}{sys.version_info[1]}-{args.scale}")
    if args.python and os.path.realpath(args.python) != os.path.realpath(sys.executable):
        cmd = [args.python, __file__, "--scale", str(args.scale), "--out", str(out)] + (
            ["--only", *args.only] if args.only else [])
        return subprocess.call(cmd)
    xvfb = _xvfb() if not os.environ.get("DISPLAY") else None
    try:
        return run(out, args.scale, set(args.only))
    finally:
        if xvfb is not None:
            xvfb.terminate()


if __name__ == "__main__":
    sys.exit(main())
