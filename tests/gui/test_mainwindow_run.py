"""The main window around a batch (UD8): tiles missing from a stitched image are remembered with it (DS-1),
Finalize moves a finished image into place (DS-9), the stall notice names the place the engine waits for (MW-9),
the quit flow says what it will do and does it (MW-19, CT-8), the log pills count the current batch (MW-22), the
plan table and the run bar follow the window (MW-11, MW-16), slides that share a name are told apart, a windowless
build still exits (PL-2) and Tk garbage dies on the Tk thread (CT-6)."""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from test_mainwindow import (_no_stale_tk_garbage, buttons, dialog, label_texts, make_slides, make_window,  # noqa: F401
                             old_output, pump, ready, row_of, status, until)
from tests.conftest import REPO
from tests.fakes import ENOSPC_ERROR, FakeEngine
from yfs import APP_NAME
from yfs.core import preflight
from yfs.core.events import LogLine, RowBytes, RowProgress
from yfs.core.names import PARTIAL_DIR
from yfs.core.plan import row_text
from yfs.ui import app, dialogs
from yfs.ui.runpanel import ProgressModel


def button(widget, text: str):
    """The button with this text anywhere inside `widget`, or None."""
    for w in widget.winfo_children():
        if w.winfo_class() == "TButton" and str(w.cget("text")) == text:
            return w
        found = button(w, text)
        if found is not None:
            return found
    return None


def texts(widget) -> str:
    """Every label, radio button and button text inside `widget`, at any depth."""
    own = [str(w.cget("text")) for w in widget.winfo_children()
           if w.winfo_class() in ("TLabel", "TRadiobutton", "TButton")]
    return " ".join([t for t in own if t] + [texts(w) for w in widget.winfo_children()])


def batch(make_window, tmp_path, synth_tree, names=("Slide1_Scan1",), **engine):
    """A window ready to stitch the slides `names` of tmp/ProjectC into tmp/out."""
    tiles = make_slides(tmp_path / "ProjectC", list(names), synth_tree)
    win = make_window(FakeEngine.configure(**engine), tiles, tmp_path / "out")
    until(win.root, lambda: ready(win), what="ready")
    return win


def finished(win):
    win.start(confirm=False)
    until(win.root, lambda: win.state == "FINISHED", 30, what="finished")
    return win


# ------------------------------------------------------------------------------------------------ DS-1
def test_a_slide_with_missing_tiles_is_ledgered_with_them_and_says_so_after_a_rescan(make_window, tmp_path,
                                                                                    synth_tree):
    win = finished(batch(make_window, tmp_path, synth_tree, dropped=1))     # one tile the scan couldn't read
    final = tmp_path / "out" / "Slide1_Scan1.ome.tif"
    st = final.stat()
    assert win.settings.ledger_get(str(final), st.st_size, st.st_mtime_ns) == 1
    assert str(win.runbar.banner.cget("style")) == "Warn.TFrame"
    assert str(win.root.title()) == win.ctx.theme.text(f"Done — 1 stitched (1 with gaps) — {APP_NAME}")
    until(win.root, lambda: status(win.output_field).endswith("your slide is already stitched here (with gaps)"),
          what="step 2 after the batch")
    win.rescan()
    until(win.root, lambda: ready(win) and row_of(win, "Slide1_Scan1").existing_verdict == "valid", what="rescan")
    assert row_text(row_of(win, "Slide1_Scan1")) == \
        "Already stitched — 1 tile was missing when it was made. Tick to redo it"
    assert status(win.output_field).endswith("your slide is already stitched here (with gaps)")


# ------------------------------------------------------------------------------------------------ DS-9
def test_finalize_moves_an_image_an_earlier_run_left_into_place_with_its_gaps(make_window, tmp_path, synth_tree,
                                                                             monkeypatch):
    """A finished image left in the partial folder ('ready') is moved by the row's Finalize, the real
    preflight.finalize_kept in the save folder's lane; the ledger gets the gaps its sidecar kept. The click is a
    Try again for that lane if it had stopped answering."""
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")
    preflight.mark_complete(str(kept), gaps=2)
    win = batch(make_window, tmp_path, synth_tree)
    rid, passes, retry = row_of(win, "Slide1_Scan1").row_id, [], win.ctx.fs.retry
    monkeypatch.setattr(win.ctx.fs, "retry", lambda path: passes.append(path) or retry(path))
    assert win.plan().row(rid).status == "ready"
    menu = win.table.menu
    labels = [menu.entrycget(i, "label") for i in range(win.table.fill_menu(rid))]
    menu.invoke(labels.index("Finalize"))                                 # right-click -> Finalize
    final = tmp_path / "out" / "Slide1_Scan1.ome.tif"
    assert passes == [str(final)] and status(win.output_field) == win.ctx.theme.text("Moving Slide1_Scan1 into place…")
    until(win.root, lambda: row_of(win, "Slide1_Scan1").existing_verdict == "valid", what="moved and indexed")
    st = final.stat()
    assert not kept.exists() and win.settings.ledger_get(str(final), st.st_size, st.st_mtime_ns) == 2
    assert row_text(row_of(win, "Slide1_Scan1")) == \
        "Already stitched — 2 tiles were missing when it was made. Tick to redo it"
    assert status(win.output_field).endswith("your slide is already stitched here (with gaps)")   # not "checking"


def test_finalize_after_a_batch_brings_the_row_banner_and_title_up_to_date(make_window, tmp_path, synth_tree,
                                                                           monkeypatch):
    """The old file was in use when the batch ended ('finalize'); the banner's [Finalize] moves the image into
    place once it is free, and the banner and the title then tell the same as the rows."""
    real_move = preflight.move_into_place

    def in_use(partial, final, retries=5, **kwargs):
        if os.path.basename(final) == "Slide1_Scan1.ome.tif":
            raise PermissionError(13, "Permission denied")
        real_move(partial, final, retries, **kwargs)
    monkeypatch.setattr(preflight, "move_into_place", in_use)
    win = finished(batch(make_window, tmp_path, synth_tree, names=("Slide1_Scan1", "Slide2_Scan1")))
    root, t, rid = win.root, win.ctx.theme.text, row_of(win, "Slide1_Scan1").row_id
    assert win.plan().row(rid).status == "finalize"
    assert str(root.title()) == t(f"Done — 1 stitched, 1 to finalize — {APP_NAME}")
    monkeypatch.setattr(preflight, "move_into_place", real_move)          # QuPath was closed
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")    # the real engine's image
    button(win.runbar.banner, "Finalize").invoke()
    until(root, lambda: win.plan().row(rid).status == "done", what="finalized")
    assert not kept.exists() and (tmp_path / "out" / "Slide1_Scan1.ome.tif").read_bytes()[:2] in (b"II", b"MM")
    assert str(root.title()) == t(f"Done — 2 stitched — {APP_NAME}")
    assert str(win.runbar.banner.cget("style")) == "Ok.TFrame" and button(win.runbar.banner, "Finalize") is None


def test_a_damaged_finished_image_is_not_moved_and_step2_offers_a_rescan(make_window, tmp_path, synth_tree):
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif", b"not an image" * 100)
    preflight.mark_complete(str(kept))
    win = batch(make_window, tmp_path, synth_tree)
    win._note_action("finalize", row_of(win, "Slide1_Scan1").row_id)
    until(win.root, lambda: status(win.output_field).startswith("Couldn't finalize Slide1_Scan1: The finished image "
                                                                 "is damaged"), what="the error")
    assert buttons(win.output_field) == ["Rescan"] and kept.exists()
    assert not (tmp_path / "out" / "Slide1_Scan1.ome.tif").exists()


def test_closing_while_images_wait_to_be_finalized_asks_first(make_window, tmp_path, synth_tree, monkeypatch):
    def in_use(*_args, **_kw):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(preflight, "move_into_place", in_use)
    win = finished(batch(make_window, tmp_path, synth_tree))
    quits = []
    monkeypatch.setattr(win.root, "quit", lambda: quits.append(1))
    win.close()
    ask = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), what="the question")
    assert label_texts(ask.body).startswith("1 finished image is waiting to be moved into place. Quit anyway?")
    ask._cancel()
    assert quits == [] and win.root.winfo_exists()
    win.close()
    until(win.root, lambda: dialog(win.root, app._ConfirmDialog), what="the question")._close(True)    # Quit anyway
    until(win.root, lambda: quits == [1], timeout=2)


# ------------------------------------------------------------------------------------------------ MW-9
def test_the_stall_notice_names_the_place_the_engine_waits_for(make_window, tmp_path, synth_tree, monkeypatch):
    """Reading tiles: the tile folder; writing the pyramid: the save folder, and only once its file stops
    growing (the pyramid's bytes are passed on as activity)."""
    win = batch(make_window, tmp_path, synth_tree)
    rid, t0, run = row_of(win, "Slide1_Scan1").row_id, time.time(), win._run_id
    win.model, win._started, win._n_jobs = ProgressModel({rid: 1e9}), 1, 1
    win.runner = SimpleNamespace(current_row=rid, stop_now=lambda: None, join=lambda _timeout: None)
    seen = []
    on_bytes = win.model.on_bytes
    monkeypatch.setattr(win.model, "on_bytes", lambda *a: seen.append(a[-1]) or on_bytes(*a))
    win._on_row_progress(RowProgress(run, rid, "stitching", 1, 4, t0))
    assert win._run_text(t0 + 61)[1] == "No progress for 1 min — ProjectC may be stalled"
    win._on_row_progress(RowProgress(run, rid, "pyramid", 1, 3, t0 + 70))
    for dt in (100, 129):
        win._on_row_bytes(RowBytes(run, rid, dt * 10 ** 6, 10 ** 5, t0 + dt, "pyramid"))
    assert seen == ["pyramid", "pyramid"] and win._run_text(t0 + 185)[1] is None
    assert win._run_text(t0 + 190)[1] == "No progress for 1 min — writing to out seems slow or stalled"


# ------------------------------------------------------------------------------------------------ MW-11, MW-16
def test_the_issues_behind_more_are_listed_with_their_buttons(make_window):
    win = make_window()
    t = win.ctx.theme.text
    assert [i.text for i in win.runbar._hidden] == ["Choose where to save the stitched images."]
    button(win.runbar.issue, t("+1 more…")).invoke()
    more = until(win.root, lambda: dialog(win.root, dialogs.IssuesDialog), what="the list")
    assert "Choose where to save the stitched images." in label_texts(more.body)
    assert button(more.body, t("Browse…")) is not None


def test_the_plan_tables_help_and_ticks_follow_the_run(make_window, tmp_path, synth_tree):
    win = batch(make_window, tmp_path, synth_tree, names=("Slide1_Scan1", "Slide2_Scan1"),
                durations={"stitch": 2}, steps={"stitch": 20})
    help_text = lambda: str(win.table.help.cget("text")) if win.table.help.winfo_manager() else ""   # noqa: E731
    t = win.ctx.theme.text
    assert help_text() == t("Untick anything you don’t want.")
    win.start(confirm=False)
    until(win.root, lambda: win.state == "RUNNING" and help_text() == t("Untick a waiting slide to skip it."),
          what="the help during the run")
    win.request_stop("after_current")
    until(win.root, lambda: help_text() == "", what="no help once stopping")
    assert win.table.tree.set(win.table._iid[row_of(win, "Slide2_Scan1").row_id], "what") == \
        t("Won't be stitched (stopping)")


# ------------------------------------------------------------------------------------------------ MW-19, CT-8
def test_finish_then_quit_says_so_and_dont_quit_keeps_the_app_open(make_window, tmp_path, synth_tree, monkeypatch):
    win = batch(make_window, tmp_path, synth_tree, names=("Slide1_Scan1", "Slide2_Scan1"),
                durations={"stitch": 2}, steps={"stitch": 20})
    root, t, quits = win.root, win.ctx.theme.text, []
    monkeypatch.setattr(root, "quit", lambda: quits.append(1))
    win.start(confirm=False)
    until(root, lambda: win.runner is not None and win.runner.current_row, what="a running slide")
    win.close()
    until(root, lambda: dialog(root, dialogs.StopDialog), what="Quit?")._close("after_current")
    quitting = t("Quitting after Slide1_Scan1…")
    assert str(win.runbar.status.cget("text")) == quitting and str(win.footer.cget("text")) == quitting
    assert quitting in str(root.title())
    win.close()                                                          # again: does it still want to quit?
    ask = until(root, lambda: dialog(root, dialogs.StopDialog), what="the second question")
    assert t("The app will close after ‘Slide1_Scan1’ finishes") in label_texts(ask.body)
    button(ask.body, "Don't quit").invoke()
    assert str(win.runbar.status.cget("text")) == t("Stopping after Slide1_Scan1…")
    assert str(win.footer.cget("text")) == t("Stopping…")
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    pump(root, 100)
    assert quits == []


def test_quit_chosen_after_the_batch_ended_still_quits(make_window, tmp_path, synth_tree, monkeypatch):
    """CT-8: the batch ends while 'Quit?' is open; [Quit] then closes the app."""
    win = batch(make_window, tmp_path, synth_tree, durations={"stitch": 1}, steps={"stitch": 10})
    root, quits = win.root, []
    monkeypatch.setattr(root, "quit", lambda: quits.append(1))
    win.start(confirm=False)
    until(root, lambda: win.runner is not None and win.runner.current_row, what="a running slide")
    win.close()
    ask = until(root, lambda: dialog(root, dialogs.StopDialog), what="Quit?")
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    assert ask.winfo_exists() and quits == []
    assert "The batch has ended." in label_texts(ask)
    assert button(ask, "Keep stitching") is None
    button(ask, "Quit").invoke()
    until(win.root, lambda: quits == [1], timeout=2)


def test_a_stop_dialog_closes_by_itself_when_the_batch_ends(make_window, tmp_path, synth_tree):
    """UD8: nothing is left to stop, so the question goes away (and its answer can't stop the next batch)."""
    win = batch(make_window, tmp_path, synth_tree, durations={"stitch": 1}, steps={"stitch": 10})
    win.start(confirm=False)
    until(win.root, lambda: win.runner is not None and win.runner.current_row, what="a running slide")
    win.runbar.stop_button.invoke()
    stop = until(win.root, lambda: dialog(win.root, dialogs.StopDialog), what="Stop?")
    until(win.root, lambda: win.state == "FINISHED", 30, what="finished")
    assert not stop.winfo_exists() and win.runbar.mode == "finished"


# ------------------------------------------------------------------------------------------------ MW-22
def test_the_log_pills_count_the_problems_of_this_batch(make_window, tmp_path, synth_tree):
    win = batch(make_window, tmp_path, synth_tree)

    def warn() -> None:
        win.ctx.bus.post(LogLine(time.time(), 30, "", "a warning", "warn"))
        until(win.root, lambda: win.drawer.counts() == (1, 0), what="the pill")
    warn()
    finished(win)
    assert win.drawer.counts() == (0, 0)                                 # RunStarted
    warn()
    button(win.runbar.banner, "Start a new batch").invoke()
    assert win.drawer.counts() == (0, 0) and not win.drawer.warn_pill.winfo_manager()


# ------------------------------------------------------------------------------------------------ labels
def test_slides_that_share_a_name_are_told_apart_in_the_run_texts(make_window, tmp_path, synth_tree):
    for batch_name in ("batchA", "batchB"):
        make_slides(tmp_path / "ProjectC" / batch_name, ["Slide9_Scan1"], synth_tree)
    win = make_window(FakeEngine.configure(durations={"stitch": 1}, steps={"stitch": 10}, fail="write",
                                           last_error=ENOSPC_ERROR, fail_if=lambda eng: "batchB" in str(eng.input_dir)),
                      tmp_path / "ProjectC", tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    plan = win.plan()
    a, b = sorted((r for r in plan.rows("new")), key=lambda r: r.cand.rel_folder)
    names = plan.label(a), plan.label(b)
    assert names[0] != names[1] and "Slide9_Scan1" not in names
    win.start(confirm=False)
    until(root, lambda: names[0] in str(win.runbar.status.cget("text")), what="the first slide by its name")
    win.runbar.stop_button.invoke()
    stop = until(root, lambda: dialog(root, dialogs.StopDialog), what="Stop?")
    assert win.ctx.theme.text(f"Finish ‘{names[0]}’ first, then stop") in texts(stop.body)
    stop._close("keep")
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    assert f"{names[1]} failed: The disk is full." in label_texts(win.runbar.banner)


# ------------------------------------------------------------------------------------------------ PL-2, CT-6
def test_a_hung_share_at_exit_ends_the_process_without_stdout(tmp_path):
    """PL-2: main() closing with a stuck lane thread skips the interpreter's shutdown with os._exit(0), also with
    no stdout (the window is closed as soon as it opens)."""
    marker = tmp_path / "returned"
    code = ("import sys, tkinter\n"
            "from yfs.core.fsservice import FsService\n"
            "from yfs.ui import app\n"
            "from tests.fakes import FakeEngine\n"
            "FsService.has_stuck_threads = lambda self: True\n"
            "tkinter.Tk.mainloop = lambda self, n=0: None\n"
            "sys.stdout = sys.stderr = None\n"
            "app.main(FakeEngine, [])\n"
            f"open({str(marker)!r}, 'w').close()\n")
    env = dict(os.environ, PYTHONPATH=str(REPO), YFS_CONFIG_DIR=str(tmp_path / "cfg"), HOME=str(tmp_path))
    res = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=60)
    assert res.returncode == 0 and not marker.exists(), res.stderr


def test_tk_garbage_is_collected_on_the_tk_thread_before_the_runner_starts(make_window, tmp_path, synth_tree,
                                                                           monkeypatch):
    """CT-6: the runner collects garbage on its own thread after each slide; a Tk object it found would be freed
    there. The window collects first, on the Tk thread, before the runner thread exists."""
    win = batch(make_window, tmp_path, synth_tree)
    calls, collect = [], gc.collect

    def spy(*args):
        calls.append((threading.current_thread() is threading.main_thread(), win.runner.is_alive()))
        return collect(*args)
    monkeypatch.setattr(gc, "collect", spy)
    finished(win)
    assert calls and calls[0] == (True, False)
