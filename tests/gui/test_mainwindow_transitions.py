"""Verifier regressions: retired work, retry controls and Finalize results must follow the current plan."""
from __future__ import annotations

import os
import time

import pytest

from test_mainwindow import (_no_stale_tk_garbage, buttons, dialog, gates, hang, label_texts,  # noqa: F401
                             make_slides, make_window, old_output, pump, ready, row_of, share_mount, status, until)
from test_mainwindow_run import batch, button, finished
from yfs import APP_NAME
from yfs.core import fsservice, preflight
from yfs.core.events import FsDone, LaneHealthChanged, ScanFinished
from yfs.core.model import Issue
from yfs.core.names import PARTIAL_DIR
from yfs.core.plan import row_text
from yfs.ui import app, dialogs


def test_save_errors_take_priority_over_a_quiet_tick_reminder(make_window, monkeypatch):
    win = make_window()
    reminder = Issue("block", "none_ticked", "Tick at least one slide.")
    problem = Issue("block", "pdir_problem", "Can't save here: choose another save folder.", (), "browse_output")
    monkeypatch.setattr(win, "_issues", lambda: [reminder, problem])
    win._refresh_idle()
    assert "Can't save here" in label_texts(win.runbar.issue)
    assert win.runbar._hidden == (reminder,)


def finalize_from_menu(win, rid):
    menu = win.table.menu
    labels = [menu.entrycget(i, "label") for i in range(win.table.fill_menu(rid))]
    menu.invoke(labels.index("Finalize"))


def kept_batch(make_window, tmp_path, synth_tree, monkeypatch):
    """A real batch whose image couldn't be moved, and can be retried once the old file is closed."""
    move = preflight.move_into_place

    def in_use(*_a, **_kw):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(preflight, "move_into_place", in_use)
    win = finished(batch(make_window, tmp_path, synth_tree))
    until(win.root, lambda: not win._index_pending, timeout=3)
    monkeypatch.setattr(preflight, "move_into_place", move)
    return win, row_of(win, "Slide1_Scan1").row_id


def test_finalize_during_a_batch_counts_as_stitched_when_it_finishes(make_window, tmp_path, synth_tree,
                                                                    monkeypatch, gates):
    """U1b-V1: the runner's old 'finalize' outcome must not replace the successful row-menu move."""
    gate, move = gates(), preflight.move_into_place

    def in_use(partial, final, retries=5, **kwargs):
        if os.path.basename(final) == "Slide1_Scan1.ome.tif":
            raise PermissionError(13, "Permission denied")
        move(partial, final, retries, **kwargs)

    def wait_on_second(engine, phase, step):
        if engine.image_name == "Slide2_Scan1" and phase == "stitch" and step == 1:
            gate.wait(4)
    monkeypatch.setattr(preflight, "move_into_place", in_use)
    win = batch(make_window, tmp_path, synth_tree, names=("Slide1_Scan1", "Slide2_Scan1"),
                on_step=wait_on_second)
    root, rid = win.root, row_of(win, "Slide1_Scan1").row_id
    try:
        win.start(confirm=False)
        until(root, lambda: win.plan().row(rid).status == "finalize" and win._started == 2, timeout=3)
        monkeypatch.setattr(preflight, "move_into_place", move)
        old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")  # real TIFF in place of FakeEngine bytes
        finalize_from_menu(win, rid)
        until(root, lambda: win.plan().row(rid).status == "done", timeout=3)
        assert win.state == "RUNNING"
    finally:
        gate.set()
    until(root, lambda: win.state == "FINISHED", timeout=3)
    assert str(root.title()) == win.ctx.theme.text(f"Done — 2 stitched — {APP_NAME}")
    assert "2 stitched" in label_texts(win.runbar.banner)
    assert "to finalize" not in label_texts(win.runbar.banner)
    assert button(win.runbar.banner, "Finalize") is None


@pytest.mark.parametrize("verdict", ["unchecked", "unreadable"])
def test_finalize_failure_reason_agrees_in_the_row_banner_and_save_step(make_window, tmp_path, synth_tree,
                                                                       monkeypatch, verdict):
    """U1b-V5/V6: NotChecked can be retried; damage needs a rescan. Neither still blames the old file being open."""
    win, rid = kept_batch(make_window, tmp_path, synth_tree, monkeypatch)
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")
    verify = preflight.verify_output
    reason = "the disk reported a read error" if verdict == "unchecked" else "it is not a TIFF image"
    monkeypatch.setattr(preflight, "verify_output", lambda _p: (verdict, reason))
    button(win.runbar.banner, "Finalize").invoke()
    until(win.root, lambda: win._save_error is not None, timeout=3)
    assert buttons(win.output_field) == (["Try again"] if verdict == "unchecked" else ["Rescan"])
    text = "couldn't be checked" if verdict == "unchecked" else "is damaged"
    assert text in status(win.output_field) and text in row_text(win.plan().row(rid))
    assert text in label_texts(win.runbar.banner) and "old file is in use" not in label_texts(win.runbar.banner)
    assert "old file is in use" not in row_text(win.plan().row(rid))
    assert kept.exists() and not (tmp_path / "out" / kept.name).exists()
    if verdict == "unchecked":
        monkeypatch.setattr(preflight, "verify_output", verify)
        win.output_field._actions[0].invoke()
        until(win.root, lambda: win.plan().row(rid).status == "done", timeout=3)
        assert button(win.runbar.banner, "Finalize") is None


@pytest.mark.parametrize("missing", [False, True])
def test_a_new_search_retires_the_finalize_retry_and_ignores_its_old_row(make_window, tmp_path, synth_tree,
                                                                       monkeypatch, missing):
    """U1b-V2: a saved or queued action is harmless after its plan is replaced, even before any new row arrives."""
    win, rid = kept_batch(make_window, tmp_path, synth_tree, monkeypatch)

    def in_use(*_a, **_kw):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(preflight, "finalize_kept", in_use)
    monkeypatch.setattr(preflight, "RETRY_DELAY_S", .01)
    finalize_from_menu(win, rid)
    until(win.root, lambda: win._save_error is not None, timeout=3)
    retry = win._save_error[1][0][1]
    assert buttons(win.output_field) == ["Try again"]
    monkeypatch.setattr(app.Discovery, "start", lambda _self: None)  # new search has not delivered a candidate
    if missing:
        win.set_input(str(tmp_path / "missing"))
        until(win.root, lambda: win.state == "EMPTY", timeout=3)
        assert win.plan() is None
    else:
        win.rescan()
        until(win.root, lambda: win.state == "SCANNING", timeout=3)
        assert win.plan().rows() == []
    assert win._save_error is None and "Try again" not in buttons(win.output_field)
    retry()                                                       # a callback queued just before the rescan
    assert not win.error_banner.winfo_manager()


def test_the_busy_slides_rescan_button_starts_a_fresh_search(make_window, tmp_path, synth_tree):
    """U1b-V3: the issue's next step works after the other stitcher has stopped writing."""
    kept = tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"being written")
    win = batch(make_window, tmp_path, synth_tree)
    assert row_of(win, "Slide1_Scan1").status == "busy"
    kept.unlink()
    button(win.runbar.issue, "Rescan").invoke()
    assert win.state == "CHECKING"
    until(win.root, lambda: ready(win), timeout=3)
    assert row_of(win, "Slide1_Scan1").status == "new" and row_of(win, "Slide1_Scan1").checked


@pytest.mark.parametrize("change_folder", [False, True])
def test_an_old_search_cannot_finish_a_new_lookup(make_window, tmp_path, synth_tree, monkeypatch, gates,
                                                 change_folder):
    """U1b-V4: Stop's late ScanFinished during F5 or another folder's lookup belongs to the retired search."""
    first = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    second = make_slides(tmp_path / "ProjectD", ["Slide2_Scan1"], synth_tree)
    start = app.Discovery.start
    monkeypatch.setattr(app.Discovery, "start", lambda _self: None)
    win = make_window(input=first)
    until(win.root, lambda: win.state == "SCANNING", timeout=3)
    old_scan = win._scan_id
    monkeypatch.setattr(app.Discovery, "start", start)
    gate = gates()
    hang(monkeypatch, app, "_locate", gate)
    if change_folder:
        win.set_input(str(second))
    else:
        win.root.event_generate("<F5>")
    assert win.state == "CHECKING"
    win.ctx.bus.post(ScanFinished(old_scan, 0, 0, 0.1, True, (), (), 3))
    pump(win.root, 100)
    assert win.state == "CHECKING" and "Search stopped" not in status(win.input_field)
    gate.set()
    until(win.root, lambda: ready(win), timeout=3)
    assert win.source == win.input_field.get() == str(second if change_folder else first)
    assert row_of(win, "Slide2_Scan1" if change_folder else "Slide1_Scan1") is not None


@pytest.mark.parametrize("verdict", ["incomplete", "valid"])
def test_returning_to_a_save_folder_rechecks_a_discarded_verdict(make_window, tmp_path, synth_tree, monkeypatch,
                                                               gates, verdict):
    """U1a-V1: A/B/A around A's reply cannot leave 'checking' forever, including when A's reply was ledgered."""
    final = old_output(tmp_path / "outA" / "Slide1_Scan1.ome.tif", b"" if verdict == "incomplete" else None)
    gate, calls, verify = gates(), [], preflight.verify_output

    def delayed(path):
        calls.append(path)
        gate.wait(4)
        return verify(path)
    monkeypatch.setattr(preflight, "verify_output", delayed)
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=tiles, output=final.parent)
    until(win.root, lambda: ready(win) and calls, timeout=3)
    seen = []
    win.ctx.bus.subscribe(FsDone, lambda ev: seen.append(ev) if ev.tag and ev.tag[0] == "check" else None)
    win.set_output(str(tmp_path / "outB"))
    until(win.root, lambda: not win._index_pending, timeout=3)
    gate.set()
    until(win.root, lambda: seen, timeout=3)
    win.set_output(str(final.parent))
    until(win.root, lambda: not win._index_pending, timeout=3)
    until(win.root, lambda: row_of(win, "Slide1_Scan1").existing_verdict == verdict, timeout=2)
    assert calls == [str(final)] * (2 if verdict == "incomplete" else 1)
    if verdict == "incomplete":
        assert row_of(win, "Slide1_Scan1").checked
        assert "it will be redone" in status(win.output_field)
        win.set_input(str(tmp_path / "missing"))
        until(win.root, lambda: win.state == "EMPTY", timeout=3)
        assert "will be redone" not in status(win.output_field)


def test_a_missing_folder_retires_the_old_plans_header_recovery(make_window, tmp_path, synth_tree, monkeypatch,
                                                               gates):
    """U1a-V2: an old lane recovering after the plan was cleared must not read a retired row or raise in Tk."""
    share = tmp_path / "share"
    tiles = make_slides(share / "ProjectC", ["Slide1_Scan1"], synth_tree)
    gate, calls, header = gates(), [], app._header
    monkeypatch.setitem(fsservice.DEADLINE["header"], "network", 0.2)

    def delayed(*args):
        calls.append(args[0])
        gate.wait(4)
        return header(*args)
    monkeypatch.setattr(app, "_header", delayed)
    win = make_window(input=tiles, mounts=[share_mount(share)])
    errors = []
    monkeypatch.setattr(win.root, "report_callback_exception", lambda _typ, value, _tb: errors.append(value))
    until(win.root, lambda: ready(win) and row_of(win, "Slide1_Scan1").header_error, timeout=3)
    rid, lane = row_of(win, "Slide1_Scan1").row_id, win.ctx.fs.lane_key(str(tiles))
    win.set_input(str(tmp_path / "missing"))
    until(win.root, lambda: win.state == "EMPTY", timeout=3)
    win.ctx.bus.post(LaneHealthChanged(lane, "ok", time.time()))
    pump(win.root, 100)
    assert errors == [] and len(calls) == 1
    win._read_header(rid, force=True)                         # a row callback already queued when it disappeared
    assert len(calls) == 1
    gate.set()


def test_repeated_final_check_retry_starts_one_request_and_clears_the_old_error(make_window, tmp_path, synth_tree,
                                                                              monkeypatch, gates):
    """U1a-V3: mouse and keyboard retries cannot overlap or leave the old timeout's retry control active."""
    win = batch(make_window, tmp_path, synth_tree)
    first, retry, calls, check = gates(), gates(), [], app._final_check
    monkeypatch.setitem(fsservice.DEADLINE["write_probe"], "local", 0.2)

    def delayed(*args):
        calls.append(args)
        (first if len(calls) == 1 else retry).wait(4)
        return check(*args)
    monkeypatch.setattr(app, "_final_check", delayed)
    win.start(confirm=False)
    until(win.root, lambda: win._check_error is not None, timeout=3)
    first.set()
    monkeypatch.setitem(fsservice.DEADLINE["write_probe"], "local", 3)
    button(win.runbar.issue, "Try again").invoke()
    until(win.root, lambda: len(calls) == 2, timeout=3)
    generation = win._check_gen
    assert "Couldn't check" not in str(win.runbar.issue_text.cget("text"))
    assert button(win.runbar.issue, "Try again") is None
    win._ctrl_enter()
    win._issue_action("retry_check")
    win.start()
    pump(win.root, 50)
    assert len(calls) == 2 and win._check_gen == generation
    assert win.runbar.button.instate(["disabled"])
    retry.set()
    until(win.root, lambda: win.state == "FINISHED", timeout=3)


def test_quitting_during_stop_now_is_visible_and_can_be_cancelled(make_window, tmp_path, synth_tree, monkeypatch,
                                                                gates):
    """U1b-V7: an irreversible Stop now must still show a subsequently requested quit in all three places."""
    gate, entered = gates(), gates()

    def writing(_engine, phase, step):
        if phase == "write" and step == 1:
            entered.set()
            gate.wait(4)
    win = batch(make_window, tmp_path, synth_tree, on_step=writing)
    quits = []
    monkeypatch.setattr(win.root, "quit", lambda: quits.append(1))
    try:
        win.start(confirm=False)
        until(win.root, lambda: entered.is_set() and win._phase == "writing", timeout=3)
        win.request_stop("now")
        win.close()
        ask = dialog(win.root, dialogs.StopDialog)
        assert "Keep stitching" not in label_texts(ask)
        button(ask, "Wait, then quit").invoke()
        assert win._stop_mode == "now"
        assert "Quitting" in str(win.runbar.status.cget("text"))
        assert "Quitting" in str(win.footer.cget("text")) and "Quitting" in str(win.root.title())
        win._stop_chosen("cancel_quit", closing=True)
        assert "Quitting" not in str(win.runbar.status.cget("text"))
        assert "Quitting" not in str(win.footer.cget("text")) and "Quitting" not in str(win.root.title())
    finally:
        gate.set()
    until(win.root, lambda: win.state == "FINISHED", timeout=3)
    assert quits == []
