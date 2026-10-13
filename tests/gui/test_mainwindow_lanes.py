"""The main window when a place stops answering (UD1, UD4) and when the mount table changes (UD2). Every request
ends at its deadline, nothing waits forever, what failed is asked again once its lane answers, and the mount table
is the live one. A scratch folder declared a cifs share (share_mount) stands in for the network share, with
deadlines of a second; lane calls block on gates, which every test releases at its end."""
from __future__ import annotations

import dataclasses
import os
import time

from test_mainwindow import (ROOT_FS, _no_stale_tk_garbage, buttons, dialog, gates, hang, make_slides,  # noqa: F401
                             make_window, old_output, pump, ready, row_of, share, share_mount, status, until)
from tests.fakes import FakeEngine
from yfs.core import fsservice, preflight
from yfs.core.events import LaneHealthChanged
from yfs.core.fsservice import FsService
from yfs.core.model import FsError, RowOutcome
from yfs.core.names import PARTIAL_DIR
from yfs.core.places import MountTable
from yfs.core.runner import FINALIZE_TEXT
from yfs.core.plan import Plan, row_text
from yfs.core.settings import Settings
from yfs.ui import app, dialogs
from yfs.ui.app import MainWindow

SHORT = 1.0                                   # the deadline of the requests a test lets hang, in seconds


def disabled(field) -> bool:
    return field.combo.instate(["disabled"]) and field.browse.instate(["disabled"])


# ------------------------------------------------------------------------------------------------ UD1: deadlines
def test_a_final_check_that_doesnt_answer_brings_stitch_back_with_try_again(make_window, share, monkeypatch, gates):
    """CT-10: Stitch's final check on a share that stopped answering ends at its deadline: the Stitch button and
    the folders come back, and the run bar says what happened, with [Try again]."""
    win = make_window(input=share / "ProjectC", output=share / "stitched", mounts=[share_mount(share)])
    root = win.root
    until(root, lambda: ready(win), what="ready")
    monkeypatch.setitem(fsservice.DEADLINE["write_probe"], "network", SHORT)
    gate = gates()
    hang(monkeypatch, app, "_final_check", gate)
    win.runbar.button.invoke()
    assert str(win.runbar.button.cget("text")) == win.ctx.theme.text("Checking…")
    assert disabled(win.input_field) and disabled(win.output_field)        # UD7: no folder change meanwhile
    until(root, lambda: win._check_error, 5, what="the check's deadline")
    assert win.state == "READY" and win.runner is None
    assert win.runbar.button.instate(["!disabled"]) and str(win.runbar.button.cget("text")).startswith("Stitch")
    issue = win._issues()[0]
    assert (issue.severity, issue.text, issue.action) == ("warn", "Couldn't check share: it didn't answer.",
                                                          "retry_check")
    pump(root, 100)
    assert str(win.runbar.issue_text.cget("text")) == issue.text
    assert not disabled(win.input_field) and not disabled(win.output_field)
    again = next(b for b in win.runbar.issue.winfo_children()
                 if b.winfo_class() == "TButton" and str(b.cget("text")) == "Try again")
    gate.set()                                                             # the share answers again
    until(root, lambda: win.ctx.fs.health(str(share)) == "ok", 5, what="the share answers")
    again.invoke()                                                         # a real check this time
    sheet = until(root, lambda: dialog(root, dialogs.ConfirmStartDialog), 5, what="the sheet")
    assert win._check_error is None
    sheet._close("back")


def test_an_existing_file_that_cant_be_checked_is_unchecked_until_the_share_answers(make_window, share, monkeypatch,
                                                                                    gates):
    """CT-1 (d), MW-3: a check that doesn't answer proves nothing about the file: 'unchecked', never 'valid' nor
    ticked to be replaced; it is checked again when the share answers (UD1)."""
    out = old_output(share / "stitched" / "Slide1_Scan1.ome.tif").parent
    monkeypatch.setitem(fsservice.DEADLINE["check_output"], "network", SHORT)
    gate = gates()
    hang(monkeypatch, preflight, "verify_output", gate)
    win = make_window(input=share / "ProjectC", output=out, mounts=[share_mount(share)])
    root = win.root
    until(root, lambda: ready(win) and row_of(win, "Slide1_Scan1").existing_verdict == "unchecked", 10,
          what="the check's deadline")
    r = row_of(win, "Slide1_Scan1")
    assert (r.section, r.checked) == ("existing", False)
    assert row_text(r) == "Couldn't check the existing file (share didn't answer) — tick to replace it"
    assert "couldn't check 1 existing file" in status(win.output_field)
    assert not win.settings.ledger_has(str(out / "Slide1_Scan1.ome.tif"), r.existing.size, r.existing.mtime_ns)
    gate.set()                                                             # the share answers: checked once more
    until(root, lambda: row_of(win, "Slide1_Scan1").existing_verdict == "valid", 10, what="checked again")
    assert "1 of your 2 slides is already stitched here" in status(win.output_field)


def test_an_index_that_doesnt_answer_blocks_stitch_with_try_again_until_the_share_answers(make_window, share, tmp_path,
                                                                                         synth_tree, monkeypatch,
                                                                                         gates):
    """UD1, MW-1: the save folder's index ends at its deadline: step 2 says so with [Try again], the run bar leaves
    'Checking the save folder…' for a block that says why, and the index is asked again when the share answers."""
    tiles = make_slides(tmp_path / "local" / "ProjectC", ["Slide1_Scan1"], synth_tree)   # local tiles, saved to a share
    monkeypatch.setitem(fsservice.DEADLINE["index_output"], "network", SHORT)
    gate = gates()
    hang(monkeypatch, preflight, "list_output_dir", gate)
    win = make_window(input=tiles, output=share / "stitched", mounts=[share_mount(share)])
    root = win.root
    until(root, lambda: win.state == "READY" and win._index_error and not win._index_pending, 10,
          what="the index's deadline")
    assert status(win.output_field) == "share isn't answering. Nothing else is affected."
    assert buttons(win.output_field) == ["Try again"]
    pump(root, 100)
    issue = next(i for i in win._issues() if i.severity == "block")
    assert issue.code == "output_error" and issue.text.startswith("Can't save to stitched: ")
    assert win.runbar.button.instate(["disabled"]) and "Checking the save folder" not in \
        str(win.runbar.issue_text.cget("text"))
    gate.set()
    until(root, lambda: ready(win) and win.out_index is not None, 10, what="indexed again")
    assert status(win.output_field).startswith("This folder doesn't exist yet") and win._index_error is None


def test_a_tile_folder_that_doesnt_answer_says_so_and_is_searched_when_it_answers(make_window, share, monkeypatch,
                                                                                 gates):
    """UD1: the lookup of a tile folder ends at its deadline with [Try again]; when the share answers again the
    search starts by itself (nothing in the core retries)."""
    monkeypatch.setitem(fsservice.DEADLINE["locate"], "network", SHORT)
    gate = gates()
    hang(monkeypatch, app, "_locate", gate)
    win = make_window(mounts=[share_mount(share)])
    root = win.root
    win.set_input(str(share / "ProjectC"))
    until(root, lambda: win.state == "EMPTY" and "isn't answering" in status(win.input_field), 5,
          what="the lookup's deadline")
    assert status(win.input_field) == "share isn't answering. Nothing else is affected."
    assert buttons(win.input_field) == ["Try again"]
    gate.set()
    until(root, lambda: ready(win), 10, what="searched once the share answers")
    assert win.source == str(share / "ProjectC") and len(win.plan().rows("new")) == 2


def test_a_delete_that_fails_says_why_with_try_again(make_window, tmp_path, monkeypatch):
    """UD1: [Delete them] that fails shows an error where it was asked for, with [Try again]."""
    raw = old_output(tmp_path / "out" / "Old_Scan1.ome.tmp.raw", b"x" * 100)

    def refuse(*_a, **_kw):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(preflight, "delete_leftovers", refuse)
    win = make_window(output=raw.parent)
    root, passes = win.root, []
    monkeypatch.setattr(win.ctx.fs, "retry", passes.append)
    until(root, lambda: "leftover file" in status(win.output_field), what="the leftover")
    win.output_field._actions[0].invoke()
    until(root, lambda: dialog(root, dialogs._Dialog), what="confirmation")._close(True)
    until(root, lambda: status(win.output_field) == "Couldn't delete the leftover files: Permission denied.",
          what="the error")
    assert buttons(win.output_field) == ["Try again"] and raw.exists()
    assert passes == [str(raw.parent)]                                   # the click got the lane's retry pass


def waiting_for_finalize(make_window, tmp_path, synth_tree):
    """A finished batch whose slide waits for Finalize (its old file was in use): (window, row id, final path)."""
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(FakeEngine.configure(), tiles, tmp_path / "out")
    until(win.root, lambda: ready(win), what="ready")
    win.start(confirm=False)
    until(win.root, lambda: win.state == "FINISHED", 30, what="finished")
    rid, final = row_of(win, "Slide1_Scan1").row_id, tmp_path / "out" / "Slide1_Scan1.ome.tif"
    win.plan().finish(rid, RowOutcome("finalize", FINALIZE_TEXT, output_path=str(final),
                                      partial_path=str(tmp_path / "out" / PARTIAL_DIR / final.name)))
    return win, rid, final


def test_a_finalize_that_fails_says_why_with_try_again(make_window, tmp_path, synth_tree, monkeypatch):
    """UD1: Finalize that fails (the old file still in use) says why in step 2, with [Try again], which moves the
    file into place once it is free."""
    win, rid, final = waiting_for_finalize(make_window, tmp_path, synth_tree)
    root, tries = win.root, []

    def finalize(partial_path, final_path, expected, replace_existing):
        tries.append(final_path)
        if len(tries) == 1:
            raise PermissionError(13, "Permission denied")
        return (1234, 5678, 0), ""
    monkeypatch.setattr(app, "_finalize_file", finalize)
    win._issue_action("finalize")                                          # the banner's [Finalize]
    until(root, lambda: dialog(root, app._ConfirmDialog), timeout=2)._close(True)
    until(root, lambda: status(win.output_field) == "Couldn't finalize Slide1_Scan1: Permission denied.",
          what="the error")
    assert buttons(win.output_field) == ["Try again"] and win.plan().row(rid).status == "finalize"
    win.output_field._actions[0].invoke()
    until(root, lambda: dialog(root, app._ConfirmDialog), timeout=2)._close(True)
    until(root, lambda: win.plan().row(rid).status == "done", what="finalized")
    assert tries == [str(final)] * 2 and win.settings.ledger_has(str(final), 1234, 5678)
    until(root, lambda: not status(win.output_field).startswith("Couldn't"), what="indexed again")


def test_a_finalize_answer_after_another_save_folder_only_records_the_file(make_window, tmp_path, synth_tree,
                                                                         monkeypatch, gates):
    """DS-15, UD7: the plan rebuilt for another save folder has no run state; a Finalize that answers late (or
    fails) is about the last folder: the file is ledgered, and the new plan and step 2 are left alone."""
    win, rid, final = waiting_for_finalize(make_window, tmp_path, synth_tree)
    root, gate, entered = win.root, gates(), gates()

    def late(partial_path, final_path, expected, replace_existing):
        entered.set()
        gate.wait(4)
        return (1234, 5678, 0), ""
    monkeypatch.setattr(app, "_finalize_file", late)
    win._finalize([rid])
    until(root, lambda: dialog(root, app._ConfirmDialog), timeout=2)._close(True)
    until(root, entered.is_set, timeout=2)
    win.set_output(str(tmp_path / "outB"))
    until(root, lambda: ready(win), what="outB")
    gate.set()
    until(root, lambda: win.settings.ledger_has(str(final), 1234, 5678), what="the late answer")
    pump(root, 100)
    assert win.plan().row(rid).status == "new" and win.plan().row(rid).checked
    assert status(win.output_field).startswith("This folder doesn't exist yet")


# ------------------------------------------------------------------------------------------------ UD4: headers
def test_header_reads_take_turns_per_lane_and_a_stalled_one_is_read_when_the_share_answers(make_window, tmp_path,
                                                                                          synth_tree, monkeypatch,
                                                                                          gates):
    """FS-7: at most fs.concurrency() header reads in flight per lane, so one read that stalls leaves the others
    to finish; whatever failed because the share didn't answer is read again when it answers."""
    share = tmp_path / "share"
    names = [f"Slide{i}_Scan1" for i in range(1, 7)]
    tiles = make_slides(share / "ProjectD", names, synth_tree)
    monkeypatch.setitem(fsservice.DEADLINE["header"], "network", SHORT)
    gate, real = gates(), app._header

    def slow(path, *args):
        if os.path.basename(path).startswith("Slide1_Scan1_"):
            gate.wait(60)
        else:
            time.sleep(0.4)                                                # slow, so that some wait their turn
        return real(path, *args)
    monkeypatch.setattr(app, "_header", slow)
    win = make_window(input=tiles, mounts=[share_mount(share)])            # its search starts in 300 ms
    root = win.root
    asked, peak = [0], [0]                                                 # header requests in the lane at once
    real_call, real_done = win.ctx.fs.call, win._on_header

    def call(path, kind, fn, **kw):
        if kind == "header":
            asked[0] += 1
            peak[0] = max(peak[0], asked[0])
        return real_call(path, kind, fn, **kw)

    def done(ev):
        asked[0] -= 1
        real_done(ev)
    monkeypatch.setattr(win.ctx.fs, "call", call)
    monkeypatch.setattr(win, "_on_header", done)
    until(root, lambda: row_of(win, "Slide1_Scan1") is not None and row_of(win, "Slide1_Scan1").header_error, 10,
          what="the stalled read's deadline")
    r = row_of(win, "Slide1_Scan1")
    assert r.header_retryable and row_text(r).endswith("Right-click → Try again")
    assert peak[0] == 2                                                    # network: two at a time, none queued
    gate.set()
    until(root, lambda: all(row_of(win, n).header is not None for n in names), 15, what="every header")
    assert all(row_of(win, n).checked for n in names)


def test_a_lane_that_answers_again_gets_the_refused_requests_first_and_the_stalled_one_last(make_window, monkeypatch):
    """FS-7, UD1: once per recovery, never while the lane is still unresponsive; the requests it refused go before
    the one that ran out of time, which may stall again (it would fail them with it). Other errors aren't asked
    again: they are not about the lane."""
    win = make_window()
    asked, current = [], ["unknown", 0.0]
    monkeypatch.setattr(win.ctx.fs, "health_snapshot", lambda _lane: tuple(current))
    for key, code in (("stalled", "timeout"), ("a", "unresponsive"), ("b", "unresponsive"), ("locked", "no_access")):
        win._ask_again("/lane", FsError(code, "…"), key, lambda k=key: asked.append(k))
    for health in ("unresponsive", "slow", "ok"):
        current[:] = [health, current[1] + 1]
        win.ctx.bus.post(LaneHealthChanged("/lane", *current))
        pump(win.root, 150)
        assert asked == ([] if health == "unresponsive" else ["a", "b", "stalled"]), health


def test_a_request_that_runs_out_of_time_again_waits_for_try_again(make_window, monkeypatch):
    """UD1: a request that ran out of time is asked again once in a row. One that always takes longer than its
    deadline (a huge listing on a slow share) would otherwise stall its lane at every recovery, failing everything
    else there each time. A refused one is asked at every recovery, and an answer resets the count."""
    win = make_window()
    asked, current = [], ["unknown", 0.0]
    monkeypatch.setattr(win.ctx.fs, "health_snapshot", lambda _lane: tuple(current))

    def ends(key, code):
        win._ask_again("/lane", FsError(code, "…") if code else None, key, lambda: asked.append(key))

    def recovery():
        for health in ("unresponsive", "ok"):
            current[:] = [health, current[1] + 1]
            win.ctx.bus.post(LaneHealthChanged("/lane", *current))
            pump(win.root, 120)
    for code in ("timeout", "timeout", None, "timeout"):                  # None: it answered (a Try again)
        ends("index", code)
        ends("header", "unresponsive")
        recovery()
    assert asked == ["header", "index", "header", "header", "header", "index"]


def test_a_tile_the_engine_cant_use_offers_no_try_again(make_window, tmp_path, synth_tree):
    """UD4: a bad tile (read_tile_header's ValueError) is not an I/O error: reading it again won't help."""
    tiles = make_slides(tmp_path / "ProjectE", ["Good_Scan1", "Bad_Scan1"], synth_tree)
    first = sorted((tiles / "Bad_Scan1").iterdir())[0]
    first.unlink()
    first.write_bytes(b"not a tiff file" * 100)
    win = make_window(input=tiles)
    until(win.root, lambda: ready(win), what="ready")
    bad, good = row_of(win, "Bad_Scan1"), row_of(win, "Good_Scan1")
    assert bad.header is None and bad.header_error and not bad.header_retryable
    assert row_text(bad).startswith("Can't use the first tile (") and "Try again" not in row_text(bad)
    assert good.header is not None and good.header_retryable


# ------------------------------------------------------------------------------------------------ UD2: mount table
def test_the_mount_table_read_after_the_lookup_is_used_for_the_search(make_window, share, monkeypatch):
    """CT-7, CT-11, FS-2: a share mounted after the start (or by this very lookup, an automount) gets its own lane
    before the search starts, and the widgets' table follows (MountsChanged)."""
    win = make_window()                                                    # the startup table doesn't know it
    root = win.root
    assert win.ctx.fs.lane_class(str(share)) == "local"
    later = MountTable.from_mountinfo_text("\n".join((ROOT_FS, share_mount(share))))
    monkeypatch.setattr(MountTable, "load", classmethod(lambda cls, timeout_s=1.0: later))
    win.set_input(str(share / "ProjectC"))
    until(root, lambda: ready(win), what="ready")
    assert win.ctx.fs.mounts == later and win.ctx.fs.lane_class(str(share)) == "network"
    assert status(win.input_field).endswith("· network drive")
    until(root, lambda: win.ctx.mounts is win.ctx.fs.mounts, what="the widgets' table")


def test_a_new_mount_table_indexes_a_save_folder_whose_volume_changed(make_window, share):
    """UD2: an index made under the old table described the folder under the mount point; once the share is
    known there, the save folder is indexed again (and is a network drive)."""
    out = share / "stitched"
    out.mkdir()
    win = make_window(output=out)
    root = win.root
    until(root, lambda: win.out_index is not None, what="indexed")
    assert win.out_index.lane_class == "local" and status(win.output_field).startswith("Local disk")
    table = MountTable.from_mountinfo_text("\n".join((ROOT_FS, share_mount(share))))
    win.ctx.fs.set_mounts(table)
    until(root, lambda: win.out_index is not None and win.out_index.lane_class == "network", what="indexed again")
    assert status(win.output_field).startswith("Network drive") and win.ctx.mounts is table


def test_f5_and_the_minute_timer_reload_the_mount_table_and_refresh_the_plan(make_window, synth_tree, monkeypatch):
    """UD2, MW-8: F5 and the minute timer read the mount table again; the timer also refreshes the plan's
    time-based flags (Plan.refresh)."""
    win = make_window(input=synth_tree / "ProjectA")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    calls = []
    monkeypatch.setattr(win.ctx.fs, "reload_mounts", lambda: calls.append("mounts"))
    real = Plan.refresh
    monkeypatch.setattr(Plan, "refresh", lambda plan: calls.append("plan") or real(plan))
    win.rescan()
    assert calls == ["mounts"]
    until(root, lambda: ready(win), what="rescanned")
    win._minute()
    assert calls == ["mounts", "mounts", "plan"]


def test_startup_asks_the_live_lane_whether_to_search_the_last_folder(ui_ctx, cfg_dir, share, tmp_path, monkeypatch):
    """UD2: R23 asks fs.auto_enter, which follows the lane's live state (a drive that is still disconnected), not
    the startup snapshot."""
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings.load()
    s.set("paths.input", str(share / "ProjectC"))
    assert s.save_now()
    mounts = MountTable.from_mountinfo_text("\n".join((ROOT_FS, share_mount(share))))
    fs = FsService(ui_ctx.bus.post, mounts)
    fs.mark_disconnected(str(share), True)
    assert mounts.auto_enter(str(share / "ProjectC")) is None
    ctx = dataclasses.replace(ui_ctx, fs=fs, mounts=mounts, settings=Settings.load(), engine_cls=object)
    try:
        win = MainWindow(ctx)
        pump(win.root, 600)
        assert win.state == "EMPTY" and status(win.input_field).startswith("Press Rescan (F5)")
    finally:
        fs.shutdown()
        for w in ui_ctx.root.winfo_children():
            w.destroy()
