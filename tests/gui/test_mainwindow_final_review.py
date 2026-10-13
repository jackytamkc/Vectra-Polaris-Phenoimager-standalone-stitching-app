"""Final review: folder identity, header consent and row-bound stop decisions."""
from __future__ import annotations

import dataclasses
import os
import threading
import time
from types import SimpleNamespace

import pytest

from test_mainwindow import (_no_stale_tk_garbage, dialog, gates, make_slides, make_window, old_output,  # noqa: F401
                             pump, ready, row_of, share_mount, status, until)
from yfs.core import fsservice, preflight
from yfs.core.events import RowFinished, RowStarted
from yfs.core.model import RowOutcome
from yfs.core.names import PARTIAL_DIR
from yfs.core.settings import Settings
from yfs.ui import app, dialogs
from yfs.ui.runpanel import ProgressModel


@pytest.mark.parametrize("ingress", ["browse", "restored", "recent", "typed"])
def test_literal_paths_and_once_expanded_edits_agree_with_the_plan(make_window, synth_tree, tmp_path, monkeypatch,
                                                                  ingress):
    monkeypatch.setenv("YFS_TEST_TOKEN", "CONTROL")
    monkeypatch.setenv("YFS_TEST_NESTED", "$YFS_TEST_TOKEN")
    literal = make_slides(tmp_path / "Study_$YFS_TEST_TOKEN", ["Slide1_Scan1"], synth_tree)
    other = make_slides(tmp_path / "Study_CONTROL", ["Slide2_Scan1"], synth_tree)
    out = tmp_path / "Save_$YFS_TEST_TOKEN"
    out.mkdir()
    (tmp_path / "Save_CONTROL").mkdir()
    settings = Settings.load()
    if ingress == "restored":
        settings.set("paths.input", str(literal))
        settings.set("paths.output", str(out))
    win = make_window(settings=settings)
    if ingress == "browse":
        monkeypatch.setattr(app.FolderBrowser, "open", lambda *_a, **kw: kw["on_done"](
            str(literal if kw["mode"] == "input" else out)))
        win.browse_input()
        win.browse_output()
    elif ingress == "recent":
        for field, path in ((win.input_field, literal), (win.output_field, out)):
            field.combo.configure(values=(str(path),))
            field.combo.current(0)
            field.combo.event_generate("<<ComboboxSelected>>")
    elif ingress == "typed":
        for field, path in ((win.input_field, literal), (win.output_field, out)):
            field._var.set(str(path).replace("$YFS_TEST_TOKEN", "$YFS_TEST_NESTED"))
            field._commit(force=True)
    until(win.root, lambda: ready(win), timeout=3)
    assert win.source == win.input_field.get() == str(literal)
    assert win.out_dir == win.output_field.get() == str(out)
    assert row_of(win, "Slide1_Scan1") and row_of(win, "Slide2_Scan1") is None
    for field in (win.input_field, win.output_field):
        field._commit(force=True)                         # unchanged Enter and FocusOut must stay literal
        field._commit()
        assert win._field_path(field) == field.get()
    win.rescan()
    until(win.root, lambda: ready(win), timeout=3)
    win.start(confirm=False)
    until(win.root, lambda: win.state == "FINISHED", timeout=3)
    assert (out / "Slide1_Scan1.ome.tif").exists()
    assert not (tmp_path / "Save_CONTROL" / "Slide1_Scan1.ome.tif").exists()
    assert other.exists()
    opened = []
    monkeypatch.setattr(app.FolderBrowser, "open", lambda *_a, **kw: opened.append(kw["initial"]))
    win.browse_input()
    win.browse_output()
    assert opened == [(str(literal.parent), literal.name), (str(out.parent), out.name)]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("field", ["input", "output"])
def test_alias_target_hangs_only_its_own_lane(make_window, tmp_path, monkeypatch, gates, restored, ancestor, field):
    remote = tmp_path / "labshare"
    remote.mkdir()
    (remote / "ProjectC").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(remote if ancestor else remote / "ProjectC", target_is_directory=True)
    path = alias / "ProjectC" if ancestor else alias
    gate, entered = gates(), threading.Event()
    owner, operation = (app, "_locate") if field == "input" else (preflight, "list_output_dir")
    real = getattr(owner, operation)

    def slow(found, *args, **kwargs):
        if str(remote) in found:
            entered.set()
            gate.wait(4)
        return real(found, *args, **kwargs)
    monkeypatch.setattr(owner, operation, slow)
    monkeypatch.setitem(fsservice.DEADLINE["locate" if field == "input" else "index_output"], "network", .1)
    settings = Settings.load()
    if restored:
        settings.set("paths." + field, str(path))
    win = make_window(settings=settings, mounts=[share_mount(remote)])
    if not restored:
        (win.set_input if field == "input" else win.set_output)(str(path))
    until(win.root, lambda: entered.is_set(), timeout=3)
    target = win.input_field if field == "input" else win.output_field
    until(win.root, lambda: "isn't answering" in status(target), timeout=2)
    assert win.ctx.fs.health(str(tmp_path)) != "unresponsive"
    assert win.ctx.fs.health(str(remote)) == "unresponsive"
    answers = []
    win.ctx.bus.subscribe(app.FsDone, lambda ev: answers.append(ev) if ev.owner == "local-proof" else None)
    win.ctx.fs.call(str(tmp_path), "locate", lambda: os.listdir(tmp_path), ticket=0, owner="local-proof")
    until(win.root, lambda: answers, timeout=1)
    assert answers[0].error is None


@pytest.mark.parametrize("result", ["duplicate", "failed", "io", "valid"])
def test_stitch_waits_for_selected_headers_before_any_choice(make_window, tmp_path, synth_tree, monkeypatch,
                                                            gates, result):
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    gate, real = gates(), app._header

    def delayed(*args):
        gate.wait(4)
        if result == "io":
            raise OSError("The tile could not be read")
        header, ram, bad = real(*args)
        if result == "failed":
            return None, ram, "the tile is cut off"
        if result == "duplicate":
            header = dataclasses.replace(header, channel_problem="duplicate channel names")
        return header, ram, bad
    monkeypatch.setattr(app, "_header", delayed)
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: win.state == "READY" and not win._index_pending, timeout=3)
    launches = []
    monkeypatch.setattr(win, "_launch", lambda: launches.append(True))
    win.start(confirm=False)
    until(win.root, lambda: win._final_result is not None, timeout=2)
    assert not launches and dialog(win.root, dialogs.ConfirmStartDialog) is None
    assert not row_of(win, "Slide1_Scan1").generic_accepted
    gate.set()
    if result == "duplicate":
        sheet = until(win.root, lambda: dialog(win.root, dialogs.ConfirmStartDialog), timeout=2)
        assert not launches and not row_of(win, "Slide1_Scan1").generic_accepted
        sheet._start()
        assert launches and row_of(win, "Slide1_Scan1").generic_accepted
    elif result in ("failed", "io"):
        until(win.root, lambda: win._final is None, timeout=2)
        assert not launches and "Couldn't read" in win._check_error
        assert not row_of(win, "Slide1_Scan1").generic_accepted
    else:
        until(win.root, lambda: launches, timeout=2)
        assert not row_of(win, "Slide1_Scan1").generic_accepted


def test_new_channel_problem_while_sheet_is_open_requires_a_new_choice(make_window, tmp_path, synth_tree,
                                                                      monkeypatch):
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: ready(win), timeout=3)
    launches = []
    monkeypatch.setattr(win, "_launch", lambda: launches.append(True))
    win.start()
    sheet = until(win.root, lambda: dialog(win.root, dialogs.ConfirmStartDialog), timeout=2)
    row = row_of(win, "Slide1_Scan1")
    win.plan().apply_header(row.row_id, dataclasses.replace(row.header, channel_problem="duplicate names"), None,
                            win._ram)
    sheet._start()
    assert not launches and not row.generic_accepted
    refreshed = until(win.root, lambda: dialog(win.root, dialogs.ConfirmStartDialog), timeout=2)
    assert refreshed is not sheet and refreshed._unresolved
    refreshed._start()
    assert launches and row.generic_accepted


def test_complete_partial_needs_no_pending_tile_header(make_window, tmp_path, synth_tree, monkeypatch, gates):
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")
    preflight.mark_complete(str(kept))
    gate, real = gates(), app._header
    monkeypatch.setattr(app, "_header", lambda *args: (gate.wait(4), real(*args))[1])
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: win.state == "READY" and not win._index_pending, timeout=3)
    launches = []
    monkeypatch.setattr(win, "_launch", lambda: launches.append(True))
    win.start(confirm=False)
    until(win.root, lambda: launches, timeout=2)
    assert row_of(win, "Slide1_Scan1").header is None


def test_partial_found_at_final_check_does_not_need_a_failed_tile_header(make_window, tmp_path, synth_tree,
                                                                       monkeypatch, gates):
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    header_gate, final_gate, real = gates(), gates(), app._final_check
    monkeypatch.setattr(app, "_header", lambda *_a: (header_gate.wait(4), (None, 1000000, "cut off"))[1])
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: win.state == "READY" and not win._index_pending, timeout=3)
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")
    preflight.mark_complete(str(kept))                    # arrived after the output index was made
    monkeypatch.setattr(app, "_final_check", lambda *a: (final_gate.wait(4), real(*a))[1])
    launches = []
    monkeypatch.setattr(win, "_launch", lambda: launches.append(True))
    win.start(confirm=False)
    header_gate.set()
    until(win.root, lambda: row_of(win, "Slide1_Scan1").header_error, timeout=2)
    final_gate.set()
    until(win.root, lambda: launches, timeout=2)
    assert row_of(win, "Slide1_Scan1").ready is not None
    assert not row_of(win, "Slide1_Scan1").generic_accepted


def all_text(widget):
    return " ".join([str(widget.cget("text"))] if "text" in widget.keys() else []) + " " + \
        " ".join(all_text(child) for child in widget.winfo_children())


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("transition", ["event", "accept", "atomic"])
def test_stop_and_quit_never_apply_an_old_slide_choice(make_window, tmp_path, synth_tree, monkeypatch,
                                                     closing, transition):
    tiles = make_slides(tmp_path / "ProjectC", ["SlideA_Scan1", "SlideB_Scan1"], synth_tree)
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: ready(win), timeout=3)
    a, b = (row_of(win, name).row_id for name in ("SlideA_Scan1", "SlideB_Scan1"))
    applied = []
    runner = SimpleNamespace(current_row=a, stop_now=lambda: applied.append("now"),
                             stop_after_current=lambda: applied.append("after_current"), join=lambda *_args: None)
    monkeypatch.setattr(win, "runner", runner)
    win.model = ProgressModel({a: None, b: None})
    win._set_state(app.RUNNING)
    win._ask_stop(closing)
    ask = win._stop_dialog
    old_choice = ask._done_cb
    assert "SlideA_Scan1" in all_text(ask)
    if transition == "event":
        runner.current_row = b
        win._on_row_finished(RowFinished(win._run_id, a, RowOutcome("done", "Done")))
        win._on_row_started(RowStarted(win._run_id, b, time.time()))
        assert not ask.winfo_exists()
        assert "SlideB_Scan1" in all_text(win._stop_dialog)
        assert "Finished slides (1)" in all_text(win._stop_dialog)
        old_choice("now")
    elif transition == "accept":
        runner.current_row = b                              # event not delivered yet
        ask._close("now")
    else:
        def race(mode, *, expected_row):
            assert expected_row == a
            runner.current_row = b                          # core refuses the raced target under its row lock
            return False
        runner.request_stop = race
        ask._close("now")
    assert not applied and win.state == app.RUNNING
    assert not win._quit_after
    assert "SlideB_Scan1" in all_text(win._stop_dialog)
    win._dismiss_stop()
    win.runner = None
