"""Recovery confirmation and independent output/temp filesystem requests."""
from __future__ import annotations

import os
import threading

import pytest

from test_mainwindow import (_no_stale_tk_garbage, dialog, gates, make_slides, make_window, old_output,  # noqa: F401
                             pump, ready, row_of, sha256, share_mount, status, until)
from yfs.core import fsservice, preflight
from yfs.core.names import COMPLETE_SUFFIX, PARTIAL_DIR, temp_subdir
from yfs.ui import app


def recovery(make_window, tmp_path, synth_tree):
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    kept = old_output(tmp_path / "out" / PARTIAL_DIR / "Slide1_Scan1.ome.tif")
    preflight.mark_complete(str(kept), gaps=2)
    win = make_window(input=tiles, output=tmp_path / "out")
    until(win.root, lambda: ready(win), timeout=3)
    return win, kept, tmp_path / "out" / kept.name


def another_image(final):
    old_output(final)
    data = final.read_bytes()
    final.write_bytes(data[:-1] + b"\xff")                 # a different, still valid TIFF pixel


def from_menu(win):
    row = row_of(win, "Slide1_Scan1")
    count = win.table.fill_menu(row.row_id)
    menu = win.table.menu
    menu.invoke(next(i for i in range(count) if menu.entrycget(i, "label") == "Finalize"))


@pytest.mark.parametrize("planned", [False, True])
def test_finalize_replacement_requires_a_destination_bound_decision(make_window, tmp_path, synth_tree, planned):
    final = tmp_path / "out" / "Slide1_Scan1.ome.tif"
    if planned:
        another_image(final)
    win, kept, final = recovery(make_window, tmp_path, synth_tree)
    if not planned:
        another_image(final)                              # an image arrived since planning
    before, partial = sha256(final), sha256(kept)
    from_menu(win)
    ask = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)
    from test_mainwindow import label_texts
    assert str(final) in label_texts(ask)
    ask._cancel()
    assert sha256(final) == before and sha256(kept) == partial
    assert os.path.exists(str(kept) + COMPLETE_SUFFIX)
    from_menu(win)
    until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)._close(True)
    until(win.root, lambda: not kept.exists(), timeout=2)
    assert sha256(final) == partial
    until(win.root, lambda: win.settings.ledger_get(str(final), final.stat().st_size, final.stat().st_mtime_ns) == 2,
          timeout=2)


def test_changed_destination_does_not_inherit_an_open_replacement_question(make_window, tmp_path, synth_tree):
    win, kept, final = recovery(make_window, tmp_path, synth_tree)
    another_image(final)
    partial = sha256(kept)
    from_menu(win)
    ask = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)
    final.write_bytes(final.read_bytes()[:-1] + b"\x7f")
    changed = sha256(final)
    ask._close(True)
    until(win.root, lambda: "destination changed" in status(win.output_field), timeout=2)
    assert sha256(final) == changed and sha256(kept) == partial
    assert os.path.exists(str(kept) + COMPLETE_SUFFIX)
    win.output_field._actions[0].invoke()
    fresh = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)
    assert fresh is not ask
    fresh._close(True)
    until(win.root, lambda: not kept.exists(), timeout=2)
    assert sha256(final) == partial


def test_arrival_during_finalize_validation_is_not_clobbered(make_window, tmp_path, synth_tree, monkeypatch, gates):
    win, kept, final = recovery(make_window, tmp_path, synth_tree)
    gate, entered, real = gates(), threading.Event(), preflight.verify_output
    partial = sha256(kept)

    def delayed(path):
        if path == str(kept):
            entered.set()
            gate.wait(4)
        return real(path)
    monkeypatch.setattr(preflight, "verify_output", delayed)
    from_menu(win)
    until(win.root, lambda: entered.is_set(), timeout=2)
    another_image(final)
    arrived = sha256(final)
    gate.set()
    until(win.root, lambda: "destination changed" in status(win.output_field), timeout=2)
    assert sha256(final) == arrived and sha256(kept) == partial
    assert os.path.exists(str(kept) + COMPLETE_SUFFIX)


@pytest.mark.parametrize("during", ["validation", "retry"])
def test_approved_destination_changes_are_refused_inside_finalize(make_window, tmp_path, synth_tree, monkeypatch,
                                                                 gates, during):
    win, kept, final = recovery(make_window, tmp_path, synth_tree)
    another_image(final)
    partial = sha256(kept)
    gate, entered = gates(), threading.Event()
    if during == "validation":
        real = preflight.verify_output

        def delayed(path):
            if path == str(kept):
                entered.set()
                gate.wait(4)
            return real(path)
        monkeypatch.setattr(preflight, "verify_output", delayed)
    else:
        real = preflight.os.replace
        attempts = []

        def delayed(src, dst):
            if str(dst) == str(final):
                attempts.append(dst)
                entered.set()
                gate.wait(4)
                raise PermissionError(13, "The file is in use")
            return real(src, dst)
        monkeypatch.setattr(preflight.os, "replace", delayed)
        monkeypatch.setattr(preflight, "RETRY_DELAY_S", .01)
    from_menu(win)
    until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)._close(True)
    until(win.root, lambda: entered.is_set(), timeout=2)
    final.write_bytes(final.read_bytes()[:-1] + b"\x7f")
    changed = sha256(final)
    gate.set()
    until(win.root, lambda: "destination changed" in status(win.output_field), timeout=2)
    assert sha256(final) == changed and sha256(kept) == partial
    assert os.path.exists(str(kept) + COMPLETE_SUFFIX)
    if during == "retry":
        assert len(attempts) == 1


@pytest.mark.parametrize("operation", ["scratch", "usage"])
def test_hung_temp_folder_leaves_local_output_lane_usable(make_window, tmp_path, synth_tree, monkeypatch, gates,
                                                         operation):
    remote = tmp_path / "labshare"
    remote.mkdir()
    out = tmp_path / "out"
    scratch = temp_subdir(str(remote), str(out))
    gate, entered = gates(), threading.Event()
    if operation == "scratch":
        real = os.scandir

        def blocked(path):
            if str(path) == scratch:
                entered.set()
                gate.wait(4)
            return real(path)
        monkeypatch.setattr(os, "scandir", blocked)
    else:
        real = preflight.shutil.disk_usage

        def blocked(path):
            if str(path) == str(remote):
                entered.set()
                gate.wait(4)
            return real(path)
        monkeypatch.setattr(preflight.shutil, "disk_usage", blocked)
    monkeypatch.setitem(fsservice.DEADLINE["usage"], "network", .1)
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=tiles, output=out, mounts=[share_mount(remote)])
    until(win.root, lambda: ready(win), timeout=3)
    win.settings.set("run.temp_dir", str(remote))
    if operation == "usage":
        win.start(confirm=False)
    else:
        win._request_index()
    until(win.root, lambda: entered.is_set(), timeout=2)
    until(win.root, lambda: win._temp_error, timeout=2)
    assert "temporary folder labshare" in win._temp_error
    assert win.ctx.fs.health(str(out)) != "unresponsive"
    assert win.ctx.fs.health(str(remote)) == "unresponsive"
    assert win.runner is None and win._index_error is None
    answers = []
    win.ctx.bus.subscribe(app.FsDone, lambda ev: answers.append(ev) if ev.owner == "local-proof" else None)
    win.ctx.fs.call(str(out), "usage", lambda: 123, ticket=0, owner="local-proof")
    until(win.root, lambda: answers, timeout=1)
    assert answers[0].error is None
    gate.set()
    until(win.root, lambda: not win._temp_error and not win._index_pending, timeout=2)
    assert win._temp_free is not None and win.ctx.fs.health(str(out)) != "unresponsive"


def test_leftover_confirmation_is_retired_when_the_folders_change(make_window, tmp_path):
    out = tmp_path / "out"
    raw = old_output(out / "Old.ome.tmp.raw", b"scratch")
    win = make_window(output=out)
    until(win.root, lambda: win.out_index is not None and win.out_index.leftovers, timeout=3)
    win._delete_leftovers()
    ask = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=1)
    win.set_output(str(tmp_path / "Other"))
    ask._close(True)
    assert raw.exists()
    assert "folders changed" in status(win.output_field)


def test_cross_drive_leftovers_are_deleted_after_output_ownership_is_reserved(make_window, tmp_path, monkeypatch):
    remote, out = tmp_path / "labshare", tmp_path / "out"
    remote.mkdir()
    out.mkdir()
    raw = old_output(type(out)(temp_subdir(str(remote), str(out))) / "Old.ome.tmp.raw", b"scratch")
    win = make_window(output=out, mounts=[share_mount(remote)])
    win.settings.set("run.temp_dir", str(remote))
    win._settings_done({"run.temp_dir"})
    until(win.root, lambda: win.out_index is not None and win.out_index.leftovers, timeout=3)
    calls = []
    monkeypatch.setattr(preflight, "delete_leftovers", lambda *args, **kwargs: calls.append(args))
    win._delete_leftovers()
    until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=1)._close(True)
    until(win.root, lambda: not raw.exists(), timeout=2)
    assert not calls                       # the same-lane deletion helper never touches the remote folder
