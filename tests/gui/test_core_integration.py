from __future__ import annotations

import os
import threading
import time

import pytest

from test_mainwindow import (_no_stale_tk_garbage, dialog, gates, label_texts, make_slides, make_window,  # noqa: F401
                             old_output, pump, ready, row_of, until)
from yfs.core import preflight
from yfs.core.names import PARTIAL_DIR
from yfs.ui import app, browser, dialogs


def test_details_follows_inline_edits_rebuild_and_plan_retirement(make_window, tmp_path, synth_tree):
    source = make_slides(tmp_path / 'ProjectC', ['Slide1_Scan1'], synth_tree)
    win = make_window(input=source, output=tmp_path / 'out')
    until(win.root, lambda: ready(win), timeout=3)
    row = row_of(win, 'Slide1_Scan1')
    win.details(row.row_id)
    details = dialog(win.root, dialogs.SlideDetailsDialog)
    win.table.rename(row.row_id)
    editor = win.table._edit['entry']
    editor.delete(0, 'end')
    editor.insert(0, 'RenamedSlide')
    editor.focus_force()
    pump(win.root, 50)
    editor.event_generate('<Return>')
    until(win.root, lambda: 'Output: RenamedSlide.ome.tif' in label_texts(details), timeout=2)
    old = win.plan()
    win._set_state(app.FINISHED)
    win.set_output(str(tmp_path / 'OtherSave'))
    until(win.root, lambda: win.plan() is not old and not win._index_pending, timeout=2)
    assert details.plan is win.plan()
    win.rescan()
    until(win.root, lambda: not details.winfo_exists(), timeout=2)
    until(win.root, lambda: ready(win), timeout=3)
    win.details(row.row_id)
    details = dialog(win.root, dialogs.SlideDetailsDialog)
    win.set_input(str(tmp_path / 'Missing'))
    until(win.root, lambda: win.plan() is None, timeout=2)
    assert not details.winfo_exists()


def test_case_matched_finalize_uses_the_displayed_and_approved_filename(make_window, tmp_path, synth_tree):
    source = make_slides(tmp_path / 'ProjectC', ['CaseOnly'], synth_tree)
    final = old_output(tmp_path / 'out' / 'caseonly.ome.tif', b'existing image')
    kept = old_output(tmp_path / 'out' / PARTIAL_DIR / 'CaseOnly.ome.tif')
    preflight.mark_complete(str(kept))
    image = kept.read_bytes()
    win = make_window(input=source, output=final.parent)
    until(win.root, lambda: ready(win), timeout=3)
    row = row_of(win, 'CaseOnly')
    until(win.root, lambda: row.replaces, timeout=2)
    assert win.table.tree.set(win.table._iid[row.row_id], 'output') == final.name
    win.details(row.row_id)
    details = dialog(win.root, dialogs.SlideDetailsDialog)
    assert 'Output: ' + final.name in label_texts(details)
    details._cancel()
    win._finalize([row.row_id])
    ask = until(win.root, lambda: dialog(win.root, app._ConfirmDialog), timeout=2)
    assert str(final) in label_texts(ask)
    ask._close(True)
    until(win.root, lambda: not kept.exists(), timeout=2)
    assert final.read_bytes() == image and not (final.parent / 'CaseOnly.ome.tif').exists()


@pytest.mark.parametrize('reverse', [False, True])
def test_browser_and_main_lookup_do_not_share_incompatible_results(make_window, tmp_path, monkeypatch, gates, reverse):
    project = tmp_path / 'ProjectC'
    project.mkdir()
    win = make_window()
    chooser = browser.FolderBrowser.open(win.root, win.ctx, mode='input', initial=(str(tmp_path), None),
                                         on_done=lambda _path: None)
    until(win.root, lambda: chooser._listing is not None, timeout=2)
    release, main_entered, browser_entered = gates(), threading.Event(), threading.Event()

    def delayed(fn, entered):
        def call(path):
            if path == str(project):
                entered.set()
                assert release.wait(3)
            return fn(path)
        return call

    monkeypatch.setattr(app, '_locate', delayed(app._locate, main_entered))
    monkeypatch.setattr(browser, '_locate', delayed(browser._locate, browser_entered))
    actions = [(lambda: win.set_input(str(project)), main_entered),
               (lambda: chooser._commit_path(str(project)), browser_entered)]
    for action, entered in reversed(actions) if reverse else actions:
        action()
        until(win.root, entered.is_set, timeout=2)
    release.set()
    until(win.root, lambda: win.state == 'READY' and chooser.current() == str(project) and
          chooser._listing is not None and not chooser._loading, timeout=2)
    assert win.source == str(project)
    chooser.cancel()


@pytest.mark.parametrize('swap', ['directory', 'file'])
def test_delete_confirmation_preserves_changed_objects(make_window, tmp_path, swap):
    out = tmp_path / 'out'
    raw = old_output(out / 'Slide1.ome.tmp.raw', b'approved scratch')
    win = make_window(output=out)
    until(win.root, lambda: win.out_index is not None and win.out_index.leftovers, timeout=2)
    win._delete_leftovers()
    ask = dialog(win.root, app._ConfirmDialog)
    original = tmp_path / 'Original'
    if swap == 'directory':
        out.rename(original)
        out.mkdir()
        original = original / raw.name
    else:
        raw.rename(original)
    old_output(raw, b'unapproved replacement scratch')
    ask._close(True)
    until(win.root, lambda: win.out_index is not None and
          win.out_index.leftovers[0][1].size == raw.stat().st_size, timeout=2)
    assert raw.read_bytes() == b'unapproved replacement scratch'
    assert original.read_bytes() == b'approved scratch'


def test_aged_damage_in_a_later_channel_is_never_ledgered(make_window, tmp_path, synth_tree):
    import numpy as np
    import tifffile
    source = make_slides(tmp_path / 'ProjectC', ['Slide1_Scan1'], synth_tree)
    final = tmp_path / 'out' / 'Slide1_Scan1.ome.tif'
    final.parent.mkdir()
    tifffile.imwrite(final, np.zeros((2, 64, 64), np.uint16), photometric='minisblack', ome=True,
                     metadata={'axes': 'CYX'})
    with tifffile.TiffFile(final) as tif:
        cut = tif.pages[1].dataoffsets[0] + 100
    with final.open('r+b') as stream:
        stream.truncate(cut)
    os.utime(final, (time.time() - 3600,) * 2)
    win = make_window(input=source, output=final.parent)
    until(win.root, lambda: ready(win), timeout=3)
    row = row_of(win, 'Slide1_Scan1')
    until(win.root, lambda: row.existing_verdict == 'incomplete', timeout=2)
    assert row.checked
    assert win.settings.ledger_get(str(final), final.stat().st_size, final.stat().st_mtime_ns) is None


@pytest.mark.parametrize('consumer', ['main', 'browser'])
def test_deferred_timeout_reconciles_live_health_once_and_ignores_stale_recovery(make_window, tmp_path,
                                                                               monkeypatch, consumer):
    from yfs.core.events import FsDone, LaneHealthChanged
    from yfs.core.model import FsError
    win = make_window()
    folder = tmp_path / 'ProjectC'
    folder.mkdir()
    fs, asked = win.ctx.fs, []
    lane = fs.lane_key(str(folder))
    state = ['ok', 1.0]
    monkeypatch.setattr(fs, 'health_snapshot', lambda _path: tuple(state))
    timeout = FsError('timeout', 'The folder did not answer.')
    if consumer == 'main':
        add = lambda: win._ask_again(lane, FsError('unresponsive', timeout.message), 'request', lambda: asked.append(1))
        recover = win._on_lane_health
    else:
        chooser = browser.FolderBrowser.open(win.root, win.ctx, mode='input', initial=(str(tmp_path), None),
                                             on_done=lambda _path: None)
        until(win.root, lambda: chooser._listing is not None, timeout=2)
        row = until(win.root, lambda: next((r for r in chooser._rows.values() if r.name == folder.name), None), timeout=2)
        until(win.root, lambda: row.summary is not None and not row.probing, timeout=2)
        monkeypatch.setattr(chooser, '_probe', lambda *_a: asked.append(1))
        recover = chooser._on_health
        def add():
            row.stalled, row.error = str(folder), timeout
            chooser._calls[-1] = (None, lambda _e: None)
            chooser._on_fs_done(FsDone(-1, 'probe', str(folder), 0, 'browser', None, None, timeout, 0))
    add()                                                   # recovery event arrived before this deferred failure
    until(win.root, lambda: asked == [1], timeout=1)
    add()
    pump(win.root, 40)
    assert asked == [1]                                      # no loop inside one recovery generation
    state[:] = ['unresponsive', 2.0]
    recover(LaneHealthChanged(lane, 'ok', 1.0))
    pump(win.root, 40)
    assert asked == [1]
    state[:] = ['ok', 3.0]
    recover(LaneHealthChanged(lane, *state))
    until(win.root, lambda: asked == [1, 1], timeout=1)
    if consumer == 'browser':
        chooser.cancel()


def test_timeout_publication_and_recovery_produce_one_gui_retry(make_window, tmp_path, monkeypatch, gates):
    from test_mainwindow import share_mount
    from yfs.core import fsservice
    from yfs.core.events import FsDone, FsSlow
    remote = tmp_path / 'labshare'
    remote.mkdir()
    win = make_window(mounts=[share_mount(remote)])
    fs, release, interleaved = win.ctx.fs, gates(), threading.Event()
    post, index, events, calls = fs._post, preflight.list_output_dir, [], []

    def ordered(event):
        if isinstance(event, FsSlow) and event.kind == 'index_output' and event.past_deadline:
            release.set()
            end = time.monotonic() + 2
            while fs.health(str(remote)) != 'ok' and time.monotonic() < end:
                time.sleep(.005)
            assert fs.health(str(remote)) == 'ok'
            interleaved.set()
        events.append(event)
        post(event)

    def delayed(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            assert release.wait(3)
        return index(*args, **kwargs)

    monkeypatch.setattr(fs, '_post', ordered)
    monkeypatch.setattr(preflight, 'list_output_dir', delayed)
    monkeypatch.setitem(fsservice.DEADLINE['index_output'], 'network', .05)
    win.set_output(str(remote))
    until(win.root, lambda: interleaved.is_set() and len(calls) == 2 and
          win.out_index is not None and not win._index_pending, timeout=3)
    assert len([e for e in events if isinstance(e, FsDone) and e.kind == 'index_output' and
                e.error and e.error.code == 'timeout']) == 1
    assert win._index_error is None


def test_unavailable_xdg_destination_does_not_hide_places_or_picker(make_window, tmp_path, monkeypatch):
    win = make_window()
    cfg, documents = tmp_path / 'cfg', tmp_path / 'labshare' / 'Documents'
    cfg.mkdir()
    (cfg / 'user-dirs.dirs').write_text(f'XDG_DOCUMENTS_DIR="{documents}"\n')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(cfg))
    isdir, touched = os.path.isdir, []

    def guard(path):
        if str(path) == str(documents):
            touched.append(path)
            raise AssertionError('configured places must not be statted')
        return isdir(path)

    monkeypatch.setattr(os.path, 'isdir', guard)
    monkeypatch.setattr(browser.osutil, 'system_picker_available', lambda: True)
    chooser = browser.FolderBrowser.open(win.root, win.ctx, mode='input', initial=(str(tmp_path), None),
                                         on_done=lambda _path: None)
    until(win.root, lambda: chooser._places and chooser._picker_ok and chooser._picker.winfo_manager(), timeout=2)
    assert {'Home', 'Documents'} <= {p.label for p in chooser._places}
    assert not touched
    chooser.cancel()


def test_connected_gvfs_share_opens_from_a_pasted_url(make_window, tmp_path, monkeypatch):
    from yfs.core.model import CLOUD
    from yfs.core.places import Mount, MountTable
    win = make_window()
    gvfs = tmp_path / 'ConnectedShares'
    backend = 'smb-share:server=nas.example.org,share=labshare'
    target = gvfs / backend / 'ProjectC'
    target.mkdir(parents=True)
    (target / 'Slide1').mkdir()
    table = MountTable(win.ctx.fs.mounts.mounts() +
                       (Mount(str(gvfs), 'fuse.gvfsd-fuse', 'gvfsd-fuse', CLOUD, 'Connected shares'),),
                       gvfs=[(str(gvfs), backend)])
    monkeypatch.setattr(MountTable, 'load', classmethod(lambda cls, timeout_s=1: table))
    win.ctx.fs.set_mounts(table)
    chooser = browser.FolderBrowser.open(win.root, win.ctx, mode='input', initial=(str(tmp_path), None),
                                         on_done=lambda _path: None)
    until(win.root, lambda: chooser._listing is not None, timeout=2)
    chooser._commit_path('smb://nas.example.org/labshare/ProjectC')
    until(win.root, lambda: chooser.current() == str(target) and chooser._listing is not None and
          not chooser._loading, timeout=2)
    assert not chooser._edit_error and chooser._ov_kind != 'error'
    assert any(r.name == 'Slide1' for r in chooser._rows.values())
    chooser.cancel()
