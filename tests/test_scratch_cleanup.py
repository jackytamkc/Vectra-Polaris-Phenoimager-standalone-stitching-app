from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from yfs.core import fsservice, preflight
from yfs.core.fsservice import FsFailure, FsService
from yfs.core.names import PARTIAL_DIR, temp_subdir
from yfs.core.places import MountTable


@pytest.fixture
def scratch_files(tmp_path):
    out, temp = tmp_path / 'out', tmp_path / 'labshare'
    out.mkdir()
    raw = Path(temp_subdir(str(temp), str(out))) / 'Slide1.ome.tmp.raw'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'approved scratch')
    os.utime(raw, (time.time() - 600,) * 2)
    mounts = MountTable.from_mountinfo_text('1 0 8:1 / / rw - ext4 /dev/sda1 rw\n'
                                          f'2 1 0:60 / {temp} rw - cifs //nas.example.org/labshare rw')
    service = FsService(lambda _event: None, mounts)
    _, items = preflight.list_scratch_dir(str(temp), str(out))
    yield service, out, raw, items
    service.shutdown()


def launch(files):
    fs, out, _raw, items = files
    return preflight.delete_scratch(fs, items, out_dir=str(out), out_identity=preflight.path_id(str(out)))


@pytest.mark.parametrize('changed', ['file', 'directory', 'live', 'live_case', 'complete'])
def test_scratch_cleanup_preserves_changed_or_owned_files(scratch_files, changed):
    fs, out, raw, items = scratch_files
    if changed == 'file':
        raw.rename(raw.with_name('original'))
        raw.write_bytes(b'unapproved scratch')
        os.utime(raw, (time.time() - 600,) * 2)
    elif changed == 'directory':
        raw.parent.rename(raw.parent.with_name('original'))
        raw.parent.mkdir()
        raw.write_bytes(b'unapproved scratch')
        os.utime(raw, (time.time() - 600,) * 2)
    else:
        partial = out / PARTIAL_DIR / ('slide1.OME.TIF' if changed == 'live_case' else 'Slide1.ome.tif')
        partial.parent.mkdir()
        partial.write_bytes(b'active output')
        if changed == 'complete':
            preflight.mark_complete(str(partial))
    expected = raw.read_bytes()
    deleted, freed, errors = launch(scratch_files).result(2)
    assert deleted == freed == 0 and errors
    assert raw.read_bytes() == expected


def test_scratch_deletion_uses_its_lane_and_releases_output_ownership(scratch_files, monkeypatch):
    fs, out, raw, _items = scratch_files
    seen = []
    delete = preflight._delete_snapshot
    reserve = preflight._open_reservation

    def checked_delete(*args, **kwargs):
        seen.append(('delete', threading.current_thread().name))
        return delete(*args, **kwargs)

    def checked_reserve(*args, **kwargs):
        seen.append(('reserve', threading.current_thread().name))
        return reserve(*args, **kwargs)

    monkeypatch.setattr(preflight, '_delete_snapshot', checked_delete)
    monkeypatch.setattr(preflight, '_open_reservation', checked_reserve)
    assert launch(scratch_files).result(2)[:2] == (1, len(b'approved scratch'))
    assert not raw.exists()
    assert all('network' in thread for op, thread in seen if op == 'delete')
    assert all('local' in thread for op, thread in seen if op == 'reserve')
    fs.submit(str(out), 'call', lambda: None).result(2)
    with preflight.reserve_output(str(out / PARTIAL_DIR / 'Slide1.ome.tif')):
        pass


def test_timed_out_scratch_keeps_reservations_and_local_output_usable(scratch_files, monkeypatch):
    fs, out, raw, _items = scratch_files
    entered, release, released = threading.Event(), threading.Event(), threading.Event()
    delete, close = preflight._delete_snapshot, preflight._ScratchCleanup.close

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return delete(*args, **kwargs)

    def closed(cleanup):
        close(cleanup)
        released.set()

    monkeypatch.setattr(preflight, '_delete_snapshot', blocked)
    monkeypatch.setattr(preflight._ScratchCleanup, 'close', closed)
    monkeypatch.setitem(fsservice.DEADLINE['delete'], 'network', .05)
    future = launch(scratch_files)
    try:
        assert entered.wait(2)
        with pytest.raises(FsFailure) as error:
            future.result(2)
        assert error.value.error.code == 'timeout'
        assert fs.submit(str(out), 'call', lambda: 'local works').result(1) == 'local works'
        assert fs.health(str(out)) != 'unresponsive'
        with pytest.raises(preflight.ReservationBusy):
            with preflight.reserve_output(str(out / PARTIAL_DIR / 'Slide1.ome.tif')):
                pass
    finally:
        release.set()
    assert released.wait(2)
    assert not raw.exists()
    with preflight.reserve_output(str(out / PARTIAL_DIR / 'Slide1.ome.tif')):
        pass


def test_scratch_ownership_listing_happens_after_reservation(scratch_files, monkeypatch):
    _fs, out, raw, _items = scratch_files
    partial = out / PARTIAL_DIR / 'slide1.OME.TIF'
    reserve = preflight._open_reservation

    def writer_arrives(path):
        if not partial.exists():
            partial.parent.mkdir(exist_ok=True)
            partial.write_bytes(b'active writer')
        return reserve(path)

    monkeypatch.setattr(preflight, '_open_reservation', writer_arrives)
    deleted, freed, errors = launch(scratch_files).result(2)
    assert deleted == freed == 0 and errors
    assert raw.read_bytes() == b'approved scratch'
