"""Read-only discovery smoke test on real folders (acceptance B7). Opt-in and generic.

Set YFS_REAL_ROOTS to one or more folders (os.pathsep-separated) that contain unmixed tiles somewhere below them.
For each root the test runs a full discovery (resuming at most twice when the budget pauses it) and checks that it
finishes, that at least one component slide is found, that every folder it could not search is reported with a
note, and - with an audit hook on ``open`` - that no file under the root was opened. Nothing is written anywhere.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from conftest import real_roots
from yfs.core.discovery import Discovery, DiscoveryOptions
from yfs.core.events import CandidateFound, ScanFinished, ScanNote, ScanPaused
from yfs.core.fsservice import FsService
from yfs.core.places import MountTable

ROOTS = real_roots()
pytestmark = [pytest.mark.real, pytest.mark.slow,
              pytest.mark.skipif(not ROOTS, reason="set YFS_REAL_ROOTS to run the read-only real-data smoke test")]

_watch: list = []                    # roots being watched; the hook records nothing while this is empty
_opened: list = []


def _audit(event: str, args: tuple) -> None:
    if event != "open" or not _watch or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if any(path == r or path.startswith(r + os.sep) for r in _watch):
        _opened.append(path)


@pytest.fixture(scope="module")
def audit_hook():
    sys.addaudithook(_audit)         # audit hooks cannot be removed; this one is idle outside the test body
    yield


@pytest.mark.parametrize("root", ROOTS, ids=[f"root{i}" for i in range(len(ROOTS))])
def test_discovery_on_a_real_root_opens_no_file(root, collector, audit_hook):
    fs = FsService(collector, MountTable.load())
    _opened.clear()
    _watch[:] = [os.path.abspath(str(root)).rstrip(os.sep)]
    try:
        d = Discovery(1, str(root), fs, collector, DiscoveryOptions())
        d.start()
        handled, deadline = 0, time.monotonic() + 900
        while not collector.of_type(ScanFinished):
            if len(collector.of_type(ScanPaused)) > handled:
                handled += 1
                d.resume() if handled <= 2 else d.stop()
            assert time.monotonic() < deadline, "discovery did not finish within 15 minutes"
            time.sleep(0.1)
        d.join(5)
    finally:
        _watch.clear()
        fs.shutdown()
    fin = collector.of_type(ScanFinished)[0]
    slides = [e.candidate for e in collector.of_type(CandidateFound) if e.candidate.kind == "component"]
    assert fin.dirs_done > 0 and fin.slides_found == len(slides) >= 1 and fin.max_depth >= 1
    assert all(len(c.tile_names) > 0 and c.folder.startswith(str(root)) for c in slides)
    failures = [e.note.path for e in collector.of_type(ScanNote)
                if e.note.kind in ("no_access", "unresponsive", "error")]
    assert sorted(failures) == sorted(fin.failed)                  # every folder it couldn't search has a note
    assert not _opened, f"{len(_opened)} files were opened during discovery"
