"""FsService: lanes, priorities, duplicates, tickets, deadlines, fail-fast, retry, recovery, thread bounds, cache
(FINAL_SPEC §5.5.3, acceptance B3). Blocking is simulated with tests/fake_fs.py; nothing touches real data."""
from __future__ import annotations

import errno
import ntpath
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from functools import partial
from pathlib import Path

import pytest

from fake_fs import FakeOs
from yfs.core import fsservice, osutil
from yfs.core.events import FsDone, FsSlow, LaneHealthChanged, MountsChanged
from yfs.core.fsservice import FsFailure, FsService, friendly_error, map_oserror, probe_dir, scan_dir
from yfs.core.model import LOCAL, NETWORK, Listing
from yfs.core.places import Mount, MountTable

REPO = Path(__file__).resolve().parents[1]


def table(*mounts) -> MountTable:
    """A mount table: ext4 root plus (mount_point, fstype, source) entries."""
    lines = ["1 0 8:1 / / rw - ext4 /dev/sda1 rw"]
    for i, (mp, fstype, src) in enumerate(mounts, 2):
        lines.append(f"{i} 1 0:{i} / {str(mp).replace(' ', chr(92) + '040')} rw - {fstype} {src} rw")
    return MountTable.from_mountinfo_text("\n".join(lines))


@pytest.fixture
def make_fs(collector):
    made, gates = [], []

    def make(mounts=None, os_api=None, **kw):
        fs = FsService(collector, mounts or table(), os_api=os_api, **kw)
        made.append(fs)
        return fs

    make.gates = gates
    yield make
    for g in gates:
        g.set()
    for fs in made:
        fs.shutdown()


def done(collector, rid, timeout=5.0) -> FsDone:
    return collector.wait_for(lambda evs: next((e for e in evs if isinstance(e, FsDone) and e.req_id == rid), None),
                              timeout)


def until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        time.sleep(0.01)


def test_list_dir_result_cache_and_health(tmp_path, collector, make_fs):
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "S1_[1,2]_component_data.tif").write_bytes(b"")
    fs = make_fs()
    t = fs.new_ticket()
    ev = done(collector, fs.list_dir(str(tmp_path), ticket=t, owner="browser", tag="T"))
    assert (ev.error, ev.kind, ev.ticket, ev.owner, ev.tag) == (None, "list", t, "browser", "T")
    assert sorted(e.name for e in ev.result.entries) == ["S1_[1,2]_component_data.tif", "notes.txt", "sub"]
    assert ev.result.mtime is not None and not ev.result.truncated
    assert fs.cached_listing(str(tmp_path)) is ev.result
    s = fs.cached_summary(str(tmp_path))
    assert s.subdirs == ("sub",) and s.groups[0].prefix == "S1" and s.other_files == 1
    assert fs.health(str(tmp_path)) == "ok" and fs.health(fs.lane_key(str(tmp_path))) == "ok"
    time.sleep(0.05)
    assert fs.cached_listing(str(tmp_path), max_age=0.01) is None
    fs.invalidate(str(tmp_path))
    assert fs.cached_listing(str(tmp_path)) is None and fs.cached_summary(str(tmp_path)) is None


def test_probe_stat_usage_and_errors(tmp_path, collector, make_fs):
    (tmp_path / "d").mkdir()
    (tmp_path / "f.txt").write_text("abc")
    os.symlink(tmp_path / "d", tmp_path / "link_to_d")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        fs = make_fs()
        t = fs.new_ticket()
        ev = done(collector, fs.probe(str(tmp_path), ticket=t, owner="b"))
        assert ev.result.subdirs == ("d", "locked") and ev.result.links == 1
        assert fs.cached_summary(str(tmp_path)) is ev.result
        st = done(collector, fs.stat_names(str(tmp_path), ["f.txt", "link_to_d", "missing", "d"], ticket=t, owner="b"))
        assert set(st.result) == {"f.txt", "link_to_d", "d"}
        assert st.result["f.txt"][0] == 3 and not st.result["f.txt"][2] and st.result["d"][2]
        assert not st.result["link_to_d"][2]                  # lstat: a link is not followed (readlink() is)
        assert done(collector, fs.disk_usage(str(tmp_path), ticket=t, owner="b")).result.free > 0
        codes = [done(collector, fs.probe(str(p), ticket=t, owner="b")).error.code
                 for p in (tmp_path / "missing", tmp_path / "f.txt", locked)]
        assert codes == ["not_found", "not_a_dir", "no_access"]
        boom = done(collector, fs.call(str(tmp_path), "call", lambda: 1 / 0, ticket=t, owner="b"))
        assert boom.error.code == "io" and "division" in boom.error.message
        fut = fs.submit(str(tmp_path / "missing"), "list", partial(fs.scan, str(tmp_path / "missing"), cap=5))
        with pytest.raises(FsFailure) as info:
            fut.result(timeout=5)
        assert info.value.error.code == "not_found"
    finally:
        locked.chmod(0o755)


def test_priorities_and_duplicate_merge(tmp_path, collector, make_fs):
    for n in "abcde":
        (tmp_path / n).mkdir()
    fake = FakeOs()
    make_fs.gates.append(fake.block(tmp_path / "a"))
    fs = make_fs(table((tmp_path, "fuse.rclone", "remote:")), fake)     # cloud lane: a single worker
    t = fs.new_ticket()
    fs.probe(str(tmp_path / "a"), ticket=t, owner="b")
    until(lambda: fake.inside == 1)
    ids = {n: fs.probe(str(tmp_path / n), ticket=t, owner="b", priority=p) for n, p in
           (("b", 6), ("c", 4), ("d", 6), ("e", 1))}
    dup = fs.probe(str(tmp_path / "d"), ticket=t, owner="b", tag="again", priority=0)   # merged, and bumped to 0
    make_fs.gates[0].set()
    results = {n: done(collector, rid) for n, rid in ids.items()}
    again = done(collector, dup)
    assert [Path(p).name for p in fake.paths("scandir")] == ["a", "d", "e", "c", "b"]
    assert again.result is results["d"].result and again.tag == "again"


def test_a_more_urgent_duplicate_moves_the_request_up_where_it_is_queued(tmp_path, collector, make_fs):
    """A newer mount table gives a queued folder a lane of its own; a duplicate with a better priority must move
    the request up on the lane that holds it (the new lane has no thread for it, and the old entry is stale)."""
    share = tmp_path / "share"
    for n in "abc":
        (share / n).mkdir(parents=True)
    fake = FakeOs()
    make_fs.gates.append(fake.block(share / "a"))
    fs = make_fs(table((share, "fuse.rclone", "remote:")), fake)       # cloud lane: a single worker
    t = fs.new_ticket()
    fs.probe(str(share / "a"), ticket=t, owner="b")
    until(lambda: fake.inside == 1)
    queued = fs.probe(str(share / "b"), ticket=t, owner="b", priority=6)
    later = fs.probe(str(share / "c"), ticket=t, owner="b", priority=4)
    fs.set_mounts(table((share, "fuse.rclone", "remote:"), (share / "b", "cifs", "//fileserver.example.org/b")))
    dup = fs.probe(str(share / "b"), ticket=t, owner="b", priority=1)   # merged, and now ahead of c
    make_fs.gates[0].set()
    assert done(collector, queued).error is None and done(collector, dup).error is None
    assert done(collector, later).error is None
    assert [Path(p).name for p in fake.paths("scandir")] == ["a", "b", "c"]


def test_cancelled_ticket_is_dropped(tmp_path, collector, make_fs):
    for n in "abcd":
        (tmp_path / n).mkdir()
    fake = FakeOs()
    gate = fake.block(tmp_path / "a")
    make_fs.gates.append(gate)
    fs = make_fs(table((tmp_path, "fuse.rclone", "remote:")), fake)
    old, new = fs.new_ticket(), fs.new_ticket()
    first = fs.probe(str(tmp_path / "a"), ticket=old, owner="b")
    until(lambda: fake.inside == 1)
    dropped = fs.probe(str(tmp_path / "b"), ticket=old, owner="b")
    fut = fs.submit(str(tmp_path / "c"), "list", partial(fs.scan, str(tmp_path / "c"), cap=10), ticket=old)
    kept = fs.probe(str(tmp_path / "d"), ticket=new, owner="b")
    fs.cancel_ticket(old)
    gate.set()
    assert done(collector, kept).error is None
    assert done(collector, first).error is None                  # already running: delivered normally
    until(fut.cancelled)
    assert not [e for e in collector.of_type(FsDone) if e.req_id == dropped]
    assert [Path(p).name for p in fake.paths("scandir")] == ["a", "d"]


def test_slow_deadline_fail_fast_retry_and_recovery(tmp_path, collector, make_fs):
    share = tmp_path / "share"
    for n in ("hung/a", "hung/b", "y", "z", "w"):
        (share / n).mkdir(parents=True)
    fake = FakeOs()
    gate = fake.block(share / "hung")
    make_fs.gates.append(gate)
    fs = make_fs(table((share, "cifs", "//fileserver.example.ac.uk/share")), fake)
    fs.SLOW_S = 0.3
    lane = fs.lane_key(str(share))
    assert fs.concurrency(str(share)) == 2 and fs.lane_class(str(share)) == "network"
    t = fs.new_ticket()
    hang = lambda p: fake.stat(str(p))                           # noqa: E731
    stuck1, _ = (fs.call(str(share / "hung" / n), "locate", partial(hang, share / "hung" / n), ticket=t,
                         owner="browser", deadline_s=1.0) for n in "ab")
    until(lambda: fake.inside == 2)
    queued = fs.probe(str(share / "y"), ticket=t, owner="browser")             # no free worker: queued
    slow = collector.wait_for(lambda evs: [e for e in evs if isinstance(e, FsSlow) and e.req_id == stuck1])
    assert not slow[0].past_deadline
    ev = done(collector, queued, timeout=3)                       # fails at once when the deadline passes
    assert ev.error.code == "unresponsive"
    past = [e for e in collector.of_type(FsSlow) if e.req_id == stuck1 and e.past_deadline]
    assert past and fs.health(lane) == "unresponsive"
    timed_out = done(collector, stuck1)                           # the deadline is the answer: not a listing
    assert (timed_out.error.code, timed_out.error.message) == ("timeout", "share didn't answer.")
    assert timed_out.elapsed >= 1.0
    health = [e.health for e in collector.of_type(LaneHealthChanged) if e.lane == lane]
    assert health[:2] == ["slow", "unresponsive"]
    until(lambda: len(fs._lanes[lane].stuck) == 2)
    t0 = time.monotonic()
    ev = done(collector, fs.list_dir(str(share / "y"), ticket=t, owner="browser"))   # user_initiated: still refused
    assert ev.error.code == "unresponsive" and time.monotonic() - t0 < 0.5
    fs.retry(lane)
    passed = fs.list_dir(str(share / "z"), ticket=t, owner="browser")
    refused = fs.list_dir(str(share / "w"), ticket=t, owner="browser")
    assert done(collector, refused).error.code == "unresponsive"
    assert done(collector, passed).error is None
    assert fs.health(lane) == "slow" and fs.has_stuck_threads()   # the stuck calls are still out there
    gate.set()
    until(lambda: fs.health(lane) == "ok" and not fs.has_stuck_threads())
    assert [e.error.code for e in collector.of_type(FsDone) if e.req_id == stuck1] == ["timeout"]   # once only
    assert done(collector, fs.probe(str(share / "w"), ticket=t, owner="b")).error is None


def test_fs_slow_at_the_default_two_and_a_half_seconds(tmp_path, collector, make_fs):
    fake = FakeOs()
    gate = fake.block(tmp_path)
    make_fs.gates.append(gate)
    fs = make_fs(table((tmp_path, "cifs", "//fileserver.example.ac.uk/share")), fake)
    assert fs.SLOW_S == 2.5
    t0 = time.monotonic()
    rid = fs.list_dir(str(tmp_path), ticket=fs.new_ticket(), owner="browser")
    [slow] = collector.wait_for(lambda evs: [e for e in evs if isinstance(e, FsSlow) and e.req_id == rid], 5)
    assert 2.5 <= time.monotonic() - t0 < 3.5 and not slow.past_deadline and 2.5 <= slow.elapsed < 3.0
    assert fs.health(str(tmp_path)) == "slow"
    gate.set()
    assert done(collector, rid).error is None and fs.health(str(tmp_path)) == "ok"


def test_hung_lane_holds_at_most_concurrency_plus_one_threads(tmp_path, collector, make_fs):
    share = tmp_path / "share"
    for i in range(8):
        (share / f"d{i}").mkdir(parents=True)
    fake = FakeOs()
    make_fs.gates.append(fake.block(share))                      # everything on the share hangs
    fs = make_fs(table((share, "nfs4", "nas.example.org:/export")), fake)
    fs.SLOW_S = 0.1
    t = fs.new_ticket()
    probe = lambda i: fs.call(str(share / f"d{i}"), "probe", partial(fake.scandir, str(share / f"d{i}")),  # noqa
                              ticket=t, owner="b", deadline_s=0.4)
    ids = [probe(i) for i in range(4)]
    for rid in ids[2:]:
        assert done(collector, rid).error.code == "unresponsive"
    codes = []
    for k in range(4, 8):                                        # retry repeatedly: one pass each time
        fs.retry(fs.lane_key(str(share)))
        codes.append(done(collector, probe(k), timeout=0.4 + 1).error.code)   # always answered (CT-3)
    lane = fs._lanes[fs.lane_key(str(share))]
    assert fake.max_inside == 3 and lane.threads == 3            # concurrency (2) + 1
    assert codes == ["timeout", "unresponsive", "unresponsive", "unresponsive"]   # then: no thread left to try
    assert lane.passes == 1 and len(fake.paths("scandir")) == 3    # refused at once, the pass kept for later
    names = [th.name for th in threading.enumerate() if th.name == "yfs-fs-network"]
    assert len(names) >= 3


def test_disconnected_lane(tmp_path, collector, make_fs):
    fs = make_fs(table((tmp_path, "cifs", "//fileserver.example.ac.uk/share")))
    lane = fs.lane_key(str(tmp_path))
    fs.mark_disconnected(lane, True)
    assert collector.of_type(LaneHealthChanged)[-1].health == "disconnected"
    t = fs.new_ticket()
    assert done(collector, fs.probe(str(tmp_path), ticket=t, owner="b")).error.code == "disconnected"
    assert done(collector, fs.list_dir(str(tmp_path), ticket=t, owner="b")).error is None   # user navigation
    assert fs.health(lane) == "ok"                            # it answered: Windows reconnected the drive
    assert done(collector, fs.probe(str(tmp_path), ticket=t, owner="b")).error is None
    fs.mark_disconnected(lane, True)
    fs.mark_disconnected(lane, False)
    assert fs.health(lane) == "unknown"


def windows_table(*disconnected: str) -> MountTable:
    return MountTable([Mount("C:\\", "local", "", LOCAL, "Local Disk (C:)"),
                       Mount("Y:\\", "remote", r"\\nas.example.org\old", NETWORK, "Y: old")],
                      flavor="win32", disconnected=disconnected)


def test_drives_known_to_be_disconnected_are_marked_at_start(collector, make_fs):
    fs = make_fs(windows_table("Y:"))
    assert fs.health("Y:") == "disconnected" and fs.health(r"C:\data") == "unknown"
    rid = fs.probe(r"Y:\Lab", ticket=fs.new_ticket(), owner="b")
    assert done(collector, rid).error.code == "disconnected"


def test_a_disconnected_drive_is_entered_once_it_answers(collector, make_fs):
    """FS-2 / PL-15 / D5: 'disconnected' follows the lane, not the mount table read at start."""
    fs = make_fs(windows_table("Y:"))
    slide = r"Y:\Lab\Slide1_Scan1"
    assert fs.auto_enter(slide) == "disconnected"
    opened = fs.call(r"Y:\Lab", "locate", lambda: ("dir", None), ticket=0, owner="main", user_initiated=True)
    assert done(collector, opened).error is None                  # the user opened it: Windows reconnected it
    assert fs.auto_enter(slide) is None and fs.mounts.auto_enter(slide) == "disconnected"   # the table is old
    fs.set_mounts(windows_table("Y:"))                            # the same report again: no change
    assert fs.auto_enter(slide) is None and not collector.of_type(MountsChanged)
    fs.set_mounts(windows_table())                                # Windows no longer reports it
    assert fs.auto_enter(slide) is None and fs.health("Y:") == "ok" and len(collector.of_type(MountsChanged)) == 1
    fs.set_mounts(windows_table("Y:"))                            # dropped again: a new report
    assert fs.auto_enter(slide) == "disconnected" and fs.health("Y:") == "disconnected"


def test_auto_enter_follows_the_mount_table(collector, make_fs):
    fs = make_fs(table(("/mnt/labshare", "autofs", "systemd-1"), ("/home/labuser/mnt/remote", "fuse.rclone", "r:")))
    assert [fs.auto_enter(p) for p in ("/mnt/labshare", "/mnt/labshare/Project", "/home/labuser/mnt/remote/x",
                                       "/proc", "/home/labuser")] == ["autofs", None, "cloud", None, None]


def test_a_deadline_is_the_answer_except_for_a_listing(tmp_path, collector, make_fs, monkeypatch):
    """CT-1 / BR-15 / D8: at the deadline every waiter of a probe, stat, header, ... gets FsDone(timeout) once;
    a listing's waiters get FsSlow(past_deadline) and its late result (the browser's Keep waiting)."""
    disk = tmp_path / "disk"
    for name in ("sub", "child"):
        (disk / name).mkdir(parents=True)
    fake = FakeOs()
    gate = fake.block(disk)
    make_fs.gates.append(gate)
    for kind in ("list_browser", "probe"):
        monkeypatch.setitem(fsservice.DEADLINE[kind], "local", 0.6)
    fs = make_fs(table((disk, "ext4", "/dev/sdb1")), fake)        # a local lane: four calls at once
    fs.SLOW_S = 0.2
    t, sub, child = fs.new_ticket(), str(disk / "sub"), str(disk / "child")
    listing = fs.list_dir(sub, ticket=t, owner="browser")
    probe = fs.probe(child, ticket=t, owner="browser")
    read_header = partial(fake.stat, sub)
    header = fs.call(sub, "header", read_header, ticket=t, owner="main", tag="h1", deadline_s=0.6)
    merged = fs.call(sub, "header", read_header, ticket=t, owner="main", tag="h2", deadline_s=0.6)
    fut = fs.submit(sub, "check_output", partial(fake.stat, sub), deadline_s=0.6)
    for rid, name in ((probe, "child"), (header, "sub"), (merged, "sub")):
        ev = done(collector, rid, timeout=3)
        assert (ev.error.code, ev.error.message, ev.result) == ("timeout", f"{name} didn't answer.", None)
        assert [e for e in collector.of_type(FsSlow) if e.req_id == rid and e.past_deadline]
    with pytest.raises(FsFailure) as info:
        fut.result(timeout=3)
    assert info.value.error.code == "timeout"
    assert [e for e in collector.of_type(FsSlow) if e.req_id == listing and e.past_deadline]
    assert not [e for e in collector.of_type(FsDone) if e.req_id == listing]
    gate.set()
    late = done(collector, listing)                               # the listing answers late
    assert late.error is None and [e.name for e in late.result.entries] == [] and late.elapsed >= 0.6
    until(lambda: fs.cached_summary(child) is not None and not fs.has_stuck_threads())   # the late probe fills it
    for rid in (probe, header, merged):                           # but is not delivered a second time
        assert len([e for e in collector.of_type(FsDone) if e.req_id == rid]) == 1


def test_try_again_after_a_deadline_runs_a_new_request(tmp_path, collector, make_fs):
    """CT-2 / FS-8: a retry with the same key (MainWindow: ticket 0) is not merged into the abandoned call."""
    share = tmp_path / "share"
    (share / "data").mkdir(parents=True)
    fake = FakeOs()
    make_fs.gates.append(fake.block(share))
    fs = make_fs(table((share, "cifs", "//fileserver.example.org/share")), fake)
    locate = partial(fake.stat, str(share / "data"))
    first = fs.call(str(share / "data"), "locate", locate, ticket=0, owner="main", deadline_s=0.5)
    assert done(collector, first, timeout=3).error.code == "timeout"
    fs.retry(str(share))
    t0 = time.monotonic()
    again = fs.call(str(share / "data"), "locate", locate, ticket=0, owner="main", deadline_s=0.5)
    ev = done(collector, again, timeout=3)
    assert ev.error.code == "timeout" and time.monotonic() - t0 >= 0.5   # after its own deadline
    assert [e for e in collector.of_type(FsSlow) if e.req_id == again and e.past_deadline]
    assert fake.paths("stat").count(str(share / "data")) == 2     # a second attempt was really made


def test_queued_requests_time_out_when_no_thread_can_run_them(tmp_path, collector, make_fs):
    """D8 safety net: every thread of the process is stuck elsewhere (max_threads), so nothing would run it."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake = FakeOs()
    make_fs.gates.append(fake.block(a))
    fs = make_fs(table((a, "ext4", "/dev/sdb1"), (b, "xfs", "/dev/sdc1")), fake, max_threads=1)
    t = fs.new_ticket()
    hung = fs.call(str(a), "locate", partial(fake.stat, str(a)), ticket=t, owner="b", deadline_s=0.3)
    until(lambda: fake.inside == 1)
    t0 = time.monotonic()
    starved = fs.call(str(b), "locate", partial(fake.stat, str(b)), ticket=t, owner="b", deadline_s=0.6)
    ev = done(collector, starved, timeout=3)
    assert (ev.error.code, ev.error.message) == ("timeout", "b didn't answer.") and time.monotonic() - t0 >= 0.6
    assert done(collector, hung).error.code == "timeout" and fake.paths("stat") == [str(a)]   # it never ran


def test_a_busy_lane_is_not_a_hung_one(tmp_path, collector, make_fs):
    """A request that waits in the queue of a lane whose thread is busy (not stuck) is never timed out."""
    fake = FakeOs()
    fake.delay(tmp_path / "slow", 0.8)
    (tmp_path / "slow").mkdir()
    (tmp_path / "next").mkdir()
    fs = make_fs(table((tmp_path, "fuse.rclone", "remote:")), fake)     # cloud lane: a single thread
    t = fs.new_ticket()
    busy = fs.call(str(tmp_path / "slow"), "header", partial(fake.stat, str(tmp_path / "slow")), ticket=t,
                   owner="b", deadline_s=5)
    waiting = fs.call(str(tmp_path / "next"), "header", partial(fake.stat, str(tmp_path / "next")), ticket=t,
                      owner="b", deadline_s=0.3)
    assert done(collector, busy).error is None and done(collector, waiting).error is None


def test_requests_in_a_row_get_the_free_threads_of_a_lane_with_an_idle_one(tmp_path, collector, make_fs):
    """The second of two requests in a row must not wait behind the first (which may hang) while the lane may run
    two: the idle thread notified for the first still counts as idle until it wakes up."""
    share = tmp_path / "share"
    for n in "ab":
        (share / n).mkdir(parents=True)
    fake = FakeOs()
    make_fs.gates.extend(fake.block(share / n) for n in "ab")
    fs = make_fs(table((share, "cifs", "//fileserver.example.org/share")), fake)      # network lane: two threads
    t = fs.new_ticket()
    assert done(collector, fs.probe(str(share), ticket=t, owner="b")).error is None
    lane = fs._lanes[fs.lane_key(str(share))]
    until(lambda: lane.idle == 1)                                # one thread, waiting for work
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)                                   # it can't wake up between the two requests
    try:
        ids = [fs.probe(str(share / n), ticket=t, owner="b") for n in "ab"]
    finally:
        sys.setswitchinterval(interval)
    until(lambda: fake.inside == 2, timeout=2)
    for gate in make_fs.gates:
        gate.set()
    assert [done(collector, rid).error for rid in ids] == [None, None] and lane.threads == 2


def test_stat_names_stays_on_its_own_lane(tmp_path, collector, make_fs):
    """BR-1 / D9: lstat only, and never a mount point of another volume (stat'ing a hung share's mount point from
    a local folder would make the whole local disk 'not answering')."""
    (tmp_path / "f.txt").write_text("abc")
    (tmp_path / "d").mkdir()
    (tmp_path / "nasA").mkdir()
    os.symlink(tmp_path / "nasA", tmp_path / "lab_data")
    fake = FakeOs()
    make_fs.gates.append(fake.block(tmp_path / "nasA"))           # the share's server is down
    make_fs.gates.append(fake.block(tmp_path / "lab_data"))       # so is anything that follows the link
    fs = make_fs(table((tmp_path / "nasA", "cifs", "//nas.example.org/a")), fake)
    names = ["f.txt", "d", "nasA", "lab_data"]
    ev = done(collector, fs.stat_names(str(tmp_path), names, ticket=fs.new_ticket(), owner="browser"), timeout=3)
    assert set(ev.result) == {"f.txt", "d", "lab_data"} and ev.result["d"][2] and not ev.result["lab_data"][2]
    assert str(tmp_path / "nasA") not in fake.paths() and not fake.inside
    assert fs.health(str(tmp_path)) == "ok" and fs.health(str(tmp_path / "nasA")) == "unknown"


def test_readlink_resolves_a_link_without_touching_its_target(tmp_path, collector, make_fs):
    """FS-9 / BR-1 / D9: the browser opens a link by its target, on the target's own lane."""
    share, home = tmp_path / "share", tmp_path / "home"
    (share / "Project").mkdir(parents=True)
    (home / "plain").mkdir(parents=True)
    os.symlink(share / "Project", home / "abs_link")
    os.symlink(os.path.join("..", "share", "Project"), home / "rel_link")
    os.symlink(tmp_path / "gone", home / "broken")
    fake = FakeOs()
    make_fs.gates.append(fake.block(share))                       # the share hangs, and so does anything that
    make_fs.gates.append(fake.block(home / "abs_link"))           # follows a link into it
    fs = make_fs(table((share, "cifs", "//fileserver.example.org/share")), fake)
    t = fs.new_ticket()
    got = {n: done(collector, fs.readlink(str(home / n), ticket=t, owner="browser", tag=n), timeout=3)
           for n in ("abs_link", "rel_link", "broken", "plain", "missing")}
    assert got["abs_link"].result == got["rel_link"].result == str(share / "Project")
    assert got["broken"].result == str(tmp_path / "gone") and got["plain"].result is None
    assert got["missing"].error.code == "not_found"
    ev = got["rel_link"]
    assert (ev.kind, ev.path, ev.tag, ev.owner) == ("readlink", str(home / "rel_link"), "rel_link", "browser")
    typed = done(collector, fs.readlink(str(home / "abs_link") + os.sep, ticket=t, owner="main"), timeout=3)
    assert (typed.result, typed.path) == (str(share / "Project"), str(home / "abs_link"))   # 'link/': not followed
    assert not fake.inside and fs.health(str(share)) == "unknown"    # nothing touched the share
    assert fs.lane_key(ev.result) == str(share) and fs.lane_class(ev.result) == NETWORK


def test_fake_os_reads_a_link_through_the_folder_holding_it(tmp_path):
    """tests/fake_fs.py: lstat and readlink of a link read the folder holding it, like the real calls, so the
    rules of the folders above it apply; a rule on the link itself (what following it does) does not."""
    share = tmp_path / "share"
    share.mkdir()
    (tmp_path / "target").mkdir()
    os.symlink(tmp_path / "target", share / "link")
    link, denied = str(share / "link"), PermissionError(errno.EACCES, "Permission denied")
    fake = FakeOs()
    fake.fail(link, denied, recursive=True)                       # following the link fails
    assert fake.readlink(link) == str(tmp_path / "target") and fake.stat(link, follow_symlinks=False)
    with pytest.raises(PermissionError):
        fake.stat(link)
    fake.fail(share, denied)                                      # the folder itself, not what is in it
    assert fake.readlink(link) and fake.stat(link, follow_symlinks=False)
    fake.fail(share, denied, recursive=True)                      # the share and everything in it
    for call in (partial(fake.readlink, link), partial(fake.stat, link, follow_symlinks=False)):
        with pytest.raises(PermissionError):
            call()


def test_windows_link_targets_lose_only_the_prefix_of_a_drive_or_share():
    """Windows' os.readlink gives NT paths. A volume GUID target (a disk mounted in an empty folder) keeps its
    \\\\?\\ prefix: without it, it would be joined to the link's folder as a relative path."""
    assert fsservice._from_nt("\\\\?\\C:\\Lab\\Scans") == "C:\\Lab\\Scans"
    assert fsservice._from_nt("\\\\?\\UNC\\fileserver.example.org\\labshare") == "\\\\fileserver.example.org\\labshare"
    vol = "\\\\?\\Volume{01234567-89ab-cdef-0123-456789abcdef}\\"
    assert fsservice._from_nt(vol) == vol and fsservice._from_nt("..\\share\\x") == "..\\share\\x"
    links = "C:\\Users\\labuser\\links"                          # what _readlink makes of it on Windows
    assert ntpath.normpath(ntpath.join(links, fsservice._from_nt(vol + "Scans"))) == vol + "Scans"
    assert ntpath.normpath(ntpath.join(links, fsservice._from_nt("\\\\?\\D:\\Scans"))) == "D:\\Scans"


def test_set_mounts_keeps_lanes_and_routes_new_mounts(tmp_path, collector, make_fs):
    """D5: lanes are made on first use, so a newer table keeps the lanes (and their state) of unchanged volumes."""
    (tmp_path / "late").mkdir()
    fs = make_fs(table())
    t = fs.new_ticket()
    assert done(collector, fs.probe(str(tmp_path / "late"), ticket=t, owner="b")).error is None
    root = fs._lanes["/"]
    late = "/mnt/latecomer/Project"                               # string only: the lambdas touch nothing
    assert done(collector, fs.call(late, "locate", lambda: 1, ticket=t, owner="b")).error is None
    lane = fs._lanes["/mnt/latecomer"]
    assert (lane.cls, fs.lane_class(late), fs.concurrency(late)) == (LOCAL, LOCAL, 4)
    newer = table(("/mnt/latecomer", "cifs", "//fileserver.example.org/late"), (tmp_path / "late", "nfs4", "nas:/x"))
    fs.set_mounts(newer)
    assert fs.mounts is newer and len(collector.of_type(MountsChanged)) == 1
    assert (fs.lane_key(late), fs.lane_class(late), fs.concurrency(late)) == ("/mnt/latecomer", NETWORK, 2)
    assert done(collector, fs.call(late, "locate", lambda: 2, ticket=t, owner="b")).result == 2
    assert fs._lanes["/mnt/latecomer"] is lane and lane.cls == NETWORK and fs._lanes["/"] is root
    assert fs.lane_key(str(tmp_path / "late" / "x")) == str(tmp_path / "late")
    fs.set_mounts(table(("/mnt/latecomer", "cifs", "//fileserver.example.org/late"), (tmp_path / "late", "nfs4",
                                                                                      "nas:/x")))
    assert len(collector.of_type(MountsChanged)) == 1             # an equal table changes nothing


def test_reload_mounts_reads_the_table_at_most_every_interval(monkeypatch, collector, make_fs):
    """D5: reload_mounts() on a daemon thread, rate-limited, never installing a table that could not be read."""
    first = table(("/mnt/latecomer", "cifs", "//fileserver.example.org/late"))
    tables, loads = [first, MountTable(), table(("/mnt/other", "nfs4", "nas:/other"))], []

    def load(cls, timeout_s=1.0):
        loads.append(time.monotonic())
        return tables[len(loads) - 1]

    monkeypatch.setattr(MountTable, "load", classmethod(load))
    fs = make_fs(table())
    fs.RELOAD_S = 0.3
    for _ in range(3):                                            # e.g. the browser opens, then a rescan
        fs.reload_mounts()
    until(lambda: len(loads) == 2 and not fs._reloading)          # one now, one at the end of the interval
    assert loads[1] - loads[0] >= 0.3 and fs.mounts is first      # the empty (unreadable) table was not used
    assert len(collector.of_type(MountsChanged)) == 1 and fs.lane_class("/mnt/latecomer/x") == NETWORK
    fs.reload_mounts()                                            # later: read at once
    collector.wait_for(lambda evs: len([e for e in evs if isinstance(e, MountsChanged)]) == 2, 5)
    assert len(loads) == 3 and fs.lane_key("/mnt/other/x") == "/mnt/other"


def test_a_hung_share_mounted_after_start_leaves_local_folders_alone(tmp_path, collector, make_fs):
    """CT-7 / D5: the table was read before /mnt/latecomer was mounted; its hang must not stop the local disk."""
    fs = make_fs(table())
    gate = threading.Event()
    make_fs.gates.append(gate)
    hung = fs.call("/mnt/latecomer/Project", "locate", gate.wait, ticket=0, owner="main", deadline_s=0.5)
    assert done(collector, hung, timeout=3).error.code == "timeout"
    assert fs.health("/mnt/latecomer") == "unresponsive" and fs.health("/") != "unresponsive"
    (tmp_path / "Documents").mkdir()
    t = fs.new_ticket()
    assert done(collector, fs.list_dir(str(tmp_path / "Documents"), ticket=t, owner="browser")).error is None
    assert done(collector, fs.disk_usage(str(tmp_path), ticket=t, owner="main")).error is None


def test_idle_workers_exit_and_global_thread_cap(tmp_path, collector, make_fs):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake = FakeOs()
    gate = fake.block(a)
    make_fs.gates.append(gate)
    fs = make_fs(table((a, "ext4", "/dev/sdb1"), (b, "xfs", "/dev/sdc1")), fake, max_threads=1)
    t = fs.new_ticket()
    first = fs.probe(str(a), ticket=t, owner="b")
    until(lambda: fake.inside == 1)
    waiting = fs.probe(str(b), ticket=t, owner="b")              # another lane, but no thread is left
    time.sleep(0.3)
    assert not [e for e in collector.of_type(FsDone) if e.req_id == waiting]
    gate.set()
    assert done(collector, first).error is None
    assert done(collector, waiting, timeout=3).error is None      # the idle worker gave its slot away
    fs.IDLE_S = 0.1
    fs.probe(str(a), ticket=fs.new_ticket(), owner="b")
    until(lambda: fs._nthreads == 0)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_dying_worker_frees_its_slot(tmp_path, collector, make_fs):
    fs = make_fs(table((tmp_path, "fuse.rclone", "remote:")))            # one worker
    t = fs.new_ticket()

    def die():
        raise SystemExit                                      # escapes the normal error handling

    fs.call(str(tmp_path), "call", die, ticket=t, owner="b")
    until(lambda: fs._nthreads == 0)
    assert done(collector, fs.probe(str(tmp_path), ticket=t, owner="b")).error is None


def test_cache_is_an_lru_of_2000(make_fs):
    fs = make_fs()
    for i in range(2001):
        fs._store(f"/x/{i}", Listing(f"/x/{i}", (), False, 0.0, time.monotonic()), None)
    assert len(fs._cache) == 2000 and fs.cached_listing("/x/0") is None and fs.cached_listing("/x/2000")


def test_scan_dir_caps_links_and_hidden(tmp_path):
    for i in range(30):
        (tmp_path / f"S_[{i},0]_component_data.tif").write_bytes(b"")
    full = scan_dir(str(tmp_path), cap=10, cap_only_without_tiles=True)
    capped = scan_dir(str(tmp_path), cap=10)
    assert (len(full.entries), full.truncated) == (30, False) and (len(capped.entries), capped.truncated) == (10, True)
    other = tmp_path / "other"
    other.mkdir()
    for i in range(30):
        (other / f"f{i}.txt").write_bytes(b"")
    (other / ".hidden").write_bytes(b"")
    os.symlink(tmp_path, other / "dirlink")
    os.symlink(other / "f0.txt", other / "filelink")
    lst = scan_dir(str(other), cap=100, stat_self=True)
    by = {e.name: e for e in lst.entries}
    assert by["dirlink"].is_link and not by["dirlink"].is_dir
    assert by["filelink"].is_link and by[".hidden"].hidden and lst.mtime is not None
    assert len(scan_dir(str(other), cap=10, cap_only_without_tiles=True).entries) == 10
    s = probe_dir(str(other))
    assert s.links == 2 and s.other_files == 30     # links stay separate; hidden files don't count as visible


def test_hung_lane_does_not_delay_exit(tmp_path):
    script = textwrap.dedent(f"""
        import sys, time
        sys.path[:0] = [{str(REPO)!r}, {str(REPO / 'tests')!r}]
        from fake_fs import FakeOs
        from yfs.core.fsservice import FsService
        from yfs.core.places import MountTable
        fake = FakeOs()
        fake.block({str(tmp_path)!r})
        fs = FsService(lambda e: None, MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw"),
                       os_api=fake)
        fs.list_dir({str(tmp_path)!r}, ticket=fs.new_ticket(), owner="t")
        while not fake.inside:
            time.sleep(0.01)
        print("READY", flush=True)
        sys.exit(0)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "READY"
        t0 = time.monotonic()
        assert p.wait(timeout=10) == 0
        assert time.monotonic() - t0 < 1.0
    finally:
        p.kill()


def test_no_thread_pool_executor_in_fsservice():
    assert "ThreadPoolExecutor" not in Path(fsservice.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("name,code", [
    ("ENOENT", "not_found"), ("EACCES", "no_access"), ("EPERM", "no_access"), ("ENOTDIR", "not_a_dir"),
    ("ENOSPC", "no_space"), ("EDQUOT", "no_space"), ("EROFS", "read_only"), ("ENOTCONN", "unavailable"),
    ("ESTALE", "unavailable"), ("EIO", "io"), ("EHOSTDOWN", "unavailable"), ("ENETUNREACH", "unavailable"),
    ("ETIMEDOUT", "unavailable"), ("EEXIST", "io")])
def test_map_oserror_errno(name, code):
    eno = getattr(errno, name)
    err = map_oserror(OSError(eno, os.strerror(eno)))
    assert (err.code, err.errno) == (code, eno) and err.message.endswith(".")


@pytest.mark.parametrize("winerror,code", [
    (2, "not_found"), (3, "not_found"), (5, "no_access"), (19, "read_only"), (39, "no_space"), (112, "no_space"),
    (21, "not_ready"), (53, "unavailable"), (64, "unavailable"), (67, "unavailable"), (1222, "unavailable"),
    (1201, "disconnected"), (999, "io")])
def test_map_oserror_winerror(winerror, code):
    e = OSError("The network path was not found")
    e.winerror = winerror
    assert map_oserror(e).code == code and map_oserror(e).winerror == winerror


def test_a_drive_that_is_not_ready_and_a_read_error_are_not_network_trouble():
    """PL-13: an empty card reader (WinError 21) and EIO (a failing local disk as much as a share)."""
    e = OSError("The device is not ready")
    e.winerror = 21
    assert (map_oserror(e).code, map_oserror(e).message) == ("not_ready",
                                                             "The drive isn't ready — is a disc or card inserted?")
    err = map_oserror(OSError(errno.EIO, os.strerror(errno.EIO)))
    assert (err.code, err.message) == ("io", "The disk or share reported a read error.")
    assert friendly_error({"type": "OSError", "message": "[Errno 5] Input/output error", "errno": errno.EIO,
                           "winerror": None}) == "The disk or share reported a read error."


def test_paths_too_long_for_windows(monkeypatch):
    """PL-7: without LongPathsEnabled, Windows reports a path of 260+ characters as not found."""
    monkeypatch.setattr(fsservice, "WINDOWS", True)
    monkeypatch.setattr(fsservice, "windows_long_paths", lambda: False)
    long = "C:\\" + "\\".join(["x" * 50] * 5)                 # 3 + 5 * 50 + 4 = 257 characters
    longer = long + "\\abc"                                   # 261
    missing = FileNotFoundError(errno.ENOENT, "The system cannot find the path specified", longer)
    missing.winerror = 3
    err = map_oserror(missing)
    assert (err.code, err.message) == ("path_too_long", "The path is too long for Windows (261 characters).")
    assert map_oserror(FileNotFoundError(errno.ENOENT, "No such file", longer)).code == "path_too_long"   # open()
    assert map_oserror(FileNotFoundError(errno.ENOENT, "No such file", long + "\\a")).code == "not_found"  # 259
    assert map_oserror(FileNotFoundError(errno.ENOENT, "No such file", long + "\\ab")).code == "path_too_long"
    renamed = FileNotFoundError(errno.ENOENT, "No such file", "C:\\out\\a.ome.tif")
    renamed.filename2 = longer                                # os.replace(src, dst): the destination is long
    assert map_oserror(renamed).message == "The path is too long for Windows (261 characters)."
    named = OSError("The filename or extension is too long")
    named.winerror = 206
    assert (map_oserror(named).code, map_oserror(named).message) == ("path_too_long",
                                                                     "The path is too long for Windows.")
    monkeypatch.setattr(fsservice, "windows_long_paths", lambda: True)    # long paths work: it really is missing
    assert map_oserror(missing).code == "not_found" and map_oserror(named).code == "path_too_long"
    monkeypatch.setattr(fsservice, "WINDOWS", False)
    monkeypatch.setattr(fsservice, "windows_long_paths", lambda: False)
    assert map_oserror(FileNotFoundError(errno.ENOENT, "No such file", "/x" * 200)).code == "not_found"


def test_windows_long_paths_reads_the_registry_once(monkeypatch):
    reads = []

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(root, sub):
        reads.append(sub)
        if fake.missing:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return Key()

    fake = types.SimpleNamespace(HKEY_LOCAL_MACHINE=0x80000002, OpenKey=open_key, missing=False,
                                 QueryValueEx=lambda key, name: (1 if name == "LongPathsEnabled" else 0, 4))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(osutil, "WINDOWS", True)
    osutil.windows_long_paths.cache_clear()
    try:
        assert osutil.windows_long_paths() and osutil.windows_long_paths()
        assert reads == [r"SYSTEM\CurrentControlSet\Control\FileSystem"]
        osutil.windows_long_paths.cache_clear()
        fake.missing = True
        assert not osutil.windows_long_paths()
    finally:
        osutil.windows_long_paths.cache_clear()


def test_friendly_error():
    assert friendly_error(OSError(errno.EACCES, "Permission denied")) == "Permission denied."
    assert friendly_error({"type": "OSError", "message": "[Errno 28] No space", "errno": 28, "winerror": None,
                           "oserror": True}) == "The disk is full."
    assert friendly_error({"type": "ValueError", "message": "x" * 300, "errno": None}) == "x" * 120
    assert friendly_error(ValueError("bad tile\nsecond line")) == "bad tile"
    assert friendly_error(None) == "Unknown error."
    assert friendly_error(RuntimeError()) == "RuntimeError"


@pytest.mark.parametrize("reverse", [False, True])
def test_generic_calls_with_different_result_contracts_do_not_merge(make_fs, collector, tmp_path, reverse):
    fs, gate, entered = make_fs(), threading.Event(), threading.Event()
    make_fs.gates.append(gate)
    expected = [("dir", None, "mounts"), ("dir", str(tmp_path), None)]
    if reverse:
        expected.reverse()
    def first():
        entered.set()
        gate.wait(2)
        return expected[0]
    a = fs.call(str(tmp_path), "locate", first, ticket=0, owner="main")
    assert entered.wait(1)
    b = fs.call(str(tmp_path), "locate", lambda: expected[1], ticket=0, owner="browser")
    gate.set()
    assert done(collector, a).result == expected[0]
    assert done(collector, b).result == expected[1]


def test_recovery_cannot_overtake_committed_timeout(monkeypatch, collector, tmp_path):
    monkeypatch.setattr(FsService, "_watch", lambda self: None)
    gate, publishing, release = threading.Event(), threading.Event(), threading.Event()
    def post(ev):
        if isinstance(ev, FsSlow) and ev.past_deadline:
            publishing.set()
            release.wait(2)
        collector(ev)
    fs = FsService(post, table())
    try:
        rid = fs.call(str(tmp_path), "index_output", gate.wait, ticket=0, owner="main", deadline_s=0.01)
        until(lambda: any(l.running for l in fs._lanes.values()))
        with fs._lock:
            next(iter(fs._inflight.values())).started -= 1
        tick = threading.Thread(target=fs._tick)
        tick.start()
        assert publishing.wait(1)
        gate.set()
        until(lambda: fs.health(str(tmp_path)) == "ok")
        release.set()
        tick.join(1)
        evs = collector.events
        timeout = [i for i, e in enumerate(evs) if isinstance(e, FsDone) and e.req_id == rid]
        recovery = [i for i, e in enumerate(evs) if isinstance(e, LaneHealthChanged) and e.health == "ok"]
        assert len(timeout) == len(recovery) == 1
        assert evs[timeout[0]].error.code == "timeout" and timeout[0] < recovery[0]
    finally:
        gate.set()
        release.set()
        fs.shutdown()


def test_hung_extended_unc_keeps_other_servers_available(make_fs, collector):
    fs = make_fs(MountTable(flavor="win32"))
    gate = threading.Event()
    make_fs.gates.append(gate)
    a = r"\\?\UNC\nas-a.example.org\labshare\ProjectC"
    b = r"\\?\UNC\nas-b.example.org\labshare\ProjectC"
    rid = fs.call(a, "locate", gate.wait, ticket=0, owner="main", deadline_s=0.01)
    assert done(collector, rid).error.code == "timeout"
    assert done(collector, fs.call(b, "locate", lambda: "healthy", ticket=0, owner="browser")).result == "healthy"


def test_gvfs_backend_changes_publish_mount_refresh(make_fs, collector, monkeypatch, tmp_path):
    from yfs.core import places
    root = str(tmp_path / "gvfs")
    names = ["smb-share:server=nas.example.org,share=labshare"]
    monkeypatch.setattr(places, "_list_gvfs", lambda *a: list(names))
    base = table((root, "fuse.gvfsd-fuse", "gvfsd-fuse"))
    fs = make_fs(base.with_gvfs())
    names.clear()
    fs.set_mounts(base.with_gvfs())
    assert len(collector.of_type(MountsChanged)) == 1
    assert fs.mounts.unc_to_local(r"\\nas.example.org\labshare") is None
    names.append("smb-share:server=nas.example.org,share=labshare")
    fs.set_mounts(base.with_gvfs())
    assert len(collector.of_type(MountsChanged)) == 2
    assert fs.mounts.unc_to_local(r"\\nas.example.org\labshare") == root + "/" + names[0]
