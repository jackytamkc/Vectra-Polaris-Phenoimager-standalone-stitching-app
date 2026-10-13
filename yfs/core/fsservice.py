"""Filesystem lanes: per-volume daemon worker threads with priorities, deadlines, a watchdog and a listing cache.

FINAL_SPEC §5.5.3 (the lane mechanics there are normative) and the deadlines of §3.1. A hung network volume can
block only its own lane; the UI learns about it through FsSlow / LaneHealthChanged and new requests fail fast.
Daemon threads only (never a thread-pool executor): a stuck call must not keep the app from exiting.
"""
from __future__ import annotations

import errno as _errno
import heapq
import itertools
import logging
import os
import re
import shutil
import stat
import sys
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .events import FsDone, FsSlow, LaneHealthChanged, MountsChanged, Post
from .model import (CLOUD, H_DISCONNECTED, H_OK, H_SLOW, H_UNKNOWN, H_UNRESPONSIVE, LOCAL, NETWORK, DirSummary,
                    Entry, FsError, Listing)
from .names import BROWSER_LIST_CAP, PROBE_CAP, TILE_RE, WINDOWS_MAX_PATH, summarize_listing
from .osutil import windows_long_paths
from .places import MountTable

WINDOWS = sys.platform == "win32"
CONCURRENCY = {LOCAL: 4, NETWORK: 2, CLOUD: 1}
DEADLINE = {kind: dict(zip((LOCAL, NETWORK, CLOUD), secs)) for kind, secs in {    # §3.1, seconds
    "list_browser": (10, 20, 60), "list_discovery": (10, 30, 120), "probe": (5, 8, 30), "stat": (5, 10, 30),
    "readlink": (5, 10, 30), "usage": (10, 20, 30), "locate": (10, 20, 30), "mkdir": (10, 20, 30),
    "write_probe": (10, 20, 30), "header": (20, 30, 60), "check_output": (20, 40, 60), "index_output": (10, 30, 60),
    "delete": (30, 60, 60), "finalize": (30, 60, 60), "call": (30, 60, 60)}.items()}
_HIDDEN_ATTRS, _REPARSE, _NAME_SURROGATE, _OFFLINE_ATTRS = 0x2 | 0x4, 0x400, 0x20000000, 0x1000 | 0x40000 | 0x400000
_REAL_OS = SimpleNamespace(scandir=lambda p: os.scandir(p), stat=lambda p, **kw: os.stat(p, **kw),
                           disk_usage=lambda p: shutil.disk_usage(p), readlink=lambda p: os.readlink(p))
_MESSAGES = {"not_found": "It doesn't exist.", "no_access": "Permission denied.", "not_a_dir": "It is not a folder.",
             "no_space": "The disk is full.", "read_only": "The disk is read-only.",
             "unavailable": "The network location isn't available.", "disconnected": "The drive is disconnected.",
             "not_ready": "The drive isn't ready — is a disc or card inserted?",
             "unresponsive": "The location isn't answering.", "cancelled": "Cancelled."}
_BY_ERRNO = [(frozenset(getattr(_errno, n) for n in names if hasattr(_errno, n)), code) for names, code in (
    (("ENOENT",), "not_found"), (("EACCES", "EPERM"), "no_access"), (("ENOTDIR",), "not_a_dir"),
    (("ENOSPC", "EDQUOT"), "no_space"), (("EROFS",), "read_only"),
    (("ENOTCONN", "ESTALE", "EHOSTDOWN", "ENETUNREACH", "ETIMEDOUT"), "unavailable"))]
_BY_WINERROR = {2: "not_found", 3: "not_found", 5: "no_access", 19: "read_only", 39: "no_space", 112: "no_space",
                21: "not_ready", 53: "unavailable", 64: "unavailable", 67: "unavailable", 1222: "unavailable",
                1201: "disconnected"}
_EIO_TEXT = "The disk or share reported a read error."     # EIO: a failing disk as much as a network share


class FsFailure(Exception):
    """Raised by futures of failed requests; .error is an FsError (plain data)."""

    def __init__(self, error: FsError):
        super().__init__(error.message)
        self.error = error


def _error(code: str, eno: Optional[int] = None, winerror: Optional[int] = None, text: str = "") -> FsError:
    return FsError(code, _MESSAGES.get(code) or text[:120] or "Something went wrong.", eno, winerror)


def map_oserror(e: OSError) -> FsError:
    """errno / winerror -> FsError code and a plain sentence (§5.5.3 table). Windows: WinError 206, or a path of
    260+ characters not found (WinError 3, ENOENT, ENAMETOOLONG) without LongPathsEnabled, is 'path_too_long'."""
    winerror = getattr(e, "winerror", None)
    n = max((len(os.fsdecode(f)) for f in (e.filename, e.filename2) if isinstance(f, (str, bytes, os.PathLike))),
            default=0)
    if winerror == 206 or (n > WINDOWS_MAX_PATH and WINDOWS and not windows_long_paths()
                           and (winerror == 3 or e.errno in (_errno.ENOENT, _errno.ENAMETOOLONG))):
        return FsError("path_too_long", f"The path is too long for Windows ({n} characters)." if n else
                       "The path is too long for Windows.", e.errno, winerror)
    return _map_codes(e.errno, winerror, e.strerror or str(e))


def _map_codes(eno: Optional[int], winerror: Optional[int], text: str) -> FsError:
    code = _BY_WINERROR.get(winerror) or next((c for codes, c in _BY_ERRNO if eno in codes), "io")
    if code == "io" and eno == _errno.EIO:
        text = _EIO_TEXT
    return _error(code, eno, winerror, text.strip().rstrip(".") + "." if text.strip() else "")


def friendly_error(x: Union[BaseException, Mapping[str, Any], None]) -> str:
    """One plain sentence for a failure: an exception or the engine's last_error dict. Text only."""
    if isinstance(x, OSError):
        return map_oserror(x).message
    if isinstance(x, Mapping) and (x.get("errno") is not None or x.get("winerror") is not None):
        return _map_codes(x.get("errno"), x.get("winerror"), str(x.get("message") or "")).message
    if isinstance(x, Mapping):
        text = str(x.get("message") or x.get("type") or "")
    else:
        text = "" if x is None else (str(x) or type(x).__name__)
    return text.strip().splitlines()[0][:120] if text.strip() else "Unknown error."


def _scan(api: Any, path: str, cap: int, cap_only_without_tiles: bool, stat_self: bool) -> Listing:
    t0, entries, truncated, seen_tile = time.monotonic(), [], False, False
    with api.scandir(path) as it:
        for de in it:
            if len(entries) >= cap and not (cap_only_without_tiles and seen_tile):
                truncated = True
                break
            name, size, mtime, offline, hidden = de.name, None, None, False, de.name.startswith(".")
            try:
                is_dir, is_link = de.is_dir(follow_symlinks=False), de.is_symlink()
            except OSError:
                is_dir = is_link = False
            if WINDOWS:                              # free on Windows: it comes with the directory listing
                try:
                    st = de.stat(follow_symlinks=False)
                    attrs, size, mtime = st.st_file_attributes, st.st_size, st.st_mtime
                    hidden, offline = hidden or bool(attrs & _HIDDEN_ATTRS), bool(attrs & _OFFLINE_ATTRS)
                    # junctions and symlinks only; OneDrive placeholders are other reparse points
                    is_link = is_link or bool(attrs & _REPARSE and st.st_reparse_tag & _NAME_SURROGATE)
                except OSError:
                    pass
            seen_tile = seen_tile or TILE_RE.match(name) is not None
            entries.append(Entry(name, is_dir, is_link, size, mtime, hidden, offline))
    mtime, now = api.stat(path).st_mtime if stat_self else None, time.monotonic()
    return Listing(path, tuple(entries), truncated, now - t0, now, mtime)


def scan_dir(path: str, *, cap: int, cap_only_without_tiles: bool = False, stat_self: bool = False) -> Listing:
    """One os.scandir: names, is_dir(follow_symlinks=False), is_symlink(); on Windows also size, mtime and attributes.
    Stops at `cap` entries (with cap_only_without_tiles: only while no tile name has been seen)."""
    return _scan(_REAL_OS, path, cap, cap_only_without_tiles, stat_self)


def probe_dir(path: str, cap: int = PROBE_CAP) -> DirSummary:
    """scan_dir + names.summarize_listing."""
    lst = scan_dir(path, cap=cap, cap_only_without_tiles=True)
    return summarize_listing(path, lst.entries, truncated=lst.truncated)


class _Req:
    __slots__ = ("key", "kind", "path", "ticket", "fn", "prio", "deadline", "waiters", "state", "queued", "started",
                 "slow", "late", "lane")

    def __init__(self, key, kind, path, ticket, fn, prio, deadline, waiter, lane):
        self.key, self.kind, self.path, self.ticket, self.fn = key, kind, path, ticket, fn
        self.prio, self.deadline, self.waiters = prio, deadline, [waiter]   # waiter: (id, owner, tag, notify, fut)
        self.state, self.queued, self.started, self.slow, self.late = "queued", time.monotonic(), 0.0, False, False
        self.lane = lane                              # where it is queued, even if a newer mount table moves the path


class _Lane:
    def __init__(self, key: str, cls: str, lock: threading.Lock):
        self.key, self.cls, self.cv = key, cls, threading.Condition(lock)
        self.heap: List[Tuple[int, int, _Req]] = []
        self.running: Dict[_Req, threading.Thread] = {}
        self.stuck: set = set()                      # abandoned threads still inside a call
        self.threads = self.idle = self.passes = 0
        self.health, self.since, self.disconnected = H_UNKNOWN, 0.0, False


class FsService:
    """Per-volume lanes of daemon worker threads. Thread-safe; every public method returns immediately.
    Emits FsDone (notify=True requests), FsSlow, LaneHealthChanged and MountsChanged."""
    SLOW_S = 2.5
    IDLE_S = 60.0
    RELOAD_S = 2.0                                    # reload_mounts() reads the mount table at most this often

    def __init__(self, post: Post, mounts: MountTable, *, max_threads: int = 32, os_api: Optional[Any] = None):
        """os_api (tests only) replaces the os/shutil calls of the built-in request kinds."""
        self._mounts, self._post, self._max, self._os = mounts, post, max_threads, os_api or _REAL_OS
        self._lock = threading.Lock()
        self._publications, self._draining = deque(), False
        self._lanes: Dict[str, _Lane] = {}
        self._inflight: Dict[tuple, _Req] = {}
        self._cancelled: "OrderedDict[int, None]" = OrderedDict()
        self._cache: "OrderedDict[str, list]" = OrderedDict()   # normcase(path) -> [listing, at, summary, at]
        self._ids, self._tickets, self._seq = itertools.count(1), itertools.count(1), itertools.count()
        self._nthreads, self._closed = 0, threading.Event()
        self._reloading = self._reload_again = False
        for m in mounts.mounts():                     # Windows drives already known to be disconnected (1201)
            if mounts.disconnected(m.mount_point):
                self.mark_disconnected(m.mount_point, True)
        threading.Thread(target=self._watch, name="yfs-fs-watchdog", daemon=True).start()

    @property
    def mounts(self) -> MountTable:
        """The current mount table (set_mounts / reload_mounts replace it). Read it once per use."""
        return self._mounts

    def set_mounts(self, table: MountTable) -> None:
        """Use a newer mount table. Lanes of unchanged volumes are kept (lanes are made on first use); a drive the
        new table reports disconnected (or no longer does) is marked so. Posts MountsChanged if it differs."""
        with self._lock:
            old, self._mounts = self._mounts, table
        if table == old:
            return
        for mp in {m.mount_point for m in old.mounts() + table.mounts()}:      # Windows 1201, reported or cleared
            if table.disconnected(mp) != old.disconnected(mp):
                self.mark_disconnected(mp, table.disconnected(mp))
        self._post(MountsChanged(time.time()))

    def reload_mounts(self) -> None:
        """Read the mount table again on a daemon thread and set_mounts() it: at most once per RELOAD_S, and a
        call in between is served at the end of that interval. A table that can't be read is not used."""
        with self._lock:
            if self._reloading:
                self._reload_again = True
                return
            self._reloading = True
        threading.Thread(target=self._reload, name="yfs-mounts", daemon=True).start()

    # ---- asynchronous requests (results arrive as FsDone)
    def new_ticket(self) -> int:
        return next(self._tickets)

    def cancel_ticket(self, ticket: int) -> None:
        """Queued requests with this ticket are dropped when dequeued (their futures are cancelled)."""
        with self._lock:
            self._cancelled[ticket] = None
            while len(self._cancelled) > 1000:
                self._cancelled.popitem(last=False)

    def list_dir(self, path: str, *, ticket: int, owner: str, tag: Any = None, priority: int = 0,
                 cap: int = BROWSER_LIST_CAP, user_initiated: bool = True) -> int:
        fn = partial(self.scan, path, cap=cap, stat_self=True)
        return self._request(path, "list", fn, ticket, owner, tag, priority, None, user_initiated, True)

    def probe(self, path: str, *, ticket: int, owner: str, tag: Any = None, priority: int = 5) -> int:
        fn = partial(self.scan, path, cap=PROBE_CAP, cap_only_without_tiles=True)
        return self._request(path, "probe", fn, ticket, owner, tag, priority, None, False, True)

    def stat_names(self, path: str, names: Sequence[str], *, ticket: int, owner: str, priority: int = 8) -> int:
        """-> {name: (size, mtime, is_dir)} from os.lstat. Links are not followed (readlink() resolves them), and
        names on another lane (mount points of other volumes) are left out: this lane never waits for them."""
        m = self._mounts
        lane = m.lane_key(path)
        names = tuple(n for n in names if m.lane_key(os.path.join(path, n)) == lane)
        return self._request(path, "stat", partial(self._stat_names, path, names), ticket, owner, None, priority,
                             None, False, True, extra=names)

    def readlink(self, path: str, *, ticket: int, owner: str, tag: Any = None, priority: int = 4) -> int:
        """-> the immediate target of the link `path` (absolute and normalised), or None if it is not a link.
        os.lstat and os.readlink run in the lane of the folder holding the link: the target is not touched."""
        path = os.path.normpath(path)                 # with a trailing separator, lstat would follow the link
        return self._request(path, "readlink", partial(self._readlink, path), ticket, owner, tag, priority, None,
                             False, True, where=os.path.dirname(path))

    def disk_usage(self, path: str, *, ticket: int, owner: str, tag: Any = None, priority: int = 3) -> int:
        return self._request(path, "usage", partial(self._os.disk_usage, path), ticket, owner, tag, priority, None,
                             False, True)

    def call(self, path: str, kind: str, fn: Callable[[], Any], *, ticket: int, owner: str, tag: Any = None,
             priority: int = 2, deadline_s: Optional[float] = None, user_initiated: bool = False,
             notify: bool = True) -> int:
        """Run fn() on the lane of `path`. OSError -> FsError via map_oserror(); other exceptions -> FsError('io')."""
        return self._request(path, kind, fn, ticket, owner, tag, priority, deadline_s, user_initiated, notify,
                             extra=id(fn))

    # ---- synchronous use from worker threads (never from the Tk thread)
    def submit(self, path: str, kind: str, fn: Callable[[], Any], *, ticket: int = 0, owner: str = "",
               priority: int = 2, deadline_s: Optional[float] = None, notify: bool = False) -> Future:
        """A concurrent.futures.Future (the class only): the result, or FsFailure."""
        fut: Future = Future()
        self._request(path, kind, fn, ticket, owner, None, priority, deadline_s, False, notify, fut=fut, extra=id(fn))
        return fut

    def scan(self, path: str, *, cap: int, cap_only_without_tiles: bool = False, stat_self: bool = False) -> Listing:
        """scan_dir() through this service's os_api. Runs in the calling thread: use it inside a lane."""
        return _scan(self._os, path, cap, cap_only_without_tiles, stat_self)

    # ---- state (no I/O)
    def health(self, path_or_lane: str) -> str:
        return self.health_snapshot(path_or_lane)[0]

    def health_snapshot(self, path_or_lane: str) -> Tuple[str, float]:
        """Current health and transition time, so deferred consumers can reject obsolete notifications."""
        with self._lock:
            lane = self._find(path_or_lane)
            return (lane.health, lane.since) if lane else (H_UNKNOWN, 0.0)

    def lane_class(self, path: str) -> str:
        return self._mounts.lane_class(path)

    def auto_enter(self, path: str) -> Optional[str]:
        """Why `path` must not be listed or probed automatically ('pseudo'|'autofs'|'disconnected'|'cloud'), or None.
        'disconnected' lasts while the lane is marked so: a call that answers clears it (the user reconnected the
        drive). Consumers (discovery, the browser) use this, not mounts.auto_enter."""
        m = self._mounts
        with self._lock:
            lane = self._lanes.get(m.lane_key(path))
            if lane is not None and lane.disconnected:
                return "disconnected"
        why = m.auto_enter(path)
        return None if why == "disconnected" else why

    def lane_key(self, path: str) -> str:
        return self._mounts.lane_key(path)

    def concurrency(self, path: str) -> int:
        """local 4, network 2, cloud 1."""
        return CONCURRENCY[self._mounts.lane_class(path)]

    def retry(self, path_or_lane: str) -> None:
        """Allow exactly one request through an unresponsive lane. While no thread of the lane is free to run it,
        requests keep failing at once and the pass is kept."""
        with self._lock:
            lane = self._find(path_or_lane)
            if lane is not None and lane.health == H_UNRESPONSIVE:
                lane.passes = 1

    def mark_disconnected(self, lane: str, on: bool) -> None:
        """Windows 1201 (from places): non-user-initiated requests then fail with FsError('disconnected')."""
        with self._lock:
            m = self._mounts
            ln = self._find(lane) or self._lane(m.lane_key(lane), m.lane_class(lane))
            ln.disconnected, ev = on, None
            if on or ln.health == H_DISCONNECTED:     # clearing it keeps a state the lane has earned since
                ev = self._set_health(ln, H_DISCONNECTED if on else H_UNKNOWN)
            self._queue_event(ev)
        self._drain()

    def cached_listing(self, path: str, max_age: Optional[float] = None) -> Optional[Listing]:
        with self._lock:
            e = self._cache_get(path)
            return e[0] if e and e[0] is not None and _fresh(e[1], max_age) else None

    def cached_summary(self, path: str, max_age: Optional[float] = None) -> Optional[DirSummary]:
        """A probe result, or the summary of a cached listing."""
        with self._lock:
            e = self._cache_get(path)
            return e[2] if e and e[2] is not None and _fresh(e[3], max_age) else None

    def invalidate(self, path: str) -> None:
        """Forget a folder's cached listing and summary, and its parent's summary."""
        with self._lock:
            self._cache.pop(os.path.normcase(path), None)
            parent = self._cache.get(os.path.normcase(os.path.dirname(path.rstrip("/\\")) or path))
            if parent is not None:
                parent[2] = None

    def has_stuck_threads(self) -> bool:
        with self._lock:
            return any(lane.stuck for lane in self._lanes.values())

    def shutdown(self) -> None:
        """Stop accepting work; idle workers exit, stuck daemon threads die with the process."""
        with self._lock:
            self._closed.set()
            for lane in self._lanes.values():
                lane.cv.notify_all()

    # ---- internals (the lock guards lanes, requests, the cache and the counters)
    def _find(self, path_or_lane: str) -> Optional[_Lane]:
        return self._lanes.get(path_or_lane) or self._lanes.get(self._mounts.lane_key(path_or_lane))

    def _lane(self, key: str, cls: str) -> _Lane:
        if key not in self._lanes:
            self._lanes[key] = _Lane(key, cls, self._lock)
        lane = self._lanes[key]
        lane.cls = cls                                # a newer mount table may know the volume better
        return lane

    def _forget(self, req: _Req) -> None:
        if self._inflight.get(req.key) is req:
            del self._inflight[req.key]

    def _request(self, path, kind, fn, ticket, owner, tag, prio, deadline_s, user, notify, fut=None, extra=None,
                 where=None):
        """`where`: a path whose lane runs the request (default: `path`)."""
        m = self._mounts
        rid, key, cls = next(self._ids), m.lane_key(where or path), m.lane_class(where or path)
        waiter = (rid, owner, tag, notify, fut)
        deadline = deadline_s or DEADLINE.get("list_browser" if kind == "list" else kind, DEADLINE["call"])[cls]
        with self._lock:
            lane = self._lane(key, cls)
            err = self._refusal(lane, user)
            req = self._inflight.get((kind, path, ticket, extra)) if err is None else None
            if req is not None:                      # duplicate of a queued or running request: share its result
                req.waiters.append(waiter)
                if req.state == "queued" and prio < req.prio:
                    req.prio = prio
                    heapq.heappush(req.lane.heap, (prio, next(self._seq), req))
            elif err is None:
                req = self._inflight[(kind, path, ticket, extra)] = _Req((kind, path, ticket, extra), kind, path,
                                                                         ticket, fn, prio, deadline, waiter, lane)
                heapq.heappush(lane.heap, (prio, next(self._seq), req))
                if lane.idle:
                    lane.cv.notify()
                if len(lane.heap) > lane.idle:        # a notified worker counts as idle until it wakes up
                    self._spawn(lane)
            if err is not None:
                self._queue_delivery([waiter], kind, path, ticket, None, err, 0.0)
        self._drain()
        return rid

    def _refusal(self, lane: _Lane, user: bool) -> Optional[FsError]:
        if self._closed.is_set():
            return _error("cancelled")
        if lane.disconnected and not user:
            return _error("disconnected")
        if lane.health == H_UNRESPONSIVE:
            full = lane.threads >= CONCURRENCY[lane.cls] + 1 and lane.threads <= len(lane.stuck)
            if not lane.passes or full:               # full: every thread it may have is stuck, keep the pass
                return _error("unresponsive")
            lane.passes -= 1                          # retry(): exactly one request goes through
        return None

    def _spawn(self, lane: _Lane) -> None:
        conc = CONCURRENCY[lane.cls]
        # stuck threads do not count towards concurrency, but a hung lane never holds more than conc + 1
        if lane.threads - len(lane.stuck) >= conc or lane.threads >= conc + 1:
            return
        if self._nthreads >= self._max:               # wake an idle worker elsewhere: it hands its slot over
            for other in self._lanes.values():
                if other.idle:
                    other.cv.notify()
                    break
            return
        lane.threads, self._nthreads = lane.threads + 1, self._nthreads + 1
        threading.Thread(target=self._work, args=(lane,), name=f"yfs-fs-{lane.cls}", daemon=True).start()

    def _pop(self, lane: _Lane, dropped: List[_Req]) -> Optional[_Req]:
        while lane.heap:
            prio, _, req = heapq.heappop(lane.heap)
            if req.state != "queued" or prio != req.prio:
                continue                              # stale entry (re-prioritised or already failed)
            if req.ticket not in self._cancelled:
                return req
            req.state = "cancelled"
            self._forget(req)
            dropped.append(req)
        return None

    def _work(self, lane: _Lane) -> None:
        try:
            self._serve(lane)
        finally:                                      # also after an unexpected error: free the thread slot
            with self._lock:
                lane.threads, self._nthreads = lane.threads - 1, self._nthreads - 1
                lane.stuck.discard(threading.current_thread())
                for other in self._lanes.values():    # a lane starved by max_threads gets the freed slot
                    if other.heap and not other.idle:
                        self._spawn(other)

    def _serve(self, lane: _Lane) -> None:
        me = threading.current_thread()
        while True:
            dropped: List[_Req] = []
            with self._lock:
                req = self._pop(lane, dropped)
                while req is None and not dropped and not self._closed.is_set():
                    if self._nthreads >= self._max and any(o.heap and not o.idle for o in self._lanes.values()
                                                           if o is not lane):
                        break                         # hand the slot to a lane that is waiting for a thread
                    lane.idle += 1
                    woke = lane.cv.wait(self.IDLE_S)
                    lane.idle -= 1
                    req = self._pop(lane, dropped)
                    if req is None and not woke:
                        break
                if req is None and not dropped:
                    return
                if req is not None:
                    req.state, req.started, lane.running[req] = "running", time.monotonic(), me
            for w in (w for d in dropped for w in d.waiters if w[4] is not None):
                w[4].cancel()
            if req is None:
                continue
            result, err = self._run(req)
            with self._lock:
                del lane.running[req]
                stuck, req.state, waiters, req.waiters = me in lane.stuck, "done", req.waiters, []
                lane.stuck.discard(me)
                self._forget(req)
                ev = self._settle(lane, err)
                self._queue_event(ev)
                self._queue_delivery(waiters, req.kind, req.path, req.ticket, result, err,
                                     time.monotonic() - req.started)
            self._drain()
            if stuck:                                 # an abandoned thread does not rejoin the lane
                return

    def _run(self, req: _Req) -> Tuple[Any, Optional[FsError]]:
        try:
            result = req.fn()
        except OSError as e:
            return None, map_oserror(e)
        except Exception as e:                       # keep text only, never the exception object
            return None, _error("io", text=friendly_error(e))
        if req.kind in ("list", "probe") and isinstance(result, Listing):
            summary = summarize_listing(req.path, result.entries, truncated=result.truncated)
            with self._lock:
                self._store(req.path, result if req.kind == "list" else None, summary)
            result = result if req.kind == "list" else summary
        return result, None

    def _deliver(self, waiters, kind, path, ticket, result, err, elapsed) -> None:
        for rid, owner, tag, notify, fut in waiters:
            if fut is None:
                if notify:
                    self._post(FsDone(rid, kind, path, ticket, owner, tag, result, err, elapsed))
            elif not fut.set_running_or_notify_cancel():
                continue                              # cancelled by its owner
            elif err is not None:
                fut.set_exception(FsFailure(err))
            else:
                fut.set_result(result)

    def _queue_event(self, ev) -> None:
        if ev is not None:
            self._publications.append((self._post, (ev,)))

    def _queue_delivery(self, *args) -> None:
        self._publications.append((self._deliver, args))

    def _drain(self) -> None:
        """One publisher, in state-transition order. Callbacks run without the service lock."""
        with self._lock:
            if self._draining:
                return
            self._draining = True
        try:
            while True:
                with self._lock:
                    if not self._publications:
                        self._draining = False
                        return
                    fn, args = self._publications.popleft()
                fn(*args)
        except BaseException:
            with self._lock:
                self._draining = False
            raise

    def _set_health(self, lane: _Lane, health: str) -> Optional[LaneHealthChanged]:
        if lane.health == health:
            return None
        lane.health, lane.since = health, time.time()
        return LaneHealthChanged(lane.key, health, lane.since)

    def _settle(self, lane: _Lane, err: Optional[FsError]) -> Optional[LaneHealthChanged]:
        """A call returned: the lane answers again (slow while other calls are still slow). A disconnected drive
        that answers was reconnected (by the user opening it)."""
        if err is not None and err.code in ("unavailable", "disconnected"):
            return None
        lane.disconnected, lane.passes = False, 0      # a pass granted while it was hung is not needed now
        return self._set_health(lane, H_SLOW if any(r.slow for r in lane.running) else H_OK)

    def _watch(self) -> None:
        while not self._closed.wait(0.25):
            try:
                self._tick()
            except Exception:                         # keep watching: deadlines protect every lane
                logging.getLogger(__name__).exception("FsService watchdog")

    def _tick(self) -> None:
        """One watchdog pass: post FsSlow; at a deadline time the request out, mark the lane unresponsive and fail
        its queue; time out queued requests that no thread is left to run."""
        now, events, fails = time.monotonic(), [], []
        with self._lock:
            for lane in self._lanes.values():
                for req, thread in list(lane.running.items()):
                    elapsed = now - req.started
                    if not req.slow and elapsed >= self.SLOW_S:
                        req.slow = True
                        events += [self._slow(req, w, elapsed, False) for w in req.waiters]
                        if lane.health in (H_UNKNOWN, H_OK):
                            events.append(self._set_health(lane, H_SLOW))
                    if req.late or elapsed < req.deadline:
                        continue
                    req.late = True
                    events += [self._slow(req, w, elapsed, True) for w in req.waiters]
                    events.append(self._set_health(lane, H_UNRESPONSIVE))
                    lane.stuck.add(thread)            # abandoned: a retry runs as a new request
                    self._forget(req)
                    # the deadline is the answer, except for a listing's notify waiters: the browser can keep
                    # waiting for it (its late result still fills the cache either way)
                    keep = [w for w in req.waiters if req.kind == "list" and w[4] is None]
                    fails.append((req, [w for w in req.waiters if w not in keep], self._timeout(req.path)))
                    req.waiters = keep
                    for q in (q for _, _, q in lane.heap if q.state == "queued"):       # fail fast
                        q.state = "failed"
                        self._forget(q)
                        fails.append((q, q.waiters, _error("unresponsive")))
                    lane.heap = []
                if lane.heap and lane.threads <= len(lane.stuck):    # no thread will run these (thread caps)
                    for q in (q for _, _, q in lane.heap if q.state == "queued" and now - q.queued >= q.deadline):
                        q.state = "failed"
                        self._forget(q)
                        fails.append((q, q.waiters, self._timeout(q.path)))
                    lane.heap = [e for e in lane.heap if e[2].state == "queued"]
                    heapq.heapify(lane.heap)
            for ev in events:
                self._queue_event(ev)
            for req, waiters, err in fails:
                self._queue_delivery(waiters, req.kind, req.path, req.ticket, None, err,
                                     now - (req.started or req.queued))
        self._drain()

    def _timeout(self, path: str) -> FsError:
        return FsError("timeout", f"{self._mounts.place_name(path)} didn't answer.")

    @staticmethod
    def _slow(req: _Req, w: tuple, elapsed: float, past: bool) -> FsSlow:
        return FsSlow(w[0], req.kind, req.path, req.ticket, w[1], w[2], elapsed, past)

    def _stat_names(self, path: str, names: Tuple[str, ...]) -> Dict[str, Tuple[int, float, bool]]:
        out = {}
        for n in names:
            try:
                st = self._os.stat(os.path.join(path, n), follow_symlinks=False)
                out[n] = (st.st_size, st.st_mtime, stat.S_ISDIR(st.st_mode))
            except OSError:
                pass                                  # gone since the listing: no facts for it
        return out

    def _readlink(self, path: str) -> Optional[str]:
        st = self._os.stat(path, follow_symlinks=False)
        attrs, tag = getattr(st, "st_file_attributes", 0), getattr(st, "st_reparse_tag", 0)
        if not (stat.S_ISLNK(st.st_mode) or attrs & _REPARSE and tag & _NAME_SURROGATE):   # also junctions
            return None
        target = self._os.readlink(path)
        return os.path.normpath(os.path.join(os.path.dirname(path), _from_nt(target) if WINDOWS else target))

    def _reload(self) -> None:
        while not self._closed.is_set():
            table = MountTable.load()
            if table.mounts():                        # empty: the table could not be read in time
                self.set_mounts(table)
            self._closed.wait(self.RELOAD_S)
            with self._lock:
                if not self._reload_again:
                    self._reloading = False
                    return
                self._reload_again = False

    def _cache_get(self, path: str) -> Optional[list]:
        e = self._cache.get(os.path.normcase(path))
        if e is not None:
            self._cache.move_to_end(os.path.normcase(path))
        return e

    def _store(self, path: str, listing: Optional[Listing], summary: Optional[DirSummary]) -> None:
        k, now = os.path.normcase(path), time.monotonic()
        e = self._cache.pop(k, None) or [None, 0.0, None, 0.0]
        if listing is not None:
            e[0], e[1] = listing, listing.listed_at
        e[2], e[3] = summary, now
        self._cache[k] = e
        while len(self._cache) > 2000:
            self._cache.popitem(last=False)


def _fresh(at: float, max_age: Optional[float]) -> bool:
    return max_age is None or time.monotonic() - at <= max_age


def _from_nt(target: str) -> str:
    """A Windows link target without the NT prefix of os.readlink: \\\\?\\C:\\x -> C:\\x, \\\\?\\UNC\\server\\share
    -> \\\\server\\share. Other \\\\?\\ targets keep it: \\\\?\\Volume{GUID}\\ (a disk mounted in a folder) is a
    name of its own, and without the prefix it would read as a relative path."""
    if target.startswith("\\\\?\\UNC\\"):
        return "\\" + target[7:]
    return target[4:] if re.match(r"\\\\\?\\[A-Za-z]:", target) else target
