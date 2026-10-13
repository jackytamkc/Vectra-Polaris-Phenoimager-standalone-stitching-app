"""Entry point, UI event bus, app context and the main window (FINAL_SPEC §2.1, §2.4, §2.5, §5.2, §5.5.10).

MainWindow owns the root's widgets and the state machine EMPTY -> CHECKING -> SCANNING <-> PAUSED -> READY ->
RUNNING -> STOPPING -> FINISHED. It wires the folder fields, discovery, the plan, the preflight lane calls and the
batch runner together; every filesystem access goes through FsService lanes or worker threads, never the Tk thread.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import heapq
import itertools
import logging
import ntpath
import os
import posixpath
import re
import stat
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from functools import partial
from queue import Empty, SimpleQueue
from tkinter import ttk
from types import SimpleNamespace
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import psutil

from yfs import APP_NAME, __version__
from yfs.core import osutil, preflight
from yfs.core.discovery import Discovery, DiscoveryOptions
from yfs.core.events import (COALESCE_KEY, BackgroundError, CandidateFound, FsDone, FsSlow, LaneHealthChanged, LogLine,
                             MountsChanged, OutputWritten, RowBytes, RowFinished, RowProgress, RowStarted, RunFinished,
                             RunStarted, ScanFinished, ScanNote, ScanPaused, ScanProgress)
from yfs.core.fsservice import FsFailure, FsService, friendly_error, map_oserror
from yfs.core.model import CLOUD, H_OK, H_SLOW, NETWORK, FsError, Issue, Note, OutputIndex, RunSummary, TileHeader
from yfs.core.names import (DEEPER_STEP, OME_SUFFIX, PARTIAL_DIR, RECENT_OUTPUT_S, TILE_RE, human_bytes, norm_key,
                            rel_folder, same_path)
from yfs.core.places import MountTable, normalize_user_path, tcc_protected
from yfs.core.plan import STITCHED, Plan, PlanEnv
from yfs.core.runner import BatchRunner, RunOptions
from yfs.core.settings import Settings, log_dir
from yfs.ui import dialogs
from yfs.ui.browser import FolderBrowser, _link_start, browser_initial
from yfs.ui.icons import IconSet
from yfs.ui.planview import PlanTable
from yfs.ui.runpanel import LogDrawer, ProgressModel, RunBar, fit_label, primary_issue, problem_lines, totals_text
from yfs.ui.theme import TOKENS as T, Theme, auto_scale, detect_xft, install_indicators, install_theme, ui_thread
from yfs.ui.widgets import PathField

__all__ = ["AppContext", "MainWindow", "UiBus", "main", "restore_geometry", "ui_thread"]
log = logging.getLogger(__name__)


class UiBus:
    """Thread-safe event queue drained on the Tk thread.
    Every interval_ms the pump moves queued events into an ordered pending list, keeping only the latest event per
    COALESCE_KEY, then calls the handlers of each event's exact type until budget_ms is used up; whatever is left
    waits for the next tick (and keeps coalescing meanwhile)."""

    def __init__(self, root: tk.Misc, interval_ms: int = 50, budget_ms: int = 30) -> None:
        self.root, self.interval_ms, self.budget_s = root, interval_ms, budget_ms / 1000.0
        self._queue: "SimpleQueue[object]" = SimpleQueue()
        self._pending: Deque[List[Any]] = deque()        # [event] boxes, emptied when a newer event replaces them
        self._latest: Dict[Any, List[Any]] = {}          # coalesce key -> its box in _pending
        self._handlers: Dict[type, List[Callable[[Any], None]]] = {}
        self._hooks: List[Callable[[], None]] = []
        self._after: Optional[str] = None
        self._due = 0.0
        self.lateness: Deque[float] = deque(maxlen=2000)  # seconds each tick started after it was due (§6.5 D5)
        root.bind("<Destroy>", lambda e: e.widget is root and self.stop(), add="+")   # no tick into a dead root

    def post(self, ev: object) -> None:
        """Any thread; never blocks."""
        self._queue.put(ev)

    @ui_thread
    def subscribe(self, ev_type: type, handler: Callable[[Any], None]) -> Callable[[], None]:
        """Main thread. Returns an unsubscribe function. Handlers must not block."""
        self._handlers.setdefault(ev_type, []).append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(ev_type, [])
            if handler in handlers:
                handlers.remove(handler)
        return unsubscribe

    @ui_thread
    def add_tick_hook(self, fn: Callable[[], None]) -> None:
        """Main thread. fn() runs once after every pump tick that delivered at least one event."""
        self._hooks.append(fn)

    @ui_thread
    def start(self) -> None:
        if self._after is None:
            self._schedule()

    @ui_thread
    def stop(self) -> None:
        if self._after is not None:
            try:
                self.root.after_cancel(self._after)
            except tk.TclError:                           # the root is already gone
                pass
            self._after = None

    def _schedule(self) -> None:
        self._due = time.perf_counter() + self.interval_ms / 1000.0
        self._after = self.root.after(self.interval_ms, self._pump)

    def _pump(self) -> None:
        start = time.perf_counter()
        self.lateness.append(max(0.0, start - self._due))
        try:
            self._tick(start + self.budget_s)
        finally:
            self._schedule()

    def _tick(self, deadline: float) -> None:
        drain_until = deadline - self.budget_s / 2        # 1. drain the queue, coalescing as we go (half the budget,
        while time.perf_counter() < drain_until:          #    so a flood of events cannot starve the delivery)
            try:
                ev = self._queue.get_nowait()
            except Empty:
                break
            box = [ev]
            key_fn = COALESCE_KEY.get(type(ev))
            if key_fn is not None:
                key = key_fn(ev)
                older = self._latest.get(key)
                if older is not None:
                    older[0] = None                       # an older event with the same key is dropped
                self._latest[key] = box
            self._pending.append(box)
        delivered = 0
        while self._pending and time.perf_counter() < deadline:   # 2. deliver within the budget
            ev = self._pending.popleft()[0]
            if ev is None:
                continue
            key_fn = COALESCE_KEY.get(type(ev))
            if key_fn is not None:
                self._latest.pop(key_fn(ev), None)
            for handler in tuple(self._handlers.get(type(ev), ())):
                self._call(handler, ev)
            delivered += 1
        if delivered:
            for hook in tuple(self._hooks):
                self._call(hook)

    def _call(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:                                 # a broken handler must not stop the pump
            self.root.report_callback_exception(*sys.exc_info())


@dataclass
class AppContext:
    root: tk.Tk
    theme: Theme
    icons: IconSet
    bus: UiBus
    fs: FsService
    mounts: MountTable
    settings: Settings
    engine_cls: type
    output_index: Callable[[], Optional[OutputIndex]]        # current output index (browser badges)
    planned_output_keys: Callable[[], FrozenSet[str]]         # browser output-mode "2 of your 6 slides..."
    running: Callable[[], bool]


# ================================================================================================ lane calls
def _locate(path: str) -> Tuple[str, Optional[str], MountTable]:
    """('dir' | 'file' | 'missing', the closest existing ancestor of a missing path, the mount table as it is after
    the stat, which may have mounted an automounted share: UD2). Other OSErrors propagate."""
    kind, closest = "missing", None
    try:
        kind = "dir" if stat.S_ISDIR(os.stat(path).st_mode) else "file"
    except (FileNotFoundError, NotADirectoryError):
        p = path
        while closest is None and os.path.dirname(p) != p:
            p = os.path.dirname(p)
            closest = p if os.path.isdir(p) else None
    return kind, closest, MountTable.load()


def _header(path: str, engine_cls: type, legacy: bool) -> Tuple[Optional[TileHeader], int, str]:
    """The first tile's header (for legacy_all rows `path` is the folder) and the available RAM, read in a lane. A tile
    the engine can't use (read_tile_header's ValueError) gives no header and why: reading it again won't help (UD4).
    An OSError propagates: a Try again may."""
    if legacy:
        name = preflight.first_tiff_name(path)
        if name is None:
            raise ValueError("There are no TIFF files in this folder.")
        path = os.path.join(path, name)
    try:
        header, bad = preflight.read_tile_header(path, engine_cls), ""
    except ValueError as e:
        header, bad = None, str(e) or "it can't be read"
    return header, psutil.virtual_memory().available, bad


def _final_check(out_dir: str, mounts: MountTable, planned: Optional[FrozenSet[str]],
                 temp_dir: Optional[str]) -> Tuple[Optional[str], OutputIndex, Optional[int]]:
    """Output-only preflight. The legacy temp argument/result stay; temp I/O has its own lane request."""
    try:
        preflight.write_probe(out_dir)
        error = None
    except OSError as e:
        error = map_oserror(e).message
    return error, preflight.list_output_dir(out_dir, mounts, planned), None


def _temp_index(temp_dir: str, out_dir: str) -> tuple:
    """Only this volume: free space and scratch candidates. Output ownership is combined on Tk, without I/O."""
    return preflight.list_scratch_dir(temp_dir, out_dir)


def _scratch_deleted(post, path: str, out_dir: str, future) -> None:
    try:
        value, error = future.result(), None
    except FsFailure as e:
        value, error = None, e.error
    except Exception as e:
        value, error = None, FsError("io", friendly_error(e))
    post(FsDone(0, "delete", path, 0, "main", ("delete", out_dir), value, error, 0))


_DESTINATION_CHANGED = "The destination changed. Both images were kept. Choose Finalize again to review it."


def _finalize_file(kept: str, final: str, expected: Optional[tuple] = None,
                   replace_existing: bool = False) -> Tuple[Optional[Tuple[int, int, int]], str]:
    """*Finalize* (DS-9): move a finished image that a run had to keep into place. Returns the final file's (size,
    mtime_ns, gaps) for the ledger, the gaps read first (their sidecar goes with the move), or None and why the image
    can't be moved: it no longer waits there, or it is damaged. NotChecked and OSError propagate so the lane's
    error offers Try again (a PermissionError: the old file is still in use)."""
    gaps = preflight.complete_gaps(kept)
    try:
        if preflight.destination_identity(final) != expected:
            return None, _DESTINATION_CHANGED
        for attempt in range(5):
            try:
                size, mtime_ns = preflight.finalize_kept(kept, final, retries=1, expected_destination=expected,
                                                         replace=replace_existing)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(preflight.RETRY_DELAY_S)        # every retry revalidates the approved destination too
    except FileExistsError:
        return None, _DESTINATION_CHANGED
    except preflight.NotChecked:                          # a transient read failure, not a damaged image
        raise
    except ValueError as e:                               # a plain sentence, kept whole
        return None, str(e)
    return (size, mtime_ns, gaps), ""


# ================================================================================================ helpers
EMPTY, CHECKING, SCANNING, PAUSED, READY, RUNNING, STOPPING, FINISHED = (
    "EMPTY", "CHECKING", "SCANNING", "PAUSED", "READY", "RUNNING", "STOPPING", "FINISHED")
_FOOTER = {EMPTY: "Ready", CHECKING: "Checking…", SCANNING: "Searching…", PAUSED: "Search paused", READY: "Ready",
           STOPPING: "Stopping…", FINISHED: "Done"}
_STYLE = {"ok": "Surface.Ok.TLabel", "warn": "Surface.Warn.TLabel", "error": "Surface.Error.TLabel"}
_NOT_ANSWERING = frozenset({"timeout", "unresponsive", "unavailable"})
_ASK_AGAIN = frozenset({"timeout", "unresponsive"})      # failures asked again once their lane answers (UD1)
_UNSEARCHED = frozenset({"no_access", "unresponsive", "error"})   # notes of folders the search couldn't list
TAGLINE = "It's your Favourite Stitcher · Vectra Polaris / PhenoImager tiles → one pyramidal OME-TIFF per slide"
CREDITS = "made by Jacky @ Ramachandran Lab · logic adapted from Pete Bankhead (QuPath)"
Status = Tuple[Optional[str], str, Sequence[Tuple[str, Callable[[], None]]]]


def _n(k: int, one: str, many: Optional[str] = None) -> str:
    return f"{k:,} {one if k == 1 else many or one + 's'}"


def _flush_std() -> None:
    """Before os._exit: a windowed Windows build has no stdout or stderr at all (None, PL-2)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except (OSError, ValueError):                     # a closed or broken pipe: nothing left to save
            pass


class _Shutdown:
    """Bound settings, log draining and stream flushing together. Neither worker ever touches Tk."""
    TIMEOUT_S = 2.0

    def __init__(self, settings: Settings, listener: Any, fs: FsService, forced: bool) -> None:
        self.done = threading.Event()
        done = self.done

        def deadline() -> None:
            if not done.wait(self.TIMEOUT_S):
                os._exit(0)                              # no logging, stream or settings locks on this path

        def drain() -> None:
            with contextlib.suppress(Exception):
                settings.save_now(timeout=self.TIMEOUT_S)
            with contextlib.suppress(Exception):
                if listener is not None:
                    listener.stop()
            with contextlib.suppress(Exception):
                fs.shutdown()
            _flush_std()
            if forced or fs.has_stuck_threads():
                os._exit(0)
            done.set()

        threading.Thread(target=deadline, name="yfs-exit-deadline", daemon=True).start()
        threading.Thread(target=drain, name="yfs-exit-drain", daemon=True).start()


def _full_path(path: str, flavor: str) -> bool:
    """UD6, a string test: absolute, and on Windows with a drive or UNC part ('\\x' and 'D:x' depend on the working
    folder as much as 'x' does)."""
    if flavor == "win32":
        return ntpath.isabs(path) and bool(ntpath.splitdrive(path)[0])
    return posixpath.isabs(path)


@dataclass
class _HeaderLane:
    """UD4: the header reads of one lane, at most `limit` (fs.concurrency) at a time, so that a read that stalls
    leaves nothing queued in the lane for the watchdog to fail."""
    limit: int
    busy: int = 0
    waiting: List[Tuple[int, int, str]] = field(default_factory=list)     # heap of (priority, order, row_id)


def restore_geometry(saved: str, screen: Tuple[int, int], default: Tuple[int, int],
                     minimum: Tuple[int, int]) -> str:
    """§2.1.4: the saved 'WxH+X+Y', unless more than a third of it would be off-screen (an X2Go resolution change):
    then the default size, centred."""
    sw, sh = screen
    m = re.match(r"^=?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", saved or "")
    if m:
        x, y = int(m.group(3)), int(m.group(4))
        w = max(minimum[0], min(int(m.group(1)), sw))
        h = max(minimum[1], min(int(m.group(2)), sh))
        visible = max(0, min(x + w, sw) - max(x, 0)) * max(0, min(y + h, sh) - max(y, 0))
        if 3 * visible >= 2 * w * h:
            return f"{w}x{h}+{x}+{y}"
    w, h = default
    return f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}"


class _ConfirmDialog(dialogs._Dialog):
    """A small modal yes/no question (the leftover clean-up); on_done(True) or on_done(None)."""
    MAX_SCREEN = 0.6

    def __init__(self, master: tk.Misc, ctx: AppContext, title: str, text: str, ok: str,
                 on_done: Callable[[Optional[bool]], None]) -> None:
        super().__init__(master, ctx, title)
        self._done_cb = on_done
        self.listing = tk.Canvas(self.body, highlightthickness=0, background=T["surface"],
                                 yscrollincrement=self.px(20))
        self.listing.grid(row=0, column=0, sticky="nsew")
        items = ttk.Frame(self.listing, style="Surface.TFrame")
        self.listing.create_window(0, 0, window=items, anchor="nw")
        self._label(items, text, row=0, column=0)
        buttons = self._buttons(1, (("Cancel", self._cancel, "TButton"), (ok, lambda: self._close(True),
                                                                            "Primary.TButton")))
        self._fit_list(items)
        self._show(buttons["Cancel"])


class _FolderField(PathField):
    """A chosen or already committed path is literal; only a new edit is expanded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self.set(self.combo.get(), commit=True))

    def _clean(self, text: str) -> str:
        return text if text == self._committed else super()._clean(text)

    def get(self) -> str:
        return self._var.get()


class _PathResolver:
    """Use the browser's component walker without a browser window or any following I/O on an alias."""
    _resolve_links = FolderBrowser._resolve_links

    def __init__(self, ctx: AppContext, path: str, done: Callable[[str, Optional[FsError]], None],
                 valid: Callable[[], bool]) -> None:
        self.ctx, self._ticket, self._calls = ctx, ctx.fs.new_ticket(), {}
        self._valid, self._done = valid, done
        self._unsubscribe = ctx.bus.subscribe(FsDone, self._answered)
        self._resolve_links(_link_start(path), self._finish, lambda *_args: None, valid, self._ticket)

    def _answered(self, ev: FsDone) -> None:
        call = self._calls.pop(ev.req_id, None)
        if call is not None:
            if self._valid():
                call[1](ev)
            else:
                self._unsubscribe()

    def _finish(self, path: str, error: Optional[FsError]) -> None:
        self._unsubscribe()
        if self._valid():
            self._done(path, error)


# ================================================================================================ main window
class MainWindow:
    """Owns the Tk root's widgets; state machine of §2.1.3; wires FS, discovery, plan, preflight, runner.
    Consumes: FsDone/FsSlow (owner 'main'), LaneHealthChanged, MountsChanged, Scan*, CandidateFound, Run*/Row*,
    OutputWritten, LogLine (-> LogDrawer), BackgroundError. `restore=False` skips reopening the last session's
    folders (smoke tests)."""
    state: str = EMPTY

    def __init__(self, ctx: AppContext, initial_input: Optional[str] = None,
                 initial_output: Optional[str] = None, *, restore: bool = True) -> None:
        self.ctx, self.root, self.settings = ctx, ctx.root, ctx.settings
        self._plan: Optional[Plan] = None
        self.source: Optional[str] = None                 # the scanned input folder
        self._lookup: Optional[str] = None                # the tile folder being looked up (CHECKING)
        self.out_dir: Optional[str] = None
        self.out_index: Optional[OutputIndex] = None
        self.discovery: Optional[Discovery] = None
        self.runner: Optional[BatchRunner] = None
        self.model: Optional[ProgressModel] = None
        self.listener: Any = None                         # osutil.setup_logging's QueueListener (set by main)
        self._shutdown: Optional[_Shutdown] = None
        self._gen = self._scan_id = self._run_id = self._index_gen = self._locate_token = self._check_gen = 0
        self._ticket = 0
        self._changed: Set[str] = set()                   # plan rows changed by this tick's handlers
        self._full = self._idle_dirty = False
        self._idle_at = 0.0
        self._index_pending = False
        self._output_pending = self._temp_pending = self._final_temp_pending = False
        self._temp_error: Optional[str] = None
        self._temp_raws: tuple = ()
        self._temp_location: Optional[str] = None
        self._index_error: Optional[str] = None           # why the last output index failed (blocks Stitch, UD1)
        self._write_error: Optional[str] = None
        self._temp_free: Optional[int] = None
        self._final: Optional[Tuple[int, str, Optional[str]]] = None    # the final check under way (gen, out, temp)
        self._final_result: Optional[FsDone] = None
        self._final_rows: Tuple[str, ...] = ()
        self._output_token = 0
        self._output_resolving = False
        self._finalizing: Set[str] = set()
        self._check_error: Optional[str] = None           # why the last final check couldn't be done (UD1)
        # why [Delete them] or Finalize failed, and how to try again (UD1): step 2 says so until the next index
        self._save_error: Optional[Tuple[str, Sequence[Tuple[str, Callable[[], None]]]]] = None
        self._ram = psutil.virtual_memory().available
        self._long_paths = osutil.windows_long_paths()    # PL-7: a registry read, once
        self._asked: Set[Any] = set()                     # requested headers and in-flight output verifications
        self._header_lanes: Dict[str, _HeaderLane] = {}   # lane key -> its header reads (UD4)
        self._seq = itertools.count()
        # lane key -> what to ask again once it answers: key -> (it ran out of time, how to ask)
        self._reask: Dict[str, Dict[Any, Tuple[bool, Callable[[], None]]]] = {}
        self._recovery_seen: Dict[tuple, float] = {}
        self._stalled: Set[Any] = set()                   # asked again after running out of time, no answer since
        self._batch: Optional[contextlib.ExitStack] = None    # this tick's plan.batch() (MW-8)
        self._batch_ids: List[str] = []
        self._step2_due = False                           # an index or a verdict came: step 2 and the checks follow
        self._root_error: Optional[Note] = None
        self._unsearched = 0                              # folders the last search couldn't list (UD5)
        self._extra_depth, self._no_prune = 0, set()      # Search deeper / Search these folders too
        self._notes1: List[str] = []                      # extra lines under the step-1 status
        self._scan_status: Status = (None, "", ())
        self._n_jobs = self._started = 0
        self._dropped: Set[str] = set()
        self._counts: Dict[str, int] = {}
        self._phase, self._stop_mode, self._quit_after, self._confirm = "", "", False, True
        self._stop_dialog: Optional[dialogs.StopDialog] = None
        self._stop_target: Optional[Tuple[int, Optional[str], int]] = None
        self._quit_dialog: Optional[_ConfirmDialog] = None    # "Quit anyway?" while images wait to be finalized
        self._summary: Optional[RunSummary] = None         # the last batch, as its banner and the title tell it
        self._resume: Optional[Dict[str, Any]] = None      # last_run, when the app was closed during a batch
        self._compact: Optional[bool] = None
        self._bg_text = ""
        self._timers: Dict[str, str] = {}                 # name -> after id (see _later)
        ctx.output_index = lambda: self.out_index
        ctx.planned_output_keys = lambda: self._plan.planned_output_keys() if self._plan else frozenset()
        ctx.running = lambda: self.state in (RUNNING, STOPPING)
        self._build()
        self._bind_keys()
        bus = ctx.bus
        for ev_type, fn in ((FsDone, self._on_fs_done), (FsSlow, self._on_fs_slow),
                            (LaneHealthChanged, self._on_lane_health), (MountsChanged, self._on_mounts_changed),
                            (ScanProgress, self._on_progress), (CandidateFound, self._on_candidate),
                            (ScanNote, self._on_note), (ScanPaused, self._on_paused),
                            (ScanFinished, self._on_scan_finished),
                            (RunStarted, self._on_run_started), (RowStarted, self._on_row_started),
                            (RowProgress, self._on_row_progress), (RowBytes, self._on_row_bytes),
                            (RowFinished, self._on_row_finished), (OutputWritten, self._on_output_written),
                            (RunFinished, self._on_run_finished), (LogLine, lambda e: self.drawer.append([e])),
                            (BackgroundError, self._on_background_error)):
            bus.subscribe(ev_type, fn)
        bus.add_tick_hook(self._after_tick)
        self._set_state(EMPTY)
        self._idle_status()
        self._status2_from_index()
        self._show_empty()
        self._refresh_idle()
        self.root.bind("<Destroy>", lambda e: e.widget is self.root and self._stop_timers(), add="+")
        self._later("autosave", 1000, self._autosave)
        self._later("minute", 60000, self._minute)
        self._startup(initial_input, initial_output, restore)

    # ------------------------------------------------------------------------------------------ layout
    def _build(self) -> None:
        root, th, icons = self.root, self.ctx.theme, self.ctx.icons
        px, G = th.px, th.G
        root.title(APP_NAME)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        header = ttk.Frame(root, style="Header.TFrame", padding=(px(16), px(10), px(16), px(8)))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        logo = icons.logo(28 if th.icon_px == 16 else 2 * th.icon_px)
        ttk.Label(header, image=logo, style="Surface.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, px(10)))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="w")
        self.tagline = ttk.Label(header, style="Tagline.TLabel", width=1)
        self.tagline.grid(row=1, column=1, sticky="ew")
        fit_label(self.tagline, th.text(TAGLINE), G["ell"])
        buttons = ttk.Frame(header, style="Header.TFrame")
        buttons.grid(row=0, column=2, rowspan=2, sticky="e", padx=(px(12), 0))
        for label, command in (("How it works", lambda: self.info("how")), ("Settings", self.open_settings),
                               ("About", lambda: self.info("about"))):
            ttk.Button(buttons, text=label, style="Flat.TButton", command=command).pack(side="left", padx=(px(2), 0))
        ttk.Separator(root).grid(row=0, column=0, sticky="sew")
        cards = ttk.Frame(root, padding=(px(12), px(12), px(12), 0))
        cards.grid(row=1, column=0, sticky="ew")
        cards.columnconfigure(0, weight=1)
        self.cards: List[ttk.Frame] = []
        self.input_field = self._step_card(
            cards, "1", "Tiles to stitch", f"Folder with your {G['ell']}_component_data.tif tiles, or a folder of "
            "such folders", lambda: self._recents("inputs"), self._input_committed, self.browse_input)
        self.output_field = self._step_card(
            cards, "2", "Save stitched images to", "Folder for the stitched .ome.tif files",
            lambda: self._recents("outputs"), self._output_committed, self.browse_output)
        self.pane = tk.PanedWindow(root, orient="vertical", bg=T["bg"], sashwidth=px(6), bd=0, showhandle=False,
                                   opaqueresize=True)
        self.pane.grid(row=2, column=0, sticky="nsew", padx=px(12))
        self.upper = ttk.Frame(self.pane)
        self.upper.columnconfigure(0, weight=1)
        self.upper.rowconfigure(0, weight=1)
        self.pane.add(self.upper, stretch="always")
        self.table = PlanTable(self.upper, self.ctx, on_focus=lambda _row: None, on_details=self.details,
                               on_fix_channels=self.fix_channels, on_note_action=self._note_action,
                               on_changed=self._plan_edited)
        self.table.grid(row=0, column=0, sticky="nsew")
        self.drawer = LogDrawer(self.upper, self.pane, self.ctx)
        self.drawer.bar.grid(row=1, column=0, sticky="ew")
        bottom = ttk.Frame(root)
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        eb = self.error_banner = ttk.Frame(bottom, style="Error.TFrame", padding=(px(8), px(3)))
        ttk.Label(eb, image=icons.get("error"), style="Error.Banner.TLabel").pack(side="left", padx=(0, px(6)))
        ttk.Label(eb, text="Something went wrong (details in the log).", style="Error.Banner.TLabel").pack(side="left")
        ttk.Button(eb, image=icons.get("close"), style="Error.Banner.TButton",
                   command=eb.grid_remove).pack(side="right")
        ttk.Button(eb, text="Copy details", style="Error.Banner.TButton", command=self._copy_error).pack(side="right")
        self.runbar = RunBar(bottom, self.ctx, on_start=self.start, on_stop=self._ask_stop,
                             on_issue_action=self._issue_action, on_open_output=self._open_output,
                             on_retry_failed=self._retry_failed, on_new_batch=self._new_batch)
        self.runbar.grid(row=1, column=0, sticky="ew")
        footer = ttk.Frame(root, padding=(px(12), 0, px(12), px(6)))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.footer = ttk.Label(footer, style="Footer.TLabel")
        self.footer.grid(row=0, column=0, sticky="w")
        ttk.Label(footer, style="Footer.TLabel", text=th.text(f"v{__version__} · {CREDITS}")).grid(row=0, column=1)
        self._place_window()
        if self.settings.get("ui.log_open"):
            self.drawer.set_open(True)
        root.bind("<Configure>", self._on_configure, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)
        if th.ws == "aqua":
            root.createcommand("tk::mac::Quit", self.close)

    def _step_card(self, parent: ttk.Frame, num: str, title: str, placeholder: str, recents: Callable[[], List[str]],
                   on_commit: Callable[[str], None], on_browse: Callable[[], None]) -> PathField:
        px = self.ctx.theme.px
        card = ttk.Frame(parent, style="Card.TFrame", padding=(px(12), px(8)))
        card.grid(row=len(self.cards), column=0, sticky="ew", pady=(0, px(8)))
        card.columnconfigure(2, weight=1)
        self.cards.append(card)
        field = _FolderField(card, self.ctx, placeholder=placeholder, recents=recents, on_commit=on_commit,
                          on_browse=on_browse)
        field.grid(row=0, column=2, sticky="ew")
        badge = ttk.Label(card, text=num, style="StepNum.TLabel", width=2)
        label = ttk.Label(card, text=title, style="Step.TLabel", width=22)
        top = max(0, (field.combo.winfo_reqheight() - label.winfo_reqheight()) // 2)   # centred on the combobox
        badge.grid(row=0, column=0, sticky="nw", padx=(0, px(8)), pady=(top, 0))
        label.grid(row=0, column=1, sticky="nw", pady=(top, 0))
        return field

    def _place_window(self) -> None:
        root, px = self.root, self.ctx.theme.px
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        minimum = (min(px(760), int(sw * 0.9)), min(px(560), int(sh * 0.9)))
        default = (min(px(1040), int(sw * 0.9)), min(px(740), int(sh * 0.9)))
        root.minsize(*minimum)
        root.geometry(restore_geometry(str(self.settings.get("ui.geometry") or ""), (sw, sh), default, minimum))

    def _bind_keys(self) -> None:
        mods = ("Control", "Command") if self.ctx.theme.ws == "aqua" else ("Control",)
        for mod in mods:
            for seq, fn in (("o", self.browse_input), ("O", self.browse_output), ("Return", self._ctrl_enter),
                            ("KP_Enter", self._ctrl_enter), ("j", self.drawer.toggle), ("comma", self.open_settings)):
                self.root.bind(f"<{mod}-{seq}>", lambda _e, f=fn: f() or "break")
            for widget in (self.table.tree, self.input_field.combo, self.output_field.combo):
                for seq in ("Return", "KP_Enter"):           # their own <Return> would win (details, commit)
                    widget.bind(f"<{mod}-{seq}>", lambda _e: self._ctrl_enter() or "break")
        self.root.bind("<F5>", lambda _e: self.rescan())
        self.root.bind("<F1>", lambda _e: self.info("how"))

    def _on_configure(self, e: tk.Event) -> None:
        """Compact mode below px(640) of height, with 20 px of hysteresis (§2.1.4)."""
        if e.widget is not self.root or e.height <= 1 or not self.tagline.winfo_exists():
            return
        px = self.ctx.theme.px
        compact = e.height < px(640) if not self._compact else e.height < px(660)
        if compact != self._compact:
            self._set_compact(compact)

    def _set_compact(self, on: bool) -> None:
        px = self.ctx.theme.px
        self._compact = on
        if on:
            self.tagline.grid_remove()
        else:
            self.tagline.grid()
        for card in self.cards:
            card.configure(padding=(px(12), px(4) if on else px(8)))
        self.table.set_compact(on)
        self.runbar.set_compact(on)
        self.root.update_idletasks()
        self.pane.paneconfigure(self.upper, minsize=self.table.min_height(4) + self.drawer.bar.winfo_reqheight())

    # ------------------------------------------------------------------------------------------ public (§5.5.10)
    @ui_thread
    def plan(self) -> Optional[Plan]:
        return self._plan

    @ui_thread
    def set_input(self, path: str) -> None:
        if not _full_path(path, self.ctx.fs.mounts.flavor):
            path = normalize_user_path(path, self.ctx.fs.mounts)
        self.input_field.set(path, commit=True)

    @ui_thread
    def set_output(self, path: str) -> None:
        if not _full_path(path, self.ctx.fs.mounts.flavor):
            path = normalize_user_path(path, self.ctx.fs.mounts)
        self.output_field.set(path, commit=True)

    @ui_thread
    def start(self, confirm: bool = True) -> None:
        """Stitch: the final preflight (§3.8) runs first; then a block stops here, warnings open the
        *Ready to stitch?* sheet (unless confirm is False) and a clean plan starts at once. The folders can't be
        changed while it runs, and its answer counts only for the folders it checked (UD7)."""
        if self._final is not None or self._output_resolving or self.state not in (READY, FINISHED) \
                or self._plan is None or not self.out_dir:
            self._refresh_idle()
            return
        self._confirm, self._check_error = confirm, None
        self._check_gen += 1
        temp = self.settings.get("run.temp_dir") or None
        self._final = (self._check_gen, self.out_dir, temp)
        self._final_result = None
        self._final_temp_pending = bool(temp)
        self._temp_error, self._temp_raws = None, ()
        self._final_rows = tuple(r.row_id for r in self._stitch_rows())
        for r in self._stitch_rows():
            if r.header is None and not r.header_error:
                self._read_header(r.row_id)
        self._enable_folders(False)
        self._refresh_idle()                             # drop the previous error and its retry link at once
        self.ctx.fs.retry(self.out_dir)                   # a click is a Try again for a lane that stopped answering
        self.ctx.fs.call(self.out_dir, "write_probe", partial(_final_check, self.out_dir, self.ctx.fs.mounts,
                                                              self._planned_names(), None),
                         ticket=self.ctx.fs.new_ticket(), owner="main", tag=("final",) + self._final, priority=0,
                         user_initiated=True)
        if temp:
            self._request_temp("final_temp", self._final)

    @ui_thread
    def request_stop(self, mode: str, target: Optional[Tuple[int, Optional[str], int]] = None) -> bool:
        """'after_current' | 'now' (§2.4.5)."""
        if self.runner is None or self.state not in (RUNNING, STOPPING):
            return False
        if target is not None and target != self._current_stop_target():
            return False
        guarded = getattr(self.runner, "request_stop", None)
        if target is not None and guarded is not None:
            if not guarded(mode, expected_row=target[1]):
                return False
        elif mode == "now":
            self.runner.stop_now()
        else:
            self.runner.stop_after_current()
        self._stop_mode = "now" if mode == "now" or self._stop_mode == "now" else "after_current"
        self._set_state(STOPPING)
        self._show_stopping()
        self._update_run_bar()                            # the footer and the title say so at once
        return True

    @ui_thread
    def rescan(self, fresh: bool = True) -> None:
        """F5 / Rescan / Start a new batch: search the input again (keeping the user's choices) and re-index the
        output folder. fresh: list every folder again (UD5); the note actions reuse listings up to two minutes old."""
        if self.state in (RUNNING, STOPPING):
            return
        self.runbar.dismiss()
        self._forget_final_check()
        self._write_error = None                          # checked again at the next Stitch
        self.ctx.fs.reload_mounts()
        tiles = self._tiles_in_use()
        if tiles:
            self._locate(tiles, fresh)
        elif self.input_field.get():                      # after a failed lookup: the field's text, cleaned (MW-14)
            path = self._typed(self._field_path(self.input_field), self.input_field, None)
            if path is not None:
                self._locate(path, fresh)
        elif self.out_dir:
            self._request_index()

    @ui_thread
    def browse_input(self) -> None:
        if self._folders_open():
            initial = browser_initial("input", self._field_path(self.input_field), None, self.settings)
            FolderBrowser.open(self.root, self.ctx, mode="input", initial=initial,
                               on_done=lambda p: p and self.set_input(p))

    @ui_thread
    def browse_output(self) -> None:
        if self._folders_open():
            tiles_here = any(r.cand.rel_folder == "." for r in self._slides())   # BR-8: never preselect a tile folder
            initial = browser_initial("output", self._field_path(self.output_field), self.source, self.settings,
                                      input_is_slide=tiles_here)
            FolderBrowser.open(self.root, self.ctx, mode="output", initial=initial,
                               on_done=lambda p: p and self.set_output(p))

    def _folders_open(self) -> bool:
        """Can the folders change now? Not during a run, nor during Stitch's final check (UD7)."""
        return self.state not in (RUNNING, STOPPING) and self._final is None

    def _field_path(self, field: PathField) -> str:
        """The field's text as a clean path (a typed '~/...' or a quoted paste still opens the browser there)."""
        text = field.get()
        return field._clean(text) if text else ""

    @ui_thread
    def open_settings(self) -> None:
        dialogs.SettingsDialog.open(self.root, self.ctx, self.state in (RUNNING, STOPPING), self._settings_done)

    @ui_thread
    def info(self, page: str) -> None:
        dialogs.InfoDialog.show(self.root, self.ctx, page)

    @ui_thread
    def details(self, row_id: str) -> None:
        if self._plan is not None:
            dialogs.SlideDetailsDialog.show(self.root, self.ctx, self._plan, row_id, self.fix_channels)

    @ui_thread
    def fix_channels(self, row_ids: List[str]) -> None:
        if self._plan is not None and self.state not in (CHECKING,):
            dialogs.ChannelNamesDialog.open(self.root, self.ctx, self._plan, list(row_ids),
                                            lambda _ok: self._plan_edited(full=True))

    @ui_thread
    def close(self) -> None:
        """The window's close button: during a run the *Quit?* dialog (§2.4.5); while finished images wait to be
        moved into place, a question first (DS-9); otherwise save and quit (§2.1.6)."""
        if self.state in (RUNNING, STOPPING):
            self._ask_stop(closing=True)
            return
        n = sum(r.status == "finalize" for r in self._plan.rows()) if self._plan is not None else 0
        if not n:
            self._quit()
        elif self._quit_dialog is None or not self._quit_dialog.winfo_exists():
            G, out = self.ctx.theme.G, self.out_dir or ""
            folder = os.path.basename(out.rstrip("/\\")) or out
            self._quit_dialog = _ConfirmDialog(
                self.root, self.ctx, f"Quit {APP_NAME}?",
                f"{_n(n, 'finished image is', 'finished images are')} waiting to be moved into place. Quit anyway?\n\n"
                f"{'It is' if n == 1 else 'They are'} kept, and moved into place the next time you stitch into "
                f"{G['lq']}{folder}{G['rq']}.", "Quit anyway", lambda ok: ok and self._quit())

    def _quit(self) -> None:
        if self._shutdown is not None:
            return
        if self.discovery is not None:
            self.discovery.stop()
        self.settings.set("ui.geometry", self.root.geometry())
        self._shutdown = _Shutdown(self.settings, self.listener, self.ctx.fs, False)
        self.listener = None
        self._poll_shutdown()

    def _poll_shutdown(self) -> None:
        if self._shutdown.done.is_set():
            self.root.quit()
        else:
            self._later("shutdown", 25, self._poll_shutdown)

    # ------------------------------------------------------------------------------------------ startup
    def _startup(self, initial_input: Optional[str], initial_output: Optional[str], restore: bool) -> None:
        s, fs = self.settings, self.ctx.fs
        if s.load_error:
            self._notes1.append(s.load_error)
            self._idle_status()
        last = s.get("last_run") or {}
        saved_in = (s.get("paths.input") or "") if restore else ""
        saved_out = (s.get("paths.output") or "") if restore else ""
        if last.get("state") == "running" and restore:
            self._resume = last
        out = initial_output or saved_out
        if out:
            if initial_output or (fs.auto_enter(out) is None and not tcc_protected(out)):
                self.output_field.set(out, commit=True)
            else:
                self.output_field.set(out)
                self.out_dir = out
                self._status2(None, "Press Rescan (F5) to check this folder.", [("Rescan", self.rescan)])
        source = initial_input or saved_in
        if not source:
            return
        self.input_field.set(source)
        if initial_input or (fs.auto_enter(source) is None and not tcc_protected(source)):          # R23
            self._later("rescan", 300, lambda: self.state == EMPTY and self._input_committed(source, fresh=True))
        else:
            self.source = None
            self._status1("info", "Press Rescan (F5) to look in this folder again.", [("Rescan", self.rescan)])

    # ------------------------------------------------------------------------------------------ step 1
    def _recents(self, kind: str) -> List[str]:
        return [r["path"] for r in self.settings.get(f"recent.{kind}") or [] if isinstance(r, dict) and r.get("path")]

    def _status1(self, kind: Optional[str], text: str, actions: Sequence[Tuple[str, Callable[[], None]]] = ()) -> None:
        if self.state not in (RUNNING, STOPPING):
            self._scan_status = (kind, text, actions)
            self.input_field.set_status(kind, "\n".join([text] + self._notes1), _STYLE.get(kind or "",
                                        "Surface.Muted.TLabel"), actions)

    def _idle_status(self) -> None:
        self._status1(None, f"The search looks {_n(self._depth(), 'folder level')} deep and shows you everything "
                            "before stitching.")

    def _typed(self, text: str, field: PathField, in_use: Optional[str]) -> Optional[str]:
        """Validate a committed path. PathField already expanded edits; chosen and saved paths stay literal.
        A relative path would depend on where the app was started (UD6). String work only."""
        mounts, G = self.ctx.fs.mounts, self.ctx.theme.G
        path = text
        if _full_path(path, mounts.flavor):
            field.set(path)
            return path
        if in_use:
            field.set(in_use)
        field.set_status("error", f"Can't find {G['lq']}{path}{G['rq']}. Type the full path of a folder.",
                         _STYLE["error"])
        return None

    def _input_committed(self, text: str, fresh: bool = False) -> None:
        """A tile folder was chosen (also set_input() and --input); fresh: list every folder again (UD5)."""
        if self.state in (RUNNING, STOPPING):
            return
        path = self._typed(text, self.input_field, self._tiles_in_use())
        if path is None:
            return
        self._forget_final_check()
        self.runbar.dismiss()                             # the banner was about the last batch (UD7)
        if not (self.source and same_path(path, self.source)):
            self._extra_depth, self._no_prune, self._notes1 = 0, set(), []
        self._locate(path, fresh)

    def _tiles_in_use(self) -> Optional[str]:
        """The tile folder step 1 stands for: the one being looked up, else the one searched."""
        return self._lookup if self.state == CHECKING else self.source

    def _locate(self, path: str, fresh: bool = False) -> None:
        self._stop_discovery()
        self._locate_token, self._lookup = self._locate_token + 1, path
        self._set_state(CHECKING)
        self._status1("spinner", "Checking…")
        token = self._locate_token

        def resolved(found: str, error: Optional[FsError]) -> None:
            if error is not None and error.code not in ("not_found", "not_a_dir"):
                self._ask_again(self.ctx.fs.lane_key(found), error, ("locate", path),
                                lambda: token == self._locate_token and self._locate(path, fresh))
                self._no_input()
                self._status1(*self._place_error(error, found, lambda: self._locate(path, True)))
                return
            self._lookup = found
            self.input_field.set(found)
            self.ctx.fs.call(found, "locate", partial(_locate, found), ticket=0, owner="main",
                             tag=("locate", token, found, fresh), priority=0, user_initiated=True)
        _PathResolver(self.ctx, path, resolved, lambda: token == self._locate_token and self.state == CHECKING)

    def _on_located(self, ev: FsDone, token: int, path: str, fresh: bool) -> None:
        if token != self._locate_token or self.state != CHECKING:
            return
        self._ask_again(self.ctx.fs.lane_key(path), ev.error, ("locate", path),
                        lambda: token == self._locate_token and self.state == EMPTY and self._locate(path, fresh))
        if ev.error is not None:
            self._no_input()
            self._status1(*self._place_error(ev.error, path, lambda: self._locate(path, True)))
            return
        kind, closest, mounts = ev.result
        if mounts.mounts():                               # UD2: what this access mounted, before the search starts
            self.ctx.fs.set_mounts(mounts)
        if kind == "missing":
            self._no_input()
            go = [("Go there", lambda: self.set_input(closest))] if closest else []
            self._status1("error", f"Can't find this folder. Closest existing folder: {closest or 'none'}", go)
            return
        if kind == "file":
            path = os.path.dirname(path)
            self.input_field.set(path)
            self._notes1 = ["Using the folder of the pasted file."]
        self._begin_scan(path, fresh)

    def _no_input(self) -> None:
        self._stop_discovery()
        self._end_batch()
        self._gen += 1                                   # no result or recovery belongs to the retired plan
        self._asked.clear()
        self._header_lanes.clear()
        for pending in self._reask.values():              # keep recovery of the current locate and output index
            for key in tuple(pending):
                if key[0] in ("header", "check"):
                    pending.pop(key)
        self._stalled = {key for key in self._stalled if key[0] not in ("header", "check")}
        self._plan, self.source = None, None
        dialogs.SlideDetailsDialog.refresh_rows(plan=None)
        self._save_error = None
        self._status2_from_index()
        self.table.bind_plan(None)
        self._show_empty()
        self._set_state(EMPTY)

    def _show_empty(self) -> None:
        def dismiss() -> None:
            self.settings.set("ui.welcome_dismissed", True)
        if self._plan is None:
            self.table.show_welcome(not self.settings.get("ui.welcome_dismissed"), self.browse_input, dismiss)

    def _place(self, path: str) -> str:
        """The name of the share, drive or folder that holds `path`, for messages."""
        return self.ctx.fs.mounts.place_name(path)

    def _place_error(self, err: FsError, path: str, retry: Callable[[], None]) -> Status:
        def again() -> None:
            self.ctx.fs.retry(path)
            retry()
        if err.code == "no_access":
            extra = " Check System Settings → Privacy & Security → Files and Folders." if osutil.MACOS else ""
            return "error", "You don't have permission to open this folder." + extra, ()
        if err.code == "disconnected":
            return ("error", f"{self._place(path)} is disconnected. Open it once in File Explorer, then Try again.",
                    [("Try again", again)])
        if err.code in _NOT_ANSWERING:
            return "error", f"{self._place(path)} isn't answering. Nothing else is affected.", [("Try again", again)]
        return "error", f"Can't open this folder: {err.message}", [("Try again", again)]

    # ------------------------------------------------------------------------------------------ discovery
    def _begin_scan(self, folder: str, fresh: bool = False) -> None:
        """A new plan of `folder` (the user's choices kept when it is the same folder) and its search; fresh: list
        every folder again instead of reusing listings up to two minutes old (UD5)."""
        self._end_batch()
        old = self._plan
        carry = old.carry_state() if old is not None and same_path(old.source, folder) else None
        self._stop_discovery()
        self._gen += 1
        self._ticket = self.ctx.fs.new_ticket()
        self._asked, self._root_error, self._unsearched, self._header_lanes = set(), None, 0, {}
        naming = "slide" if self.settings.get("discovery.naming") == "slide" else "folder"
        self._plan = Plan(folder, self.out_dir, naming, carry)
        dialogs.SlideDetailsDialog.refresh_rows(plan=self._plan)
        if self.out_index is not None and self.out_dir and same_path(self.out_index.out_dir, self.out_dir):
            self._plan.apply_output_index(self.out_index)
        self.source = folder
        self.table.bind_plan(self._plan)
        self._save_error = None                           # its row actions belonged to the old plan
        self._status2_from_index()
        self._scan_id += 1
        opts = DiscoveryOptions(max_depth=self._depth(), out_dir=self.out_dir, no_prune=frozenset(self._no_prune),
                                cache_age=0 if fresh else DiscoveryOptions.cache_age, extra_depth=self._extra_depth)
        self.discovery = Discovery(self._scan_id, folder, self.ctx.fs, self.ctx.bus.post, opts)
        self._set_state(SCANNING)
        self._status1("spinner", "Searching…", [("Stop searching", self.stop_scan)])
        self.discovery.start()

    def _stop_discovery(self) -> None:
        self._scan_id += 1                                # its late events must not finish the next lookup
        if self.discovery is not None:
            self.discovery.stop()
            self.discovery = None
        if self._ticket:
            self.ctx.fs.cancel_ticket(self._ticket)

    def stop_scan(self) -> None:
        if self.discovery is not None:
            self.discovery.stop()

    def resume_scan(self) -> None:
        if self.discovery is not None and self.state == PAUSED:
            self.discovery.resume()
            self._set_state(SCANNING)
            self._status1("spinner", "Searching…", [("Stop searching", self.stop_scan)])

    def _slides(self) -> List[Any]:
        return [r for r in self._plan.rows() if r.section != "note" and r.cand.kind == "component"] \
            if self._plan is not None else []

    def _on_progress(self, ev: ScanProgress) -> None:
        if ev.scan_id != self._scan_id or self.state != SCANNING:
            return
        where = rel_folder(ev.current, self.source or ev.current)
        self._status1("spinner", f"Searching… {_n(ev.dirs_done, 'folder')} · {_n(ev.slides_found, 'slide')} found"
                      + ("" if where == "." else f" · now in {where}"), [("Stop searching", self.stop_scan)])

    def _on_candidate(self, ev: CandidateFound) -> None:
        if ev.scan_id == self._scan_id and self._plan is not None:
            self._tick_plan().add_candidate(ev.candidate)    # the tick's candidates cost one recompute (MW-8)
            if ev.candidate.kind != "legacy_all":
                self._read_header(ev.candidate.key)

    def _on_note(self, ev: ScanNote) -> None:
        if ev.scan_id == self._scan_id and self._plan is not None:
            n = ev.note
            if n.kind in ("no_access", "unresponsive", "error") and same_path(n.path, self.source):
                self._root_error = n
            self._changed.update(self._plan.add_note(n))

    def _on_paused(self, ev: ScanPaused) -> None:
        if ev.scan_id == self._scan_id and self.state == SCANNING:
            self._set_state(PAUSED)
            self._status1("info", f"Searched {_n(ev.dirs_done, 'folder')} in {int(ev.elapsed)} s — "
                          f"{_n(len(self._slides()), 'slide')} so far.",
                          [("Keep searching", self.resume_scan), ("Stop here", self.stop_scan)])

    def _on_scan_finished(self, ev: ScanFinished) -> None:
        if ev.scan_id != self._scan_id or self._plan is None:
            return
        self._end_batch()
        self.discovery, self._unsearched = None, len(ev.failed)
        self._set_state(READY)
        slides, source = self._slides(), self.source or ""
        deeper = [("Search deeper", lambda: self._rescan(extra=DEEPER_STEP))] if ev.frontier else []
        cls = self.ctx.fs.lane_class(source)
        where = {NETWORK: "network drive", CLOUD: "cloud folder"}.get(cls, "local disk")
        none = f"No slides found within {_n(ev.max_depth, 'level')}"          # the depth searched (PL-16)
        n = self._root_error
        if n is not None and not slides:
            code = {"no_access": "no_access", "unresponsive": "unresponsive"}.get(n.kind, "io")
            self._status1(*self._place_error(FsError(code, n.detail or "It couldn't be read."), source, self.rescan))
        elif ev.failed:                                    # FS-6: part of the tree is missing from the list
            found = f"Found {_n(len(slides), 'slide')}" if slides else none
            self._status1("warn", f"{'Search stopped. ' if ev.stopped else ''}{found} · "
                          f"{_n(len(ev.failed), 'folder')} couldn't be searched",
                          [("Show", self._show_unsearched), ("Rescan", self.rescan)] + deeper)
        elif slides:
            tiles = sum(len(r.cand.tile_names) for r in slides)
            self._status1("ok", f"{'Search stopped: ' if ev.stopped else ''}{_n(len(slides), 'slide')} found "
                          f"({_n(tiles, 'tile')}) · {_n(ev.dirs_done, 'folder')} in {ev.elapsed:.1f} s · {where}",
                          [("Rescan", self.rescan)] + deeper)
        else:
            self._status1("warn", f"{'Search stopped. ' if ev.stopped else ''}{none}.{self._found_text()}",
                          deeper + [("Choose another folder…", self.browse_input)])
        if cls == CLOUD:
            depth = "one level only" if ev.max_depth == 1 else f"{ev.max_depth} levels deep"
            self._notes1 = [x for x in self._notes1 if not x.startswith("Cloud")] + [
                f"Cloud folder: files download as they are read. Searching {depth}."]
            self._status1(*self._scan_status)
        self._resume_line()
        self.settings.set("paths.input", source)
        self.settings.add_recent("input", source, len(slides))
        if not self.out_dir:
            self._status2_from_index()                     # the chips depend on the tiles found
        self._request_index()
        self._verify_existing()

    def _depth(self) -> int:
        depth = self.settings.get("discovery.max_depth", 3)
        return (depth if isinstance(depth, int) and 1 <= depth <= 6 else 3) + self._extra_depth

    def _found_text(self) -> str:
        """' Found: 2 stitched images, 18 whole-slide images.' from the notes and the excluded rows."""
        counts: Dict[str, int] = {}
        for r in self._plan.rows("excluded") if self._plan else []:
            kind = r.note.kind if r.note is not None else r.cand.kind
            if kind in ("outputs", "wsi", "raw_scan"):
                counts[kind] = counts.get(kind, 0) + r.note.count
            elif kind in ("legacy_all", "other_suffix", "unsuffixed"):
                counts[kind] = counts.get(kind, 0) + 1
        words = (("outputs", "stitched image"), ("wsi", "whole-slide image"), ("raw_scan", "raw scan"),
                 ("legacy_all", "folder of other TIFF images", "folders of other TIFF images"),
                 ("other_suffix", "set of other inForm tiles", "sets of other inForm tiles"),
                 ("unsuffixed", "set of tiles without a suffix", "sets of tiles without a suffix"))
        found = [_n(counts[w[0]], *w[1:]) for w in words if counts.get(w[0])]
        return f" Found: {', '.join(found)}." if found else ""

    def _resume_line(self) -> None:
        last = self._resume
        if last and same_path(last.get("input") or "", self.source or ""):
            try:
                when = time.strftime("%d %b, %H:%M", time.strptime(str(last.get("started")), "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                when = "unknown time"
            line = (f"Your last batch ({when.lstrip('0')}) stopped after {last.get('done', 0)} of "
                    f"{last.get('planned', 0)} slides — the rest are ticked below.")
            if line not in self._notes1:
                self._notes1.append(line)
                self._status1(*self._scan_status)

    def _show_unsearched(self) -> None:
        """[Show]: *Not included*, opened at the folders the search couldn't list, with the keyboard there."""
        rows = [r.row_id for r in self._plan.rows("note")
                if r.note.kind in _UNSEARCHED] if self._plan is not None else []
        self.table.reveal(rows, "excluded")

    def _rescan(self, extra: int = 0, no_prune: Sequence[str] = (), retry: Sequence[str] = ()) -> None:
        """Search deeper, Search these folders too, Search this folder again: a wider search, which reuses the
        listings of the last two minutes. Ignored during a run, so the next search isn't changed silently (MW-21)."""
        if self.state in (RUNNING, STOPPING):
            return
        for p in retry:
            self.ctx.fs.retry(p)
        self._extra_depth += extra
        self._no_prune.update(no_prune)
        self.rescan(fresh=False)

    def _note_action(self, action: str, row_id: str) -> None:
        """A row's action from the table. During a run only Finalize does something: a search started then would
        silently change the next batch (MW-21)."""
        if self._plan is None:
            return
        if action == "finalize":
            self._finalize([row_id])
        elif self.state in (RUNNING, STOPPING):
            return
        elif action == "retry_header":
            self.ctx.fs.retry(self._header_path(row_id)[0])
            self._read_header(row_id, force=True)
        elif action == "search_deeper":
            self._rescan(extra=DEEPER_STEP)
        elif action == "search_wide":
            self._rescan(no_prune=self._plan.note_paths(row_id))
        elif action == "search_again":
            self._rescan(retry=self._plan.note_paths(row_id))

    # ------------------------------------------------------------------------------------------ headers, outputs
    def _header_path(self, row_id: str) -> Tuple[str, bool]:
        c = self._plan.row(row_id).cand
        legacy = c.kind == "legacy_all"
        return (c.folder if legacy else os.path.join(c.folder, sorted(c.tile_names)[0])), legacy

    def _read_header(self, row_id: str, force: bool = False) -> None:
        """Read a row's first tile once (force: Try again). §5.5.10 priorities: a Try again 0, slides 3, the other
        tile sets 6 (they must not delay the slides); each lane reads at most fs.concurrency() at a time (UD4)."""
        if self._plan is None or (("header", row_id) in self._asked and not force):
            return
        try:
            path = self._header_path(row_id)[0]
        except KeyError:                                 # a queued retry whose row has disappeared
            return
        self._asked.add(("header", row_id))
        key = self.ctx.fs.lane_key(path)
        lane = self._header_lanes.setdefault(key, _HeaderLane(self.ctx.fs.concurrency(path)))
        prio = 0 if force else 3 if self._plan.row(row_id).cand.kind == "component" else 6
        heapq.heappush(lane.waiting, (prio, next(self._seq), row_id))
        self._send_headers(key)

    def _send_headers(self, key: str) -> None:
        lane = self._header_lanes[key]
        while lane.waiting and lane.busy < lane.limit:
            prio, _, row_id = heapq.heappop(lane.waiting)
            path, legacy = self._header_path(row_id)
            lane.busy += 1
            self.ctx.fs.call(path, "header", partial(_header, path, self.ctx.engine_cls, legacy), ticket=self._ticket,
                             owner="main", tag=("header", self._gen, row_id, key), priority=prio)

    def _planned_names(self) -> FrozenSet[str]:
        """norm_key of every candidate row's final name: partials that match none of them are leftovers."""
        self._end_batch()
        rows = self._plan.rows() if self._plan is not None else []
        return frozenset(norm_key(r.output_stem + OME_SUFFIX) for r in rows if r.section != "note")

    def _request_index(self) -> None:
        """A fresh index of the save folder (each request its own ticket: one asked for another plan or temporary
        folder must not answer it). Partials count as leftovers only against a finished search (DS-9)."""
        if not self.out_dir:
            return
        self._index_gen += 1
        self._index_pending, self._output_pending, self._index_error = True, True, None
        temp = self.settings.get("run.temp_dir") or None
        self._temp_pending, self._temp_error, self._temp_raws = bool(temp), None, ()
        self._temp_free, self._temp_location = None, None
        if self.out_index is None or not same_path(self.out_index.out_dir, self.out_dir):
            self._status2("spinner", "Checking…")
        planned = self._planned_names() if self.state in (READY, RUNNING, STOPPING, FINISHED) else None
        self.ctx.fs.call(self.out_dir, "index_output", partial(preflight.list_output_dir, self.out_dir,
                                                               self.ctx.fs.mounts, planned),
                         ticket=self.ctx.fs.new_ticket(), owner="main", tag=("index", self._index_gen, self.out_dir),
                         priority=2)
        if temp:
            self._request_temp("temp_index", (self._index_gen, self.out_dir, temp))
        self._idle_dirty = True

    def _request_temp(self, kind: str, token: tuple) -> None:
        gen, out, temp = token

        def valid() -> bool:
            return token == self._final if kind == "final_temp" else \
                (gen, out, temp) == (self._index_gen, self.out_dir, self.settings.get("run.temp_dir") or None)

        def resolved(path: str, error: Optional[FsError]) -> None:
            if error is not None and error.code not in ("not_found", "not_a_dir"):
                self._on_temp_index(FsDone(0, "usage", path, 0, "main", (kind,) + token, None, error, 0))
                return
            self._temp_location = path
            self.ctx.fs.retry(path)
            self.ctx.fs.call(path, "usage", partial(_temp_index, path, out),
                             ticket=self.ctx.fs.new_ticket(), owner="main", tag=(kind,) + token, priority=2)
        _PathResolver(self.ctx, temp, resolved, valid)

    def _with_scratch(self, idx: OutputIndex) -> OutputIndex:
        raws = tuple((p, info) for p, info in self._temp_raws
                     if not preflight._in_use(info, time.time(), lambda key: key in idx.busy))
        return replace(idx, leftovers=idx.leftovers + tuple(item for item in raws if item not in idx.leftovers))

    def _on_temp_index(self, ev: FsDone) -> None:
        kind, gen, out, temp = ev.tag
        if kind == "final_temp":
            if ev.tag[1:] != self._final:
                return
            self._final_temp_pending = False
        elif (gen, out, temp) != (self._index_gen, self.out_dir, self.settings.get("run.temp_dir") or None):
            return
        else:
            self._temp_pending = False
            self._index_pending = self._output_pending
        self._ask_again(self.ctx.fs.lane_key(ev.path), ev.error, ("temp", out, temp),
                        lambda: self.out_dir == out and (self.settings.get("run.temp_dir") or None) == temp
                        and self.state not in (RUNNING, STOPPING) and self._request_index())
        if ev.error is not None:
            why = "it didn't answer" if ev.error.code in _NOT_ANSWERING else ev.error.message.rstrip(".")
            self._temp_error = f"Couldn't check the temporary folder {self._place(temp)}: {why}."
        else:
            self._temp_free, self._temp_raws = ev.result
            self._temp_error = None
            if self.out_index is not None:
                self._apply_index(self._with_scratch(self.out_index))
        if kind == "final_temp":
            self._finish_final_check()
        self._status2_from_index()
        self._idle_dirty = True

    def _output_committed(self, text: str) -> None:
        if self.state in (RUNNING, STOPPING):
            return
        path = self._typed(text, self.output_field, self.out_dir)
        if path is None:
            return
        self._forget_final_check()
        self.runbar.dismiss()                             # the banner was about the last batch (UD7)
        self._output_token += 1
        self._output_resolving = True
        token = self._output_token
        self._index_gen += 1
        self._index_pending, self._index_error = True, None
        self._status2("spinner", "Checking…")

        def resolved(found: str, error: Optional[FsError]) -> None:
            if error is not None and error.code not in ("not_found", "not_a_dir"):
                self._output_resolving = False
                self._index_pending, self._index_error = False, error.message
                self._ask_again(self.ctx.fs.lane_key(found), error, ("index", path),
                                lambda: token == self._output_token and self._output_committed(path))
                self._status2(*self._place_error(error, found, lambda: self._output_committed(path)))
                self._refresh_idle()
                return
            self._use_output(found)
        _PathResolver(self.ctx, path, resolved, lambda: token == self._output_token)
        self._refresh_idle()

    def _use_output(self, path: str) -> None:
        self._output_resolving = False
        self._forget_final_check()
        self.output_field.set(path)
        self.out_dir, self.out_index, self._write_error, self._temp_free = path, None, None, None
        self._save_error = None
        self.settings.set("paths.output", path)
        if self._plan is not None and self.state == FINISHED:     # DS-15: nothing is stitched in the new folder yet
            self._end_batch()
            self._plan = self._plan.rebuild(path)
            self.table.bind_plan(self._plan)
            self._set_state(READY)
        elif self._plan is not None:
            self._plan.set_out_dir(path)
            self._full = True
        dialogs.SlideDetailsDialog.refresh_rows(plan=self._plan)
        self._request_index()
        self._refresh_idle()

    def _apply_index(self, idx: OutputIndex) -> None:
        """Step 2 and the checks of existing files follow in _after_tick, once the plan has taken the index in."""
        self.out_index, self._index_error, self._save_error = idx, None, None
        if self._plan is not None:
            self._changed.update(self._plan.apply_output_index(idx))
        if idx.exists and idx.error is None:
            self.settings.add_recent("output", idx.out_dir)
        self._step2_due = True

    def _status2(self, kind: Optional[str], text: str, actions: Sequence[Tuple[str, Callable[[], None]]] = ()) -> None:
        self.output_field.set_status(kind, text, _STYLE.get(kind or "", "Surface.Muted.TLabel"), actions)

    def _root_matches(self, here: FrozenSet[str] = frozenset()) -> int:
        """Stitched files in the input folder itself whose names match this plan's slides (§3.6), not counting
        names already in `here` (the chosen output folder)."""
        s = self.ctx.fs.cached_summary(self.source) if self.source and self._plan is not None else None
        planned = self._planned_names() - here if s is not None else frozenset()
        return sum(norm_key(name) in planned for name in (s.outputs if s is not None else ()))

    def _status2_from_index(self) -> None:
        """§2.1.3 step-2 status line, the most important condition first."""
        idx, out = self.out_index, self.out_dir
        if not out:
            chips = []
            if self.source:
                chips.append(('New "stitched" folder next to the tiles',
                              lambda: self.set_output(os.path.join(self.source, "stitched"))))
                n = self._root_matches()
                if n:
                    chips.append((f"Save next to the {_n(n, 'stitched file')} already in {self._place(self.source)}",
                                  lambda: self.set_output(self.source)))
            last = self._recents("outputs")
            if last:
                name = os.path.basename(last[0].rstrip("/\\")) or last[0]
                chips.append((f"{name} (used last time)", lambda: self.set_output(last[0])))
            self._status2(None, "A local disk is fastest.", chips[:3])
            return
        if idx is None or not same_path(idx.out_dir, out):
            return
        free = f"{human_bytes(idx.free_bytes, idx.lane_class == NETWORK)} free" if idx.free_bytes is not None else ""
        if self._write_error or idx.error:
            self._status2("error", f"Can't save here: {self._write_error or idx.error.message}")
        elif idx.pdir_problem:                             # DS-10: a plan issue blocks Stitch too
            self._status2("error", f"Can't save here: {idx.pdir_problem[:1].lower()}{idx.pdir_problem[1:]}")
        elif self._temp_error:
            self._status2("error", self._temp_error, [("Try again", self._request_index),
                                                       ("Settings", self.open_settings)])
        elif self._save_error:
            self._status2("error", *self._save_error)
        elif any(TILE_RE.match(n) for n in idx.names):
            self._status2("warn", "This folder holds tiles; stitched images would be mixed in.",
                          [('Use a "stitched" subfolder', lambda: self.set_output(os.path.join(out, "stitched")))])
        elif idx.leftovers:
            size = sum(info.size for _p, info in idx.leftovers)
            self._status2("warn", f"{_n(len(idx.leftovers), 'leftover file')} ({human_bytes(size)}) from an "
                          "unfinished run.", [("Delete them", self._delete_leftovers)])
        elif not idx.exists and idx.missing_levels > 1 and idx.nearest_existing:    # a typo? (MW-26)
            near = idx.nearest_existing
            name = os.path.basename(near.rstrip("/\\")) or near     # by its name: the field shows the whole path
            self._status2("warn", f"{idx.missing_levels} new folders will be created inside {name}: "
                                  f"{rel_folder(out, near)}")
        elif not idx.exists:
            self._status2("info", "This folder doesn't exist yet — it will be created.")
        elif idx.lane_class == CLOUD:
            self._status2("warn", "Cloud folder: stitched files will upload in the background.")
        else:
            parts, clauses = ["Network drive" if idx.lane_class == NETWORK else "Local disk", free], \
                self._existing_clauses()
            if idx.outputs and not clauses:     # else the slides' clauses say it, without counting the files twice
                parts.append(f"{_n(len(idx.outputs), 'stitched image')} already here")
            parts += clauses
            if idx.lane_class == NETWORK:
                parts.append("saving over the network is slower")
            n = self._root_matches(frozenset(idx.outputs)) if self.source and not same_path(self.source, out) else 0
            more = [("Save there instead", lambda: self.set_output(self.source))] if n else []
            if n:
                parts.append(f"{n} of these slides already {'has' if n == 1 else 'have'} stitched files in "
                             f"{self._place(self.source)}")
            self._status2("info", " · ".join(p for p in parts if p), more)

    def _existing_clauses(self) -> List[str]:
        """Step 2 about the slides that have a file of their name here (UD7, MW-12): only checked files count as
        stitched, and those with tiles missing are said to be (DS-1); files that look incomplete or couldn't be
        checked are said apart."""
        slides, c = len(self._slides()), self._plan.existing_counts() if self._plan is not None else {}
        done, broken, unchecked, checking = (c.get(k, 0) for k in ("valid", "incomplete", "unchecked", "checking"))
        out = []
        if done:
            gaps = self._with_gaps()
            out.append(("your slide is already stitched here" if slides == 1 else
                        f"all {slides:,} of your slides are already stitched here" if done == slides else
                        f"{done:,} of your {slides:,} slides {'is' if done == 1 else 'are'} already stitched here") +
                       ("" if not gaps else " (with gaps)" if slides == 1 else f" ({gaps:,} with gaps)"))
        if broken:
            out.append("1 existing file looks incomplete — it will be redone" if broken == 1 else
                       f"{broken:,} existing files look incomplete — they will be redone")
        if unchecked:
            out.append(f"couldn't check {_n(unchecked, 'existing file')}")
        if checking:
            out.append(f"checking {_n(checking, 'existing file')}…")
        return out

    def _with_gaps(self) -> int:
        """Valid images known to lack tiles, including this plan's results (DS-1)."""
        return self._plan.existing_counts()["gaps"] if self._plan is not None else 0

    def _verify_existing(self) -> None:
        """R4: check the existing output of every row that has no verdict on it yet (also a row ticked to replace
        it, or renamed onto it), once per version of the file; ledger hits (with the tiles that were missing when
        the file was made, DS-1) and outputs younger than RECENT_OUTPUT_S count as valid without I/O (§3.8).
        Partials to redo or move into place are never checked."""
        idx, plan = self.out_index, self._plan
        if plan is None or idx is None or self.state not in (READY, FINISHED):
            return
        for r in plan.rows():
            e = r.existing
            if e is None or r.existing_verdict != "checking" or r.unfinished or r.ready:
                continue
            path = os.path.join(idx.out_dir, e.name)
            key = (path, e.size, e.mtime_ns)
            if ("check", key) in self._asked:
                continue
            gaps = self.settings.ledger_get(*key)
            if gaps is not None or time.time() - e.mtime_ns / 1e9 < RECENT_OUTPUT_S:
                self._changed.update(plan.apply_verification(path, "valid", "", gaps or 0, size=e.size,
                                                             mtime_ns=e.mtime_ns))
            else:
                self._check(path, key)

    def _check(self, path: str, key: Tuple[str, int, int]) -> None:
        if ("check", key) in self._asked:
            return
        self._asked.add(("check", key))
        self.ctx.fs.call(path, "check_output", partial(preflight.verify_output, path), ticket=self._ticket,
                         owner="main", tag=("check", self._gen, key), priority=7)

    def _delete_leftovers(self) -> None:
        idx = self.out_index
        if idx is None or not idx.leftovers:
            return
        token = (self._index_gen, self.out_dir, self.settings.get("run.temp_dir"))
        names = [f"{os.path.basename(p)}  ({human_bytes(info.size)})" for p, info in idx.leftovers[:10]]
        more = len(idx.leftovers) - len(names)
        text = ("These files were left behind by an unfinished run:\n" + "\n".join(names) +
                (f"\n…and {more} more" if more > 0 else "") +
                "\n\nFiles changed in the last 2 minutes are kept (another stitcher may be using them).")

        def go(ok: Optional[bool]) -> None:
            if ok:
                if token != (self._index_gen, self.out_dir, self.settings.get("run.temp_dir")):
                    self._save_failed("The folders changed. Review the leftover files before deleting them.",
                                      self._delete_leftovers, "Review files")
                    return
                self._save_error = None
                groups: Dict[str, list] = {}
                for path, info in idx.leftovers:
                    groups.setdefault(os.path.dirname(path), []).append((path, info))
                for parent, items in groups.items():
                    self.ctx.fs.retry(parent)
                    if self.ctx.fs.lane_key(parent) != self.ctx.fs.lane_key(idx.out_dir):
                        future = preflight.delete_scratch(self.ctx.fs, items, out_dir=idx.out_dir,
                                                           out_identity=idx.identity)
                        future.add_done_callback(partial(_scratch_deleted, self.ctx.bus.post, parent, idx.out_dir))
                    else:
                        self.ctx.fs.call(parent, "delete", partial(preflight.delete_leftovers, items,
                                                               out_dir=idx.out_dir, out_identity=idx.identity),
                                     ticket=self.ctx.fs.new_ticket(), owner="main", tag=("delete", idx.out_dir),
                                     priority=0, user_initiated=True)
        _ConfirmDialog(self.root, self.ctx, "Delete leftover files?", text, "Delete them", go)

    def _finalize(self, row_ids: Sequence[str]) -> None:
        """*Finalize* (DS-9): move the rows' finished images into place, in the save folder's lane: one that a run
        had to keep (its old file was in use), or one that an earlier run left ('ready')."""
        plan, moves, wanted = self._plan, [], set(row_ids)
        for r in plan.rows() if plan is not None else []:
            if r.row_id not in wanted:
                continue
            o = r.outcome
            if r.status == "finalize" and o is not None and o.partial_path and o.output_path:
                moves.append((r, o.partial_path, o.output_path))
            elif r.status == "ready" and r.ready is not None and plan.out_dir:
                moves.append((r, os.path.join(plan.out_dir, PARTIAL_DIR, r.ready.name),
                              os.path.join(plan.out_dir, r.existing.name if r.replaces else r.output_stem + OME_SUFFIX)))
        for r, kept, final in moves:
            if final in self._finalizing:
                continue
            self._finalizing.add(final)
            self.ctx.fs.retry(final)                      # a click is a Try again for a lane that stopped answering
            self.ctx.fs.call(final, "check_output", partial(preflight.destination_identity, final),
                             ticket=self.ctx.fs.new_ticket(), owner="main",
                             tag=("finalize_check", self._gen, r.row_id, kept, final, plan.label(r)),
                             priority=0, user_initiated=True)
        if moves:
            self._save_error = None
            what = plan.label(moves[0][0]) if len(moves) == 1 else _n(len(moves), "finished image")
            self._status2("spinner", f"Moving {what} into place…")

    def _on_finalize_checked(self, ev: FsDone) -> None:
        _, gen, row_id, kept, final, name = ev.tag
        self._finalizing.discard(final)
        if gen != self._gen or not self.out_dir or not same_path(os.path.dirname(final), self.out_dir):
            return
        if ev.error is not None:
            self._save_failed(f"Couldn't check the destination for {name}: {ev.error.message}",
                              lambda: self._finalize([row_id]))
            return
        expected = ev.result

        def move(ok: Optional[bool]) -> None:
            if not ok or gen != self._gen or not self.out_dir or not same_path(os.path.dirname(final), self.out_dir):
                self._status2_from_index()
                return
            self._finalizing.add(final)
            self.ctx.fs.call(final, "finalize", partial(_finalize_file, kept, final, expected, expected is not None),
                             ticket=self.ctx.fs.new_ticket(), owner="main",
                             tag=("finalize", row_id, final, name), priority=0, user_initiated=True)
            self._status2("spinner", f"Moving {name} into place…")
        if expected is None:
            move(True)
        else:
            self._status2("warn", f"Finalize will replace the existing image for {name}.")
            _ConfirmDialog(self.root, self.ctx, "Replace the existing image?",
                           f"The finished image for {name} will replace this file:\n\n{final}\n\n"
                           "Cancel keeps both images.", "Replace image", move)

    def _save_failed(self, text: str, again: Callable[[], None], label: str = "Try again") -> None:
        """[Delete them] or Finalize failed (UD1): step 2 says why, with the next step, until the next index."""
        self._save_error = (text, [(label, again)])
        self._status2("error", text, self._save_error[1])

    # ------------------------------------------------------------------------------------------ FS results
    def _on_fs_done(self, ev: FsDone) -> None:
        """Every request ends with one FsDone, at its deadline FsError('timeout') (UD1): each kind leaves its waiting
        state on an error too."""
        if ev.owner != "main" or not isinstance(ev.tag, tuple):
            return
        what = ev.tag[0]
        if what == "locate":
            self._on_located(ev, *ev.tag[1:])
        elif what == "index":
            self._on_index(ev)
        elif what == "final":
            if ev.tag[1:] == self._final:                  # not an older check, nor one that a change made void
                self._on_final_check(ev)
        elif what in ("temp_index", "final_temp"):
            self._on_temp_index(ev)
        elif what == "delete":
            self._on_deleted(ev)
        elif what == "header":
            self._on_header(ev)
        elif what == "check":
            self._on_checked(ev)
        elif what == "finalize":
            self._on_finalized(ev)
        elif what == "finalize_check":
            self._on_finalize_checked(ev)

    def _on_index(self, ev: FsDone) -> None:
        _, gen, path = ev.tag
        if gen != self._index_gen:
            return
        self._output_pending, self._index_pending, self._idle_dirty = False, self._temp_pending, True
        self._ask_again(self.ctx.fs.lane_key(path), ev.error, ("index", path),
                        lambda: gen == self._index_gen and self._request_index())
        if ev.error is None:
            self._apply_index(self._with_scratch(ev.result))
            return
        self.out_index, self._index_error = None, ev.error.message     # blocks Stitch: nothing is known about it
        kind, text, actions = self._place_error(ev.error, path, self._request_index)
        self._status2(kind, text.replace("this folder", "the save folder"), actions)

    def _on_deleted(self, ev: FsDone) -> None:
        if ev.error is None:
            n, freed, errors = ev.result
            log.info("Deleted %s (%s).%s", _n(n, "leftover file"), human_bytes(freed),
                     (" Kept: " + " ".join(errors)) if errors else "")
            self._request_index()
        elif self.out_dir and same_path(ev.tag[1], self.out_dir):   # UD1: said where the user asked for it
            self._save_failed(f"Couldn't delete the leftover files: {ev.error.message}", self._delete_leftovers)

    def _on_header(self, ev: FsDone) -> None:
        _, gen, row_id, key = ev.tag
        if gen != self._gen or self._plan is None:
            return                                         # a result for a plan that was rebuilt since
        self._header_lanes[key].busy -= 1
        self._send_headers(key)
        if ev.error is not None and ev.error.code == "cancelled":
            return
        header, ram, bad = ev.result if ev.error is None else (None, None, "")
        self._ram = ram or self._ram
        why = bad or (ev.error.message if ev.error is not None else None)
        self._changed.update(self._plan.apply_header(row_id, header, why, self._ram, retryable=not bad))
        self._ask_again(key, ev.error, ("header", row_id),
                        lambda: gen == self._gen and self._read_header(row_id, force=True))
        if self._final_result is not None:
            self._finish_final_check()

    def _on_checked(self, ev: FsDone) -> None:
        _, gen, key = ev.tag
        if gen != self._gen or self._plan is None:
            return
        self._asked.discard(("check", key))                # a discarded answer may need checking on a return here
        path, size, mtime_ns = key
        if ev.error is None:
            verdict, detail = ev.result
        else:                                              # UD1: a check that didn't happen proves nothing
            verdict, detail = "unchecked", ev.error.message
        self._ask_again(self.ctx.fs.lane_key(path), ev.error, ("check", key),
                        lambda: gen == self._gen and self._check(path, key))
        self._changed.update(self._plan.apply_verification(path, verdict, detail, size=size, mtime_ns=mtime_ns))
        self._step2_due = True
        if verdict == "valid":
            self.settings.ledger_add(path, size, mtime_ns)

    def _on_finalized(self, ev: FsDone) -> None:
        """The image is in place: the ledger gets it with its gaps (DS-1), and the save folder is indexed again; or
        step 2 says why not, with the next step (UD1). The answer may come after a rescan or another save folder:
        the moved file still counts for the ledger, its row only while it waits in this plan."""
        _, row_id, final, name = ev.tag
        self._finalizing.discard(final)
        here = bool(self.out_dir) and same_path(os.path.dirname(final), self.out_dir)
        moved, why = ev.result if ev.error is None else (None, "")
        if moved is None:
            if ev.error is None:                           # it no longer waits there, or it is damaged: see anew
                if why == _DESTINATION_CHANGED:
                    again, label = (lambda: self._finalize([row_id])), "Review replacement"
                else:
                    again, label = self.rescan, "Rescan"
            else:
                in_use = ev.error.code == "no_access" and osutil.WINDOWS      # os.replace onto an open file
                why = "The old file is still in use (QuPath?). Close it, then try again." if in_use else \
                    ev.error.message
                again, label = (lambda: self._finalize([row_id])), "Try again"
            log.warning("%s could not be moved into place: %s", name, why)
            if here:
                self._save_failed(f"Couldn't finalize {name}: {why}", again, label)
                rows = self._plan.rows() if self._plan is not None else []
                row = next((r for r in rows if r.row_id == row_id and r.status == "finalize"), None)
                if row is not None and row.outcome is not None and same_path(row.outcome.output_path or "", final):
                    self._plan.finish(row_id, replace(row.outcome, message=why))
                    self._changed.add(row_id)
                    if self._summary is not None:
                        self._show_result(self._summary)
            return
        size, mtime_ns, gaps = moved
        self.settings.ledger_add(final, size, mtime_ns, gaps)
        log.info("%s was moved into place.", name)
        if self._plan is not None:
            self._changed.update(self._plan.apply_verification(final, "valid", "", gaps, size=size,
                                                               mtime_ns=mtime_ns))
            self._finalized(row_id, final, size, gaps)
        if here:
            self._request_index()

    def _finalized(self, row_id: str, final: str, size: int, gaps: int) -> None:
        """A slide of a run that waited for Finalize is done: its row, and the banner and title of that batch."""
        try:
            row = self._plan.row(row_id)
        except KeyError:                                   # a rescan found other slides meanwhile
            return
        if row.status != "finalize" or row.outcome is None or not same_path(row.outcome.output_path or "", final):
            return
        text = f"Done with gaps: {_n(gaps, 'tile')} couldn't be read" if gaps else f"Done · {human_bytes(size)}"
        outcome = replace(row.outcome, status="done_gaps" if gaps else "done", message=text, output_bytes=size,
                          failed_tiles=gaps, partial_path=None)
        self._plan.finish(row_id, outcome)
        self._changed.add(row_id)
        s = self._summary
        if s is not None and any(rid == row_id for rid, _o in s.rows):
            self._show_result(s)

    def _on_fs_slow(self, ev: FsSlow) -> None:
        """A slow input or output folder: say so in its status line until it answers, or until the FsDone of its
        deadline says that it didn't."""
        if ev.owner != "main" or ev.past_deadline or not isinstance(ev.tag, tuple):
            return
        waiting = f"Waiting for {self._place(ev.path)} to answer…"
        if ev.tag[0] == "locate" and self.state == CHECKING and ev.tag[1] == self._locate_token:
            self._status1("spinner", waiting)
        elif ev.tag[0] == "index" and ev.tag[1] == self._index_gen:
            self._status2("spinner", waiting)

    def _ask_again(self, lane: str, err: Optional[FsError], key: Any, again: Callable[[], None]) -> None:
        """The answer (err None) or failure of a request that may be asked again (UD1; the core never retries by
        itself). One that failed because its place didn't answer is asked again when its lane answers again; one
        that ran out of time only once in a row, as one that always takes longer than its deadline would stall its
        lane at every recovery: its [Try again] is the user's."""
        if err is None or err.code not in _ASK_AGAIN:
            self._stalled.discard(key)
            self._reask.get(lane, {}).pop(key, None)
        elif err.code == "unresponsive" or key not in self._stalled:
            self._reask.setdefault(lane, {})[key] = (err.code == "timeout", again)
            self._later("recover:" + lane, 0, lambda: self._recover_lane(lane))

    def _on_lane_health(self, ev: LaneHealthChanged) -> None:
        """The requests refused while the lane wasn't answering go first, those that ran out of time last: one that
        stalls again then fails no other with it."""
        if ev.health in (H_OK, H_SLOW):
            self._recover_lane(ev.lane)

    def _recover_lane(self, lane: str) -> None:
        health, since = self.ctx.fs.health_snapshot(lane)
        if health in (H_OK, H_SLOW):
            for key, (stalled, again) in sorted(self._reask.pop(lane, {}).items(), key=lambda item: item[1][0]):
                if self._recovery_seen.get((lane, key)) == since:
                    self._reask.setdefault(lane, {})[key] = (stalled, again)
                    continue
                self._recovery_seen[lane, key] = since
                if stalled:
                    self._stalled.add(key)
                again()

    def _on_mounts_changed(self, _ev: MountsChanged) -> None:
        """UD2: the widgets read ctx.mounts; an index made under the old table may describe another volume (a share
        mounted on the folder since), so the save folder is indexed again when its lane changed."""
        fs, out, idx = self.ctx.fs, self.out_dir, self.out_index
        self.ctx.mounts = fs.mounts
        if out and idx is not None and self.state not in (RUNNING, STOPPING) and \
                (idx.lane, idx.lane_class) != (fs.lane_key(out), fs.lane_class(out)):
            self._request_index()
        self._idle_dirty = True

    # ------------------------------------------------------------------------------------------ plan changes
    def _plan_edited(self, full: bool = False) -> None:
        """A user change in the table or a dialog: legacy headers once ticked, waiting rows unticked during a run, a
        file that a new output name now matches checked (MW-20), and step 2's counts."""
        plan = self._plan
        dialogs.SlideDetailsDialog.refresh_rows(plan=plan)
        if plan is None:
            return
        for r in plan.ticked():
            if r.cand.kind == "legacy_all" and r.header is None and r.header_error is None:
                self._read_header(r.row_id)
        if self.model is not None and self.state in (RUNNING, STOPPING):
            for r in plan.rows():
                if r.status == "waiting" and not r.checked and r.row_id not in self._dropped:
                    self._dropped.add(r.row_id)
                    self.model.drop(r.row_id)
        self._verify_existing()
        self._status2_from_index()
        self._full = self._full or full
        if full or self._changed:
            self._after_tick()
        else:
            self._refresh_idle()

    def _tick_plan(self) -> Plan:
        """The plan inside this UI tick's plan.batch(), opened by the first candidate of the tick: a tick's candidates
        (and the headers and verdicts that join them) then cost one recompute instead of one each (MW-8, FS-16).
        _after_tick closes it, and so does everything that needs the rows derived before that (_end_batch)."""
        if self._batch is None:
            self._batch = contextlib.ExitStack()
            self._batch_ids = self._batch.enter_context(self._plan.batch())
        return self._plan

    def _end_batch(self) -> None:
        if self._batch is not None:
            batch, self._batch = self._batch, None
            batch.close()
            self._changed.update(self._batch_ids)

    def _after_tick(self) -> None:
        """Once per pump tick that delivered events: the tick's plan batch ends, the checks and step 2 follow a new
        index or verdict (the checks first: ledger hits count at once, and step 2 counts them), then one table refresh
        for every row changed in that tick."""
        self._end_batch()
        if self._step2_due:
            self._step2_due = False
            self._verify_existing()
            self._status2_from_index()
        if self._plan is not None and (self._changed or self._full):
            ids = None if self._full else list(self._changed)
            self.table.refresh(ids)
            dialogs.SlideDetailsDialog.refresh_rows(list(self._changed) if ids is not None else
                                                    [r.row_id for r in self._plan.rows()], plan=self._plan)
            self._changed.clear()
            self._full = False
            self._idle_dirty = True
        if self.state in (RUNNING, STOPPING):
            self._update_run_bar()
        elif self._idle_dirty and (self.state != SCANNING or time.monotonic() - self._idle_at > 0.5):
            self._refresh_idle()

    def _later(self, name: str, ms: int, fn: Callable[[], None]) -> None:
        """One pending timer per name (a new one replaces it), cancelled with the root (a timer firing into a
        destroyed root errs)."""
        if name in self._timers:
            self.root.after_cancel(self._timers[name])
        self._timers[name] = self.root.after(ms, fn)

    def _stop_timers(self) -> None:
        for after_id in self._timers.values():
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self._timers.clear()

    def _minute(self) -> None:
        """Once a minute: the time-based flags (recent_change) and the issues built on them, and the mount table
        (UD2: a share mounted or gone since)."""
        self.ctx.fs.reload_mounts()
        if self._plan is not None and self.state in (READY, FINISHED):
            self._changed.update(self._plan.refresh())
            self._idle_dirty = True
            self._after_tick()
        self._later("minute", 60000, self._minute)

    def _autosave(self) -> None:
        """The 1 s settings debounce: whatever changed is written by the settings thread."""
        if self.settings.dirty:
            self.settings.save_async()
        self._later("autosave", 1000, self._autosave)

    # ------------------------------------------------------------------------------------------ run bar (idle)
    def _env(self) -> PlanEnv:
        out, idx, mounts = self.out_dir, self.out_index, self.ctx.fs.mounts
        idx = idx if idx is not None and out and same_path(idx.out_dir, out) else None
        temp = self.settings.get("run.temp_dir") or None
        if temp:
            temp = self._temp_location or temp
        tvol = temp or out
        same = not temp or not out or mounts.lane_key(temp) == mounts.lane_key(out)
        return PlanEnv(self.source, out, self.state == SCANNING, self.state == PAUSED, idx, self._index_pending,
                       self._write_error or self._index_error, self._temp_free,
                       mounts.lane_class(tvol) if tvol else None, same, osutil.WINDOWS, self._ram,
                       unsearched=self._unsearched, windows_long_paths=self._long_paths, temp_dir=temp)

    def _issues(self) -> List[Issue]:
        if self._plan is None:
            out = [Issue("block", "scanning", "Checking the folder…")] if self.state == CHECKING else \
                [Issue("block", "no_source", "Choose the folder with your tiles.", (), "browse_input")]
            if not self.out_dir:
                out.append(Issue("block", "no_output", "Choose where to save the stitched images.", (),
                                 "browse_output"))
            return out
        check = [Issue("warn", "final_check", self._check_error, (), "retry_check")] if self._check_error else []
        if self._temp_error:
            check.append(Issue("block", "temp_error", self._temp_error, (), "settings"))
        return check + self._plan.issues(self._env())

    def _refresh_idle(self) -> None:
        """Issue line, totals and the Stitch button (§2.4.1)."""
        self._idle_dirty, self._idle_at = False, time.monotonic()
        if self.state in (RUNNING, STOPPING):
            return
        issues = [i for i in self._issues() if i.severity in ("block", "warn")]
        first = primary_issue(issues)
        hidden = [i for i in issues if i is not first]      # behind '+N more…' (MW-11)
        plan, idx = self._plan, self.out_index
        n = len(plan.pending()) if plan else 0             # MW-7: what Stitch would run
        totals = ""
        if plan is not None and n:
            vol = os.path.basename(idx.lane.rstrip("/\\")) if idx is not None else ""     # "" for "/"
            totals = totals_text(plan.totals(), idx.free_bytes if idx is not None else None, vol)
        self.runbar.show_idle(n, totals, first, len(hidden), hidden)
        if self._final is not None:
            self._show_checking()
        if self.state != FINISHED:
            self.root.title(APP_NAME)

    def _show_checking(self) -> None:
        """The Stitch button while the final check runs: 'Checking…', disabled, also across idle refreshes."""
        if self.runbar.mode == "idle":
            self.runbar.button.configure(text=self.ctx.theme.text("Checking…"))
            self.runbar.button.state(["disabled"])

    def _enable_folders(self, on: bool) -> None:
        self.input_field.set_enabled(on)
        self.output_field.set_enabled(on)

    def _forget_final_check(self) -> None:
        """A folder or the temporary folder changed: a final check under way no longer counts (CT-5), and its error
        no longer applies."""
        self._check_error = None
        self._final_result = None
        self._final_temp_pending = False
        if self._final is not None:
            self._final = None
            self._enable_folders(True)
            self._refresh_idle()

    def _issue_action(self, action: str) -> None:
        if action == "browse_input":
            self.browse_input()
        elif action == "browse_output":
            self.browse_output()
        elif action == "stop_scan":
            self.stop_scan()
        elif action == "resume_scan":
            self.resume_scan()
        elif action == "rescan":
            self.rescan()
        elif action == "fix_channels" and self._plan is not None:
            issue = next((i for i in self._issues() if i.code == "channels"), None)
            self.fix_channels(list(issue.rows) if issue else [])
        elif action == "settings":
            self.open_settings()
        elif action == "delete_leftovers":
            self._delete_leftovers()
        elif action == "retry_check":
            self.start(self._confirm)
        elif action == "quit_anyway":
            self._hard_exit()
        elif action == "finalize" and self._plan is not None:
            self._finalize([r.row_id for r in self._plan.rows() if r.status == "finalize"])

    def _ctrl_enter(self) -> None:
        if self.runbar.mode == "idle" and self.runbar.button.instate(["!disabled"]):
            self.runbar.button.invoke()

    def _settings_done(self, keys: Set[str]) -> None:
        if "run.temp_dir" in keys and self.state not in (RUNNING, STOPPING):
            self._forget_final_check()                    # it checked the old temporary folder (CT-5)
            self._request_index()                         # whose leftovers it offered (UD7)
        if "discovery.naming" in keys and self._plan is not None:
            self._plan.set_naming("slide" if self.settings.get("discovery.naming") == "slide" else "folder")
            self._plan_edited(full=True)
        self._refresh_idle()

    # ------------------------------------------------------------------------------------------ starting a run
    def _on_final_check(self, ev: FsDone) -> None:
        """The final check's answer, for the folders it checked (the tag; UD7). One that couldn't be done brings the
        Stitch button back with the reason and [Try again] (UD1): it says nothing about the folder."""
        temp = self.settings.get("run.temp_dir") or None
        if self.state not in (READY, FINISHED) or self._plan is None or ev.tag[2:] != (self.out_dir, temp):
            self._refresh_idle()
            return
        self._end_batch()
        if ev.error is not None:
            self._forget_final_check()
            why = "it didn't answer" if ev.error.code in _NOT_ANSWERING else ev.error.message.rstrip(".")
            self._check_error = f"Couldn't check {self._place(self.out_dir)}: {why}."
            self._refresh_idle()
            return
        self._write_error, idx, _unused_free = ev.result
        self._index_gen, self._index_pending = self._index_gen + 1, False     # supersedes a pending index
        self._apply_index(self._with_scratch(idx))
        self._final_result = ev
        self._finish_final_check()

    def _stitch_rows(self) -> list:
        """Recovery of an already complete image needs no tile header."""
        return [r for r in self._plan.pending() if r.ready is None and r.status != "finalize"] \
            if self._plan is not None else []

    def _finish_final_check(self) -> None:
        if self._final is None or self._final_result is None or self._plan is None:
            return
        if self._final_temp_pending:
            return
        self._end_batch()
        rows = {r.row_id: r for r in self._plan.rows()}
        selected = [rows[rid] for rid in self._final_rows if rid in rows]
        failed = [self._plan.label(r) for r in selected
                  if r.header_error and r.ready is None and r.status != "finalize"]
        if failed:
            self._forget_final_check()
            self._check_error = "Couldn't read the first tile for " + ", ".join(failed) + ". Review these rows first."
            self._refresh_idle()
            return
        pending = [r for r in self._stitch_rows() if r.header is None]
        if pending:
            for r in pending:
                self._read_header(r.row_id)
            self._refresh_idle()
            self.runbar.button.configure(text=self.ctx.theme.text("Checking tile headers…"))
            return
        self._forget_final_check()
        self._present_start()

    def _start_decision(self) -> tuple:
        return (self._gen, self.out_dir, self.settings.get("run.temp_dir"),
                tuple(r.row_id for r in self._plan.pending()),
                tuple((i.code, i.text, i.rows) for i in self._issues() if i.severity == "warn"))

    def _present_start(self) -> None:
        issues = self._issues()
        if any(i.severity == "block" for i in issues):
            self._status2_from_index()
            self._refresh_idle()
            return
        warns = [i for i in issues if i.severity == "warn"]
        unresolved = [r.row_id for r in self._stitch_rows()
                      if any(f.code == "channels" and f.severity == "warn" for f in r.flags)]
        decision = self._start_decision()
        if (warns and self._confirm) or unresolved:
            dialogs.ConfirmStartDialog.open(self.root, self.ctx, warns, unresolved,
                                            lambda result: self._sheet_done(result, unresolved, decision))
        else:
            self._sheet_done("start", unresolved, decision)

    def _sheet_done(self, result: str, unresolved: List[str], decision: Optional[tuple] = None) -> None:
        self._end_batch()
        if self.state not in (READY, FINISHED) or self._plan is None:
            return
        if result == "start":
            if any(i.severity == "block" for i in self._issues()):
                self._refresh_idle()
                return
            if any(r.header is None for r in self._stitch_rows()):
                self.start(self._confirm)
                return
            if decision is not None and decision != self._start_decision():
                self.start(self._confirm)                  # new warnings need a fresh decision, never implied consent
                return
            if unresolved:
                self._plan.set_channels(unresolved, None)
            self._launch()
        elif result == "fix_channels":
            self.fix_channels(unresolved)
            self._refresh_idle()
        else:
            self._refresh_idle()

    def _launch(self) -> None:
        plan, s = self._plan, self.settings
        self._run_id += 1
        self.model = ProgressModel({r.row_id: r.est.raw_bytes if r.est else None for r in plan.pending()})
        self._n_jobs = self._started = 0
        self._dropped, self._counts, self._phase, self._stop_mode, self._quit_after = set(), {}, "", "", False
        self._summary = None                              # the last batch's banner is gone
        temp = (self._temp_location or s.get("run.temp_dir")) if s.get("run.temp_dir") else None
        opts = RunOptions(self.out_dir, temp_dir=temp,
                          summary_csv=bool(s.get("run.summary_csv", True)), app_version=__version__)
        self.runner = BatchRunner(self._run_id, plan, self.ctx.engine_cls, self.ctx.bus.post, opts)
        self._set_state(RUNNING)
        self.input_field.set_status("info", "You can change folders after this batch.")
        self._enable_folders(False)
        self.runbar.show_running("Starting…", 0.0, None, None)
        gc.collect()          # CT-6: Tk garbage dies here, on the Tk thread, not in the runner's own collections
        self.runner.start()
        self._later("run", 1000, self._run_timer)

    # ------------------------------------------------------------------------------------------ running
    def _on_run_started(self, ev: RunStarted) -> None:
        if ev.run_id != self._run_id:
            return
        self._n_jobs = ev.n_jobs
        if self.settings.get("run.keep_awake", True):
            osutil.keep_awake(True)
        self._resume = None
        self._notes1 = [x for x in self._notes1 if not x.startswith("Your last batch")]
        self.settings.set("last_run", {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                       "ended": "", "input": self.source or "", "output": self.out_dir or "",
                                       "planned": ev.n_jobs, "done": 0, "failed": 0})
        self.settings.save_async()
        self.drawer.reset_counts()                        # the pills count this batch's problems (MW-22)
        self._full = True

    def _on_row_started(self, ev: RowStarted) -> None:
        if ev.run_id == self._run_id:
            self._started += 1
            self._phase = "reading"
            self._changed.add(ev.row_id)
            self._refresh_stop()

    def _on_row_progress(self, ev: RowProgress) -> None:
        if ev.run_id == self._run_id and self.model is not None:
            self.model.on_progress(ev.row_id, ev.phase, ev.current, ev.total, ev.t)
            self._phase = ev.phase
            self._changed.add(ev.row_id)
            if self.state == STOPPING:
                self._show_stopping()

    def _on_row_bytes(self, ev: RowBytes) -> None:
        if ev.run_id == self._run_id and self.model is not None:
            self.model.on_bytes(ev.row_id, ev.written, ev.expected, ev.t, ev.phase)   # 'pyramid': activity (MW-9)
            self._changed.add(ev.row_id)

    def _on_row_finished(self, ev: RowFinished) -> None:
        if ev.run_id != self._run_id or self.model is None:
            return
        self.model.on_finished(ev.row_id, time.time())
        self._changed.add(ev.row_id)
        status = ev.outcome.status
        self._counts[status] = self._counts.get(status, 0) + 1
        last = self.settings.get("last_run") or {}
        last.update(done=sum(self._counts.get(k, 0) for k in STITCHED if k != "finalize"),
                    failed=self._counts.get("failed", 0))
        self.settings.set("last_run", last)
        self._refresh_stop()

    def _on_output_written(self, ev: OutputWritten) -> None:
        self.settings.ledger_add(ev.path, ev.size, ev.mtime_ns, ev.gaps)      # a later scan tells of them (DS-1)

    def _label(self, row_id: str) -> str:
        """A slide's name in messages: slides that share a name are told apart (Plan.label)."""
        return self._plan.label(self._plan.row(row_id))

    def _run_text(self, now: float) -> Tuple[str, Optional[str]]:
        """'Stitching 2 of 5 · <slide> · <phase> · <time left>'; after a minute without progress the stall notice
        instead of the time left, about the place the engine is waiting for (MW-9)."""
        model, plan, cur = self.model, self._plan, self.runner.current_row if self.runner else None
        n = max(self._started, self._n_jobs - len(self._dropped))
        if cur is None or plan is None or model is None:
            return f"Stitching {self._started} of {n}", None
        row = plan.row(cur)
        text = f"Stitching {self._started} of {n} · {plan.label(row)} · {row.run_text or 'Starting…'}"
        stall = model.stall_notice(now, self._place(self.source or row.cand.folder), self._place(self.out_dir or ""))
        return (text, stall) if stall else (f"{text} · {model.eta_text(now)}", None)

    def _update_run_bar(self) -> None:
        if self.model is None:
            return
        text, stall = self._run_text(time.time())
        self.runbar.show_running(text, self.model.overall(), self.model.slide(), stall)
        n = max(self._started, self._n_jobs - len(self._dropped))
        pct, quitting, t = int(self.model.overall() * 100), self._quitting(), self.ctx.theme.text
        self.root.title(t(f"{pct}% · {quitting or f'{self._started}/{n}'} — {APP_NAME}"))
        self.footer.configure(text=t(quitting or ("Stopping…" if self.state == STOPPING else
                                                  f"Stitching {self._started} of {n}")))

    def _run_timer(self) -> None:
        if self.state in (RUNNING, STOPPING):
            self._update_run_bar()
            self._later("run", 1000, self._run_timer)

    def _quitting(self) -> str:
        """Say when the app will close, including a quit requested after Stop now (MW-19)."""
        if not (self._quit_after and self.state == STOPPING):
            return ""
        if self._stop_mode == "now":
            return "Quitting once the current write step ends…" if self._phase == "writing" else "Quitting…"
        cur = self.runner.current_row if self.runner else None
        return f"Quitting after {self._label(cur)}…" if cur else "Quitting…"

    def _show_stopping(self) -> None:
        cur = self.runner.current_row if self.runner else None
        if self._stop_mode == "now" and self._phase == "writing":
            self.runbar.show_stopping(self._quitting() or
                                     "Stopping… finishing the current write step (can take a few minutes)", True)
        elif self._stop_mode == "now":
            self.runbar.show_stopping(self._quitting() or "Stopping…", False)
        else:
            self.runbar.show_stopping(self._quitting() or (f"Stopping after {self._label(cur)}…" if cur else
                                                           "Stopping…"), False)

    def _ask_stop(self, closing: bool = False) -> None:
        """Stop… or the close button during a run (§2.4.5). Closing again while the app already quits after the
        current slide asks whether to quit at all (MW-19)."""
        if self.runner is None or self.state not in (RUNNING, STOPPING):
            return
        target = self._current_stop_target()
        _, cur, done = target
        slide = self._label(cur) if cur else "this slide"
        eta = self.model.remaining_s(time.time(), cur) if self.model is not None and cur else None
        self._dismiss_stop()                              # the newest question replaces an open one
        self._stop_target = target
        self._stop_dialog = dialogs.StopDialog.open(self.root, self.ctx, slide, eta, done, closing,
                                                    lambda choice: self._stop_chosen(choice, closing, target),
                                                    quitting_after=closing and self._quit_after,
                                                    stopping_now=self._stop_mode == "now")

    def _current_stop_target(self) -> Tuple[int, Optional[str], int]:
        return (self._run_id, self.runner.current_row if self.runner else None,
                sum(self._counts.get(k, 0) for k in STITCHED if k != "finalize"))

    def _refresh_stop(self) -> None:
        if self._stop_dialog is not None and self.state in (RUNNING, STOPPING) \
                and self.runner is not None and self.runner.current_row is not None \
                and self._stop_target != self._current_stop_target():
            self._ask_stop(self._stop_dialog.closing)

    def _dismiss_stop(self) -> None:
        if self._stop_dialog is not None:
            self._stop_dialog.dismiss()
            self._stop_dialog = None
        self._stop_target = None

    def _stop_chosen(self, choice: str, closing: bool,
                     target: Optional[Tuple[int, Optional[str], int]] = None) -> None:
        """'keep' | 'after_current' | 'now' | 'cancel_quit' (MW-19: the app stays open, and the stop goes on). An
        answer to quit that comes after the batch ended still quits (CT-8)."""
        if target is not None and target[0] != self._run_id:
            return                                         # an old question cannot stop a new batch
        if self.state in (RUNNING, STOPPING) and choice in ("after_current", "now") \
                and target is not None and target != self._current_stop_target():
            self._ask_stop(closing)                         # even a transition whose event has not arrived yet
            return
        self._stop_dialog, self._stop_target = None, None
        if self.state not in (RUNNING, STOPPING):
            if closing and choice in ("after_current", "now"):
                self.close()
        elif choice == "cancel_quit":
            self._quit_after = False
            self._show_stopping()
            self._update_run_bar()
        elif choice in ("after_current", "now"):
            if not self.request_stop(choice, target):
                self._ask_stop(closing)
                return
            if closing and choice == "after_current":
                self._quit_after = True
                self._show_stopping()
                self._update_run_bar()
            if closing and choice == "now":
                self._quit_when_idle(time.monotonic() + 3.0)

    def _quit_when_idle(self, deadline: float) -> None:
        """Quit now: wait (UI responsive) up to 3 s for the current slide to end, then save and exit."""
        if self.runner is not None and self.runner.current_row is not None and time.monotonic() < deadline:
            self._later("quit", 100, lambda: self._quit_when_idle(deadline))
        else:
            self._hard_exit()

    def _hard_exit(self) -> None:
        """Quit even if settings, logging or a stream never answers; Tk stays responsive until exit."""
        if self._shutdown is not None:
            return
        last = self.settings.get("last_run") or {}
        last.update(state="stopped", ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.settings.set("last_run", last)
        self.settings.set("ui.geometry", self.root.geometry())
        self._shutdown = _Shutdown(self.settings, self.listener, self.ctx.fs, True)
        self.listener = None

    def _on_run_finished(self, ev: RunFinished) -> None:
        if ev.run_id != self._run_id:
            return
        s, summary = self.settings, ev.summary
        self.runner = None
        if s.get("run.keep_awake", True):
            osutil.keep_awake(False)
        last = s.get("last_run") or {}
        last.update(state="stopped" if summary.stop_reason else "finished", ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
        s.set("last_run", last)
        s.save_async()
        if self._quit_after or (self._stop_dialog is not None and not self._stop_dialog.closing):
            self._dismiss_stop()                          # nothing left to stop; a Quit? question still quits (CT-8)
        elif self._stop_dialog is not None:
            self._stop_dialog.finished()
        self._full = True
        self._enable_folders(True)
        self._set_state(FINISHED)
        self._status1(*self._scan_status)
        self._show_result(summary, new=True)
        self.root.bell()
        self._after_tick()
        self._request_index()
        if self._quit_after:
            self.close()

    def _show_result(self, summary: RunSummary, new: bool = False) -> None:
        """A batch's result: the completion banner (a new one, or the one shown), the window title and the footer
        (§2.4.7). They are told again when one of its slides is finalized, so they always agree with its rows."""
        if self._plan is not None:                        # Finalize may have updated a row during this batch
            outcomes = {r.row_id: r.outcome for r in self._plan.rows() if r.outcome is not None}
            rows = tuple((rid, outcomes.get(rid, o)) for rid, o in summary.rows)
            summary = replace(summary, rows=rows, counts=dict(Counter(o.status for _rid, o in rows)))
        self._summary = summary
        if new or self.runbar.mode == "finished":
            self.runbar.show_finished(summary, problem_lines(summary, self._label))
        if self.state == FINISHED:
            c, t, word = summary.counts, self.ctx.theme.text, "Stopped" if summary.stop_reason else "Done"
            gaps = f" ({c['done_gaps']:,} with gaps)" if c.get("done_gaps") else ""
            parts = [f"{sum(c.get(k, 0) for k in ('done', 'done_gaps', 'done_generic')):,} stitched{gaps}"] + [
                f"{c[k]:,} {what}" for k, what in (("failed", "failed"), ("skipped", "skipped"),
                                                  ("cancelled", "stopped"), ("finalize", "to finalize")) if c.get(k)]
            self.root.title(t(f"{word} — {', '.join(parts)} — {APP_NAME}"))
            self.footer.configure(text=word)

    def _open_output(self) -> None:
        out = self.out_dir
        if out:
            osutil.open_path(out, on_error=lambda _msg: self.ctx.bus.post(BackgroundError("open", out)))

    def _new_batch(self) -> None:
        """[Start a new batch]: the log's warning and error pills start again at 0 (MW-22), and the folders are
        searched again."""
        self.drawer.reset_counts()
        self.rescan()

    def _retry_failed(self) -> None:
        """[Retry failed]: tick the failed, skipped and stopped rows, then Stitch (the sheet shows any warnings). Only
        for the batch the banner describes: another folder leaves FINISHED with a new plan (UD7)."""
        if self._plan is not None and self.state == FINISHED:
            self._plan.select("failed")
            self.runbar.dismiss()
            self._plan_edited(full=True)
            self.start()

    def _on_background_error(self, ev: BackgroundError) -> None:
        if ev.where == "open":                             # [Open output folder] failed: show the path
            self.runbar.show_open_error(ev.text)
            return
        self._bg_text = ev.text
        self.error_banner.grid(row=0, column=0, sticky="ew", padx=self.ctx.theme.px(12),
                               pady=(self.ctx.theme.px(6), 0))

    def _copy_error(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self._bg_text)

    # ------------------------------------------------------------------------------------------ state
    def _set_state(self, state: str) -> None:
        self.state = state
        self._idle_dirty = True
        if state not in (RUNNING, STOPPING):
            self.footer.configure(text=self.ctx.theme.text(_FOOTER[state]))
        if state in (EMPTY, CHECKING, SCANNING, PAUSED, READY):
            self.root.title(APP_NAME)
        self.table.set_state(state)                       # its help text, Select menu and tick boxes (MW-16)


# ================================================================================================ main()
def _user_scale(flag: Optional[str], setting: Any) -> Optional[float]:
    """--scale beats settings ui.scale ('auto' or a number); None means automatic."""
    for value in (flag, setting):
        try:
            if value not in (None, "", "auto") and 0.5 <= float(value) <= 4.0:
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def _diag(args: argparse.Namespace) -> int:
    """--diag: the About diagnostics plus `zenity --version` run with the cleaned environment (§5.5.10)."""
    settings = Settings.load()
    try:
        root = tk.Tk()
        root.withdraw()
        ws = str(root.tk.call("tk", "windowingsystem"))
        theme = SimpleNamespace(xft=detect_xft(root), ws=ws,
                                scale=_user_scale(args.scale, settings.get("ui.scale")) or auto_scale(root, ws))
    except tk.TclError as exc:                            # no display: report what is known without Tk
        why = f"unavailable ({exc})"
        root = SimpleNamespace(tk=SimpleNamespace(call=lambda *_a: why), destroy=lambda: None)
        theme = SimpleNamespace(xft=False, ws="none", scale=_user_scale(args.scale, settings.get("ui.scale")) or 1.0)
    lines = dialogs.diagnostics(SimpleNamespace(root=root, theme=theme, settings=settings))
    root.destroy()
    if osutil.LINUX:
        try:
            out = subprocess.run(["zenity", "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=10, env=osutil.clean_env())
            lines.append(("zenity", (out.stdout.strip().splitlines() or [f"exit code {out.returncode}"])[0]))
        except (OSError, subprocess.SubprocessError) as exc:
            lines.append(("zenity", f"not available ({exc})"))
    print(f"{APP_NAME} {__version__} diagnostics")
    for label, value in lines:
        print(f"{label}: {value}")
    return 0


def main(engine_cls: type, argv: Optional[Sequence[str]] = None) -> int:
    """Flags: --input PATH, --output PATH, --scale F, --smoke-test (build the UI, run the event loop for 1.5 s,
    print 'SMOKE OK <version>', exit 0), --diag (print the diagnostics, exit 0). Unknown flags (and --classic) are
    ignored. Order: set_windows_dpi_awareness(); set_windows_error_mode(); Settings.load(); tk.Tk();
    install_theme(); IconSet; UiBus; setup_logging; install_excepthooks; MountTable.load(); FsService; MainWindow."""
    parser = argparse.ArgumentParser(prog=APP_NAME, add_help=False, allow_abbrev=False)
    for flag in ("--input", "--output", "--scale"):
        parser.add_argument(flag)
    for flag in ("--smoke-test", "--diag"):
        parser.add_argument(flag, action="store_true")
    args, _unknown = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if args.diag:
        return _diag(args)
    for name in ("input", "output"):                      # relative to where the app was started (UD6); a file://
        value = getattr(args, name)                       # or smb:// URL is cleaned like typed text instead
        if value and "://" not in value:
            setattr(args, name, os.path.abspath(os.path.expanduser(value)))
    code = _run(engine_cls, args)
    gc.collect()      # free the Tk objects here: collected on a worker thread, Tcl would abort the process
    return code


def _run(engine_cls: type, args: argparse.Namespace) -> int:
    osutil.set_windows_dpi_awareness()
    osutil.set_windows_error_mode()
    settings = Settings.load()
    try:
        root = tk.Tk(className=APP_NAME)
    except tk.TclError as exc:
        print(f"{APP_NAME}: cannot open a window: {exc}", file=sys.stderr)
        return 1
    theme = install_theme(root, _user_scale(args.scale, settings.get("ui.scale")))
    icons = IconSet(root, theme.icon_px)
    install_indicators(root, icons)
    root.iconphoto(True, *icons.app_icons())
    bus = UiBus(root)
    bus.start()
    listener = osutil.setup_logging(log_dir(), bus.post)
    osutil.install_excepthooks(root, bus.post)
    errors: List[str] = []
    if args.smoke_test:                                   # any callback or thread error fails the smoke test
        report = root.report_callback_exception
        root.report_callback_exception = lambda *exc: (errors.append("".join(traceback.format_exception(*exc))),
                                                       report(*exc))
        bus.subscribe(BackgroundError, lambda ev: errors.append(ev.text))
    mounts = MountTable.load()
    fs = FsService(bus.post, mounts)
    ctx = AppContext(root, theme, icons, bus, fs, mounts, settings, engine_cls, lambda: None, frozenset,
                     lambda: False)
    window = MainWindow(ctx, args.input, args.output, restore=not args.smoke_test)
    window.listener = listener
    log.info("%s %s started", APP_NAME, __version__)
    if args.smoke_test:
        root.after(1500, window._quit)
    root.mainloop()
    if window._shutdown is None:
        window._quit()
    while not window._shutdown.done.is_set():
        with contextlib.suppress(tk.TclError):
            root.update()                                # also handles a mainloop ended outside the close callback
        time.sleep(0.01)
    bus.stop()
    try:
        root.destroy()
    except tk.TclError:                                   # already destroyed by the window manager's close button
        pass
    if args.smoke_test:
        for text in errors:
            print(text, file=sys.stderr)
        print(f"SMOKE OK {__version__}" if not errors else f"SMOKE FAILED: {len(errors)} error(s)", flush=True)
        return 0 if not errors else 1
    return 0
