"""The batch runner (FINAL_SPEC §2.4.4-§2.4.7, §5.5.7): stitches the ticked plan rows one at a time.

Each slide is written as <out>/.stitcher-in-progress/<final name>: claimed with O_EXCL plus an owner record, kept
fresh by a heartbeat so another stitcher can tell a live claim from a crash leftover (R29, D3), checked to be still
ours before the engine opens it by name and again once it is fsync'ed, marked complete and moved into place with
os.replace. Exceptions from engine calls are reduced to text at once and never stored (R28): a kept exception keeps
its traceback frames, and they keep the multi-GB canvas alive.
"""
from __future__ import annotations

import csv
import gc
import logging
import os
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass
from os.path import exists, join
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import psutil
import tifffile

from yfs import APP_NAME

from . import osutil, preflight
from .events import OutputWritten, Post, RowBytes, RowFinished, RowProgress, RowStarted, RunFinished, RunStarted
from .fsservice import friendly_error, map_oserror, scan_dir
from .model import DirSummary, Job, RowOutcome, RunSummary, TileGroup
from .names import (COMPLETE_SUFFIX, HEARTBEAT_S, LEVEL0_FACTOR, OUT_WORST_FACTOR, PARTIAL_DIR, RAM_HEADROOM_GB,
                    STALE_HEX, STALE_INFIX, human_bytes, human_duration, norm_key, summarize_listing, temp_subdir)
from .plan import Plan

_clock = time.time                   # the wall clock of events (tests substitute a fake)
_monotonic = time.monotonic          # schedules heartbeats, so a step of the wall clock can't stop them (D3)
_log = logging.getLogger(__name__)
_PHASES = (("Stitching", "stitching"), ("Formatting", "preparing"), ("Writing Full Res", "writing"),
           ("Writing Pyramid", "pyramid"), ("Finished", "finishing"))           # engine phase_text prefixes
CANCELLED_TEXT = "Stopped — will start again next time"
FINALIZE_TEXT = "Finished, but the old file is in use (QuPath?). Close it, then right-click → Finalize"
BUSY_TEXT = "another stitcher may be writing this slide"
TAKEN_OVER_TEXT = "another stitcher took this slide over"
WAITING_TEXT = "a finished image of this slide waits to be moved into place (rescan, then Stitch)"
APPEARED_TEXT = "the output appeared while this slide was being stitched; rescan to choose which to keep"
FOLDER_CHANGED_TEXT = "It is no longer the folder the batch started in (was the drive disconnected?)."
CSV_COLUMNS = ("slide", "source_folder", "output_file", "status", "message", "tiles", "failed_tiles", "channels",
               "um_per_px", "memory_mode", "started", "duration_s", "output_bytes", "app_version",
               "tifffile_version", "numpy_version")


@dataclass(frozen=True)
class RunOptions:
    out_dir: str
    temp_dir: Optional[str] = None
    summary_csv: bool = True
    app_version: str = ""
    replace_retries: int = 5
    poll_s: float = 2.0
    stale_partial_s: float = 120.0


@dataclass(frozen=True)
class Claim:
    path: str                        # <out>/.stitcher-in-progress/<final name>
    ident: Tuple[int, int]           # (st_dev, st_ino) of the file we created
    token: str                       # in our owner record
    parent_id: Optional[Tuple[int, int]] = None


def free_bytes(path: str) -> int:
    """shutil.disk_usage(path).free."""
    return shutil.disk_usage(path).free


def claim_partial(path: str) -> Optional[Tuple[int, int]]:
    """Create `path` exclusively (O_EXCL). Its preflight.path_id if this process now owns it (nobody takes over a
    claim this fresh); None if it already existed."""
    try:
        with preflight.reserve_output(path):
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            return preflight.path_id(path)
    except (FileExistsError, preflight.ReservationBusy):
        return None


def same_volume(a: str, b: str) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def find_group(s: DirSummary, prefix: str, suffix: str) -> Optional[TileGroup]:
    return next((g for g in s.groups if g.prefix == prefix and g.suffix == suffix), None)


_RUN_TEXT = {"reading": "Reading tile positions…", "stitching": "Stitching tiles {0}/{1}",
             "preparing": "Preparing to write…", "writing": "Writing full-resolution image…",
             "pyramid": "Writing pyramid level {0} of {1}", "finishing": "Finishing…"}


def run_text(phase: str, current: int, total: int) -> str:
    """The *What will happen* text of the running row (§2.3.3)."""
    return _RUN_TEXT.get(phase, "Running…").format(current, total)


def _done_text(status: str, gaps: int, size: int, seconds: float) -> str:
    return {"done": f"Done · {human_bytes(size)} · {human_duration(seconds)}",
            "done_gaps": f"Done with gaps: {gaps} tile{'s' if gaps != 1 else ''} couldn't be read",
            "done_generic": "Done · generic channel names (rename in QuPath/Visiopharm)"}[status]


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _raw_name(partial_name: str) -> str:
    """The engine's scratch file name for a partial: X.ome.tif -> X.ome.tmp.raw (its temp_path)."""
    return os.path.splitext(partial_name)[0] + ".tmp.raw"


def _probe_ok(out_dir: str) -> bool:
    try:
        preflight.write_probe(out_dir, create=False)
        return True
    except OSError:
        return False


def _partials_of(pdir: str, final_name: str) -> List[str]:
    """The partials named like `final_name` by norm_key: on a case-sensitive disk, case variants too (DS-13)."""
    key = norm_key(final_name)
    try:
        with os.scandir(pdir) as it:
            return [de.path for de in it if norm_key(de.name) == key]
    except FileNotFoundError:
        return []


def _fingerprint(st: os.stat_result) -> Tuple[int, int, int]:
    """Tells the partial that was judged from a claim made since (a new file: a fresh mtime). Not st_ino: FAT and
    exFAT give a file renamed to a longer name another id."""
    return st.st_dev, st.st_mtime_ns, st.st_size


def _take_over(path: str, judged: os.stat_result, tdir: Optional[str]) -> bool:
    """D3: move a dead or stale partial aside first (os.rename: only one stitcher can win), make sure it is the file
    that was judged, then delete it with its sidecars and scratch file. False if another stitcher got there first."""
    try:
        with preflight.reserve_output(path) as reservation:
            return _take_over_reserved(path, judged, tdir, reservation)
    except (FileNotFoundError, preflight.ReservationBusy):
        return False


def _take_over_reserved(path: str, judged: os.stat_result, tdir: Optional[str], reservation) -> bool:
    if _fingerprint(os.stat(path)) != _fingerprint(judged):
        return False
    aside = f"{path}{STALE_INFIX}{os.urandom(STALE_HEX // 2).hex()}"
    try:
        os.rename(path, aside)
        moved = os.stat(aside)
    except (FileNotFoundError, PermissionError):     # gone meanwhile, or still held open by its writer
        return False
    if _fingerprint(moved) != _fingerprint(judged):  # a claim made since it was judged: put it back
        if not exists(path):
            os.rename(aside, path)
        return False
    reservation.check()
    _remove(aside)
    preflight.remove_sidecars(path)
    for folder in (os.path.dirname(path), tdir):
        if folder:
            _remove(join(folder, _raw_name(os.path.basename(path))))
    return True


class PartialMonitor(threading.Thread):
    """From claim to finish: every heartbeat_s (by time.monotonic) os.utime() of the partial and of the slide's
    scratch file, without explicit times, so an NFS server stamps them with its own clock, and a rewrite of the
    owner record, which every file server stamps with its own clock (R29, D3); and while level 0 or the pyramid is
    written, the partial's size posted as RowBytes with that phase every poll_s (MW-9). No reference to engine or
    canvas."""

    def __init__(self, partial: str, job: Job, post: Post, *, run_id: int = 0, heartbeat_s: float = HEARTBEAT_S,
                 poll_s: float = 2.0, temp_raw: Optional[str] = None) -> None:
        super().__init__(name="yfs-partial-monitor", daemon=True)
        self.partial, self.row_id, self.run_id, self.post = partial, job.row_id, run_id, post
        self.touched = (partial,) + ((temp_raw,) if temp_raw else ())
        self.heartbeat_s, self.poll_s = heartbeat_s, poll_s
        self.expected = int(job.est.raw_bytes * LEVEL0_FACTOR) if job.est else 0   # the runner refines it
        self.phase = ""                          # set by the progress callback
        self.polled_at = 0.0                     # time.monotonic() of the latest poll
        self._halt, self._lock, self._stopped = threading.Event(), threading.Lock(), False

    def run(self) -> None:
        due = _monotonic() + self.heartbeat_s
        while not self._halt.wait(self.poll_s):
            if _monotonic() >= due:
                due = _monotonic() + self.heartbeat_s
                for path in self.touched:
                    try:
                        os.utime(path)
                    except OSError:              # Windows while tifffile holds it (its writes touch it); RAM mode
                        pass
                try:
                    preflight.touch_owner(self.partial)
                except OSError:                  # a share hiccup: the next beat is in time (4 per STALE_PARTIAL_S)
                    pass
            phase = self.phase
            if phase in ("writing", "pyramid"):
                try:
                    size = os.path.getsize(self.partial)
                except OSError:
                    size = -1
                with self._lock:                 # nothing is posted once stop() has returned
                    if size >= 0 and not self._stopped:
                        self.post(RowBytes(self.run_id, self.row_id, size, self.expected, _clock(), phase))
            self.polled_at = _monotonic()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
        self._halt.set()
        if self.is_alive():
            self.join(1.0)


class BatchRunner(threading.Thread):
    """Runs ticked plan rows one at a time. Emits RunStarted, RowStarted, RowProgress, RowBytes, RowFinished,
    OutputWritten, RunFinished (always, even after stop/failure)."""

    def __init__(self, run_id: int, plan: Plan, engine_cls: type, post: Post, opts: RunOptions) -> None:
        super().__init__(name=f"yfs-runner-{run_id}", daemon=True)
        self.run_id, self.plan, self.engine_cls, self.post, self.opts = run_id, plan, engine_cls, post, opts
        self._cancel = threading.Event()         # the engine's cancel_event
        self._after_current = False
        self._stopping: Optional[str] = None
        self._job: Optional[Job] = None          # the slide being stitched
        self._out_id: Optional[Tuple[int, int]] = None   # the save folder the batch started in (DS-8)
        self._tdirs: Set[str] = set()            # scratch subfolders made in the custom temp folder (D4)
        self._tdir_ids = {}
        self._state_lock = plan.lock                     # UI plan batches and row transitions share one lock order
        self._ended = False

    def stop_after_current(self) -> None:
        with self._state_lock:
            self._set_stop("after_current")

    def stop_now(self) -> None:
        """Sets the engine cancel_event; the batch stops after the current slide ends."""
        with self._state_lock:
            self._set_stop("now")

    def _set_stop(self, mode: str) -> None:
        if mode == "now":
            self._stopping = "now"
            self._cancel.set()
        else:
            self._after_current = True
            self._stopping = self._stopping or "after_current"

    def request_stop(self, mode: str, *, expected_row: Optional[str]) -> bool:
        """Accept the displayed slide only while it is still current, before any next slide can be claimed."""
        if mode not in ("now", "after_current"):
            raise ValueError("Unknown stop mode")
        with self._state_lock:
            row = self._job.row_id if self._job is not None else None
            if self._ended or row != expected_row:
                return False
            self._set_stop(mode)
            return True

    @property
    def current_row(self) -> Optional[str]:
        with self._state_lock:
            return self._job.row_id if self._job is not None else None

    @property
    def stopping(self) -> Optional[str]:
        """None | 'after_current' | 'now'."""
        return self._stopping

    # ---- the thread body
    def run(self) -> None:
        started, results, reason, out_side = _clock(), [], None, 0
        try:
            self._note_save_folder()
            self.post(RunStarted(self.run_id, self.plan.begin_run(), started))
            while True:
                with self._state_lock:
                    if self._after_current or self._cancel.is_set():
                        break
                    job = self.plan.claim_next(self.opts.out_dir, self.opts.temp_dir)
                    self._job = job
                    if job is None:
                        self._ended = True
                        break
                outcome = self._run_job(job)
                self._end_row(job, outcome, results)
                out_side = out_side + 1 if outcome.output_side else 0
                if outcome.output_unavailable or out_side >= 2:        # §2.4.6: the save folder is gone
                    reason = "output_unavailable"
                    break
        except Exception as e:                   # a bug: report it as text, still finish the run properly
            _log.error("The batch stopped unexpectedly: %s", friendly_error(e))
        finally:
            with self._state_lock:
                self._ended = True
            self._finish(started, results, reason)

    def _note_save_folder(self) -> None:
        """Start is the final check (§3.8): make the save folder if it is missing and note which folder it is, so
        that no later step writes into whatever appears at its path after an unmount (DS-8). If this fails, the
        first slide's check says why."""
        try:
            os.makedirs(self.opts.out_dir, exist_ok=True)
            self._out_id = preflight.path_id(self.opts.out_dir)
        except OSError:
            self._out_id = None

    def _end_row(self, job: Job, outcome: RowOutcome, results: List[Tuple[Job, RowOutcome]]) -> None:
        with self._state_lock:
            self.plan.finish(job.row_id, outcome)
            self._job = None
        results.append((job, outcome))
        _log.log(logging.WARNING if outcome.status in ("failed", "skipped", "finalize", "done_gaps") else
                 logging.INFO, "%s", outcome.message)
        self.post(RowFinished(self.run_id, job.row_id, outcome))

    def _finish(self, started: float, results: List[Tuple[Job, RowOutcome]], reason: Optional[str]) -> None:
        """At the end, always: the plan's end_run, the empty partial and scratch folders, the CSV, then RunFinished."""
        osutil.set_log_slide("")
        csv_path = None
        try:
            if self._job is not None:            # _run_job was interrupted (it catches every Exception)
                self._end_row(self._job, RowOutcome("failed", "Failed: internal error (see the log)."), results)
                self._job = None
            left = any(r.checked and r.status == "waiting" for r in self.plan.rows())
            if self._stopping and reason is None and (left or any(o.status == "cancelled" for _, o in results)):
                reason = "user"
            self.plan.end_run()
            for folder in ([join(self.opts.out_dir, PARTIAL_DIR)] + sorted(self._tdirs) if self._same_save_folder() else []):
                try:
                    if folder in self._tdir_ids and preflight.path_id(folder) != self._tdir_ids[folder]:
                        continue
                    os.rmdir(folder)
                except OSError:                  # absent, or still holds a partial kept for Finalize
                    pass
            with self.plan.lock:                 # Finalize can finish a slide while the rest of the batch runs
                results = [(job, self.plan.row(job.row_id).outcome or outcome) for job, outcome in results]
            if self.opts.summary_csv and results:
                csv_path = self._write_csv(results)
        except Exception as e:
            _log.error("Finishing the batch failed: %s", friendly_error(e))
        counts = Counter(o.status for _, o in results)
        self.post(RunFinished(self.run_id, RunSummary(started, _clock(), dict(counts),
                                                      tuple((j.row_id, o) for j, o in results),
                                                      self._stopping is not None, reason, csv_path)))

    # ---- one slide
    def _run_job(self, job: Job) -> RowOutcome:
        """One slide (§5.5.7). Never raises; keeps no exception object (R28)."""
        t0 = _clock()
        self.post(RowStarted(self.run_id, job.row_id, t0))
        osutil.set_log_slide(job.display)
        try:
            return self._attempt(job, t0)
        except Exception as e:                   # outside the engine calls: a vanished folder, a bug ...
            msg = friendly_error(e)
        return self._outcome(t0, "failed", msg)

    def _attempt(self, job: Job, t0: float) -> RowOutcome:
        components = (job.final_name, job.final_name + COMPLETE_SUFFIX, job.final_name + ".owner",
                      job.final_name + STALE_INFIX + "0" * STALE_HEX, _raw_name(job.final_name))
        if any(len(c.encode("utf-8")) > 255 or os.path.basename(c) != c for c in components):
            return self._outcome(t0, "failed", "The output name is too long or is not a file name. Choose a shorter name.")
        problem = self._save_folder_problem(job.out_dir)
        if problem:
            return self._outcome(t0, "failed", f"Can't save to {job.out_dir}: {problem}", output_unavailable=True)
        if exists(join(job.out_dir, job.final_name)) and not job.replace:       # whatever partial is there (DS-12)
            return self._outcome(t0, "skipped", "the output appeared since planning")
        pdir = join(job.out_dir, PARTIAL_DIR)
        if job.finalize_only:
            return self._finalize_ready(job, t0, pdir)
        self._phase(job, "reading", 0, 0)
        tile_files, note = None, ""
        if job.kind != "legacy_all":             # re-list: full, uncapped, the same classifier as discovery
            try:
                group = find_group(summarize_listing(job.folder, scan_dir(job.folder, cap=10 ** 9).entries),
                                   job.prefix, job.suffix)
            except FileNotFoundError:
                group = None
            if group is None:
                return self._outcome(t0, "skipped", "its tiles are no longer there")
            tile_files = group.names
            if len(group.names) != len(job.planned_tile_names):
                note = f" (tile count changed since planning: {len(job.planned_tile_names)} → {len(group.names)})"
        if job.est and free_bytes(job.out_dir) < job.est.out_worst:          # space check 1, before any tile
            return self._no_space(t0, job.est.out_worst, free_bytes(job.out_dir))
        problem = self._partial_dir_problem(pdir)
        if problem:
            return self._outcome(t0, "failed", f"Can't save to {job.out_dir}: {problem}", output_unavailable=True)
        tdir = temp_subdir(job.temp_dir, job.out_dir) if job.temp_dir else None
        try:
            with preflight.reserve_output(join(pdir, job.final_name)):
                if not self._same_save_folder():
                    return self._outcome(t0, "failed", FOLDER_CHANGED_TEXT, output_unavailable=True)
                busy = self._clear_way(pdir, job.final_name, tdir)
                if busy:
                    return self._outcome(t0, "skipped", busy)
                claim = self._claim(join(pdir, job.final_name))
        except preflight.ReservationBusy:
            return self._outcome(t0, "skipped", BUSY_TEXT)
        except OSError as e:                     # e.g. the disk is full: nothing can be saved there
            return self._outcome(t0, "failed", map_oserror(e).message, output_side=True)
        if claim is None:
            return self._outcome(t0, "skipped", BUSY_TEXT)
        mon = PartialMonitor(claim.path, job, self._bytes_post(job), run_id=self.run_id, poll_s=self.opts.poll_s,
                             temp_raw=join(tdir or pdir, _raw_name(job.final_name)))     # == the engine's temp_path
        mon.start()
        try:
            return self._stitch(job, t0, claim, tile_files, note, mon, tdir)
        finally:
            mon.stop()
            preflight.release_owner(claim.token)

    def _save_folder_problem(self, out_dir: str) -> Optional[str]:
        """Before each slide (DS-8, FS-17): the save folder must still be there, writable, and the folder the batch
        started in. It is never re-created here: after an unmount that would write onto the bare mount point."""
        try:
            preflight.write_probe(out_dir, create=False)
            if self._out_id is None:            # unknown at the start (a hiccup): this is the folder then
                self._out_id = preflight.path_id(out_dir)
            if preflight.path_id(out_dir) != self._out_id:
                return FOLDER_CHANGED_TEXT
        except OSError as e:
            return map_oserror(e).message
        return None

    def _same_save_folder(self) -> bool:
        try:
            return preflight.path_id(self.opts.out_dir) == self._out_id
        except OSError:
            return False

    def _cleanup_temp(self, eng, claim: Claim) -> None:
        if not self._same_save_folder():
            return
        try:
            if claim.parent_id is not None and preflight.path_id(os.path.dirname(claim.path)) != claim.parent_id:
                return
            if preflight.owner_token(claim.path) != claim.token:
                return
            folder = os.path.dirname(os.fspath(eng.temp_path))
            if folder in self._tdir_ids and preflight.path_id(folder) != self._tdir_ids[folder]:
                return
        except OSError:
            return
        eng.cleanup_temp()

    @staticmethod
    def _partial_dir_problem(pdir: str) -> Optional[str]:
        """Make PARTIAL_DIR with os.mkdir, so its parent must exist (DS-8); refuse a link, a junction or a file in
        its place (DS-10). Returns why it can't be used, or None."""
        try:
            os.mkdir(pdir)
            osutil.hide_on_windows(pdir)
        except FileExistsError:
            pass
        except OSError as e:
            return map_oserror(e).message
        try:
            return preflight.partial_dir_problem(pdir)
        except OSError as e:
            return map_oserror(e).message

    def _clear_way(self, pdir: str, final_name: str, tdir: Optional[str]) -> Optional[str]:
        """D3: a partial of this slide (any case variant, DS-13) that is complete or live keeps the slide skipped (the
        reason is returned); a dead or stale one, judged by the share's clock (DS-3), is taken over."""
        try:
            with preflight.reserve_output(join(pdir, final_name)):
                return self._clear_way_reserved(pdir, final_name, tdir)
        except preflight.ReservationBusy:
            return BUSY_TEXT

    def _clear_way_reserved(self, pdir: str, final_name: str, tdir: Optional[str]) -> Optional[str]:
        now = None
        for path in _partials_of(pdir, final_name):
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            if now is None:
                now = preflight.share_now(pdir)
            state = preflight.partial_state(path, st.st_mtime, now, self.opts.stale_partial_s)
            if state == "complete":
                return WAITING_TEXT
            if state == "live" or not _take_over(path, st, tdir):
                return BUSY_TEXT
        return None

    @staticmethod
    def _claim(partial: str) -> Optional[Claim]:
        """The O_EXCL claim, then our owner record (D3). Sidecars already there are garbage from a crash right after
        a move into place: removed first, so a claim can never look complete. None if another stitcher won."""
        try:
            with preflight.reserve_output(partial):
                ident = claim_partial(partial)
                if ident is None:
                    return None
                preflight.remove_sidecars(partial)
                try:
                    return Claim(partial, ident, preflight.write_owner(partial),
                                 preflight.path_id(os.path.dirname(partial)))
                except OSError:
                    _remove(partial)
                    raise
        except preflight.ReservationBusy:
            return None

    def _stitch(self, job: Job, t0: float, claim: Claim, tile_files, note: str, mon: PartialMonitor,
                tdir: Optional[str]) -> RowOutcome:
        eng = canvas = None
        try:
            eng = self.engine_cls(job.folder, claim.path, self._progress(job, mon),
                                  lambda name, current: list(job.channel_override) if job.channel_override else None,
                                  tile_files=tile_files, image_name=job.image_name, cancel_event=self._cancel,
                                  temp_dir=tdir)
            if not eng.scan_metadata():
                return self._failed_or_cancelled(eng, job, t0, claim)
            needed_gb = eng.calculate_ram_needed_gb()
            raw = int(needed_gb * 1024 ** 3)
            mon.expected = int(raw * LEVEL0_FACTOR)
            use_ram = needed_gb + RAM_HEADROOM_GB < psutil.virtual_memory().available / 1024 ** 3   # v1 rule
            if not use_ram:                      # space check 2: the canvas-sized disk-mode temp file
                tvol = job.temp_dir or job.out_dir
                out_worst = job.est.out_worst if job.est else int(raw * OUT_WORST_FACTOR)
                need = raw + (out_worst if same_volume(tvol, job.out_dir) else 0)
                if free_bytes(tvol) < need:
                    self._discard(claim)
                    return self._no_space(t0, need, free_bytes(tvol))
                if tdir:                         # this save folder's own scratch subfolder (D4)
                    try:
                        os.makedirs(tdir, exist_ok=True)
                    except OSError as e:
                        self._discard(claim)
                        return self._outcome(t0, "failed", f"Can't use the temporary folder {job.temp_dir}: "
                                                           f"{map_oserror(e).message}")
                    osutil.hide_on_windows(tdir)
                    self._tdirs.add(tdir)
                    self._tdir_ids[tdir] = preflight.path_id(tdir)
            canvas = eng.stitch(use_ram=use_ram)
            if canvas is None:
                return self._failed_or_cancelled(eng, job, t0, claim)
            lost = self._lost(job, t0, claim)    # the engine opens the partial by name: still ours, still there?
            ok = lost is None and eng.write_to_disk(canvas, use_ram)
            canvas = None
            gc.collect()
            self._cleanup_temp(eng, claim)       # Windows: the memmap's file can be deleted only now
            if lost is not None:
                return lost
            if not ok:
                return self._failed_or_cancelled(eng, job, t0, claim, writing=True)
        except Exception as e:                   # an engine call raised (e.g. os.listdir in legacy mode)
            msg = friendly_error(e)              # text only: the exception's frames would pin the canvas
        else:
            return self._finalize(eng, job, t0, claim, use_ram, note, mon)
        canvas = None
        gc.collect()
        if eng is not None:
            self._cleanup_temp(eng, claim)
        self._discard(claim)
        return self._outcome(t0, "failed", msg)

    def _failed_or_cancelled(self, eng, job: Job, t0: float, claim: Claim, writing: bool = False) -> RowOutcome:
        self._cleanup_temp(eng, claim)
        self._discard(claim)
        if eng.cancelled:
            return self._outcome(t0, "cancelled", CANCELLED_TEXT)
        err = eng.last_error
        reason = friendly_error(err) if err else (osutil.last_log_problem(job.display) or "see the log for details.")
        output_side = writing and (bool(err and err.get("oserror")) or not _probe_ok(job.out_dir))     # R28
        return self._outcome(t0, "failed", reason, output_side=output_side)

    @staticmethod
    def _discard(claim: Claim) -> None:
        """Give up our claim (a failure, a stop, a changed save folder): remove our owner record, so that no other
        instance on this computer takes the partial for live work while this app runs, then our partial. Nothing
        once our owner record is gone: another stitcher has taken the name over, and its new claim may even have our
        old file's id (D3). A partial whose id changed (a remounted share) is left to the age rule."""
        try:
            if claim.parent_id is not None and preflight.path_id(os.path.dirname(claim.path)) != claim.parent_id:
                return
            with preflight.reserve_output(claim.path):
                if (preflight.read_owner(claim.path) or {}).get("token") != claim.token:
                    return
                try:
                    ours = preflight.path_id(claim.path) == claim.ident
                except OSError:
                    ours = False
                preflight.remove_sidecars(claim.path)
                if ours:
                    _remove(claim.path)
        except OSError:
            pass

    def _finalize(self, eng, job: Job, t0: float, claim: Claim, use_ram: bool, note: str,
                  mon: PartialMonitor) -> RowOutcome:
        lost = self._lost(job, t0, claim)
        if lost is not None:
            return lost
        try:
            with preflight.reserve_output(claim.path):
                return self._finalize_reserved(eng, job, t0, claim, use_ram, note, mon)
        except preflight.ReservationBusy:
            return self._outcome(t0, "skipped", BUSY_TEXT)

    def _finalize_reserved(self, eng, job, t0, claim, use_ram, note, mon) -> RowOutcome:
        final = join(job.out_dir, job.final_name)
        gaps = eng.failed_tiles + self._dropped(job, eng)                 # tiles dropped by the scan, too (DS-1)
        facts = dict(tiles_used=len(eng.tiles), failed_tiles=gaps, channels=tuple(eng.channel_names),
                     um_per_px=eng.phys_size_x_um, memory_mode="ram" if use_ram else "disk")
        try:
            preflight.fsync_path(claim.path)     # minutes for a big file on a share: the heartbeat goes on
            lost = self._lost(job, t0, claim)
            if lost is not None:
                return lost
            preflight.mark_complete(claim.path, gaps)
        except OSError as e:
            self._discard(claim)                 # not trusted as finished: it would only look live to others
            return self._outcome(t0, "failed", map_oserror(e).message, output_side=True, **facts)
        mon.stop()
        if exists(final) and not job.replace:    # it appeared while we stitched: the user decides (it is 'ready')
            return self._outcome(t0, "skipped", APPEARED_TEXT, partial_path=claim.path, **facts)
        try:
            preflight.move_into_place(claim.path, final, self.opts.replace_retries, replace=job.replace)
        except FileExistsError:
            return self._outcome(t0, "skipped", APPEARED_TEXT, partial_path=claim.path, **facts)
        except PermissionError:                  # the old file is open elsewhere: the complete partial waits
            return self._outcome(t0, "finalize", FINALIZE_TEXT, output_path=final, partial_path=claim.path, **facts)
        except OSError as e:                     # it stays complete: the next scan offers it as ready
            return self._outcome(t0, "failed", map_oserror(e).message, output_side=True, **facts)
        st = os.stat(final)
        self.post(OutputWritten(final, st.st_size, st.st_mtime_ns, gaps))
        generic = list(eng.channel_names) == [f"Channel {i}" for i in range(eng.n_channels)]
        status = "done_gaps" if gaps else "done_generic" if generic and not job.channel_override else "done"
        return self._outcome(t0, status, _done_text(status, gaps, st.st_size, _clock() - t0) + note,
                             output_path=final, output_bytes=st.st_size, **facts)

    @staticmethod
    def _dropped(job: Job, eng) -> int:
        """Tiles the metadata scan dropped (DS-1). A legacy_all row lists every TIFF of its folder, so there only the
        tiles among the files left out count."""
        dropped = max(0, eng.n_listed - len(eng.tiles))
        if job.kind == "legacy_all" and dropped:
            placed = [os.path.basename(t["path"]) for t in eng.tiles]
            try:
                return min(dropped, preflight.legacy_gaps(job.folder, placed))
            except OSError:                      # the folder is gone just now: report what the engine left out
                pass
        return dropped

    def _lost(self, job: Job, t0: float, claim: Claim) -> Optional[RowOutcome]:
        """Before the engine opens the partial by name, and again before the partial is marked complete: is the save
        folder still there and the one the batch started in (DS-8), and is the partial still ours, the file we
        claimed, with our owner token (D3)? None if so; otherwise the outcome."""
        try:
            changed = preflight.path_id(job.out_dir) != self._out_id
        except OSError as e:                     # gone (an unmounted share?): never write onto its mount point
            return self._outcome(t0, "failed", f"Can't save to {job.out_dir}: {map_oserror(e).message}",
                                 output_unavailable=True)
        if changed:
            return self._outcome(t0, "failed", f"Can't save to {job.out_dir}: {FOLDER_CHANGED_TEXT}",
                                 output_unavailable=True)
        try:
            mine = preflight.path_id(claim.path) == claim.ident and preflight.owner_token(claim.path) == claim.token
        except FileNotFoundError:
            mine = False
        except OSError as e:                     # a share hiccup: whose the partial is can't be told, so it isn't used
            self._discard(claim)
            return self._outcome(t0, "failed", map_oserror(e).message, output_side=True)
        if mine:
            return None
        self._discard(claim)                     # our owner record, if it is still there, is not theirs
        return self._outcome(t0, "skipped", TAKEN_OVER_TEXT)

    def _finalize_ready(self, job: Job, t0: float, pdir: str) -> RowOutcome:
        """Job.finalize_only (D3): a complete partial left by an earlier run (the old file was in use) only needs
        moving into place, once it checks out."""
        try:
            with preflight.reserve_output(join(pdir, job.final_name)):
                if not self._same_save_folder():
                    return self._outcome(t0, "failed", FOLDER_CHANGED_TEXT, output_unavailable=True)
                return self._finalize_ready_reserved(job, t0, pdir)
        except preflight.ReservationBusy:
            return self._outcome(t0, "skipped", BUSY_TEXT)

    def _finalize_ready_reserved(self, job: Job, t0: float, pdir: str) -> RowOutcome:
        self._phase(job, "finishing", 0, 0)
        problem = preflight.partial_dir_problem(pdir)   # a link: what is "in" it is nobody's finished image (DS-10)
        if problem:
            return self._outcome(t0, "failed", f"Can't save to {job.out_dir}: {problem}", output_unavailable=True)
        final = join(job.out_dir, job.final_name)
        partial = next((p for p in _partials_of(pdir, job.final_name) if exists(p + COMPLETE_SUFFIX)), None)
        if partial is None:
            return self._outcome(t0, "skipped", "the finished image no longer waits to be moved into place")
        gaps = preflight.complete_gaps(partial)
        try:
            size, mtime_ns = preflight.finalize_kept(partial, final, self.opts.replace_retries, replace=job.replace)
        except FileExistsError:
            return self._outcome(t0, "skipped", APPEARED_TEXT, partial_path=partial, failed_tiles=gaps)
        except preflight.NotChecked as e:
            return self._outcome(t0, "failed", str(e), output_side=True)
        except ValueError as e:
            return self._outcome(t0, "failed", str(e))
        except PermissionError:
            return self._outcome(t0, "finalize", FINALIZE_TEXT, output_path=final, partial_path=partial,
                                 failed_tiles=gaps)
        except OSError as e:
            return self._outcome(t0, "failed", map_oserror(e).message, output_side=True)
        self.post(OutputWritten(final, size, mtime_ns, gaps))
        status = "done_gaps" if gaps else "done"
        return self._outcome(t0, status, _done_text(status, gaps, size, _clock() - t0), output_path=final,
                             output_bytes=size, failed_tiles=gaps)

    @staticmethod
    def _no_space(t0: float, need: int, free: int) -> RowOutcome:
        return BatchRunner._outcome(t0, "skipped", f"not enough space (needs {human_bytes(need, True)}, "
                                                   f"{human_bytes(free)} free)")

    @staticmethod
    def _outcome(t0: float, status: str, text: str, **kw) -> RowOutcome:
        prefix = {"failed": "Failed: ", "skipped": "Skipped: "}.get(status, "")
        return RowOutcome(status, prefix + text, started=t0, duration_s=max(0.0, _clock() - t0), **kw)

    # ---- progress (engine pool threads and the monitor: they only post and set the row's text)
    def _phase(self, job: Job, phase: str, current: int, total: int) -> None:
        self.plan.set_run_text(job.row_id, run_text(phase, current, total))
        self.post(RowProgress(self.run_id, job.row_id, phase, current, total, _clock()))

    def _progress(self, job: Job, mon: PartialMonitor) -> Callable[[int, int, str], None]:
        def callback(current: int, total: int, phase_text: str) -> None:
            phase = next((p for prefix, p in _PHASES if str(phase_text).startswith(prefix)), None)
            if phase is not None:
                mon.phase = phase                # the monitor posts RowBytes while "writing" or "pyramid"
                self._phase(job, phase, current, total)
        return callback

    def _bytes_post(self, job: Job) -> Post:
        def post(ev: RowBytes) -> None:
            if ev.phase == "writing":            # pyramid bytes are only activity: the level text stays (MW-9)
                self.plan.set_run_text(job.row_id, f"Writing full-resolution image · {human_bytes(ev.written)}")
            self.post(ev)
        return post

    # ---- the summary table (§2.4.7)
    def _write_csv(self, results: List[Tuple[Job, RowOutcome]]) -> Optional[str]:
        path = join(self.opts.out_dir, f"{APP_NAME}_summary_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        try:
            if preflight.path_id(self.opts.out_dir) != self._out_id:      # not into a stand-in folder (DS-8)
                _log.warning("The summary table was not saved: %s", FOLDER_CHANGED_TEXT)
                return None
            base = path[:-4]
            while True:
                try:
                    f = open(path, "x", encoding="utf-8-sig", newline="")
                    break
                except FileExistsError:
                    path = base + "-" + os.urandom(8).hex() + ".csv"
            with f:
                w = csv.writer(f)
                w.writerow(CSV_COLUMNS)
                for job, o in results:
                    w.writerow([job.display, job.folder, o.output_path or "", o.status, o.message,
                                o.tiles_used or len(job.planned_tile_names) or "", o.failed_tiles,
                                "; ".join(o.channels), "" if o.um_per_px is None else round(o.um_per_px, 6),
                                o.memory_mode or "", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(o.started)),
                                round(o.duration_s, 1), "" if o.output_bytes is None else o.output_bytes,
                                self.opts.app_version, tifffile.__version__, np.__version__])
            return path
        except OSError as e:
            _log.warning("The summary table could not be saved: %s", friendly_error(e))
            return None
