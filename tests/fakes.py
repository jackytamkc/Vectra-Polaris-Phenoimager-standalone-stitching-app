"""Test doubles for the run pipeline: FakeEngine (stands in for stitcher_RAM.StitchingEngine) and FakeClock.

FakeEngine has the engine's constructor, attributes and public methods but reads no pixels: the tiles may be
empty files that only carry tile names. It spends configurable time in each phase, reports progress with the
engine's phase texts, honours cancel_event at the engine's checkpoints and writes a small fake output, so the
runner's partial-file protocol, progress, space checks, failure handling and stop modes run in milliseconds.
Make a variant with FakeEngine.configure(...); each configured class records its own calls and canvas weakrefs::

    Engine = FakeEngine.configure(durations={"stitch": 2.0}, fail="write", last_error=ENOSPC_ERROR,
                                  fail_if=lambda eng: eng.image_name == "SlideA2_Scan1")
    BatchRunner(1, plan, Engine, post, RunOptions(out_dir)).start()

Phase times are real seconds, or FakeClock seconds when `clock` is given (the clock then advances instead of
sleeping, and `on_step(engine, phase, step)` lets a test act between steps). The engine's static header helpers
(§5.7 d) are the real ones, so preflight.read_tile_header(path, FakeEngine) reads real tiles exactly as the engine
would (the UI's plan, sizes and channel checks then behave as with the real engine).
"""
from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from stitcher_RAM import StitchingEngine as _RealEngine

REAL_NAMES = ["DAPI", "Opal 520", "Opal 570", "Opal 690", "Autofluorescence"]
ENOSPC_ERROR = {"type": "OSError", "message": "[Errno 28] No space left on device", "errno": 28, "winerror": None,
                "oserror": True}
CONTIG_ERROR = {"type": "ValueError", "message": "<PLANARCONFIG.CONTIG: 1> does not match storedshape", "errno": None,
                "winerror": None, "oserror": False}


class FakeClock:
    """A wall clock that only moves when told to. Patch yfs.core.runner._clock with its .time method."""

    def __init__(self, start: Optional[float] = None) -> None:
        self._t = time.time() if start is None else float(start)
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += seconds


class _Stop(Exception):
    """Internal: a cancellation checkpoint or a configured failure inside write_to_disk."""


class FakeEngine:
    DEFAULTS: Dict[str, Any] = dict(
        durations={},          # phase ("scan" | "stitch" | "write") -> seconds spent in it
        steps={},              # phase -> number of steps (progress reports / checkpoints); default 4
        clock=None,            # FakeClock: advance it instead of sleeping
        on_step=None,          # callable(engine, phase, step) after every step
        fail=None,             # phase that fails the engine's way (False / None) ...
        last_error=None,       # ... leaving this record (dict) in engine.last_error
        log_warning=None,      # logged as a WARNING just before that failure
        raise_in=None,         # phase whose method raises RuntimeError instead (an unexpected engine error)
        fail_if=None,          # callable(engine) -> bool: limit fail / raise_in to some slides
        names=REAL_NAMES,      # channel names in the "tiles"; suspicious ones go to ask_user_callback like v1
        failed_tiles=0,        # tiles that "could not be read"
        dropped=0,             # tiles the metadata scan drops (unreadable header): the last ones of the listing
        needed_gb=0.001,       # calculate_ram_needed_gb(); a huge value forces disk mode
        output_bytes=64 * 1024,  # size of level 0 in the fake output
        pyramid=3,             # pyramid levels reported
    )
    cfg: Dict[str, Any] = dict(DEFAULTS)
    normalize_shape = staticmethod(_RealEngine.normalize_shape)          # the header helpers of §5.7 (d)
    extract_perkin_elmer_channels = staticmethod(_RealEngine.extract_perkin_elmer_channels)
    assess_channels = staticmethod(_RealEngine.assess_channels)
    resolution_from_tags = staticmethod(_RealEngine.resolution_from_tags)
    calls: List[tuple] = []                      # (method, image_name), on the configured class
    canvas_refs: List["weakref.ref"] = []        # one weakref per canvas returned by stitch()

    @classmethod
    def configure(cls, **options: Any) -> type:
        unknown = set(options) - set(cls.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown FakeEngine options: {sorted(unknown)}")
        return type("FakeEngine", (cls,), {"cfg": dict(cls.cfg, **options), "calls": [], "canvas_refs": []})

    def __init__(self, input_dir, output_path, progress_callback=None, ask_user_callback=None,
                 *, tile_files=None, image_name=None, cancel_event=None, temp_dir=None) -> None:
        self.input_dir, self.output_path = Path(input_dir), Path(output_path)
        self.progress_callback, self.ask_user_callback = progress_callback, ask_user_callback
        self.temp_path = self.output_path.with_suffix(".tmp.raw")
        if temp_dir is not None:
            self.temp_path = Path(temp_dir) / self.temp_path.name
        self.tile_files = None if tile_files is None else sorted({os.path.basename(os.fspath(f)) for f in tile_files})
        self.image_name = image_name or self.input_dir.name
        self.cancel_event, self.cancelled, self.failed_tiles, self.last_error = cancel_event, False, 0, None
        self.n_listed = 0
        self.tiles: List[dict] = []
        self.canvas_shape, self.dtype, self.channel_names = None, None, []
        self.n_channels = self.tile_h = self.tile_w = 0
        self.phys_size_x_um = self.phys_size_y_um = 0.5

    # ---- helpers
    def _cancel_requested(self) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.cancelled = True
        return self.cancelled

    def _report(self, current: int, total: int, text: str) -> None:
        if self.progress_callback:
            self.progress_callback(current, total, text)

    def _spend(self, phase: str, each=None, checkpoints: bool = True) -> bool:
        """Spend the phase's time in steps; False when cancelled at a checkpoint."""
        n = max(1, self.cfg["steps"].get(phase, 4))
        dt = self.cfg["durations"].get(phase, 0.0) / n
        for i in range(1, n + 1):
            if checkpoints and self._cancel_requested():
                return False
            if dt:
                self.cfg["clock"].advance(dt) if self.cfg["clock"] else time.sleep(dt)
            if each is not None:
                each(i, n)
            if self.cfg["on_step"] is not None:
                self.cfg["on_step"](self, phase, i)
        return not (checkpoints and self._cancel_requested())

    def _failing(self, phase: str) -> bool:
        """Raise (raise_in) or return True (fail) when the configuration fails this phase for this slide."""
        c = self.cfg
        if phase not in (c["fail"], c["raise_in"]) or (c["fail_if"] is not None and not c["fail_if"](self)):
            return False
        if c["raise_in"] == phase:
            raise RuntimeError(f"fake {phase} exception")
        if c["log_warning"]:
            logging.warning(c["log_warning"])
        self.last_error = dict(c["last_error"]) if c["last_error"] else None
        return True

    # ---- the engine's public methods
    def scan_metadata(self) -> bool:
        type(self).calls.append(("scan_metadata", self.image_name))
        if self._cancel_requested():
            return False
        files = self.tile_files if self.tile_files is not None else sorted(
            f for f in os.listdir(self.input_dir) if f.lower().endswith((".tif", ".tiff")))
        self.n_listed = len(files)
        if not files or not self._spend("scan") or self._failing("scan"):
            return False
        names = list(self.cfg["names"] or [])
        self.n_channels, self.tile_h, self.tile_w, self.dtype = 5, 256, 320, np.dtype("float32")
        if names and len(names) == self.n_channels and len(set(names)) == len(names):
            self.channel_names = names
        else:                                    # v1's suspicious-name path
            asked = self.ask_user_callback(self.image_name, names or ["?"] * self.n_channels) \
                if self.ask_user_callback else None
            self.channel_names = asked if asked and len(asked) == self.n_channels else \
                [f"Channel {i}" for i in range(self.n_channels)]
        self.tiles = [{"path": str(self.input_dir / f)} for f in files[:len(files) - self.cfg["dropped"]]]
        self.canvas_shape = (self.n_channels, 64, 80)
        return True

    def calculate_ram_needed_gb(self) -> float:
        return self.cfg["needed_gb"]

    def stitch(self, use_ram=True):
        type(self).calls.append(("stitch", self.image_name))
        if not use_ram:
            self.temp_path.write_bytes(b"\0" * 4096)   # the memmap's backing file
        n = len(self.tiles)
        if not self._spend("stitch", lambda i, k: self._report(max(1, i * n // k), n, "Stitching")) \
                or self._failing("stitch"):
            self.cleanup_temp()
            return None
        self.failed_tiles = self.cfg["failed_tiles"]
        canvas = np.zeros(self.canvas_shape, np.float32)
        type(self).canvas_refs.append(weakref.ref(canvas))
        return canvas

    def write_to_disk(self, canvas, is_ram_mode) -> bool:
        type(self).calls.append(("write_to_disk", self.image_name))
        ok, levels = False, self.cfg["pyramid"]
        self._report(0, 0, "Formatting...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            if self._cancel_requested():
                raise _Stop()
            with open(self.output_path, "wb") as f:
                self._report(0, 0, "Writing Full Res...")
                n = max(1, self.cfg["steps"].get("write", 4))
                chunk = b"L" * (self.cfg["output_bytes"] // n)
                self._spend("write", lambda i, k: (f.write(chunk), f.flush()), checkpoints=False)
                if self._failing("write"):     # after level 0, like an ENOSPC at the first pyramid level
                    raise _Stop()
                for level in range(1, levels + 1):
                    if self._cancel_requested():
                        raise _Stop()
                    self._report(level, levels, f"Writing Pyramid {level}/{levels}")
                    f.write(b"P" * 512)
            self._report(100, 100, "Finished")
            ok = True
        except _Stop:
            if self.output_path.exists():
                os.remove(self.output_path)
        finally:
            if not is_ram_mode:
                self.cleanup_temp()
        return ok

    def cleanup_temp(self) -> bool:
        try:
            if self.temp_path.exists():
                os.remove(self.temp_path)
        except OSError:
            pass
        return not self.temp_path.exists()
