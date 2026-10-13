"""User settings (FINAL_SPEC §5.5.8, §5.6): one JSON file, read once at startup, written atomically by a
daemon writer thread (a uniquely named tmp file + os.replace, so two app instances never share one). The UI
debounces save_async(); save_now() is the exit path."""
from __future__ import annotations

import copy
import json
import logging
import ntpath
import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from .. import APP_NAME

RECENT_MAX = 8
LEDGER_MAX = 2000
REPLACE_TRIES, REPLACE_DELAY_S = 5, 0.05        # Windows: another instance may be reading settings.json just now
DAMAGED_TEXT = "Settings were reset (the file was damaged)."
DEFAULTS: Dict[str, Any] = {
    "schema": 1,
    "paths": {"input": "", "output": "", "browser_last": {"input": "", "output": ""}},
    "recent": {"inputs": [], "outputs": []},
    "browser": {"geometry": "", "sash": 210, "show_hidden": False, "show_all_files": False, "only_tiles": False,
                "sort": ["name", "asc"]},
    "discovery": {"max_depth": 3, "naming": "folder"},
    "run": {"keep_awake": True, "summary_csv": True, "temp_dir": None},
    "ui": {"scale": "auto", "geometry": "", "welcome_dismissed": False, "log_open": False, "log_height": 180},
    "verified_outputs": [],
    "last_run": {"state": "none", "started": "", "ended": "", "input": "", "output": "",
                 "planned": 0, "done": 0, "failed": 0},
}
log = logging.getLogger(__name__)


def _base(var: str, *fallback: str) -> str:
    """%LOCALAPPDATA% or an XDG base directory; unset or relative values (invalid for XDG) use the fallback."""
    value = os.environ.get(var, "")
    return value if os.path.isabs(value) else os.path.join(os.path.expanduser("~"), *fallback)


def _dirs() -> Tuple[str, str]:
    """(settings folder, logs folder) for this OS (§5.6); YFS_CONFIG_DIR overrides both."""
    if os.environ.get("YFS_CONFIG_DIR"):
        return os.environ["YFS_CONFIG_DIR"], os.path.join(os.environ["YFS_CONFIG_DIR"], "logs")
    if sys.platform == "win32":
        base = os.path.join(_base("LOCALAPPDATA", "AppData", "Local"), APP_NAME)
        return base, os.path.join(base, "logs")
    library = os.path.join(os.path.expanduser("~"), "Library")
    if sys.platform == "darwin":
        return os.path.join(library, "Application Support", APP_NAME), os.path.join(library, "Logs", APP_NAME)
    return (os.path.join(_base("XDG_CONFIG_HOME", ".config"), APP_NAME.lower()),
            os.path.join(_base("XDG_STATE_HOME", ".local", "state"), APP_NAME.lower(), "logs"))


def config_dir() -> str:
    """YFS_CONFIG_DIR | %LOCALAPPDATA%\\YourFavouriteStitcher | ~/Library/Application Support/
    YourFavouriteStitcher | ${XDG_CONFIG_HOME:-~/.config}/yourfavouritestitcher."""
    return _dirs()[0]


def log_dir() -> str:
    """YFS_CONFIG_DIR/logs | %LOCALAPPDATA%\\YourFavouriteStitcher\\logs | ~/Library/Logs/
    YourFavouriteStitcher | ${XDG_STATE_HOME:-~/.local/state}/yourfavouritestitcher/logs."""
    return _dirs()[1]


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy of `base` updated from `over`: unknown keys are kept and a section never becomes a non-dict."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if not isinstance(out.get(k), dict):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _merge(out[k], v)
    return out


def _same_folder_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _ledger_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _full_path(path: str) -> bool:
    """Absolute, and on Windows with a drive or UNC part: anything else means another folder for each working
    folder the app is started from (UD6)."""
    if sys.platform == "win32":
        return ntpath.isabs(path) and bool(ntpath.splitdrive(path)[0])
    return os.path.isabs(path)


def _forget_relative(paths: Dict[str, Any]) -> None:
    """Empty every non-absolute folder path of the 'paths' section (also nested ones, e.g. browser_last)."""
    for key, value in paths.items():
        if isinstance(value, dict):
            _forget_relative(value)
        elif isinstance(value, str) and value and not _full_path(value):
            paths[key] = ""


class Settings:
    """The settings of one user. Mutated on the Tk thread; safe to read from any thread."""

    def __init__(self, path: str, data: Optional[Dict[str, Any]] = None, load_error: Optional[str] = None) -> None:
        self.path, self.load_error, self.dirty = path, load_error, False
        self._data = _merge(DEFAULTS, data or {})
        recent = self._data["recent"]
        for kind in ("inputs", "outputs"):              # other modules rely on r["path"] being a str
            entries = recent[kind] if isinstance(recent[kind], list) else []
            recent[kind] = [r for r in entries if isinstance(r, dict) and isinstance(r.get("path"), str)
                            and r["path"]][:RECENT_MAX]
        ledger = self._data.pop("verified_outputs")
        self._ledger: "OrderedDict[str, Tuple[int, int, int]]" = OrderedDict()   # key -> (size, mtime_ns, gaps)
        for item in ledger[-LEDGER_MAX:] if isinstance(ledger, list) else []:
            try:
                path_key, size, mtime_ns, *gaps = item if isinstance(item, list) and len(item) in (3, 4) else ()
                self._ledger[str(path_key)] = (int(size), int(mtime_ns), int(gaps[0]) if gaps else 0)
            except (TypeError, ValueError):
                pass                                    # a damaged entry is simply not trusted
        self._cond = threading.Condition()
        self._pending: Optional[str] = None
        self._queued = self._written = 0
        self._last_ok = True
        self._writer: Optional[threading.Thread] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Settings":
        """Startup only (main thread). Corrupt JSON -> renamed to settings.json.bak, defaults used, .load_error set.
        Unknown keys are preserved; missing keys come from DEFAULTS (deep-merge). Folder paths that aren't
        absolute (paths.*, recent.*: an older version kept typed relative ones) are dropped (UD6)."""
        path = path or os.path.join(config_dir(), "settings.json")
        try:
            with open(path, encoding="utf-8-sig") as f:      # tolerate a BOM from a Windows editor
                data = json.load(f)
        except FileNotFoundError:
            return cls(path)
        except (ValueError, RecursionError) as e:       # bad JSON or bad UTF-8
            return cls._damaged(path, str(e))
        except OSError as e:
            return cls(path, load_error=f"Settings could not be read ({e.strerror or e}); defaults are used.")
        if not isinstance(data, dict):
            return cls._damaged(path, "not a JSON object")
        s = cls(path, data)
        _forget_relative(s._data["paths"])
        for kind in ("inputs", "outputs"):
            s._data["recent"][kind] = [r for r in s._data["recent"][kind] if _full_path(r["path"])]
        return s

    @classmethod
    def _damaged(cls, path: str, why: str) -> "Settings":
        log.warning("Settings file %s is damaged (%s); it is kept as settings.json.bak", path, why)
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
        return cls(path, load_error=DAMAGED_TEXT)

    def get(self, dotted: str, default: Any = None) -> Any:
        with self._cond:
            node: Any = self._data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, dotted: str, value: Any) -> None:
        """Marks dirty; schedule save_async() (debounced 1 s by the UI)."""
        *parents, last = dotted.split(".")
        with self._cond:
            node = self._data
            for part in parents:
                if not isinstance(node.get(part), dict):
                    node[part] = {}
                node = node[part]
            if last not in node or node[last] != value:
                node[last] = copy.deepcopy(value)
                self.dirty = True

    def add_recent(self, kind: str, path: str, slides: Optional[int] = None) -> None:
        """kind input|output; max 8; dedupe by normcase. slides=None keeps the count already known."""
        if not path:
            return
        key = _same_folder_key(path)
        with self._cond:
            entries = self._data["recent"][{"input": "inputs", "output": "outputs"}[kind]]
            old = next((r for r in entries if _same_folder_key(r["path"]) == key), {})
            entry: Dict[str, Any] = {"path": path, "used": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if slides is not None or "slides" in old:
                entry["slides"] = old.get("slides") if slides is None else int(slides)
            entries[:] = ([entry] + [r for r in entries if _same_folder_key(r["path"]) != key])[:RECENT_MAX]
            self.dirty = True

    def ledger_has(self, path: str, size: int, mtime_ns: int) -> bool:
        return self.ledger_get(path, size, mtime_ns) is not None

    def ledger_get(self, path: str, size: int, mtime_ns: int) -> Optional[int]:
        """The gaps recorded with this exact file (tiles that were missing or unreadable when it was made), or None
        when it is not in the ledger."""
        with self._cond:
            entry = self._ledger.get(_ledger_key(path))
        return entry[2] if entry is not None and entry[:2] == (int(size), int(mtime_ns)) else None

    def ledger_add(self, path: str, size: int, mtime_ns: int, gaps: int = 0) -> None:
        """Cap 2,000 (oldest dropped). gaps: tiles that were missing or unreadable when the file was made (DS-1)."""
        key = _ledger_key(path)
        with self._cond:
            self._ledger.pop(key, None)
            self._ledger[key] = (int(size), int(mtime_ns), int(gaps))
            if len(self._ledger) > LEDGER_MAX:
                self._ledger.popitem(last=False)
            self.dirty = True

    def save_async(self) -> None:
        """Snapshot -> writer thread -> tmp file + os.replace."""
        with self._cond:
            ledger = [[k, s, m] + ([g] if g else []) for k, (s, m, g) in self._ledger.items()]   # v2.0 reads 3 items
            data = dict(self._data, verified_outputs=ledger)
            self._pending, self.dirty = json.dumps(data, indent=1), False
            self._queued += 1
            if self._writer is None:
                self._writer = threading.Thread(target=self._write_loop, name="yfs-settings", daemon=True)
                self._writer.start()
            self._cond.notify_all()

    def save_now(self, timeout: float = 2.0) -> bool:
        """Exit path: queue the latest state if it changed, then wait up to `timeout` s. True if it is on disk."""
        if self.dirty:
            self.save_async()
        with self._cond:
            target = self._queued
            return self._cond.wait_for(lambda: self._written >= target, timeout) and self._last_ok

    def _write_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None)
                text, seq, self._pending = self._pending, self._queued, None
            ok = self._write(text)
            with self._cond:
                self._written, self._last_ok = seq, ok
                self._cond.notify_all()

    def _write(self, text: str) -> bool:
        folder, tmp = os.path.dirname(os.path.abspath(self.path)), None
        try:
            os.makedirs(folder, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=folder, prefix=".settings.", suffix=".tmp")   # never another instance's
            with open(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(1, REPLACE_TRIES + 1):
                try:
                    os.replace(tmp, self.path)
                    return True
                except PermissionError:
                    if attempt == REPLACE_TRIES:
                        raise
                    time.sleep(REPLACE_DELAY_S)
        except OSError as e:
            log.warning("Could not save the settings to %s: %s", self.path, e)
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return False
