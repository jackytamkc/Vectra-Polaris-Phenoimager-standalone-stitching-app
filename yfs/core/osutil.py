"""Operating-system helpers (FINAL_SPEC §5.5.9): child-process environment, Windows DPI and error mode, X
resources, opening folders and URLs, keep-awake, the zenity folder picker, log text, logging and error hooks.

Every Windows API call goes through _win32(): if it fails, the app carries on without that nicety."""
from __future__ import annotations

import ctypes
import functools
import logging
import logging.handlers
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from .events import BackgroundError, LogLine, PickerResult, Post

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")
LOG_KEEP = 30                                    # session log files kept
PROBLEM_TEXT_MAX = 120
_QUIET = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def clean_env() -> Dict[str, str]:
    """os.environ copy; if frozen (PyInstaller): LD_LIBRARY_PATH <- LD_LIBRARY_PATH_ORIG or removed;
    same for DYLD_LIBRARY_PATH / DYLD_FRAMEWORK_PATH on macOS."""
    env = dict(os.environ)
    if getattr(sys, "frozen", False):            # the bootloader points these at the bundle's own libraries
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"):
            if var + "_ORIG" in env:
                env[var] = env[var + "_ORIG"]
            else:
                env.pop(var, None)
    return env


def _win32(call: Callable[[Any], Any]) -> Any:
    """call(ctypes.windll) on Windows; None elsewhere or when the DLL, the function or the call fails."""
    if WINDOWS:
        try:
            return call(ctypes.windll)
        except Exception:
            pass
    return None


def set_windows_dpi_awareness() -> None:
    """Before tk.Tk(). shcore.SetProcessDpiAwareness(1) else user32.SetProcessDPIAware()."""
    if _win32(lambda w: w.shcore.SetProcessDpiAwareness(1)) is None:    # HRESULT, or None: no shcore (< 8.1)
        _win32(lambda w: w.user32.SetProcessDPIAware())


def set_windows_error_mode() -> None:
    """kernel32.SetErrorMode(SEM_FAILCRITICALERRORS 0x1 | SEM_NOOPENFILEERRORBOX 0x8000), keeping other flags."""
    _win32(lambda w: w.kernel32.SetErrorMode(w.kernel32.SetErrorMode(0x8001) | 0x8001))


def hide_on_windows(path: str) -> None:
    """SetFileAttributesW(FILE_ATTRIBUTE_HIDDEN); best effort."""
    _win32(lambda w: w.kernel32.SetFileAttributesW(str(path), 0x2))


@functools.lru_cache(maxsize=None)
def windows_long_paths() -> bool:
    """Windows: LongPathsEnabled is 1, so paths of 260+ characters work (the app is longPathAware). Read once from
    the registry; False elsewhere or when it can't be read."""
    if not WINDOWS:
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except (ImportError, OSError):
        return False


def linux_xft_dpi(timeout: float = 1.0) -> Optional[float]:
    """`xrdb -query` (clean env) -> Xft.dpi, or None."""
    try:
        out = subprocess.run(["xrdb", "-query"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             errors="replace", timeout=timeout, env=clean_env()).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"^Xft\.dpi:\s*([0-9]+(?:\.[0-9]*)?)\s*$", out, re.M)
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def open_path(path: str, on_error: Optional[Callable[[str], None]] = None) -> None:
    """Worker thread: os.startfile | open | xdg-open (clean env, start_new_session=True).
    on_error(text) is called on that worker thread, so it must be thread-safe (e.g. it posts an event)."""
    def work() -> None:
        try:
            if WINDOWS:
                os.startfile(path)
                return
            opener = "open" if MACOS else "xdg-open"
            code = subprocess.Popen([opener, path], env=clean_env(), start_new_session=True, **_QUIET).wait(10)
            if code and on_error:
                on_error(f"{opener} could not open it (exit code {code}).")
        except subprocess.TimeoutExpired:
            pass                                 # the opener is still busy showing it: not a failure
        except Exception as e:
            if on_error:
                on_error(str(e) or type(e).__name__)
    threading.Thread(target=work, name="yfs-open", daemon=True).start()


def open_url(url: str) -> None:
    """Same mechanism (never the webbrowser module in frozen Linux builds)."""
    open_path(url)


_awake: Dict[str, Any] = {"want": False, "proc": None}
_awake_lock = threading.Lock()


def keep_awake(on: bool) -> None:
    """MAIN THREAD. Windows: SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED) / (ES_CONTINUOUS).
    macOS: start/stop `caffeinate -i -w <pid>`. Linux: no-op (Later)."""
    _win32(lambda w: w.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0)))
    if MACOS:                                    # P2 (WP1 cut item 3)
        _awake["want"] = bool(on)
        threading.Thread(target=_sync_caffeinate, name="yfs-caffeinate", daemon=True).start()


def _sync_caffeinate() -> None:
    """Make the caffeinate child match the latest wish; these workers may run in any order."""
    with _awake_lock:
        proc, want = _awake["proc"], _awake["want"]
        if proc is not None and (not want or proc.poll() is not None):
            _awake["proc"] = None
            try:
                proc.terminate()
                proc.wait(5)
            except (OSError, subprocess.SubprocessError):
                pass
        if want and _awake["proc"] is None:
            try:
                _awake["proc"] = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                                  env=clean_env(), **_QUIET)
            except OSError:
                pass


def system_picker_available() -> bool:
    """Windows/macOS True; Linux shutil.which('zenity') is not None."""
    return WINDOWS or MACOS or shutil.which("zenity") is not None


def run_zenity(initial_dir: str, title: str, post: Post, token: int) -> Callable[[], None]:
    """Starts a worker: zenity --file-selection --directory --title=... --filename=<dir>/ (clean env,
    start_new_session=True). Posts PickerResult (path None on Cancel). Returns a kill() function."""
    state: Dict[str, Any] = {"proc": None, "killed": False}
    lock = threading.Lock()

    def work() -> None:
        path = error = None
        try:
            with lock:                           # kill() may come before the process exists
                if not state["killed"]:
                    state["proc"] = subprocess.Popen(
                        ["zenity", "--file-selection", "--directory", "--title=" + title,
                         "--filename=" + os.path.join(initial_dir, "")],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        env=clean_env(), start_new_session=True)
            proc = state["proc"]
            out = proc.communicate()[0] if proc is not None else b""
            if proc is not None and proc.returncode == 0:
                path = os.fsdecode(out.rstrip(b"\r\n")) or None
            elif proc is not None and proc.returncode not in (1, 5) and not state["killed"]:   # Cancel, timeout
                error = f"The system folder picker stopped unexpectedly (exit code {proc.returncode})."
        except OSError as e:
            error = f"The system folder picker could not be started ({e.strerror or e})."
        post(PickerResult(token, path, error))

    def kill() -> None:
        with lock:
            state["killed"] = True
            proc = state["proc"]
        try:
            if proc is not None:
                proc.kill()
        except OSError:
            pass
    threading.Thread(target=work, name="yfs-zenity", daemon=True).start()
    return kill


# Pictographs that start engine log lines (BMP symbol blocks and everything above U+FFFF) and their joiners.
_LEADING_EMOJI = re.compile("^\\s*(?:[\u2300-\u23ff\u2600-\u27bf\u2b00-\u2bff\ufe0e\ufe0f\u200d\u20e3"
                            "\U00010000-\U0010ffff]+\\s*)+")
_DROP = re.compile("[\ufe0e\ufe0f\U00010000-\U0010ffff]")
_NOT_LATIN1 = re.compile("[^\x00-\xff]")
_CORE_FONT_MAP = str.maketrans({"\u2026": "...", "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2018": "'",
                                "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2248": "~", "\u2713": "v",
                                "\u2717": "x"})


def display_text(s: str, xft: bool) -> str:
    """Strip leading emoji + U+FE0F + non-BMP; core fonts: §4.3 mapping, then '?' above U+00FF."""
    s = _DROP.sub("", _LEADING_EMOJI.sub("", s, count=1))
    return s if xft else _NOT_LATIN1.sub("?", s.translate(_CORE_FONT_MAP))


def _message(record: logging.LogRecord) -> str:
    """The record's own message without its traceback, also after QueueHandler.prepare()."""
    plain = getattr(record, "yfs_plain", None)
    if plain is not None:
        return plain
    try:
        return record.getMessage()
    except Exception:                            # a bad format string must not break the log view
        return str(record.msg)


def log_tag_for(record: logging.LogRecord) -> str:
    """§4.9 (ok for INFO starting with ✅/🎉)."""
    if record.levelno >= logging.ERROR:
        return "error"
    if record.levelno >= logging.WARNING:
        return "warn"
    if record.levelno >= logging.INFO:
        return "ok" if _message(record).lstrip().startswith(("\u2705", "\U0001f389")) else "info"
    return "debug"


_slide_lock = threading.Lock()
_slide = {"name": ""}
_problems: Dict[str, str] = {}                   # slide -> message of its last WARNING/ERROR record


def set_log_slide(name: str) -> None:
    """Attribution of engine log records to the running slide ('' outside a run); a new attempt starts clean."""
    with _slide_lock:
        _slide["name"] = name or ""
        _problems.pop(_slide["name"], None)


def last_log_problem(slide: str) -> str:
    """Display text of the slide's last WARNING/ERROR record, or ''."""
    with _slide_lock:
        lines = display_text(_problems.get(slide, ""), True).strip().splitlines()
    text = lines[0].strip() if lines else ""
    return text if len(text) <= PROBLEM_TEXT_MAX else text[:PROBLEM_TEXT_MAX - 3] + "..."


class _SlideQueueHandler(logging.handlers.QueueHandler):
    """Notes the slide and the plain message where the record is emitted: the listener runs later, and the
    runner reads last_log_problem() as soon as the engine returns."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record.yfs_plain = _message(record)
        with _slide_lock:
            record.yfs_slide = _slide["name"]
            if record.levelno >= logging.WARNING:
                _problems.pop(record.yfs_slide, None)
                _problems[record.yfs_slide] = record.yfs_plain
                if len(_problems) > 1000:
                    del _problems[next(iter(_problems))]
        return super().prepare(record)       # formats any traceback to text and drops exc_info (R28)


class _PostHandler(logging.Handler):
    """Listener side: one LogLine event per record."""

    def __init__(self, post: Post) -> None:
        super().__init__()
        self._post = post

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._post(LogLine(record.created, record.levelno, getattr(record, "yfs_slide", ""),
                               _message(record), log_tag_for(record)))
        except Exception:
            self.handleError(record)


def _prune_logs(folder: str, keep: int = LOG_KEEP) -> None:
    try:
        names = sorted(n for n in os.listdir(folder) if n.startswith("session-") and n.endswith(".log"))
        for name in names[:max(0, len(names) - keep)]:       # names sort by time; the current one is last
            os.remove(os.path.join(folder, name))
    except OSError:
        pass


def setup_logging(log_dir: str, post: Post, level: int = logging.INFO) -> logging.handlers.QueueListener:
    """Root logger -> QueueHandler; listener -> FileHandler(session-YYYYmmdd-HHMMSS.log, utf-8) + LogLine poster.
    No StreamHandler when sys.stderr is None. Prunes to the newest 30 session logs (worker).
    Call stop() on the returned listener at exit."""
    handlers: List[logging.Handler] = [_PostHandler(post)]
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, time.strftime("session-%Y%m%d-%H%M%S.log")),
                                           encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s "
                                                    "[%(yfs_slide)s] %(message)s"))
        handlers.insert(0, file_handler)
    except OSError:
        pass
    if sys.stderr is not None:                   # windowed builds have no stderr
        handlers.append(logging.StreamHandler(sys.stderr))
    q: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
    root = logging.getLogger()
    for old in [h for h in root.handlers if isinstance(h, _SlideQueueHandler)]:
        root.removeHandler(old)
    root.addHandler(_SlideQueueHandler(q))
    root.setLevel(level)
    listener = logging.handlers.QueueListener(q, *handlers, respect_handler_level=True)
    listener.start()
    threading.Thread(target=_prune_logs, args=(log_dir,), name="yfs-log-prune", daemon=True).start()
    if not isinstance(handlers[0], logging.FileHandler):
        logging.getLogger(__name__).warning("No log file: the folder %s could not be used.", log_dir)
    return listener


def install_excepthooks(root: Any, post: Post) -> None:
    """root.report_callback_exception, sys.excepthook, threading.excepthook -> logging.exception + BackgroundError.
    Only text leaves a hook: the exception and its frames are not kept (R28)."""
    def report(where: str, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            logging.getLogger("yfs").error("Unexpected error (%s)", where, exc_info=(exc_type, exc, tb))
            post(BackgroundError(where, "".join(traceback.format_exception(exc_type, exc, tb))))
        except Exception:
            pass                                 # an error hook must never raise

    def on_main(exc_type: Any, exc: Any, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
        else:
            report("main", exc_type, exc, tb)

    def on_thread(args: Any) -> None:
        if not issubclass(args.exc_type, SystemExit):
            report("thread " + getattr(args.thread, "name", "?"), args.exc_type, args.exc_value, args.exc_traceback)

    if root is not None:
        root.report_callback_exception = lambda exc_type, exc, tb: report("ui", exc_type, exc, tb)
    sys.excepthook = on_main
    threading.excepthook = on_thread
