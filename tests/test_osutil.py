"""yfs.core.osutil (FINAL_SPEC §5.5.9; acceptance A5, A6 for this module).

Windows and macOS branches run here by monkeypatching the platform flags, ctypes and os.startfile; POSIX
child processes (zenity, xdg-open, open, caffeinate, xrdb) are stub shell scripts put first on PATH."""
from __future__ import annotations

import gc
import logging
import os
import re
import subprocess
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

from yfs.core import osutil
from yfs.core.events import BackgroundError, LogLine, PickerResult

REPO = Path(__file__).resolve().parents[1]
ENGINE_EMOJI = ["\U0001f50d", "\U0001f4cf", "\u26a0\ufe0f", "\u2705", "\U0001f389", "\u274c", "\u23e9",
                "\u2b07\ufe0f", "\U0001f4be", "\U0001f4ca", "\u23f9\ufe0f"]          # 🔍 📏 ⚠️ ✅ 🎉 ❌ ⏩ ⬇️ 💾 📊 ⏹️
posix_only = pytest.mark.skipif(os.name != "posix", reason="stub executables are shell scripts")


def lines_with(collector, *needles, count=1):
    """LogLine events whose text contains one of `needles` (other threads may log at the same time)."""
    return collector.wait_for(lambda evs: (lambda found: found if len(found) >= count else None)(
        [e for e in evs if isinstance(e, LogLine) and any(n in e.text for n in needles)]))


def wait_until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture
def stub_bin(tmp_path, monkeypatch):
    """make(name, body) writes an executable /bin/sh script into a folder that is first on PATH."""
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")

    def make(name, body):
        path = folder / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return path
    make.folder = folder
    return make


@pytest.fixture
def fresh_logging():
    """Undoes setup_logging(): stops the listeners the test registers and removes the queue handler."""
    root = logging.getLogger()
    level = root.level
    listeners = []
    yield listeners
    for listener in listeners:
        try:
            listener.stop()
        except AttributeError:                   # already stopped (stop() is not idempotent on 3.11)
            pass
        for h in listener.handlers:
            h.close()
    for h in [h for h in root.handlers if isinstance(h, osutil._SlideQueueHandler)]:
        root.removeHandler(h)
    root.setLevel(level)
    osutil.set_log_slide("")


class FakeWinDLL:
    """Records calls such as windll.kernel32.SetThreadExecutionState(...); `results` sets return values or
    exceptions per function name."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __getattr__(self, dll):
        outer = self

        class Dll:
            def __getattr__(self, fn):
                def call(*args):
                    outer.calls.append((dll, fn) + args)
                    r = outer.results.get(fn, 0)
                    if isinstance(r, BaseException):
                        raise r
                    return r(*args) if callable(r) else r
                return call
        return Dll()


# ---------------------------------------------------------------------------------------------- clean_env (A5)
def test_clean_env_restores_library_paths_when_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle/_internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/site/lib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/bundle/Frameworks")
    monkeypatch.delenv("DYLD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setenv("DYLD_FRAMEWORK_PATH", "/bundle/Frameworks")
    monkeypatch.setenv("DYLD_FRAMEWORK_PATH_ORIG", "")
    env = osutil.clean_env()
    assert env["LD_LIBRARY_PATH"] == "/opt/site/lib"
    assert "DYLD_LIBRARY_PATH" not in env and env["DYLD_FRAMEWORK_PATH"] == ""
    assert env["PATH"] == os.environ["PATH"] and os.environ["LD_LIBRARY_PATH"] == "/bundle/_internal"


def test_clean_env_removes_bundle_path_without_orig_and_is_a_copy_when_not_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle/_internal")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    assert "LD_LIBRARY_PATH" not in osutil.clean_env()
    monkeypatch.setattr(sys, "frozen", False)
    env = osutil.clean_env()
    assert env == dict(os.environ) and env is not os.environ


# --------------------------------------------------------------------------------------------- display_text (A5)
@pytest.mark.parametrize("xft", [True, False])
@pytest.mark.parametrize("emoji", ENGINE_EMOJI)
def test_display_text_strips_every_engine_emoji(emoji, xft):
    assert osutil.display_text(f"{emoji} [Slide_1] Quick-Scanning 9 tiles...", xft) == \
        "[Slide_1] Quick-Scanning 9 tiles..."
    assert osutil.display_text(f"{emoji}  [S] Using DISK Mode (Slower)...", xft) == "[S] Using DISK Mode (Slower)..."


@pytest.mark.parametrize("xft", [True, False])
def test_every_engine_log_message_becomes_safe_text(xft):
    """Future-proofing: every literal logging message in stitcher_RAM.py is displayable after display_text."""
    source = (REPO / "stitcher_RAM.py").read_text(encoding="utf-8")
    messages = re.findall(r'logging\.(?:info|warning|error|debug)\(f?"([^"]*)"', source)
    assert len(messages) > 25
    for msg in messages:
        shown = osutil.display_text(msg, xft)
        assert not shown[:1] or shown[0].isspace() or shown[0].isascii(), (msg, shown)
        assert all(ord(ch) <= 0xFFFF and ch != "\ufe0f" for ch in shown), shown
        if not xft:
            assert all(ord(ch) <= 0xFF for ch in shown), shown


def test_display_text_details():
    assert osutil.display_text("   [S1] Writing Level 0...", True) == "   [S1] Writing Level 0..."   # indentation kept
    assert osutil.display_text("\U0001f4cf Resolution Found: 0.4984 \u00b5m/pixel", False) == \
        "Resolution Found: 0.4984 \u00b5m/pixel"                                   # µ is Latin-1
    assert osutil.display_text("\u26a0\ufe0f [S] 2/9 tiles FAILED \u2014 output will have gaps!", False) == \
        "[S] 2/9 tiles FAILED - output will have gaps!"
    assert osutil.display_text("\u26a0\ufe0f [S] 2/9 tiles FAILED \u2014 gaps", True) == "[S] 2/9 tiles FAILED \u2014 gaps"
    assert osutil.display_text("a \U0001f600 b\ufe0f c", True) == "a  b c"           # non-BMP and VS16 dropped
    assert osutil.display_text("\u2026 \u2192 \u2018x\u2019 \u201cy\u201d \u2248 \u2713 \u2717 \u2013 \u65e5", False) == \
        "... -> 'x' \"y\" ~ v x - ?"
    assert osutil.display_text("\u00a3\u00b7\u00d7\u00b0", False) == "\u00a3\u00b7\u00d7\u00b0"
    assert osutil.display_text("", False) == "" and osutil.display_text("\u2705", True) == ""


# ------------------------------------------------------------------------------------------------- log tags (§4.9)
def test_log_tag_for():
    def rec(level, msg, *args):
        return logging.LogRecord("root", level, __file__, 1, msg, args, None)
    assert osutil.log_tag_for(rec(logging.INFO, "\u2705 [S] Stitching (RAM) Complete.")) == "ok"
    assert osutil.log_tag_for(rec(logging.INFO, "\U0001f389 [S] Success! (Size: %.1f MB)", 3.0)) == "ok"
    assert osutil.log_tag_for(rec(logging.INFO, "   [S] Writing Level 0...")) == "info"
    assert osutil.log_tag_for(rec(logging.WARNING, "\u2705 odd")) == "warn"
    assert osutil.log_tag_for(rec(logging.ERROR, "x")) == "error"
    assert osutil.log_tag_for(rec(logging.CRITICAL, "x")) == "error"
    assert osutil.log_tag_for(rec(logging.DEBUG, "x")) == "debug"
    assert osutil.log_tag_for(rec(logging.INFO, "bad %d format", "text")) == "info"   # never raises


# -------------------------------------------------------------------------------------------------- logging
def test_setup_logging_file_lines_slides_and_problems(tmp_path, collector, fresh_logging):
    log_dir = tmp_path / "logs"
    fresh_logging.append(osutil.setup_logging(str(log_dir), collector))
    osutil.set_log_slide("Slide_A")
    logging.info("\U0001f50d [Slide_A] Quick-Scanning 9 tiles...")
    logging.warning("\u26a0\ufe0f Channel Name Issue: %s", ["DAPI", "DAPI"])
    logging.error("\u274c Error reading first tile header: %s", "truncated file")
    assert osutil.last_log_problem("Slide_A") == "Error reading first tile header: truncated file"  # no waiting
    assert osutil.last_log_problem("Other") == ""
    osutil.set_log_slide("")
    logging.info("\u2705 [x] done")
    lines = lines_with(collector, "Quick-Scanning", "Channel Name Issue", "Error reading", "[x] done", count=4)
    assert [(e.slide, e.tag) for e in lines] == [("Slide_A", "info"), ("Slide_A", "warn"),
                                                 ("Slide_A", "error"), ("", "ok")]
    assert lines[1].text == "\u26a0\ufe0f Channel Name Issue: ['DAPI', 'DAPI']" and lines[1].level == logging.WARNING
    fresh_logging[0].stop()
    files = list(log_dir.glob("session-*.log"))
    assert len(files) == 1 and re.fullmatch(r"session-\d{8}-\d{6}\.log", files[0].name)
    content = files[0].read_text(encoding="utf-8")
    assert "\U0001f50d [Slide_A] Quick-Scanning" in content and "[Slide_A] \u274c Error reading" in content


def test_a_new_attempt_clears_the_slide_problem_and_text_is_cut(tmp_path, collector, fresh_logging):
    fresh_logging.append(osutil.setup_logging(str(tmp_path / "logs"), collector))
    osutil.set_log_slide("S2")
    logging.warning("\u26a0\ufe0f " + "x" * 300)
    text = osutil.last_log_problem("S2")
    assert len(text) == osutil.PROBLEM_TEXT_MAX and text.endswith("...")
    osutil.set_log_slide("S2")
    assert osutil.last_log_problem("S2") == ""


def test_exception_records_keep_their_traceback_in_the_file_only(tmp_path, collector, fresh_logging):
    log_dir = tmp_path / "logs"
    fresh_logging.append(osutil.setup_logging(str(log_dir), collector))
    try:
        raise ValueError("inner detail")
    except ValueError:
        logging.exception("Something failed")
    line = lines_with(collector, "Something failed")[0]
    assert line.text == "Something failed" and line.tag == "error"
    fresh_logging[0].stop()
    content = next(log_dir.glob("session-*.log")).read_text(encoding="utf-8")
    assert "Traceback" in content and "ValueError: inner detail" in content


def test_no_stream_handler_without_stderr(tmp_path, collector, fresh_logging, monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    listener = osutil.setup_logging(str(tmp_path / "logs"), collector)
    fresh_logging.append(listener)
    assert not any(type(h) is logging.StreamHandler for h in listener.handlers)
    logging.warning("still logged")
    assert lines_with(collector, "still logged")[0].text == "still logged"
    monkeypatch.undo()
    listener2 = osutil.setup_logging(str(tmp_path / "logs2"), collector)
    fresh_logging.append(listener2)
    assert any(type(h) is logging.StreamHandler for h in listener2.handlers)
    assert sum(isinstance(h, osutil._SlideQueueHandler) for h in logging.getLogger().handlers) == 1


def test_old_session_logs_are_pruned_to_30(tmp_path, collector, fresh_logging):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for i in range(40):
        (log_dir / f"session-2020{i // 28 + 1:02d}{i % 28 + 1:02d}-120000.log").write_text("old")
    (log_dir / "notes.txt").write_text("keep me")
    fresh_logging.append(osutil.setup_logging(str(log_dir), collector))
    assert wait_until(lambda: len(list(log_dir.glob("session-*.log"))) == osutil.LOG_KEEP)
    names = sorted(p.name for p in log_dir.glob("session-*.log"))
    assert names[-1].startswith("session-20") and not names[-1].startswith("session-2020")    # the current one
    assert names[0] == "session-20200112-120000.log" and (log_dir / "notes.txt").exists()


def test_unusable_log_folder_still_posts_lines(tmp_path, collector, fresh_logging):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    listener = osutil.setup_logging(str(blocker / "logs"), collector)
    fresh_logging.append(listener)
    assert not any(isinstance(h, logging.FileHandler) for h in listener.handlers)
    assert lines_with(collector, "No log file")


# ------------------------------------------------------------------------------------------------- excepthooks
def test_excepthooks_log_and_post(tmp_path, collector, fresh_logging, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    fresh_logging.append(osutil.setup_logging(str(tmp_path / "logs"), collector))
    root = types.SimpleNamespace()
    osutil.install_excepthooks(root, collector)
    try:
        raise KeyError("in a Tk callback")
    except KeyError:
        root.report_callback_exception(*sys.exc_info())
    try:
        raise RuntimeError("uncaught on main")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    t = threading.Thread(target=lambda: 1 / 0, name="worker-x")
    t.start()
    t.join()
    quiet = threading.Thread(target=sys.exit, name="worker-exit")
    quiet.start()
    quiet.join()
    errors = collector.wait_type(BackgroundError, count=3)
    assert [e.where for e in errors] == ["ui", "main", "thread worker-x"]
    assert "KeyError: 'in a Tk callback'" in errors[0].text and "ZeroDivisionError" in errors[2].text
    collector.wait_for(lambda evs: len([e for e in evs if isinstance(e, LogLine) and e.tag == "error"]) >= 3)
    time.sleep(0.2)
    assert len(collector.of_type(BackgroundError)) == 3                 # SystemExit in a thread is not an error


def test_excepthook_keeps_no_reference_to_the_failing_frames(tmp_path, collector, fresh_logging, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    fresh_logging.append(osutil.setup_logging(str(tmp_path / "logs"), collector))
    root_logger = logging.getLogger()
    others = [h for h in root_logger.handlers if not isinstance(h, osutil._SlideQueueHandler)]
    for h in others:                             # pytest's capture handler would keep the record (and frames)
        root_logger.removeHandler(h)
    try:
        osutil.install_excepthooks(None, collector)

        class Canvas:
            pass

        def fail(canvas):
            raise MemoryError("big canvas")
        canvas = Canvas()
        ref = weakref.ref(canvas)
        try:
            fail(canvas)
        except MemoryError:
            sys.excepthook(*sys.exc_info())
        del canvas
        gc.collect()
        collector.wait_type(BackgroundError)
        assert ref() is None
    finally:
        for h in others:
            root_logger.addHandler(h)


def test_excepthooks_survive_without_stderr(collector, monkeypatch, fresh_logging, tmp_path):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(sys, "stderr", None)
    fresh_logging.append(osutil.setup_logging(str(tmp_path / "logs"), collector))
    osutil.install_excepthooks(None, collector)
    try:
        raise OSError(5, "I/O error")
    except OSError:
        sys.excepthook(*sys.exc_info())
    assert collector.wait_type(BackgroundError)[0].where == "main"


# ---------------------------------------------------------------------------------------------- system picker
@posix_only
def test_zenity_picker_result_args_env_and_session(stub_bin, tmp_path, collector, monkeypatch):
    rec = tmp_path / "rec"
    stub_bin("zenity", f'printf "%s\\n" "$@" > "{rec}.args"\n'
                       f'echo "$LD_LIBRARY_PATH" > "{rec}.env"\n'
                       f'cut -d" " -f6 /proc/$$/stat > "{rec}.sid"\n'
                       'printf "/data/chosen folder/\\303\\274\\n"')
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/local/lib")
    assert osutil.system_picker_available()
    osutil.run_zenity("/data/start", "Choose the folder with your tiles", collector, 7)
    result = collector.wait_type(PickerResult)[0]
    assert result == PickerResult(7, "/data/chosen folder/\u00fc", None)
    assert Path(f"{rec}.args").read_text().splitlines() == [
        "--file-selection", "--directory", "--title=Choose the folder with your tiles", "--filename=/data/start/"]
    assert Path(f"{rec}.env").read_text().strip() == "/usr/local/lib"
    assert int(Path(f"{rec}.sid").read_text()) != os.getsid(0)                 # start_new_session


@posix_only
@pytest.mark.parametrize("code,error", [(1, None), (5, None), (3, "exit code 3")])
def test_zenity_cancel_and_failure(stub_bin, collector, code, error):
    stub_bin("zenity", f"exit {code}")
    osutil.run_zenity("/data", "t", collector, 1)
    result = collector.wait_type(PickerResult)[0]
    assert result.path is None and (result.error is None if error is None else error in result.error)


@posix_only
def test_zenity_kill_ends_the_picker(stub_bin, collector):
    stub_bin("zenity", "exec sleep 30")
    kill = osutil.run_zenity("/data", "t", collector, 2)
    time.sleep(0.3)
    t0 = time.monotonic()
    kill()
    assert collector.wait_type(PickerResult, timeout=5)[0] == PickerResult(2, None, None)
    assert time.monotonic() - t0 < 3
    kill()                                                                     # harmless twice


def test_zenity_missing(stub_bin, collector, monkeypatch):
    monkeypatch.setenv("PATH", str(stub_bin.folder))
    assert osutil.system_picker_available() is False
    osutil.run_zenity("/data", "t", collector, 3)
    result = collector.wait_type(PickerResult)[0]
    assert result.path is None and "could not be started" in result.error


def test_picker_always_available_on_windows_and_macos(monkeypatch, stub_bin):
    monkeypatch.setenv("PATH", str(stub_bin.folder))
    for flag in ("WINDOWS", "MACOS"):
        with monkeypatch.context() as m:
            m.setattr(osutil, flag, True)
            assert osutil.system_picker_available() is True


# -------------------------------------------------------------------------------------------- open path / url
@posix_only
def test_open_path_uses_xdg_open_with_clean_env(stub_bin, tmp_path, monkeypatch):
    rec = tmp_path / "opened"
    stub_bin("xdg-open", f'echo "$1|$LD_LIBRARY_PATH" > "{rec}"')
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    errors = []
    osutil.open_path("/data/out folder", errors.append)
    assert wait_until(rec.exists) and wait_until(lambda: rec.read_text().strip() == "/data/out folder|")
    osutil.open_url("https://github.com/")
    assert wait_until(lambda: rec.read_text().strip() == "https://github.com/|")
    time.sleep(0.2)
    assert errors == []


@posix_only
def test_open_path_reports_failures(stub_bin, monkeypatch):
    stub_bin("xdg-open", "exit 4")
    got = threading.Event()
    msgs = []
    osutil.open_path("/nowhere", lambda m: (msgs.append(m), got.set()))
    assert got.wait(5) and "exit code 4" in msgs[0]
    monkeypatch.setenv("PATH", str(stub_bin.folder / "empty"))
    got.clear()
    osutil.open_path("/nowhere", lambda m: (msgs.append(m), got.set()))
    assert got.wait(5) and len(msgs) == 2


@posix_only
def test_open_path_on_macos_uses_open(stub_bin, tmp_path, monkeypatch):
    rec = tmp_path / "opened"
    stub_bin("open", f'echo "$1" > "{rec}"')
    monkeypatch.setattr(osutil, "MACOS", True)
    osutil.open_path("/Volumes/share")
    assert wait_until(lambda: rec.exists() and rec.read_text().strip() == "/Volumes/share")


def test_open_path_on_windows_uses_startfile(monkeypatch):
    opened, msgs = [], []
    done = threading.Event()
    monkeypatch.setattr(osutil, "WINDOWS", True)
    monkeypatch.setattr(os, "startfile", lambda p: (opened.append(p), done.set()), raising=False)
    osutil.open_path("Z:\\Stitched")
    assert done.wait(5) and opened == ["Z:\\Stitched"]

    def refuse(p):
        raise OSError(2, "The system cannot find the file specified")
    monkeypatch.setattr(os, "startfile", refuse, raising=False)
    done.clear()
    osutil.open_path("Z:\\gone", lambda m: (msgs.append(m), done.set()))
    assert done.wait(5) and "cannot find" in msgs[0]


# ------------------------------------------------------------------------------------------------- keep awake
def test_keep_awake_windows(monkeypatch):
    fake = FakeWinDLL()
    monkeypatch.setattr(osutil, "WINDOWS", True)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    osutil.keep_awake(True)
    osutil.keep_awake(False)
    assert fake.calls == [("kernel32", "SetThreadExecutionState", 0x80000001),
                          ("kernel32", "SetThreadExecutionState", 0x80000000)]
    fake.results["SetThreadExecutionState"] = OSError("no")
    osutil.keep_awake(True)                                                     # swallowed


@posix_only
def test_keep_awake_macos_caffeinate(stub_bin, tmp_path, monkeypatch):
    rec = tmp_path / "caf"
    stub_bin("caffeinate", f'echo "$@" > "{rec}"\nexec sleep 30')
    monkeypatch.setattr(osutil, "MACOS", True)
    osutil.keep_awake(True)
    assert wait_until(lambda: rec.exists() and rec.read_text().strip() == f"-i -w {os.getpid()}")
    proc = osutil._awake["proc"]
    osutil.keep_awake(True)                                                     # already running: no second one
    time.sleep(0.2)
    assert osutil._awake["proc"] is proc
    osutil.keep_awake(False)
    assert wait_until(lambda: osutil._awake["proc"] is None and proc.poll() is not None)
    osutil.keep_awake(True)
    osutil.keep_awake(False)                                                    # quick on/off settles on "off"
    assert wait_until(lambda: osutil._awake["proc"] is None, 5)
    time.sleep(0.3)
    assert osutil._awake["proc"] is None


def test_keep_awake_linux_is_a_no_op(monkeypatch):
    fake, synced = FakeWinDLL(), []
    monkeypatch.setattr(osutil, "WINDOWS", False)
    monkeypatch.setattr(osutil, "MACOS", False)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    monkeypatch.setattr(osutil, "_sync_caffeinate", lambda: synced.append(1))
    osutil.keep_awake(True)
    osutil.keep_awake(False)
    time.sleep(0.1)
    assert fake.calls == [] and synced == [] and osutil._awake["proc"] is None


# -------------------------------------------------------------------------------------------- Windows helpers
def test_dpi_awareness_falls_back_to_user32(monkeypatch):
    fake = FakeWinDLL()
    monkeypatch.setattr(osutil, "WINDOWS", True)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    osutil.set_windows_dpi_awareness()                                          # S_OK: no fallback
    fake.results["SetProcessDpiAwareness"] = -2147024891                        # E_ACCESSDENIED: already set
    osutil.set_windows_dpi_awareness()
    assert fake.calls == [("shcore", "SetProcessDpiAwareness", 1)] * 2
    fake.calls.clear()
    fake.results["SetProcessDpiAwareness"] = OSError("no shcore")
    osutil.set_windows_dpi_awareness()
    assert fake.calls == [("shcore", "SetProcessDpiAwareness", 1), ("user32", "SetProcessDPIAware")]
    fake.results["SetProcessDPIAware"] = AttributeError("none")
    osutil.set_windows_dpi_awareness()                                          # swallowed


def test_error_mode_keeps_existing_flags(monkeypatch):
    fake = FakeWinDLL({"SetErrorMode": lambda mode: 0x0002})
    monkeypatch.setattr(osutil, "WINDOWS", True)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    osutil.set_windows_error_mode()
    assert fake.calls == [("kernel32", "SetErrorMode", 0x8001), ("kernel32", "SetErrorMode", 0x8003)]


def test_hide_on_windows_sets_the_hidden_attribute(monkeypatch):
    fake = FakeWinDLL()
    monkeypatch.setattr(osutil, "WINDOWS", True)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    osutil.hide_on_windows("C:\\out\\.stitcher-in-progress")
    assert fake.calls == [("kernel32", "SetFileAttributesW", "C:\\out\\.stitcher-in-progress", 0x2)]
    fake.results["SetFileAttributesW"] = OSError("boom")
    osutil.hide_on_windows("C:\\x")                                             # swallowed


def test_windows_helpers_do_nothing_elsewhere(monkeypatch):
    fake = FakeWinDLL()
    monkeypatch.setattr(osutil, "WINDOWS", False)
    monkeypatch.setattr(osutil, "ctypes", types.SimpleNamespace(windll=fake))
    osutil.set_windows_dpi_awareness()
    osutil.set_windows_error_mode()
    osutil.hide_on_windows("/tmp/x")
    assert fake.calls == []


# ------------------------------------------------------------------------------------------------- X resources
@posix_only
@pytest.mark.parametrize("body,expected", [('printf "Xft.antialias:\\t1\\nXft.dpi:\\t144\\n"', 144.0),
                                           ('printf "Xft.dpi: 96.5\\n"', 96.5),
                                           ('printf "*customization:\\t-color\\n"', None),
                                           ('printf "Xft.dpi:\\t0\\n"', None),
                                           ("exit 1", None)])
def test_linux_xft_dpi(stub_bin, body, expected):
    stub_bin("xrdb", body)
    assert osutil.linux_xft_dpi() == expected


@posix_only
def test_linux_xft_dpi_times_out_and_handles_missing_xrdb(stub_bin, monkeypatch):
    stub_bin("xrdb", "exec sleep 10")
    t0 = time.monotonic()
    assert osutil.linux_xft_dpi(timeout=0.3) is None and time.monotonic() - t0 < 3
    monkeypatch.setenv("PATH", str(stub_bin.folder / "empty"))
    assert osutil.linux_xft_dpi() is None


# ---------------------------------------------------------------------------------------------------- A6
def test_this_package_needs_no_tkinter():
    code = ("import sys; sys.modules['tkinter'] = None\n"
            "import yfs.core.osutil, yfs.core.settings\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    for name in ("osutil.py", "settings.py"):
        assert "tkinter" not in (REPO / "yfs" / "core" / name).read_text(encoding="utf-8")
