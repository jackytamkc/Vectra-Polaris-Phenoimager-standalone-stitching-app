"""Fixtures for GUI tests (FINAL_SPEC §7): a private Xvfb display, a themed Tk root, icons and an app context.

- Every test under tests/gui/ is marked ``gui``.
- On Linux a private ``Xvfb :1NN`` (NN from the PID, skipping displays in use) is started once per session and
  DISPLAY points at it, so tests never draw on a real desktop. Without Xvfb the existing DISPLAY is used; with
  neither, the GUI tests are skipped. Set YFS_TEST_USE_DISPLAY=1 to use an explicitly supplied DISPLAY without
  starting a private server (for managed test environments).
- ``tk_root``: a fresh ``tk.Tk`` with ``install_theme`` applied (auto scale; parametrize indirectly with a scale,
  e.g. ``@pytest.mark.parametrize("tk_root", [1.5], indirect=True)``); destroyed afterwards. It starts from the
  display's own DPI (``display_scaling``), whatever scale an earlier root of the session installed, and gives it
  back when it is destroyed. Tests that make their own roots should do the same.
- ``theme``, ``icons``: the root's Theme and an IconSet at its icon size.
- ``ui_ctx``: an AppContext with a running UiBus; fs/mounts/settings are None (replace them with
  ``dataclasses.replace(ui_ctx, fs=...)`` when a test needs them).
- ``pump``: ``pump(root, ms)`` runs the Tk event loop for ``ms`` milliseconds.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

GUI_DIR = Path(__file__).resolve().parent

# Some tests do `from conftest import FIXTURES` (or real_roots). pytest imports every conftest.py as the module
# "conftest", so once tests/gui is collected that name points here: re-export the root conftest's plain helpers
# (not its fixtures, which pytest already registered from the root file).
_spec = importlib.util.spec_from_file_location("yfs_tests_root_conftest", GUI_DIR.parent / "conftest.py")
_root_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_conftest)
FIXTURES, REPO, Collector = _root_conftest.FIXTURES, _root_conftest.REPO, _root_conftest.Collector
load_module, real_subset, real_roots = _root_conftest.load_module, _root_conftest.real_subset, _root_conftest.real_roots


def pytest_collection_modifyitems(config, items):
    for item in items:
        if GUI_DIR in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.gui)


def _die_with_parent() -> None:
    """Runs in the Xvfb child: SIGTERM it when pytest dies, even on a crash (no orphaned servers)."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)    # 1 = PR_SET_PDEATHSIG
    except Exception:
        pass


def _start_xvfb(xvfb: str):
    """Start Xvfb on the first free :1NN display at or after 100 + PID % 100; returns (process, display)."""
    first = os.getpid() % 100
    for n in (100 + (first + i) % 100 for i in range(100)):
        if os.path.exists(f"/tmp/.X11-unix/X{n}") or os.path.exists(f"/tmp/.X{n}-lock"):
            continue
        proc = subprocess.Popen([xvfb, f":{n}", "-screen", "0", "1600x1000x24", "-nolisten", "tcp"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=_die_with_parent)
        for _ in range(100):                          # up to 5 s for the socket to appear
            if os.path.exists(f"/tmp/.X11-unix/X{n}") or proc.poll() is not None:
                break
            time.sleep(0.05)
        if proc.poll() is None and os.path.exists(f"/tmp/.X11-unix/X{n}"):
            return proc, f":{n}"
        proc.kill()
        proc.wait()
    return None, None


@pytest.fixture(scope="session", autouse=True)
def xdisplay():
    """The display used by every GUI test (see the module docstring)."""
    if not sys.platform.startswith("linux"):
        yield None                                    # Windows / macOS: the native display
        return
    if os.environ.get("YFS_TEST_USE_DISPLAY") == "1":
        if not os.environ.get("DISPLAY"):
            pytest.fail("YFS_TEST_USE_DISPLAY requires DISPLAY")
        yield os.environ["DISPLAY"]
        return
    xvfb = shutil.which("Xvfb")
    proc, display = _start_xvfb(xvfb) if xvfb else (None, None)
    if proc is None:
        if not os.environ.get("DISPLAY"):
            pytest.skip("no X display and Xvfb is not available")
        yield os.environ["DISPLAY"]
        return
    old = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = display
    try:
        yield display
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if old is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old


@pytest.fixture(scope="session", autouse=True)
def display_scaling(xdisplay):
    """`tk scaling` of a fresh root, measured before any test changed it. Tk keeps one X connection per display for
    the whole process and `tk scaling` rewrites the screen size stored in it, so every later root inherits the DPI
    of the scale installed last (an auto-scaled root after a 1.5 one came out at 1.5, with 24 px icons)."""
    import tkinter as tk

    root = tk.Tk()
    try:
        return float(root.tk.call("tk", "scaling"))
    finally:
        root.destroy()


def _pump(root, ms: int = 100) -> None:
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


@pytest.fixture
def pump():
    return _pump


@pytest.fixture
def tk_root(request, xdisplay, display_scaling):
    import tkinter as tk
    from yfs.ui.theme import install_theme

    root = tk.Tk()
    root.geometry("+0+0")
    root.tk.call("tk", "scaling", display_scaling)    # the display's own DPI, not the last root's scale
    install_theme(root, getattr(request, "param", None))
    yield root
    try:
        root.update_idletasks()                       # run pending idle work (e.g. <<ThemeChanged>>) first
        root.tk.call("tk", "scaling", display_scaling)    # leave the display as it was found
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def theme(tk_root):
    from yfs.ui.theme import theme_of

    return theme_of(tk_root)


@pytest.fixture
def icons(tk_root, theme):
    from yfs.ui.icons import IconSet
    from yfs.ui.theme import install_indicators

    icons = IconSet(tk_root, theme.icon_px)
    install_indicators(tk_root, icons)
    return icons


@pytest.fixture
def ui_ctx(tk_root, theme, icons):
    from yfs.ui.app import AppContext, UiBus

    bus = UiBus(tk_root)
    bus.start()
    yield AppContext(tk_root, theme, icons, bus, None, None, None, object, lambda: None, frozenset, lambda: False)
    bus.stop()
