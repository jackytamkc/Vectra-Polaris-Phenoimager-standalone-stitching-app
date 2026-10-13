"""Shared pytest fixtures for the v2 test-suite.

- ``synth_tree``: a synthetic Vectra/PhenoImager tile tree (tests/fixtures/make_synth.py), built once per session.
- ``cfg_dir``: points YFS_CONFIG_DIR at a private temporary folder (settings + logs never touch the real profile).
- ``collector``: a thread-safe ``post`` callable that records events, with small helpers for waiting on them.
- ``legacy_engine``: the verbatim v1 ``stitcher_RAM.py`` (tests/fixtures/legacy_engine_v1.py) for identity checks.
- ``live_process`` / ``dead_process``: (pid, start time) of a process of this user that runs / has exited, as the
  owner of a partial file (D3).

Real data is never required. Opt-in checks read these environment variables and skip when they are unset:
- YFS_REAL_SUBSET: a folder of real ``*_component_data.tif`` tiles (read-only; symlinks are fine).
- YFS_REAL_ROOTS: os.pathsep-separated folders for read-only discovery smoke tests.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _collect_tk_garbage_on_the_main_thread():
    """Pytest can retain a destroyed root until the next test. Collect before core workers can inherit it too."""
    gc.collect()
    yield
    gc.collect()


def load_module(name: str, path: Path):
    """Import a Python file under an explicit module name (used for fixtures that are not packages)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synth_tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth") / "synth"
    make = load_module("yfs_test_make_synth", FIXTURES / "make_synth.py")
    make.main(str(root))
    yield root
    locked = root / "locked"
    if locked.exists():
        os.chmod(locked, 0o755)          # let pytest clean the temp tree up afterwards


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "yfs-config"
    d.mkdir()
    monkeypatch.setenv("YFS_CONFIG_DIR", str(d))
    return d


class Collector:
    """A ``post`` replacement: records every event; safe to call from any thread."""

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def __call__(self, ev) -> None:
        with self._cond:
            self.events.append(ev)
            self._cond.notify_all()

    post = __call__

    def of_type(self, *types):
        with self._lock:
            return [e for e in self.events if isinstance(e, types)]

    def wait_for(self, pred, timeout: float = 10.0):
        """Block until ``pred(events)`` is truthy (returns its value) or raise TimeoutError."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                value = pred(list(self.events))
                if value:
                    return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"condition not met within {timeout} s; got {len(self.events)} events")
                self._cond.wait(min(remaining, 0.1))

    def wait_type(self, typ, count: int = 1, timeout: float = 10.0):
        return self.wait_for(lambda evs: [e for e in evs if isinstance(e, typ)][:count]
                             if len([e for e in evs if isinstance(e, typ)]) >= count else None, timeout)


@pytest.fixture
def collector() -> Collector:
    return Collector()


@pytest.fixture(scope="session")
def legacy_engine():
    """The unmodified v1 module. Callers that need determinism set ``legacy_engine.NUM_THREADS = 1``."""
    return load_module("legacy_engine_v1", FIXTURES / "legacy_engine_v1.py")


def _sleeper():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    return p, psutil.Process(p.pid).create_time()


@pytest.fixture
def live_process():
    """(pid, start time) of a running process of this user on this computer."""
    p, started = _sleeper()
    yield p.pid, started
    p.kill()
    p.wait()


@pytest.fixture
def dead_process():
    """(pid, start time) of a process of this user that has exited."""
    p, started = _sleeper()
    p.kill()
    p.wait()
    return p.pid, started


def real_subset() -> Path | None:
    p = os.environ.get("YFS_REAL_SUBSET")
    return Path(p) if p and Path(p).is_dir() else None


def real_roots() -> list:
    return [Path(p) for p in os.environ.get("YFS_REAL_ROOTS", "").split(os.pathsep) if p and Path(p).is_dir()]
