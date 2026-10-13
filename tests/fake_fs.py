"""A fake ``os_api`` for FsService (and Discovery/browser) tests.

It forwards to the real filesystem (normally a pytest tmp tree) but lets a test make calls on chosen paths
block, slow down or fail, and records every call::

    fake = FakeOs()
    gate = fake.block(tmp / "share")          # calls on that path (and below) wait until gate.set()
    fake.delay(tmp / "slow", 0.5)
    fake.fail(tmp / "locked", PermissionError(13, "Permission denied"))
    fs = FsService(post, mounts, os_api=fake)
    ...
    assert str(tmp / "cloud") not in fake.paths("scandir")
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Dict, List, Optional, Tuple


class FakeOs:
    """Drop-in for the os/shutil calls used by FsService: scandir, stat (also lstat: follow_symlinks=False),
    disk_usage, readlink."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []          # (op, path) in call order
        self._rules: Dict[str, Tuple[str, object]] = {}  # normpath -> (action, arg)
        self._lock = threading.Lock()
        self.inside = 0                                   # calls currently blocked or delayed
        self.max_inside = 0

    # ---- configuration
    def block(self, path, recursive: bool = True) -> threading.Event:
        """Calls on `path` (and below, if recursive) wait until the returned event is set."""
        gate = threading.Event()
        self._set(path, "block", gate, recursive)
        return gate

    def delay(self, path, seconds: float, recursive: bool = False) -> None:
        self._set(path, "delay", seconds, recursive)

    def fail(self, path, error: OSError, recursive: bool = False) -> None:
        self._set(path, "fail", error, recursive)

    def clear(self, path=None) -> None:
        with self._lock:
            if path is None:
                self._rules.clear()
            else:
                self._rules.pop(os.path.normpath(str(path)), None)

    def _set(self, path, action: str, arg, recursive: bool) -> None:
        with self._lock:
            self._rules[os.path.normpath(str(path))] = (action + ("*" if recursive else ""), arg)

    # ---- queries
    def paths(self, op: Optional[str] = None) -> List[str]:
        with self._lock:
            return [p for o, p in self.calls if op is None or o == op]

    # ---- the os_api surface
    def scandir(self, path):
        self._hit("scandir", path)
        return os.scandir(path)

    def stat(self, path, **kw):
        # lstat of a link reads only the folder holding it, like the real call: the rules of the folders above it
        # apply, a rule on the link itself (what following it does) does not
        self._hit("stat", path, kw.get("follow_symlinks", True) or not os.path.islink(path))
        return os.stat(path, **kw)

    def disk_usage(self, path):
        self._hit("disk_usage", path)
        return shutil.disk_usage(path)

    def readlink(self, path):
        self._hit("readlink", path, not os.path.islink(path))
        return os.readlink(path)

    def _rule(self, path: str, follow: bool = True):
        """The rule of `path`, or a recursive one of a folder above it. follow=False (lstat or readlink of a link):
        only the folders' recursive rules."""
        want = os.path.normpath(path)
        p = want if follow else os.path.dirname(want)
        while True:
            rule = self._rules.get(p)
            if rule is not None and (p == want or rule[0].endswith("*")):
                return rule[0].rstrip("*"), rule[1]
            parent = os.path.dirname(p)
            if parent == p:
                return None
            p = parent

    def _hit(self, op: str, path, follow: bool = True) -> None:
        path = str(path)
        with self._lock:
            self.calls.append((op, path))
            rule = self._rule(path, follow)
            if rule and rule[0] in ("block", "delay"):
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
        if not rule:
            return
        action, arg = rule
        if action == "fail":
            raise arg
        try:
            if action == "block":
                arg.wait()
            else:
                time.sleep(arg)
        finally:
            with self._lock:
                self.inside -= 1
