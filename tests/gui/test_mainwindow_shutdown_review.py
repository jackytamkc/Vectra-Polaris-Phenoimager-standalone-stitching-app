"""A blocked log handler or stream cannot keep either shutdown path alive."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("blocked", ["handler", "flush"])
def test_shutdown_has_a_deadline_without_blocking_tk(tmp_path, forced, blocked):
    heartbeat = tmp_path / "heartbeat"
    script = r'''
import logging
import logging.handlers
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from yfs.ui import app

root = tk.Tk()
root.withdraw()
gate = threading.Event()
entered = threading.Event()
listener = None
if sys.argv[2] == "handler":
    class Blocked(logging.Handler):
        def emit(self, record):
            entered.set()
            gate.wait()
    messages = queue.Queue()
    listener = logging.handlers.QueueListener(messages, Blocked())
    listener.start()
    messages.put(logging.LogRecord("test", 20, "", 1, "test", (), None))
    assert entered.wait(1)
else:
    class BlockedStream:
        def flush(self):
            gate.wait()
    sys.stderr = BlockedStream()

win = app.MainWindow.__new__(app.MainWindow)
win.root, win.listener, win.discovery, win._shutdown, win._timers = root, listener, None, None, {}
win.settings = SimpleNamespace(get=lambda key: {}, set=lambda *args: None, save_now=lambda **kw: True)
win.ctx = SimpleNamespace(fs=SimpleNamespace(shutdown=lambda: None, has_stuck_threads=lambda: False))
app._Shutdown.TIMEOUT_S = .25
root.after(40, lambda: Path(sys.argv[3]).write_text("responsive"))
root.after(0, win._hard_exit if sys.argv[1] == "True" else win._quit)
root.mainloop()
raise AssertionError("blocked cleanup must reach the deadline's process exit")
'''
    result = subprocess.run([sys.executable, "-c", script, str(forced), blocked, str(heartbeat)],
                            env=dict(os.environ), capture_output=True, timeout=3)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert heartbeat.read_text() == "responsive"
