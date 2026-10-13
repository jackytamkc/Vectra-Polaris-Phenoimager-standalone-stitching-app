"""Tiny helpers for scripted, headless Tk screenshots under Xvfb.

Usage inside a test script (DISPLAY must point at an Xvfb server):

    from shotlib import Steps, grab_widget
    steps = Steps(root, delay_ms=600)
    steps.add(lambda: grab_widget(root, "/path/main.png"))
    steps.add(lambda: ...)
    steps.run()          # schedules everything, destroys root at the end
    root.mainloop()
"""
import os
import traceback

from PIL import ImageGrab


def grab_widget(widget, path, pad=0):
    """Screenshot exactly the on-screen area of a Tk widget/toplevel."""
    widget.update_idletasks()
    widget.update()
    x, y = widget.winfo_rootx(), widget.winfo_rooty()
    w, h = widget.winfo_width(), widget.winfo_height()
    bbox = (max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)
    img = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"), bbox=bbox)
    img.save(path)
    print(f"[shot] {path} {img.size}", flush=True)
    return path


def grab_screen(path, bbox=None):
    img = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"), bbox=bbox)
    if bbox is None:
        bb = img.getbbox()
        if bb:
            img = img.crop(bb)
    img.save(path)
    print(f"[shot] {path} {img.size}", flush=True)
    return path


class Steps:
    """Run callables sequentially on the Tk event loop with a delay between them.

    A step may return an int to override the delay before the next step (ms), e.g. to wait for
    background scans. Exceptions are printed and the run continues, so one broken step doesn't
    hide the screenshots of the others.
    """

    def __init__(self, root, delay_ms=600, quit_at_end=True):
        self.root = root
        self.delay = delay_ms
        self.steps = []
        self.quit_at_end = quit_at_end
        self.errors = []

    def add(self, fn, name=None):
        self.steps.append((name or getattr(fn, "__name__", "step"), fn))
        return self

    def wait_until(self, pred, timeout_ms=20000, poll_ms=200, name="wait"):
        """Insert a step that polls `pred()` until true (or timeout) before continuing."""
        state = {"t": 0}

        def _wait():
            if pred():
                return self.delay
            state["t"] += poll_ms
            if state["t"] >= timeout_ms:
                print(f"[steps] timeout waiting for {name}", flush=True)
                self.errors.append(f"timeout:{name}")
                return self.delay
            return ("again", poll_ms)

        self.steps.append((name, _wait))
        return self

    def run(self):
        self._i = 0
        self.root.after(self.delay, self._next)

    def _next(self):
        if self._i >= len(self.steps):
            print(f"[steps] done, errors={self.errors}", flush=True)
            if self.quit_at_end:
                self.root.after(50, self.root.destroy)
            return
        name, fn = self.steps[self._i]
        delay = self.delay
        try:
            r = fn()
            if isinstance(r, tuple) and r and r[0] == "again":
                self.root.after(r[1], self._next)   # re-run same step
                return
            if isinstance(r, int):
                delay = r
        except Exception:
            print(f"[steps] step {name} failed:\n{traceback.format_exc()}", flush=True)
            self.errors.append(name)
        self._i += 1
        self.root.after(delay, self._next)
