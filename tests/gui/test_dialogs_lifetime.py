from __future__ import annotations

import gc
import threading
import tkinter as tk
import weakref

import pytest

from yfs.ui import browser
from yfs.ui.dialogs import ChannelNamesDialog, NewFolderDialog, SettingsDialog

from test_browser import env, settled, tree, until  # noqa: F401
from test_dialogs import buttons, ctx, find, make_plan  # noqa: F401


@pytest.fixture
def finalizers(monkeypatch):
    records, held = [], []
    real_variable, real_image = tk.Variable.__del__, tk.Image.__del__

    def spy(real_del):
        def record(obj):
            records.append((id(obj), threading.current_thread().name))
            if threading.current_thread() is threading.main_thread():
                real_del(obj)
            else:
                held.append((real_del, obj))                 # record a failure without a worker Tcl call
        return record

    monkeypatch.setattr(tk.Variable, "__del__", spy(real_variable))
    monkeypatch.setattr(tk.Image, "__del__", spy(real_image))
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield records
    finally:
        gc.collect()
        for real_del, obj in held:
            real_del(obj)
        held.clear()
        if enabled:
            gc.enable()


def worker_collect():
    worker = threading.Thread(target=gc.collect, name="dialog-gc-worker")
    worker.start()
    worker.join(1)
    assert not worker.is_alive()


@pytest.mark.parametrize("kind", ["folder", "channels", "settings"])
@pytest.mark.parametrize("tcl_destroy", [False, True])
def test_parent_destruction_releases_variables_and_images(ctx, tk_root, finalizers, kind, tcl_destroy):
    parent = tk.Toplevel(tk_root)
    done = []
    if kind == "channels":
        plan, ids = make_plan()
        ChannelNamesDialog.open(parent, ctx, plan, [ids["A_Scan1"]], done.append)
        dlg = find(parent, ChannelNamesDialog)
    elif kind == "settings":
        dlg = SettingsDialog.open(parent, ctx, False, done.append)
    else:
        dlg = NewFolderDialog.ask(parent, ctx, frozenset(), done.append)
    dlg._test_image = tk.PhotoImage(master=dlg, width=1, height=1)
    objects = [v for v in vars(dlg).values() if isinstance(v, (tk.Variable, tk.Image))]
    objects += list(getattr(dlg, "vars", ()))
    ids = {id(v) for v in objects}
    refs = [weakref.ref(v) for v in objects]
    del objects
    if tcl_destroy:
        parent.tk.call("destroy", str(parent))
    else:
        parent.destroy()
    worker_collect()                                         # even if a caller still holds the dialog
    assert all(ref() is None for ref in refs)
    assert all(thread == "MainThread" for i, thread in finalizers if i in ids)
    assert len([i for i, _ in finalizers if i in ids]) == len(ids)
    assert done == [] and dlg._done_cb is None and dlg._regrab == ""
    dlg._close(None)                                         # a queued close is harmless
    assert done == []


def test_pending_save_destroys_nested_new_folder_on_the_tk_thread(env, tree, tk_root, monkeypatch, finalizers):
    entered, release = threading.Event(), threading.Event()
    real_probe = browser.write_probe

    def slow_probe(path):
        entered.set()
        assert release.wait(2)
        return real_probe(path)

    monkeypatch.setattr(browser, "write_probe", slow_probe)
    b = env.open((str(tree), None), mode="output")
    try:
        until(tk_root, lambda: settled(b), timeout=2)
        b._primary.invoke()
        until(tk_root, entered.is_set, timeout=2)
        new = buttons(b)["New folder"]
        assert new.winfo_ismapped() and not new.instate(["disabled"])
        new.invoke()
        dlg = find(b, NewFolderDialog)
        dlg._test_image = tk.PhotoImage(master=dlg, width=1, height=1)
        refs = [weakref.ref(dlg.var), weakref.ref(dlg._test_image)]
        ids = {id(ref()) for ref in refs}
        b.destroy()                                         # parent closes while both actions are outstanding
        release.set()
        until(tk_root, lambda: not b.winfo_exists(), timeout=2)
        worker_collect()
        assert not dlg.winfo_exists() and env.done == []
        assert all(ref() is None for ref in refs)
        assert len([i for i, _ in finalizers if i in ids]) == 2
        assert all(thread == "MainThread" for i, thread in finalizers if i in ids)
        assert dlg._done_cb is None and dlg._regrab == ""
    finally:
        release.set()
        if b.winfo_exists():
            b.cancel()
