"""Final-review regressions: truthful folder facts and predictable keyboard/navigation actions."""
from __future__ import annotations

import errno
import os
import threading
import tkinter as tk

import numpy as np
import pytest
import tifffile

from test_browser import (env, key, mounts, overlay_buttons, overlay_text, pump, rows, select,  # noqa: F401
                          settled, strip, tree, until)
from test_browser_facts import slide
from test_browser_keys import click
from yfs.core import fsservice, preflight
from yfs.core.model import OutputIndex, OutputInfo
from yfs.ui import browser


def wait(b, pred, what="browser result"):
    return until(b, pred, timeout=2, what=what)


@pytest.mark.parametrize("verdict", ["valid", "incomplete", "unchecked"])
def test_colliding_slide_names_never_inherit_a_bare_output_badge(env, tree, monkeypatch, verdict):
    """BROWSER-1: even a verified pyramid cannot identify either slide by their shared basename."""
    for sample in ("SampleA", "SampleB"):
        slide(tree / sample / "Scan1", sample)
    output = tree / "Stitched" / "Scan1.ome.tif"
    output.parent.mkdir()
    with tifffile.TiffWriter(output) as tif:
        tif.write(np.zeros((32, 32), np.uint16), subifds=1, metadata=None)
        tif.write(np.zeros((16, 16), np.uint16), subfiletype=1, metadata=None)
    if verdict == "incomplete":
        output.write_bytes(b"")

    def unreadable(path):
        raise OSError(errno.EIO, "Injected read error")

    with monkeypatch.context() as patch:
        if verdict == "unchecked":
            patch.setattr(preflight, "_ReadErrors", unreadable)
        assert preflight.verify_output(str(output))[0] == verdict
    info = output.stat()
    env.out_index = OutputIndex(str(tree / "Stitched"), True, frozenset(),
                                {"scan1.ome.tif": OutputInfo("Scan1.ome.tif", info.st_size, info.st_mtime_ns)}, {}, (),
                                None, "/", "local", False)
    b = env.open((str(tree / "SampleA"), None))
    wait(b, lambda: settled(b))
    assert rows(b)["Scan1"][1] == "Slide tiles"
    b.navigate(str(tree / "SampleB"))
    wait(b, lambda: settled(b))
    assert rows(b)["Scan1"][1] == "Slide tiles"


@pytest.mark.parametrize("mode", ["output", "temp"])
@pytest.mark.parametrize("submit", [False, True])
def test_path_editing_invalidates_a_pending_write_check(env, tree, monkeypatch, mode, submit):
    """BROWSER-2: an old write answers while editing or resolving another path; only a fresh Use can accept."""
    old, new = tree / "SlowSave", tree / "Destination"
    old.mkdir()
    new.mkdir()
    write_gate, lookup_gate = threading.Event(), threading.Event()
    wrote, locating = threading.Event(), threading.Event()
    locate = browser._locate

    def write(path):
        wrote.set()
        assert write_gate.wait(3)

    def lookup(path):
        if path == str(new):
            locating.set()
            assert lookup_gate.wait(3)
        return locate(path)

    monkeypatch.setattr(browser, "write_probe", write)
    monkeypatch.setattr(browser, "_locate", lookup)
    b = env.open((str(old), None), mode=mode)
    wait(b, lambda: settled(b))
    try:
        b._primary.invoke()
        wait(b, wrote.is_set)
        key(b.tree, "<Control-l>")
        b._edit.delete(0, "end")
        b._edit.insert(0, str(new))                  # trace catches paste without a KeyRelease
        if submit:
            key(b._edit, "<Return>")
            wait(b, locating.is_set)
        write_gate.set()
        pump(b, 100)
        assert b.winfo_exists() and env.done == []
        assert b.current() == str(old)
        if not submit:
            key(b._edit, "<Return>")
            wait(b, locating.is_set)
        lookup_gate.set()
        wait(b, lambda: b.current() == str(new) and settled(b))
        assert env.done == []
        b._primary.invoke()
        wait(b.ctx.root, lambda: env.done)
        assert env.done == [str(new)]
    finally:
        write_gate.set()
        lookup_gate.set()


def test_copy_path_from_the_actual_places_menu(env, tree, tk_root, monkeypatch):
    """BROWSER-3: Tk's clipboard internals must remain callable on the browser Toplevel."""
    errors = []
    monkeypatch.setattr(tk_root, "report_callback_exception", lambda *args: errors.append(args))
    ctx = env.ctx()
    ctx.settings.add_recent("input", str(tree))
    b = env.open((str(tree), None), ctx=ctx)
    wait(b, lambda: settled(b) and b._place_items)
    iid = next(i for i, (p, _lane) in b._place_items.items() if p.path == str(tree))
    b._side.see(iid)
    b.update()
    x, y, width, height = b._side.bbox(iid)
    tk_root.clipboard_clear()
    tk_root.clipboard_append("old clipboard")
    b._side.event_generate("<Button-3>", x=x + 20, y=y + height // 2,
                           rootx=b._side.winfo_rootx() + x + 20, rooty=b._side.winfo_rooty() + y + height // 2)
    b.update()
    menu = next(w for w in reversed(b.winfo_children()) if isinstance(w, tk.Menu) and w.winfo_ismapped())
    try:
        assert menu.entrycget(0, "label") == "Copy path"
        menu.invoke(0)
        assert errors == []
        assert tk_root.clipboard_get() == str(tree)
    finally:
        menu.unpost()
        b.grab_set()


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO])
@pytest.mark.parametrize("mixed", [False, True])
def test_failed_descendants_remain_unknown_visible_and_retryable(env, tree, code, mixed):
    """BROWSER-5: selected and current parents retain read failures, including alongside successful slides."""
    failed = tree / "Project" / "Unreadable_Scan1"
    slide(failed, "Unreadable", 3)
    if mixed:
        slide(tree / "Project" / "Readable_Scan1", "Readable", 2)
    env.fake.fail(failed, OSError(code, "Injected read failure"))
    b = env.open((str(tree), None))
    wait(b, lambda: settled(b))
    select(b, "Project")
    assert "couldn't be read" in rows(b)["Project"][1]
    assert strip(b)[0] == "Warn" and "1 folder couldn't be read" in strip(b)[1]
    assert "No tiles found in Project" not in strip(b)[1]
    if mixed:
        assert "1+ slide folder" in strip(b)[1]
    key(b.tree, "<Escape>")
    assert strip(b)[0] == "Warn" and "1 folder couldn't be read" in strip(b)[1]
    key(b.tree, "<Alt-t>")
    wait(b, lambda: "resort" not in b._jobs)
    assert "Project" in rows(b)
    select(b, "Project")
    env.fake.clear(failed)
    key(b.tree, "<F5>")
    wait(b, lambda: settled(b) and strip(b)[0] == "Ok")
    assert strip(b)[0] == "Ok"
    assert rows(b)["Project"][1] == ("2 slide folders inside" if mixed else "1 slide folder inside")
    assert env.fake.paths("scandir").count(str(failed)) == 2


@pytest.mark.parametrize("mode", ["output", "temp"])
def test_f5_refreshes_capacity_of_the_selected_destination(env, tree, monkeypatch, mode):
    """BROWSER-6: refreshing a selected folder replaces a previously successful capacity fact."""
    (tree / "Destination").mkdir()
    amount, calls = [10 * 1024 ** 3], []

    def usage(path):
        calls.append(path)
        return 20 * 1024 ** 3, 0, amount[0]

    monkeypatch.setattr(env.fake, "disk_usage", usage)
    b = env.open((str(tree), "Destination"), mode=mode)
    wait(b, lambda: "10.0 GB" in strip(b)[1])
    amount[0] = 1024 ** 3
    key(b.tree, "<F5>")
    wait(b, lambda: "1.0 GB" in strip(b)[1])
    assert b.target() == str(tree / "Destination")
    assert calls.count(b.target()) == 2


@pytest.mark.parametrize("mode", ["output", "temp"])
def test_refresh_capacity_cannot_join_or_be_overwritten_by_an_old_check(env, tree, monkeypatch, mode):
    """BROWSER-6: a second usage request completes before the old one; only the second value survives."""
    old_gate, started = threading.Event(), threading.Event()
    calls = []

    def usage(path):
        calls.append(path)
        if len(calls) == 1:
            started.set()
            assert old_gate.wait(3)
            return 20 * 1024 ** 3, 0, 10 * 1024 ** 3
        return 20 * 1024 ** 3, 0, 1024 ** 3

    monkeypatch.setattr(env.fake, "disk_usage", usage)
    b = env.open((str(tree), None), mode=mode)
    try:
        wait(b, started.is_set)
        key(b.tree, "<F5>")
        wait(b, lambda: "1.0 GB" in strip(b)[1])
        assert len(calls) == 2
        old_gate.set()
        pump(b, 150)
        assert "1.0 GB" in strip(b)[1] and "10.0 GB" not in strip(b)[1]
    finally:
        old_gate.set()


@pytest.mark.parametrize("mode", ["output", "temp"])
@pytest.mark.parametrize("failure", ["io", "timeout"])
def test_capacity_failure_settles_until_explicit_retry(env, tree, monkeypatch, mode, failure):
    """BROWSER-7: a failed capacity fact is distinct from write access, with no automatic retry loop."""
    calls, gate = [], threading.Event()
    monkeypatch.setitem(fsservice.DEADLINE["usage"], "local", 0.15)

    def usage(path):
        calls.append(path)
        if len(calls) == 1:
            if failure == "io":
                raise OSError(errno.EIO, "Injected capacity error")
            assert gate.wait(3)
        return 20 * 1024 ** 3, 0, 1024 ** 3

    monkeypatch.setattr(env.fake, "disk_usage", usage)
    b = env.open((str(tree), None), mode=mode)
    try:
        wait(b, lambda: "Couldn't check free space" in strip(b)[1])
        assert strip(b)[0] == "Warn" and "F5" in strip(b)[1]
        assert "Can't write" not in strip(b)[1] and "Can't save" not in strip(b)[1]
        before = strip(b)
        for _ in range(3):
            b._update_target()
            pump(b, 80)
        assert strip(b) == before and len(calls) == 1
        gate.set()
        key(b.tree, "<F5>")
        wait(b, lambda: "Free space: 1.0 GB" in strip(b)[1])
        assert strip(b)[0] == "Ok" and len(calls) == 2
    finally:
        gate.set()


@pytest.mark.parametrize("activation", ["mouse", "keyboard"])
def test_missing_recent_names_and_removes_only_the_attempted_place(env, tree, activation):
    """BROWSER-8: failed Recent B never replaces A's location/selection or removes A's history entry."""
    current, missing = tree / "Projects", tree / "DeletedBatch"
    (current / "KeepSelected").mkdir(parents=True)
    ctx = env.ctx()
    for path in (current, missing):
        ctx.settings.add_recent("input", str(path))
    b = env.open((str(current), "KeepSelected"), ctx=ctx)
    wait(b, lambda: settled(b) and b._place_items and "Checking" not in strip(b)[1])
    selected = b.tree.selection()
    before = strip(b)

    def activate():
        iid = next(i for i, (p, _lane) in b._place_items.items() if p.path == str(missing))
        if activation == "keyboard":
            b._side.focus(iid)
            b._side.selection_set(iid)
            key(b._side, "<Return>")
        else:
            x, y, width, height = b._side.bbox(iid)
            click(b, b._side, x + 20, y + height // 2)
        wait(b, lambda: b._ov_kind == "error")

    def button(text):
        return next(w for frame in b._ov.winfo_children() for w in frame.winfo_children()
                    if w.winfo_class() == "TButton" and w.cget("text") == text)

    activate()
    assert "DeletedBatch" in overlay_text(b)
    assert overlay_buttons(b) == ["Dismiss", "Remove from recent"]
    key(button("Dismiss"), "<space>")
    assert b.current() == str(current) and b.tree.selection() == selected and strip(b) == before
    assert not b._ov.winfo_manager()
    activate()
    key(button("Remove from recent"), "<space>")
    assert [r["path"] for r in ctx.settings.get("recent.inputs")] == [str(current)]
    assert b.current() == str(current) and b.tree.selection() == selected and strip(b) == before
    assert not b._ov.winfo_manager()


@pytest.mark.parametrize("completion", ["<Tab>", "<Down>"])
def test_autocomplete_includes_resolved_folder_aliases_only(env, tree, completion):
    """BROWSER-10: aliases retain their names in completion; file/broken/hidden links do not leak in."""
    slide(tree / "Actual", "Slide1")
    (tree / "note.txt").write_text("notes", encoding="utf-8")
    for name, target in (("Shortcut", "Actual"), ("ShortFile", "note.txt"), ("ShortBroken", "missing"),
                         (".ShortHidden", "Actual")):
        (tree / name).symlink_to(tree / target)
    b = env.open((str(tree), None))
    wait(b, lambda: settled(b) and b._rows[b._names["Shortcut"]].kind == "dir"
         and not any(r.probing for r in b._rows.values()))
    key(b.tree, "<Control-l>")
    b._edit.delete(0, "end")
    b._edit.insert(0, "Short")
    key(b._edit, "<KeyRelease-t>")
    key(b._edit, completion)
    if completion == "<Tab>":
        assert b._edit.get() == str(tree / "Shortcut") + os.sep
    else:
        assert b._lb_items[1] == ["Shortcut"]
        key(b._edit, "<Down>")                      # the first Down shows; the second chooses
        assert b._lb_index() == 0
    key(b._edit, "<Return>")
    wait(b, lambda: b.current() == str(tree / "Actual") and settled(b))


def test_sort_column_and_direction_are_operable_with_keys(env, tree):
    """BROWSER-11: Tab reaches Sort; arrows and Enter select columns and direction, changing the real listing."""
    for i, name in enumerate(("Alpha", "Beta", "Gamma")):
        slide(tree / name, name, 3 - i)
        os.utime(tree / name, (1000000000 + i * 86400, 1000000000 + i * 86400))
    b = env.open((str(tree), None))
    wait(b, lambda: settled(b))
    b.tree.focus_force()
    for _ in range(20):
        focused = b.focus_get()
        if focused is not None and focused.winfo_class() == "TMenubutton" and focused.cget("text") == "Sort":
            break
        focused.event_generate("<Tab>")
        b.update()
    else:
        pytest.fail("No keyboard-focusable Sort control")
    sort = b.focus_get()

    def choose(label):
        key(sort, "<space>")
        menu = b.nametowidget(str(sort.cget("menu")))
        assert menu.winfo_ismapped()
        for _ in range(10):
            active = menu.index("active")
            if active is not None and menu.entrycget(active, "label") == label:
                break
            menu.event_generate("<Down>")
            b.update()
        else:
            pytest.fail("Sort choice not reachable: " + label)
        menu.event_generate("<Return>")
        b.update()

    choose("Modified")
    wait(b, lambda: list(rows(b)) == ["Gamma", "Beta", "Alpha"])
    choose("Ascending")
    wait(b, lambda: list(rows(b)) == ["Alpha", "Beta", "Gamma"])
    choose("Tiles")
    wait(b, lambda: list(rows(b)) == ["Alpha", "Beta", "Gamma"])
    choose("Ascending")
    wait(b, lambda: list(rows(b)) == ["Gamma", "Beta", "Alpha"])
