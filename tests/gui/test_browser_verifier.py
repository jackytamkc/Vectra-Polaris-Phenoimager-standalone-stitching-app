"""Browser verifier regressions: lane isolation, honest facts, editing and narrow layouts."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from test_browser import (env, key, mounts, overlay_text, pump, rows, select, settled, strip, tree, until)  # noqa: F401
from test_browser_lanes import cifs, slide
from yfs.core import fsservice
from yfs.core.fsservice import FsService
from yfs.core.places import Place
from yfs.ui import browser

pytestmark = pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)


def capture(b, name):
    """Optional evidence on the supplied screenshot display; ordinary test runs write no images."""
    folder = os.environ.get("YFS_BROWSER_SHOTS")
    if folder:
        from shotlib import grab_widget
        pump(b, 180)                                    # include the debounced preview and final geometry
        path = Path(folder) / str(b.ctx.theme.scale)
        path.mkdir(parents=True, exist_ok=True)
        grab_widget(b, str(path / (name + ".png")))


@pytest.mark.parametrize("choose", [False, True])
def test_hung_link_activation_never_uses_the_local_alias(env, tree, monkeypatch, choose):
    share, local = tree / "labshare", tree / "Local"
    slide(share / "ProjectC" / "Slide1_Scan1", "Slide1_Scan1")
    local.mkdir()
    alias = local / "lab_data"
    alias.symlink_to(share / "ProjectC")
    monkeypatch.setitem(fsservice.DEADLINE["readlink"], "network", .25)
    gate = env.fake.block(share)
    alias_gate = env.fake.block(alias)
    try:
        ctx = env.ctx(mounts(cifs(share)))
        b = env.open((str(local), None), ctx=ctx)
        until(b, lambda: "lab_data" in rows(b) and not b._rows[b._names["lab_data"]].probing, timeout=3)
        select(b, "lab_data")
        assert "anyway" not in strip(b)[1]               # unresolved links can only be retried, never used blindly
        capture(b, "link_timeout_row")
        key(b.tree, "<Control-Return>" if choose else "<Return>")
        pump(b, 100)
        assert not env.done and str(alias) not in env.fake.paths("scandir")
        assert "labshare" in overlay_text(b) or "labshare" in strip(b)[1]
        answered = []
        rid = ctx.fs.call(str(local), "locate", lambda: "local works", ticket=0, owner="browser")
        b._calls[rid] = (None, lambda e: answered.append(e.result))
        until(b, lambda: answered, timeout=1)
        assert answered == ["local works"] and ctx.fs.health(str(local)) != "unresponsive"
        capture(b, "link_timeout_activation")
        gate.set()
        if choose:
            until(b, lambda: env.done, timeout=3)
            assert env.done == [str(share / "ProjectC")]
        else:
            until(b, lambda: b.current() == str(share / "ProjectC") and settled(b), timeout=3)
            capture(b, "resolved_share")
    finally:
        gate.set()
        alias_gate.set()


@pytest.mark.parametrize("initial", [False, True])
def test_typed_and_remembered_paths_resolve_linked_ancestors_before_any_following_io(env, tree, monkeypatch, initial):
    share, local = tree / "labshare", tree / "Local"
    slide(share / "ProjectC" / "Slide1_Scan1", "Slide1_Scan1")
    local.mkdir()
    alias = tree / "alias"
    (tree / "bridge").symlink_to(share)
    alias.symlink_to(tree / "bridge")
    typed = alias / "ProjectC"
    followed = []
    real = browser._locate

    def locate(path):
        followed.append(path)
        return real(path)
    monkeypatch.setattr(browser, "_locate", locate)
    ctx = env.ctx(mounts(cifs(share)))
    b = env.open((str(typed if initial else local), None), ctx=ctx)
    if not initial:
        until(b, lambda: settled(b), timeout=3)
        b.edit_path(str(typed))
        key(b._edit, "<Return>")
    until(b, lambda: not b._editing and settled(b), timeout=3)
    assert str(typed) not in followed and str(typed) not in env.fake.paths("scandir")
    assert b.current() == str(share / "ProjectC")
    assert ctx.fs.lane_key(b.current()) == str(share)
    capture(b, "remembered_resolved_path" if initial else "typed_resolved_path")


def test_timed_out_dates_report_failure_then_recover_without_losing_tile_facts(env, tree, monkeypatch):
    for i in range(6):
        folder = tree / f"Export{i}"
        slide(folder, f"Slide{i}_Scan1")
        os.utime(folder, (1700000000 + i * 86400, 1700000000 + i * 86400))
    gate = threading.Event()
    real, batches = FsService._stat_names, []

    def stat_names(fs, path, names):
        batches.append(names)
        gate.wait()
        return real(fs, path, names)
    monkeypatch.setattr(FsService, "_stat_names", stat_names)
    monkeypatch.setitem(fsservice.DEADLINE["stat"], "local", .25)
    ctx = env.ctx()
    ctx.settings.set("browser.sort", ["modified", "desc"])
    try:
        b = env.open((str(tree), None), ctx=ctx)
        until(b, lambda: settled(b) and batches and not b._dating and "resort" not in b._jobs, timeout=3)
        assert "F5" in b._notes[b._others] and "date" in b._notes[b._others].lower()
        assert all(b.tree.set(k, "modified") == "No date" for k in b.tree.get_children(""))
        assert all(v[0] == "2" and v[1] == "Slide tiles" for v in rows(b).values())
        capture(b, "dates_timeout")
        gate.set()
        until(b, lambda: all(r.mtime is not None for r in b._rows.values()) and
              list(rows(b)) == [f"Export{i}" for i in reversed(range(6))], timeout=3)
        assert "F5" not in b._notes[b._others] and len(batches) == 2
        capture(b, "dates_recovered")
    finally:
        gate.set()


def test_selected_parent_preserves_its_unchecked_descendants(env, tree):
    cloud = tree / "ProjectC" / "CloudMount"
    slide(cloud / "Slide1_Scan1", "Slide1_Scan1")
    table = mounts(f"50 1 0:50 / {cloud} rw - fuse.rclone remote: rw")
    b = env.open((str(tree), None), ctx=env.ctx(table))
    until(b, lambda: settled(b), timeout=3)
    select(b, "ProjectC")
    assert "within 2 levels" not in strip(b)[1] and "weren't checked" not in strip(b)[1]
    assert "wasn't checked" in strip(b)[1] and "not checked" in rows(b)["ProjectC"][1]
    assert str(cloud) not in env.fake.paths("scandir")
    capture(b, "selected_unchecked_parent")


def test_completion_resolves_a_linked_parent_before_listing_it(env, tree):
    share = tree / "labshare"
    (share / "ProjectC").mkdir(parents=True)
    (tree / "Local").mkdir()
    alias = tree / "alias"
    alias.symlink_to(share)
    b = env.open((str(tree / "Local"), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: settled(b), timeout=3)
    b.edit_path(str(alias) + "/")
    key(b._edit, "<Down>")
    until(b, b._lb_visible, timeout=2)
    assert str(alias) not in env.fake.paths("scandir")
    assert b._lb_items == (str(share), ["ProjectC"])


def test_missing_name_below_a_typed_alias_is_highlighted_in_the_resolved_path(env, tree):
    share = tree / "labshare"
    share.mkdir()
    alias = tree / "short"
    alias.symlink_to(share)
    b = env.open((str(tree), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: settled(b), timeout=3)
    b.edit_path(str(alias / "Missing" / "Inner"))
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error", timeout=2)
    assert "Missing" in strip(b)[1] and "labshare" in strip(b)[1]
    assert b._edit.get()[b._edit.index("sel.first"):b._edit.index("sel.last")] == "Missing"
    b._strip_btn.invoke()
    until(b, lambda: b.current() == str(share) and settled(b), timeout=2)


def test_refresh_during_initial_link_resolution_never_lists_the_alias(env, tree, monkeypatch):
    share = tree / "labshare"
    (share / "ProjectC" / "Inner").mkdir(parents=True)
    alias = tree / "alias"
    alias.symlink_to(share)
    monkeypatch.setitem(fsservice.DEADLINE["readlink"], "network", .25)
    gate = env.fake.block(share)
    try:
        b = env.open((str(alias / "ProjectC"), None), ctx=env.ctx(mounts(cifs(share))))
        until(b, lambda: b._ov_kind == "unresponsive", timeout=3)
        key(b.tree, "<F5>")
        pump(b, 100)
        assert not any(p.startswith(str(alias)) for p in env.fake.paths("scandir"))
        gate.set()
        until(b, lambda: b.current() == str(share / "ProjectC") and settled(b), timeout=3)
        assert not b._ov_kind
    finally:
        gate.set()


def test_f5_retries_failed_dates_without_waiting_for_the_old_worker(env, tree, monkeypatch):
    (tree / "Export").mkdir()
    gate = threading.Event()
    real, batches = FsService._stat_names, []

    def stat_names(fs, path, names):
        batches.append(names)
        if len(batches) == 1:
            gate.wait()
        return real(fs, path, names)
    monkeypatch.setattr(FsService, "_stat_names", stat_names)
    monkeypatch.setitem(fsservice.DEADLINE["stat"], "local", .25)
    try:
        b = env.open((str(tree), None))
        until(b, lambda: "F5" in b._notes[b._others], timeout=3)
        key(b.tree, "<F5>")
        until(b, lambda: settled(b) and b._rows[b._names["Export"]].mtime is not None, timeout=2)
        assert len(batches) == 2 and not gate.is_set()
    finally:
        gate.set()


def test_hidden_file_only_folder_has_consistent_badge_and_card(env, tree):
    hidden = tree / "HiddenOnly"
    hidden.mkdir()
    (hidden / ".notes").touch()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b), timeout=3)
    assert rows(b)["HiddenOnly"][1] == "No tiles · only hidden items"
    capture(b, "hidden_only_badge")
    b.navigate(str(hidden))
    until(b, lambda: settled(b), timeout=3)
    assert overlay_text(b) == "Only hidden items here (Ctrl+H shows them)."
    capture(b, "hidden_only_card")


@pytest.mark.parametrize("choose", [False, True])
def test_completion_beyond_ten_matches_uses_the_entry(env, tree, choose):
    for i in range(15):
        (tree / f"Project_{i:02d}").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b), timeout=3)
    b.edit_path(str(tree / "Project_"))
    key(b._edit, "<Tab>")
    for _ in range(11):
        key(b._edit, "<Tab>")
    expected = str(tree / "Project_10")
    assert b._edit.get() == expected
    capture(b, "completion_eleventh_match")
    key(b._edit, "<Control-Return>" if choose else "<Return>")
    if choose:
        until(b, lambda: env.done, timeout=3)
        assert env.done == [expected]
    else:
        until(b, lambda: not b._editing and settled(b), timeout=3)
        assert b.current() == expected


@pytest.mark.parametrize("choose", [False, True])
def test_new_path_text_outlives_an_earlier_successful_lookup(env, tree, monkeypatch, choose):
    for name in ("Archive", "Proteomics"):
        (tree / name).mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b), timeout=3)
    gate, entered = threading.Event(), threading.Event()
    real = browser._locate

    def held(path):
        if path == str(tree / "Archive"):
            entered.set()
            gate.wait()
        return real(path)
    monkeypatch.setattr(browser, "_locate", held)
    try:
        b.edit_path(str(tree / "Archive"))
        key(b._edit, "<Control-Return>" if choose else "<Return>")
        until(b, entered.is_set, timeout=3)
        b._edit.delete(0, "end")
        b._edit.insert(0, str(tree / "Proteomics"))
        # No KeyRelease: an answer can arrive during a held key or immediately after a mouse paste.
        gate.set()
        pump(b, 200)
        assert not env.done and b._editing and b.current() == str(tree)
        assert b._edit.get() == str(tree / "Proteomics") and not b._loading
        capture(b, "newer_path_preserved")
    finally:
        gate.set()


def test_long_place_keeps_current_breadcrumb_visible(env, tree, monkeypatch):
    (tree / "Child_folder").mkdir()
    place = Place("computer", "Research_archive_" + "long_name_" * 6, str(tree), "folder", "local", True, str(tree))
    monkeypatch.setattr(browser, "list_places", lambda *args: [place])
    b = env.open((str(tree / "Child_folder"), None), mode="output")
    until(b, lambda: settled(b) and b._places, timeout=3)
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    pump(b, 100)
    current = b._crumb_ws[-1][1]
    assert current.winfo_ismapped() and "Child" in current.cget("text")
    assert current.winfo_rootx() + current.winfo_reqwidth() <= b._pathbar.winfo_rootx() + b._pathbar.winfo_width()
    iid = next(iter(b._place_items))
    caption = b._side.item(iid, "text")
    assert caption.endswith(b.G["ell"])
    assert b.ctx.theme.fonts["YfsBody"].measure(caption) < b._side.winfo_width() - b.ctx.icons.size
    capture(b, "long_place_current_crumb")


@pytest.mark.parametrize("mode", ["output", "temp"])
def test_narrow_suggestions_fit_and_resize(env, tree, mode):
    name = "Project_with_a_long_and_informative_name"
    (tree / name / "Inner").mkdir(parents=True)
    b = env.open((str(tree), None), mode=mode)
    until(b, lambda: settled(b), timeout=3)
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    pump(b, 100)
    b.edit_path(str(tree) + "/")
    key(b._edit, "<Down>")
    for width in (720, 900):
        b.geometry(f"{b.px(width)}x{b.px(480)}")
        pump(b, 100)
        if not b._lb_visible():
            key(b._edit, "<Down>")
        box = b._lb.bbox("0", "badge")
        text = b._lb.set("0", "badge")
        assert text and box[0] + b._rulers[False].width(text) + b.px(5) <= b._lb.winfo_width()
        assert sum(b._lb.column(c, "width") for c in ("#0", "badge")) <= b._lb.winfo_width()
        capture(b, f"{mode}_suggestions_{width}")


def test_temp_footer_reserves_picker_and_cancel_width(env, tree, monkeypatch):
    name = "WideMicroscopy" * 3
    (tree / name).mkdir()
    monkeypatch.setattr(browser.osutil, "system_picker_available", lambda: True)
    b = env.open((str(tree / name), None), mode="temp")
    until(b, lambda: settled(b) and b._picker.winfo_ismapped(), timeout=3)
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    pump(b, 100)
    capture(b, "temp_footer_long_name")
    assert b._picker.winfo_width() >= b._picker.winfo_reqwidth(), (
        b._primary.cget("text"), b._primary.master.winfo_width(), b._primary.winfo_reqwidth(),
        b._cancel.winfo_reqwidth(), b._picker.winfo_reqwidth())
    assert b._primary.winfo_width() >= b._primary.winfo_reqwidth()
    assert b._primary.winfo_rootx() + b._primary.winfo_width() <= b.winfo_rootx() + b.winfo_width()
