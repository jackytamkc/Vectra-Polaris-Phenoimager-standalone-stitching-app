"""Folder browser under Xvfb with the real FsService, MountTable and Settings on synthetic trees
(FINAL_SPEC §2.2, acceptance C1-C4). Slow and hung folders come from tests/fake_fs.py."""
from __future__ import annotations

import dataclasses
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.fake_fs import FakeOs
from yfs.core import fsservice
from yfs.core.fsservice import FsService
from yfs.core.model import OutputIndex, OutputInfo
from yfs.core.places import MountTable
from yfs.core.settings import Settings
from yfs.ui.browser import FolderBrowser, browser_initial
from yfs.ui.dialogs import NewFolderDialog

ROOT_FS = "1 0 8:1 / / rw - ext4 /dev/sda1 rw"


def mounts(*extra: str) -> MountTable:
    return MountTable.from_mountinfo_text("\n".join((ROOT_FS,) + extra))


def until(root, pred, timeout: float = 10.0, what: str = "condition"):
    """Run the Tk loop until pred() is truthy (returned) or fail."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        value = pred()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def pump(root, ms: int) -> None:
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


@pytest.fixture
def tree(tmp_path) -> Path:
    """An empty folder for a test's own tree (tmp_path also holds the fake home and the settings)."""
    root = tmp_path / "tree"
    root.mkdir()
    return root


@pytest.fixture
def env(ui_ctx, cfg_dir, tmp_path, monkeypatch):
    """An AppContext with a real FsService (FakeOs underneath), a mount table and settings in a temp folder.
    env.table is what MountTable.load() reads, so the browser's reload_mounts() finds the test's table (set it to
    a newer table to 'mount' something)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))                     # hermetic places: no real bookmarks or XDG dirs
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    made = []

    class Env:
        fake = FakeOs()
        out_index = None
        planned = frozenset()
        done = []
        table = mounts()

        def ctx(self, table=None):
            self.table = table or mounts()
            fs = FsService(ui_ctx.bus.post, self.table, os_api=self.fake)
            made.append(fs)
            return dataclasses.replace(ui_ctx, fs=fs, mounts=self.table, settings=Settings.load(),
                                       output_index=lambda: self.out_index, planned_output_keys=lambda: self.planned)

        def open(self, initial, mode="input", ctx=None, **kw):
            ctx = ctx or self.ctx()
            b = FolderBrowser.open(ui_ctx.root, ctx, mode=mode, initial=initial, on_done=self.done.append, **kw)
            return b

    e = Env()
    e.done = []
    monkeypatch.setattr(MountTable, "load", classmethod(lambda cls, timeout_s=1.0: e.table))
    monkeypatch.setattr(FsService, "RELOAD_S", 0.05)
    yield e
    for w in ui_ctx.root.winfo_children():
        w.destroy()
    for fs in made:
        fs.shutdown()


def rows(b) -> dict:
    """name -> (tiles, What's inside, tags) of the rows shown in the listing, in full (a narrow column draws them
    shortened, BR-26: see shown())."""
    out = {}
    for k in b.tree.get_children(""):
        r = b._rows.get(k)
        if r is None:                                         # the "Opening..." row
            continue
        name, tiles, inside, _mod, _bold, _lead = b._full[k]
        out[r.name if r.kind != "tiles" else name] = (tiles, inside, tuple(b.tree.item(k, "tags")))
    return out


def shown(b, name) -> tuple:
    """(name, What's inside) as the listing draws the row named `name`."""
    k = b._names.get(name) or "tiles"
    return str(b.tree.item(k, "text")), str(b.tree.item(k, "values")[1])


def settled(b) -> bool:
    """Listing shown, every probe and roll-up back."""
    return (b._listing is not None and not b._loading and not b._probe_queue and
            all(r.kind != "dir" or r.auto or ((r.summary is not None or r.error is not None) and not r.probing
                                              and (r.roll is None or not r.roll[4])) for r in b._rows.values()))


def select(b, name):
    key = b._names[name]
    b.tree.selection_set(key)
    b.tree.focus(key)
    b.update()


def key(widget, seq, **kw):
    widget.focus_force()
    widget.update()
    widget.event_generate(seq, **kw)
    widget.update()


def strip(b):
    return str(b._strip.cget("style")).split(".")[0], str(b._strip_text.cget("text"))


def q(b, s):
    return f"{b.G['lq']}{s}{b.G['rq']}"


# ------------------------------------------------------------------------------------------------ C1: start folder
def test_opens_next_to_the_field_folder_with_it_selected(env, synth_tree):
    ctx = env.ctx()
    initial = browser_initial("input", str(synth_tree / "ProjectA"), None, ctx.settings)
    assert initial == (str(synth_tree), "ProjectA")
    b = env.open(initial, ctx=ctx)
    assert str(b._primary.cget("text")) == f"Use {q(b, 'ProjectA')}"      # before any listing arrived
    assert b.target() == str(synth_tree / "ProjectA")
    until(b, lambda: b.tree.selection(), what="selection")
    assert b._rows[b.tree.selection()[0]].name == "ProjectA" and b.target() == str(synth_tree / "ProjectA")


def test_missing_start_folder_opens_its_closest_ancestor(env, synth_tree):
    b = env.open((str(synth_tree / "ProjectA" / "gone" / "deeper"), None))
    until(b, lambda: b._listing is not None and b.current() == str(synth_tree / "ProjectA"))
    assert str(b._note.cget("text")) == b.T(f"{q(b, 'deeper')} no longer exists — showing the closest folder that "
                                            f"does.")


def test_cancel_escape_and_window_close_return_none(env, synth_tree, tk_root):
    for how in ("cancel", "escape", "close"):
        env.done.clear()
        b = env.open((str(synth_tree), None))
        until(b, lambda: settled(b))
        if how == "cancel":
            b.cancel()
        elif how == "escape":
            key(b.tree, "<Escape>")
        else:
            b.tk.call(b.protocol("WM_DELETE_WINDOW"))
        assert env.done == [None] and not b.winfo_exists()


# ------------------------------------------------------------------------------------------------ C1: badges
def test_badges_and_rollups_on_the_synthetic_tree(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b), what="probes and roll-ups")
    got = rows(b)
    assert got["ProjectA"][1:] == ("2 slide folders inside", ("slide",))
    assert got["inform_export"] == ("10", "2 slides (inForm batch export)", ("slide",))
    assert got["round1_unmix"][1] == "2 slide folders inside"
    assert got["suspicious"][1] == "1 slide folder inside"
    assert got["Spaces & ünïcödé"][1] == "1 slide folder inside"
    assert got["duplicates"][1:] == ("No tiles · 2 folders", ("dim",))
    assert got["unrelated"][1] == "No tiles · 3 folders"
    assert got["empty_dir"][1:] == ("Empty", ("dim",))
    assert got["locked"][1] == "No access"
    assert ".hidden_dir" not in got
    assert list(got)[:3] == ["duplicates", "empty_dir", "inform_export"]          # natural order, folders first


def test_slide_folder_rows_files_and_the_other_files_count(env, synth_tree):
    env.out_index = OutputIndex("/out", True, frozenset(), {"slidea1_scan1.ome.tif": OutputInfo("x", 1, 1)}, {}, (),
                                None, "/", "local", False)
    b = env.open((str(synth_tree / "ProjectA"), None))
    until(b, lambda: settled(b) and b._rows[b._names["SlideA1_Scan1.ome.tif"]].size is not None, what="stat pass")
    got = rows(b)
    assert got["SlideA1_Scan1"][:2] == ("9", "Slide tiles")
    assert got["SlideA2_Scan1"][:2] == ("4", "Slide tiles")
    assert got["SlideA1_Scan1.ome.tif"][1].startswith("Stitched image · ")
    assert "notes.txt" not in got and str(b._others.cget("text")) == "1 other file not shown"
    b._v_all.set(True)
    b._relayout()
    pump(b, 50)
    assert rows(b)["notes.txt"][2] == ("dim",)


def test_tiles_collapse_into_one_row_that_counts_the_tiles_slides_are_made_of(env, synth_tree):
    b = env.open((str(synth_tree / "inform_export"), None))
    until(b, lambda: settled(b))
    got = rows(b)
    assert got["10 tile files"][1] == b.T("…_[x,y]_component_data.tif · also 4 binary_seg_maps")      # BR-28
    assert got["batch_procedure.ifp"][1] == "inForm batch file" and str(b._others.cget("text")) == \
        "1 other file not shown"
    assert strip(b)[1].startswith("inform_export: 2 slides in one folder (inForm batch export), 10 tiles.")


def test_strip_and_primary_button_follow_the_target(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    until(b, lambda: strip(b) == ("Warn", "synth contains 7+ slide folders (32+ tiles). 1 folder couldn't be read. "
                                          "Press F5 to try again."), what="slide folders and unreadable descendants")
    select(b, "ProjectA")
    assert str(b._primary.cget("text")) == f"Use {q(b, 'ProjectA')}"
    assert strip(b) == ("Ok", "ProjectA contains 2 slide folders (13 tiles). You'll see the full list before "
                              "anything is stitched.")
    select(b, "inform_export")
    assert strip(b) == ("Ok", "inform_export: 2 slides in one folder (inForm batch export), 10 tiles.")
    select(b, "empty_dir")
    assert strip(b) == ("Warn", "empty_dir is empty.")
    key(b.tree, "<Escape>")                                           # Esc clears the selection first
    assert b.tree.selection() == () and b.target() == str(synth_tree) and b.winfo_exists()


def test_the_save_folder_is_flagged_in_input_mode(env, synth_tree):
    env.out_index = OutputIndex(str(synth_tree / "ProjectA"), True, frozenset(), {}, {}, (), None, "/", "local", False)
    b = env.open((str(synth_tree), "ProjectA"))
    until(b, lambda: settled(b) and b.tree.selection())
    assert strip(b) == ("Warn", "This is your save folder. Stitched images are never used as tiles.")


def test_long_names_are_shortened_in_the_middle(env, tree):
    name = "A" * 30 + "_middle_" + "Z" * 30
    (tree / name).mkdir()
    b = env.open((str(tree), name))
    text = str(b._primary.cget("text"))
    assert b.G["ell"] in text and len(text) <= len("Use ''") + 40 + 2 and text.endswith("Z" + b.G["rq"])


def test_a_link_to_a_folder_is_followed_and_opens_the_real_folder(env, synth_tree):
    """UD3: the link is read (readlink, in the lane of the folder holding it), probed where it leads, shown under
    its own name, and opened as the real folder: the crumbs show the real path."""
    b = env.open((str(synth_tree / "unrelated"), None))
    until(b, lambda: "loop_to_root" in rows(b), what="the link is followed to a folder")
    r = b._rows[b._names["loop_to_root"]]
    assert r.kind == "dir" and r.target == str(synth_tree)
    badge = "1+ slide folders inside · 1 folder couldn't be read · link"
    until(b, lambda: rows(b)["loop_to_root"][1] == badge, what="its probe and partial roll-up")
    assert b._cut[r.key] == b.T(f"loop_to_root → {synth_tree}\n{badge}")    # where it leads, and what is unknown
    assert ("readlink", str(synth_tree / "unrelated" / "loop_to_root")) in env.fake.calls
    assert str(synth_tree / "unrelated" / "loop_to_root") not in env.fake.paths("scandir")    # never through it
    select(b, "loop_to_root")
    assert b.target() == str(synth_tree) and str(b._primary.cget("text")) == f"Use {q(b, 'loop_to_root')}"
    key(b.tree, "<Return>")
    until(b, lambda: b.current() == str(synth_tree) and b._listing is not None, what="the real folder")
    assert b._crumb_ws[-1][2] == "synth"


def test_drive_and_share_roots_are_named_by_their_place(env, tree):
    hpc = tree / "hpc"
    hpc.mkdir()
    b = env.open((str(tree), "hpc"), ctx=env.ctx(mounts(f"2 1 0:42 / {hpc} rw - fuse.sshfs user@hpc.example.org:/ rw")))
    until(b, lambda: b._place_items and b.tree.selection())
    assert str(b._primary.cget("text")) == f"Use {q(b, 'hpc (ssh)')}"


def test_layout_follows_the_window_width_and_is_remembered(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b.geometry(f"{b.px(1080)}x{b.px(680)}")
    pump(b, 200)
    assert b._preview_on and str(b._preview) in b._pw.panes() and "modified" in b.tree.cget("displaycolumns")
    b.geometry(f"{b.px(720)}x{b.px(560)}")
    pump(b, 300)
    assert not b._preview_on and str(b._preview) not in b._pw.panes()
    assert "modified" not in b.tree.cget("displaycolumns") and "inside" in b.tree.cget("displaycolumns")
    b._pw.sashpos(0, b.px(180))
    pump(b, 50)
    b.cancel()
    s = b.ctx.settings
    assert s.get("browser.geometry") == f"{round(b.px(720) / b.ctx.theme.scale)}x{round(b.px(560) / b.ctx.theme.scale)}"
    assert abs(s.get("browser.sash") - 180) <= 2


def test_refresh_and_ctrl_f(env, tree):
    (tree / "a").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    (tree / "b_new").mkdir()
    key(b.tree, "<F5>")
    until(b, lambda: "b_new" in rows(b), what="refreshed listing")
    key(b.tree, "<Control-f>")
    assert str(b.focus_get()) == str(b._filter)


# ------------------------------------------------------------------------------------------------ C1: list tools
def test_filter_substring_glob_and_escape(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b._filter_var.set("PROJ")
    until(b, lambda: list(rows(b)) == ["ProjectA"], what="substring filter")
    b._filter_var.set("*_dir")
    until(b, lambda: list(rows(b)) == ["empty_dir"], what="glob filter")
    key(b.tree, "<Escape>")
    until(b, lambda: len(rows(b)) > 5 and b._filter_var.get() == "")


def test_only_folders_with_tiles_and_hidden_items(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b._v_tiles.set(True)
    b._relayout()
    pump(b, 30)
    assert set(rows(b)) == {"inform_export", "ProjectA", "round1_unmix", "suspicious", "Spaces & ünïcödé", "locked"}
    b._v_tiles.set(False)
    key(b.tree, "<Control-h>")
    assert "hidden" in rows(b)[".hidden_dir"][2]
    b.cancel()
    assert b.ctx.settings.save_now(2) and Settings.load().get("browser.show_hidden") is True


def test_type_ahead_and_slash_starts_path_editing(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    for ch in "su":
        key(b.tree, "<KeyPress>", keysym=ch)
    assert b._rows[b.tree.selection()[0]].name == "suspicious" and str(b._note.cget("text")) == "su"
    pump(b, 1100)
    key(b.tree, "<KeyPress>", keysym="r")
    assert b._rows[b.tree.selection()[0]].name == "round1_unmix"          # the buffer reset after 1 s
    pump(b, 1100)
    key(b.tree, "<KeyPress>", keysym="slash")
    assert b._editing and b._edit_var.get() == "/"


def test_sorting_keeps_folders_first_and_is_saved(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b._sort_by("tiles")
    pump(b, 30)
    assert list(rows(b))[0] == "inform_export"
    b._sort_by("inside")
    pump(b, 30)
    names = list(rows(b))
    assert names[0] == "inform_export" and names[-1] == "locked"             # slide rows first, errors last
    b._sort_by("name")
    b._sort_by("name")
    pump(b, 30)
    assert list(rows(b))[0] == "unrelated"
    b.cancel()
    assert b.ctx.settings.save_now(2) and Settings.load().get("browser.sort") == ["name", "desc"]


def test_open_up_back_forward_and_ctrl_enter(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    select(b, "ProjectA")
    key(b.tree, "<Return>")                                                # Enter opens, never chooses
    until(b, lambda: b.current() == str(synth_tree / "ProjectA") and settled(b))
    assert env.done == []
    key(b.tree, "<BackSpace>")
    until(b, lambda: b.current() == str(synth_tree) and b.tree.selection())
    assert b._rows[b.tree.selection()[0]].name == "ProjectA"               # up selects where we came from
    b.back()
    until(b, lambda: b.current() == str(synth_tree / "ProjectA"))
    b.forward()
    until(b, lambda: b.current() == str(synth_tree) and settled(b))
    select(b, "suspicious")
    key(b.tree, "<Control-Return>")
    assert env.done == [str(synth_tree / "suspicious")] and not b.winfo_exists()
    assert b.ctx.settings.save_now(2) and Settings.load().get("paths.browser_last.input") == str(synth_tree)


def test_escape_order(env, synth_tree):
    """Esc: leave path edit, then clear the filter, then clear the selection, then Cancel (§2.2.10)."""
    b = env.open((str(synth_tree), "ProjectA"))
    until(b, lambda: settled(b) and b.tree.selection())
    b._filter_var.set("Proj")
    pump(b, 300)
    b.edit_path()
    steps = []
    for _ in range(4):
        key(b._edit if b._editing else b.tree, "<Escape>")
        open_ = b.winfo_exists()                                          # (a closed browser has no Tk variables)
        steps.append((b._editing, b._filter_var.get() if open_ else "", bool(open_ and b.tree.selection()),
                      env.done[:]))
    assert steps == [(False, "Proj", True, []), (False, "", True, []), (False, "", False, []),
                     (False, "", False, [None])]


# ------------------------------------------------------------------------------------------------ C1: path edit
def test_path_edit_maps_unc_paths_through_the_mount_table(env, tree):
    share = tree / "share"
    (share / "ProjectC").mkdir(parents=True)
    table = mounts(f"2 1 0:40 / {share} rw - cifs //fileserver.example.ac.uk/labshare rw")
    b = env.open((str(tree), None), ctx=env.ctx(table))
    until(b, lambda: settled(b))
    b.edit_path()
    assert b._editing and b._edit.select_present()
    b._edit_var.set("\\\\FILESERVER.example.ac.uk\\labshare\\ProjectC")
    key(b._edit, "<Return>")
    until(b, lambda: b.current() == str(share / "ProjectC"), what="UNC navigation")
    assert not b._editing


def test_breadcrumbs_collapse_into_the_menu_and_clicking_the_bar_edits(env, tree):
    deep = tree.joinpath(*[f"a_rather_long_folder_name_{i}" for i in range(8)])
    deep.mkdir(parents=True)
    b = env.open((str(deep), None))
    until(b, lambda: b._place_items and b._listing is not None)
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    pump(b, 300)
    shown = [w for _s, w, _l, _p in b._crumb_ws if w.winfo_manager()]
    assert len(shown) < len(b._crumb_ws) and b._more.winfo_manager() == "pack"
    assert shown[-1] is b._crumb_ws[-1][1]                                  # the current folder stays visible
    assert str(shown[-1].cget("style")) == "CrumbCurrent.TButton"
    labels = [b._more_menu.entrycget(i, "label") for i in range(b._more_menu.index("end") + 1)]
    assert "a_rather_long_folder_name_0" in labels
    b._more_menu.invoke(labels.index("a_rather_long_folder_name_0"))
    until(b, lambda: b.current() == str(tree / "a_rather_long_folder_name_0"))
    b._pathbar.event_generate("<Button-1>", x=b._pathbar.winfo_width() - 5, y=5)
    assert b._editing and b._edit_var.get() == b.current()


def test_path_edit_file_missing_and_bare_server(env, synth_tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b.edit_path(str(synth_tree / "ProjectA" / "notes.txt"))
    key(b._edit, "<Return>")
    until(b, lambda: b.current() == str(synth_tree / "ProjectA") and b._listing is not None)
    b._v_all.set(True)
    b.edit_path(str(synth_tree / "ProjectA" / "nope" / "x"))
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error")
    assert strip(b)[1] == f"Can't find {q(b, 'nope')} in {q(b, 'ProjectA')}."             # BR-16: names, once
    assert str(b._strip_btn.cget("text")) == "Go there" and b._editing
    b.edit_path("\\\\server")
    key(b._edit, "<Return>")
    assert strip(b) == ("Warn", "Type the full share path, for example \\\\server\\share.") and b._editing
    key(b._edit, "<Escape>")
    assert not b._editing and b.winfo_exists()                              # Esc leaves path edit first


def test_tab_completes_the_common_prefix_then_cycles(env, tree):
    for name in ("alpha1", "alpha2", "alphabet", "beta"):
        (tree / name).mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    b.edit_path(str(tree / "AL"))
    key(b._edit, "<Tab>")                                                  # the parent listing is cached: at once
    assert b._edit_var.get() == str(tree / "alpha")
    seen = []
    for _ in range(3):
        key(b._edit, "<Tab>")
        seen.append(b._edit_var.get())
    assert seen == [str(tree / n) for n in ("alpha1", "alpha2", "alphabet")]
    assert str(b.focus_get()) == str(b._edit)                              # Tab never leaves the Entry
    b.edit_path(str(tree / "be"))
    key(b._edit, "<Tab>")
    assert b._edit_var.get() == str(tree / "beta") + os.sep


def test_tab_completion_lists_an_uncached_parent_first(env, synth_tree):
    b = env.open((str(synth_tree / "ProjectA"), None))
    until(b, lambda: settled(b))
    b.edit_path(str(synth_tree / "duplicates" / "batch"))
    key(b._edit, "<Tab>")
    until(b, lambda: b._edit_var.get() == str(synth_tree / "duplicates" / "batch"), what="no change yet")
    key(b._edit, "<Tab>")
    until(b, lambda: b._edit_var.get() == str(synth_tree / "duplicates" / "batchA"), what="cycle after listing")


def test_autocomplete_listbox_inside_the_dialog(env, tree):
    for i in range(14):
        (tree / f"run{i:02d}").mkdir()
    (tree / "other").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    b.edit_path(str(tree / "run"))
    key(b._edit, "<Down>")                                                 # opens the list; nothing chosen yet
    lb = b._lb
    assert b._lb_visible() and lb.winfo_toplevel() is b and len(lb.get_children()) == 10
    assert lb.item("0", "text") == "run00" and lb.set("0", "badge") == "Empty"   # the cached badge, its own column
    assert lb.selection() == ()
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")
    assert lb.selection() == ("1",) and str(b.focus_get()) == str(b._edit)
    key(b._edit, "<Up>")
    key(b._edit, "<Escape>")
    assert not b._lb_visible() and b._editing                              # Esc closes the list first
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")
    key(b._edit, "<Return>")                                               # Enter accepts the suggestion
    until(b, lambda: b.current() == str(tree / "run01"))


# ------------------------------------------------------------------------------------------------ sidebar
def test_places_health_and_recent(env, tree):
    share = tree / "share"
    (share / "P").mkdir(parents=True)
    table = mounts(f"2 1 0:40 / {share} rw - cifs //fileserver.example.ac.uk/labshare rw",
                   f"3 1 0:41 / {tree / 'cloud'} rw - fuse.rclone cloud: rw")
    ctx = env.ctx(table)
    ctx.settings.add_recent("input", str(share / "P"), slides=6)
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: b._place_items, what="places")
    labels = {b._side.item(i, "text").strip(): p for i, (p, _l) in b._place_items.items()}
    assert {"P · 6 slides", "Home", "share", "cloud"} <= set(labels)          # no "(cloud)" under CLOUD
    sections = [b._side.item(i, "text") for i in b._side.get_children("") if i not in b._place_items]
    assert sections[:2] == ["RECENT", "THIS COMPUTER"] and "NETWORK" in sections and "CLOUD" in sections
    ctx.fs.mark_disconnected(ctx.fs.lane_key(str(share)), True)          # a real lane state change (no I/O)
    until(b, lambda: "share (disconnected)" in [b._side.item(i, "text").strip() for i in b._place_items])
    recent = next(i for i, (p, _l) in b._place_items.items() if p.section == "recent")
    assert b._side.item(recent, "tags") == ("dim",)                        # Recent: dimmed, no suffix
    b._forget_recent(str(share / "P"))
    assert not ctx.settings.get("recent.inputs")
    net = next(i for i, (p, _l) in b._place_items.items() if p.label == "share")
    b._open_place(net)
    until(b, lambda: b.current() == str(share))


def test_children_behind_mount_boundaries_are_never_probed(env, tree):
    for d in ("cloud", "plain"):
        (tree / d / "inner").mkdir(parents=True)
    table = mounts(f"2 1 0:41 / {tree / 'cloud'} rw - fuse.rclone remote: rw")
    b = env.open((str(tree), None), ctx=env.ctx(table))
    until(b, lambda: settled(b))
    assert rows(b)["cloud"][1] == b.T("Cloud folder — open to look inside")
    assert str(tree / "cloud") not in env.fake.paths("scandir")
    assert str(tree / "plain") in env.fake.paths("scandir")
    b.navigate(str(tree / "cloud"))                                     # an explicit click still opens it
    until(b, lambda: b._listing is not None and "inner" in rows(b))


def test_a_large_folder_is_inserted_in_batches(env, tree):
    for i in range(1200):
        (tree / f"d{i:04d}").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b), timeout=30)
    b._render(b._listing)                                                   # again, without running the loop
    assert len(b.tree.get_children("")) == 500                              # the first batch only
    until(b, lambda: len(b.tree.get_children("")) == 1200, what="all rows")
    assert b._probe_queue == [] or len(b._probe_queue) <= 1200 - 300       # probes go out 300 per turn


# ------------------------------------------------------------------------------------------------ C2: states
@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setitem(fsservice.DEADLINE["list_browser"], "local", 4)


def overlay_text(b) -> str:
    texts = [str(w.cget("text")) for w in b._ov.winfo_children() if "Label" in w.winfo_class()]
    return " ".join(t for t in texts if t) if b._ov.winfo_manager() else ""


def overlay_buttons(b):
    return [str(btn.cget("text")) for f in b._ov.winfo_children() for btn in f.winfo_children()
            if btn.winfo_class() == "TButton"]


def test_slow_then_not_answering_then_a_late_result(env, tree, short_deadline):
    hung = tree / "hung"
    (hung / "inside").mkdir(parents=True)
    (tree / "start").mkdir()
    gate = env.fake.block(hung)
    try:
        b = env.open((str(tree / "start"), None))
        until(b, lambda: settled(b))
        t0 = time.monotonic()
        b.navigate(str(hung))
        assert rows(b) == {} and b.tree.item("__opening", "text") == b.T(f"Opening {q(b, 'hung')}…")
        until(b, lambda: b._bar.winfo_manager() == "place", timeout=2, what="loading bar")
        assert time.monotonic() - t0 >= 0.29
        until(b, lambda: b._ov_kind == "wait", timeout=4, what="slow state")
        assert 2.4 <= time.monotonic() - t0 < 3.6                            # FsSlow after 2.5 s
        assert "to answer" in overlay_text(b) and overlay_buttons(b) == ["Stop waiting"]
        assert "Disks can take a moment to wake up." in overlay_text(b)      # a local disk: not a network drive
        until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="not answering")
        assert overlay_buttons(b) == ["Keep waiting", "Try again", "Go back"]
        assert b.T("isn't answering. Nothing else in the app is affected. Disks sometimes need a minute to wake "
                   "up.") in overlay_text(b)
        assert b._primary.instate(["!disabled"]) and str(b._primary.cget("text")) == f"Use {q(b, 'hung')}"
        gate.set()                                                           # the folder answers late
        until(b, lambda: b._listing is not None and "inside" in rows(b), what="late result")
        assert str(b._note.cget("text")).startswith("Answered after ")
        assert not b._ov.winfo_manager() and not b._bar.winfo_manager()
    finally:
        gate.set()


def test_try_again_sends_exactly_one_request(env, tree, short_deadline):
    hung = tree / "hung"
    hung.mkdir()
    (tree / "start").mkdir()
    gate = env.fake.block(hung)
    try:
        b = env.open((str(tree / "start"), None))
        until(b, lambda: settled(b))
        b.navigate(str(hung))
        until(b, lambda: b._ov_kind == "unresponsive", timeout=8, what="not answering")
        before = env.fake.paths("scandir").count(str(hung))
        retry = next(w for f in b._ov.winfo_children() for w in f.winfo_children()
                     if w.winfo_class() == "TButton" and str(w.cget("text")) == "Try again")
        retry.invoke()
        pump(b, 400)
        assert env.fake.paths("scandir").count(str(hung)) == before + 1
        b.navigate(str(hung))                                                 # fail-fast again: no new call
        until(b, lambda: b._ov_kind == "error", what="fail fast")
        assert env.fake.paths("scandir").count(str(hung)) == before + 1
        assert "isn't answering" in overlay_text(b) and overlay_buttons(b) == ["Try again", "Go back"]
    finally:
        gate.set()


def test_keep_waiting_and_go_back(env, tree, short_deadline):
    hung = tree / "hung"
    hung.mkdir()
    (tree / "start").mkdir()
    gate = env.fake.block(hung)
    try:
        b = env.open((str(tree / "start"), None))
        until(b, lambda: settled(b))
        b.navigate(str(hung))
        until(b, lambda: b._ov_kind == "unresponsive", timeout=8)
        b._keep()
        assert not b._ov.winfo_manager() and b._bar.winfo_manager() == "place"
        b._go_back()
        until(b, lambda: b.current() == str(tree / "start") and b._listing is not None)
    finally:
        gate.set()


def test_no_access_not_found_and_empty_states(env, synth_tree, tree):
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b.navigate(str(synth_tree / "locked"))
    until(b, lambda: b._ov_kind == "error")
    assert f"You don't have permission to open {q(b, 'locked')}." in overlay_text(b)
    assert overlay_buttons(b) == ["Go back"]
    b.navigate(str(synth_tree / "vanished"))
    until(b, lambda: b._ov_kind == "error" and "exist" in overlay_text(b))
    assert f"{q(b, 'vanished')} doesn't exist any more." in overlay_text(b)
    b.navigate(str(synth_tree / "empty_dir"))
    until(b, lambda: b._ov_kind == "empty")
    assert overlay_text(b) == "This folder is empty."
    (tree / "only_files").mkdir()
    (tree / "only_files" / "readme.txt").write_text("x")
    b.navigate(str(tree / "only_files"))
    until(b, lambda: b._ov_kind == "empty")
    assert overlay_text(b) == f"No folders here. Tiles usually sit in a folder named like {q(b, '<slide>_Scan1')}."


def test_not_found_from_recent_offers_remove(env, tree):
    ctx = env.ctx()
    ctx.settings.add_recent("input", str(tree / "gone"), slides=2)
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: b._place_items)
    b.navigate(str(tree / "gone"))
    until(b, lambda: b._ov_kind == "error")
    assert overlay_buttons(b) == ["Go back", "Remove from recent"]


def test_initial_permission_error_does_not_still_claim_to_be_opening(env, tree):
    locked = tree / "locked"
    locked.mkdir(exist_ok=True)
    env.fake.fail(locked, PermissionError(13, "Permission denied"))
    b = env.open((str(locked), None))
    until(b, lambda: b._ov_kind == "error")
    assert not b._loading
    assert not any("Opening" in str(b.tree.item(i, "text")) for i in b.tree.get_children())
    assert "permission" in overlay_text(b)


def test_the_tk_thread_never_touches_user_paths(env, synth_tree, monkeypatch, tree):
    """C2: an audit hook (open, os.scandir, os.listdir, ...) plus os.stat/os.lstat wrappers record any access to the
    test tree from the main thread while a whole browsing session runs."""
    hits = []
    guard = str(synth_tree)
    real_stat, real_lstat = os.stat, os.lstat

    def watch(name, fn):
        def wrapped(path, *a, **kw):
            if threading.current_thread() is threading.main_thread() and str(path).startswith(guard):
                hits.append((name, str(path)))
            return fn(path, *a, **kw)
        return wrapped
    monkeypatch.setattr(os, "stat", watch("os.stat", real_stat))
    monkeypatch.setattr(os, "lstat", watch("os.lstat", real_lstat))
    _AUDIT["guard"], _AUDIT["hits"] = guard, hits
    try:
        b = env.open((str(synth_tree), "ProjectA"))
        until(b, lambda: settled(b))
        key(b.tree, "<Return>")
        until(b, lambda: b.current() == str(synth_tree / "ProjectA") and settled(b))
        pump(b, 300)                                                        # stat pass
        b._filter_var.set("Slide")
        b.edit_path(str(synth_tree / "rou"))
        key(b._edit, "<Tab>")
        key(b._edit, "<Down>")
        b.edit_path(str(synth_tree / "unrelated" / "WSIs"))
        key(b._edit, "<Return>")
        until(b, lambda: b.current() == str(synth_tree / "unrelated" / "WSIs") and settled(b))
        b._v_all.set(True)
        b._relayout()
        pump(b, 300)
        b.refresh()
        until(b, lambda: settled(b))
        b.up()
        until(b, lambda: settled(b) and b.current() == str(synth_tree / "unrelated"))
        b.choose()
    finally:
        _AUDIT["guard"] = None
    assert env.done == [str(synth_tree / "unrelated" / "WSIs")]
    assert hits == []


_AUDIT = {"guard": None, "hits": []}


def _audit(event, args):
    guard = _AUDIT["guard"]
    if guard and event in ("open", "os.scandir", "os.listdir", "os.mkdir", "os.remove", "os.rename") and args \
            and threading.current_thread() is threading.main_thread():
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(guard):
            _AUDIT["hits"].append((event, os.fsdecode(path)))


sys.addaudithook(_audit)                      # audit hooks can't be removed: it is inert while guard is None


# ------------------------------------------------------------------------------------------------ C3: output mode
def test_output_mode_hides_tiles_and_dims_tile_folders(env, synth_tree):
    b = env.open((str(synth_tree / "ProjectA"), "SlideA1_Scan1"), mode="output")
    assert str(b.title()) == "Choose where to save the stitched images"
    assert str(b._primary.cget("text")) == f"Save here: {q(b, 'SlideA1_Scan1')}"
    until(b, lambda: settled(b) and b.tree.selection())
    assert "tiles" not in b.tree.cget("displaycolumns")
    assert rows(b)["SlideA1_Scan1"][1:] == (b.T("Slide tiles — don't save here"), ("dim",))
    until(b, lambda: "Free space" in strip(b)[1], what="free space")
    assert strip(b)[0] == "Warn" and "This folder holds tiles" in strip(b)[1]
    assert str(b._strip_btn.cget("text")) == f"Create {q(b, 'stitched')} here"


def test_create_stitched_here(env, tree):
    slide = tree / "S1_Scan1"
    slide.mkdir()
    for i in range(3):
        (slide / f"S1_Scan1_[{i * 900},100]_component_data.tif").write_bytes(b"")
    b = env.open((str(tree), "S1_Scan1"), mode="output")
    until(b, lambda: settled(b) and b._strip_btn.winfo_manager())
    b._strip_btn.invoke()
    until(b, lambda: b.current() == str(slide) and b.tree.selection())
    assert (slide / "stitched").is_dir() and b.target() == str(slide / "stitched")


def test_new_folder_is_created_and_selected(env, tree):
    (tree / "Existing").mkdir()
    b = env.open((str(tree), None), mode="output")
    until(b, lambda: settled(b))
    key(b.tree, "<Control-N>")
    dlg = until(b, lambda: next((w for w in b.winfo_children() if isinstance(w, NewFolderDialog)), None))
    assert dlg.var.get() == "stitched"
    dlg.var.set("existing")
    dlg._ok()
    assert dlg.winfo_exists() and "already exists" in str(dlg.error.cget("text"))
    dlg.var.set("Stitched 2026")
    dlg._ok()
    until(b, lambda: b.tree.selection() and b._rows[b.tree.selection()[0]].name == "Stitched 2026")
    assert (tree / "Stitched 2026").is_dir() and b.target() == str(tree / "Stitched 2026")


def test_save_here_checks_that_the_folder_is_writable(env, tree):
    ro, ok = tree / "readonly", tree / "ok"
    ro.mkdir()
    ok.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(str(ro), os.W_OK):
            pytest.skip("running as a user that ignores permissions")
        b = env.open((str(tree), "readonly"), mode="output")
        until(b, lambda: settled(b) and b.tree.selection())
        b.choose()
        until(b, lambda: strip(b)[0] == "Error", what="write-probe failure")
        assert strip(b) == ("Error", "Can't save here: Permission denied.") and b.winfo_exists() and env.done == []
        select(b, "ok")
        b.choose()
        until(b, lambda: env.done, what="saved")
        assert env.done == [str(ok)] and not list(ok.iterdir())             # the probe file was removed
    finally:
        ro.chmod(stat.S_IRWXU)


# ------------------------------------------------------------------------------------------------ C4: system picker
@pytest.fixture
def zenity(tmp_path, monkeypatch):
    """A stub zenity on PATH that prints the folder named in $FAKE_PICK."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "zenity"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$FAKE_PICK"\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    return bindir


@pytest.mark.skipif(sys.platform != "linux", reason="zenity is the Linux system picker")
def test_zenity_result_is_shown_selected_not_taken_blindly(env, synth_tree, zenity, monkeypatch):
    monkeypatch.setenv("FAKE_PICK", str(synth_tree / "round1_unmix" / "R1-02"))
    b = env.open((str(synth_tree), None))
    until(b, lambda: b._picker.winfo_manager() == "grid", what="picker link")
    b._picker.invoke()
    until(b, lambda: b.current() == str(synth_tree / "round1_unmix") and b.tree.selection(), what="picked")
    assert b._rows[b.tree.selection()[0]].name == "R1-02" and env.done == []
    assert str(b._primary.cget("text")) == f"Use {q(b, 'R1-02')}"
    key(b._primary, "<Return>")                                              # Enter confirms
    assert env.done == [str(synth_tree / "round1_unmix" / "R1-02")]


@pytest.mark.skipif(sys.platform != "linux", reason="zenity is the Linux system picker")
def test_picker_link_hidden_without_zenity(env, synth_tree, tree, monkeypatch):
    empty = tree / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    b = env.open((str(synth_tree), None))
    until(b, lambda: b._place_items, what="places and picker check")
    assert b._picker.winfo_manager() == "" and b._picker_ok is False


def test_picker_link_hidden_during_a_run(env, synth_tree, zenity):
    ctx = dataclasses.replace(env.ctx(), running=lambda: True)
    b = env.open((str(synth_tree), None), ctx=ctx)
    until(b, lambda: b._place_items)
    pump(b, 100)
    assert b._picker.winfo_manager() == ""
