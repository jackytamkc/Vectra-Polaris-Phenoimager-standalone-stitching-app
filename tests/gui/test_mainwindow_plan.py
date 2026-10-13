"""The main window's searches and its plan (UD5, MW-1, MW-8, MW-20, MW-21, PL-7, BR-8): a rescan the user asks for
lists every folder again, a search that missed folders says so, the run bar follows every index, a UI tick's
candidates cost one recompute, and an existing file is checked whenever a row has no verdict on it."""
from __future__ import annotations

from test_mainwindow import (_no_stale_tk_garbage, buttons, dialog, make_slides, make_window, old_output,  # noqa: F401
                             pump, ready, row_of, status, until)
from tests.fakes import FakeEngine
from yfs.core import osutil
from yfs.core.names import DEEPER_STEP
from yfs.core.plan import Plan, row_text
from yfs.ui import app, dialogs


# ------------------------------------------------------------------------------------------------ UD5: searches
def test_a_rescan_right_after_a_new_slide_folder_finds_it(make_window, tmp_path, synth_tree):
    """FS-3: F5 lists every folder again; the listings of the last search would hide a slide added since."""
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=tiles)
    root = win.root
    until(root, lambda: ready(win), what="ready")
    make_slides(tiles, ["Slide2_Scan1"], synth_tree)
    old = win.plan()
    root.event_generate("<F5>")
    until(root, lambda: ready(win) and win.plan() is not old, what="rescanned")
    assert row_of(win, "Slide2_Scan1") is not None


def test_searches_the_user_asks_for_list_again_and_search_deeper_adds_levels(make_window, synth_tree, monkeypatch):
    """UD5: the startup scan and F5 reuse no listing; choosing a folder and the note actions reuse the last two
    minutes' (the browser's); Search deeper passes the levels it adds."""
    seen = []

    class Recording(app.Discovery):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            seen.append(self.opts)
    monkeypatch.setattr(app, "Discovery", Recording)
    win = make_window(input=synth_tree / "ProjectA")                       # the startup scan
    root = win.root
    until(root, lambda: ready(win) and len(seen) == 1, what="startup scan")
    win.set_input(str(synth_tree / "round1_unmix"))                       # chosen in the browser
    until(root, lambda: ready(win) and len(seen) == 2, what="chosen")
    win.rescan()                                                          # F5 / [Rescan]
    until(root, lambda: ready(win) and len(seen) == 3, what="rescan")
    win._rescan(extra=DEEPER_STEP)                                        # [Search deeper]
    until(root, lambda: ready(win) and len(seen) == 4, what="deeper")
    assert [(o.cache_age, o.extra_depth, o.max_depth) for o in seen] == [(0, 0, 3), (120.0, 0, 3), (0, 0, 3),
                                                                         (120.0, 3, 6)]


def test_show_opens_not_included_at_the_folders_that_couldnt_be_searched(make_window, synth_tree):
    """FS-6: a search that couldn't list a folder (synth/locked) is a warning with [Show], and a warning before
    Start (PlanEnv.unsearched)."""
    win = make_window(input=synth_tree)
    root = win.root
    until(root, lambda: ready(win), what="ready")
    assert status(win.input_field) == "Found 10 slides · 1 folder couldn't be searched"
    assert str(win.input_field._text.cget("style")) == "Surface.Warn.TLabel"
    assert any(i.code == "unsearched" and i.severity == "warn" for i in win._issues())
    root.focus_force()
    win.input_field._actions[buttons(win.input_field).index("Show")].invoke()
    pump(root, 100)
    tree, table = win.table.tree, win.table
    locked = [r.row_id for r in win.plan().rows("note") if r.note.kind == "no_access"]
    assert locked and [table._rid[i] for i in tree.selection()] == locked
    assert tree.item("sec:excluded", "open") and root.focus_get() is tree and tree.bbox(tree.selection()[0])


def test_search_deeper_on_a_cloud_folder_reports_the_depth_searched(make_window, tmp_path, synth_tree):
    """PL-16: a cloud folder is searched one level deep; Search deeper really searches deeper, and the status
    says how deep it searched."""
    cloud = tmp_path / "cloud"
    make_slides(cloud / "Vectra" / "Batch1", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=cloud / "Vectra", mounts=[f"3 1 0:61 / {cloud} rw - fuse.rclone remote: rw"])
    root = win.root
    until(root, lambda: ready(win), what="ready")
    assert status(win.input_field) == "No slides found within 1 level.\nCloud folder: files download as they " \
                                      "are read. Searching one level only."
    win.input_field._actions[buttons(win.input_field).index("Search deeper")].invoke()
    until(root, lambda: ready(win) and row_of(win, "Slide1_Scan1") is not None, what="searched deeper")
    lines = status(win.input_field).split("\n")
    assert lines[0].startswith("1 slide found (12 tiles)") and lines[1].endswith("Searching 4 levels deep.")


def test_search_actions_during_a_run_change_nothing(make_window, tmp_path, synth_tree, monkeypatch):
    """MW-21: Search deeper / these folders too during a run would silently change the next search; nor is a
    header read again then."""
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1", "Slide2_Scan1"], synth_tree)
    win = make_window(FakeEngine.configure(durations={"stitch": 5}, steps={"stitch": 50}), tiles, tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.start(confirm=False)
    until(root, lambda: win.state == "RUNNING", what="running")
    reads = []
    monkeypatch.setattr(win, "_read_header", lambda *a, **kw: reads.append(a))
    win._note_action("search_deeper", "note|not_searched_depth")
    win._note_action("search_wide", "note|wide_pruned|x")
    win._note_action("retry_header", row_of(win, "Slide2_Scan1").row_id)
    assert win.state == "RUNNING" and win._extra_depth == 0 and win._no_prune == set() and reads == []


# ------------------------------------------------------------------------------------------------ the plan
def test_an_index_that_changes_no_row_still_brings_the_stitch_button_back(make_window, tmp_path, synth_tree):
    """MW-1: after [Delete them] the new index changes no row; the run bar must not stay on 'Checking the save
    folder…' with Stitch disabled."""
    out = tmp_path / "out"
    raw = old_output(out / "Old_Scan1.ome.tmp.raw", b"x" * 100)
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1"], synth_tree)
    win = make_window(input=tiles, output=out)
    root = win.root
    until(root, lambda: ready(win) and "leftover" in status(win.output_field), what="ready")
    assert win.runbar.button.instate(["!disabled"])
    win.output_field._actions[0].invoke()
    until(root, lambda: dialog(root, dialogs._Dialog), what="confirmation")._close(True)
    until(root, lambda: not raw.exists() and not win._index_pending, what="deleted and indexed")
    pump(root, 200)
    assert win.runbar.button.instate(["!disabled"]) and "Checking the save folder" not in \
        str(win.runbar.issue_text.cget("text"))


def test_the_candidates_of_one_ui_tick_cost_one_recompute(make_window, tmp_path, synth_tree, monkeypatch):
    """MW-8, FS-16: candidates arrive many per UI tick; each tick adds them in one plan.batch()."""
    names = [f"S{i:03d}_Scan1" for i in range(120)]
    tiles = make_slides(tmp_path / "Big", names, synth_tree, tiles=1)
    count, real = [0], Plan._recompute
    monkeypatch.setattr(Plan, "_recompute", lambda plan: count.__setitem__(0, count[0] + 1) or real(plan))
    win = make_window(input=tiles)
    until(win.root, lambda: ready(win), 30, what="ready")
    assert len(win._slides()) == len(names) and count[0] < len(names) // 2, count[0]


def test_an_output_renamed_onto_an_existing_file_has_that_file_checked(make_window, tmp_path, synth_tree):
    """MW-20: the new name matches a file of the save folder; it is checked like any existing output, and the
    row says the name is taken."""
    out = tmp_path / "out"
    old_output(out / "Old_Result.ome.tif")
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1", "Slide2_Scan1"], synth_tree)
    win = make_window(input=tiles, output=out)
    root = win.root
    until(root, lambda: ready(win), what="ready")
    rid = row_of(win, "Slide2_Scan1").row_id
    assert win.plan().rename_output(rid, "Old_Result") is None
    win._plan_edited()                                                    # what the table calls after an edit
    until(root, lambda: win.plan().row(rid).existing_verdict == "valid", 5, what="checked")
    r = win.plan().row(rid)
    assert (r.section, r.checked) == ("existing", False)
    assert row_text(r) == "A file with this name already exists — tick to replace it"


def test_a_file_ticked_to_be_replaced_is_checked_again_after_a_rescan(make_window, tmp_path, synth_tree):
    """CORE_API §3: every row whose existing file has no verdict yet is checked, also one the user ticked to
    replace it (status 'replace'), which before stayed 'checking' for good."""
    out = tmp_path / "out"
    old_output(out / "Slide1_Scan1.ome.tif")
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1", "Slide2_Scan1"], synth_tree)
    win = make_window(input=tiles, output=out)
    root = win.root
    until(root, lambda: ready(win) and row_of(win, "Slide1_Scan1").existing_verdict == "valid", what="checked")
    rid = row_of(win, "Slide1_Scan1").row_id
    win.plan().toggle([rid], True)
    win._plan_edited()
    assert win.plan().row(rid).status == "replace"
    old = win.plan()
    win.rescan()
    until(root, lambda: ready(win) and win.plan() is not old, what="rescanned")
    until(root, lambda: win.plan().row(rid).existing_verdict == "valid", 5, what="checked again")
    assert win.plan().row(rid).status == "replace"


def test_the_plan_gets_the_long_path_setting_once_and_the_temp_folder(make_window, synth_tree, tmp_path, monkeypatch):
    """PL-7: PlanEnv.windows_long_paths is a registry read made once, not per event; PlanEnv.temp_dir is the custom
    temporary folder (its scratch files are the longest paths)."""
    calls = []
    monkeypatch.setattr(osutil, "windows_long_paths", lambda: calls.append(1) or True)
    win = make_window(input=synth_tree / "ProjectA")
    win.settings.set("run.temp_dir", str(tmp_path / "scratch"))
    until(win.root, lambda: ready(win), what="ready")
    env = win._env()
    assert env.windows_long_paths is True and env.temp_dir == str(tmp_path / "scratch") and calls == [1]


def test_the_output_browser_is_told_when_the_tiles_folder_is_a_slide(make_window, synth_tree, monkeypatch):
    """BR-8: for a slide folder the output browser must not preselect that folder of tiles."""
    seen = []
    monkeypatch.setattr(app, "browser_initial", lambda *a, **kw: seen.append(kw["input_is_slide"]) or (a[2], None))
    monkeypatch.setattr("yfs.ui.app.FolderBrowser.open", lambda *a, **kw: None)
    win = make_window(input=synth_tree / "ProjectA" / "SlideA1_Scan1")
    root = win.root
    until(root, lambda: ready(win), what="a slide folder")
    win.browse_output()
    win.set_input(str(synth_tree / "ProjectA"))
    until(root, lambda: ready(win) and win.source == str(synth_tree / "ProjectA"), what="a folder of slides")
    win.browse_output()
    assert seen == [True, False]
