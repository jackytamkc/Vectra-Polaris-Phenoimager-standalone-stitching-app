"""The main window's folder fields and the save folder's status (UD6, UD7): typed paths must be full paths, the
folders can't change under Stitch's final check, a new folder after a batch starts afresh, and step 2 says what is
known about the files already there."""
from __future__ import annotations

from test_mainwindow import (_no_stale_tk_garbage, dialog, gates, hang, make_slides, make_window,  # noqa: F401
                             old_output, pump, ready, row_of, status, until)
from tests.fakes import ENOSPC_ERROR, FakeEngine
from yfs.core import preflight
from yfs.core.names import PARTIAL_DIR, temp_subdir
from yfs.ui import app, dialogs


def commit(win, field, text: str) -> None:
    """Type `text` into a folder field and press Enter."""
    win.root.focus_force()
    field.combo.focus_set()
    pump(win.root, 50)
    field._var.set(text)
    field.combo.event_generate("<Return>")
    pump(win.root, 50)


# ------------------------------------------------------------------------------------------------ UD6: typed paths
def refused(win, text: str) -> str:
    return win.ctx.theme.text(f"Can't find ‘{text}’. Type the full path of a folder.")


def test_a_typed_relative_folder_is_refused_in_both_steps(make_window, synth_tree, tmp_path, monkeypatch):
    """DS-11, PL-3: 'Stitched' would be found from wherever the app was started: refused with the browser's words.
    The folders and settings stay as they were, and so do the fields: they show the folders Stitch would use. The
    refusal is no search result: after a batch step 1 says what the search found."""
    monkeypatch.chdir(tmp_path)
    win = make_window(input=synth_tree / "ProjectA", output=tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    found = status(win.input_field)
    commit(win, win.output_field, "Stitched")
    assert status(win.output_field) == refused(win, "Stitched") and win.output_field.get() == str(tmp_path / "out")
    assert win.out_dir == str(tmp_path / "out") and win.settings.get("paths.output") == str(tmp_path / "out")
    commit(win, win.input_field, "tiles")
    assert status(win.input_field) == refused(win, "tiles") and win.state == "READY"
    assert win.source == win.input_field.get() == str(synth_tree / "ProjectA")
    assert win.settings.get("paths.input") == win.source
    win.plan().select("all_new")
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    assert status(win.input_field) == found and not (tmp_path / "Stitched").exists()


def test_a_typed_relative_folder_or_f5_during_a_lookup_keep_to_that_folder(make_window, synth_tree, monkeypatch,
                                                                          gates):
    """UD6, MW-14: while a tile folder is being looked up (a slow share), it is the folder in use: a refused text
    gives the field back to it, and F5 looks it up again, not the folder searched before."""
    gate = gates()
    hang(monkeypatch, app, "_locate", gate)
    win = make_window(input=synth_tree / "round1_unmix")
    root = win.root
    until(root, lambda: win.state == "CHECKING", what="looking it up")
    gate.set()
    until(root, lambda: ready(win), what="the first folder")
    gate.clear()
    win.set_input(str(synth_tree / "ProjectA"))
    commit(win, win.input_field, "tiles")
    assert win.state == "CHECKING" and status(win.input_field) == refused(win, "tiles")
    assert win.input_field.get() == str(synth_tree / "ProjectA")
    root.event_generate("<F5>")
    gate.set()
    until(root, lambda: ready(win) and win.source == str(synth_tree / "ProjectA"), what="searched")
    pump(root, 300)
    assert win.source == win.input_field.get() == str(synth_tree / "ProjectA")


def test_f5_after_a_failed_lookup_cleans_the_typed_text_again(make_window, synth_tree, tmp_path):
    """MW-14: F5 after 'Can't find this folder' looks the field's '~/…' text up as a full path."""
    win = make_window()
    root = win.root
    later = tmp_path / "home" / "lab" / "Later"
    later.parent.mkdir(parents=True)
    win.set_input("~/lab/Later")
    until(root, lambda: "Can't find this folder" in status(win.input_field), what="missing")
    assert win.input_field.get() == str(later) and win.source is None
    make_slides(later, ["Slide1_Scan1"], synth_tree)                     # the export has arrived
    root.event_generate("<F5>")
    until(root, lambda: ready(win), what="found")
    assert win.source == str(later) and row_of(win, "Slide1_Scan1") is not None


def test_relative_input_and_output_flags_are_made_absolute(tmp_path, monkeypatch):
    """UD6: --input/--output are relative to the folder the app was started from, made absolute in main(); a URL
    is left for the window to clean like a pasted one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    seen = []
    monkeypatch.setattr(app, "_run", lambda engine, args: seen.append(args) or 0)
    assert app.main(object, ["--input", "tiles", "--output", "~/Stitched"]) == 0
    assert (seen[0].input, seen[0].output) == (str(tmp_path / "tiles"), str(tmp_path / "home" / "Stitched"))
    urls = ["file:///data/ProjectC", "smb://fileserver.example.org/labshare/stitched"]
    assert app.main(object, ["--input", urls[0], "--output", urls[1]]) == 0
    assert [seen[1].input, seen[1].output] == urls


# ------------------------------------------------------------------------------------------------ UD7
def test_a_folder_change_during_the_final_check_makes_it_void(make_window, synth_tree, tmp_path, monkeypatch, gates):
    """CT-5, MW-13: during the final check the folder fields and Browse are disabled; a save folder set anyway (a
    script) voids the check: its answer, about the old folder, starts nothing."""
    win = make_window(FakeEngine.configure(), synth_tree / "ProjectA", tmp_path / "outA")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win._plan_edited(full=True)
    opened = []
    monkeypatch.setattr("yfs.ui.app.FolderBrowser.open", lambda *a, **kw: opened.append(kw))
    gate = gates()
    hang(monkeypatch, app, "_final_check", gate)
    win.start(confirm=False)
    for field in (win.input_field, win.output_field):
        assert field.combo.instate(["disabled"]) and field.browse.instate(["disabled"])
    win.browse_output()
    win.browse_input()
    assert opened == []                                                   # Ctrl+O / Ctrl+Shift+O do nothing either
    win.set_output(str(tmp_path / "outB"))
    assert win._final is None and win.output_field.browse.instate(["!disabled"])
    gate.set()                                                            # outA's answer arrives: ignored
    until(root, lambda: ready(win), what="outB indexed")
    pump(root, 300)
    assert win.state == "READY" and win.runner is None and win.out_index.out_dir == str(tmp_path / "outB")
    assert win.runbar.button.instate(["!disabled"]) and not (tmp_path / "outB").exists()


def test_a_new_save_folder_after_a_batch_starts_afresh(make_window, synth_tree, tmp_path):
    """DS-15, MWV-1, MW-4: after a batch, another save folder dismisses the banner and rebuilds the plan: nothing is
    stitched there yet, so the slides are ticked and Stitch offers them again."""
    tiles = make_slides(tmp_path / "ProjectC", ["Slide1_Scan1", "Slide2_Scan1"], synth_tree)
    win = make_window(FakeEngine.configure(), tiles, tmp_path / "outA")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    old = win.plan()
    assert win.runbar.mode == "finished" and {r.status for r in old.rows() if r.section != "note"} == {"done"}
    win.set_output(str(tmp_path / "outB"))
    until(root, lambda: ready(win), what="outB indexed")
    plan = win.plan()
    assert plan is not old and win.state == "READY" and win.runbar.mode == "idle"
    assert [(r.status, r.checked) for r in plan.rows() if r.section != "note"] == [("new", True), ("new", True)]
    assert str(win.runbar.button.cget("text")) == "Stitch 2 slides" and win.runbar.button.instate(
        ["!disabled"])


def test_a_new_tile_folder_after_a_batch_dismisses_its_banner(make_window, synth_tree, tmp_path):
    """MW-4: the banner of the last batch goes when another tile folder is chosen; its buttons would act on the
    new plan."""
    Engine = FakeEngine.configure(fail="write", last_error=ENOSPC_ERROR,
                                  fail_if=lambda eng: eng.image_name == "SlideA2_Scan1")
    win = make_window(Engine, synth_tree / "ProjectA", tmp_path / "out")
    root = win.root
    until(root, lambda: ready(win), what="ready")
    win.plan().select("all_new")
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED", 30, what="finished")
    assert win.runbar.mode == "finished"
    retry = next(b for b in win.runbar.banner.winfo_children()[0].winfo_children()
                 if b.winfo_class() == "TButton" and str(b.cget("text")) == "Retry failed")
    other = make_slides(tmp_path / "ProjectD", ["Slide1_Scan1"], synth_tree)
    win.set_input(str(other))
    until(root, lambda: ready(win) and win.source == str(other), what="the new folder")
    assert win.runbar.mode == "idle" and win.runbar.button.winfo_manager()
    ticks = [(r.row_id, r.checked) for r in win.plan().rows()]
    assert ("Slide1_Scan1", True) in [(r.cand.folder_name, r.checked) for r in win.plan().rows()]
    retry.invoke()                                     # the old banner's button (hidden now) acts on nothing
    assert win.state == "READY" and [(r.row_id, r.checked) for r in win.plan().rows()] == ticks
    assert str(win.runbar.button.cget("text")) == "Stitch 1 slide"      # no Stitch started either


def test_step2_counts_only_checked_files_as_stitched(make_window, tmp_path, synth_tree, monkeypatch):
    """MW-12, UD7: 'already stitched here' counts valid files only; a file that looks incomplete and one that
    couldn't be checked are said apart."""
    out = tmp_path / "out"
    names = ["Slide1_Scan1", "Slide2_Scan1", "Slide3_Scan1", "Slide4_Scan1"]
    tiles = make_slides(tmp_path / "ProjectC", names, synth_tree)
    old_output(out / "Slide1_Scan1.ome.tif")                              # valid
    old_output(out / "Slide2_Scan1.ome.tif", b"")                         # empty: looks incomplete
    old_output(out / "Slide3_Scan1.ome.tif")                              # couldn't be checked
    real = preflight.verify_output
    monkeypatch.setattr(preflight, "verify_output", lambda path: ("unchecked", "Permission denied")
                        if "Slide3" in path else real(path))
    win = make_window(input=tiles, output=out)
    root = win.root
    until(root, lambda: ready(win) and "checking" not in status(win.output_field), what="checked")
    assert status(win.output_field) == win.ctx.theme.text(
        f"Local disk · {status(win.output_field).split(' · ')[1]} · 1 of your 4 slides is already stitched here · "
        "1 existing file looks incomplete — it will be redone · couldn't check 1 existing file")   # not counted twice
    assert row_of(win, "Slide2_Scan1").status == "incomplete" and row_of(win, "Slide3_Scan1").status == "unchecked"


def test_step2_says_which_folders_will_be_created(make_window, tmp_path):
    """MW-26: more than one missing level (a typo? a share that isn't mounted?) is a warning naming them."""
    (tmp_path / "Data").mkdir()
    win = make_window(output=tmp_path / "Data" / "Stiched" / "ProjectC" / "stitched")
    until(win.root, lambda: win.out_index is not None, what="indexed")
    assert status(win.output_field) == "3 new folders will be created inside Data: Stiched/ProjectC/stitched"
    assert str(win.output_field._text.cget("style")) == "Surface.Warn.TLabel"
    win.set_output(str(tmp_path / "Data" / "stitched"))                  # one level: the usual case
    until(win.root, lambda: "doesn't exist yet" in status(win.output_field), what="one new folder")


def test_a_linked_partial_folder_is_a_step2_error_and_blocks_stitch(make_window, synth_tree, tmp_path):
    """DS-10: a .stitcher-in-progress that is a link to another place is never used: step 2 says so, and the plan
    blocks Stitch."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / PARTIAL_DIR).symlink_to(elsewhere)
    win = make_window(input=synth_tree / "ProjectA", output=out)
    until(win.root, lambda: ready(win), what="ready")
    assert status(win.output_field).startswith(f"Can't save here: its {PARTIAL_DIR} folder is a link")
    assert any(i.severity == "block" and i.code == "pdir_problem" for i in win._issues())


def test_scratch_files_left_in_the_custom_temp_folder_are_offered_and_deleted(make_window, synth_tree, tmp_path):
    """DS-2, DS-16: with a custom temporary folder, this save folder's stale scratch files there are leftovers too,
    and [Delete them] removes them, except one whose slide another stitcher has started on meanwhile: the check is
    made again when deleting, against the save folder's partial (D3)."""
    out, temp = tmp_path / "out", tmp_path / "scratch"
    scratch = type(out)(temp_subdir(str(temp), str(out)))
    raws = [old_output(scratch / f"{name}.ome.tmp.raw", b"x" * 2048) for name in ("SlideA1_Scan1", "SlideA2_Scan1")]
    win = make_window()
    win.settings.set("run.temp_dir", str(temp))
    win.set_input(str(synth_tree / "ProjectA"))
    win.set_output(str(out))
    root = win.root
    until(root, lambda: ready(win) and "leftover" in status(win.output_field), what="the leftovers")
    assert status(win.output_field) == "2 leftover files (4 KB) from an unfinished run."
    win.output_field._actions[0].invoke()
    confirm = until(root, lambda: dialog(root, dialogs._Dialog), what="confirmation")
    (out / PARTIAL_DIR).mkdir(parents=True)                               # another stitcher starts on SlideA1
    (out / PARTIAL_DIR / "SlideA1_Scan1.ome.tif").write_bytes(b"")
    confirm._close(True)
    until(root, lambda: not raws[1].exists(), what="deleted")
    assert raws[0].exists()


def test_without_a_finished_search_no_partial_is_offered_as_a_leftover(make_window, synth_tree, tmp_path):
    """DS-9: an unfinished image of a slide the search hasn't found (yet) is not a leftover to delete."""
    out = tmp_path / "out"
    old_output(out / PARTIAL_DIR / "SlideA1_Scan1.ome.tif", b"x" * 2048)
    win = make_window(output=out)
    root = win.root
    until(root, lambda: win.out_index is not None, what="indexed")
    assert win.out_index.leftovers == () and "leftover" not in status(win.output_field)
    win.set_input(str(synth_tree / "unrelated" / "WSIs"))                # a plan without that slide
    until(root, lambda: ready(win), what="ready")
    assert "1 leftover file" in status(win.output_field)
