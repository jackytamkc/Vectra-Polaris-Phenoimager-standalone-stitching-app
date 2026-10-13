"""What the browser selects and how it draws it (BR-2, BR-3, BR-20, BR-25, BR-26): a folder asked for is never
silently dropped by a filter, filtering never leaves a hidden row chosen, the temporary-files chooser speaks for
itself, the preview names folders like the rest of the dialog, and long texts end in an ellipsis with a tooltip."""
from __future__ import annotations

import itertools
import os
import stat
from pathlib import Path

import pytest

from test_browser import env, key, pump, q, rows, select, settled, shown, strip, tree, until  # noqa: F401
from yfs.core.settings import Settings
from yfs.ui.dialogs import NewFolderDialog

_CLOCK = itertools.count(10_000, 1_000)                # event times a second apart: never a double-click


def click_row(b, name: str) -> None:
    box = b.tree.bbox(b._names[name])
    x, y = box[0] + b.px(40), box[1] + box[3] // 2
    b.tree.event_generate("<ButtonPress-1>", x=x, y=y, time=next(_CLOCK))
    b.tree.event_generate("<ButtonRelease-1>", x=x, y=y, time=next(_CLOCK))
    b.update()


def checkbuttons(b) -> list:
    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    return [str(w.cget("text")) for w in walk(b) if w.winfo_class() == "TCheckbutton"]


def slide(folder, prefix: str, n: int = 2, suffix: str = "component_data") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_[{1000 + 900 * i},2000]_{suffix}.tif").write_bytes(b"")


@pytest.fixture
def work(tree):
    """tree/Work: Batch (tiles), Stitched (outputs), Old (empty), .archive (hidden), Projects/P1 (slide folders)."""
    w = tree / "Work"
    (w / "Old").mkdir(parents=True)
    slide(w / "Batch", "B01_Scan1")
    (w / "Stitched").mkdir()
    (w / "Stitched" / "B01_Scan1.ome.tif").write_bytes(b"")
    (w / ".archive").mkdir()
    slide(w / "Projects" / "P1" / "S1_Scan1", "S1_Scan1", 3)
    return w


# ------------------------------------------------------------------------------------------------ BR-2
def test_only_folders_with_tiles_is_for_finding_tiles_not_for_saving(env, work):
    """BR-2: ticked while choosing tiles, the option neither shows nor hides anything in the save folder chooser:
    its field's folder is selected and the button names it."""
    ctx = env.ctx()
    ctx.settings.set("browser.only_tiles", True)
    b = env.open((str(work), "Stitched"), mode="output", ctx=ctx)
    until(b, lambda: settled(b) and b.tree.selection())
    pump(b, 600)                                                          # (after the probes' re-sort)
    assert [b._rows[k].name for k in b.tree.selection()] == ["Stitched"]
    assert str(b._primary.cget("text")) == f"Save here: {q(b, 'Stitched')}"
    assert {"Batch", "Old", "Projects", "Stitched"} <= set(rows(b))
    assert "Only folders with tiles" not in checkbuttons(b)
    b.cancel()
    assert ctx.settings.save_now(2) and Settings.load().get("browser.only_tiles") is True     # still on for tiles


@pytest.mark.parametrize("how", ["field", "picked", "only_tiles"])
def test_a_folder_asked_for_is_shown_and_selected_although_a_filter_hides_it(env, work, how):
    """BR-2: the field's folder, a folder from the system picker, or one that 'Only folders with tiles' hides is
    shown (dimmed like any hidden item), selected and named on the button; the footer says why it is there. Use
    returns it. A filter the user changes afterwards applies to it again."""
    ctx = env.ctx()
    ctx.settings.set("browser.only_tiles", how == "only_tiles")
    name = "Stitched" if how == "only_tiles" else ".archive"
    b = env.open((str(work), name if how != "picked" else None), ctx=ctx)
    until(b, lambda: settled(b))
    if how == "picked":
        b._picked(str(work / name))
        until(b, lambda: settled(b) and b._listing is not None)
    until(b, lambda: b.tree.selection(), what="the folder asked for")
    assert [b._rows[k].name for k in b.tree.selection()] == [name] and b.target() == str(work / name)
    assert str(b._primary.cget("text")) == f"Use {q(b, name)}"
    until(b, lambda: b._notes[b._others].startswith(b.T(f"Also showing {q(b, name)}, which your filters hide")),
          what="the footer note")                          # (only_tiles: once the probe found no tiles)
    pump(b, 500)
    assert b.tree.selection() and b.target() == str(work / name)
    if how == "only_tiles":
        key(b.tree, "<Alt-t>")                                            # the user's own filter change
        key(b.tree, "<Alt-t>")
        until(b, lambda: name not in rows(b), what="the filter applies again")
        assert b.tree.selection() == () and b.target() == str(work)
    else:
        key(b.tree, "<Control-Return>")
        assert env.done == [str(work / name)]


# ------------------------------------------------------------------------------------------------ BR-3
def test_filtering_never_leaves_a_hidden_row_chosen(env, work):
    """BR-3: a row the filter hides is no longer selected, so after the filter is cleared only the row clicked
    last is selected and it is the target."""
    b = env.open((str(work), None))
    until(b, lambda: settled(b))
    click_row(b, "Batch")
    b._filter_var.set("stit")
    until(b, lambda: list(rows(b)) == ["Stitched"], what="the filter")
    assert b.tree.selection() == () and b.target() == str(work)
    click_row(b, "Stitched")
    key(b.tree, "<Escape>")                                               # clears the filter
    until(b, lambda: "Batch" in rows(b))
    assert [b._rows[k].name for k in b.tree.selection()] == ["Stitched"] and b.target() == str(work / "Stitched")
    assert str(b._primary.cget("text")) == f"Use {q(b, 'Stitched')}"


def test_the_row_chosen_stays_when_its_probe_finds_what_the_filter_hides(env, work):
    """BR-3: with 'Only folders with tiles' on, a folder chosen while it is still being looked at stays shown and
    chosen when it turns out to hold no tiles (the footer says why it is there): the target never switches by
    itself. Only the user's own filter change unchooses it."""
    ctx = env.ctx()
    ctx.settings.set("browser.only_tiles", True)
    gate = env.fake.block(work / "Old")
    b = env.open((str(work), None), ctx=ctx)
    until(b, lambda: b._listing is not None and "Old" in rows(b), what="the folder still being looked at")
    click_row(b, "Old")
    assert b.target() == str(work / "Old")
    gate.set()
    until(b, lambda: settled(b))
    pump(b, 600)                                                          # (after the probes' re-sort)
    assert "Old" in rows(b) and b.target() == str(work / "Old")
    assert str(b._primary.cget("text")) == f"Use {q(b, 'Old')}"
    assert b._notes[b._others].startswith(b.T(f"Also showing {q(b, 'Old')}, which your filters hide"))
    key(b.tree, "<Alt-t>")                                                # the user's own filter change
    key(b.tree, "<Alt-t>")
    until(b, lambda: "Old" not in rows(b), what="the filter applies again")
    assert b.tree.selection() == () and b.target() == str(work)


# ------------------------------------------------------------------------------------------------ BR-20
def test_the_temporary_files_chooser_speaks_for_itself(env, work, monkeypatch):
    """BR-20 (UD11): its own title and button, a strip with the free space only (no tiles warning, no Create
    action), no Tiles column, no 'Only folders with tiles', no recent save folders; Use checks that it can write
    there and remembers the folder under its own key."""
    ctx = env.ctx()
    ctx.settings.set("paths.browser_last.output", str(work / "Stitched"))
    ctx.settings.add_recent("output", str(work / "Stitched"))
    b = env.open((str(work), "Batch"), mode="temp", ctx=ctx)
    until(b, lambda: settled(b) and b._place_items and b.tree.selection() and "Free space" in strip(b)[1])
    assert str(b.title()) == "Choose a folder for temporary files"
    assert str(b._primary.cget("text")) == f"Use for temporary files: {q(b, 'Batch')}"
    assert strip(b)[0] == "Ok" and strip(b)[1].startswith("Free space: ") and not b._strip_btn.winfo_manager()
    assert "tiles" not in b.tree.cget("displaycolumns") and "Only folders with tiles" not in checkbuttons(b)
    assert not any(p.section == "recent" for p, _l in b._place_items.values())
    assert rows(b)["Batch"][2] == ()                                      # not bold: nothing stands out here
    locked = work / "Old"
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if not os.access(str(locked), os.W_OK):                           # (root ignores permissions)
            select(b, "Old")
            b.choose()
            until(b, lambda: strip(b)[0] == "Error", what="the write check refuses")
            assert strip(b) == ("Error", "Can't write here: Permission denied.") and env.done == []
    finally:
        locked.chmod(stat.S_IRWXU)
    select(b, "Batch")
    key(b.tree, "<Control-Return>")
    until(b, lambda: env.done, what="the write check")
    assert env.done == [str(work / "Batch")] and not list((work / "Batch").glob(".yfs-write-test*"))
    assert ctx.settings.get("paths.browser_last.temp") == str(work)
    assert ctx.settings.get("paths.browser_last.output") == str(work / "Stitched")



@pytest.mark.parametrize("has_stitched", [False, True])
def test_the_temporary_files_chooser_can_make_a_folder(env, work, has_stitched):
    """BR-20: like the save folder chooser, it makes a folder (New folder, Ctrl+Shift+N), with no name offered
    ('stitched' is for stitched images); the new folder is selected and named on the button."""
    parent = work if has_stitched else work / "Old"       # no existing name can suppress the output default
    b = env.open((str(parent), None), mode="temp")
    until(b, lambda: settled(b))
    key(b.tree, "<Control-N>")
    dlg = until(b, lambda: next((w for w in b.winfo_children() if isinstance(w, NewFolderDialog)), None))
    assert dlg.var.get() == ""
    dlg.var.set("scratch")
    dlg._ok()
    until(b, lambda: b.tree.selection() and b._rows[b.tree.selection()[0]].name == "scratch")
    assert (parent / "scratch").is_dir()
    assert str(b._primary.cget("text")) == f"Use for temporary files: {q(b, 'scratch')}"


# ------------------------------------------------------------------------------------------------ BR-25
def test_the_preview_names_folders_like_the_dialog_and_maps_a_slide_inside(env, tree):
    """BR-25: Home is titled 'Home' (its place's name, like the crumb and the button), and a folder of slide
    folders shows the tile layout of the first slide inside."""
    home = Path(os.environ["HOME"])                        # (the env fixture's own home)
    slide(home / "Lab" / "S1_Scan1", "S1_Scan1", 3)
    b = env.open((str(home), None))
    until(b, lambda: settled(b) and b._place_items and b._preview_on)
    until(b, lambda: str(b._facts.facts.winfo_children()[0].cget("text")) == "Home", what="the place's name")
    select(b, "Lab")
    until(b, lambda: b._facts.drawing.winfo_manager() == "grid", what="the tile map")
    assert str(b._facts.facts.winfo_children()[0].cget("text")) == "Lab"
    assert str(b._facts.caption.cget("text")) == "Tile layout of S1_Scan1 (from file names)"


# ------------------------------------------------------------------------------------------------ BR-26
def test_long_names_and_badges_end_in_an_ellipsis_with_a_tooltip(env, tree):
    """BR-26: at the smallest size, a long name is shortened in the middle and a long badge at the end, with an
    ellipsis and never wider than its column; the pointer on the row shows both in full. The current folder's
    crumb is shortened too."""
    long_name = "2026-01-15_Example_tissue_FFPE_whole_section_round2_Scan1"
    slide(tree / long_name, "T1", 2, "composite_image")
    slide(tree / long_name, "T1", 2, "binary_seg_maps")
    deep = tree / ("Experiment_" + "x" * 110)
    deep.mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    badge = "binary_seg_maps, composite_image tiles, not unmixed components"
    until(b, lambda: shown(b, long_name) != (long_name, badge), what="the rows fitted to the narrow columns")
    name, inside = shown(b, long_name)
    ell = b.G["ell"]
    assert ell in name and name.startswith("2026") and name.endswith("Scan1") and inside.endswith(ell)
    ruler = b._rulers[False]
    assert ruler.width(name) <= b._fit_widths[0] and ruler.width(inside) <= b._fit_widths[1]
    k = b._names[long_name]
    box = b.tree.bbox(k)
    b.tree.event_generate("<Motion>", x=box[0] + b.px(30), y=box[1] + 2)
    until(b, lambda: b._list_tip._tip is not None, timeout=3, what="the tooltip")
    text = str(b._list_tip._tip.winfo_children()[0].cget("text"))
    assert text == b.T(f"{long_name}\n{badge}")
    b.navigate(str(deep))
    until(b, lambda: b._listing is not None)
    b.update()
    crumb = str(b._crumb_ws[-1][1].cget("text"))
    assert ell in crumb and b._crumb_box.winfo_reqwidth() <= b._pathbar.winfo_width()


def test_a_shortened_badge_ends_at_a_word_and_never_leaves_a_bracket_open(env, tree):
    """BR-26: '2 whole-slide images (another program)' with room for '…(another' reads '2 whole-slide images…'."""
    b = env.open((str(tree), None))
    ruler, ell = b._rulers[False], b.G["ell"]
    text = "2 whole-slide images (another program)"
    room = ruler.width("2 whole-slide images (another pr" + ell)
    assert ruler.fit(text, room, ell) == "2 whole-slide images" + ell
    assert ruler.fit("composite_image tiles, not unmixed components", ruler.width("composite_image tiles, not unm"),
                     ell) == "composite_image tiles, not" + ell
    assert ruler.fit(text, ruler.width(text), ell) == text


def test_the_footer_notes_end_in_an_ellipsis_when_the_window_is_narrow(env, tree):
    """BR-26: at the smallest size the footer's notes are shortened at a word, with an ellipsis, never cut off
    mid-word by the window's edge; the pointer on them shows the whole text."""
    for name in ("a", "b", "c"):
        (tree / name).mkdir()
    for name in ("notes.txt", "protocol.pdf"):
        (tree / name).write_bytes(b"")
    slide(tree / "S1_Scan1", "S1_Scan1")
    ctx = env.ctx()
    ctx.settings.set("browser.only_tiles", True)
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b) and set(rows(b)) == {"S1_Scan1"})
    full, ell, notes = b.T("2 other files not shown · 3 folders without tiles not shown"), b.G["ell"], b._others
    assert b._notes[notes] == full
    b.geometry(f"{b.px(720)}x{b.px(480)}")
    until(b, lambda: str(notes.cget("text")) != full, what="the notes fitted to the narrow footer")
    pump(b, 100)
    text = str(notes.cget("text"))
    assert text.endswith(ell) and full.startswith(text[:-len(ell)]) and full[len(text) - len(ell)] == " "
    assert notes.winfo_x() + notes.winfo_reqwidth() <= notes.master.winfo_width()
    notes.event_generate("<Enter>", x=2, y=2)
    tip = until(b, lambda: [w for w in notes.winfo_children() if w.winfo_class() == "Toplevel"], timeout=3,
                what="the tooltip")
    assert str(tip[0].winfo_children()[0].cget("text")) == full


def test_a_sidebar_dragged_wider_keeps_its_width_when_the_preview_comes_back(env, tree):
    """The sidebar the user dragged keeps its width when the window is narrowed below the preview's px(1000) and
    widened again: it doesn't snap back to the width saved when the dialog last closed."""
    (tree / "a").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b) and b._preview_on)
    b.geometry(f"{b.px(1080)}x{b.px(680)}")
    pump(b, 300)
    b._pw.sashpos(0, b.px(300))                                            # dragged wider
    pump(b, 100)
    for width in (900, 1080):
        b.geometry(f"{b.px(width)}x{b.px(680)}")
        until(b, lambda: b._preview_on == (width >= 1000), what="the preview pane follows the width")
        pump(b, 300)
    assert abs(b._pw.sashpos(0) - b.px(300)) <= 2
