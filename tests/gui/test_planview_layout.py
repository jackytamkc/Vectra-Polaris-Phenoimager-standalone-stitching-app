"""PlanTable columns and cells (UD9): every cell shortened with an ellipsis to its column, Folder and Output in the
middle, the full text in a tooltip and the details strip (MW-10); right-aligned numbers never touch the text after
them (MW-17); section labels are never clipped (MW-18) and no column moves during a run; the rename editor shows the
whole '.ome.tif' (MW-25) and a whole error message."""
from __future__ import annotations

import os

import pytest
from PIL import ImageGrab

from test_planview import OUT, SRC, Calls, _button, cand, every_status, header, press

from yfs.core.model import RowOutcome
from yfs.core.plan import Plan, row_text
from yfs.ui.planview import COLUMNS, PlanTable, _fit_folder

GB = 1024 ** 3
LONG = "01_ProjectC_round1_unmix/batch_2025-01-15/QA731104-05_Scan1_with_a_long_name"


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
@pytest.mark.parametrize("first_run", [True, False])
def test_welcome_keeps_choose_visible_and_keyboard_reachable_in_a_short_pane(ui_ctx, pump, first_run):
    table = make(ui_ctx, pump, 760)
    root, px, seen = ui_ctx.root, ui_ctx.theme.px, []
    root.geometry(f"{px(760)}x{px(190)}+0+0")
    table.show_welcome(first_run, lambda: seen.append("choose"), lambda: None)
    settle(table, pump)
    choose = _button(table._welcome, ui_ctx.theme.text("Choose tile folder…"))
    assert choose.winfo_ismapped()
    assert choose.winfo_height() >= choose.winfo_reqheight()
    assert choose.winfo_rooty() >= table._welcome.winfo_rooty()
    assert choose.winfo_rooty() + choose.winfo_height() <= (table._welcome.winfo_rooty() +
                                                           table._welcome.winfo_height())
    root.focus_force()
    table.select_button.focus_set()
    for _ in range(12):
        focused = root.focus_get()
        if focused is choose:
            break
        focused.event_generate("<Tab>")
        pump(root, 10)
    assert root.focus_get() is choose
    choose.event_generate("<space>")
    pump(root, 30)
    assert seen == ["choose"]


def make(ui_ctx, pump, width):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(width)}x{px(560)}+0+0")
    calls = Calls()
    t = PlanTable(root, ui_ctx, on_focus=lambda r: None, on_details=lambda r: None, on_fix_channels=lambda ids: None,
                  on_note_action=lambda a, r: None, on_changed=lambda: calls.append(("changed",)))
    t.pack(fill="both", expand=True, padx=px(12))
    pump(root, 30)
    return t


def settle(t, pump):
    pump(t, 120)
    t.update_idletasks()


def cells(t):
    """(iid, column, full text, shown text) of every displayed cell of the open sections."""
    tree = t.tree
    for sec in tree.get_children(""):
        for iid in (sec,) + (tree.get_children(sec) if tree.item(sec, "open") else ()):
            full = t._full[iid]
            yield iid, "#0", full[0], tree.item(iid, "text")
            for col in tree["displaycolumns"]:
                yield iid, col, full[2][COLUMNS.index(col)], tree.set(iid, col)


def room(t, iid, col):
    """The width a cell's text may take: its column less the cell inset (and the image and indent in #0)."""
    w = int(t.tree.column(col, "width")) - t._pad
    if col == "#0":
        w -= t._pair_w + (0 if iid.startswith("sec:") else t._indent)
    return w


def font_of(t, iid):
    tags = t.tree.item(iid, "tags")
    return t.ctx.theme.fonts["YfsBold" if tags and tags[0] in ("section", "running") else "YfsBody"]


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
@pytest.mark.parametrize("width", [1040, 760])
def test_every_cell_fits_its_column_with_an_ellipsis_when_shortened(ui_ctx, pump, width):
    t = make(ui_ctx, pump, width)
    plan, c = every_status()
    twins = [cand(f"twins_from_the_second_scanner/batch{i}/QA731104-05_Scan1_long_copy", n=50)
             for i in (1, 2)]                                    # D1: '…__batch1'
    for x in [cand(LONG, n=88)] + twins:
        plan.add_candidate(x)
        plan.apply_header(x.key, header(), None, 64 * GB)
    t.bind_plan(plan)
    t.set_state("READY")
    t._open["excluded"] = True
    t.refresh([])
    settle(t, pump)
    ell, cut = ui_ctx.theme.G["ell"], 0
    for iid, col, full, shown in cells(t):
        assert font_of(t, iid).measure(shown) <= room(t, iid, col), (iid, col, shown)
        if shown != full:
            cut += 1
            assert ell in shown, (iid, col, shown)
            head, end = shown.rsplit(ell, 1)                        # cut at the end, or before the part
            assert full.startswith(head) and full.endswith(end), (iid, col, shown)   # that tells copies apart
    assert cut                                                  # the long texts were shortened
    long = t._iid[cand(LONG, n=88).key]
    folder, output = t.tree.set(long, "folder"), t.tree.set(long, "output")
    parent, end = LONG.rsplit("/", 1)[0], "/batch_2025-01-15"
    if font_of(t, long).measure(ell + end) <= room(t, long, "folder"):
        assert folder.endswith(end), folder                     # the last part stays whole,
    else:                                                       # or the middle goes
        assert folder.startswith("01_") and parent.endswith(folder.split(ell)[-1]), folder
    for i, x in enumerate(twins, 1):                            # copies keep what tells them apart
        assert t.tree.set(t._iid[x.key], "folder").endswith(f"{ell}/batch{i}")
    assert output.startswith("QA7") and output.endswith(ell)    # no '__': cut at the end, like the slide's name
    if width == 1040:                                           # (760 leaves no room for it: Folder tells them apart)
        outputs = [t.tree.set(t._iid[x.key], "output") for x in twins]
        assert [o.split(ell)[-1] for o in outputs] == ["__batch1.ome.tif", "__batch2.ome.tif"], outputs


def test_a_folder_keeps_its_last_part_whole_while_it_fits(theme):
    """'duplicates/batchA' in a narrow column: '…/batchA' tells the copies apart, 'dupl…atchA' hardly does."""
    font, ell, text = theme.fonts["YfsBody"], theme.G["ell"], "duplicates/batchA"
    assert _fit_folder(font, text, font.measure(text), ell) == text
    for width in range(font.measure(ell + "/batchA"), font.measure(text)):
        shown = _fit_folder(font, text, width, ell)
        assert shown.endswith(ell + "/batchA") and font.measure(shown) <= width, (width, shown)
    shown = _fit_folder(font, text, font.measure(ell + "/batchA") - 1, ell)     # not even that: the middle goes
    assert ell in shown and shown[0] == "d" and shown[-1] == "A"


def test_a_shortened_cell_shows_its_whole_text_in_a_tooltip_and_the_details_strip(ui_ctx, pump):
    t = make(ui_ctx, pump, 760)
    plan, c = every_status()
    t.bind_plan(plan)
    t.set_state("READY")
    settle(t, pump)
    tree, tip = t.tree, t._tip
    tip.delay = 20
    herr = t._iid[c["herr"].key]                                  # 'Can't read the first tile (...). Right-click...'
    full = ui_ctx.theme.text(row_text(plan.row(c["herr"].key)))
    assert tree.set(herr, "what") != full and t._full[herr][2][-1] == full
    tree.see(herr)
    settle(t, pump)
    x, y, w, h = tree.bbox(herr, "what")
    tree.event_generate("<Motion>", x=x + w // 2, y=y + h // 2)
    pump(t, 150)
    assert tip._tip is not None and tip.full == full
    label = tip._tip.winfo_children()[0]
    assert str(label.cget("text")) == full
    assert tip._tip.winfo_rooty() >= tree.winfo_rooty() + y + h        # just under the cell
    new = t._iid[c["new"].key]                                    # 'New file' fits: no tooltip
    x, y, w, h = tree.bbox(new, "what")
    tree.event_generate("<Motion>", x=x + w // 2, y=y + h // 2)
    pump(t, 150)
    assert tip._tip is None
    tree.focus(herr)
    tree.selection_set(herr)
    pump(t, 30)
    assert full in t.details._yfs_full                            # the strip names the status too
    recent = t._iid[c["recent"].key]                              # its top warning is its status: said once
    tree.focus(recent)
    tree.selection_set(recent)
    pump(t, 30)
    assert t.details._yfs_full.count("Folder changed") == 1 and t.details._yfs_full.count("120 tiles") == 1


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_numbers_never_touch_the_text_after_them(ui_ctx, pump):
    """MW-17: the right-aligned Tiles/Size values end at least px(14) before the Folder text starts, as drawn (cell
    insets alone leave 10-12 px on both Tk builds, which reads as one phrase: '12 duplicates')."""
    t = make(ui_ctx, pump, 1040)
    plan, c = every_status()
    t.bind_plan(plan)
    t.set_state("READY")
    settle(t, pump)
    px, tree = ui_ctx.theme.px, t.tree
    for width, number in ((1040, "size"), (760, "tiles")):        # (Size is hidden in a narrow window)
        ui_ctx.root.geometry(f"{px(width)}x{px(560)}")
        settle(t, pump)
        assert number in tree["displaycolumns"]
        for key in ("new", "big"):
            iid = t._iid[c[key].key]
            tree.see(iid)
            settle(t, pump)
            assert ink_gap(tree, iid, number, "folder") >= px(14), (width, key)


def ink_gap(tree, iid, left, right):
    """Pixels between the last dark pixel of cell `left` and the first of cell `right` in row `iid`, as drawn."""
    lx, y, lw, h = tree.bbox(iid, left)
    rx, _, rw, _ = tree.bbox(iid, right)
    x0, y0 = tree.winfo_rootx(), tree.winfo_rooty()
    img = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"), bbox=(x0 + lx, y0 + y, x0 + rx + rw, y0 + y + h))
    pix = img.load()
    ink = [x for x in range(img.size[0]) if any(sum(pix[x, yy][:3]) < 450 for yy in range(img.size[1]))]
    return min(x for x in ink if x >= rx - lx) - max(x for x in ink if x < lw)


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_section_labels_are_never_clipped_at_the_minimum_width(ui_ctx, pump):
    """MW-18: 760 px with 120 slides: 'New slides  (119 of 120 ticked)', also while running ('1 done' goes to the
    What cell)."""
    t = make(ui_ctx, pump, 760)
    plan = Plan(SRC, OUT)
    for i in range(120):
        plan.add_candidate(cand(f"project/batch{i // 40}/Slide{i:03d}_Scan1", n=20 if i else 3))
    t.bind_plan(plan)
    t.set_state("READY")
    settle(t, pump)
    bold = ui_ctx.theme.fonts["YfsBold"]

    def whole(label):
        assert t.tree.item("sec:new", "text") == label == t._full["sec:new"][0]
        assert bold.measure(label) <= room(t, "sec:new", "#0")
    whole("New slides  (119 of 120 ticked)")
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    t.set_state("RUNNING")
    plan.finish(job.row_id, RowOutcome("done", "Done · 1 GB · 1 min"))
    t.refresh([job.row_id])
    settle(t, pump)
    whole("New slides  (118 of 119 ticked)")
    assert t.tree.set("sec:new", "what") == "1 done"


def widths(t):
    return {col: int(t.tree.column(col, "width")) for col in ("#0",) + tuple(t.tree["displaycolumns"])}


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
@pytest.mark.parametrize("width", [1040, 760])
def test_no_column_moves_from_stitch_to_the_end_of_the_run(ui_ctx, pump, width):
    """MW-18: the columns keep the widths the plan gave them through the run. Short names, so the Slide column is
    as wide as its section label; the first slide to run has the longest name, which turns bold, and a long
    progress text; another fails with a long reason; a stop is asked for. The run's texts fit the room planned."""
    t = make(ui_ctx, pump, width)
    plan = Plan(SRC, OUT)
    for rel in ["a/A_MMMM_wide_name_Scan1"] + [f"{'ab'[i % 2]}/PC-0{i}_Scan1" for i in range(1, 7)]:
        x = cand(rel, n=20)
        plan.add_candidate(x)
        plan.apply_header(x.key, header(), None, 64 * GB)
    t.bind_plan(plan)
    t.set_state("READY")
    settle(t, pump)
    before = widths(t)

    def step(change=None, state=None):
        if change is not None:
            t.refresh([change])
        if state is not None:
            t.set_state(state)
        settle(t, pump)
        assert widths(t) == before, (change, state)
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    assert plan.row(job.row_id).cand.folder_name == "A_MMMM_wide_name_Scan1"
    step(state="RUNNING")
    plan.set_run_text(job.row_id, "Stitching tiles 12 of 20 · about 4 min left · reading the tiles slowly")
    step(job.row_id)
    plan.finish(job.row_id, RowOutcome("done", "Done · 1 GB · 1 min"))
    step(job.row_id)
    assert t.tree.set("sec:new", "what") == "1 done"
    job = plan.claim_next(OUT, None)
    plan.finish(job.row_id, RowOutcome("failed", "Failed: the tiles of this slide couldn't be read (Input/output "
                                                 "error). Check the share, then tick it to try again."))
    step(job.row_id)
    step(state="STOPPING")                                             # "Won't be stitched (stopping)"
    plan.end_run()
    t.set_running(False)
    step(state="FINISHED")
    bold = ui_ctx.theme.fonts["YfsBold"]
    assert bold.measure(t.tree.item("sec:new", "text")) <= room(t, "sec:new", "#0")


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_the_rename_editor_shows_the_whole_suffix(ui_ctx, pump):
    """MW-25: the Entry is as wide as the Output cell and '.ome.tif' keeps its full width after it. It covers the
    start of the What cell, which is blank meanwhile: '.ome.tif ace the file from 3 Sep' read as one."""
    t = make(ui_ctx, pump, 1040)
    plan, c = every_status()
    t.bind_plan(plan)
    settle(t, pump)
    iid, what = t._iid[c["repl"].key], t.tree.set(t._iid[c["repl"].key], "what")
    t.rename(c["repl"].key)
    settle(t, pump)
    ed = t._edit
    suffix, entry = ed["suffix"], ed["entry"]
    assert str(suffix.cget("text")) == ".ome.tif" and suffix.winfo_width() >= suffix.winfo_reqwidth()
    assert entry.winfo_width() >= int(t.tree.column("output", "width")) - ui_ctx.theme.px(16)
    assert suffix.winfo_rootx() + suffix.winfo_width() <= t.tree.winfo_rootx() + t.tree.winfo_width()
    assert what and t.tree.set(iid, "what") == ""
    entry.event_generate("<Escape>")
    settle(t, pump)
    assert t._edit is None and t.tree.set(iid, "what") == what


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
@pytest.mark.parametrize("width", [760, 1040])
def test_the_rename_error_is_whole_inside_the_table(ui_ctx, pump, width):
    """MW-25: the message under the editor names the suggested file name, so it is read whole: it starts under the
    field as far as it fits, it goes above the row when that is the last one in view, and it covers the whole row
    it lies on, so that no end of that row's text reads as part of the name."""
    t = make(ui_ctx, pump, width)
    plan, c = every_status()
    t.bind_plan(plan)
    settle(t, pump)
    tree = t.tree
    for key, above in (("dupB", False), ("repl", True)):                  # repl: the last row, see() puts it at
        tree.yview_moveto(0)                                               # the bottom of the view
        settle(t, pump)
        t.rename(c[key].key)
        settle(t, pump)
        ed = t._edit
        ed["entry"].focus_force()
        ed["entry"].delete(0, "end")
        ed["entry"].insert(0, "bad:name?")
        ed["entry"].event_generate("<Return>")
        settle(t, pump)
        error, row_y = ed["error"], tree.bbox(t._iid[c[key].key], "output")[1]
        x, y, w, h = error.winfo_x(), error.winfo_y(), error.winfo_width(), error.winfo_height()
        assert str(error.cget("text")) == "Some characters can't be used in file names. Suggested name: bad_name_"
        assert error.winfo_ismapped() and w >= error.winfo_reqwidth() and h >= error.winfo_reqheight(), key
        assert 0 <= x and x + w <= tree.winfo_width() and 0 <= y and y + h <= tree.winfo_height(), (key, x, y, w, h)
        assert (y + h <= row_y) is above, key
        assert x <= 4 and x + w >= tree.winfo_width() - 4 and h >= tree.bbox(t._iid[c[key].key], "output")[3], key
        t._close_editor()


def test_a_resize_shortens_the_cells_again_and_separators_dont_drag(ui_ctx, pump):
    t = make(ui_ctx, pump, 1040)
    plan, c = every_status()
    t.bind_plan(plan)
    settle(t, pump)
    tree, px = t.tree, ui_ctx.theme.px
    iid = t._iid[c["incomp"].key]                                 # 'The existing file looks incomplete — will redo…'
    wide = tree.set(iid, "what")
    ui_ctx.root.geometry(f"{px(760)}x{px(560)}")
    settle(t, pump)
    narrow = tree.set(iid, "what")
    assert narrow != wide and ui_ctx.theme.G["ell"] in narrow
    assert ui_ctx.theme.fonts["YfsBody"].measure(narrow) <= room(t, iid, "what")
    widths = {col: int(tree.column(col, "width")) for col in ("#0",) + tuple(tree["displaycolumns"])}
    x = int(tree.column("#0", "width")) + 1                       # the separator after the Slide column
    y = 4
    assert tree.identify_region(x, y) == "separator"
    tree.event_generate("<Button-1>", x=x, y=y)
    tree.event_generate("<B1-Motion>", x=x + px(80), y=y)
    tree.event_generate("<ButtonRelease-1>", x=x + px(80), y=y)
    settle(t, pump)
    assert {col: int(tree.column(col, "width")) for col in widths} == widths
    press(t, pump, "<Down>")                                      # (the tree still works after it)
