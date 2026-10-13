"""PlanTable (FINAL_SPEC §2.3, §2.5, §5.5.15): sections and tri-state ticks, composite images and hit-testing,
keyboard map, inline rename, Select and context menus (incl. R26), details strip, welcome state, refresh speed."""
from __future__ import annotations

import os
import time

import pytest

from yfs.core.discovery import Discovery, DiscoveryOptions
from yfs.core.events import CandidateFound, ScanFinished, ScanNote
from yfs.core.fsservice import FsService
from yfs.core.model import Candidate, Note, OutputIndex, OutputInfo, RowOutcome, TileHeader
from yfs.core.names import norm_key
from yfs.core.places import MountTable
from yfs.core.plan import Plan, row_text
from yfs.ui.planview import COLUMNS, PlanTable, details_text, status_icon

SRC, OUT = "/data/lab/projectX", "/data/lab/stitched"
GB = 1024 ** 3
NAMES = ("DAPI", "Opal 520", "Opal 570", "Opal 690", "Autofluorescence")


def cand(rel, prefix=None, n=20, kind="component", suffix="component_data", mtime=None, step=900.0):
    folder = f"{SRC}/{rel}"
    prefix = prefix if prefix is not None else os.path.basename(folder)
    key = f"{norm_key(folder)}|*|legacy" if kind == "legacy_all" else f"{norm_key(folder)}|{prefix.casefold()}|{suffix}"
    cols = max(1, int(n ** 0.5 + 0.5))
    coords = () if kind == "legacy_all" else tuple((i % cols * step, i // cols * step * 0.75) for i in range(n))
    tiles = tuple(f"{prefix}_[{int(x)},{int(y)}]_{suffix}.tif" for x, y in coords)
    return Candidate(key, folder, rel, os.path.basename(folder), "" if kind == "legacy_all" else prefix,
                     "legacy" if kind == "legacy_all" else suffix, kind, tiles, coords, False, False,
                     rel.count("/") + 1, mtime)


def header(n=5, problem=None, names=NAMES, first=5 * 1396 * 1860 * 4):
    return TileHeader("t.tif", (n, 1396, 1860), "float32", n, 1396, 1860, 0.4984, 0.4984, True, tuple(names), problem,
                      (None,) * len(names), first)


def index(outputs=(), partials=()):
    info = lambda name: OutputInfo(name, 18 * GB, 1788448361675246000)     # noqa: E731
    return OutputIndex(OUT, True, frozenset(norm_key(n) for n in outputs), {norm_key(n): info(n) for n in outputs},
                       {norm_key(n): info(n) for n in partials}, (), 10 ** 13, "/data", "local", False, None)


KEYS = ("new", "chan", "recent", "few", "sparse", "herr", "big", "unfin", "incomp", "exist", "check", "repl", "dupA",
        "dupB", "seg", "uns", "legacy", "mono")


def every_status():
    """One plan holding every planning status, the flags and the note kinds (synthetic names)."""
    p, c, ram = Plan(SRC, OUT), {}, 64 * GB
    c.update(new=cand("b1/SlideA1_Scan1", n=173), chan=cand("b1/SlideA2_Scan1", n=178),
             recent=cand("b1/SlideA3_Scan1", n=120, mtime=time.time() - 90), few=cand("stamps/SlideA2_Scan1", n=3),
             sparse=cand("b1/SlideB7_Scan1", n=20, step=9000.0), herr=cand("b1/SlideB8_Scan1", n=64),
             big=cand("b2/SlideC1_Scan1", n=600), unfin=cand("b2/SlideC2_Scan1", n=204),
             incomp=cand("b2/SlideC3_Scan1", n=237), exist=cand("b2/SlideC4_Scan1", n=246),
             check=cand("b2/SlideC5_Scan1", n=150), repl=cand("b2/SlideC6_Scan1", n=96),
             dupA=cand("dups/runA/Slide9", "Slide9_Scan1", 40), dupB=cand("dups/runB/Slide9", "Slide9_Scan1", 38),
             seg=cand("inform/S1_Scan1", "S1_Scan1", 4, "other_suffix", "binary_seg_maps"),
             uns=cand("misc/Slide_X", "Slide_X", 12, "unsuffixed", ""), legacy=cand("misc/WSIs", kind="legacy_all"),
             mono=cand("b3/SlideD1_Scan1", n=30))
    for k in KEYS:
        p.add_candidate(c[k])
    for k in ("new", "recent", "few", "sparse", "unfin", "incomp", "exist", "check", "repl", "dupA", "dupB", "uns"):
        p.apply_header(c[k].key, header(), None, ram)
    p.apply_header(c["chan"].key, header(names=("DAPI",) * 5, problem="duplicate"), None, ram)
    p.apply_header(c["big"].key, header(), None, 8 * GB)
    p.apply_header(c["herr"].key, None, "Permission denied.", ram)
    for k in ("seg", "mono"):
        p.apply_header(c[k].key, header(1, None, ("Nucleus",)), None, ram)
    stems = {k: p.row(c[k].key).output_stem + ".ome.tif" for k in ("unfin", "incomp", "exist", "check", "repl")}
    p.apply_output_index(index([stems[k] for k in ("incomp", "exist", "check", "repl")], [stems["unfin"]]))
    for k, verdict in (("incomp", "incomplete"), ("exist", "valid"), ("repl", "valid")):
        p.apply_verification(os.path.join(OUT, stems[k]), verdict, "")
    p.toggle([c["repl"].key, c["legacy"].key], True)
    for n in (Note("outputs", SRC + "/b2", 2, names=("Z1.ome.tif", "Z2.ome.tif")), Note("raw_scan", SRC + "/raw/a"),
              Note("wide_pruned", SRC + "/patches", 1083), Note("not_searched_depth", SRC + "/deep/a/b", 12),
              Note("no_images", SRC + "/analysis"), Note("no_access", SRC + "/locked"),
              Note("unresponsive", SRC + "/slow")):
        p.add_note(n)
    return p, c


EXPECT = {   # key: (section, tick box, status icon, tags)
    "new": ("new", "cb_on", "ready", ()), "chan": ("new", "cb_on", "warn", ("warn",)),
    "recent": ("new", "cb_on", "warn", ("warn",)), "few": ("new", "cb_off", "skip", ("dim",)),
    "sparse": ("new", "cb_off", "skip", ("dim",)), "herr": ("new", "cb_off", "error", ("error",)),
    "big": ("new", "cb_on", "ready", ()), "unfin": ("new", "cb_on", "warn", ("warn",)),
    "incomp": ("new", "cb_on", "warn", ("warn",)), "exist": ("existing", "cb_off", "ok", ("dim",)),
    "check": ("existing", "cb_off", "wait", ("dim",)), "repl": ("existing", "cb_on", "warn", ("warn",)),
    "seg": ("excluded", None, "skip", ("dim",)), "uns": ("excluded", "cb_off", "info", ("dim",)),
    "legacy": ("excluded", "cb_on", "info", ()), "mono": ("new", None, "skip", ("dim",))}


class Calls(list):
    def of(self, kind):
        return [c[1:] for c in self if c[0] == kind]


@pytest.fixture
def table(ui_ctx, pump):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(1040)}x{px(740)}+0+0")
    calls = Calls()
    t = PlanTable(root, ui_ctx, on_focus=lambda r: calls.append(("focus", r)),
                  on_details=lambda r: calls.append(("details", r)),
                  on_fix_channels=lambda ids: calls.append(("fix", list(ids))),
                  on_note_action=lambda a, r: calls.append(("note", a, r)), on_changed=lambda: calls.append(("changed",)))
    t.pack(fill="both", expand=True)
    pump(root, 30)
    return t, calls


@pytest.fixture
def bound(table, pump):
    t, calls = table
    plan, c = every_status()
    t.bind_plan(plan)
    pump(t, 30)
    return t, calls, plan, c


def image_of(t, iid):
    return str(t.tree.item(iid, "image")[0]) if t.tree.item(iid, "image") else ""


def full(t, iid, column="#0"):
    """The full text of a cell (the table may show it shortened, MW-10)."""
    d = t._full[iid]
    return d[0] if column == "#0" else d[2][COLUMNS.index(column)]


def shows(t, iid, column, text):
    """The cell holds `text`, whole or shortened with the ellipsis."""
    shown = t.tree.item(iid, "text") if column == "#0" else t.tree.set(iid, column)
    return full(t, iid, column) == text and (shown == text or t.ctx.theme.G["ell"] in shown)


def click(t, pump, iid, dx=None, column=None, double=False):
    """A click on row `iid`: dx px into the Slide cell (the composite image starts at 0), or in `column`.
    Two generated presses at one spot make Tk report the second as <Double-Button-1>, as for a real double-click."""
    tree = t.tree
    tree.see(iid)
    pump(t, 20)
    x, y, w, h = tree.bbox(iid, column or "#0")
    x += dx if dx is not None else w // 2
    for _ in range(2 if double else 1):
        tree.event_generate("<Button-1>", x=x, y=y + h // 2)
        tree.event_generate("<ButtonRelease-1>", x=x, y=y + h // 2)
    pump(t, 20)


def press(t, pump, key):
    t.tree.focus_force()
    pump(t, 20)
    t.tree.event_generate(key)
    pump(t, 20)


# ------------------------------------------------------------------------------------------------ rows and sections
def test_every_status_shows_its_tick_box_icon_text_and_tags(bound, ui_ctx):
    t, _, plan, c = bound
    tree, icons, text = t.tree, ui_ctx.icons, ui_ctx.theme.text
    for key, (section, box, icon, tags) in EXPECT.items():
        r, iid = plan.row(c[key].key), t._iid[c[key].key]
        assert tree.parent(iid) == f"sec:{section}", key
        assert image_of(t, iid) == str(icons.pair(box, icon)), key
        assert tuple(tree.item(iid, "tags")) == tags, key
        assert shows(t, iid, "what", text(row_text(r))) and shows(t, iid, "output", text(r.output_stem + ".ome.tif"))
    assert tree.set(t._iid[c["chan"].key], "what") == "Check channel names (DAPI ×5)"
    assert tree.set(t._iid[c["exist"].key], "size") == "18.0 GB" and tree.set(t._iid[c["new"].key], "size").startswith("~")
    assert tree.item(t._iid[c["dupB"].key], "text") == "Slide9 (tiles: Slide9_Scan1)"
    assert tree.set(t._iid[c["dupB"].key], "output") == "Slide9__runB.ome.tif"
    notes = [i for i in tree.get_children("sec:excluded") if t._rid[i].startswith("note|")]
    assert len(notes) == 7 and all(tree.item(i, "tags") == ("note",) for i in notes)
    by_kind = {plan.row(t._rid[i]).note.kind: i for i in notes}
    assert image_of(t, by_kind["no_access"]) == str(icons.pair(None, "error"))
    assert image_of(t, by_kind["no_images"]) == str(icons.pair(None, "skip"))
    assert tree.item(by_kind["not_searched_depth"], "text") == "Deeper folders"
    assert tree.item(by_kind["wide_pruned"], "text") == "patches"
    assert all(tree.set(i, "what") == text(row_text(plan.row(t._rid[i]))) or full(t, i, "what") ==
               text(row_text(plan.row(t._rid[i]))) for i in notes)


def test_sections_counts_tristate_box_and_default_open_state(bound, ui_ctx, pump):
    t, _, plan, c = bound
    tree, icons = t.tree, ui_ctx.icons
    assert tree.get_children("") == ("sec:new", "sec:existing", "sec:excluded")
    new = plan.rows("new")
    assert tree.item("sec:new", "text") == f"New slides  ({sum(r.checked for r in new)} of {len(new)} ticked)"
    assert image_of(t, "sec:new") == str(icons.pair("chev_down", "cb_mixed"))
    assert tree.item("sec:existing", "text") == "Already stitched  (3)" and tree.item("sec:existing", "open")
    assert tree.set("sec:existing", "what") == "Skipped unless you tick them"
    assert image_of(t, "sec:existing") == str(icons.pair("chev_down", None))
    assert tree.item("sec:excluded", "text") == f"Not included  ({len(plan.rows('excluded'))})"
    assert not tree.item("sec:excluded", "open")
    plan.select("none")
    t.refresh([])
    assert image_of(t, "sec:new") == str(icons.pair("chev_down", "cb_off"))
    plan.select("all_new")
    t.refresh(None)
    assert image_of(t, "sec:new") == str(icons.pair("chev_down", "cb_on"))


def test_existing_section_opens_only_when_short_or_nothing_is_new(table, ui_ctx):
    t, _ = table
    p = Plan(SRC, OUT)
    olds = [cand(f"old/S{i}") for i in range(6)]
    for x in olds:
        p.add_candidate(x)
    p.apply_output_index(index([f"S{i}.ome.tif" for i in range(6)]))
    t.bind_plan(p)
    assert t.tree.item("sec:new", "text") == "New slides  (0)" and t.tree.set("sec:new", "what") == \
        "Nothing new to stitch" and t.tree.item("sec:existing", "open")       # nothing new: open even with 6 rows
    p.add_candidate(cand("new/N1"))
    t.refresh(None)
    assert not t.tree.item("sec:existing", "open")                            # more than 5 and something new
    assert t.tree.set("sec:existing", "what") == ui_ctx.theme.text("Skipped unless you tick them  (S0, S1, S2, …)")


# ------------------------------------------------------------------------------------------------ pointer
def test_clicks_on_the_composite_images_follow_the_verified_hit_test(bound, ui_ctx, pump):
    t, calls, plan, c = bound
    icons, tree = ui_ctx.icons, t.tree
    slot1 = icons.size + icons.gap + 2
    rid = c["new"].key
    click(t, pump, t._iid[rid], dx=2)                                         # tick box
    assert not plan.row(rid).checked and plan.row(rid).user_touched and calls.of("changed")
    click(t, pump, t._iid[rid], dx=slot1)                                    # status icon: toggles too
    assert plan.row(rid).checked
    click(t, pump, t._iid[c["chan"].key], dx=2 * icons.size + 3 * icons.gap + 20)   # the name: select only
    assert plan.row(c["chan"].key).checked and tree.selection() == (t._iid[c["chan"].key],)
    assert calls.of("focus")[-1] == (c["chan"].key,)
    click(t, pump, "sec:new", dx=slot1)                                       # section tick box: all rows
    assert all(r.checked for r in plan.rows("new") if r.checkable)
    click(t, pump, "sec:new", dx=slot1)
    assert not any(r.checked for r in plan.rows("new"))
    click(t, pump, "sec:new", dx=2)                                           # chevron: close
    assert not tree.item("sec:new", "open") and image_of(t, "sec:new").startswith(str(icons.pair("chev_right", "cb_off")))
    click(t, pump, "sec:new", dx=200)                                         # the label opens it again
    assert tree.item("sec:new", "open")
    click(t, pump, t._iid[c["mono"].key], dx=2)                              # a row without a tick box
    assert not plan.row(c["mono"].key).checked


def test_double_click_opens_details_or_renames_on_the_output_cell(bound, pump):
    t, calls, plan, c = bound
    click(t, pump, t._iid[c["new"].key], column="what", double=True)
    assert calls.of("details") == [(c["new"].key,)] and t._edit is None
    click(t, pump, t._iid[c["new"].key], column="output", double=True)
    assert t._edit is not None and t._edit["row"] == c["new"].key
    assert t.tree.item("sec:existing", "open")
    click(t, pump, "sec:existing", dx=200, double=True)                      # the first click closes it; the
    assert not t.tree.item("sec:existing", "open")                           # second does not reopen it
    icons = t.ctx.icons
    click(t, pump, "sec:existing", dx=icons.size + icons.gap + 2, double=True)   # nor on its empty box slot
    assert t.tree.item("sec:existing", "open")


# ------------------------------------------------------------------------------------------------ keyboard
def test_keyboard_space_delete_enter_f2_and_select_all(bound, pump):
    t, calls, plan, c = bound
    tree = t.tree
    a, b = t._iid[c["new"].key], t._iid[c["chan"].key]
    tree.selection_set((a, b))
    tree.focus(a)
    press(t, pump, "<space>")
    assert not plan.row(c["new"].key).checked and not plan.row(c["chan"].key).checked
    press(t, pump, "<space>")
    assert plan.row(c["new"].key).checked and plan.row(c["chan"].key).checked
    press(t, pump, "<Delete>")
    assert not plan.row(c["new"].key).checked and not plan.row(c["chan"].key).checked
    press(t, pump, "<Return>")
    assert calls.of("details")[-1] == (c["new"].key,)
    press(t, pump, "<F2>")
    assert t._edit is not None and t._edit["row"] == c["new"].key
    t._close_editor()
    press(t, pump, "<Control-a>")
    shown = [i for s in ("sec:new", "sec:existing") for i in tree.get_children(s)]
    assert set(tree.selection()) == set(shown)                               # Not included is closed
    tree.selection_set(())
    tree.focus("sec:new")
    before = [r.checked for r in plan.rows("new")]
    press(t, pump, "<space>")                                                 # the section: toggle its rows
    assert [r.checked for r in plan.rows("new")] != before
    tree.focus("sec:excluded")
    press(t, pump, "<Return>")
    assert tree.item("sec:excluded", "open")


def test_the_arrow_keys_work_as_soon_as_the_table_has_the_keyboard(bound, pump):
    """Tk's arrow keys do nothing without a focused item: tabbing in focuses the first one, and the row of a tick
    box clicked is the one the keys go on from."""
    t, _, plan, c = bound
    tree = t.tree
    assert tree.focus() == ""
    press(t, pump, "<Down>")                                                  # (press gives the tree the keyboard)
    first = tree.get_children("sec:new")[0]
    assert tree.focus() == first and tree.selection() == (first,)
    press(t, pump, "<space>")
    assert not plan.row(t._rid[first]).checked
    big = t._iid[c["big"].key]
    click(t, pump, big, dx=2)
    assert tree.focus() == big and not plan.row(c["big"].key).checked
    press(t, pump, "<Down>")
    assert tree.focus() == tree.next(big)


def test_space_and_delete_never_change_a_slide_out_of_sight(bound, pump):
    """Space and Delete act on what is selected in sight (MW-2). A click on a section's label selects the section, so
    Space then opens or closes it. Tk's Shift+click range selects the rows of a closed section it runs through, and
    a selected row can move into a closed section: such rows are left alone. Ticking an 'Already stitched' row out
    of sight would replace its file."""
    t, _, plan, c = bound
    tree = t.tree
    click(t, pump, t._iid[c["new"].key], column="what")                       # a slide is selected, then another
    click(t, pump, "sec:existing", dx=200)                                    # section's label closes that one
    for key, is_open in (("<space>", True), ("<Delete>", True), ("<space>", False)):
        press(t, pump, key)
        assert plan.row(c["new"].key).checked and bool(tree.item("sec:existing", "open")) is is_open, key
    t._set_open("excluded", True)
    click(t, pump, t._iid[c["few"].key], column="what")                       # the last new slide, then (over the
    x, y, w, h = tree.bbox(t._iid[c["uns"].key], "what")                       # closed section) a row of Not included
    tree.event_generate("<Shift-Button-1>", x=x + w // 2, y=y + h // 2, state=1)
    tree.event_generate("<ButtonRelease-1>", x=x + w // 2, y=y + h // 2, state=1)
    pump(t, 20)
    hidden = [t._rid[i] for i in tree.selection() if tree.parent(i) == "sec:existing"]
    ticks = {rid: plan.row(rid).checked for rid in hidden}
    assert len(hidden) == 3 and not all(ticks.values())
    press(t, pump, "<space>")
    assert plan.row(c["uns"].key).checked and {rid: plan.row(rid).checked for rid in hidden} == ticks
    press(t, pump, "<Delete>")
    assert not plan.row(c["uns"].key).checked and {rid: plan.row(rid).checked for rid in hidden} == ticks
    click(t, pump, t._iid[c["new"].key], column="what")                       # selected, then its file appears
    names = [plan.row(c[k].key).output_stem + ".ome.tif" for k in ("new", "incomp", "exist", "check", "repl")]
    plan.apply_output_index(index(names))
    t.refresh(None)
    iid = t._iid[c["new"].key]
    assert tree.parent(iid) == "sec:existing" and tree.selection() == (iid,) and not plan.row(c["new"].key).checked
    press(t, pump, "<space>")
    assert not plan.row(c["new"].key).checked


# ------------------------------------------------------------------------------------------------ inline rename
def test_inline_rename_validates_shows_errors_and_commits(bound, pump):
    t, calls, plan, c = bound
    rid, other = c["new"].key, plan.row(c["chan"].key).output_stem
    t.rename(rid)
    pump(t, 20)
    ed = t._edit
    entry = ed["entry"]
    assert entry.get() == "SlideA1_Scan1" and entry.selection_present() and str(entry.focus_get()) == str(entry)
    entry.delete(0, "end")
    entry.insert(0, other)
    entry.event_generate("<Return>")
    pump(t, 20)
    assert t._edit is ed and ed["error"].cget("text") == "Another slide already uses this name."
    assert ed["error"].winfo_ismapped()
    entry.delete(0, "end")
    entry.insert(0, "bad:name?")
    entry.event_generate("<Return>")
    pump(t, 20)
    assert entry.get() == "bad_name_" and "can't be used" in str(ed["error"].cget("text"))
    entry.delete(0, "end")
    entry.insert(0, "My slide.ome.tif")
    entry.event_generate("<Return>")
    pump(t, 20)
    assert t._edit is None and plan.row(rid).output_stem == "My slide" and plan.row(rid).output_edited
    assert t.tree.set(t._iid[rid], "output") == "My slide.ome.tif" and calls.of("changed")
    t.rename(rid)
    pump(t, 20)
    t._edit["entry"].delete(0, "end")
    t._edit["entry"].event_generate("<Return>")                               # empty: the automatic name again
    pump(t, 20)
    assert plan.row(rid).output_stem == "SlideA1_Scan1" and not plan.row(rid).output_edited
    t.rename(rid)
    pump(t, 20)
    t._edit["entry"].insert(0, "zzz")
    t._edit["entry"].event_generate("<Escape>")
    pump(t, 20)
    assert t._edit is None and plan.row(rid).output_stem == "SlideA1_Scan1"


def test_rename_to_an_existing_file_moves_the_row_to_already_stitched(bound, pump):
    t, _, plan, c = bound
    rid = c["new"].key
    t.rename(rid)
    pump(t, 20)
    entry = t._edit["entry"]
    entry.delete(0, "end")
    entry.insert(0, "Z1")
    plan.apply_output_index(index(["Z1.ome.tif", plan.row(c["exist"].key).output_stem + ".ome.tif"]))
    entry.event_generate("<Return>")
    pump(t, 20)
    assert plan.row(rid).section == "existing" and t.tree.parent(t._iid[rid]) == "sec:existing"


def test_f2_on_a_slide_in_a_closed_section_opens_the_section_for_good(bound, ui_ctx, pump):
    """A focused slide can end up in a closed section (here its file appears). F2 opens the section as a click
    does, so the chevron agrees and the next refresh doesn't close it over the editor."""
    t, _, plan, c = bound
    tree = t.tree
    t._set_open("existing", False)
    iid = t._iid[c["new"].key]
    click(t, pump, iid, column="what")
    names = [plan.row(c[k].key).output_stem + ".ome.tif" for k in ("new", "incomp", "exist", "check", "repl")]
    plan.apply_output_index(index(names))
    t.refresh(None)
    assert tree.parent(iid) == "sec:existing" and tree.focus() == iid and not tree.item("sec:existing", "open")
    press(t, pump, "<F2>")
    t.refresh(None)
    pump(t, 30)
    assert t._edit is not None and t._edit["row"] == c["new"].key and t._edit["frame"].winfo_ismapped()
    assert tree.item("sec:existing", "open")
    assert image_of(t, "sec:existing") == str(ui_ctx.icons.pair("chev_down", None))           # the chevron agrees


def test_editor_follows_its_row_when_the_table_scrolls(bound, pump):
    t, _, plan, c = bound
    t.ctx.root.geometry(f"{t.ctx.theme.px(1040)}x{t.ctx.theme.px(420)}")
    t.rename(c["new"].key)
    pump(t, 50)
    frame = t._edit["frame"]
    y0 = frame.winfo_y()
    t.tree.yview_scroll(1, "units")
    pump(t, 30)
    assert not frame.winfo_ismapped() or frame.winfo_y() != y0


# ------------------------------------------------------------------------------------------------ menus
def labels(menu):
    end = menu.index("end")
    return [] if end is None else [menu.entrycget(i, "label") for i in range(end + 1)
                                   if menu.type(i) == "command"]


def invoke(menu, label):
    end = menu.index("end")
    menu.invoke(next(i for i in range(end + 1) if menu.type(i) == "command" and menu.entrycget(i, "label") == label))


def test_select_menu_items_depend_on_the_plan_and_on_running(bound, ui_ctx):
    t, calls, plan, c = bound
    text = ui_ctx.theme.text
    t._fill_select()
    assert labels(t.select_menu) == ["Tick all new slides", "Untick all", "Only slides not stitched yet",
                                     "Invert ticks", "Untick smaller copies"]      # duplicates exist; no run yet
    invoke(t.select_menu, "Untick all")
    assert not plan.ticked() and calls.of("changed")
    invoke(t.select_menu, "Only slides not stitched yet")
    assert {r.row_id for r in plan.ticked()} == {r.row_id for r in plan.rows("new") if r.checkable}
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    t.set_running(True)
    t._fill_select()
    assert labels(t.select_menu)[-1] == "Untick all waiting"
    assert all(t.select_menu.entrycget(i, "state") == "disabled" for i in range(5))
    invoke(t.select_menu, "Untick all waiting")
    assert not [r for r in plan.rows() if r.status == "waiting" and r.checked] and plan.row(job.row_id).checked
    t._fill_select()                                                          # nothing waits any more
    assert t.select_menu.entrycget(t.select_menu.index("end"), "state") == "disabled"
    plan.finish(job.row_id, RowOutcome("failed", "Failed: test"))
    plan.end_run()
    t.set_running(False)
    t._fill_select()
    assert text("Tick failed slides") in labels(t.select_menu)


def test_context_menu_offers_fix_channel_names_only_for_channel_problems(bound, pump):
    """D3 / R26: the engine asks for names only when they are suspicious, so only those rows offer the fix."""
    t, calls, plan, c = bound
    fix = t.ctx.theme.text("Fix channel names…")
    assert fix in labels_for(t, c["chan"].key)
    for key in ("new", "few", "exist", "herr", "seg", "legacy"):
        assert not any(x.startswith("Fix channel") for x in labels_for(t, c[key].key)), key
    t.tree.selection_set((t._iid[c["chan"].key], t._iid[c["new"].key]))
    t.fill_menu(c["chan"].key)
    invoke(t.menu, fix)
    assert calls.of("fix") == [([c["chan"].key],)]                          # rows without a problem are left out


def labels_for(t, rid):
    t.tree.selection_set((t._iid[rid],))
    t.fill_menu(rid)
    return labels(t.menu)


def test_context_menu_actions_per_row_kind(bound, ui_ctx, pump):
    t, calls, plan, c = bound
    tx = ui_ctx.theme.text
    assert labels_for(t, c["new"].key) == ["Open folder", "Copy folder path", tx("Slide details…"),
                                           tx("Rename output…")]
    assert labels_for(t, c["herr"].key)[-1] == "Try again"
    invoke(t.menu, "Try again")
    assert calls.of("note")[-1] == ("retry_header", c["herr"].key)
    notes = {plan.row(r).note.kind: r for r in t._iid if r.startswith("note|")}
    for kind, label, action in (("not_searched_depth", "Search deeper", "search_deeper"),
                                ("wide_pruned", "Search these folders too", "search_wide"),
                                ("unresponsive", "Search this folder again", "search_again")):
        assert labels_for(t, notes[kind])[-1] == label
        invoke(t.menu, label)
        assert calls.of("note")[-1] == (action, notes[kind])
    assert labels_for(t, notes["not_searched_depth"]) == ["Search deeper"]    # aggregated: no single folder
    labels_for(t, c["new"].key)
    invoke(t.menu, "Copy folder path")
    assert t.clipboard_get() == c["new"].folder
    invoke(t.menu, tx("Slide details…"))
    assert calls.of("details")[-1] == (c["new"].key,)
    invoke(t.menu, tx("Rename output…"))
    pump(t, 30)
    assert t._edit is not None


def test_during_a_run_waiting_rows_can_only_be_unticked(bound, ui_ctx, pump):
    t, calls, plan, c = bound
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    t.set_running(True)
    t.refresh(None)
    running, waiting = job.row_id, c["chan"].key
    assert t.tree.item(t._iid[running], "tags") == ("running",)
    assert image_of(t, t._iid[running]) == str(t._image("cb_on", "run", 0))         # can't change: dimmed
    started = labels_for(t, running)
    assert ui_ctx.theme.text("Rename output…") in started
    assert t.menu.entrycget(started.index(ui_ctx.theme.text("Rename output…")), "state") == "disabled"
    assert labels_for(t, waiting)[-1] == ui_ctx.theme.text("Don’t stitch this slide")
    invoke(t.menu, ui_ctx.theme.text("Don’t stitch this slide"))
    assert not plan.row(waiting).checked and t.tree.set(t._iid[waiting], "what") == "Won't be stitched"
    assert image_of(t, t._iid[waiting]) == str(t._image("cb_off", "skip", 0))      # can't be ticked again
    click(t, pump, t._iid[running], dx=2)                                    # a started row keeps its tick
    assert plan.row(running).checked
    t.rename(running)
    assert t._edit is None


# ------------------------------------------------------------------------------------------------ details strip
def test_details_strip_describes_the_focused_row(bound, ui_ctx, pump):
    t, _, plan, c = bound
    r = plan.row(c["dupB"].key)
    text = details_text(r)
    assert text.startswith("Slide9 · New file · 38 tiles · Renamed so nothing is overwritten (…__runB)")
    assert "5 channels (DAPI, Opal 520, Opal 570, …)" in text and "0.498 µm/px" in text and "fits in memory" in text
    assert "tiles named Slide9_Scan1" in text and text.endswith("in dups/runB")
    chan = details_text(plan.row(c["chan"].key))
    assert chan.startswith("SlideA2_Scan1 · Check channel names (DAPI ×5) · 178 tiles") and "5 channels ·" in chan
    assert chan.count("DAPI ×5") == 1                                         # the flag already names them
    herr = details_text(plan.row(c["herr"].key))
    assert herr.startswith("SlideB8_Scan1 · Can't read the first tile (Permission denied). Right-click") and \
        herr.lower().count("read the first tile") == 1
    t.tree.focus(t._iid[c["dupB"].key])
    t.tree.selection_set(t._iid[c["dupB"].key])
    pump(t, 30)
    shown = t.details.cget("text")
    ell = ui_ctx.theme.G["ell"]
    assert shown == ui_ctx.theme.text(text) or (shown.endswith(ell) and ui_ctx.theme.text(text).startswith(shown[:-len(ell)]))
    assert ui_ctx.theme.fonts["YfsSmall"].measure(shown) <= t.details.winfo_width()
    t.set_compact(True)
    assert not t.details.winfo_manager()
    t.set_compact(False)
    assert t.details.winfo_manager()


def test_status_icon_of_a_ticked_new_row_follows_its_top_warning(bound):
    _, _, plan, c = bound
    assert status_icon(plan.row(c["recent"].key)) == "warn"
    plan.toggle([c["recent"].key], False)
    assert status_icon(plan.row(c["recent"].key)) == "skip"


# ------------------------------------------------------------------------------------------------ welcome, width
def test_reveal_opens_and_focuses_the_requested_note_rows(bound, pump):
    table, _calls, plan, _c = bound
    rows = [r.row_id for r in plan.rows("note")]
    assert rows
    table._set_open("excluded", False)
    table.winfo_toplevel().focus_force()
    table.reveal(["gone"] + rows, "excluded")
    pump(table, 30)
    assert [table._rid[i] for i in table.tree.selection()] == rows
    assert table.focused_row() == rows[0] and table.focus_get() is table.tree
    assert table.tree.bbox(table._iid[rows[0]])


def test_welcome_state_and_its_buttons(table, pump):
    t, calls = table
    seen = []
    t.show_welcome(True, lambda: seen.append("choose"), lambda: seen.append("dismiss"))
    pump(t, 30)
    texts = _texts(t._welcome)
    assert "Welcome to your FAVOURITE Stitcher!" in texts and not t.tree.winfo_ismapped()
    assert any("you don't have to pay £5/hr." in x for x in texts)
    assert not t.details.winfo_manager()
    _button(t._welcome, t.ctx.theme.text("Choose tile folder…")).invoke()
    _button(t._welcome, t.ctx.theme.text("Don’t show this again")).invoke()
    pump(t, 30)
    assert seen == ["choose", "dismiss"]
    assert "Choose the folder with your tiles to see the slides it contains." in _texts(t._welcome)
    t.bind_plan(every_status()[0])
    pump(t, 30)
    assert t._welcome is None and t.tree.winfo_ismapped() and t.details.winfo_manager()


def _texts(w):
    out = []
    for ch in w.winfo_children():
        if ch.winfo_class() in ("TLabel", "TButton") and str(ch.cget("text")):
            out.append(str(ch.cget("text")))
        out += _texts(ch)
    return out


def _button(w, text):
    for ch in w.winfo_children():
        if ch.winfo_class() == "TButton" and str(ch.cget("text")) == text:
            return ch
        found = _button(ch, text)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_width_rules_hide_size_and_help_and_folder_shows_for_several_parents(bound, ui_ctx, pump):
    t, _, plan, c = bound
    px, root = ui_ctx.theme.px, ui_ctx.root
    assert t.tree["displaycolumns"] == ("tiles", "size", "gap", "folder", "output", "what") and t.help.winfo_manager()
    root.geometry(f"{px(860)}x{px(560)}")
    pump(t, 80)
    assert t.tree["displaycolumns"] == ("tiles", "gap", "folder", "output", "what") and t.help.winfo_manager()
    root.geometry(f"{px(760)}x{px(560)}")
    pump(t, 80)
    assert not t.help.winfo_manager()
    one = Plan(SRC, OUT)
    one.add_candidate(cand("only/S1"))
    one.add_candidate(cand("only/S2"))
    t.bind_plan(one)
    assert "folder" not in t.tree["displaycolumns"]
    assert t.min_height() < root.winfo_height()


# ------------------------------------------------------------------------------------------------ real plans, speed
def test_a_plan_from_discovery_on_the_synthetic_tree(table, synth_tree, collector, pump):
    t, _ = table
    fs = FsService(collector, MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw"))
    try:
        Discovery(1, str(synth_tree), fs, collector, DiscoveryOptions()).start()
        collector.wait_type(ScanFinished, timeout=30)
    finally:
        fs.shutdown()

    def build():
        p = Plan(str(synth_tree), str(synth_tree / "stitched"))
        for e in collector.events:
            if isinstance(e, CandidateFound):
                p.add_candidate(e.candidate)
            elif isinstance(e, ScanNote):
                p.add_note(e.note)
        return p
    p = build()
    t.bind_plan(p)
    pump(t, 30)
    new = [full(t, i) for i in t.tree.get_children("sec:new")]
    assert {"S1_Scan1", "S2_Scan1", "SlideA1_Scan1", "SlideA2_Scan1", "SusSlide_Scan1"} <= set(new)
    assert "R1-02 (tiles: R1-02_Scan1)" in new
    excluded = [full(t, i) for i in t.tree.get_children("sec:excluded")]
    assert "S1_Scan1 (binary_seg_maps)" in excluded and "Deeper folders" in excluded
    keep = t._iid[next(r.row_id for r in p.rows("new") if r.cand.prefix == "S2_Scan1")]
    t.tree.selection_set(keep)
    t.tree.focus(keep)
    t.bind_plan(build())                                                     # a rescan keeps selection and focus
    assert t.tree.selection() == (keep,) and t.tree.focus() == keep


def test_refresh_stays_fast_with_1200_rows(table, pump, capsys):
    t, _ = table
    p = Plan(SRC, OUT)
    for i in range(1200):
        p.add_candidate(cand(f"batch{i // 40:02d}/Slide{i:04d}_Scan1", n=12))
    t0 = time.perf_counter()
    t.bind_plan(p)
    full = time.perf_counter() - t0
    pump(t, 30)
    ids = [r.row_id for r in p.rows()][:10]
    times = []
    for _ in range(5):
        changed = p.toggle(ids)
        t0 = time.perf_counter()
        t.refresh(changed)
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    t.refresh(None)
    again = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[planview] 1200 rows: first build {full * 1000:.0f} ms, 10-row refresh "
              f"{min(times) * 1000:.1f}-{max(times) * 1000:.1f} ms, full refresh {again * 1000:.0f} ms")
    assert len(t.tree.get_children("sec:new")) == 1200
    assert max(times) < 0.1 and again < 0.5 and full < 2.0
    assert all(t.tree.item(t._iid[i], "image")[0] == str(t.ctx.icons.pair("cb_off", "skip")) for i in ids)
