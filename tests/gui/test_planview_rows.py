"""PlanTable rows and their menus (UD9, CORE_API §3): 'Try again' on every selected header error but never on a
bad tile (FS-7), the context menu from the keyboard, the statuses 'ready', 'unchecked' and 'busy', the note rows of
D1 and of unmounted folders, an icon and a label for every status and note kind, and the Mac delete key (PL-14)."""
from __future__ import annotations

import os

import pytest

from test_planview import OUT, SRC, Calls, cand, click, every_status, header, image_of, invoke, labels, labels_for, \
    press

from yfs.core import plan as core_plan
from yfs.core.model import Note, OutputIndex, OutputInfo, RowOutcome
from yfs.core.names import norm_key
from yfs.core.plan import Plan, row_text
from yfs.ui.planview import _NOTE_ICON, _NOTE_LABEL, STATUS_ICON, PlanTable, status_icon

GB = 1024 ** 3
PLANNING = ("new", "unfinished", "incomplete", "few_tiles", "sparse", "header_error", "unsupported", "existing",
            "existing_checking", "replace", "unchecked", "ready", "busy", "other_suffix", "unsuffixed", "legacy_all")


@pytest.fixture
def table(ui_ctx, pump):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(1040)}x{px(740)}+0+0")
    calls = Calls()
    t = PlanTable(root, ui_ctx, on_focus=lambda r: None, on_details=lambda r: None, on_fix_channels=lambda ids: None,
                  on_note_action=lambda a, r: calls.append(("note", a, r)),
                  on_changed=lambda: calls.append(("changed",)))
    t.pack(fill="both", expand=True)
    pump(root, 30)
    return t, calls


def info(name):
    return OutputInfo(name, 18 * GB, 1788448361675246000)


def outputs(outs=(), partials=(), ready=(), busy=()):
    keys = lambda names: {norm_key(n): info(n) for n in names}      # noqa: E731
    return OutputIndex(OUT, True, frozenset(norm_key(n) for n in outs), keys(outs), keys(partials), (), 10 ** 13,
                       "/data", "local", False, None, ready=keys(ready), busy=frozenset(norm_key(n) for n in busy))


def test_try_again_reads_every_selected_header_again_but_never_a_bad_tile(table, pump):
    """FS-7: one retry_header per selected row that a new read can help; a bad tile only says what is wrong."""
    t, calls = table
    p = Plan(SRC, OUT)
    keys = {}
    for name, error, retryable in (("A", "The location isn't answering.", True), ("B", "Permission denied.", True),
                                   ("C", "The file is cut off.", False), ("D", None, True)):
        c = cand(f"h/{name}_Scan1")
        keys[name] = c.key
        p.add_candidate(c)
        p.apply_header(c.key, None if error else header(), error, 64 * GB, retryable=retryable)
    t.bind_plan(p)
    t.set_state("READY")
    t.tree.selection_set([t._iid[k] for k in keys.values()])
    t.fill_menu(keys["A"])
    assert "Try again (2 slides)" in labels(t.menu)
    invoke(t.menu, "Try again (2 slides)")
    assert calls.of("note") == [("retry_header", keys["A"]), ("retry_header", keys["B"])]
    t.fill_menu(keys["C"])                                        # the bad tile: reading it again won't help
    assert not any(label.startswith("Try again") for label in labels(t.menu))
    assert "Copy it again" in row_text(p.row(keys["C"]))
    assert labels_for(t, keys["B"])[-1] == "Try again"            # one row selected: just that one
    invoke(t.menu, "Try again")
    assert calls.of("note")[-1] == ("retry_header", keys["B"])


def test_the_context_menu_opens_from_the_keyboard_too(table, ui_ctx, pump, monkeypatch):
    """Shift+F10, and the menu key where the keyboard has one, open the focused row's menu just under its name:
    Try again, Finalize and the searches don't need a mouse."""
    t, calls = table
    p, c = every_status()
    t.bind_plan(p)
    pump(t, 30)
    posted = []
    monkeypatch.setattr(t.menu, "tk_popup", lambda x, y: posted.append((x, y)))
    tree, herr = t.tree, t._iid[c["herr"].key]
    keys = ["<Shift-F10>"] + (["<KeyPress-Menu>"] if ui_ctx.theme.ws == "x11" else [])
    for key in keys:
        tree.selection_set(())
        tree.focus(herr)
        press(t, pump, key)
        x, y, w, h = tree.bbox(herr, "#0")
        assert posted.pop() == (tree.winfo_rootx() + x + t._pair_w, tree.winfo_rooty() + y + h), key
        assert tree.selection() == (herr,) and labels(t.menu)[-1] == "Try again", key
    invoke(t.menu, "Try again")
    assert calls.of("note") == [("retry_header", c["herr"].key)]
    tree.focus("sec:new")                                         # a section has no menu
    press(t, pump, "<Shift-F10>")
    assert not posted


def test_ready_unchecked_and_busy_rows(table, ui_ctx, pump):
    """'ready' is ticked and offers Finalize; 'unchecked' stays unticked in Already stitched; 'busy' can't be
    ticked while another stitcher writes it."""
    t, calls = table
    tx, icons = ui_ctx.theme.text, ui_ctx.icons
    p = Plan(SRC, OUT)
    for name in ("Ready", "Unchecked", "Busy", "Other"):
        p.add_candidate(cand(f"st/{name}_Scan1"))
    rows = {r.cand.folder_name.split("_")[0]: r for r in p.rows()}
    stem = {k: r.output_stem + ".ome.tif" for k, r in rows.items()}
    p.apply_output_index(outputs([stem["Unchecked"]], [stem["Busy"]], [stem["Ready"]], [stem["Busy"]]))
    p.apply_verification(os.path.join(OUT, stem["Unchecked"]), "unchecked", "Permission denied")
    t.bind_plan(p)
    t.set_state("READY")
    ready, unchecked, busy = (rows[k] for k in ("Ready", "Unchecked", "Busy"))
    assert (ready.status, unchecked.status, busy.status) == ("ready", "unchecked", "busy")
    assert image_of(t, t._iid[ready.row_id]) == str(icons.pair("cb_on", "ready"))
    assert t.tree.parent(t._iid[unchecked.row_id]) == "sec:existing"
    assert image_of(t, t._iid[unchecked.row_id]) == str(icons.pair("cb_off", "warn"))
    assert image_of(t, t._iid[busy.row_id]) == str(icons.pair(None, "wait"))         # no tick box
    for r in (ready, unchecked, busy):
        assert t._full[t._iid[r.row_id]][2][-1] == tx(row_text(r)) != r.status
    assert labels_for(t, ready.row_id)[-1] == "Finalize"
    invoke(t.menu, "Finalize")
    assert calls.of("note")[-1] == ("finalize", ready.row_id)
    assert "Finalize" not in labels_for(t, unchecked.row_id) + labels_for(t, busy.row_id)
    click(t, pump, t._iid[busy.row_id], dx=2)
    assert not p.row(busy.row_id).checked
    p.toggle([ready.row_id], False)
    assert status_icon(p.row(ready.row_id)) == "skip"                                # won't be stitched


def test_finalize_acts_on_every_selected_slide_that_waits_for_it(table, pump):
    t, calls = table
    p = Plan(SRC, OUT)
    for name in ("F1", "F2", "N"):
        p.add_candidate(cand(f"fin/{name}_Scan1"))
    t.bind_plan(p)
    p.begin_run()
    for _ in range(3):
        job = p.claim_next(OUT, None)
        p.finish(job.row_id, RowOutcome("finalize" if "/F" in job.folder else "done", "Finished, but in use"))
    p.end_run()
    t.set_state("FINISHED")
    ids = [r.row_id for r in p.rows()]
    t.tree.selection_set([t._iid[i] for i in ids])
    t.fill_menu(ids[0])
    invoke(t.menu, "Finalize (2 slides)")
    assert calls.of("note") == [("finalize", ids[0]), ("finalize", ids[1])]


def test_every_status_and_note_kind_has_an_icon_and_a_label():
    statuses = set(PLANNING) | core_plan.RUN_STATUSES | set(core_plan._RUN_TEXT)
    assert statuses <= set(STATUS_ICON), statuses - set(STATUS_ICON)
    kinds = (set(core_plan._NOTE_ORDER) | set(core_plan._AGGREGATED)) - {"leftovers"}
    assert kinds <= set(_NOTE_ICON), kinds - set(_NOTE_ICON)
    assert set(core_plan._AGGREGATED.values()) <= set(_NOTE_LABEL)            # rows that stand for many folders
    assert set(_NOTE_ICON.values()) <= {"info", "skip", "error"}


def test_note_rows_for_an_unmatched_output_and_for_folders_that_are_not_connected(table, ui_ctx, pump):
    """D1: an old 'Scan1.ome.tif' that two slides could own; folders skipped as not mounted (autofs) or as system
    folders (pseudo)."""
    t, calls = table
    p = Plan(SRC, OUT)
    for name in ("A1", "B1"):
        p.add_candidate(cand(f"lab/{name}/Scan1", f"{name}_Scan1"))
    p.apply_output_index(outputs(["Scan1.ome.tif"]))
    p.add_note(Note("not_mounted", SRC + "/shares", 2, "autofs", ("LabShareA", "LabShareB")))
    t.bind_plan(p)
    t.set_state("READY")
    rows = {r.note.kind: r for r in p.rows("note")}
    unmatched, mounts = t._iid[rows["unmatched"].row_id], t._iid[rows["not_mounted"].row_id]
    assert {r.output_stem for r in p.rows("new")} == {"A1_Scan1", "B1_Scan1"}
    assert t._full[unmatched][0] == "Scan1.ome.tif" and image_of(t, unmatched) == str(ui_ctx.icons.pair(None, "info"))
    assert "may belong to one of 2 slides named 'Scan1'" in t._full[unmatched][2][-1]
    assert labels_for(t, rows["unmatched"].row_id) == ["Show in folder", "Copy folder path"]
    invoke(t.menu, "Copy folder path")
    assert t.clipboard_get() == OUT
    assert t._full[mounts][0] == "Not connected" and image_of(t, mounts) == str(ui_ctx.icons.pair(None, "info"))
    assert labels_for(t, rows["not_mounted"].row_id) == ["Rescan"]
    invoke(t.menu, "Rescan")
    assert calls.of("note")[-1] == ("search_again", rows["not_mounted"].row_id)
    q = Plan(SRC, OUT)
    q.add_candidate(cand("x/S1"))
    q.add_note(Note("not_mounted", SRC, 1, "pseudo", ("proc",)))
    t.bind_plan(q)
    pseudo = q.rows("note")[0]
    assert t._full[t._iid[pseudo.row_id]][0] == "System folders" and status_icon(pseudo) == "skip"
    assert t.fill_menu(pseudo.row_id) == 0


def test_the_mac_delete_key_unticks_too(ui_ctx, pump, monkeypatch):
    """PL-14: Tk on macOS sends BackSpace for the key labelled delete; elsewhere BackSpace does nothing."""
    plans = []
    for ws in ("aqua", "x11"):
        monkeypatch.setattr(ui_ctx.theme, "ws", ws)
        t = PlanTable(ui_ctx.root, ui_ctx, on_focus=lambda r: None, on_details=lambda r: None,
                      on_fix_channels=lambda ids: None, on_note_action=lambda a, r: None, on_changed=lambda: None)
        t.pack(fill="both", expand=True)
        p, c = every_status()
        t.bind_plan(p)
        pump(t, 30)
        t.tree.selection_set((t._iid[c["new"].key], t._iid[c["chan"].key]))
        press(t, pump, "<BackSpace>")
        plans.append((p.row(c["new"].key).checked, p.row(c["chan"].key).checked))
        t.destroy()
    assert plans == [(False, False), (True, True)]
