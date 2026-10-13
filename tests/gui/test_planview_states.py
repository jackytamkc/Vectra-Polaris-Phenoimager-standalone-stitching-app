"""PlanTable follows the main window's state (UD9): the toolbar hint and the Select button (MW-16), dimmed tick boxes
on rows that can't change during a run (MW-16), a section box that never unticks the queue (MW-2), waiting rows
once a stop is asked for (MW-19), searches that can't start during a run (MW-21) and the New-slides count of the
slides Start would run (MW-7)."""
from __future__ import annotations

import pytest

from test_planview import OUT, SRC, Calls, cand, click, every_status, header, image_of, labels_for, press

from yfs.core.model import RowOutcome
from yfs.core.plan import Plan
from yfs.ui.planview import PlanTable, status_icon

GB = 1024 ** 3


@pytest.fixture
def table(ui_ctx, pump):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(1040)}x{px(740)}+0+0")
    calls = Calls()
    t = PlanTable(root, ui_ctx, on_focus=lambda r: None, on_details=lambda r: calls.append(("details", r)),
                  on_fix_channels=lambda ids: None, on_note_action=lambda a, r: calls.append(("note", a, r)),
                  on_changed=lambda: calls.append(("changed",)))
    t.pack(fill="both", expand=True)
    pump(root, 30)
    return t, calls


def slides(n=5, few=1):
    """n ticked slides and `few` unticked 3-tile ones, so the New-slides box is mixed."""
    p = Plan(SRC, OUT)
    for i in range(n):
        p.add_candidate(cand(f"run/S{i}_Scan1", n=20))
    for i in range(few):
        p.add_candidate(cand(f"run/T{i}_Scan1", n=3))
    for r in p.rows():
        p.apply_header(r.row_id, header(), None, 64 * GB)
    return p


def shown_help(t):
    return str(t.help.cget("text")) if t.help.winfo_manager() else ""


def entry_state(menu, label):
    end = menu.index("end")
    return next(menu.entrycget(i, "state") for i in range(end + 1)
                if menu.type(i) == "command" and menu.entrycget(i, "label") == label)


def test_the_hint_and_the_select_button_follow_the_state(table, ui_ctx, pump):
    t, _ = table
    tx = ui_ctx.theme.text
    assert t.select_button.instate(["disabled"]) and not shown_help(t)        # EMPTY: no plan yet
    plan = Plan(SRC, OUT)
    t.bind_plan(plan)                                                          # a search that found nothing yet
    t.set_state("SCANNING")
    assert t.select_button.instate(["disabled"]) and not shown_help(t)
    for x in slides().rows():
        plan.add_candidate(x.cand)
        plan.apply_header(x.row_id, header(), None, 64 * GB)
    t.refresh(None)
    for state in ("SCANNING", "READY"):
        t.set_state(state)
        assert shown_help(t) == tx("Untick anything you don’t want.") and t.select_button.instate(["!disabled"])
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    t.set_state("RUNNING")
    assert shown_help(t) == "Untick a waiting slide to skip it." and t.select_button.instate(["!disabled"])
    plan.select("untick_waiting")                                              # nothing left to untick: no hint,
    t.refresh(None)                                                            # and no menu of greyed-out items
    assert not shown_help(t) and t.select_button.instate(["disabled"])
    t.set_state("STOPPING")                                                    # nothing more starts: no hint
    assert not shown_help(t) and t.select_button.instate(["disabled"])
    plan.finish(job.row_id, RowOutcome("failed", "Failed: test"))
    plan.end_run()
    t.set_running(False)                                                       # (the order app.py uses)
    t.set_state("FINISHED")
    assert not shown_help(t)                                                   # the failed slide is still ticked
    plan.toggle([job.row_id], False)
    t.refresh([job.row_id])
    assert shown_help(t) == "Tick failed slides to try them again."
    t.bind_plan(None)
    t.set_state("EMPTY")
    assert not shown_help(t) and t.select_button.instate(["disabled"])


def test_during_a_run_only_the_waiting_slides_boxes_look_active(table, ui_ctx, pump):
    t, calls = table
    icons = ui_ctx.icons
    plan, c = every_status()
    t.bind_plan(plan)
    plan.begin_run()
    running = plan.claim_next(OUT, None).row_id
    t.set_state("RUNNING")
    for r in plan.rows():
        if r.section == "note" or not r.checkable:
            continue
        box, right, active = ("cb_on" if r.checked else "cb_off"), status_icon(r), r.status == "waiting" and r.checked
        assert image_of(t, t._iid[r.row_id]) == str(icons.pair(box, right) if active else t._image(box, right, 0))
        assert active or str(t._image(box, right, 0)) != str(icons.pair(box, right))
    assert image_of(t, "sec:new") == str(t._image("chev_down", "cb_mixed", 1))   # the section box too (MW-2)
    few = t._iid[c["few"].key]                                                  # unticked: it can't be ticked now
    click(t, pump, few, dx=2)
    assert not plan.row(c["few"].key).checked and t.tree.selection() == (few,)
    assert image_of(t, t._iid[running]) == str(t._image("cb_on", "run", 0))
    plan.finish(running, RowOutcome("done", "Done · 1 GB · 1 min"))
    plan.end_run()
    t.set_state("FINISHED")                                                     # stitched in this plan: still dim
    assert image_of(t, t._iid[running]) == str(t._image("cb_on", "ok", 0))
    assert image_of(t, t._iid[c["chan"].key]) == str(icons.pair("cb_on", "warn"))
    click(t, pump, t._iid[running], dx=2)
    assert plan.row(running).checked


def test_the_section_box_never_unticks_the_queue_during_a_run(table, ui_ctx, pump):
    """MW-2: a click on the New-slides box (or Space on the section) only opens or closes the section, also when
    slides selected before are now out of sight inside it."""
    t, _ = table
    tree = t.tree
    plan = slides(5, 1)
    t.bind_plan(plan)
    t.set_state("READY")
    plan.begin_run()
    plan.claim_next(OUT, None)
    t.set_state("RUNNING")
    queue = lambda: [r.row_id for r in plan.rows() if r.status == "waiting" and r.checked]     # noqa: E731
    waiting = queue()
    assert len(waiting) == 4 and tree.item("sec:new", "open")
    slot1 = ui_ctx.icons.size + ui_ctx.icons.gap + 2
    click(t, pump, t._iid[waiting[0]], column="what")                          # a waiting slide is selected,
    click(t, pump, "sec:new", dx=slot1)                                       # then the box is clicked
    assert queue() == waiting and not tree.item("sec:new", "open")             # like a click on the label
    press(t, pump, "<space>")
    assert queue() == waiting and tree.item("sec:new", "open")
    assert tree.selection() == ("sec:new",) and tree.focus() == "sec:new"      # the selection followed the click
    press(t, pump, "<Control-a>")                                             # every slide selected, then the
    click(t, pump, "sec:new", dx=200)                                         # label closes the section
    for key, is_open in (("<Delete>", False), ("<space>", True), ("<space>", False)):
        press(t, pump, key)
        assert queue() == waiting and bool(tree.item("sec:new", "open")) is is_open, key
    press(t, pump, "<space>")
    click(t, pump, t._iid[waiting[0]], column="what")                          # a waiting slide: Space skips it
    press(t, pump, "<space>")
    assert not plan.row(waiting[0]).checked and queue() == waiting[1:]


def test_once_a_stop_is_asked_for_waiting_slides_read_wont_be_stitched(table, ui_ctx, pump):
    """MW-19: after 'Stop' (or 'Finish X, then quit') no waiting slide will start."""
    t, _ = table
    plan = slides(4, 0)
    t.bind_plan(plan)
    plan.begin_run()
    running = plan.claim_next(OUT, None).row_id
    t.set_state("RUNNING")
    *waiting, skipped = [r.row_id for r in plan.rows() if r.status == "waiting"]
    assert all(t.tree.set(t._iid[w], "what") == "Waiting" for w in waiting)
    plan.toggle([skipped], False)                                              # unticked before the stop
    t.refresh([skipped])
    t.set_state("STOPPING")
    for w in waiting:
        iid = t._iid[w]
        assert t.tree.set(iid, "what") == "Won't be stitched (stopping)" and t.tree.item(iid, "tags") == ("dim",)
        assert image_of(t, iid) == str(t._image("cb_on", "skip", 0))
    assert t.tree.set(t._iid[skipped], "what") == "Won't be stitched"          # the stop changes nothing for it
    assert t.tree.item(t._iid[running], "tags") == ("running",)
    assert ui_ctx.theme.text("Don’t stitch this slide") not in labels_for(t, waiting[0])
    t._fill_select()
    assert entry_state(t.select_menu, "Untick all waiting") == "disabled"


def test_searches_and_header_retries_are_disabled_during_a_run(table, ui_ctx, pump):
    """MW-21: a search can't start during a run, so the note actions are offered disabled."""
    t, calls = table
    plan, c = every_status()
    t.bind_plan(plan)
    notes = {plan.row(r).note.kind: r for r in t._iid if r.startswith("note|")}
    actions = (("not_searched_depth", "Search deeper"), ("wide_pruned", "Search these folders too"),
               ("unresponsive", "Search this folder again"))
    plan.begin_run()
    t.set_state("RUNNING")
    for kind, label in actions:
        labels_for(t, notes[kind])
        assert entry_state(t.menu, label) == "disabled", kind
    labels_for(t, c["herr"].key)
    assert entry_state(t.menu, "Try again") == "disabled"
    plan.end_run()
    t.set_state("FINISHED")
    for kind, label in actions:
        labels_for(t, notes[kind])
        assert entry_state(t.menu, label) == "normal", kind


def test_new_slides_counts_the_slides_start_would_run(table, ui_ctx, pump):
    """MW-7: slides stitched in this plan are counted apart from the ticked ones, in the section's What cell, so
    the label keeps its width (MW-18); once every slide is done, the label says so."""
    t, _ = table
    plan = slides(3, 1)
    t.bind_plan(plan)
    t.set_state("READY")

    def section():
        return t.tree.item("sec:new", "text"), t.tree.set("sec:new", "what")
    assert section() == ("New slides  (3 of 4 ticked)", "")
    plan.begin_run()
    first = plan.claim_next(OUT, None).row_id
    t.set_state("RUNNING")
    plan.finish(first, RowOutcome("done", "Done · 1 GB · 1 min"))
    t.refresh([first])
    assert section() == ("New slides  (2 of 3 ticked)", "1 done")
    for _ in range(2):
        job = plan.claim_next(OUT, None)
        plan.finish(job.row_id, RowOutcome("done", "Done · 1 GB · 1 min"))
    plan.end_run()
    t.set_state("FINISHED")
    assert section() == ("New slides  (0 of 1 ticked)", "3 done")
    plan.toggle([r.row_id for r in plan.rows() if r.cand.folder_name == "T0_Scan1"], True)
    t.refresh(None)
    assert section() == ("New slides  (1 of 1 ticked)", "3 done")
    every = slides(2, 0)
    t.bind_plan(every)
    every.begin_run()
    for _ in range(2):
        job = every.claim_next(OUT, None)
        every.finish(job.row_id, RowOutcome("done", "Done · 1 GB · 1 min"))
    every.end_run()
    t.set_state("FINISHED")
    assert section() == ("New slides  (2 done)", "")
