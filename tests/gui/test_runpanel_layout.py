"""The compact completion banner in the real minimum-size main window (MW-15, V-U2b-3)."""
from __future__ import annotations

import pytest

from test_mainwindow import (_no_stale_tk_garbage, make_slides, make_window, ready, until)  # noqa: F401
from test_runpanel import buttons, headline, summary
from tests.fakes import FakeEngine
from yfs.core.model import RowOutcome
from yfs.ui.runpanel import banner_headline, problem_lines


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_compact_completion_with_every_problem_keeps_four_plan_rows(make_window, synth_tree, tmp_path, pump):
    source = make_slides(tmp_path / "ProjectC", [f"Slide{i}_Scan1" for i in range(1, 9)], synth_tree)
    out = tmp_path / "Stitched"
    out.mkdir()
    engine = FakeEngine.configure(fail="scan", log_warning="Synthetic read error",
                                  fail_if=lambda eng: eng.image_name not in ("Slide1_Scan1", "Slide2_Scan1",
                                                                            "Slide3_Scan1"))
    win = make_window(engine, input=source, output=out, restore=False)
    root, px = win.root, win.ctx.theme.px
    until(root, lambda: ready(win), timeout=3, what="synthetic plan")
    root.geometry(f"{px(760)}x{px(560)}")
    pump(root, 80)
    assert win._compact
    win.start(confirm=False)
    until(root, lambda: win.state == "FINISHED" and not win._index_pending, timeout=3, what="synthetic batch")
    plan = win.plan()
    plan.begin_run()
    outcomes = ("done", "done", "done", "done_gaps", "failed", "failed", "skipped", "finalize")
    rows = tuple((row.row_id, RowOutcome(status, "See the log for details."))
                 for row, status in zip(plan.rows(), outcomes))
    for row_id, outcome in rows:
        plan.finish(row_id, outcome)
    plan.end_run()
    win.table.refresh()
    win._set_state("FINISHED")
    result = summary({status: outcomes.count(status) for status in set(outcomes)}, rows)
    win.runbar.show_finished(result, problem_lines(result, lambda rid: plan.label(plan.row(rid))))
    pump(root, 100)
    tree = win.table.tree
    items = [row for section in tree.get_children("") for row in (section,) + tree.get_children(section)]
    visible = [row for row in items if tree.bbox(row) and
               tree.bbox(row)[1] + tree.bbox(row)[3] <= tree.winfo_height() - 2]  # inside the bottom border
    assert len(visible) >= 4, (len(visible), win.runbar.winfo_height())
    assert str(headline(win.runbar).cget("text")) == win.ctx.theme.text(banner_headline(result, short=True))
    for button in buttons(win.runbar.banner).values():
        assert button.winfo_ismapped()
        assert button.winfo_rootx() + button.winfo_width() <= root.winfo_rootx() + root.winfo_width()
