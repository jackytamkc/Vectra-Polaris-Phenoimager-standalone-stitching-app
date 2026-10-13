"""RunBar and LogDrawer (FINAL_SPEC §2.4.1, §2.4.4-§2.4.8, §4.9, §5.5.15) and the run-bar text helpers."""
from __future__ import annotations

import dataclasses
import time
import tkinter as tk
from tkinter import ttk

import pytest

from yfs.core.events import LogLine
from yfs.core.model import Issue, RowOutcome, RunSummary
from yfs.core.plan import Totals
from yfs.ui.dialogs import IssuesDialog
from yfs.ui.runpanel import (LogDrawer, ProgressModel, RunBar, banner_headline, fit_place, fit_text, problem_lines,
                             totals_text)
from yfs.ui.widgets import Tooltip

GB = 1024 ** 3


def totals(ticked=5, unknown=0, disk=0, typical=46 * GB, unsized=0):
    return Totals(ticked, ticked, 0, 0, 0, 100, typical, typical * 2, unknown, disk, 0, unsized)


def summary(counts, rows=(), stop=None, csv="/out/YourFavouriteStitcher_summary_20261010-2241.csv", took=46 * 60):
    end = time.mktime((2026, 10, 10, 22, 41, 0, 0, 0, -1))
    return RunSummary(end - took, end, counts, tuple(rows), stop == "user", stop, csv)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_totals_text_variants():
    assert totals_text(totals(), 1300 * GB, "Data") == "5 slides · ~46.0 GB to write · 1.3 TB free on Data · all fit in " \
                                                      "memory"
    assert totals_text(totals(1, typical=9 * GB), None) == "1 slide · ~9.0 GB to write · fits in memory"
    assert totals_text(totals(disk=2), 10 * GB) == "5 slides · ~46.0 GB to write · 10.0 GB free · 2 need temporary " \
                                                   "disk space (slower)"
    assert totals_text(totals(unknown=3), None) == "5 slides · sizes are estimated once the first tiles are read"
    assert totals_text(totals(0), 1) == ""


def test_totals_count_slides_of_unknown_size_apart():
    """MWV-2: a ticked slide whose size can never be estimated (a legacy folder, an offline placeholder, an unreadable
    first tile) is counted apart; the others are not all said to fit in memory, and nothing waits for sizes."""
    assert totals_text(totals(6, unsized=1), 1300 * GB, "Data") == \
        "6 slides · ~46.0 GB to write (+1 of unknown size) · 1.3 TB free on Data · 5 fit in memory"
    assert totals_text(totals(2, unsized=1, typical=GB // 2), None) == \
        "2 slides · ~512 MB to write (+1 of unknown size) · 1 fits in memory"
    assert totals_text(totals(6, disk=2, unsized=3), None) == \
        "6 slides · ~46.0 GB to write (+3 of unknown size) · 2 need temporary disk space (slower)"
    assert totals_text(totals(1, unsized=1, typical=0), 9 * GB, "Data") == \
        "1 slide · size unknown · 9.0 GB free on Data"
    assert totals_text(totals(2, unsized=2, typical=0), None) == "2 slides · sizes unknown"
    assert totals_text(totals(6, unknown=2, unsized=1), None) == \
        "6 slides · sizes are estimated once the first tiles are read"


def test_banner_headline_and_problem_lines():
    rows = (("a", RowOutcome("done", "Done · 8.9 GB · 7 min")), ("b", RowOutcome("failed", "Failed: couldn't read 3 tiles")),
            ("c", RowOutcome("skipped", "Skipped: not enough space")), ("d", RowOutcome("done_gaps", "Done with gaps: x")))
    s = summary({"done": 4, "done_gaps": 1, "failed": 1, "skipped": 1}, rows)
    assert banner_headline(s) == "Finished at 22:41 (took 46 min): 5 stitched (1 with gaps), 1 failed, 1 skipped."
    assert banner_headline(s, short=True) == "Finished: 5 stitched (1 with gaps), 1 failed, 1 skipped."   # compact
    assert problem_lines(s, str.upper) == ["B failed: couldn't read 3 tiles", "C skipped: not enough space",
                                           "D done with gaps: x"]
    s = summary({"done": 2, "cancelled": 1}, stop="user", took=30)
    assert banner_headline(s) == "Stopped at 22:41 (took 30 s): 2 stitched, 1 stopped."
    assert banner_headline(s, short=True) == "Stopped: 2 stitched, 1 stopped."
    full = RowOutcome("failed", "Failed: No space left on device.", output_side=True)
    s = summary({"done": 1, "failed": 2}, (("a", full), ("b", full)), stop="output_unavailable")
    assert banner_headline(s) == banner_headline(s, short=True) == \
        "Stopped: the save folder became unavailable (No space left on device)."
    assert banner_headline(summary({})) == "Finished at 22:41 (took 46 min): nothing was stitched."


def test_fit_text_shortens_with_the_ellipsis(tk_root):
    import tkinter.font as tkfont
    font = tkfont.nametofont("YfsBody")
    text = "Stitching 2 of 5 · SlideA2_Scan1 · Stitching tiles 41/64"
    assert fit_text(font, text, 0, "...") == text and fit_text(font, text, 10_000, "...") == text
    short = fit_text(font, text, font.measure(text) // 2, "...")
    assert short.endswith("...") and text.startswith(short[:-3]) and font.measure(short) <= font.measure(text) // 2


def test_fit_text_can_keep_the_end_whole(tk_root):
    import tkinter.font as tkfont
    font = tkfont.nametofont("YfsBody")
    text, keep = "Can't save to 2026-09-03_ProjectC_round2_stitched: Permission denied.", ": Permission denied."
    short = fit_text(font, text, font.measure(text) * 2 // 3, "...", keep)
    assert short.endswith("..." + keep) and text.startswith(short[:-len("..." + keep)])
    assert font.measure(short) <= font.measure(text) * 2 // 3
    tiny = fit_text(font, text, font.measure("..." + keep), "...", keep)           # no room for it: a plain cut
    assert tiny.endswith("...") and font.measure(tiny) <= font.measure("..." + keep)


def test_fit_place_cuts_the_place_name_before_the_words_around_it(tk_root):
    """MW-24, V-U2b-4: the name is cut first, down to its first 4 characters; then the end goes too."""
    import tkinter.font as tkfont
    font = tkfont.nametofont("YfsBody")
    lead, place, tail = "Can't save to ", "2026-09-03_ProjectC_round2_stitched", ": Permission denied."
    text = lead + place + tail
    assert fit_place(font, lead, place, tail, 0, "...") == text == fit_place(font, lead, place, tail, 10_000, "...")
    width = font.measure(text) * 2 // 3
    short = fit_place(font, lead, place, tail, width, "...")
    assert short.startswith(lead + place[:4]) and short.endswith("..." + tail) and font.measure(short) <= width
    width = font.measure(lead + place[:4] + "..." + tail) - 1                  # not even 4 characters fit with it
    tiny = fit_place(font, lead, place, tail, width, "...")
    assert tiny.startswith(lead + place[:4] + "...: Perm") and tiny.endswith("...") and font.measure(tiny) <= width
    short = fit_place(font, lead, "Out", tail, font.measure(lead + "Out" + tail) - 1, "...")
    assert short.startswith(lead + "Out: Permission") and short.endswith("...")   # a short name stays whole


# ------------------------------------------------------------------------------------------------ run bar
class Calls(list):
    pass


@pytest.fixture
def runbar(ui_ctx, pump):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(1040)}x{px(200)}+0+0")
    calls = Calls()
    bar = RunBar(root, ui_ctx, on_start=lambda: calls.append("start"), on_stop=lambda: calls.append("stop"),
                 on_issue_action=calls.append, on_open_output=lambda: calls.append("open"),
                 on_retry_failed=lambda: calls.append("retry"), on_new_batch=lambda: calls.append("new"))
    bar.pack(side="bottom", fill="x")
    pump(root, 30)
    return bar, calls


def shown(bar):
    return {str(w) for w in bar.grid_slaves()}


def buttons(widget):
    out = {}
    for w in widget.winfo_children():
        if w.winfo_class() == "TButton":
            out[str(w.cget("text"))] = w
        out.update(buttons(w))
    return out


def labels(widget):
    return [str(w.cget("text")) for w in widget.winfo_children() if w.winfo_class() == "TLabel"]


def headline(bar):
    """The banner's headline: its bold label."""
    return next(w for row in bar.banner.winfo_children() if row.winfo_class() == "Frame"
                for w in row.winfo_children() if w.winfo_class() == "TLabel" and str(w.cget("font")) == "YfsBold")


def note(bar):
    """The text at the left of the banner's bottom row: the summary CSV."""
    return next(w for row in bar.banner.winfo_children() if row.winfo_class() == "Frame"
                for f in row.winfo_children() if f.winfo_class() == "Frame"
                for w in f.winfo_children() if w.winfo_class() == "TLabel")


def tooltip(widget, pump, monkeypatch) -> str:
    """The text of `widget`'s tooltip, as the pointer shows it."""
    monkeypatch.setattr(Tooltip, "DELAY_MS", 30)
    widget.event_generate("<Enter>")
    pump(widget, 120)
    tips = [w for w in widget.winfo_children() if w.winfo_class() == "Toplevel"]
    text = str(tips[0].winfo_children()[0].cget("text")) if tips else ""
    widget.event_generate("<Leave>")
    return text


def test_idle_button_text_and_state(runbar, ui_ctx, pump):
    bar, calls = runbar
    bar.show_idle(5, "5 slides · ~46 GB to write", None, 0)
    pump(bar, 30)
    assert str(bar.button.cget("text")) == "Stitch 5 slides" and bar.button.instate(["!disabled"])
    assert str(bar.issue) not in shown(bar) and bar.totals.cget("text") == "5 slides · ~46 GB to write"
    bar.show_idle(1, "", None, 0)
    assert str(bar.button.cget("text")) == "Stitch 1 slide"
    bar.show_idle(0, "", Issue("block", "none_ticked", "Tick at least one slide."), 0)
    assert bar.button.instate(["disabled"]) and str(bar.issue) in shown(bar)
    bar.button.invoke()
    assert calls == []
    bar.show_idle(3, "", None, 0)
    bar.button.invoke()                                                       # Ctrl+Enter does the same
    assert calls == ["start"] and bar.button.instate(["disabled"])
    assert str(bar.button.cget("text")) == ui_ctx.theme.text("Checking…")
    bar.show_idle(3, "", None, 0)
    assert bar.button.instate(["!disabled"])


def test_issue_line_styles_actions_and_more(runbar, ui_ctx, pump):
    bar, calls = runbar
    t = ui_ctx.theme.text
    bar.show_idle(5, "totals", Issue("warn", "channels", "1 slide needs a look: check channel names (DAPI ×5).",
                                     ("r",), "fix_channels"), 2)
    pump(bar, 30)
    assert str(bar.issue.cget("style")) == "Warn.TFrame" and bar.button.instate(["!disabled"])
    assert str(bar.issue_text.cget("text")) == t("1 slide needs a look: check channel names (DAPI ×5).")
    assert "(+2 more)" in [str(w.cget("text")) for w in bar.issue.winfo_children() if w.winfo_class() == "TLabel"]
    buttons(bar.issue)[t("Fix names…")].invoke()
    assert calls == ["fix_channels"]
    bar.show_idle(5, "", Issue("block", "paused", "Search paused.", (), "resume_scan"), 0)
    assert str(bar.issue.cget("style")) == "TFrame" and bar.button.instate(["disabled"])
    found = buttons(bar.issue)
    found["Keep searching"].invoke()
    found["Stop here"].invoke()
    assert calls[-2:] == ["resume_scan", "stop_scan"]
    bar.show_idle(5, "", Issue("block", "output_error", "Can't save to /out: Permission denied.", (), "browse_output"), 0)
    assert str(bar.issue.cget("style")) == "Error.TFrame"
    buttons(bar.issue)[t("Browse…")].invoke()
    assert calls[-1] == "browse_output"
    bar.show_idle(5, "", Issue("warn", "final_check", "Couldn't check Out: it didn't answer.", (), "retry_check"), 0)
    assert str(bar.issue.cget("style")) == "Warn.TFrame" and bar.button.instate(["!disabled"])
    buttons(bar.issue)["Try again"].invoke()                                 # UD1: the final check once more
    assert calls[-1] == "retry_check"
    bar.show_idle(5, "", Issue("block", "pdir_problem", "Can't save to Out: its .stitcher-in-progress folder is a "
                                                        "link to another place.", (), "browse_output"), 0)
    assert str(bar.issue.cget("style")) == "Error.TFrame" and bar.button.instate(["disabled"])
    assert t("Browse…") in buttons(bar.issue)
    bar.show_idle(0, "", Issue("block", "none_ticked", "Nothing to stitch yet — another stitcher is writing S1. "
                                                       "Rescan when it's done.", (), "rescan"), 0)
    buttons(bar.issue)["Rescan"].invoke()                                    # V-U2b-6: the busy slide's next step
    assert calls[-1] == "rescan" and str(bar.issue.cget("style")) == "TFrame"


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_long_texts_are_cut_so_buttons_stay_inside_760px(runbar, ui_ctx, pump):
    bar, calls = runbar
    px, root = ui_ctx.theme.px, ui_ctx.root
    root.geometry(f"{px(760)}x{px(200)}")
    long = "Nothing new to stitch — all 12 slides already have stitched files. Tick one to replace it. " * 3
    bar.show_idle(12, "12 slides · ~460 GB to write · 1.3 TB free on a disk with a long name · all fit in memory",
                  Issue("warn", "channels", long, (), "fix_channels"), 7)
    pump(bar, 80)
    edge = root.winfo_rootx() + root.winfo_width()
    for w in [bar.button, *buttons(bar.issue).values()]:
        assert w.winfo_rootx() + w.winfo_width() <= edge, w
    assert bar.winfo_reqwidth() <= root.winfo_width()
    assert str(bar.issue_text.cget("text")).endswith(ui_ctx.theme.G["ell"])
    bar.show_running(long, 0.4, 0.6, "No progress for 3 min — the input share may be stalled")
    pump(bar, 80)
    assert bar.stall.winfo_rootx() + bar.stall.winfo_width() <= edge
    assert bar.stop_button.winfo_rootx() + bar.stop_button.winfo_width() <= edge
    rows = tuple((f"r{i}", RowOutcome("failed", "Failed: " + long)) for i in range(5))
    bar.show_finished(summary({"done": 3, "failed": 5, "finalize": 1}, rows), [long] * 5)
    pump(bar, 80)
    for w in buttons(bar.banner).values():
        assert w.winfo_rootx() + w.winfo_width() <= edge
    assert bar.winfo_reqwidth() <= root.winfo_width()


def test_running_and_stopping_layouts(runbar, ui_ctx, pump):
    bar, calls = runbar
    t = ui_ctx.theme.text
    bar.show_running("Stitching 2 of 5 · S2 · Stitching tiles 41/64 · about 19 min left", 0.41, 0.64, None)
    pump(bar, 30)
    assert {str(bar.line), str(bar.bars), str(bar.stop_button), str(bar.button)} <= shown(bar)
    assert str(bar.button.cget("text")) == t("Stitching…") and bar.button.instate(["disabled"])
    assert float(bar.overall["value"]) == pytest.approx(0.41) and float(bar.slide["value"]) == pytest.approx(0.64)
    assert not bar.stall.winfo_manager()
    bar.stop_button.invoke()
    assert calls == ["stop"]
    bar.show_running("Stitching 2 of 5 · S2 · Writing full-resolution image", 0.5, None,
                     "No progress for 3 min — the input share may be stalled")
    pump(bar, 30)
    assert bar.stall.winfo_manager() and str(bar.stall.cget("style")) == "Bg.Warn.TLabel"
    assert float(bar.slide["value"]) == 0.0
    bar.show_stopping(t("Stopping after S2…"), False)
    assert str(bar.stop_button) not in shown(bar) and str(bar.quit_button) not in shown(bar)
    bar.show_stopping(t("Stopping… finishing the current write step (can take a few minutes)"), True)
    assert str(bar.quit_button) in shown(bar)
    bar.quit_button.invoke()
    assert calls[-1] == "quit_anyway"
    bar.show_running("ignored while stopping", 0.9, 0.95, None)
    pump(bar, 30)
    assert bar.mode == "stopping" and "ignored" not in str(bar.status.cget("text"))
    assert float(bar.overall["value"]) == pytest.approx(0.9)


def test_completion_banner_ok_warn_and_actions(runbar, ui_ctx, pump):
    bar, calls = runbar
    t = ui_ctx.theme.text
    bar.show_idle(4, "4 slides", None, 0)
    bar.show_finished(summary({"done": 5}), [])
    pump(bar, 30)
    assert bar.mode == "finished" and str(bar.banner.cget("style")) == "Ok.TFrame" and shown(bar) == {str(bar.banner)}
    found = buttons(bar.banner)
    assert set(found) == {"", "Open output folder", "Start a new batch", "Open"}
    assert str(found["Open output folder"].cget("style")) == "Primary.TButton"
    found["Open output folder"].invoke()
    assert calls == ["open"] and bar.mode == "finished"                       # opening does not close the banner
    bar.show_idle(2, "2 slides", None, 0)                                     # the banner stays until dismissed
    assert bar.mode == "finished"
    found[""].invoke()                                                        # [x]
    assert bar.mode == "idle" and str(bar.button.cget("text")) == "Stitch 2 slides"
    rows = tuple((f"r{i}", RowOutcome("failed", f"Failed: reason {i}")) for i in range(5))
    lines = problem_lines(summary({"failed": 5}, rows), lambda r: r)
    bar.show_finished(summary({"done": 1, "failed": 5}, rows), lines)
    pump(bar, 30)
    assert str(bar.banner.cget("style")) == "Warn.TFrame"
    texts = [str(w.cget("text")) for w in bar.banner.winfo_children() if w.winfo_class() == "TLabel"]
    assert texts[:3] == ["r0 failed: reason 0", "r1 failed: reason 1", "r2 failed: reason 2"]
    assert texts[3] == t("…and 2 more — see the list.")
    buttons(bar.banner)["Retry failed"].invoke()
    assert calls[-1] == "retry" and bar.mode == "idle"
    bar.show_finished(summary({"done": 1}, stop="output_unavailable"), [])
    found = buttons(bar.banner)
    assert "Retry failed" not in found and "Retry remaining" in found
    found["Retry remaining"].invoke()                                         # starts again from the current plan
    assert calls[-1] == "start" and bar.mode == "idle"
    bar.show_finished(summary({"finalize": 1}, csv=None), [])
    found = buttons(bar.banner)
    assert "Open" not in found
    found["Finalize"].invoke()
    assert calls[-1] == "finalize"
    bar.show_finished(summary({"done": 1}), [])
    bar.show_open_error("/data/out")
    buttons(bar.banner)["Copy"].invoke()
    assert bar.clipboard_get() == "/data/out"
    buttons(bar.banner)["Start a new batch"].invoke()
    assert calls[-1] == "new" and bar.mode == "idle"


def test_compact_mode_puts_totals_in_a_tooltip_and_shortens_the_banner(runbar, ui_ctx, pump, monkeypatch):
    bar, calls = runbar
    t = ui_ctx.theme.text
    issue = Issue("warn", "replace", "1 file will be replaced: S1.", ("r",))
    bar.show_idle(3, "3 slides · ~20 GB to write", issue, 0)
    assert str(bar.totals) in shown(bar)
    bar.set_compact(True)
    assert str(bar.totals) not in shown(bar) and str(bar.issue) in shown(bar)
    bar.show_idle(3, "3 slides · ~20 GB to write", None, 0)
    assert str(bar.totals) in shown(bar)                                     # no issue: the totals keep their line
    rows = tuple((f"r{i}", RowOutcome("failed", "Failed: x")) for i in range(5))
    bar.show_finished(summary({"failed": 5}, rows), ["a", "b", "c", "d", "e"])
    pump(bar, 30)
    head, open_button = headline(bar), buttons(bar.banner)["Open output folder"]
    assert labels(bar.banner) == [] and str(head.cget("text")) == "Finished: 5 failed."   # one row: no problem lines
    assert head.winfo_rooty() < open_button.winfo_rooty() + open_button.winfo_height() and \
        open_button.winfo_rooty() < head.winfo_rooty() + head.winfo_height()          # the headline beside the buttons
    more = t("…and 2 more — see the list.")                       # V-U2b-2: as many lines as the full banner
    assert tooltip(head, pump, monkeypatch) == \
        f"Finished at 22:41 (took 46 min): 5 failed.\na\nb\nc\n{more}\nSummary saved: " \
        "YourFavouriteStitcher_summary_20261010-2241.csv"
    bar.set_compact(False)
    assert labels(bar.banner) == ["a", "b", "c", more] and "Summary saved" in str(note(bar).cget("text"))
    assert str(headline(bar).cget("text")) == "Finished at 22:41 (took 46 min): 5 failed."


def test_a_slide_with_gaps_makes_the_banner_a_warning_that_names_it(runbar, pump):
    """DS-1: a slide stitched with missing tiles is a problem: Warn banner, the count in the headline, its line."""
    bar, calls = runbar
    rows = (("a", RowOutcome("done", "Done · 8.9 GB · 7 min")),
            ("b", RowOutcome("done_gaps", "Done with gaps: 1 tile couldn't be read")))
    s = summary({"done": 1, "done_gaps": 1}, rows)
    bar.show_finished(s, problem_lines(s, {"a": "PC-01_Scan1", "b": "PC-02_Scan1"}.get))
    pump(bar, 30)
    assert str(bar.banner.cget("style")) == "Warn.TFrame"
    assert str(headline(bar).cget("text")) == "Finished at 22:41 (took 46 min): 2 stitched (1 with gaps)."
    assert labels(bar.banner) == ["PC-02_Scan1 done with gaps: 1 tile couldn't be read"]
    assert "Retry failed" not in buttons(bar.banner)
    bar.show_finished(summary({"done": 2}, rows[:1]), [])
    assert str(bar.banner.cget("style")) == "Ok.TFrame"


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_compact_bars_keep_the_idle_height_so_the_plan_keeps_its_rows(runbar, ui_ctx, pump, monkeypatch):
    """MW-15, D4: at 760 px in compact mode the running bar is no taller than the idle bar, and the completion banner
    at most by a few pixels (its headline, without the time, may take two lines beside the buttons), so the plan keeps
    its four rows. A headline that needs more stands on its own line over the buttons (V-U2b-3). The headline is never
    cut, a cut status line has a tooltip, and every button stays inside the window."""
    bar, calls = runbar
    px, root, t = ui_ctx.theme.px, ui_ctx.root, ui_ctx.theme.text
    root.geometry(f"{px(760)}x{px(200)}")
    bar.set_compact(True)
    bar.show_idle(6, "6 slides", Issue("warn", "channels", "PC-04_Scan1 needs a look: check channel names (DAPI ×5).",
                                       ("r",), "fix_channels"), 0)
    pump(bar, 80)
    idle = bar.winfo_height()
    text = "Stitching 6 of 6 · PC-06_Scan1 · Writing pyramid level 2 of 3 · about 2 min left (done around 22:43)"
    bar.show_running(text, 0.9, 0.95, None)
    pump(bar, 80)
    assert bar.winfo_height() <= idle and bar.overall.winfo_ismapped() and not bar.slide.winfo_ismapped()
    assert str(bar.status.cget("text")) != t(text) and tooltip(bar.status, pump, monkeypatch) == t(text)
    edge = root.winfo_rootx() + root.winfo_width()
    rows = tuple((f"r{i}", RowOutcome("failed", "Failed: see the log for details.")) for i in range(3))
    line = ui_ctx.theme.fonts["YfsBold"].metrics("linespace")
    for counts, over in (({"done": 3, "failed": 3}, False), ({"done": 3, "failed": 3, "finalize": 1}, False),
                         ({"done": 3, "done_gaps": 1, "failed": 2, "skipped": 1, "finalize": 1}, True)):
        s = summary(counts, rows)
        bar.show_finished(s, problem_lines(s, lambda r: f"PC-0{int(r[1:]) * 2 + 1}_Scan1"))
        pump(bar, 120)
        head, open_button = headline(bar), buttons(bar.banner)["Open output folder"]
        assert str(head.cget("text")) == t(banner_headline(s, short=True))      # whole; the time is in the tooltip
        assert (head.winfo_rooty() + head.winfo_height() <= open_button.winfo_rooty()) == over
        if over:                            # too long for two lines beside five buttons: one line right over them
            assert head.winfo_height() < 2 * line and bar.winfo_height() <= idle + head.winfo_height()
            assert open_button.winfo_rooty() - head.winfo_rooty() - head.winfo_height() <= px(1)
        else:                               # beside the buttons, in at most two lines
            assert head.winfo_height() < 3 * line and bar.winfo_height() <= idle + px(6)
        for w in buttons(bar.banner).values():
            assert w.winfo_ismapped() and w.winfo_rootx() + w.winfo_width() <= edge, w
        assert bar.winfo_reqwidth() <= root.winfo_width()
    bar.set_compact(False)                                                   # a larger window: the full banner
    pump(bar, 80)
    assert labels(bar.banner)[:3] == [f"PC-0{k}_Scan1 failed: see the log for details." for k in (1, 3, 5)]


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_a_stall_notice_fits_the_compact_line_and_keeps_its_place(runbar, ui_ctx, pump, monkeypatch):
    """V-U2b-1, MW-9: at 760 px in compact mode the stall notice comes first, cut inside the save folder's name and
    never clipped; the status before it keeps whole parts or steps aside, and the line's tooltip has it all."""
    bar, calls = runbar
    px, root, t, ell = ui_ctx.theme.px, ui_ctx.root, ui_ctx.theme.text, ui_ctx.theme.G["ell"]
    root.geometry(f"{px(760)}x{px(200)}")
    bar.set_compact(True)
    text = "Stitching 6 of 6 · PC-06_Scan1 · Writing pyramid level 2 of 3"
    stall = ProgressModel.STALL_WRITING.format(3, "2026-09-03_ProjectC_tonsil_FFPE_panel1_stitched")
    bar.show_running(text, 0.9, 0.95, stall)
    pump(bar, 80)
    shown = str(bar.stall.cget("text"))
    assert shown.startswith(t("No progress for 3 min — writing to 2026")) and shown.endswith(f"{ell} seems slow or "
                                                                                            "stalled"), shown
    assert bar.stall.winfo_reqwidth() <= bar.stall.winfo_width()            # nothing clipped
    assert bar.stall.winfo_rootx() + bar.stall.winfo_width() <= bar.stop_button.winfo_rootx()
    assert not bar.status.winfo_ismapped() and bar.stall.winfo_rootx() == bar.overall.winfo_rootx()   # (not indented)
    assert tooltip(bar.stall, pump, monkeypatch) == t(f"{text} · {stall}")
    short = ProgressModel.STALL_WRITING.format(1, "Out")
    bar.show_running(text, 0.9, 0.95, short)
    pump(bar, 80)
    status = str(bar.status.cget("text"))
    assert str(bar.stall.cget("text")) == t(short) and bar.status.winfo_ismapped()
    assert status.startswith("Stitching 6 of 6 · ") and status.endswith(" · ") and t(f"{text} · ").startswith(status)


def test_more_link_lists_the_hidden_issues_with_their_actions(runbar, ui_ctx, pump, monkeypatch):
    """MW-11: '+1 more…' is a link. Its tooltip names what it hides, and it opens the list of those issues, each with
    its action button."""
    bar, calls = runbar
    t = ui_ctx.theme.text
    first = Issue("block", "no_source", "Choose the folder with your tiles.", (), "browse_input")
    hidden = [Issue("block", "no_output", "Choose where to save the stitched images.", (), "browse_output")]
    bar.show_idle(0, "", first, 1, hidden)
    pump(bar, 30)
    link = buttons(bar.issue)[t("+1 more…")]
    assert str(link.cget("style")) == "LinkBg.TButton"
    assert tooltip(link, pump, monkeypatch) == "Choose where to save the stitched images."
    link.invoke()
    dlg = next(w for w in ui_ctx.root.winfo_children() if isinstance(w, IssuesDialog))
    pump(bar, 30)
    assert str(dlg.title()) == "1 more thing to check" and dlg.grab_current() is dlg
    assert t("Choose where to save the stitched images.") in labels(dlg.listing.winfo_children()[0])
    buttons(dlg)[t("Browse…")].invoke()
    assert calls == ["browse_output"] and not dlg.winfo_exists()
    link.invoke()
    dlg = next(w for w in ui_ctx.root.winfo_children() if isinstance(w, IssuesDialog) and w.winfo_exists())
    dlg.focus_force()
    pump(bar, 30)
    (dlg.focus_get() or dlg).event_generate("<Escape>")                     # Esc closes it; nothing happens
    assert calls == ["browse_output"] and not dlg.winfo_exists()
    warn = Issue("warn", "channels", "PC-04_Scan1 needs a look: check channel names (DAPI ×5).", ("r",), "fix_channels")
    bar.show_idle(3, "", Issue("block", "output_error", "Can't save to Out: Permission denied.", (), "browse_output"),
                  2, [warn, Issue("warn", "space_some", "Space is tight.")])
    assert t("+2 more…") in buttons(bar.issue)


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_a_long_save_folder_name_never_hides_the_reason(runbar, ui_ctx, pump, monkeypatch):
    """MW-24: 'Can't save to <place>: <reason>.' (and the final check's 'Couldn't check <place>: <reason>.') is cut
    inside the place name, and its tooltip shows it all."""
    bar, calls = runbar
    px, root, t = ui_ctx.theme.px, ui_ctx.root, ui_ctx.theme.text
    root.geometry(f"{px(760)}x{px(200)}")
    place = "2026-09-03_ProjectC_multiplex_panel_round2_stitched_images_final"
    other = Issue("warn", "channels", "PC-04_Scan1 needs a look: check channel names (DAPI ×5).", ("r",),
                  "fix_channels")
    for issue, reason in ((Issue("block", "output_error", f"Can't save to {place}: Permission denied.", (),
                                 "browse_output"), ": Permission denied."),
                          (Issue("warn", "final_check", f"Couldn't check {place}: it didn't answer.", (),
                                 "retry_check"), ": it didn't answer.")):
        bar.show_idle(3, "3 slides", issue, 1, [other])
        pump(bar, 80)
        cut = str(bar.issue_text.cget("text"))
        assert cut != t(issue.text) and cut.endswith(ui_ctx.theme.G["ell"] + reason), cut
        assert tooltip(bar.issue_text, pump, monkeypatch) == t(issue.text)
    pdir = Issue("block", "pdir_problem", f"Can't save to {place}: its .stitcher-in-progress folder is a link to "
                 "another place. Delete the link, or choose another save folder.", (), "browse_output")
    bar.show_idle(3, "3 slides", pdir, 1, [other])                          # V-U2b-4: a reason too long to keep whole
    pump(bar, 80)
    cut, ell = str(bar.issue_text.cget("text")), ui_ctx.theme.G["ell"]
    assert cut.startswith(f"Can't save to {place[:4]}{ell}: its .stitcher") and cut.endswith(ell), cut
    assert bar.issue_text.winfo_reqwidth() <= bar.issue_text.winfo_width()  # nothing clipped
    assert tooltip(bar.issue_text, pump, monkeypatch) == t(pdir.text)
    bar.show_idle(3, "3 slides", Issue("block", "output_error", "Can't save to Out: Permission denied.", (),
                                       "browse_output"), 0)
    pump(bar, 80)
    assert tooltip(bar.issue_text, pump, monkeypatch) == ""                  # nothing cut: no tooltip


# ------------------------------------------------------------------------------------------------ log drawer
class FakeSettings:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


@pytest.fixture
def drawer(ui_ctx, pump):
    root, px = ui_ctx.root, ui_ctx.theme.px
    root.geometry(f"{px(1040)}x{px(600)}+0+0")
    ctx = dataclasses.replace(ui_ctx, settings=FakeSettings(**{"ui.log_height": 150}))
    pane = tk.PanedWindow(root, orient="vertical")
    pane.pack(fill="both", expand=True)
    upper = ttk.Frame(pane)
    pane.add(upper, stretch="always", height=px(300))
    ttk.Label(upper, text="plan").pack(fill="both", expand=True)
    d = LogDrawer(upper, pane, ctx)
    d.bar.pack(side="bottom", fill="x")
    pump(root, 30)
    return d, pane, ctx


def line(text, tag="info", slide="", t=None):
    level = {"info": 20, "ok": 20, "warn": 30, "error": 40, "debug": 10}[tag]
    return LogLine(t or time.time(), level, slide, text, tag)


def test_lines_are_batched_cleaned_tagged_and_counted(drawer, pump):
    d, _, ctx = drawer
    inserts = []
    real = d.text.insert
    d.text.insert = lambda *a: (inserts.append(a), real(*a))[1]
    t0 = time.mktime((2026, 10, 10, 12, 41, 7, 0, 0, -1))
    d.append([line("🔍 Scanning metadata...", t=t0)])
    d.append([line("⚠️ Suspicious channel names", "warn", "S2_Scan1", t0 + 1),
              line("❌ Failed to read a tile", "error", "S2_Scan1", t0 + 2)])
    d.append([line("✅ Stitching complete", "ok", "S1_Scan1", t0 + 3)])
    assert inserts == [] and d.counts() == (0, 0)                            # nothing until the tick is over
    pump(d.text, 30)
    assert len(inserts) == 1 and d.counts() == (1, 1)
    content = d.text.get("1.0", "end-1c").splitlines()
    assert content == ["12:41:07  Scanning metadata...", "12:41:08  S2_Scan1  Suspicious channel names",
                       "12:41:09  S2_Scan1  Failed to read a tile", "12:41:10  S1_Scan1  Stitching complete"]
    assert "warn" in d.text.tag_names("2.20") and "error" in d.text.tag_names("3.20") and "ok" in d.text.tag_names("4.20")
    assert "time" in d.text.tag_names("2.0") and "slide" in d.text.tag_names("2.10")
    assert d.warn_pill.winfo_manager() and str(d.warn_pill.cget("text")) == "1 warning"
    assert str(d.error_pill.cget("text")) == "1 error"
    assert str(d.last.cget("text")).startswith("12:41:10  S1_Scan1  Stitching complete")
    assert str(d.text.cget("state")) == "disabled"
    d.copy_all()
    assert d.text.clipboard_get().splitlines() == content


def test_reset_counts_starts_the_pills_afresh_and_keeps_the_log(drawer, pump):
    """MW-22: a new batch starts with no pills; the earlier lines stay in the log."""
    d, _, _ = drawer
    d.append([line("careful", "warn"), line("bad", "error"), line("plain")])
    pump(d.text, 30)
    assert d.counts() == (1, 1) and d.warn_pill.winfo_manager() and d.error_pill.winfo_manager()
    d.reset_counts()
    assert d.counts() == (0, 0) and not d.warn_pill.winfo_manager() and not d.error_pill.winfo_manager()
    assert len(d.text.get("1.0", "end-1c").splitlines()) == 3
    d.append([line("again", "warn")])
    pump(d.text, 30)
    assert d.counts() == (1, 0) and str(d.warn_pill.cget("text")) == "1 warning" and not d.error_pill.winfo_manager()


def test_at_most_5000_lines_are_kept(drawer, pump):
    d, _, _ = drawer
    d.append([line(f"line {i}") for i in range(5100)])
    pump(d.text, 50)
    rows = d.text.get("1.0", "end-1c").splitlines()
    assert len(rows) == 5000 and rows[0].endswith("line 100") and rows[-1].endswith("line 5099")


def test_auto_scroll_only_when_the_view_was_at_the_bottom(drawer, pump):
    d, _, _ = drawer
    d.set_open(True)
    d.append([line(f"line {i}") for i in range(300)])
    pump(d.text, 50)
    assert d.text.yview()[1] >= 0.999                                         # followed the new lines
    d.text.yview_moveto(0.2)
    pump(d.text, 30)
    top = d.text.index("@0,0")
    d.append([line("more")])
    pump(d.text, 30)
    assert d.text.index("@0,0") == top and d.text.yview()[1] < 0.999        # the reader's place is kept
    d.text.yview_moveto(1.0)
    d.append([line("again")])
    pump(d.text, 30)
    assert d.text.yview()[1] >= 0.999


def test_open_close_filter_and_remembered_height(drawer, ui_ctx, pump):
    d, pane, ctx = drawer
    px = ui_ctx.theme.px
    assert str(d.body) not in [str(p) for p in pane.panes()] and not d.filter.winfo_manager()
    d.toggle()
    pump(d.text, 50)
    assert str(d.body) in [str(p) for p in pane.panes()] and d.filter.winfo_manager() and d.is_open
    assert abs(d.body.winfo_height() - px(150)) <= 2 and ctx.settings.values["ui.log_open"] is True
    pane.paneconfigure(d.body, height=px(220))
    pump(d.text, 50)
    d.set_open(False)
    assert ctx.settings.values["ui.log_height"] == pytest.approx(220, abs=2)
    assert str(d.body) not in [str(p) for p in pane.panes()] and ctx.settings.values["ui.log_open"] is False
    d.append([line("plain"), line("careful", "warn"), line("detail", "debug"), line("bad", "error")])
    d.set_open(True)
    pump(d.text, 50)
    d.filter.set("Warnings and errors")
    d.filter.event_generate("<<ComboboxSelected>>")
    assert all(d.text.tag_cget(tag, "elide") in (1, "1", True) for tag in ("info", "ok", "debug"))
    d.filter.set("Everything")
    d.filter.event_generate("<<ComboboxSelected>>")
    assert all(d.text.tag_cget(tag, "elide") in (0, "0", False) for tag in ("info", "ok", "debug"))
    d.warn_pill.event_generate("<Button-1>")
    assert d.is_open
