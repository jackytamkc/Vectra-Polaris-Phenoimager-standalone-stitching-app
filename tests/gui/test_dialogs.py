"""Dialogs (FINAL_SPEC §2.2.9, §2.3.7, §2.4.2-§2.4.5, §2.5, §5.5.16; acceptance C5) on both Tk builds."""
from __future__ import annotations

import dataclasses
import gc
import os
import threading
import time
import tkinter as tk

import pytest

from yfs import APP_NAME, __version__
from yfs.core.model import Entry, Issue, TileHeader
from yfs.core.names import candidates_from_summary, summarize_listing
from yfs.core.plan import Plan
from yfs.core.settings import Settings
from yfs.ui.dialogs import (ChannelNamesDialog, ConfirmStartDialog, InfoDialog, IssuesDialog, NewFolderDialog,
                            SettingsDialog, SlideDetailsDialog, StopDialog, folder_name_error)

@pytest.mark.parametrize("quitting", [False, True])
def test_quit_after_stop_now_never_promises_to_finish_or_resume(ctx, tk_root, quitting):
    done = []
    dlg = StopDialog.open(tk_root, ctx, "Slide1_Scan1", 240.0, 2, True, done.append,
                          quitting_after=quitting, stopping_now=True)
    pump(tk_root)
    shown, choices = " ".join(texts(dlg)), buttons(dlg)
    assert "current write step" in shown
    assert "Finish 'Slide1" not in shown and "after 'Slide1" not in shown
    assert "Keep stitching" not in choices and not hasattr(dlg, "choice")
    choices["Don't quit" if quitting else "Wait, then quit"].invoke()
    assert done == ["cancel_quit" if quitting else "after_current"]


# Appendix B of the spec, as printed there (the v1 source has a few trailing spaces, which are not words).
APPENDIX_B = """Welcome to your FAVOURITE Stitcher!

It is important to note that this Stitcher only works with AKOYA VECTRA POLARIS/PHENOIMAGER unmixed tiles
Reason to use this:
1. You can run this in background
2. You can stitched multiple folders sequentially without you keep clicking
3. You don't have to pay £5/hr
Note: This app uses your RAM/disk to store tmp files, make sure you have some space left
If your input/output file is in network drive, the stitching speed is 99% depends on your network speed
App developed by Jacky@Ramachandran Lab

Logic adopted from Pete Bankhead AKA Qupath Founder
Enjoy!"""


def pump(root, ms: int = 50) -> None:
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


def esc(dlg) -> None:
    """Esc as the user presses it: to the widget that has the focus."""
    dlg.focus_force()
    dlg.update()
    (dlg.focus_get() or dlg).event_generate("<Escape>")
    dlg.update_idletasks()


def find(root, cls):
    return next((w for w in root.winfo_children() if isinstance(w, cls)), None)


def buttons(widget) -> dict:
    out = {}
    for w in widget.winfo_children():
        if w.winfo_class() == "TButton":
            out[str(w.cget("text"))] = w
        out.update(buttons(w))
    return out


def texts(widget) -> list:
    out = []
    for w in widget.winfo_children():
        if w.winfo_class() in ("TLabel", "TRadiobutton", "TCheckbutton") and str(w.cget("text")):
            out.append(str(w.cget("text")))
        out += texts(w)
    return out


@pytest.fixture
def ctx(ui_ctx, cfg_dir):
    return dataclasses.replace(ui_ctx, settings=Settings.load())


def header(problem=None, names=("DAPI", "Opal 520", "Opal 570", "Opal 690", "AF"), n=5):
    return TileHeader("t.tif", (n, 100, 120), "float32", n, 100, 120, 0.4984, 0.4984, True, tuple(names), problem,
                      ("#0000FF", "#00FF00", None, "#FF0000", "#FFFFFF")[:n], n * 100 * 120 * 4)


def tile_names(prefix, n, suffix="component_data"):
    return [f"{prefix}_[{1000 + 900 * i},{2000 + 700 * (i % 2)}]_{suffix}.tif" for i in range(n)]


def make_plan(root="/data/src"):
    """Rows: Good (names fine), A/B/C with DAPI x5, Count (4 names for 5 channels), Six (6 channels, duplicate) and
    a composite_image product next to Good."""
    plan = Plan(root, "/data/out")
    for folder, prefix, n, extra in (("Good_Scan1", "Good_Scan1", 12, True), ("A_Scan1", "A_Scan1", 12, False),
                                     ("B_Scan1", "B_Scan1", 12, False), ("C_Scan1", "C_Scan1", 12, False),
                                     ("Count_Scan1", "Count_Scan1", 12, False), ("Six_Scan1", "Six_Scan1", 12, False)):
        names = tile_names(prefix, n) + (tile_names(prefix, 7, "composite_image") if extra else [])
        s = summarize_listing(os.path.join(root, folder), [Entry(x, False, False) for x in names])
        for c in candidates_from_summary(s, root=root, depth=1, dir_mtime=None):
            plan.add_candidate(c)
    ids = {r.cand.folder_name: r.row_id for r in plan.rows("new")}
    plan.apply_header(ids["Good_Scan1"], header(), None, 64 << 30)
    for k in ("A_Scan1", "B_Scan1", "C_Scan1"):
        plan.apply_header(ids[k], header("duplicate", ("DAPI",) * 5), None, 64 << 30)
    plan.apply_header(ids["Count_Scan1"], header("count", ("DAPI", "x", "y", "z")), None, 64 << 30)
    plan.apply_header(ids["Six_Scan1"], header("duplicate", ("DAPI",) * 6, 6), None, 64 << 30)
    return plan, ids


# ------------------------------------------------------------------------------------------------ new folder
@pytest.mark.parametrize("name, problem", [
    ("", "Type a name"), ("   ", "Type a name"), ("a/b", "can't contain"), ("a:b", "can't contain"),
    ("tab\there", "can't contain"), ("ends.", "can't end"), ("ends ", "can't end"), ("CON", "reserved"),
    ("lpt1.txt", "reserved"), ("Stitched", "already exists"), ("stitched 2", None), ("Résultats", None)])
def test_folder_name_rules(name, problem):
    got = folder_name_error(name, frozenset({"stitched"}))
    assert (got is None) if problem is None else (problem in got)


def test_new_folder_dialog(ctx, tk_root):
    done = []
    dlg = NewFolderDialog.ask(tk_root, ctx, frozenset(), done.append)
    pump(tk_root)
    assert dlg.var.get() == "stitched" and dlg.entry.select_present()
    dlg.var.set("bad|name")
    dlg._ok()
    assert dlg.winfo_exists() and "can't contain" in str(dlg.error.cget("text")) and done == []
    dlg.var.set("Stitched images")
    buttons(dlg)["Create"].invoke()
    assert done == ["Stitched images"] and not dlg.winfo_exists()
    dlg = NewFolderDialog.ask(tk_root, ctx, frozenset({"stitched"}), done.append)
    assert dlg.var.get() == ""
    esc(dlg)
    assert done[-1] is None


# ------------------------------------------------------------------------------------------------ stop
def test_stop_dialog_modes(ctx, tk_root):
    done = []
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", 240.0, 1, False, done.append)
    pump(tk_root)
    assert str(dlg.title()) == "Stop stitching?"
    shown = texts(dlg)
    assert any("Slide_Scan1" in t and "first, then stop" in t for t in shown)
    assert "(about 4 min)  recommended" in shown and "Finished slides (1) are kept either way." in shown
    assert dlg.choice.get() == "after_current"
    buttons(dlg)["Stop"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 0, False, done.append)
    dlg.choice.set("now")
    buttons(dlg)["Stop"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 0, False, done.append)
    buttons(dlg)["Keep stitching"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 0, False, done.append)
    esc(dlg)
    assert done == ["after_current", "now", "keep", "keep"]


def test_stop_dialog_when_closing_the_window(ctx, tk_root):
    done = []
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 3, True, done.append)
    pump(tk_root)
    assert str(dlg.title()) == f"Quit {APP_NAME}?" and "Quit" in buttons(dlg) and "Stop" not in buttons(dlg)
    assert any("You can minimise the window instead." in t for t in texts(dlg))
    assert not any("recommended" in t and "about" in t for t in texts(dlg))          # no estimate yet
    dlg.choice.set("now")
    buttons(dlg)["Quit"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 3, True, done.append)
    dlg.protocol("WM_DELETE_WINDOW") and dlg.tk.call(dlg.protocol("WM_DELETE_WINDOW"))
    assert done == ["now", "keep"]


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
@pytest.mark.parametrize("closing", [False, True])
def test_long_slide_name_keeps_stop_choices_and_buttons_on_screen(ctx, tk_root, closing):
    done = []
    name = "ProjectC_" + "Extended_sample_name_" * 5 + "Scan1"
    dlg = StopDialog.open(tk_root, ctx, name, 240.0, 2, closing, done.append)
    pump(tk_root)
    assert dlg.winfo_width() <= min(ctx.theme.px(600), tk_root.winfo_screenwidth())
    assert dlg.winfo_rootx() + dlg.winfo_width() <= tk_root.winfo_screenwidth()
    assert dlg.winfo_rooty() + dlg.winfo_height() <= tk_root.winfo_screenheight()
    choices = [w for line in dlg.body.winfo_children() for w in line.winfo_children()
               if w.winfo_class() == "TRadiobutton"]
    assert len(choices) == 2 and all(name in str(w.cget("text")) for w in choices)
    choices[1].focus_force()
    choices[1].event_generate("<space>")
    pump(tk_root)
    assert dlg.choice.get() == "now"
    action = buttons(dlg)["Quit" if closing else "Stop"]
    assert action.winfo_rootx() + action.winfo_width() <= tk_root.winfo_screenwidth()
    action.invoke()
    assert done == ["now"]


def test_closing_again_while_quitting_after_the_slide(ctx, tk_root):
    """MW-19: the app already quits after the current slide. The dialog says so and offers to undo just that
    ([Don't quit]) or to quit at once; Esc changes nothing. dismiss() closes it without an answer."""
    done = []
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", 240.0, 2, True, done.append, quitting_after=True)
    pump(tk_root)
    q = ctx.theme.text("‘Slide_Scan1’")
    assert str(dlg.title()) == f"Quit {APP_NAME}?" and dlg.closing
    assert texts(dlg)[0] == f"The app will close after {q} finishes (about 4 min)."
    assert set(buttons(dlg)) == {"Don't quit", "Quit now"} and dlg.focus_lastfor() is buttons(dlg)["Don't quit"]
    assert not [w for w in dlg.body.winfo_children() if w.winfo_class() == "TRadiobutton"]
    buttons(dlg)["Don't quit"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 2, True, done.append, quitting_after=True)
    assert texts(dlg)[0] == f"The app will close after {q} finishes."
    buttons(dlg)["Quit now"].invoke()
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 2, True, done.append, quitting_after=True)
    esc(dlg)
    dlg = StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 2, False, done.append)
    dlg.dismiss()
    assert done == ["cancel_quit", "now", "keep"] and not dlg.winfo_exists()


# ------------------------------------------------------------------------------------------------ ready to stitch
WARNINGS = [Issue("warn", "channels", "2 slides have channel-name problems: A_Scan1, B_Scan1.", ("a", "b"),
                  "fix_channels"),
            Issue("warn", "replace", "1 file will be replaced: X.ome.tif."),
            Issue("warn", "recent_change", "C_Scan1: folder changed 2 min ago — is inForm still exporting?")]


def test_ready_to_stitch_sheet_outcomes(ctx, tk_root):
    done = []
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS, ["a", "b"], done.append)
    pump(tk_root)
    shown = texts(dlg)
    assert str(dlg.title()) == "Ready to stitch?"
    assert [t for t in shown if "replaced" in t or "changed" in t or "problems" in t] == [ctx.theme.text(w.text)
                                                                                         for w in WARNINGS]
    assert dlg.choice.get() == "generic" and any(t.startswith("Use generic names") for t in shown)
    assert str(dlg.start.cget("style")) == "Accent.TButton"
    dlg.start.invoke()                                                    # generic preselected -> start
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS, ["a", "b"], done.append)
    dlg.choice.set("fix")
    assert str(dlg.start.cget("text")) == ctx.theme.text("Fix names…")
    dlg.start.invoke()
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS, ["a", "b"], done.append)
    buttons(dlg)["Go back"].invoke()
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS[1:], [], done.append)
    assert not any(t.startswith("Use generic names") for t in texts(dlg))
    esc(dlg)
    assert done == ["start", "fix_channels", "back", "back"]


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_a_long_warning_list_scrolls_and_the_buttons_stay_on_screen(ctx, tk_root):
    """MW-5: 40 warnings make a sheet no taller than 60 % of the screen; the list scrolls (bar, wheel, keys) and the
    buttons stay visible. A short list shows whole, without a scroll bar."""
    done = []
    many = [Issue("warn", "recent_change", f"S{k:02d}_Scan1: folder changed 1 min ago — is inForm still exporting?")
            for k in range(40)]
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS[:1] + many, ["a", "b"], done.append)
    pump(tk_root, 100)
    sh = dlg.winfo_screenheight()
    assert dlg.winfo_height() <= ConfirmStartDialog.MAX_SCREEN * sh + 2
    assert dlg.start.winfo_rooty() + dlg.start.winfo_height() <= sh and dlg.start.winfo_ismapped()
    assert [w for w in dlg.body.winfo_children() if w.winfo_class() == "TScrollbar"]
    assert dlg.listing.yview()[0] == 0.0 < dlg.listing.yview()[1] < 1.0
    dlg.focus_force()
    pump(tk_root, 30)
    (dlg.focus_get() or dlg).event_generate("<Next>")
    assert dlg.listing.yview()[0] > 0.0
    radio = next(w for w in dlg.listing.winfo_children()[0].winfo_children() if w.winfo_class() == "TRadiobutton")
    radio.focus_force()                                                   # Tab to a control: it scrolls into view
    pump(tk_root, 50)
    assert dlg.listing.yview()[0] == 0.0
    dlg.start.invoke()
    dlg = ConfirmStartDialog.open(tk_root, ctx, WARNINGS, ["a", "b"], done.append)
    pump(tk_root, 50)
    assert dlg.listing.yview() == (0.0, 1.0) and not [w for w in dlg.body.winfo_children()
                                                     if w.winfo_class() == "TScrollbar"]
    buttons(dlg)["Go back"].invoke()
    assert done == ["start", "back"]


# ------------------------------------------------------------------------------------------------ channel names
@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_many_issues_keep_close_and_keyboard_actions_visible(ctx, tk_root):
    done = []
    rows = [("warn", f"Project {i}: tiles changed recently. Check that the export has finished before stitching.",
             (("Show", f"show_{i}"),)) for i in range(24)]
    dlg = IssuesDialog.open(tk_root, ctx, rows, done.append)
    pump(tk_root, 60)
    close = buttons(dlg)["Close"]
    assert dlg.winfo_height() <= tk_root.winfo_screenheight() * .6 + 2
    assert close.winfo_rooty() + close.winfo_height() <= tk_root.winfo_screenheight()
    actions = [w for line in dlg.listing.winfo_children()[0].winfo_children()
               for w in line.winfo_children() if w.winfo_class() == "TButton"]
    dlg.focus_force()
    close.event_generate("<Next>")
    assert dlg.listing.yview()[0] > 0
    actions[-1].focus_force()
    pump(tk_root, 50)
    assert actions[-1].winfo_rooty() >= dlg.listing.winfo_rooty()
    assert actions[-1].winfo_rooty() + actions[-1].winfo_height() <= (
        dlg.listing.winfo_rooty() + dlg.listing.winfo_height())
    actions[-1].invoke()
    assert done == ["show_23"]


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_long_cleanup_confirmation_keeps_cancel_visible(ctx, tk_root):
    from yfs.ui.app import _ConfirmDialog

    names = ["ProjectC_" + "Extended_sample_name_" * 9 + f"{i}.tmp.raw (2.4 GB)" for i in range(10)]
    text = ("These files were left behind by an unfinished run:\n" + "\n".join(names) +
            "\n\nFiles changed in the last 2 minutes are kept (another stitcher may be using them).")
    done = []
    dlg = _ConfirmDialog(tk_root, ctx, "Delete leftover files?", text, "Delete them", done.append)
    pump(tk_root)
    assert dlg.winfo_height() <= tk_root.winfo_screenheight() * .6 + 2
    cancel = buttons(dlg)["Cancel"]
    assert cancel.winfo_rooty() + cancel.winfo_height() <= tk_root.winfo_screenheight()
    assert ctx.theme.text(text) in texts(dlg)
    dlg.focus_force()
    cancel.event_generate("<Next>")
    assert dlg.listing.yview()[0] > 0
    cancel.invoke()
    assert done == [None]


def test_channel_names_entries_copy_and_same_problem(ctx, tk_root):
    plan, ids = make_plan()
    done = []
    ChannelNamesDialog.open(tk_root, ctx, plan, [ids["A_Scan1"], ids["B_Scan1"], ids["C_Scan1"], ids["Count_Scan1"]],
                            done.append)
    dlg = find(tk_root, ChannelNamesDialog)
    pump(tk_root)
    assert len(dlg.entries) == 5 and [v.get() for v in dlg.vars] == ["DAPI"] * 5
    assert str(dlg.title()) == "Channel names · A_Scan1 (and 2 more with the same problem)"
    assert any('every channel is called' in t and "DAPI" in t for t in texts(dlg))
    assert [s[0] for s in dlg.sources] == ["Good_Scan1 (5 channels)"]
    dlg.copy.current(0)
    dlg.copy.event_generate("<<ComboboxSelected>>")
    assert [v.get() for v in dlg.vars] == ["DAPI", "Opal 520", "Opal 570", "Opal 690", "AF"]
    assert dlg.also.get() is True
    buttons(dlg)["Save names"].invoke()
    assert done == [True]
    for k in ("A_Scan1", "B_Scan1", "C_Scan1"):
        assert plan.row(ids[k]).channel_override == ("DAPI", "Opal 520", "Opal 570", "Opal 690", "AF")
    assert plan.row(ids["Count_Scan1"]).channel_override is None          # other problem: not included


def test_channel_names_only_the_first_row_generic_and_validation(ctx, tk_root):
    plan, ids = make_plan()
    done = []
    rows = [ids["A_Scan1"], ids["B_Scan1"]]
    ChannelNamesDialog.open(tk_root, ctx, plan, rows, done.append)
    dlg = find(tk_root, ChannelNamesDialog)
    dlg.also.set(False)
    dlg.vars[2].set("   ")
    buttons(dlg)["Save names"].invoke()
    assert dlg.winfo_exists() and str(dlg.error.cget("text")) == "Every channel needs a name." and done == []
    for i, v in enumerate(dlg.vars):
        v.set(f"Marker {i}")
    buttons(dlg)["Save names"].invoke()
    assert plan.row(ids["A_Scan1"]).channel_override == tuple(f"Marker {i}" for i in range(5))
    assert plan.row(ids["B_Scan1"]).channel_override is None
    ChannelNamesDialog.open(tk_root, ctx, plan, [ids["B_Scan1"], ids["C_Scan1"]], done.append)
    dlg = find(tk_root, ChannelNamesDialog)
    assert "A_Scan1 (5 channels)" in [s[0] for s in dlg.sources]           # names set earlier in this session
    buttons(dlg)["Use generic names (Channel 0-4)"].invoke()
    for k in ("B_Scan1", "C_Scan1"):
        assert plan.row(ids[k]).generic_accepted and plan.row(ids[k]).channel_override is None
    ChannelNamesDialog.open(tk_root, ctx, plan, [ids["Six_Scan1"]], done.append)
    dlg = find(tk_root, ChannelNamesDialog)
    assert len(dlg.entries) == 6 and not dlg.same
    buttons(dlg)["Cancel"].invoke()
    ChannelNamesDialog.open(tk_root, ctx, plan, [ids["Good_Scan1"]], done.append)   # no problem: nothing to ask
    assert done == [True, True, False, False] and find(tk_root, ChannelNamesDialog) is None


# ------------------------------------------------------------------------------------------------ slide details
def test_slide_details_single_instance_and_ignored_products(ctx, tk_root):
    plan, ids = make_plan()
    fixes = []
    dlg = SlideDetailsDialog.show(tk_root, ctx, plan, ids["Good_Scan1"], fixes.append)
    pump(tk_root)
    assert str(dlg.title()) == "Slide details · Good_Scan1"
    assert str(dlg.extra.cget("text")) == "also 7 composite_image tiles (ignored)"
    assert dlg.facts.map.find_all()                                      # the tile map is drawn
    again = SlideDetailsDialog.show(tk_root, ctx, plan, ids["A_Scan1"], fixes.append)
    assert again is dlg and str(dlg.title()) == "Slide details · A_Scan1" and str(dlg.extra.cget("text")) == ""
    fix = next(b for t, b in buttons(dlg).items() if t.startswith("Fix channel names"))
    fix.invoke()
    assert fixes == [[ids["A_Scan1"]]]
    plan.set_channels([ids["A_Scan1"]], ["a", "b", "c", "d", "e"])
    SlideDetailsDialog.refresh_rows([ids["A_Scan1"]])
    assert any("Channel names set by you" in t for t in texts(dlg))
    buttons(dlg)["Close"].invoke()
    assert SlideDetailsDialog.instance is None and not dlg.winfo_exists()


# ------------------------------------------------------------------------------------------------ settings
def test_settings_are_saved_when_the_dialog_closes(ctx, tk_root, tmp_path):
    done = []
    dlg = SettingsDialog.open(tk_root, ctx, False, done.append)
    pump(tk_root)
    dlg.scale.set("150%")
    dlg.naming.set("slide")
    dlg.depth.set("5")
    dlg.temp_mode.set("folder")
    buttons(dlg)["Save"].invoke()
    assert dlg.winfo_exists() and "Choose a folder" in str(dlg.error.cget("text"))   # temp folder missing
    dlg.temp.set(str(tmp_path / "scratch"))
    dlg.csv.set(False)
    buttons(dlg)["Save"].invoke()
    assert done == [{"ui.scale", "discovery.naming", "discovery.max_depth", "run.temp_dir", "run.summary_csv"}]
    assert ctx.settings.save_now(2)
    saved = Settings.load()
    assert (saved.get("ui.scale"), saved.get("discovery.naming"), saved.get("discovery.max_depth"),
            saved.get("run.temp_dir"), saved.get("run.summary_csv")) == (1.5, "slide", 5, str(tmp_path / "scratch"),
                                                                         False)
    dlg = SettingsDialog.open(tk_root, ctx, False, done.append)
    assert (dlg.scale.get(), dlg.naming.get(), dlg.depth.get(), dlg.temp_mode.get()) == ("150%", "slide", "5",
                                                                                         "folder")
    dlg.temp_mode.set("next")
    buttons(dlg)["Save"].invoke()
    assert done[-1] == {"run.temp_dir"} and ctx.settings.get("run.temp_dir") is None
    dlg = SettingsDialog.open(tk_root, ctx, False, done.append)
    dlg.scale.set("200%")
    buttons(dlg)["Cancel"].invoke()
    assert done[-1] == set() and ctx.settings.get("ui.scale") == 1.5


def test_settings_are_read_only_during_a_run(ctx, tk_root):
    done = []
    dlg = SettingsDialog.open(tk_root, ctx, True, done.append)
    pump(tk_root)
    assert "You can change settings after this batch." in texts(dlg)
    assert all(w.instate(["disabled"]) for w in dlg.controls)
    assert "Close" in buttons(dlg) and "Save" not in buttons(dlg)
    buttons(dlg)["Close"].invoke()
    assert done == [set()]


def test_settings_temp_folder_browse_opens_the_browser_and_regrabs(ctx, tk_root, monkeypatch, tmp_path):
    """BR-20: the temporary-folder chooser is the browser's own 'temp' mode, which opens where it was last used."""
    from yfs.ui import browser as browser_mod
    opened = {}

    def fake_open(master, ctx_, *, mode, initial, on_done, title=None):
        opened.update(master=master, mode=mode, initial=initial, title=title, on_done=on_done)
    monkeypatch.setattr(browser_mod.FolderBrowser, "open", staticmethod(fake_open))
    ctx.settings.set("paths.browser_last.output", str(tmp_path / "Stitched"))
    ctx.settings.set("paths.browser_last.temp", str(tmp_path / "Scratch"))
    dlg = SettingsDialog.open(tk_root, ctx, False, lambda keys: None)
    pump(tk_root)
    buttons(dlg)[ctx.theme.text("Browse…")].invoke()
    assert (opened["master"], opened["mode"], opened["title"]) == (dlg, "temp", None)
    assert opened["initial"] == (str(tmp_path / "Scratch"), None)
    opened["on_done"](str(tmp_path))
    pump(tk_root, 120)
    assert dlg.temp.get() == str(tmp_path) and dlg.temp_mode.get() == "folder"
    assert dlg.grab_current() is dlg                                     # the dialog is modal again
    dlg.destroy()


def test_settings_refuse_a_temporary_folder_that_is_not_a_full_path(ctx, tk_root, tmp_path, monkeypatch):
    """A typed 'scratch' would be resolved against whatever folder the app was started from (UD6): it is refused
    with a hint; '~/scratch' is taken as the home folder's."""
    for name in ("HOME", "USERPROFILE"):                     # (Windows takes the home folder from USERPROFILE)
        monkeypatch.setenv(name, str(tmp_path))
    done = []
    dlg = SettingsDialog.open(tk_root, ctx, False, done.append)
    dlg.temp_mode.set("folder")
    dlg.temp.set("scratch")
    buttons(dlg)["Save"].invoke()
    assert dlg.winfo_exists() and "full path" in str(dlg.error.cget("text")) and done == []
    dlg.temp.set("~/scratch")
    buttons(dlg)["Save"].invoke()
    temp = ctx.settings.get("run.temp_dir")
    assert done == [{"run.temp_dir"}] and temp == os.path.expanduser("~/scratch") and temp.startswith(str(tmp_path))


# ------------------------------------------------------------------------------------------------ Tk objects (CT-6)
def test_closed_dialogs_free_their_tk_variables_on_the_tk_thread(ctx, tk_root, monkeypatch):
    """CT-6: a closed dialog's Variables die when it closes, on the Tk thread. Left in reference cycles they were
    finalised by the next collection on any thread, and a Tcl call from a worker can freeze the app."""
    made, threads = [], []
    real_init, real_del = tk.Variable.__init__, tk.Variable.__del__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        made.append(self._name)

    def spy(self):
        threads.append(threading.current_thread().name)
        if threading.current_thread() is threading.main_thread():
            real_del(self)                                   # (off the Tk thread the Tcl call would block)
    monkeypatch.setattr(tk.Variable, "__init__", init)
    monkeypatch.setattr(tk.Variable, "__del__", spy)
    plan, ids = make_plan()
    issues = [("info", "Choose where to save the stitched images.", (("Browse…", "browse_output"),))]

    def channels() -> ChannelNamesDialog:
        ChannelNamesDialog.open(tk_root, ctx, plan, [ids["A_Scan1"], ids["B_Scan1"]], lambda r: None)
        return find(tk_root, ChannelNamesDialog)
    dialogs = [(lambda: NewFolderDialog.ask(tk_root, ctx, frozenset(), lambda r: None), "_cancel"),
               (lambda: StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 0, False, lambda r: None), "_cancel"),
               (lambda: StopDialog.open(tk_root, ctx, "Slide_Scan1", None, 0, True, lambda r: None,
                                        quitting_after=True), "dismiss"),
               (lambda: ConfirmStartDialog.open(tk_root, ctx, WARNINGS, ["a"], lambda r: None), "_cancel"),
               (channels, "_cancel"),
               (lambda: SlideDetailsDialog.show(tk_root, ctx, plan, ids["Good_Scan1"], lambda r: None), "_cancel"),
               (lambda: SettingsDialog.open(tk_root, ctx, False, lambda r: None), "_cancel"),
               (lambda: InfoDialog.show(tk_root, ctx, "about"), "_cancel"),
               (lambda: IssuesDialog.open(tk_root, ctx, issues, lambda r: None), "_cancel")]
    gc.disable()                                             # only explicit collections, so nothing hides a leak
    try:
        for open_, close in dialogs:
            dlg = open_()                                    # held, as callers hold the dialogs they open
            getattr(dlg, close)()
            assert len(threads) == len(made), type(dlg).__name__    # its own close freed its Variables all the same
        del dlg
        worker = threading.Thread(target=gc.collect, name="worker")
        worker.start()
        worker.join(10)
    finally:
        gc.enable()
    assert len(made) >= 16 and threads == ["MainThread"] * len(made), (len(made), threads)


# ------------------------------------------------------------------------------------------------ info
def test_how_it_works_shows_appendix_b_word_for_word(ctx, tk_root):
    dlg = InfoDialog.show(tk_root, ctx, "how")
    pump(tk_root)
    shown = dlg.text.get("1.0", "end")
    start = shown.index("Welcome to your FAVOURITE Stitcher!")
    got = [line.rstrip() for line in shown[start:].splitlines()[:len(APPENDIX_B.splitlines())]]
    assert got == APPENDIX_B.splitlines()
    assert shown.index("From Jacky") < start
    assert "What happens when you press Stitch" in shown
    assert "Ctrl+Shift+N" in shown and "Ctrl+J" in shown                   # both shortcut tables
    assert "Shift+F10 or the Menu key\tThe menu of the focused plan row (Try again, Finalize" in shown
    assert str(dlg.text.cget("state")) == "disabled"


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_how_it_works_keeps_shortcuts_aligned_and_close_on_screen(ctx, tk_root, monkeypatch):
    dlg = InfoDialog.show(tk_root, ctx, "how")
    pump(tk_root)
    assert dlg.winfo_rooty() + dlg.winfo_height() <= dlg.winfo_screenheight()
    close = dlg.close_button
    assert close.winfo_rooty() + close.winfo_height() <= dlg.winfo_screenheight()
    columns, wrapped = set(), 0
    text = dlg.text
    for number, line in enumerate(text.get("1.0", "end-1c").splitlines(), 1):
        if "\t" not in line:
            continue
        tab = line.index("\t")
        action = f"{number}.{tab + 1}"
        text.yview(f"{number}.0")
        text.update_idletasks()
        key_end, start = text.bbox(f"{number}.{tab - 1}"), text.bbox(action)
        assert key_end is not None and start is not None
        assert start[0] - key_end[0] - key_end[2] >= ctx.theme.px(12)
        columns.add(start[0])
        continuation = text.index(f"{action} display lineend + 1c")
        if text.compare(continuation, "<", f"{number}.end"):
            wrapped += 1
            assert text.bbox(continuation)[0] == start[0]     # wrapped actions stay out of the key column
    assert len(columns) == 1 and wrapped
    monkeypatch.setattr("yfs.ui.dialogs.diagnostics", lambda ctx: [("Version", "example")])
    dlg.page("about")
    pump(tk_root)
    dlg.geometry(f"+0+{dlg.winfo_screenheight() - dlg.winfo_height()}")
    pump(tk_root)
    dlg.page("how")
    pump(tk_root)
    assert dlg.winfo_rooty() + dlg.winfo_height() <= dlg.winfo_screenheight()
    close.invoke()
    assert not dlg.winfo_exists()


def test_about_and_single_instance(ctx, tk_root):
    dlg = InfoDialog.show(tk_root, ctx, "how")
    again = InfoDialog.show(tk_root, ctx, "about")
    pump(tk_root)
    assert again is dlg and str(dlg.title()) == "About"
    shown = texts(dlg)
    assert APP_NAME in shown and "It's your Favourite Stitcher" in shown and f"Version {__version__}" in shown
    assert any(t.startswith("Made by Jacky @ Ramachandran Lab") for t in shown)
    for label in ("Python", "Tk", "numpy", "tifffile", "imagecodecs", "psutil", "Settings", "Logs"):
        assert label in shown
    assert {"GitHub page", "Copy diagnostics"} <= set(buttons(dlg))
    buttons(dlg)["Copy diagnostics"].invoke()
    clip = dlg.clipboard_get()
    assert "tifffile: " in clip and "OS: " in clip and "Interface size: " in clip and "Last run: " in clip
    buttons(dlg)["Close"].invoke()
    assert InfoDialog.instance is None
