"""The folder browser by keyboard and by pointer (BUX-M1, BR-13, BR-14, BR-17, BR-18): the arrows work at once after
every navigation, the sidebar and the footer's controls are reachable, a card button that goes away takes the
keyboard to the listing, F5 keeps the place in the list, and a click returns the target to the folder shown."""
from __future__ import annotations

import itertools
import sys

import pytest

from test_browser import (env, key, mounts, overlay_buttons, pump, q, select, settled, tree, until,  # noqa: F401
                          zenity)


def names(b, keys) -> list:
    return [b._rows[k].name for k in keys]


_CLOCK = itertools.count(10_000, 1_000)                # event times a second apart: never a double-click


def click(b, widget, x: int, y: int, state: int = 0) -> None:
    """A real mouse click (press and release) at x, y of `widget`."""
    widget.event_generate("<Motion>", x=x, y=y)
    widget.event_generate("<ButtonPress-1>", x=x, y=y, state=state, time=next(_CLOCK))
    widget.event_generate("<ButtonRelease-1>", x=x, y=y, state=state, time=next(_CLOCK))
    b.update()


def click_row(b, name: str, state: int = 0) -> None:
    box = b.tree.bbox(b._names[name])
    click(b, b.tree, box[0] + b.px(40), box[1] + box[3] // 2, state)


def focus(b) -> str:
    return str(b.tk.call("focus"))


@pytest.fixture
def folders(tree):
    for name in ("alpha", "beta", "gamma"):
        (tree / "Work" / name / "sub").mkdir(parents=True)
    return tree / "Work"


# ------------------------------------------------------------------------------------------------ BUX-M1, BR-21
def test_down_and_up_select_a_row_at_once_after_every_navigation(env, folders):
    """BUX-M1: nothing is selected after a navigation (the folder shown is the target), yet Down selects the
    first row and Up the last one; Enter into a folder, then Down, selects its first row."""
    b = env.open((str(folders), None))
    until(b, lambda: settled(b))
    assert b.tree.selection() == () and focus(b) == str(b.tree)            # the keyboard is in the listing
    key(b.tree, "<Down>")
    assert names(b, b.tree.selection()) == ["alpha"]
    key(b.tree, "<Down>")
    assert names(b, b.tree.selection()) == ["beta"]                        # Tk's own navigation from there
    key(b.tree, "<Escape>")                                                # clears the selection
    key(b.tree, "<Down>")
    assert names(b, b.tree.selection()) == ["beta"]                        # the row the keyboard was on
    key(b.tree, "<Return>")
    until(b, lambda: b.current() == str(folders / "beta") and settled(b))
    assert b.tree.selection() == () and focus(b) == str(b.tree)
    key(b.tree, "<Up>")
    assert names(b, b.tree.selection()) == ["sub"]


def test_the_sidebar_works_by_keyboard(env, folders, tree):
    """BUX-M1: in the sidebar the place of the folder shown has the keyboard; Up/Down move between places over
    the section titles; Enter opens one and the keyboard goes on to its listing."""
    (tree / "share" / "P").mkdir(parents=True)
    ctx = env.ctx(mounts(f"2 1 0:40 / {tree / 'share'} rw - cifs //srv.example.org/share rw"))
    b = env.open((str(folders), None), ctx=ctx)
    until(b, lambda: settled(b) and b._place_items)
    side, label = b._side, lambda i: b._place_items[i][0].label
    assert label(side.focus()) == "Computer" and side.selection() == (side.focus(),)    # where Work is
    key(side, "<Down>")
    assert label(side.focus()) == "share" and side.selection() == (side.focus(),)       # over NETWORK
    for _ in range(3):
        key(side, "<Up>")
    assert label(side.focus()) == "Home" and side.selection() == (side.focus(),)        # the first place
    key(side, "<Down>")
    key(side, "<Down>")
    key(side, "<Return>")
    until(b, lambda: b.current() == str(tree / "share") and settled(b))
    assert focus(b) == str(b.tree)
    key(b.tree, "<Down>")
    assert names(b, b.tree.selection()) == ["P"]


# ------------------------------------------------------------------------------------------------ BR-13
def tab_chain(b) -> list:
    """The widgets Tab visits from the listing until it comes back to it."""
    seen, w = [], b.tree
    for _ in range(40):
        w = w.tk_focusNext()
        if w is None or w is b.tree:
            break
        seen.append(w)
    return seen


@pytest.mark.skipif(sys.platform != "linux", reason="zenity is the Linux system picker")
def test_footer_options_strip_actions_and_the_picker_are_reachable(env, tree, zenity):
    """BR-13: the options, the strip's action and the picker link are in the Tab order, and Alt+T / Alt+H /
    Alt+A / Alt+P reach them without Tab."""
    slide = tree / "S1_Scan1"
    slide.mkdir()
    (slide / "S1_Scan1_[1000,2000]_component_data.tif").write_bytes(b"")
    b = env.open((str(tree), "S1_Scan1"), mode="output")
    until(b, lambda: settled(b) and b._strip_btn.winfo_manager() and b._picker.winfo_manager())
    chain = tab_chain(b)
    checks = [str(w.cget("text")) for w in chain if w.winfo_class() == "TCheckbutton"]
    assert checks == ["Show hidden", "Show all files"]                     # output mode: no 'Only folders...'
    assert b._strip_btn in chain and b._picker in chain
    assert chain.index(b._strip_btn) < chain.index(b._picker) < chain.index(b._primary)
    key(b.tree, "<Alt-a>")
    assert b._v_all.get() is True
    b.cancel()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    key(b.tree, "<Alt-t>")
    assert b._v_tiles.get() is True
    key(b.tree, "<Alt-h>")
    assert b._v_hidden.get() is True


def test_the_strip_action_runs_by_its_letter_while_a_path_is_typed(env, folders):
    """BR-13: after a mistyped path the keyboard stays in the path box, where Tab completes names, so [Go there]
    has its letter underlined like the options': Alt+G goes there and the keyboard goes on to the listing."""
    b = env.open((str(folders), None))
    until(b, lambda: settled(b))
    b.edit_path(str(folders / "beta" / "nope"))
    key(b._edit, "<Return>")
    until(b, lambda: b._strip_btn.winfo_manager(), what="[Go there]")
    assert str(b._strip_btn.cget("text")) == "Go there" and str(b._strip_btn.cget("underline")) == "0"
    assert focus(b) == str(b._edit)
    key(b._edit, "<Alt-x>")                                                # another letter: nothing happens
    assert b._editing and b.current() == str(folders)
    key(b._edit, "<Alt-g>")
    until(b, lambda: b.current() == str(folders / "beta") and settled(b), what="Go there")
    assert not b._editing and focus(b) == str(b.tree)


# ------------------------------------------------------------------------------------------------ BR-14
def test_a_card_button_that_goes_away_takes_the_keyboard_to_the_listing(env, synth_tree):
    """BR-14: after a click on Go back, the keyboard is in the listing (not on the hidden button): Space doesn't
    go back again, and the arrows move in the list."""
    b = env.open((str(synth_tree), None))
    until(b, lambda: settled(b))
    b.navigate(str(synth_tree / "locked"))
    until(b, lambda: b._ov_kind == "error")
    assert overlay_buttons(b) == ["Go back"]
    back = next(w for f in b._ov.winfo_children() for w in f.winfo_children() if w.winfo_class() == "TButton")
    b.update()
    assert b.tree.tk_focusNext() is back                                   # Tab from the listing reaches it
    back.focus_force()                                                     # (a mouse click focuses a ttk button)
    click(b, back, 5, 5)
    until(b, lambda: b.current() == str(synth_tree) and settled(b))
    assert focus(b) == str(b.tree) and not b._ov.winfo_manager()
    key(b.tree, "<space>")
    pump(b, 100)
    assert b.current() == str(synth_tree)
    key(b.tree, "<Down>")
    assert b.tree.selection()
    b._overlay("picker", "wait", "Waiting for the system folder picker…", "", [("Cancel", b._kill_picker)])
    card = [w for f in b._ov.winfo_children() for w in f.winfo_children() if w.winfo_class() == "TButton"]
    card[0].focus_force()
    b.update()
    b._overlay("unresponsive", "warn", "X isn't answering.", "", [("Keep waiting", b._keep), ("Go back", b.back)])
    b.update()                                                              # (once it is mapped)
    card = [w for f in b._ov.winfo_children() for w in f.winfo_children() if w.winfo_class() == "TButton"]
    assert focus(b) == str(card[0])                                         # a card replacing a card: its button
    card[0].invoke()                                                         # Keep waiting hides the card, no
    assert focus(b) == str(b.tree) and not b._ov.winfo_manager()            # navigation: the keyboard goes on


# ------------------------------------------------------------------------------------------------ BR-17
def test_refresh_keeps_the_scroll_position_and_the_selection(env, tree):
    """BR-17: F5 in a long folder doesn't throw the list back to the top, and the selected row stays selected."""
    for i in range(200):
        (tree / f"d{i:03d}").mkdir()
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    b.tree.yview_moveto(0.5)
    b.update()
    select(b, "d105")
    top = b.tree.yview()[0]
    assert 0.4 < top < 0.6
    key(b.tree, "<F5>")
    until(b, lambda: settled(b) and b.tree.selection(), what="the refreshed list")
    pump(b, 100)
    assert abs(b.tree.yview()[0] - top) <= 1 / 200 and names(b, b.tree.selection()) == ["d105"]


# ------------------------------------------------------------------------------------------------ BR-18
def test_a_click_returns_the_target_to_the_folder_shown(env, folders):
    """BR-18: a click on the list's empty space, on the current folder's crumb, or a Ctrl+click on the selected
    row: the folder shown is the target again, as in a file manager."""
    b = env.open((str(folders), None))
    until(b, lambda: settled(b))
    b.geometry(f"{b.px(1080)}x{b.px(680)}")
    pump(b, 200)
    use = f"Use {q(b, 'Work')}"
    for how in ("empty", "crumb", "ctrl"):
        click_row(b, "beta")
        assert b.target() == str(folders / "beta") and str(b._primary.cget("text")) == f"Use {q(b, 'beta')}"
        listing = b._rows                                                 # (a reload makes new rows)
        if how == "empty":
            click(b, b.tree, b.px(60), b.tree.winfo_height() - b.px(10))
        elif how == "crumb":                                              # (no reload: F5 is for that)
            b._crumb_ws[-1][1].invoke()
        else:
            click_row(b, "beta", state=0x20004 if sys.platform == "win32" else 0x4)
        assert b.tree.selection() == () and b.target() == str(folders), how
        assert str(b._primary.cget("text")) == use and b._rows is listing
