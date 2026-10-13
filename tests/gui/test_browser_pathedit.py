"""Typing a path in the folder browser (BR-5, BR-6, BR-16, BR-23, BR-29): what is typed is what Enter and Ctrl+Enter
use, a mistyped path stays to be corrected, and the suggestions can be found and look like the listing."""
from __future__ import annotations

import os

import pytest

from test_browser import env, key, pump, q, settled, strip, tree, until  # noqa: F401


@pytest.fixture
def lab(tree):
    """tree/Lab with ProjectC (and its MultiplexOptim, PC-01_Scan1, PC-02_Scan1), Archive and Proteomics."""
    for p in ("ProjectC/MultiplexOptim", "ProjectC/PC-01_Scan1", "ProjectC/PC-02_Scan1", "Archive", "Proteomics"):
        (tree / "Lab" / p).mkdir(parents=True)
    return tree / "Lab"


def typed(b, text: str) -> None:
    """Type `text` at the end of the path Entry, key by key (each key fires the Entry's bindings)."""
    for ch in text:
        b._edit.insert("end", ch)
        b._edit.event_generate("<KeyRelease>", keysym="slash" if ch == "/" else ch)
        b.update()


def selected_text(b) -> str:
    e = b._edit
    return e.get()[e.index("sel.first"):e.index("sel.last")] if e.selection_present() else ""


# ------------------------------------------------------------------------------------------------ BR-5
@pytest.mark.parametrize("mode", ["input", "output"])
def test_ctrl_enter_uses_the_typed_folder(env, lab, mode):
    """BR-5: Ctrl+Enter in path edit uses the folder typed (after the save check in output mode), not the folder
    that was showing or selected before."""
    b = env.open((str(lab), "ProjectC"), mode=mode)
    until(b, lambda: settled(b) and b.tree.selection())
    b.edit_path(str(lab / "Archive"))
    key(b._edit, "<Control-Return>")
    until(b, lambda: env.done, what="the typed folder chosen")
    assert env.done == [str(lab / "Archive")]


def test_ctrl_enter_on_a_missing_path_keeps_the_dialog_open(env, lab):
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path(str(lab / "Archiv"))
    key(b._edit, "<Control-Return>")
    until(b, lambda: strip(b)[0] == "Error")
    assert env.done == [] and b.winfo_exists() and b._editing


def test_ctrl_enter_on_the_folder_shown_uses_it(env, lab):
    """The Entry starts with the folder shown: Ctrl+Enter uses it, even with a subfolder selected below."""
    b = env.open((str(lab), "ProjectC"))
    until(b, lambda: settled(b) and b.tree.selection())
    key(b.tree, "<Control-l>")
    key(b._edit, "<Control-Return>")
    assert env.done == [str(lab)]


# ------------------------------------------------------------------------------------------------ BR-6
def test_enter_opens_what_was_typed_unless_a_suggestion_was_chosen(env, lab):
    """BR-6: Down opens the suggestions with none chosen; typing on keeps none chosen, so Enter opens the typed
    folder. Down, Down, Enter opens the first suggestion."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path(str(lab / "Pro"))
    key(b._edit, "<Down>")
    assert b._lb_visible() and b._lb.selection() == ()
    typed(b, "jectC/")
    until(b, lambda: [b._lb.item(i, "text") for i in b._lb.get_children()][:1] == ["MultiplexOptim"],
          what="the suggestions follow the text")
    assert b._lb.selection() == ()
    key(b._edit, "<Return>")
    until(b, lambda: b.current() == str(lab / "ProjectC") and settled(b))
    b.edit_path(str(lab / "Pro"))
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")
    assert b._lb.selection() == ("0",)
    key(b._edit, "<Up>")                                                    # back to the typed text
    assert b._lb.selection() == ()
    key(b._edit, "<Down>")
    key(b._edit, "<Return>")
    until(b, lambda: b.current() == str(lab / "ProjectC") and settled(b), what="the first suggestion")
    b.edit_path(str(lab / "Pro"))
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")                                                 # ProjectC chosen, then a key typed:
    typed(b, "t")                                                          # Enter at once (before the list
    key(b._edit, "<Return>")                                               # follows) takes 'Prot'
    until(b, lambda: strip(b)[0] == "Error")
    assert strip(b)[1] == f"Can't find {q(b, 'Prot')} in {q(b, 'Lab')}." and b.current() == str(lab / "ProjectC")


def test_a_fast_typist_gets_what_was_typed_and_keeps_what_was_chosen(env, lab):
    """BR-6 at typing speed: a key typed after Down chose a suggestion is typing, even when Enter comes while that
    key is still down (its release not seen yet); and Down pressed just after a key keeps its choice when the
    suggestions follow the text a moment later."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path(str(lab / "Pro"))
    key(b._edit, "<Down>")
    key(b._edit, "<Down>")
    assert b._lb.selection() == ("0",)                                     # ProjectC chosen
    b._edit.insert("end", "t")                                             # 't' pressed (no release yet)...
    key(b._edit, "<Return>")                                               # ...and Enter: the text wins
    until(b, lambda: strip(b)[0] == "Error")
    assert strip(b)[1] == f"Can't find {q(b, 'Prot')} in {q(b, 'Lab')}." and b.current() == str(lab)
    b.edit_path(str(lab / "Pr"))
    key(b._edit, "<Down>")                                                 # the list shows,
    typed(b, "o")                                                          # a key follows it (120 ms later),
    key(b._edit, "<Down>")                                                 # and Down chooses at once:
    pump(b, 300)
    assert b._lb.selection() == ("0",) and b._lb.item("0", "text") == "ProjectC"     # still chosen


# ------------------------------------------------------------------------------------------------ BR-16
def test_a_mistyped_path_stays_to_be_corrected(env, lab):
    """BR-16: path editing goes on with the text, the part that doesn't exist selected; the strip names it and
    the folder it isn't in, once, and offers to go there (a click on it works: it doesn't end path editing first).
    Typing again clears the message; so does Esc, which leaves path editing."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path(str(lab / "ProjectC" / "PC-03_Scna1" / "stitched"))
    height = b._strip.winfo_reqheight()
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error")
    assert b._strip.winfo_reqheight() == height                              # an action shows: no jump
    assert strip(b)[1] == f"Can't find {q(b, 'PC-03_Scna1')} in {q(b, 'ProjectC')}."
    assert b._editing and b._edit.get() == str(lab / "ProjectC" / "PC-03_Scna1" / "stitched")
    assert selected_text(b) == "PC-03_Scna1" and str(b.focus_get()) == str(b._edit)
    assert str(b._strip_btn.cget("text")) == "Go there" and b.current() == str(lab)
    key(b._edit, "<Control-l>")                                              # (as in an address bar: selects it)
    assert b._edit.get() == str(lab / "ProjectC" / "PC-03_Scna1" / "stitched") and selected_text(b) == b._edit.get()
    b._edit.icursor("end")
    b._edit.selection_clear()
    typed(b, "x")                                                            # the message was about the old text
    assert strip(b)[0] == "Info" and "Tab completes a name" in strip(b)[1]
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error")
    key(b._edit, "<Escape>")
    assert not b._editing and strip(b)[0] != "Error" and str(b.focus_get()) == str(b.tree)
    b.edit_path(str(lab / "ProjectC" / "nope"))
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error")
    go = b._strip_btn                                                        # Go there, clicked by a hand: the
    go.event_generate("<ButtonPress-1>", x=5, y=5)                           # press gives it the keyboard, and
    pump(b, 100)                                                             # it must still be there for the
    assert go.winfo_ismapped() and b._editing                                # release
    go.event_generate("<ButtonRelease-1>", x=5, y=5)
    until(b, lambda: b.current() == str(lab / "ProjectC") and not b._editing and strip(b)[0] != "Error")


def test_a_name_typed_alone_is_looked_for_in_the_folder_shown(env, lab):
    """A name (or a relative path) is taken in the folder shown, as a file manager's address bar does; Tab
    completes it there too. A missing one says which folder it isn't in."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path("Proj")
    key(b._edit, "<Tab>")
    assert b._edit.get() == os.path.join(str(lab / "ProjectC"), "")             # completed where it is
    b.edit_path("ProjectC/MultiplexOptim")
    key(b._edit, "<Return>")
    until(b, lambda: b.current() == str(lab / "ProjectC" / "MultiplexOptim") and b._listing is not None)
    b.edit_path("../Archive")
    key(b._edit, "<Return>")
    until(b, lambda: strip(b)[0] == "Error")
    assert strip(b)[1] == f"Can't find {q(b, 'Archive')} in {q(b, 'ProjectC')}."
    assert b._edit.get() == str(lab / "ProjectC" / "Archive")


# ------------------------------------------------------------------------------------------------ BR-23, BR-29
def test_the_suggestions_look_like_the_listing_and_line_up_with_the_entry(env, lab):
    """BR-23: the suggestion list starts at the Entry's left edge, just under it; each row has the folder's icon,
    and its badge in a column of its own."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    b.edit_path(str(lab) + "/")
    key(b._edit, "<Down>")
    lb = b._lb
    b.update()
    assert abs(lb.winfo_rootx() - b._edit.winfo_rootx()) <= 1
    assert 0 <= lb.winfo_rooty() - (b._edit.winfo_rooty() + b._edit.winfo_height()) <= 1
    rows = {lb.item(i, "text"): (lb.set(i, "badge"), lb.item(i, "image")) for i in lb.get_children()}
    assert set(rows) == {"Archive", "ProjectC", "Proteomics"}
    assert rows["ProjectC"][0] == b.T("No tiles · 3 folders") and all(image for _badge, image in rows.values())


def test_an_ambiguous_tab_shows_the_choices_and_the_strip_says_how(env, lab):
    """BR-29: while editing, the strip says what Enter, Ctrl+Enter, Tab and Down do; a Tab that can't complete
    anything more opens the list of matches, and the next Tabs cycle through them."""
    b = env.open((str(lab), None))
    until(b, lambda: settled(b))
    key(b.tree, "<Control-l>")
    assert strip(b) == ("Info", "Enter opens the folder you typed, Ctrl+Enter uses it. Tab completes a name; Down "
                                "lists the folders that match.")
    b.edit_path(str(lab / "Pro"))
    key(b._edit, "<Tab>")                                                    # 'Pro' is all that ProjectC and
    assert b._edit.get() == str(lab / "Pro")                                 # Proteomics share
    assert b._lb_visible()
    assert [b._lb.item(i, "text") for i in b._lb.get_children()] == ["ProjectC", "Proteomics"]
    key(b._edit, "<Tab>")
    assert b._edit.get() == str(lab / "ProjectC") and b._lb.selection() == ("0",)
    key(b._edit, "<Tab>")
    assert b._edit.get() == str(lab / "Proteomics") and b._lb.selection() == ("1",)
