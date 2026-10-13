"""Folder browser on hung, slow, linked and newly mounted places (UD1, UD2, UD3, CT-4, CT-6, FS-1): every wait ends,
nothing waits in the wrong lane, late answers never move the view, and the mount table is followed while it changes.
Shares are cifs lines in a fake mount table over test folders; tests/fake_fs.py makes them hang."""
from __future__ import annotations

import gc
import os
import threading
import tkinter as tk

import pytest

from test_browser import (env, key, mounts, overlay_buttons, overlay_text, pump, q, rows, select,  # noqa: F401
                          settled, strip, tree, until)
from yfs.core import fsservice
from yfs.core.fsservice import FsService
from yfs.ui import browser

DEAD = "Didn't answer — F5 to try again"


def cifs(path, n: int = 50) -> str:
    return f"{n} 1 0:{n} / {path} rw - cifs //srv.example.org/{os.path.basename(str(path))} rw"


def slide(folder, prefix: str, n: int = 2) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_[{1000 + 900 * i},2000]_component_data.tif").write_bytes(b"")


def card_button(b, text: str):
    return next(w for f in b._ov.winfo_children() for w in f.winfo_children()
                if w.winfo_class() == "TButton" and str(w.cget("text")) == text)


@pytest.fixture
def hang_locate(monkeypatch):
    """hang_locate(folder): the browser's locate lane call blocks for paths in `folder` (os.stat on a dead share)
    until the returned gate opens; every gate opens when the test ends."""
    gates, real = [], browser._locate

    def make(folder):
        gate = threading.Event()
        gates.append(gate)

        def locate(path):
            if path.startswith(str(folder)):
                gate.wait()
            return real(path)
        monkeypatch.setattr(browser, "_locate", locate)
        return gate
    yield make
    for gate in gates:
        gate.set()


@pytest.fixture
def dead_share(env, tree, monkeypatch):
    """tree/mnt holding 'dead' (a cifs mount whose server is down, 1 s deadlines) and 'fine'; yields the gate."""
    for kind in ("probe", "readlink"):
        monkeypatch.setitem(fsservice.DEADLINE[kind], "network", 1)
    for name in ("dead", "fine"):
        (tree / "mnt" / name / "inner").mkdir(parents=True)
    gate = env.fake.block(tree / "mnt" / "dead")
    yield gate
    gate.set()


# ------------------------------------------------------------------------------------------------ UD3, UD1: lanes
def test_a_hung_share_mounted_in_the_folder_never_holds_the_folder_up(env, tree, dead_share):
    """BR-1, BR-15: the stat pass leaves the mount point out, its probe waits in the share's own lane, and at that
    lane's deadline the row says it didn't answer; the strip names the share once (UD1)."""
    mnt = tree / "mnt"
    ctx = env.ctx(mounts(cifs(mnt / "dead")))
    b = env.open((str(mnt), None), ctx=ctx)
    until(b, lambda: rows(b).get("dead", ("", ""))[1] == b.T(DEAD), timeout=4, what="the share's deadline")
    assert ctx.fs.health(ctx.fs.lane_key(str(mnt))) != "unresponsive"
    assert ("stat", str(mnt / "fine")) in env.fake.calls and ("stat", str(mnt / "dead")) not in env.fake.calls
    assert rows(b)["fine"][1] == "No tiles · 1 folder"
    select(b, "dead")
    assert strip(b) == ("Warn", "dead didn't answer. Press F5 to try again, or use it anyway.")


def test_rows_that_didnt_answer_are_asked_again_once_the_share_answers(env, tree, dead_share):
    """UD1: nothing retries while the share hangs; when its lane answers again the row is probed by itself."""
    mnt = tree / "mnt"
    b = env.open((str(mnt), None), ctx=env.ctx(mounts(cifs(mnt / "dead"))))
    until(b, lambda: rows(b).get("dead", ("", ""))[1] == b.T(DEAD), timeout=4, what="the share's deadline")
    dead_share.set()                                    # the server is back: its stuck call returns
    until(b, lambda: rows(b)["dead"][1] == "No tiles · 1 folder", timeout=3, what="the probe asked again")


def test_a_folder_whose_subfolder_didnt_answer_says_so_until_it_does(env, tree, dead_share):
    """BR-15 (UD1): a roll-up probe past its deadline ends the row's '...' as 'Didn't answer' (never 'No tiles'),
    and once the share answers, the subfolder that didn't is asked again (only that one)."""
    b = env.open((str(tree), None), ctx=env.ctx(mounts(cifs(tree / "mnt" / "dead"))))
    until(b, lambda: rows(b).get("mnt", ("", ""))[1] == b.T(DEAD), timeout=4, what="the roll-up's deadline")
    select(b, "mnt")
    assert strip(b) == ("Warn", "dead didn't answer. Press F5 to try again, or use mnt anyway.")
    budget = b._roll_budget
    dead_share.set()
    until(b, lambda: rows(b)["mnt"][1] == "No tiles · 2 folders", timeout=3, what="the roll-up asked again")
    assert b._roll_budget == budget and env.fake.paths("scandir").count(str(tree / "mnt" / "fine")) == 1


def test_a_link_into_a_hung_share_waits_in_the_shares_lane(env, tree, dead_share):
    """UD3 (BR-1, FS-9): a link is read in the lane of the folder holding it and followed into the share's lane;
    a mount point ends the chain without being read. Both links end as folders that didn't answer, and the
    folder holding them stays healthy."""
    mnt, home = tree / "mnt", tree / "home"
    home.mkdir()
    (home / "lab_data").symlink_to(mnt / "dead" / "inner")
    (home / "lab_root").symlink_to(mnt / "dead")
    ctx = env.ctx(mounts(cifs(mnt / "dead")))
    b = env.open((str(home), None), ctx=ctx)
    until(b, lambda: all(rows(b).get(n, ("", ""))[1] == b.T(DEAD + " · link") for n in ("lab_data", "lab_root")),
          timeout=5, what="both links end at the share's deadline")
    assert ctx.fs.health(ctx.fs.lane_key(str(home))) != "unresponsive"
    assert b._rows[b._names["lab_root"]].kind == "dir" and ("stat", str(mnt / "dead")) not in env.fake.calls
    assert str(home / "lab_data") not in env.fake.paths("scandir")       # never listed through the link


def test_a_folder_of_links_claims_nothing_while_they_are_followed(env, tree, dead_share):
    """UD3: until its link into the share is followed, a folder holding only that link doesn't say it has no
    folders; at the share's deadline the link shows as a folder that didn't answer. Links that lead to a file or
    nowhere end as files: then it says so."""
    home = tree / "home"
    home.mkdir()
    (home / "lab_data").symlink_to(tree / "mnt" / "dead" / "inner")
    (tree / "notes.txt").write_text("x")
    for name, target in (("to_a_file", tree / "notes.txt"), ("broken", tree / "nowhere")):
        (tree / name).mkdir()
        (tree / name / "link").symlink_to(target)
        env.fake.delay(target, 0.4)                     # answered after the folder's own layout is done
    b = env.open((str(home), None), ctx=env.ctx(mounts(cifs(tree / "mnt" / "dead"))))
    until(b, lambda: b._listing is not None and not b._loading)
    pump(b, 300)
    assert b._ov_kind == "" and b._rows[b._names["lab_data"]].probing
    until(b, lambda: rows(b).get("lab_data", ("", ""))[1] == b.T(DEAD + " · link"), timeout=4,
          what="the share's deadline")
    assert b._ov_kind == ""
    for name in ("to_a_file", "broken"):
        b.navigate(str(tree / name))
        until(b, lambda: b._ov_kind == "empty", what=f"the link in {name} followed")
        assert overlay_text(b) == b.T("No folders here. Tiles usually sit in a folder named like ‘<slide>_Scan1’.")


def test_a_folder_full_of_links_is_followed_300_links_per_turn(env, tree, monkeypatch):
    """UD3: following costs a request per link, so a folder of links is followed in turns, like the probes; every
    link still ends up as the folder it leads to."""
    links = tree / "links"
    links.mkdir()
    for i in range(320):
        (tree / "data" / f"d{i:03d}").mkdir(parents=True)
        (links / f"link{i:03d}").symlink_to(tree / "data" / f"d{i:03d}")
    asked, readlink, follow = [], FsService.readlink, browser.FolderBrowser._follow_links
    monkeypatch.setattr(FsService, "readlink",
                        lambda self, path, **kw: (asked.append(path), readlink(self, path, **kw))[1])
    turns = []

    def counted(self):
        n = len(asked)
        follow(self)
        turns.append(len(asked) - n)
    monkeypatch.setattr(browser.FolderBrowser, "_follow_links", counted)
    b = env.open((str(links), None))
    until(b, lambda: settled(b) and len(rows(b)) == 320, what="every link followed to its folder")
    assert max(turns) == 300 and sum(turns) == 320


# ------------------------------------------------------------------------------------------------ BR-4, CT-4: locate
def test_a_hung_first_folder_shows_the_not_answering_card_and_its_late_answer_moves_nothing(
        env, tree, monkeypatch, hang_locate):
    """BR-4, CT-4: the first locate is the loading phase: the bar, then Keep waiting / Try again / Go back at its
    deadline with the strip saying so; once the user went elsewhere, its late answer changes nothing."""
    monkeypatch.setitem(fsservice.DEADLINE["locate"], "network", 1)
    share = tree / "share"
    (share / "Lab" / "ExpQ").mkdir(parents=True)
    (tree / "local" / "x").mkdir(parents=True)
    gate = hang_locate(share)
    b = env.open((str(share / "Lab" / "ExpQ"), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: b._bar.winfo_manager() == "place", timeout=2, what="the loading bar")
    until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="the not-answering card")
    assert overlay_buttons(b) == ["Keep waiting", "Try again", "Go back"]
    assert overlay_text(b) == b.T("share isn't answering. Nothing else in the app is affected. Network drives "
                                  "sometimes need a minute to wake up.")
    assert strip(b) == ("Warn", "share didn't answer. Press F5 to try again, or use ExpQ anyway.")
    b.navigate(str(tree / "local"))
    until(b, lambda: b._listing is not None and b.current() == str(tree / "local"))
    gate.set()                                          # the share answers: its lane is fine again
    pump(b, 1200)
    assert b.current() == str(tree / "local") and not b._ov.winfo_manager()


def test_the_not_answering_card_opens_the_folder_once_the_share_answers(env, tree, monkeypatch, hang_locate):
    """UD1: the card's navigation is asked again, once, when the share's lane answers again."""
    monkeypatch.setitem(fsservice.DEADLINE["locate"], "network", 1)
    share = tree / "share"
    slide(share / "Lab" / "ExpQ" / "Q01_Scan1", "Q01_Scan1")
    gate = hang_locate(share)
    b = env.open((str(share / "Lab" / "ExpQ"), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="the not-answering card")
    gate.set()
    until(b, lambda: b._listing is not None and "Q01_Scan1" in rows(b), timeout=3, what="the folder by itself")
    assert b.current() == str(share / "Lab" / "ExpQ") and not b._ov.winfo_manager()


def test_go_back_from_the_first_folder_leaves_the_share_that_doesnt_answer(env, tree, monkeypatch, hang_locate):
    """BR-4: with no history, Go back leaves the share (its parent folders would only wait for it again)."""
    monkeypatch.setitem(fsservice.DEADLINE["locate"], "network", 1)
    share = tree / "share"
    (share / "Lab" / "ExpQ").mkdir(parents=True)
    hang_locate(share)
    b = env.open((str(share / "Lab" / "ExpQ"), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="the not-answering card")
    card_button(b, "Go back").invoke()
    until(b, lambda: b._listing is not None and b.current() == str(tree), what="out of the share")
    assert "share" in rows(b) and not b._ov.winfo_manager()


def test_a_typed_path_that_answers_late_never_moves_the_view(env, tree, hang_locate):
    """CT-4: a typed path loads like a folder; a navigation makes its answer obsolete."""
    share = tree / "share"
    (share / "Lab").mkdir(parents=True)
    (tree / "local").mkdir()
    gate = hang_locate(share)
    b = env.open((str(tree), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: settled(b))
    b.edit_path(str(share / "Lab"))
    key(b._edit, "<Return>")
    until(b, lambda: b._bar.winfo_manager() == "place", timeout=2, what="the typed path loads")
    b.navigate(str(tree / "local"))
    until(b, lambda: b._listing is not None and b.current() == str(tree / "local"))
    gate.set()
    pump(b, 800)
    assert b.current() == str(tree / "local")


def test_a_typed_path_that_hangs_shows_the_card_and_go_back_stays_put(env, tree, monkeypatch, hang_locate):
    """BR-4: Keep waiting / Try again / Go back for a typed path too; Go back stays in the folder shown."""
    monkeypatch.setitem(fsservice.DEADLINE["locate"], "network", 1)
    share = tree / "share"
    (share / "Lab").mkdir(parents=True)
    (tree / "a").mkdir()
    hang_locate(share)
    b = env.open((str(tree), None), ctx=env.ctx(mounts(cifs(share))))
    until(b, lambda: settled(b))
    b.edit_path(str(share / "Lab"))
    key(b._edit, "<Return>")
    until(b, lambda: b._bar.winfo_manager() == "place", timeout=2, what="the typed path loads")
    assert b._editing and b._edit.get() == str(share / "Lab")             # BR-16: the text stays while it loads
    until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="the not-answering card")
    assert overlay_buttons(b) == ["Keep waiting", "Try again", "Go back"] and b.current() == str(tree)
    card_button(b, "Go back").invoke()
    pump(b, 100)
    assert not b._ov.winfo_manager() and not b._bar.winfo_manager() and b.current() == str(tree) and "a" in rows(b)


# ------------------------------------------------------------------------------------------------ BR-7, CT-4: saving
def test_save_here_only_ever_returns_the_folder_still_chosen(env, tree, monkeypatch):
    """BR-7, CT-4: a click on another folder while a save check runs checks that folder instead; the earlier
    check's late answer neither closes the dialog nor returns its folder, even while the newer check is out."""
    for name in ("A", "B"):
        (tree / name).mkdir()
    gates, real = {"A": threading.Event(), "B": threading.Event()}, browser.write_probe

    def probe(path, create=True):
        gates[os.path.basename(path)].wait()
        return real(path, create)
    monkeypatch.setattr(browser, "write_probe", probe)
    try:
        b = env.open((str(tree), "A"), mode="output")
        until(b, lambda: settled(b) and b.tree.selection())
        b.choose()
        assert strip(b) == ("Info", b.T("Checking that you can save in A…"))
        select(b, "B")
        assert str(b._primary.cget("text")) == f"Save here: {q(b, 'B')}" and "Checking" not in strip(b)[1]
        b.choose()
        gates["A"].set()                                # A answers after the user moved on to B
        pump(b, 300)
        assert env.done == [] and b.winfo_exists() and strip(b) == ("Info", b.T("Checking that you can save in B…"))
        gates["B"].set()
        until(b, lambda: env.done, what="B checked and chosen")
        assert env.done == [str(tree / "B")]
    finally:
        for gate in gates.values():
            gate.set()


def test_a_save_check_or_new_folder_that_doesnt_answer_says_so_and_offers_the_next_step(env, tree, monkeypatch):
    """UD1: at the deadline a save check ends as a warning with [Try again] (which lets one request through the
    share that stopped answering), and a new folder as a warning pointing at F5: nothing waits silently."""
    for kind in ("write_probe", "mkdir"):
        monkeypatch.setitem(fsservice.DEADLINE[kind], "network", 1)
    share = tree / "share"
    (share / "outs" / "A").mkdir(parents=True)
    gate, real_probe, real_mkdir = threading.Event(), browser.write_probe, os.mkdir

    def probe(path, create=True):
        gate.wait()
        return real_probe(path, create)

    def mkdir(path, *a, **kw):
        gate.wait()
        return real_mkdir(path, *a, **kw)
    monkeypatch.setattr(browser, "write_probe", probe)
    monkeypatch.setattr(os, "mkdir", mkdir)
    try:
        b = env.open((str(share / "outs"), "A"), mode="output", ctx=env.ctx(mounts(cifs(share))))
        until(b, lambda: settled(b) and b.tree.selection())
        b._make_folder("New")
        until(b, lambda: strip(b)[0] == "Warn", timeout=4, what="the folder's deadline")
        assert strip(b)[1] == b.T("Couldn't create ‘New’: share didn't answer. Press F5 to see whether it was made.")
        gate.set()                                      # the share answers: the folder is made after all
        until(b, lambda: (share / "outs" / "New").is_dir() and b.ctx.fs.health(str(share)) == "ok")
        gate.clear()
        b.choose()
        until(b, lambda: strip(b)[0] == "Warn", timeout=4, what="the save check's deadline")
        assert strip(b)[1] == "Couldn't check that you can save in A: share didn't answer." and env.done == []
        assert b._strip_btn.winfo_manager() and str(b._strip_btn.cget("text")) == "Try again"
        gate.set()
        b._strip_btn.invoke()
        until(b, lambda: env.done, what="A checked and chosen")
        assert env.done == [str(share / "outs" / "A")]
    finally:
        gate.set()


def test_a_folder_made_after_the_user_moved_on_does_not_pull_the_view_back(env, tree, monkeypatch):
    """CT-4: New folder on a slow share, then the user goes elsewhere: the folder is made, the view stays."""
    (tree / "outs").mkdir()
    (tree / "elsewhere").mkdir()
    gate, real = threading.Event(), os.mkdir

    def mkdir(path, *a, **kw):
        if str(path).endswith("New"):
            gate.wait()
        return real(path, *a, **kw)
    monkeypatch.setattr(os, "mkdir", mkdir)
    try:
        b = env.open((str(tree / "outs"), None), mode="output")
        until(b, lambda: settled(b))
        b._make_folder("New")
        b.navigate(str(tree / "elsewhere"))
        until(b, lambda: b._listing is not None and b.current() == str(tree / "elsewhere"))
        gate.set()
        until(b, lambda: (tree / "outs" / "New").is_dir(), what="the folder is made")
        pump(b, 300)
        assert b.current() == str(tree / "elsewhere")
    finally:
        gate.set()


# ------------------------------------------------------------------------------------------------ UD2: mount table
AUTOFS = "60 1 0:60 / {} rw - autofs systemd-1 rw,direct"


def test_the_mount_table_is_reloaded_on_opening_and_after_an_automount(env, tree, monkeypatch):
    """UD2 (CT-11, FS-2, PL-15): reload when the browser opens and after opening an automount point; the newer
    table gives the share its own network lane and a place in the sidebar."""
    calls, real = [], FsService.reload_mounts
    monkeypatch.setattr(FsService, "reload_mounts", lambda self: (calls.append(1), real(self))[1])
    auto = tree / "auto"
    slide(auto / "S1_Scan1", "S1_Scan1")
    ctx = env.ctx(mounts(AUTOFS.format(auto)))
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b) and b._place_items)
    assert calls and rows(b)["auto"][1] == b.T("Network drive — open to connect")
    assert str(auto) not in env.fake.paths("scandir")             # an automount point is never probed by itself
    env.table = mounts(AUTOFS.format(auto), cifs(auto, 61))         # what opening it mounts
    b.navigate(str(auto))
    until(b, lambda: b._listing is not None and ctx.fs.mounts == env.table and b.ctx.mounts is ctx.fs.mounts,
          what="the newer table")
    assert ctx.fs.lane_class(str(auto / "S1_Scan1")) == "network" and len(calls) >= 2
    until(b, lambda: any(p.section == "network" and p.path == str(auto) for p, _l in b._place_items.values()),
          what="the share in the sidebar")
    b.up()
    until(b, lambda: settled(b) and rows(b)["auto"][1] == "1 slide folder inside", what="probed now")


def test_a_newer_mount_table_rechecks_the_rows_shown(env, tree):
    """UD2: on MountsChanged each row's reason not to be probed is evaluated again, and rows now allowed are
    probed without any navigation."""
    auto = tree / "auto"
    slide(auto / "S1_Scan1", "S1_Scan1")
    ctx = env.ctx(mounts(AUTOFS.format(auto)))
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b) and rows(b)["auto"][1] == b.T("Network drive — open to connect"))
    env.table = mounts(AUTOFS.format(auto), cifs(auto, 61))
    ctx.fs.set_mounts(env.table)                        # what a reload finds once something mounted the share
    until(b, lambda: rows(b)["auto"][1] == "1 slide folder inside", what="the row probed in place")
    assert b.ctx.mounts is ctx.fs.mounts and b.current() == str(tree)


# ------------------------------------------------------------------------------------------------ FS-1, CT-6
def test_a_cut_short_cached_listing_is_never_shown_as_the_folder(env, tree):
    """FS-1: discovery caches listings capped at 5,000 entries; the browser lists the folder itself unless a cached
    listing is complete."""
    big = tree / "big"
    big.mkdir()
    for i in range(300):
        (big / f"f{i:03d}.txt").write_bytes(b"")
    ctx = env.ctx()
    ctx.fs.list_dir(str(big), ticket=ctx.fs.new_ticket(), owner="discovery", cap=100)
    for _ in range(200):
        if ctx.fs.cached_listing(str(big)) is not None:
            break
        pump(ctx.root, 10)
    assert ctx.fs.cached_listing(str(big)).truncated
    b = env.open((str(big), None), ctx=ctx)
    until(b, lambda: b._listing is not None and not b._loading)
    assert len(b._listing.entries) == 300 and str(b._others.cget("text")) == "300 other files not shown"


def test_closing_frees_its_tk_variables_on_the_tk_thread(env, tree, monkeypatch):
    """CT-6 (UD12): a garbage collection on a worker thread must never be the one that frees the browser's Tk
    variables (a cross-thread Tcl call that can freeze the app)."""
    seen, real = [], tk.Variable.__del__

    def spy(var):
        main = threading.current_thread() is threading.main_thread()
        seen.append((str(var), main))
        if main:                                        # off the Tk thread it would wait for the main loop
            real(var)
    monkeypatch.setattr(tk.Variable, "__del__", spy)
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    names = {str(v) for v in (b._v_tiles, b._v_hidden, b._v_all, b._edit_var, b._filter_var)}
    b.cancel()
    del b
    worker = threading.Thread(target=gc.collect)
    worker.start()
    worker.join(10)
    assert sorted(main for name, main in seen if name in names) == [True] * len(names)
