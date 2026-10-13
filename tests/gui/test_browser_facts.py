"""What the browser says about folders (BR-9, BR-10, BR-11, BR-12, BR-22, BR-27, BUX-M2): the strip, the preview,
the badges, the sort and the 'Only folders with tiles' filter tell the truth, including about what wasn't checked."""
from __future__ import annotations

import os
import time

from test_browser import env, mounts, overlay_text, pump, rows, select, settled, strip, tree, until  # noqa: F401
from yfs.core import fsservice

FULL = " You'll see the full list before anything is stitched."


def slide(folder, prefix: str, n: int = 2) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_[{1000 + 900 * i},2000]_component_data.tif").write_bytes(b"")


def preview(b) -> list:
    """The preview's lines, title first."""
    return [str(w.cget("text")) for w in b._facts.facts.winfo_children() if w.winfo_class() == "TLabel"]


# ------------------------------------------------------------------------------------------------ the strip
def test_the_strip_counts_the_slide_folders_its_rows_show(env, tree):
    """BR-11: the current folder's strip adds the slide folders found inside its subfolders."""
    for p, names in (("P1", ("A1_Scan1", "A2_Scan1")), ("P2", ("C1_Scan1",))):
        for n in names:
            slide(tree / "Projects" / p / n, n, 3)
    b = env.open((str(tree / "Projects"), None))
    until(b, lambda: settled(b))
    assert rows(b)["P1"][1] == "2 slide folders inside"
    until(b, lambda: strip(b) == ("Ok", "Projects contains 3 slide folders (9 tiles)." + FULL), what="the roll-ups")


def test_the_strip_leaves_out_hidden_folders_like_the_search(env, tree):
    """BR-11: the search never looks inside hidden folders, so neither the strip nor the preview counts them."""
    slide(tree / "P" / "A1_Scan1", "A1_Scan1")
    slide(tree / "P" / ".H2_Scan1", "H2_Scan1")
    slide(tree / "P" / ".old" / "H1_Scan1", "H1_Scan1")
    b = env.open((str(tree / "P"), None))
    until(b, lambda: settled(b) and b._preview_on)
    assert strip(b) == ("Ok", "P contains 1 slide folder (2 tiles)." + FULL)
    until(b, lambda: "1 slide folder inside (2 tiles)" in preview(b), what="the preview")


def test_slide_folders_behind_a_link_are_not_promised(env, tree):
    """BR-11, UD3: a link row shows what it leads to, but the search doesn't follow links: the strip leaves those
    slide folders out and says to open the link instead."""
    slide(tree / "elsewhere" / "Proj" / "S1_Scan1", "S1_Scan1")
    (tree / "Lab").mkdir()
    (tree / "Lab" / "lab_data").symlink_to(tree / "elsewhere" / "Proj")
    b = env.open((str(tree / "Lab"), None))
    until(b, lambda: settled(b) and rows(b).get("lab_data", ("", ""))[1] == "1 slide folder inside · link")
    until(b, lambda: strip(b) == ("Info", "The search doesn't follow links: open lab_data to use the slide folders "
                                          "it leads to."), what="the strip")
    slide(tree / "Lab" / "R1_Scan1", "R1_Scan1")
    b.refresh()
    until(b, lambda: strip(b) == ("Ok", "Lab contains 1 slide folder (2 tiles), not counting lab_data: the search "
                                        "doesn't follow links." + FULL), what="the folder's own slide only")


def test_a_cloud_folder_says_its_subfolders_arent_checked_and_keeps_them_under_the_filter(env, tree):
    """BR-12, BUX-M2 (UD11): cloud subfolders are never probed, so nothing is claimed about them; 'Only folders with
    tiles' keeps them listed (their badge says they weren't looked at)."""
    cloud = tree / "cloudy"
    slide(cloud / "C1_Scan1", "C1_Scan1")
    slide(cloud / "Project_X" / "X1_Scan1", "X1_Scan1")
    ctx = env.ctx(mounts(f"3 1 0:41 / {cloud} rw - fuse.rclone remote: rw"))
    ctx.settings.set("browser.only_tiles", True)
    b = env.open((str(cloud), None), ctx=ctx)
    until(b, lambda: b._listing is not None and len(rows(b)) == 2, what="both cloud folders listed")
    assert {v[1] for v in rows(b).values()} == {b.T("Cloud folder — open to look inside")}
    assert strip(b) == ("Info", "Cloud folder: subfolders aren't checked automatically. Open one to look inside, or "
                                "use this folder (the search looks one level deep here).")
    assert str(cloud / "C1_Scan1") not in env.fake.paths("scandir")
    b.navigate(str(tree))                                               # a local folder holding the cloud mount
    until(b, lambda: settled(b) and "cloudy" in rows(b))
    assert strip(b) == ("Info", "No tiles found so far: 1 folder here wasn't checked. Open it to look inside.")


def test_a_folder_holding_a_cloud_folder_is_not_known_to_have_no_tiles(env, tree):
    """BUX-M2 one level down: a local folder (like ~/mnt) whose only subfolder is a cloud folder, which is never
    looked into, stays under 'Only folders with tiles', and the strip says it wasn't checked."""
    slide(tree / "mnt" / "cloudy" / "C1_Scan1", "C1_Scan1")
    (tree / "plain" / "sub").mkdir(parents=True)
    ctx = env.ctx(mounts(f"3 1 0:41 / {tree / 'mnt' / 'cloudy'} rw - fuse.rclone remote: rw"))
    ctx.settings.set("browser.only_tiles", True)
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b))
    pump(b, 600)                                                        # (after the probes' re-sort)
    assert set(rows(b)) == {"mnt"} and rows(b)["mnt"][1] == b.T("No tiles found so far · 1 folder not checked")
    assert strip(b) == ("Info", "No tiles found so far: 1 folder here wasn't checked. Open it to look inside.")


def test_the_filter_says_what_it_hides(env, tree):
    """BUX-M2: 'Only folders with tiles' says in the footer how many folders it hides, and when it hides every
    folder, the card says so."""
    for name in ("a", "b"):
        (tree / "P" / name / "sub").mkdir(parents=True)
    (tree / "Empty").mkdir()
    slide(tree / "S1_Scan1", "S1_Scan1")
    ctx = env.ctx()
    ctx.settings.set("browser.only_tiles", True)
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b) and set(rows(b)) == {"S1_Scan1"}, what="the folders without tiles hidden")
    assert str(b._others.cget("text")) == "2 folders without tiles not shown"
    b.navigate(str(tree / "P"))
    until(b, lambda: settled(b) and b._ov_kind == "empty")
    assert overlay_text(b) == b.T("No folders with tiles here. ‘Only folders with tiles’ is hiding 2 other folders.")
    assert str(b._others.cget("text")) == "2 folders without tiles not shown"
    b._toggle(b._v_tiles)                                                # the option off: nothing is hidden
    until(b, lambda: set(rows(b)) == {"a", "b"})
    assert str(b._others.cget("text")) == ""


def test_a_folder_of_hidden_items_says_so(env, tree):
    """BR-22: its row doesn't count hidden items as files, and inside, the card points at Ctrl+H."""
    hidden = tree / "HiddenOnly"
    (hidden / ".snapshot").mkdir(parents=True)
    (hidden / ".DS_Store").write_bytes(b"")
    b = env.open((str(tree), None))
    until(b, lambda: settled(b))
    assert rows(b)["HiddenOnly"][1] == "No tiles · only hidden items"
    b.navigate(str(hidden))
    until(b, lambda: b._ov_kind == "empty")
    assert overlay_text(b) == "Only hidden items here (Ctrl+H shows them)."


# ------------------------------------------------------------------------------------------------ the preview
def test_the_preview_never_shows_the_last_folders_slide(env, synth_tree):
    """BR-9: the preview is reset as the navigation starts, then follows the new folder."""
    b = env.open((str(synth_tree / "ProjectA"), "SlideA1_Scan1"))
    until(b, lambda: b._preview_on and settled(b) and preview(b)[:1] == ["SlideA1_Scan1"], what="the slide")
    b.navigate(str(synth_tree / "round1_unmix"))
    assert preview(b)[0] == "round1_unmix" and "SlideA1_Scan1" not in " ".join(preview(b))
    until(b, lambda: settled(b) and "2 slide folders inside (6 tiles)" in preview(b), what="the new folder's facts")


def test_strip_and_preview_follow_a_share_that_doesnt_answer(env, tree, monkeypatch):
    """BR-27: while the card says the share isn't answering, the strip and the preview say it too (not
    'Checking...'), and in output mode the network advice is never in a green banner."""
    monkeypatch.setitem(fsservice.DEADLINE["list_browser"], "network", 1)
    share = tree / "share"
    (share / "Archive" / "x").mkdir(parents=True)
    (share / "Out").mkdir()
    ctx = env.ctx(mounts(f"2 1 0:50 / {share} rw - cifs //srv.example.org/share rw"))
    b = env.open((str(share), None), ctx=ctx)
    until(b, lambda: settled(b))
    gate = env.fake.block(share / "Archive")
    try:
        b.navigate(str(share / "Archive"))
        until(b, lambda: b._ov_kind == "unresponsive", timeout=4, what="the not-answering card")
        assert strip(b) == ("Warn", "share didn't answer. Press F5 to try again, or use Archive anyway.")
        until(b, lambda: preview(b) == ["Archive", "share didn't answer."], what="the preview")
    finally:
        gate.set()
    out = env.open((str(share), "Out"), mode="output", ctx=ctx)
    until(out, lambda: "Free space" in strip(out)[1], what="the free space")
    assert strip(out)[0] == "Info" and "Network folder" in strip(out)[1]


# ------------------------------------------------------------------------------------------------ the sort
def test_sorting_by_modified_orders_every_row_newest_first(env, tree):
    """BR-10: every shown row is dated (not only the visible ones); rows without a date (links) come last."""
    base = time.time() - 10 * 86400
    for i in range(300):
        d = tree / f"s{i:03d}"
        d.mkdir()
        os.utime(d, (base + i * 60, base + i * 60))
    (tree / "zlink").symlink_to(tree / "s000")
    ctx = env.ctx()
    ctx.settings.set("browser.sort", ["modified", "desc"])
    b = env.open((str(tree), None), ctx=ctx)
    until(b, lambda: settled(b) and "zlink" in rows(b) and not b._dating and list(rows(b))[0] == "s299",
          timeout=10, what="every row dated and sorted")
    until(b, lambda: list(rows(b))[-2:] == ["s000", "zlink"], what="no date last")
    assert list(rows(b))[:3] == ["s299", "s298", "s297"] and "Sorting by date" not in str(b._others.cget("text"))
    select(b, "s100")
    assert b._rows[b._names["s100"]].mtime is not None
