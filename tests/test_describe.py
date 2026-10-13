"""Pure browser texts (FINAL_SPEC §2.2.2, §2.2.3, §2.2.6, §7.1): every row of the badge table, the file rows, every
target-strip text and every start-folder rule of browser_initial(). No display and no filesystem needed."""
from __future__ import annotations

import os

import pytest

from yfs.core.model import CLOUD, LOCAL, NETWORK, Entry, FsError, Rollup
from yfs.core.names import summarize_listing
from yfs.core.places import Place
from yfs.core.settings import Settings
from yfs.ui.browser import Badge, browser_initial, describe_file, describe_folder, describe_target, describe_tiles

ROOT = os.path.join(os.sep, "data", "lab")


def tiles(prefix, n, suffix="component_data"):
    return [f"{prefix}_[{1000 + 900 * i},{2000}]{'_' + suffix if suffix else ''}.tif" for i in range(n)]


def summ(name="Folder", files=(), dirs=(), truncated=False, n_entries=None):
    s = summarize_listing(os.path.join(ROOT, name), [Entry(f, False, False) for f in files] +
                          [Entry(d, True, False) for d in dirs], truncated=truncated)
    return s if n_entries is None else s.__class__(**{**s.__dict__, "n_entries": n_entries})


def badge(summary, rollup=None, mode="input", out=frozenset(), lane=LOCAL, error=None, auto=None) -> Badge:
    return describe_folder(summary, rollup, mode, out, lane, error, auto)


# ----------------------------------------------------------------------------------------------- §2.2.6 badges
@pytest.mark.parametrize("code, text", [("no_access", "No access"), ("timeout", "Didn't answer — F5 to try again"),
                                        ("unresponsive", "Didn't answer — F5 to try again")])
def test_probe_errors_come_first(code, text):
    b = badge(summ(files=tiles("S1", 3)), error=FsError(code, "x"))
    assert (b.icon, b.inside, b.dim, b.tiles_text) == ("folder_error", text, True, "")


@pytest.mark.parametrize("auto, lane, text, dim", [
    ("cloud", LOCAL, "Cloud folder — open to look inside", False),
    (None, CLOUD, "Cloud folder — open to look inside", False),
    ("disconnected", NETWORK, "Network drive — open to connect", False),
    ("autofs", NETWORK, "Network drive — open to connect", False),
    ("pseudo", LOCAL, "System folder", True)])
def test_folders_that_are_never_probed_automatically(auto, lane, text, dim):
    b = badge(None, auto=auto, lane=lane)
    assert (b.icon, b.inside, b.dim) == ("folder", text, dim)


def test_not_yet_probed_is_an_ellipsis():
    assert (badge(None).inside, badge(None).dim) == ("…", True)


def test_one_slide_folder_is_bold_with_its_tile_count():
    b = badge(summ("SlideA1_Scan1", tiles("SlideA1_Scan1", 9) + ["notes.txt"]))
    assert (b.icon, b.tiles_text, b.inside, b.bold, b.dim) == ("folder_tiles", "9", "Slide tiles", True, False)


def test_output_names_do_not_establish_completion_and_output_mode_warns_about_tiles():
    s = summ("SlideA1_Scan1", tiles("SlideA1_Scan1", 9))
    assert badge(s, out=frozenset({"slidea1_scan1.ome.tif"})).inside == "Slide tiles"
    b = badge(s, mode="output")
    assert (b.inside, b.bold, b.dim, b.tiles_text) == ("Slide tiles — don't save here", False, True, "9")


def test_component_group_wins_over_unsuffixed_and_other_suffixes():
    s = summ(files=tiles("S1", 4) + tiles("S1", 7, "") + tiles("S1", 2, "binary_seg_maps"))
    assert (badge(s).tiles_text, badge(s).inside) == ("4", "Slide tiles")
    assert badge(summ(files=tiles("S1", 5, ""))).inside == "Slide tiles"        # unsuffixed only


def test_several_prefixes_are_an_inform_batch_export():
    b = badge(summ(files=tiles("S1_Scan1", 4) + tiles("S2_Scan1", 6) + tiles("S1_Scan1", 4, "binary_seg_maps")))
    assert (b.icon, b.tiles_text, b.inside, b.bold) == ("folder_tiles", "10", "2 slides (inForm batch export)", True)


@pytest.mark.parametrize("roll, text", [(Rollup(12, 12, 541, False), "12 slide folders inside"),
                                        (Rollup(40, 40, 900, True), "40+ slide folders inside"),
                                        (Rollup(3, 1, 9, False), "1 slide folder inside")])
def test_rollups(roll, text):
    b = badge(summ(dirs=[f"d{i}" for i in range(3)], files=["x.ome.tif"]), roll)
    assert (b.icon, b.inside, b.bold) == ("folder_slides", text, True)


def test_only_other_suffix_groups():
    """The kind of tiles comes first: a narrow column cut with an ellipsis still says it (BR-26)."""
    b = badge(summ(files=tiles("S1", 3, "composite_image")))
    assert (b.tiles_text, b.inside) == ("3", "composite_image tiles, not unmixed components")


@pytest.mark.parametrize("files, dirs, text", [
    (["S1.qptiff", "a.ome.tif"], [], "Raw scan — unmix in inForm first"),
    (["Overview.tif"], [], "Raw scan — unmix in inForm first"),
    (["a.ome.tif", "b.ome.tiff", "c_stitch.tif"], ["sub"], "2 stitched images"),
    (["c_stitch.tif"], [], "1 whole-slide image (another program)"),
    (["Batch.log", "a.tif"], [], "inForm export"),
    ([f"case_{i}.tif" for i in range(6)], [], "6 other TIFF images")])
def test_contents_without_tiles(files, dirs, text):
    b = badge(summ(files=files, dirs=dirs), Rollup(1, 0, 0, False))
    assert (b.icon, b.inside, b.dim) == ("folder", text, False)


def test_truncated_empty_and_otherwise():
    assert badge(summ(files=[f"f{i}.txt" for i in range(5000)], truncated=True)).inside == "5,000+ items, no tiles"
    empty = badge(summ())
    assert (empty.inside, empty.dim) == ("Empty", True)
    assert badge(summ(dirs=[f"d{i}" for i in range(12)]), Rollup(12, 0, 0, False)).inside == "No tiles · 12 folders"
    assert badge(summ(files=["a.txt", "b.csv", "c.py"])).inside == "No tiles · 3 files"


def test_hidden_items_are_not_counted_as_files():
    """BR-22: system litter and hidden folders are no files of the user's."""
    assert badge(summ(files=[".DS_Store", "Thumbs.db"], dirs=[".snapshot"])).inside == "No tiles · only hidden items"
    assert badge(summ(files=["notes.txt", ".DS_Store", "._notes.txt"])).inside == "No tiles · 1 file"
    assert badge(summ(files=["a.lnk", "S1.ome.tmp.raw"])).inside == "No tiles · 2 files"
    links = summarize_listing(os.path.join(ROOT, "Shortcuts"), [Entry(n, False, True) for n in ("lab_data", "runs")])
    assert badge(links).inside == "No tiles · 2 links"            # links lead elsewhere: not this folder's files


def test_rank_orders_the_whats_inside_sort():
    ranks = [badge(summ(files=tiles("S", 9))).rank, badge(summ(files=tiles("A", 2) + tiles("B", 2))).rank,
             badge(summ(dirs=["d"]), Rollup(1, 1, 9, False)).rank, badge(summ(files=tiles("S", 2, "x"))).rank,
             badge(summ(files=["s.qptiff"])).rank, badge(summ(files=["a.ome.tif"])).rank,
             badge(summ(files=["a.tif"])).rank, badge(summ()).rank, badge(None, error=FsError("io", "x")).rank]
    assert ranks == sorted(ranks) and len(set(ranks)) >= 8


# ----------------------------------------------------------------------------------------------- file rows
@pytest.mark.parametrize("name, size, icon, text, dim", [
    ("A1.ome.tif", 18960143992, "file_ome", "Stitched image · 17.7 GB", False),
    ("A1.ome.tiff", None, "file_ome", "Stitched image", False),
    ("R1-01_Scan1_stitch.tif", None, "file", "Whole-slide image (another program)", False),
    ("S1.qptiff", None, "file", "Raw scan", False), ("S1.im3", None, "file", "Raw scan", False),
    ("Batch.log", None, "file", "inForm batch file", False),
    ("batch_procedure.ifp", None, "file", "inForm batch file", False),
    ("share.lnk", None, "file", "Windows shortcut (can't be followed)", True),
    ("A1.ome.tmp.raw", None, "file", "Temporary file from an unfinished run", True),
    ("notes.txt", 2048, "file", "2 KB", True)])
def test_file_rows(name, size, icon, text, dim):
    assert describe_file(name, size) == (icon, text, dim)


def test_tile_summary_row():
    assert describe_tiles(summ(files=tiles("S", 370))) == ("370 tile files", "…_[x,y]_component_data.tif")
    both = describe_tiles(summ(files=tiles("S", 2) + tiles("S", 2, "binary_seg_maps") + tiles("T", 1, "")))
    assert both == ("3 tile files", "…_[x,y]_component_data.tif, …_[x,y].tif · also 2 binary_seg_maps")    # BR-28
    shadowed = describe_tiles(summ(files=tiles("S", 9) + tiles("S", 9, "") + tiles("S", 9, "binary_seg_maps")))
    assert shadowed == ("9 tile files", "…_[x,y]_component_data.tif · also 18 other tiles")
    assert describe_tiles(summ(files=tiles("S", 4, "composite_image"))) == \
        ("4 tile files", "…_[x,y]_composite_image.tif")                       # nothing to stitch: every tile


# ----------------------------------------------------------------------------------------------- §2.2.3 strips
FULL = " You'll see the full list before anything is stitched."


def test_input_strip_texts():
    one = summ("Slide_Scan1", tiles("Slide_Scan1", 178))
    assert describe_target("input", "Slide_Scan1", one, None) == \
        ("Ok", "Slide_Scan1: 1 slide, 178 tiles (…_component_data.tif)." + FULL)
    multi = summ("informexport", tiles("S1", 500) + tiles("S2", 600) + tiles("S3", 4))
    assert describe_target("input", "informexport", multi, None) == \
        ("Ok", "informexport: 3 slides in one folder (inForm batch export), 1,104 tiles.")
    parent = summ("Proj", dirs=[f"s{i}" for i in range(6)])
    assert describe_target("input", "Proj", parent, Rollup(6, 6, 1102, False)) == \
        ("Ok", "Proj contains 6 slide folders (1,102 tiles)." + FULL)
    assert describe_target("input", "Proj", parent, Rollup(40, 40, 900, True))[1].startswith(
        "Proj contains 40+ slide folders (900+ tiles).")
    assert describe_target("input", "X", None, None) == ("Info", "Checking what's inside X…")
    assert describe_target("input", "X", parent, None) == ("Info", "Checking what's inside X…")   # roll-up pending


def test_input_strip_warnings_and_errors():
    nothing = ("Warn", "No tiles found in X within 2 levels. The search will look 3 levels deep.")
    assert describe_target("input", "X", summ(dirs=["a"]), Rollup(1, 0, 0, False)) == nothing
    assert describe_target("input", "X", summ(files=["a.txt"]), None) == ("Warn", "No tiles found in X.")   # no levels
    assert describe_target("input", "X", summ(files=["s.qptiff"]), None) == \
        ("Warn", "X holds raw scans (.qptiff). Unmix and export component data in inForm first.")
    assert describe_target("input", "X", summ(files=tiles("S", 3)), None, save_folder=True) == \
        ("Warn", "This is your save folder. Stitched images are never used as tiles.")
    assert describe_target("input", "X", None, None, error=FsError("no_access", "Permission denied.")) == \
        ("Error", "X: Permission denied.")
    assert describe_target("input", "X", None, None, auto="cloud")[0] == "Info"
    assert describe_target("input", "Z", None, None, auto="disconnected") == \
        ("Info", "Z is a network drive. Open it to connect.")
    assert describe_target("input", "X", summ(files=tiles("S", 3, "composite_image")), None)[0] == "Warn"
    assert describe_target("input", "Empty", summ(), Rollup(0, 0, 0, False)) == ("Warn", "Empty is empty.")


@pytest.mark.parametrize("mode, name, place, error, text", [
    ("input", "hungmnt", "hungmnt", FsError("timeout", "hungmnt didn't answer."),
     "hungmnt didn't answer. Press F5 to try again, or use it anyway."),
    ("input", "Archive", "labshare", FsError("timeout", "labshare didn't answer."),
     "labshare didn't answer. Press F5 to try again, or use Archive anyway."),
    ("input", "Archive", "labshare", FsError("unresponsive", "The location isn't answering."),
     "labshare isn't answering. Press F5 to try again, or use Archive anyway."),
    ("output", "Archive", "labshare", FsError("unresponsive", "The location isn't answering."),
     "labshare isn't answering. Press F5 to try again.")])
def test_a_place_that_doesnt_answer_is_named_once_with_the_next_step(mode, name, place, error, text):
    """UD1 (BR-27): never 'hungmnt: hungmnt didn't answer.', and never a red error for a share that may wake up."""
    assert describe_target(mode, name, None, None, error=error, place=place) == ("Warn", text)


def test_folders_that_cant_be_checked_never_give_no_tiles_found():
    """BR-12 (UD11): a cloud folder's subfolders are never probed; elsewhere, folders not checked are said so."""
    cloud = summ("Cloudy", dirs=["C1_Scan1", "Project_X"])
    assert describe_target("input", "Cloudy", cloud, Rollup(0, 0, 0, False), lane_class=CLOUD) == \
        ("Info", "Cloud folder: subfolders aren't checked automatically. Open one to look inside, or use this "
                 "folder (the search looks one level deep here).")
    assert describe_target("input", "Cloudy", summ("Cloudy", files=["a.txt"]), None, lane_class=CLOUD) == \
        ("Warn", "No tiles found in Cloudy.")
    mnt = summ("mnt", dirs=["nasA", "nasB", "cloud"])
    assert describe_target("input", "mnt", mnt, Rollup(1, 0, 0, False), unchecked=2) == \
        ("Info", "No tiles found so far: 2 folders here weren't checked. Open one to look inside.")
    assert describe_target("input", "mnt", mnt, Rollup(2, 0, 0, False), unchecked=1)[1].startswith(
        "No tiles found so far: 1 folder here wasn't checked. Open it to look inside.")
    assert describe_target("input", "mnt", mnt, Rollup(3, 1, 9, False), unchecked=2)[0] == "Ok"
    assert describe_target("input", "Archive", summ("Archive", dirs=["2019"]), Rollup(40, 0, 0, True)) == \
        ("Warn", "No tiles found in Archive so far (not every subfolder was checked). The search will look 3 levels "
                 "deep.")                                   # a roll-up stops at 40 subfolders (200 per view)


def test_slide_folders_behind_a_link_are_not_promised():
    """BR-11: the search doesn't follow links, so the strip counts without them and says where they lead."""
    home = summ("Home", dirs=["Real"], files=["lab_data"])
    assert describe_target("input", "Home", home, Rollup(1, 0, 0, False), linked="lab_data") == \
        ("Info", "The search doesn't follow links: open lab_data to use the slide folders it leads to.")
    assert describe_target("input", "Home", home, Rollup(1, 1, 4, False), linked="lab_data") == \
        ("Ok", "Home contains 1 slide folder (4 tiles), not counting lab_data: the search doesn't follow links." + FULL)


@pytest.mark.parametrize("found", [0, 1])
def test_failed_rollups_are_unknown_or_explicitly_partial(found):
    s = summ("Project", dirs=["Readable", "Unreadable"])
    roll = Rollup(found, found, 2 * found, False)
    b = describe_folder(s, roll, "input", frozenset(), LOCAL, None, failed=1)
    assert "1 folder couldn't be read" in b.inside and not b.dim
    style, text = describe_target("input", "Project", s, roll, failed=1)
    assert style == "Warn" and "1 folder couldn't be read" in text and "F5" in text
    assert ("1+ slide folder" in text) if found else ("Couldn't check every folder" in text)


@pytest.mark.parametrize("mode", ["output", "temp"])
def test_capacity_failure_is_not_a_write_permission_failure(mode):
    err = FsError("io", "The disk or share reported a read error.")
    assert describe_target(mode, "Destination", summ(), None, free_error=err) == \
        ("Warn", "Couldn't check free space: The disk or share reported a read error. Press F5 to try again.")


def test_temp_strip_texts():
    """BR-20: the temporary-files chooser says only what matters for scratch files: free space, and how far away
    the folder is; never tiles, stitched images or a Create action (the strip has no action from describe)."""
    gb = 1024 ** 3
    assert describe_target("temp", "S1", summ("S1", tiles("S1", 9)), None, free=10 * gb) == \
        ("Ok", "Free space: 10.0 GB.")
    assert describe_target("temp", "N", summ(), None, free=gb, lane_class=NETWORK) == \
        ("Info", "Free space: 1.0 GB. Network folder: temporary files are faster on a local disk.")
    assert describe_target("temp", "C", summ(), None, lane_class=CLOUD) == \
        ("Warn", "Cloud folder: temporary files would be uploaded too. A local disk is better.")
    assert describe_target("temp", "T", None, None) == ("Info", "Checking T…")
    assert describe_target("temp", "T", None, None, error=FsError("no_access", "Permission denied.")) == \
        ("Error", "Can't write here: Permission denied.")
    assert describe_target("temp", "T", None, None, error=FsError("timeout", "labshare didn't answer."),
                           place="labshare") == ("Warn", "labshare didn't answer. Press F5 to try again.")


def test_temp_badges_single_nothing_out():
    """BR-20: for temporary files no folder is a better pick than another: nothing is bold."""
    one = summ("S1_Scan1", tiles("S1_Scan1", 4))
    assert badge(one, mode="temp", out=frozenset({"s1_scan1.ome.tif"})) == \
        Badge("folder_tiles", "4", "Slide tiles", False, False, 0)
    assert badge(summ(dirs=["a"]), Rollup(1, 1, 4, False), mode="temp").bold is False


def test_output_strip_texts():
    outs = summ("Stitched", ["a.ome.tif", "b.ome.tif"])
    style, text = describe_target("output", "Stitched", outs, None, free=int(1.3 * 1024 ** 4),
                                  planned=frozenset({"a.ome.tif", "b.ome.tif", "c.ome.tif", "d.ome.tif",
                                                     "e.ome.tif", "f.ome.tif"}))
    assert (style, text) == ("Ok", "Free space: 1.3 TB · contains 2 stitched images · 2 of your 6 slides already "
                                   "have stitched files here.")
    style, text = describe_target("output", "S1", summ("S1", tiles("S1", 9)), None, free=10 * 1024 ** 3)
    assert style == "Warn" and text == "Free space: 10.0 GB. This folder holds tiles; stitched images would be mixed in."
    assert describe_target("output", "N", summ(), None, free=1024 ** 4, lane_class=NETWORK) == \
        ("Info", "Free space: 1.0 TB. Network folder: saving over the network is slower; a local disk is faster.")
    assert describe_target("output", "N", summ("N", tiles("S", 2)), None, lane_class=NETWORK)[0] == "Warn"   # BR-27
    assert describe_target("output", "C", summ(), None, lane_class=CLOUD) == \
        ("Warn", "Cloud folder: stitched files will upload in the background.")
    assert describe_target("output", "N", None, None) == ("Info", "Checking N…")
    assert describe_target("output", "N", None, None, error=FsError("no_access", "Permission denied.")) == \
        ("Error", "Can't save here: Permission denied.")


# ----------------------------------------------------------------------------------------------- §2.2.2 start folder
def settings(**data) -> Settings:
    return Settings(os.path.join(ROOT, "no-such-config", "settings.json"), data)


SHARE = Place("network", "labshare", os.path.join(os.sep, "mnt", "labshare"), "drive_net", NETWORK, True)
PROJ = os.path.join(SHARE.path, "ProjectC")


def test_rule1_field_path_opens_its_parent_with_it_selected():
    assert browser_initial("input", f"  {PROJ}/  ", None, settings()) == (SHARE.path, "ProjectC")
    assert browser_initial("output", PROJ, "/x/in", settings()) == (SHARE.path, "ProjectC")


def test_rule1_a_place_root_opens_itself():
    assert browser_initial("input", SHARE.path, None, settings(), [SHARE]) == (SHARE.path, None)
    assert browser_initial("input", os.sep, None, settings()) == (os.sep, None)
    assert browser_initial("input", SHARE.path, None, settings()) == (os.path.dirname(SHARE.path), "labshare")


def test_rule2_last_browsed_folder_per_mode():
    s = settings(paths={"browser_last": {"input": PROJ, "output": "/data/out"}},
                 recent={"inputs": [{"path": "/x/recent"}]})
    assert browser_initial("input", "", None, s) == (PROJ, None)
    assert browser_initial("output", "relative/text", None, s) == ("/data/out", None)


def test_rule3_parent_of_the_latest_recent_entry():
    s = settings(recent={"inputs": [{"path": PROJ, "slides": 6}], "outputs": [{"path": "/data/Stitched"}]})
    assert browser_initial("input", "", None, s) == (SHARE.path, "ProjectC")
    assert browser_initial("output", "", PROJ, s) == ("/data", "Stitched")


def test_rule4_output_mode_opens_next_to_the_input():
    assert browser_initial("output", "", PROJ, settings()) == (SHARE.path, "ProjectC")
    assert browser_initial("input", "", PROJ, settings())[1] is None                # input mode: rule 5


def test_rule4_a_slide_folder_input_is_never_the_default_save_folder():
    """BR-8: one slide's folder of tiles: open next to it with nothing selected, so the target is the folder that
    holds it; a slide at a share's root opens the root itself (there is nothing next to it on the share)."""
    slide = os.path.join(PROJ, "PC-03_Scan1")
    assert browser_initial("output", "", slide, settings(), input_is_slide=True) == (PROJ, None)
    assert browser_initial("output", "", SHARE.path, settings(), [SHARE], input_is_slide=True) == (SHARE.path, None)
    assert browser_initial("output", "", PROJ, settings(), input_is_slide=False) == (SHARE.path, "ProjectC")


def test_the_temporary_files_chooser_has_its_own_start_folder(monkeypatch):
    """BR-20: the temp field, then paths.browser_last.temp, then Home: never an output folder or the input."""
    monkeypatch.setenv("HOME", os.path.join(ROOT, "home"))
    s = settings(paths={"browser_last": {"output": "/data/out"}}, recent={"outputs": [{"path": "/data/Stitched"}]})
    assert browser_initial("temp", "", PROJ, s) == (os.path.join(ROOT, "home"), None)
    s.set("paths.browser_last.temp", "/scratch")
    assert browser_initial("temp", "", PROJ, s) == ("/scratch", None)
    assert browser_initial("temp", "/fast/tmp", PROJ, s) == ("/fast", "tmp")


def test_rule5_home(monkeypatch):
    monkeypatch.setenv("HOME", os.path.join(ROOT, "home"))
    assert browser_initial("input", "", None, settings()) == (os.path.join(ROOT, "home"), None)
