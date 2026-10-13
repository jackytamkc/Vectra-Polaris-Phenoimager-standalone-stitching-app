"""Plan: default selection (§3.10), flags (§2.3.4), notes (§2.3.8), issues (§2.4.1), renames (§2.3.5),
Select modes, channel names (R26), output matching and verification (§3.8), carry-over (§2.3.9) and the runner
protocol with a thread-safety test (acceptance B5)."""
from __future__ import annotations

import json
import os
import random
import threading
import time

import pytest

from yfs.core import names as N
from yfs.core import plan as PL
from yfs.core import preflight
from yfs.core.discovery import Discovery, DiscoveryOptions
from yfs.core.events import CandidateFound, ScanFinished, ScanNote
from yfs.core.fsservice import FsService
from yfs.core.model import Candidate, Entry, Estimate, FsError, Note, OutputIndex, OutputInfo, RowOutcome, TileHeader
from yfs.core.names import norm_key
from yfs.core.places import MountTable
from yfs.core.plan import Plan, PlanEnv, default_state, display_name, row_text

SRC, OUT = "/data/src", "/data/out"
GB = 1024 ** 3


def cand(rel, prefix=None, n=20, kind="component", suffix="component_data", multi=False, mtime=None, x0=0.0):
    folder = SRC if rel == "." else f"{SRC}/{rel}"
    prefix = prefix if prefix is not None else os.path.basename(folder)
    key = f"{norm_key(folder)}|*|legacy" if kind == "legacy_all" else f"{norm_key(folder)}|{prefix.casefold()}|{suffix}"
    tiles = () if kind == "legacy_all" else tuple(f"{prefix}_[{x0 + i * 900},0]_{suffix}.tif" for i in range(n))
    coords = tuple((x0 + i * 900.0, 0.0) for i in range(len(tiles)))
    return Candidate(key, folder, rel, os.path.basename(folder), prefix if kind != "legacy_all" else "",
                     suffix if kind != "legacy_all" else "legacy", kind, tiles, coords, multi, False,
                     rel.count("/") + 1, mtime)


def header(n_channels=5, problem=None, names=("DAPI", "Opal 520", "Opal 570", "Opal 690", "AF"), offline=False,
           first=5 * 1396 * 1860 * 4):
    return TileHeader("t.tif", (n_channels, 1396, 1860), "float32", n_channels, 1396, 1860, 0.5, 0.5, True,
                      tuple(names), problem, (None,) * len(names), first, offline)


def index(outputs=(), partials=(), free=10 ** 13, out_dir=OUT, lane_class="local", system=False, error=None,
          ready=(), **kw):
    info = lambda name: OutputInfo(name, 18 * GB, 1788448361675246000)                 # noqa: E731
    return OutputIndex(out_dir, True, frozenset(norm_key(n) for n in outputs), {norm_key(n): info(n) for n in outputs},
                       {norm_key(n): info(n) for n in partials}, (), free, "/data", lane_class, system, error,
                       {norm_key(n): info(n) for n in ready}, **kw)


def env(**kw):
    base = dict(source=SRC, out_dir=OUT, scanning=False, paused=False, out_index=index(), out_index_pending=False,
                write_error=None, temp_free=None, temp_lane_class="local", temp_same_volume=True, windows=False,
                available_ram=64 * GB)
    base.update(kw)
    return PlanEnv(**base)


def make(*cands, naming="folder", carry=None):
    p = Plan(SRC, OUT, naming, carry)
    for c in cands:
        p.add_candidate(c)
    return p


def st(p, rid):
    r = p.row(rid)
    return r.section, r.status, r.checked, r.checkable


def codes(issues, sev=None):
    return [i.code for i in issues if sev in (None, i.severity)]


# ---- §3.10 default selection
def test_default_states_and_user_touch():
    a, few, legacy, seg = cand("A"), cand("B", n=3), cand("WSIs", kind="legacy_all"), \
        cand("A", kind="other_suffix", suffix="binary_seg_maps")
    p = make(a, few, legacy, seg)
    assert st(p, a.key) == ("new", "new", True, True) and st(p, few.key) == ("new", "few_tiles", False, True)
    assert st(p, legacy.key) == ("excluded", "legacy_all", False, True)
    assert st(p, seg.key) == ("excluded", "other_suffix", False, True)
    p.apply_header(a.key, None, "Permission denied.", 64 * GB)
    assert st(p, a.key) == ("new", "header_error", False, True)
    assert "Can't read the first tile (Permission denied)" in row_text(p.row(a.key))
    p.apply_header(a.key, header(), None, 64 * GB)
    assert st(p, a.key) == ("new", "new", True, True) and p.row(a.key).est.canvas_w == 19 * 900 * 2 + 1860
    p.toggle([a.key])                                             # the user unticks
    p.apply_header(a.key, header(), None, 64 * GB)
    assert st(p, a.key) == ("new", "new", False, True) and p.row(a.key).user_touched
    assert row_text(p.row(a.key)) == "Won't be stitched"
    p.toggle([few.key])
    assert st(p, few.key) == ("new", "few_tiles", True, True) and row_text(p.row(few.key)) == "New file"


def test_sparse_and_unsupported_overrides_user_touch():
    sparse = cand("S", n=12)
    p = make(sparse)
    big = TileHeader("t", (5, 100, 100), "float32", 5, 100, 100, 0.01, 0.01, True, ("a", "b", "c", "d", "e"), None,
                     (None,) * 5, 200000)
    p.apply_header(sparse.key, big, None, 64 * GB)
    assert p.row(sparse.key).est.coverage < 0.1 and p.row(sparse.key).est.raw_bytes > GB
    assert st(p, sparse.key) == ("new", "sparse", False, True) and row_text(p.row(sparse.key)).startswith("Sparse: 12")
    one = cand("One")
    p.add_candidate(one)
    p.toggle([one.key], True)
    p.apply_header(one.key, header(1, names=("Nucleus",)), None, 64 * GB)
    assert st(p, one.key) == ("new", "unsupported", False, False)  # R24 wins over the user's tick
    assert p.toggle([one.key], True) == [] and not p.row(one.key).checked
    seg = cand("Seg", kind="other_suffix", suffix="binary_seg_maps")
    p.add_candidate(seg)
    p.apply_header(seg.key, header(1, names=("Nucleus",)), None, 64 * GB)
    assert st(p, seg.key) == ("excluded", "unsupported", False, False)
    assert row_text(p.row(seg.key)) == "1-channel images can't be stitched yet (engine limitation)"


def test_default_state_function_table():
    r = PL.PlanRow("k", cand("A"))
    assert default_state(r) == ("new", "new", True, True)
    r.unfinished = OutputInfo("A.ome.tif", 1, 1)
    assert default_state(r) == ("new", "unfinished", True, True)
    r.unfinished, r.existing, r.existing_verdict = None, OutputInfo("A.ome.tif", 1, 1), "checking"
    assert default_state(r) == ("existing", "existing_checking", False, True)
    r.existing_verdict = "valid"
    assert default_state(r) == ("existing", "existing", False, True)
    r.existing_verdict = "incomplete"
    assert default_state(r) == ("new", "incomplete", True, True)
    for verdict in ("unreadable", "unchecked"):                 # D2: couldn't check it -> kept, never auto-ticked
        r.existing_verdict = verdict
        assert default_state(r) == ("existing", "unchecked", False, True)
    r.ready = OutputInfo("A.ome.tif", 1, 1)
    assert default_state(r) == ("new", "ready", True, True)


# ---- §3.8 output matching and verification
def test_output_index_matching_and_verification():
    a, b, c = cand("A"), cand("B"), cand("C")
    p = make(a, b, c)
    assert p.apply_output_index(index(["a.OME.tif"], partials=["B.ome.tif"], out_dir="/elsewhere")) == []
    changed = p.apply_output_index(index(["a.OME.tif"], partials=["B.ome.tif"]))
    assert set(changed) == {a.key, b.key}
    assert st(p, a.key) == ("existing", "existing_checking", False, True)
    assert st(p, b.key) == ("new", "unfinished", True, True) and st(p, c.key) == ("new", "new", True, True)
    p.apply_verification(f"{OUT}/a.OME.tif", "valid", "")
    assert st(p, a.key) == ("existing", "existing", False, True)
    assert row_text(p.row(a.key)).startswith("Skipped: already stitched (18.0 GB, ")
    p.toggle([a.key])
    assert st(p, a.key) == ("existing", "replace", True, True) and row_text(p.row(a.key)).startswith("Will replace")
    p.toggle([a.key])
    assert st(p, a.key) == ("existing", "existing", False, True)
    p.apply_output_index(index(["a.OME.tif"]))                  # same file again: verdict remembered
    assert st(p, a.key)[1] == "existing"
    p.apply_verification(f"{OUT}/a.OME.tif", "incomplete", "missing SubIFD")
    assert st(p, a.key)[:2] == ("new", "incomplete")
    p.set_out_dir("/data/other")
    assert st(p, a.key)[:2] == ("new", "new") and p.row(b.key).unfinished is None


def test_slide_naming_matches_v1_folder_named_output():
    c = cand("TS4410-07", prefix="TS4410_07_Scan1")
    p = make(c, naming="slide")
    p.apply_output_index(index(["TS4410-07.ome.tif"]))
    assert p.row(c.key).output_stem == "TS4410_07_Scan1" and p.row(c.key).section == "existing"
    p.set_naming("folder")
    assert p.row(c.key).output_stem == "TS4410-07" and display_name(c, full=True) == "TS4410-07 (tiles: TS4410_07_Scan1)"


# ---- §3.7 names, §2.3.5 rename
def test_duplicates_are_renamed_never_skipped():
    a = cand("duplicates/batchA/Slide9", prefix="Slide9_Scan1")
    b = cand("duplicates/batchB/Slide9", prefix="Slide9_Scan1", n=40)
    p = make(b, a)
    assert p.row(a.key).output_stem == "Slide9__batchA" and p.row(b.key).output_stem == "Slide9__batchB"   # D1
    flags = {f.code: f.text for f in p.row(b.key).flags}
    assert flags["renamed"] == "Renamed so nothing is overwritten (…__batchB)"
    assert flags["duplicate"] == "The same slide is also in duplicates/batchA/Slide9 (20 tiles)"
    assert {f.code: f.text for f in p.row(a.key).flags}["renamed"] == "Renamed so nothing is overwritten (…__batchA)"
    assert p.row(a.key).checked and p.row(b.key).checked and p.row(a.key).image_name is None


def test_rename_output():
    a, b, c = cand("A"), cand("B"), cand("C")
    p = make(a, b, c)
    p.apply_output_index(index(["Taken.ome.tif"]))
    assert p.rename_output(a.key, "B") == "Another slide already uses this name."
    assert "Suggested name: x_y" in p.rename_output(a.key, "x:y")
    assert p.rename_output(a.key, "  Renamed A.ome.tif ") is None
    r = p.row(a.key)
    assert (r.output_stem, r.output_edited) == ("Renamed A", True)
    p.toggle([c.key], True)
    assert p.rename_output(c.key, "taken") is None                 # the file exists: skipped unless ticked
    assert st(p, c.key)[:3] == ("existing", "existing_checking", False)
    assert p.rename_output(a.key, "") is None and p.row(a.key).output_stem == "A" and not p.row(a.key).output_edited
    assert p.rename_output(b.key, "A") == "Another slide already uses this name."
    assert p.rename_output(b.key, "Z") is None
    z = cand("x/Z")
    p.add_candidate(z)
    assert p.row(z.key).output_stem.startswith("Z__") and p.row(b.key).output_stem == "Z"   # user's name wins


# ---- flags
def test_flags():
    now = time.time()
    p = make(cand("run/s1", n=400), cand("run/s2", n=380), cand("run/s3", n=49), cand("run/s4", n=410),
             cand("fresh", mtime=now - 90), cand("WSIs", kind="legacy_all"))
    rid = lambda rel: next(r.row_id for r in p.rows() if r.cand.rel_folder == rel)   # noqa: E731
    flags = lambda rel: {f.code: (f.severity, f.text) for f in p.row(rid(rel)).flags}  # noqa: E731
    assert flags("run/s3")["sibling_low"] == ("info", "49 tiles — other folders here have about 390. Export incomplete?")
    assert "sibling_low" not in flags("run/s1")
    assert flags("fresh")["recent_change"] == ("warn", "Folder changed 2 min ago — is inForm still exporting?")
    p.apply_header(rid("run/s1"), header(problem="duplicate", names=("DAPI",) * 5), None, 64 * GB)
    assert flags("run/s1")["channels"] == ("warn", "Check channel names (DAPI ×5)")
    assert row_text(p.row(rid("run/s1"))) == "Check channel names (DAPI ×5)"
    p.apply_header(rid("run/s2"), header(6, "count"), None, 64 * GB)
    assert flags("run/s2")["channels"][1] == "Check channel names (5 names for 6 channels)"
    p.apply_header(rid("run/s4"), header(5, "missing", names=()), None, 4 * GB)
    assert flags("run/s4")["channels"][1] == "Check channel names (no names in the tiles)"
    assert flags("run/s4")["disk_mode"][1].startswith("Too big for memory: uses ~")
    p.set_channels([rid("run/s1")], ["a", "b", "c", "d", "e"])
    assert flags("run/s1")["channels"] == ("info", "Channel names set by you")
    p.set_channels([rid("run/s2")], None)
    assert flags("run/s2")["channels"] == ("info", "Generic channel names") and p.row(rid("run/s2")).generic_accepted
    p.apply_header(rid("WSIs"), header(first=3 * GB), None, 64 * GB)
    assert "legacy_big" not in flags("WSIs")
    p.toggle([rid("WSIs")])
    assert flags("WSIs")["legacy_big"] == ("warn", "The first image is 3.0 GB — this is probably not a tile folder")
    off = cand("cloudy")
    p.add_candidate(off)
    p.apply_header(off.key, header(offline=True), None, 64 * GB)
    assert [f.code for f in p.row(off.key).flags] == ["offline"] and p.row(off.key).est is None


def test_set_channels_only_for_rows_with_a_channel_problem():
    good, bad = cand("good"), cand("bad")
    p = make(good, bad)
    p.apply_header(good.key, header(), None, 64 * GB)
    p.apply_header(bad.key, header(problem="duplicate", names=("DAPI",) * 5), None, 64 * GB)
    assert p.set_channels([good.key, bad.key], ["a", "b", "c", "d", "e"]) == [bad.key]
    assert p.row(good.key).channel_override is None and p.row(bad.key).channel_override == ("a", "b", "c", "d", "e")
    assert p.set_channels([bad.key], ["too", "few"]) == []        # must match the channel count


# ---- §2.3.8 notes
def test_notes_become_grouped_rows():
    p = make(cand("A"))
    for n in [Note("outputs", SRC, 2, names=("a.ome.tif", "b.ome.tif")), Note("raw_scan", f"{SRC}/YS-01/Scan1"),
              Note("raw_scan", f"{SRC}/YS-02/Scan1"), Note("no_images", f"{SRC}/seq_runs/x"),
              Note("empty", f"{SRC}/pipeline_runs"), Note("not_searched_depth", f"{SRC}/a/b/c", 7),
              Note("not_searched_depth", f"{SRC}/a/b/d", 5), Note("system_skipped", SRC, 2, names=("$RECYCLE.BIN",
                                                                                               "System Volume Information")),
              Note("cloud_skipped", f"{SRC}/clouddrive"), Note("leftovers", SRC, 1), Note("unresponsive", f"{SRC}/slow",
                                                                                     detail="x"),
              Note("wide_pruned", f"{SRC}/WSIs/TCGA", 1083), Note("raw_scan", f"{SRC}/YS-01/Scan1")]:
        p.add_note(n)
    texts = [row_text(r) for r in p.rows("note")]
    assert texts == [
        "2 stitched images in src (a.ome.tif, b.ome.tif) — never used as tiles",
        "2 raw scans (.qptiff) — unmix in inForm and export component data first (YS-01/Scan1, YS-02/Scan1)",
        "WSIs/TCGA: 1,083 subfolders not searched (no slides found in the first 50 folders). Right-click → Search "
        "these folders too",
        "12 folders below 3 levels not searched. Right-click → Search deeper",
        "System folders skipped ($RECYCLE.BIN, System Volume Information)",
        "clouddrive (cloud) was not searched — choose it directly to search it",
        "2 folders without images (seq_runs, pipeline_runs)",
        "Didn't answer: slow. Right-click → Search this folder again"]
    assert [r.section for r in p.rows("excluded")] == ["note"] * 8 and p.totals().excluded == 8
    assert p.note_paths("note|not_searched_depth") == [f"{SRC}/a/b/c", f"{SRC}/a/b/d"]
    assert not any(r.checkable for r in p.rows("note"))
    for i in range(60):
        p.add_note(Note("no_access", f"{SRC}/locked{i:02d}"))
    access = [r for r in p.rows("note") if r.note.kind == "no_access"]
    assert len(access) == 51 and row_text(access[-1]).startswith("…and 10 more folders like this (locked50, ")


def test_a_pruned_wide_folder_says_how_many_sample_folders_could_not_be_searched():
    """FS-4/C2-V2: Note('wide_pruned').detail counts the sample folders that couldn't be searched completely."""
    p = make(cand("A"))
    p.add_note(Note("wide_pruned", f"{SRC}/cohort", 200, "2"))
    assert [row_text(r) for r in p.rows("note")] == [
        "cohort: 200 subfolders not searched (no slides found in the first 50 folders; 2 of them couldn't be "
        "searched). Right-click → Search these folders too"]


# ---- §2.3.6 select
def test_select_modes():
    big, small = cand("x/S9", prefix="Slide9_Scan1", n=176), cand("stamps", prefix="Slide9_Scan1", n=40)
    new, old = cand("N"), cand("O")
    p = make(big, small, new, old)
    p.apply_output_index(index(["O.ome.tif"]))
    p.select("none")
    assert p.ticked() == []
    p.select("all_new")
    assert {r.row_id for r in p.ticked()} == {big.key, small.key, new.key}
    p.toggle([old.key])
    p.select("not_stitched")
    assert {r.row_id for r in p.ticked()} == {big.key, small.key, new.key}
    p.select("untick_smaller_copies")
    assert {r.row_id for r in p.ticked()} == {big.key, new.key}
    p.select("invert")
    assert {r.row_id for r in p.ticked()} == {small.key}
    assert p.select("untick_waiting") == []                        # only during a run
    p.toggle_section("new")
    assert {r.row_id for r in p.ticked()} == {big.key, small.key, new.key}
    p.toggle_section("new")
    assert p.ticked() == [] and p.toggle_section("existing") == []


# ---- §2.4.1 issues
def test_blocking_issues():
    p = make()
    assert codes(p.issues(env(source=None, out_dir=None))) == ["no_source", "no_output", "none_ticked"]
    assert codes(p.issues(env(out_index_pending=True, scanning=True))) == ["checking_output", "scanning",
                                                                           "none_ticked"]
    issues = p.issues(env(scanning=True, paused=True))
    assert codes(issues, "block") == ["paused", "none_ticked"] and issues[0].action == "resume_scan"
    assert issues[1].text == "Tick at least one slide."
    a = cand("A")
    p.add_candidate(a)
    p.apply_output_index(index(["A.ome.tif"]))
    assert p.issues(env())[0].text == "Nothing new to stitch — the slide already has a stitched file. Tick it to " \
                                      "replace it."
    two = make(cand("A"), cand("B"))
    two.apply_output_index(index(["A.ome.tif", "B.ome.tif"]))
    assert two.issues(env())[0].text == ("Nothing new to stitch — all 2 slides already have stitched files. "
                                         "Tick one to replace it.")
    p.toggle([a.key])
    assert codes(p.issues(env(write_error="Permission denied."))) == ["output_error", "replace"]
    p.apply_header(a.key, header(), None, 64 * GB)
    idx = index(["A.ome.tif"], free=1000)
    p.apply_output_index(idx)
    issues = p.issues(env(out_index=idx))
    assert codes(issues, "block") == ["no_space_first"] and "free on data." in issues[0].text
    err = index(error=FsError("no_access", "Permission denied."))
    assert codes(p.issues(env(out_index=err)), "block") == ["output_error"]


def test_warnings_and_info():
    now = time.time()
    s1, s2 = cand("a/S9", prefix="Slide9"), cand("b/S9", prefix="Slide9", mtime=now - 30)
    legacy, ch = cand("WSIs", kind="legacy_all"), cand("ch")
    p = make(s1, s2, legacy, ch)
    p.apply_header(ch.key, header(problem="duplicate", names=("DAPI",) * 5), None, GB)
    p.apply_header(legacy.key, header(first=3 * GB), None, 64 * GB)
    p.toggle([legacy.key])
    idx = index(free=100 * 1024 ** 2, lane_class="cloud", system=True)
    p.apply_output_index(idx)
    issues = p.issues(env(out_index=idx, windows=True, out_dir="C:\\out", temp_lane_class="network"))
    assert codes(issues, "warn") == ["channels", "space_some", "system_disk", "temp_on_network", "recent_change",
                                     "legacy_big", "duplicates_ticked", "cloud_output"]
    by = {i.code: i for i in issues}
    assert by["channels"].text == "ch needs a look: check channel names (DAPI ×5)." and \
        by["channels"].rows == (ch.key,) and by["channels"].action == "fix_channels"
    assert by["recent_change"].text.startswith("S9__b: folder changed 1 min ago")     # two slides are "S9"
    assert by["duplicates_ticked"].text == "Slide9 is ticked in 2 places (a/S9, b/S9) — you'll get 2 files."   # MW-25
    assert codes(issues, "block") == []                           # the first slide's size is not known yet
    p2 = make(cand("A"))
    net = index(lane_class="network")
    p2.apply_output_index(net)
    assert codes(p2.issues(env(out_index=net))) == ["network_output"]


# ---- totals, runner protocol, carry-over
def test_totals_and_planned_keys():
    a, b, c = cand("A"), cand("B"), cand("C", n=3)
    p = make(a, b, c)
    p.apply_header(a.key, header(), None, 64 * GB)
    t = p.totals()
    assert (t.ticked, t.new, t.existing, t.tiles, t.unknown_estimates) == (2, 3, 0, 40, 1)
    assert t.out_typical == p.row(a.key).est.out_typical and p.planned_output_keys() == {"a.ome.tif", "b.ome.tif"}
    assert not p.multi_parent()
    p.add_candidate(cand("deep/D"))
    assert p.multi_parent()


def test_runner_protocol_and_end_run():
    rows = [cand(f"S{i}") for i in range(4)]
    p = make(*rows)
    p.toggle([rows[3].key], False)
    assert p.begin_run() == 3 and [r.status for r in p.rows()][:3] == ["waiting"] * 3
    job = p.claim_next(OUT, None)
    assert (job.row_id, job.run_index, job.final_name, job.replace, job.display) == (rows[0].key, 1, "S0.ome.tif",
                                                                                     False, "S0")
    assert p.toggle([rows[3].key], True) == [] and p.select("none") == []        # nothing new during a run
    p.set_run_text(job.row_id, "Stitching tiles 3/20")
    assert row_text(p.row(job.row_id)) == "Stitching tiles 3/20"
    assert p.rename_output(job.row_id, "x") is not None and p.rename_output(rows[2].key, "S2b") is None
    p.toggle([rows[1].key])                                                         # untick a waiting row
    assert row_text(p.row(rows[1].key)) == "Won't be stitched"
    p.finish(job.row_id, RowOutcome("done", "Done · 8.9 GB · 7 min"))
    job2 = p.claim_next(OUT, "/tmp/scratch")
    assert (job2.row_id, job2.run_index, job2.final_name, job2.temp_dir) == (rows[2].key, 2, "S2b.ome.tif",
                                                                             "/tmp/scratch")
    p.end_run()
    assert p.claim_next(OUT, None) is None
    assert st(p, rows[1].key) == ("new", "new", False, True) and p.row(rows[0].key).status == "done"
    assert row_text(p.row(rows[0].key)) == "Done · 8.9 GB · 7 min" and p.row(rows[2].key).status == "running"
    p.finish(rows[2].key, RowOutcome("failed", "Failed: The disk is full."))
    p.select("failed")
    assert [r.row_id for r in p.ticked()] == [rows[2].key]
    assert p.begin_run() == 1


def test_replace_jobs():
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif"]))
    p.apply_verification(f"{OUT}/B.ome.tif", "incomplete", "")
    p.toggle([a.key])
    p.begin_run()
    jobs = {j.row_id: j.replace for j in iter(lambda: p.claim_next(OUT, None), None)}
    assert jobs == {a.key: True, b.key: True}


def test_claim_finish_is_thread_safe():
    rows = [cand(f"S{i:03d}") for i in range(200)]
    p = make(*rows)
    p.begin_run()
    claimed, errors = [], []

    def runner():
        try:
            while True:
                job = p.claim_next(OUT, None)
                if job is None:
                    break
                claimed.append(job.row_id)
                p.set_run_text(job.row_id, "Stitching")
                p.finish(job.row_id, RowOutcome("done", "Done"))
        except Exception as e:                                   # pragma: no cover - reported below
            errors.append(repr(e))

    t = threading.Thread(target=runner)
    t.start()
    rnd = random.Random(7)
    unticked = set()
    for _ in range(1000):
        rid = rnd.choice(rows).key
        if p.toggle([rid], rnd.choice([None, False])):
            unticked.add(rid)
    t.join(30)
    p.end_run()
    assert not errors and len(claimed) == len(set(claimed))
    assert set(claimed) | unticked == {r.key for r in rows} and not set(claimed) & unticked
    assert all(p.row(r).status == "done" for r in claimed)
    assert all(p.row(r).status == "new" and not p.row(r).checked for r in unticked)


def test_carry_over_keeps_the_users_choices():
    a, b, c, d = cand("A"), cand("B", n=3), cand("C"), cand("D")
    p = make(a, b, c, d)
    p.toggle([a.key])                                               # untick a ready slide
    p.toggle([b.key])                                               # tick a few-tiles slide
    p.rename_output(c.key, "C renamed")
    p.apply_header(d.key, header(problem="duplicate", names=("DAPI",) * 5), None, 64 * GB)
    p.set_channels([d.key], ["1", "2", "3", "4", "5"])
    carry = p.carry_state()
    assert carry == {a.key: {"checked": False}, b.key: {"checked": True}, c.key: {"output_stem": "C renamed"},
                     d.key: {"channel_override": ("1", "2", "3", "4", "5")}}
    q = make(d, c, b, a, carry=carry)
    assert [st(q, k)[2] for k in (a.key, b.key, c.key)] == [False, True, True]
    assert q.row(c.key).output_stem == "C renamed" and q.row(d.key).channel_override == ("1", "2", "3", "4", "5")
    q.begin_run()
    q.finish(q.claim_next(OUT, None).row_id, RowOutcome("done", "Done"))
    assert b.key not in q.carry_state()                             # a stitched row's tick is not carried


def test_synthetic_tree_end_to_end(synth_tree, collector):
    """Discovery -> Plan on synth_tree: the duplicates are renamed and the inForm export is split (§3.11)."""
    fs = FsService(collector, MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw"))
    try:
        Discovery(1, str(synth_tree), fs, collector, DiscoveryOptions()).start()
        collector.wait_type(ScanFinished)
    finally:
        fs.shutdown()
    p = Plan(str(synth_tree), str(synth_tree / "stitched"))
    for e in collector.events:
        if isinstance(e, CandidateFound):
            p.add_candidate(e.candidate)
        elif isinstance(e, ScanNote):
            p.add_note(e.note)
    names = {r.cand.rel_folder + ":" + r.cand.prefix: r.output_stem for r in p.rows() if r.section != "note"
             and r.cand.kind != "other_suffix"}
    assert names["duplicates/batchA/Slide9:Slide9_Scan1"] == "Slide9__batchA"                      # D1
    assert names["duplicates/batchB/Slide9:Slide9_Scan1"] == "Slide9__batchB"
    assert names["inform_export:S1_Scan1"] == "S1_Scan1" and names["inform_export:S2_Scan1"] == "S2_Scan1"
    assert names["round1_unmix/R1-02:R1-02_Scan1"] == "R1-02"
    assert [r.cand.rel_folder for r in p.rows("new")][:2] == ["duplicates/batchA/Slide9", "duplicates/batchB/Slide9"]
    excluded = [(r.cand.rel_folder, r.status) for r in p.rows("excluded") if r.section != "note"]
    assert excluded == [("inform_export", "other_suffix"), ("unrelated/WSIs", "legacy_all")]
    assert p.multi_parent() and p.totals().ticked == 0            # synthetic slides have < 10 tiles
    assert {r.status for r in p.rows("new")} == {"few_tiles"}


def test_a_duplicate_keeps_its_folder_suffix_while_its_twin_runs():
    """Rows that started keep their names; a waiting duplicate must not be renamed to '__2' meanwhile."""
    p = make(cand("batchA/Slide9", "Slide9_Scan1"), cand("batchB/Slide9", "Slide9_Scan1"))
    a, b = [r.row_id for r in p.rows("new")]
    assert (p.row(a).output_stem, p.row(b).output_stem) == ("Slide9__batchA", "Slide9__batchB")
    p.begin_run()
    p.claim_next(OUT, None)                                          # batchA/Slide9 is running
    p.set_naming("folder")                                           # any recompute during the run
    assert (p.row(a).output_stem, p.row(b).output_stem) == ("Slide9__batchA", "Slide9__batchB")


def test_a_ticked_excluded_row_says_it_will_be_stitched():
    p = make(cand("x", "S1_Scan1", suffix="composite_image", kind="other_suffix"), cand("w", kind="legacy_all"))
    other, legacy = sorted(p.rows("excluded"), key=lambda r: r.cand.kind != "other_suffix")
    assert row_text(other).endswith("Tick to stitch anyway")
    p.toggle([other.row_id, legacy.row_id], True)
    assert row_text(p.row(other.row_id)) == "Will be stitched (composite_image tiles, not unmixed components)"
    assert row_text(p.row(legacy.row_id)) == "Will be stitched (every TIFF in this folder, like the old version)"


# ---- fix round 1: names (D1), existing outputs (D2), retries, rebuilds and issues
def replace_lines(p, **kw):
    return [i.text for i in p.issues(env(**kw)) if i.code == "replace"]


def test_colliding_folder_names_are_named_after_their_tiles():
    """DS-5/D1: in the <SlideID>/Scan1 layout no slide keeps the bare 'Scan1', so a slide exported later (sorting
    first) renames nobody, and an old Scan1.ome.tif is matched to none of them but shown as a note."""
    ys2, ys3 = cand("SL02/Scan1", "SL02_Scan1"), cand("SL03/Scan1", "SL03_Scan1")
    p = make(ys2, ys3)
    p.apply_output_index(index(["Scan1.ome.tif", "SL02_Scan1.ome.tif"]))
    names = lambda: {r.cand.rel_folder: (r.output_stem, r.image_name, r.section) for r in p.rows()   # noqa: E731
                     if r.section != "note"}
    assert names() == {"SL02/Scan1": ("SL02_Scan1", "SL02_Scan1", "existing"),
                       "SL03/Scan1": ("SL03_Scan1", "SL03_Scan1", "new")}
    ys1 = cand("SL01/Scan1", "SL01_Scan1")
    changed = p.add_candidate(ys1)
    assert names()["SL01/Scan1"] == ("SL01_Scan1", "SL01_Scan1", "new") and ys1.key in changed
    assert names()["SL02/Scan1"] == ("SL02_Scan1", "SL02_Scan1", "existing")
    assert [row_text(r) for r in p.rows("note")] == [
        "Scan1.ome.tif: may belong to one of 3 slides named 'Scan1'; not matched"]
    assert "note|unmatched|scan1.ome.tif" in changed and p.totals().excluded == 1
    assert {f.code: f.text for f in p.row(ys1.key).flags}["renamed"] == \
        "Named after its tiles: other slide folders are also called Scan1"
    p.toggle([ys2.key])                                             # the sheet tells the 'Scan1's apart
    assert replace_lines(p) == ["1 file will be replaced: SL02_Scan1."]
    p.apply_output_index(index(["SL02_Scan1.ome.tif"]))             # the old file is gone: so is its note
    assert p.rows("note") == []


def test_a_v1_output_is_not_given_to_one_of_its_namesakes():
    """DS-5: v1 named outputs by folder in os.walk order; v2 must not pick one of the four 'Scan1' slides for it."""
    rows = [cand(f"{s}/Scan1", f"{s}_Scan1") for s in ("S9", "S10", "S11", "S2")]
    p = make(*rows)
    p.apply_output_index(index(["Scan1.ome.tif"]))
    assert [st(p, c.key) for c in rows] == [("new", "new", True, True)] * 4
    assert [r.note.count for r in p.rows("note")] == [4]


def test_slide_naming_uses_the_folder_file_only_for_a_unique_folder():
    """DS-4: the v1 '<folder>.ome.tif' counts only for a folder name no other folder has and that is no row's own
    name; it reads 'Already stitched as …', and ticking the slide makes a new file, not a replacement."""
    rows = [cand(f"{s}/Scan1", f"{s}_Scan1") for s in ("SL01", "SL02", "SL03")]
    p = make(*rows, naming="slide")
    p.apply_output_index(index(["Scan1.ome.tif"]))
    assert [st(p, c.key) for c in rows] == [("new", "new", True, True)] * 3
    assert [row_text(r) for r in p.rows("note")] == [
        "Scan1.ome.tif: may belong to one of 3 slides named 'Scan1'; not matched"]
    one = make(rows[0], naming="slide")
    one.apply_output_index(index(["Scan1.ome.tif"]))
    r = one.row(rows[0].key)
    assert st(one, r.row_id) == ("existing", "existing_checking", False, True) and r.output_stem == "SL01_Scan1"
    assert row_text(r) == "Already stitched as Scan1.ome.tif — checking it's complete…"
    one.apply_verification(f"{OUT}/Scan1.ome.tif", "valid", "")
    assert row_text(r).startswith("Already stitched as Scan1.ome.tif (18.0 GB, ")
    one.toggle([r.row_id])
    assert st(one, r.row_id) == ("existing", "new", True, True)
    assert row_text(r) == "New file (Scan1.ome.tif stays as it is)"
    assert one.totals().replace == 0 and replace_lines(one) == []
    one.begin_run()
    assert (one.claim_next(OUT, None).final_name, r.replaces) == ("SL01_Scan1.ome.tif", False)
    broken = make(rows[1], naming="slide")                            # a broken v1 file: the slide is new
    broken.apply_output_index(index(["Scan1.ome.tif"]))
    broken.apply_verification(f"{OUT}/Scan1.ome.tif", "incomplete", "")
    assert st(broken, rows[1].key) == ("new", "new", True, True)
    export = cand("exp", "Scan1", multi=True)                         # 'Scan1.ome.tif' is another row's own name
    other = make(rows[2], export, naming="slide")
    other.apply_output_index(index(["Scan1.ome.tif"]))
    assert st(other, rows[2].key)[:2] == ("new", "new") and other.row(export.key).section == "existing"


def test_a_slide_stitched_under_its_own_name_leaves_the_folder_file_alone():
    """C3-V1 (DS-4, D11): with slide naming, a partial or a finished partial of the slide's own name shows it is being
    stitched under that name. Its folder's old '<folder>.ome.tif' is then not its file: nothing is a replacement,
    and the job may not overwrite a final of its own name that turns up meanwhile."""
    c = cand("SL01/Scan1", "SL01_Scan1")
    texts = {"unfinished": "Unfinished last time — will start again",
             "ready": "Finished last time — will move the new image into place"}   # (no 'the old file was in use')
    for kind, status in (("partials", "unfinished"), ("ready", "ready")):
        p = make(c, naming="slide")
        p.apply_output_index(index(["Scan1.ome.tif"], **{kind: ["SL01_Scan1.ome.tif"]}))
        r = p.row(c.key)
        assert (st(p, c.key), r.existing, r.replaces, row_text(r)) == (("new", status, True, True), None, False,
                                                                       texts[status])
        assert replace_lines(p) == [] and p.totals().replace == 0
        p.begin_run()
        job = p.claim_next(OUT, None)
        assert (job.final_name, job.replace, job.finalize_only) == ("SL01_Scan1.ome.tif", False, status == "ready")


def test_a_file_that_couldnt_be_checked_is_kept_until_ticked():
    """D2 (DS-7, MW-3, FS-5): only a file that looks incomplete is redone by default. One that couldn't be checked
    stays unticked in Already stitched, and ticking it puts the overwrite on the sheet."""
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.toggle([b.key], False)
    p.apply_output_index(index(["A.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "unchecked", "Permission denied.")
    assert st(p, a.key) == ("existing", "unchecked", False, True) and not p.row(a.key).replaces
    assert row_text(p.row(a.key)) == "Couldn't check the existing file (Permission denied) — tick to replace it"
    assert codes(p.issues(env()), "block") == ["none_ticked"] and p.existing_counts()["unchecked"] == 1
    p.apply_verification(f"{OUT}/A.ome.tif", "unreadable", "not a TIFF file")
    assert row_text(p.row(a.key)) == "Couldn't check the existing file (not a TIFF file) — tick to replace it"
    p.toggle([a.key])
    assert st(p, a.key)[1] == "replace" and replace_lines(p) == ["1 file will be replaced: A."]
    p.begin_run()
    assert p.claim_next(OUT, None).replace


def test_every_overwrite_is_on_the_sheet():
    """D2 (DS-7): incomplete, unfinished and ready rows whose final exists are replacements: Job.replace, the
    sheet's warning and Totals.replace. An unfinished row without a final is not (DS-12)."""
    a, b, c, d = cand("A"), cand("B"), cand("C"), cand("D")
    p = make(a, b, c, d)
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif", "C.ome.tif"], partials=["B.ome.tif", "D.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "incomplete", "1 of 3 pyramid levels")
    p.apply_verification(f"{OUT}/C.ome.tif", "valid", "")
    assert [p.row(k).status for k in (a.key, b.key, c.key, d.key)] == ["incomplete", "unfinished", "existing",
                                                                         "unfinished"]
    assert p.row(b.key).existing.name == "B.ome.tif" and p.row(d.key).existing is None
    assert replace_lines(p) == ["2 files will be replaced: A (looks incomplete), B."] and p.totals().replace == 2
    # B's old file is never checked (it will be redone): it is not 'checking' for ever (C3-V4)
    assert p.existing_counts() == {"valid": 1, "checking": 0, "incomplete": 1, "unchecked": 0, "gaps": 0}
    p.begin_run()
    assert {j.row_id: j.replace for j in iter(lambda: p.claim_next(OUT, None), None)} == {a.key: True, b.key: True,
                                                                                         d.key: False}


def test_unticked_redo_rows_say_they_wont_be_stitched():
    """FS-18."""
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.apply_output_index(index(["A.ome.tif"], partials=["B.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "incomplete", "")
    p.toggle([a.key, b.key], False)
    assert row_text(p.row(a.key)) == "Won't be stitched (the existing file looks incomplete)"
    assert row_text(p.row(b.key)) == "Won't be stitched (unfinished last time)"


def test_a_complete_partial_is_moved_into_place():
    """DS-9/D3: a partial with a COMPLETE sidecar is 'ready': ticked, finalize-only, a replacement when the old final
    is still there."""
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.apply_output_index(index(["A.ome.tif"], partials=["A.ome.tif"], ready=["A.ome.tif", "B.ome.tif"]))
    assert st(p, a.key) == ("new", "ready", True, True) and st(p, b.key) == ("new", "ready", True, True)
    assert row_text(p.row(a.key)) == ("Finished last time — the old file was in use; will move the new image into "
                                      "place")
    assert row_text(p.row(b.key)) == "Finished last time — will move the new image into place"
    assert replace_lines(p) == ["1 file will be replaced: A."] and p.existing_counts()["checking"] == 0
    p.toggle([b.key])
    assert row_text(p.row(b.key)) == "Won't be stitched (finished last time, not moved into place yet)"
    p.begin_run()
    job = p.claim_next(OUT, None)
    assert (job.row_id, job.finalize_only, job.replace) == (a.key, True, True) and p.claim_next(OUT, None) is None


def test_retry_keeps_what_was_planned():
    """DS-6/MW-6: [Retry failed] redoes a replaced and an incomplete file (the sheet lists them again), and a row a
    stopped retry never reached keeps its failed outcome, so Retry failed still finds it."""
    a, b, c = cand("A"), cand("B"), cand("C")
    p = make(a, b, c)
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "valid", "")
    p.apply_verification(f"{OUT}/B.ome.tif", "incomplete", "")
    p.toggle([a.key])
    p.toggle([c.key], False)
    sheet = ["2 files will be replaced: B (looks incomplete), A."]   # display order: New, then Already stitched
    assert replace_lines(p) == sheet
    assert p.begin_run() == 2
    for job in iter(lambda: p.claim_next(OUT, None), None):
        assert job.replace
        p.finish(job.row_id, RowOutcome("failed", "Failed: The disk is full."))
    p.end_run()
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif"]))            # the index after the run
    p.select("failed")                                                # [Retry failed]
    assert replace_lines(p) == sheet
    assert p.begin_run() == 2
    job = p.claim_next(OUT, None)
    assert (job.row_id, job.replace) == (b.key, True)
    p.finish(job.row_id, RowOutcome("done", "Done · 1 GB"))
    p.end_run()                                                       # stopped after B: A was never claimed
    ra = p.row(a.key)
    assert (ra.status, ra.outcome.status, row_text(ra)) == ("failed", "failed", "Failed: The disk is full.")
    p.select("failed")
    assert [r.row_id for r in p.pending()] == [a.key] and p.begin_run() == 1
    assert p.claim_next(OUT, None).replace
    lost = PL.PlanRow("k", cand("Z"), status="cancelled")             # a run status without its outcome
    assert row_text(lost) == "Stopped — will start again next time"


def test_a_final_that_appeared_during_the_run_is_still_not_replaced():
    """DS-12: a retried row that was new at planning stays Job.replace=False, so the runner's guard keeps a file
    another stitcher wrote meanwhile."""
    a = cand("A")
    p = make(a)
    p.begin_run()
    p.finish(p.claim_next(OUT, None).row_id, RowOutcome("skipped", "Skipped: the output appeared since planning"))
    p.end_run()
    p.apply_output_index(index(["A.ome.tif"]))
    p.select("failed")
    p.begin_run()
    assert p.claim_next(OUT, None).replace is False


def test_an_output_made_with_missing_tiles_says_so():
    """DS-1: the ledger's gap count shows on the Already stitched row."""
    a = cand("A")
    p = make(a)
    p.apply_output_index(index(["A.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "valid", "", gaps=3)
    assert st(p, a.key) == ("existing", "existing", False, True)
    assert row_text(p.row(a.key)) == "Already stitched — 3 tiles were missing when it was made. Tick to redo it"


def test_a_name_typed_onto_an_existing_file_says_so():
    """MW-20: renaming onto another file's name is not 'Already stitched'; the note survives a rescan."""
    a = cand("A")
    p = make(a, cand("B"))
    p.apply_output_index(index(["Old_Result.ome.tif"]))
    assert p.rename_output(a.key, "Old_Result") is None
    r = p.row(a.key)
    assert st(p, a.key)[:3] == ("existing", "existing_checking", False)
    assert row_text(r) == "A file with this name already exists — tick to replace it"
    p.apply_verification(f"{OUT}/Old_Result.ome.tif", "valid", "")
    assert row_text(r) == "A file with this name already exists — tick to replace it"
    assert p.existing_counts()["valid"] == 0
    q = make(a, carry=p.carry_state())
    q.apply_output_index(index(["Old_Result.ome.tif"]))
    assert row_text(q.row(a.key)) == "A file with this name already exists — tick to replace it"
    p.toggle([a.key])
    assert row_text(r).startswith("Will replace the file from ") and replace_lines(p) == [
        "1 file will be replaced: A."]
    assert p.rename_output(a.key, "Fresh") is None and not p.row(a.key).name_taken


def test_committing_the_name_a_slide_has_changes_nothing():
    """C3-V2 (MW-20): the rename editor commits on Enter and when the user clicks away. The name a slide already has,
    also in another case, is not 'a file with this name already exists', and a tick the user gave stays."""
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.apply_output_index(index(["A.ome.tif", "Old_Result.ome.tif"]))
    p.apply_verification(f"{OUT}/A.ome.tif", "valid", "")
    for name in ("A", "a"):
        assert p.rename_output(a.key, name) is None and not p.row(a.key).name_taken
        assert st(p, a.key) == ("existing", "existing", False, True)
        assert row_text(p.row(a.key)).startswith("Skipped: already stitched (18.0 GB, ")
    assert p.existing_counts()["valid"] == 1 and p.carry_state()[a.key] == {"output_stem": "a"}
    assert p.rename_output(b.key, "Old_Result") is None and p.row(b.key).name_taken
    p.toggle([b.key])                                                 # ticked to replace that file
    assert p.rename_output(b.key, "Old_Result") is None
    assert st(p, b.key)[1:3] == ("replace", True) and p.row(b.key).name_taken


def test_a_name_typed_onto_a_file_that_looks_incomplete_stays_unticked():
    """MW-20/§2.3.5: a typed name onto an existing file leaves the slide unticked whatever the check finds there
    (it is not the slide's file to 'redo'); ticking it puts the overwrite on the sheet."""
    a = cand("A")
    p = make(a)
    p.apply_output_index(index(["Old_Result.ome.tif"]))
    assert p.rename_output(a.key, "Old_Result") is None
    p.apply_verification(f"{OUT}/Old_Result.ome.tif", "incomplete", "1 of 3 pyramid levels")
    assert st(p, a.key) == ("existing", "existing", False, True) and not p.row(a.key).replaces
    assert row_text(p.row(a.key)) == "A file with this name already exists — tick to replace it"
    p.toggle([a.key])
    assert st(p, a.key)[1] == "replace" and replace_lines(p) == ["1 file will be replaced: A."]


def test_a_verdict_about_another_version_of_the_file_is_ignored():
    """A check that read the file before it changed must not vouch for the version now in the index."""
    a = cand("A")
    p = make(a)
    p.apply_output_index(index(["A.ome.tif"]))
    e = p.row(a.key).existing
    assert p.apply_verification(f"{OUT}/A.ome.tif", "valid", "", size=e.size - 1, mtime_ns=e.mtime_ns) == []
    assert st(p, a.key)[1] == "existing_checking"
    assert p.apply_verification(f"{OUT}/A.ome.tif", "valid", "", size=e.size, mtime_ns=e.mtime_ns) == [a.key]
    assert st(p, a.key)[1] == "existing"


def test_rebuild_for_another_save_folder_drops_the_run():
    """DS-15/MWV-1: after a batch, another save folder gets a fresh plan of the same slides: no run statuses, the
    user's choices kept, headers, estimates and notes too; the same folder keeps its index."""
    a, b, c = cand("A"), cand("B"), cand("C")
    p = make(a, b, c)
    p.apply_header(a.key, header(), None, 64 * GB)
    p.toggle([c.key])                                                 # the user unticks C
    p.add_note(Note("raw_scan", f"{SRC}/SL-01/Scan1"))
    p.begin_run()
    for job in iter(lambda: p.claim_next(OUT, None), None):
        p.finish(job.row_id, RowOutcome("done", "Done · 1 GB"))
    p.end_run()
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif"]))
    assert p.pending() == [] and p.issues(env())[0].text == ("All ticked slides are done — Start a new batch to "
                                                             "stitch more.")
    q = p.rebuild("/data/out2")
    assert q.out_dir == "/data/out2" and [(r.status, r.checked, r.outcome) for r in q.rows("new")] == [
        ("new", True, None), ("new", True, None), ("new", False, None)]
    assert [r.row_id for r in q.pending()] == [a.key, b.key] and q.row(a.key).est == p.row(a.key).est
    assert [row_text(r) for r in q.rows("note")] == [row_text(r) for r in p.rows("note")]
    same = p.rebuild()
    assert same.out_dir == OUT and [same.row(k).status for k in (a.key, b.key, c.key)] == [
        "existing_checking", "existing_checking", "new"]


def test_the_section_box_does_nothing_during_a_run():
    """MW-2: one click on the New slides box must not untick every waiting slide."""
    rows = [cand(f"S{i}") for i in range(4)]
    p = make(*rows)
    p.begin_run()
    p.claim_next(OUT, None)
    assert p.toggle_section("new") == [] and all(r.checked for r in p.rows("new"))
    assert len(p.select("untick_waiting")) == 3                        # the explicit way still works


def test_jobs_name_twins_like_the_plan():
    p = make(cand("batchA/Slide1_Scan1"), cand("batchB/Slide1_Scan1"))
    labels = [p.label(r) for r in p.pending()]
    p.begin_run()
    jobs = list(iter(lambda: p.claim_next(OUT, None), None))
    assert [j.display for j in jobs] == labels and len(set(labels)) == 2


def test_existing_gap_counts_follow_the_current_image():
    a, b, c = cand("A"), cand("B"), cand("C")
    p = make(a, b, c)
    p.apply_output_index(index(["A.ome.tif", "B.ome.tif", "C.ome.tif"], ready=["C.ome.tif"]))
    for name in ("A", "B", "C"):
        p.apply_verification(f"{OUT}/{name}.ome.tif", "valid", "", gaps=2)
    assert p.existing_counts()["valid"] == p.existing_counts()["gaps"] == 2
    p.finish(a.key, RowOutcome("done", "Done"))            # a successful replacement removed the old gaps
    assert p.existing_counts()["valid"] == 2 and p.existing_counts()["gaps"] == 1
    p.finish(c.key, RowOutcome("done_gaps", "Done with gaps"))
    assert p.existing_counts()["valid"] == 3 and p.existing_counts()["gaps"] == 2


def test_slides_stitched_in_this_plan_are_not_counted_again():
    """MW-7: after a stopped run the totals, the space checks and the planned names count what Start would run."""
    big, s2, s3 = cand("Big", n=60), cand("S2"), cand("S3")
    p = make(big, s2, s3)
    for c in (big, s2, s3):
        p.apply_header(c.key, header(), None, 64 * GB)
    first = p.row(big.key).est.out_typical
    p.begin_run()
    p.finish(p.claim_next(OUT, None).row_id, RowOutcome("done", "Done"))
    p.end_run()
    idx = index(free=first - 1)
    p.apply_output_index(idx)
    assert codes(p.issues(env(out_index=idx)), "block") == [] and p.row(s2.key).est.out_typical * 2 < first
    t = p.totals()
    assert (t.ticked, t.tiles, t.out_typical) == (2, 40, 2 * p.row(s2.key).est.out_typical)
    assert p.planned_output_keys() == {"s2.ome.tif", "s3.ome.tif"} and len(p.ticked()) == 3


def test_per_row_warnings_make_one_line_each():
    """MW-5/MW-25: the sheet gets one line per code that names the slides, not one line per slide."""
    now = time.time()
    p = make(*[cand(f"S{i:02d}", mtime=now - 90) for i in range(40)])
    [rec] = [i for i in p.issues(env()) if i.code == "recent_change"]
    assert rec.text == "40 slide folders changed in the last 5 min — is inForm still exporting? (S00, S01, S02, …)"
    assert len(rec.rows) == 40
    w1, w2 = cand("x/WSIs", kind="legacy_all"), cand("y/Big", kind="legacy_all")
    q = make(w1, w2, cand("a/S9", "Slide9"), cand("b/S9", "Slide9"), cand("a/S7", "S7"), cand("b/S7", "S7"))
    for c in (w1, w2):
        q.toggle([c.key])
        q.apply_header(c.key, header(first=3 * GB), None, 64 * GB)
    by = {i.code: i for i in q.issues(env())}
    assert by["legacy_big"].text == ("2 ticked folders start with images over 2 GB — probably not tile folders "
                                     "(WSIs, Big)")
    assert by["duplicates_ticked"].text == ("2 slides are ticked in more than one place — you'll get 4 files "
                                            "(S7, Slide9).") and len(by["duplicates_ticked"].rows) == 4


def test_new_issues():
    """MW-24 (reason first, short place), FS-6 (unsearched), MW-26 (missing levels), PL-6 (cloud-synced)."""
    p = make(cand("A"))
    assert [i.text for i in p.issues(env(write_error="Permission denied")) if i.code == "output_error"] == [
        "Can't save to out: Permission denied."]
    assert [i.text for i in p.issues(env(unsearched=3)) if i.code == "unsearched"] == [
        "3 folders couldn't be searched — slides in them are missing from this list."]
    deep = index(out_dir="/data/Stiched/ProjectC/stitched", missing_levels=3, nearest_existing="/data")
    [miss] = [i for i in p.issues(env(out_index=deep)) if i.code == "missing_levels"]
    assert (miss.severity, miss.text) == ("warn", "3 new folders will be created inside /data: Stiched/ProjectC/"
                                                  "stitched")
    one = index(missing_levels=1, nearest_existing="/data")
    assert "missing_levels" not in codes(p.issues(env(out_index=one)))
    synced = index(cloud_synced=True)
    assert "cloud_output" in codes(p.issues(env(out_index=synced)), "warn")


def test_windows_long_paths_block_start():
    """PL-7: on Windows the longest path the runner writes (a name beside the partial: a sidecar, or the name a stale
    partial gets while it is taken over; the run's summary table; the scratch file in a custom temp folder) must fit
    in 259 characters unless long paths are enabled."""
    p = make(cand("Slide_0001"))
    at = lambda n, **kw: env(out_dir="C:\\" + "x" * (n - 3), windows=True, **kw)       # noqa: E731
    stale = len("/.stitcher-in-progress/Slide_0001.ome.tif.stale-0a1b2c3d")   # 6 longer than '.complete' (C3-V5)
    assert "long_path" not in codes(p.issues(at(259 - stale)))
    [block] = [i for i in p.issues(at(260 - stale)) if i.code == "long_path"]
    assert (block.severity, block.text, block.action) == ("block", "Paths too long for Windows (260 characters, 259 "
                                                          "allowed) for Slide_0001. Choose a shorter save folder or "
                                                          "output name.", "browse_output")
    assert preflight.STALE_INFIX == N.STALE_INFIX                                # the runner's takeover name
    short = make(cand("A"))                                                      # the summary table is longer
    csv = len("/YourFavouriteStitcher_summary_20261011-104141.csv")
    assert "long_path" not in codes(short.issues(at(259 - csv))) and "long_path" in codes(short.issues(at(260 - csv)))
    assert "long_path" not in codes(p.issues(at(400, windows_long_paths=True)))
    assert "long_path" not in codes(p.issues(env(out_dir="/" + "x" * 400)))
    assert "long_path" in codes(p.issues(at(50, temp_dir="D:\\" + "t" * 230)))   # size unknown: maybe disk mode
    p.apply_header(p.rows()[0].row_id, header(), None, 64 * GB)                  # fits in memory: no scratch file
    assert "long_path" not in codes(p.issues(at(50, temp_dir="D:\\" + "t" * 230)))


def test_totals_tell_sizes_to_come_from_unknown_sizes():
    """MWV-2: a row that can never be estimated keeps the known total instead of 'estimated once read'."""
    a, legacy, off = cand("A"), cand("WSIs", kind="legacy_all"), cand("Cloudy")
    p = make(a, legacy, off)
    p.toggle([legacy.key], True)
    assert (p.totals().unknown_estimates, p.totals().unsized) == (2, 1)
    for c, h in ((a, header()), (legacy, header()), (off, header(offline=True))):
        p.apply_header(c.key, h, None, 64 * GB)
    t = p.totals()
    assert (t.ticked, t.unknown_estimates, t.unsized, t.out_typical) == (3, 0, 2, p.row(a.key).est.out_typical)


def test_not_mounted_notes_are_one_row():
    """FS-2/D10: folders skipped as unmounted or disconnected get one aggregated note row."""
    p = make(cand("A"))
    p.add_note(Note("not_mounted", f"{SRC}/mnt", 2, "autofs", ("labshare", "scans")))
    p.add_note(Note("not_mounted", SRC, 1, "disconnected", ("Z",)))
    assert [row_text(r) for r in p.rows("note")] == [
        "3 folders not searched: not mounted or not connected (mnt/labshare, mnt/scans, Z). Open them, then Rescan"]
    q = make(cand("A"))
    q.add_note(Note("not_mounted", SRC, 2, "pseudo", ("proc", "sys")))
    assert [row_text(r) for r in q.rows("note")] == ["2 system folders skipped (proc, sys)"]


def test_slides_differing_only_in_case_are_both_kept(monkeypatch):
    """FS-12/PL-5: on a case-sensitive filesystem Run1/S1_Scan1 and run1/S1_Scan1 are two slides."""
    monkeypatch.setattr(N, "_CASE_INSENSITIVE_FS", False)
    tiles = [Entry(f"S1_Scan1_[{i * 900},0]_component_data.tif", False, False) for i in range(12)]
    p = Plan(SRC, OUT)
    for folder in (f"{SRC}/Run1/S1_Scan1", f"{SRC}/run1/S1_Scan1"):
        for c in N.candidates_from_summary(N.summarize_listing(folder, tiles), root=SRC, depth=2, dir_mtime=None):
            p.add_candidate(c)
    rows = p.rows("new")
    assert [r.cand.rel_folder for r in rows] == ["Run1/S1_Scan1", "run1/S1_Scan1"]
    assert len({r.output_stem for r in rows}) == 2 and all(r.output_stem.startswith("S1_Scan1__") for r in rows)
    assert [f.code for f in rows[0].flags] == ["renamed", "duplicate"]


def test_an_odd_folder_name_does_not_stop_the_plan(monkeypatch):
    """FS-14: a long name with an undecodable byte gets a name; a name that can't be made at all falls back."""
    weird = cand(os.fsdecode(b"x" * 190 + b"\xe9"))
    p = make(weird, cand("B"))
    assert len(p.row(weird.key).output_stem) == 179 and p.toggle([weird.key]) and p.issues(env()) == []
    real = PL.output_stem

    def odd(c, naming):
        if c.folder_name == "Bad":
            raise UnicodeEncodeError("utf-8", "Bad", 0, 1, "surrogates not allowed")
        return real(c, naming)
    monkeypatch.setattr(PL, "output_stem", odd)
    q = make(cand("Bad"), cand("Good"))
    assert [r.output_stem for r in q.rows("new")] == ["slide", "Good"] and q.totals().ticked == 2


def test_batches_and_time_based_flags(monkeypatch):
    """MW-8/FS-16: issues() and totals() never recompute; a batch recomputes once and reports what changed;
    refresh() ages the time-based flags."""
    now = time.time()
    p = make(cand("A", mtime=now - 90))
    calls = []
    real = Plan._recompute
    monkeypatch.setattr(Plan, "_recompute", lambda self: (calls.append(1), real(self))[1])
    p.issues(env())
    p.totals()
    assert calls == []
    with p.batch() as changed:
        assert [p.add_candidate(cand(f"S{i}")) for i in range(20)] == [[]] * 20
        assert p.apply_header("nope", header(), None, GB) == [] and p.toggle([p.rows()[0].row_id]) == []
        with p.batch() as inner:
            p.add_candidate(cand("T"))
    assert len(calls) == 1 and len(changed) == 22 and inner == [] and len(p.rows()) == 22
    assert p.apply_header(cand("T").key, header(), None, 64 * GB) == [cand("T").key] and len(calls) == 1
    rid = cand("A").key
    monkeypatch.setattr(PL.time, "time", lambda: now + 400)          # no longer recent: the sheet knows at once
    assert "recent_change" not in codes(p.issues(env())) and p.row(rid).flags[0].code == "recent_change"
    monkeypatch.setattr(PL.time, "time", lambda: now + 150)
    assert p.refresh() == [rid] and p.row(rid).flags[0].text.startswith("Folder changed 4 min ago")
    assert len(calls) == 2


def test_a_header_that_ends_a_slides_run_frees_its_name_at_once():
    """C3-V3 (MW-8): a slide that started keeps its name. When a 1-channel header takes it out of its run (R24), its
    name and what depends on it (here an old file no slide may claim now) follow at once, as a recompute would."""
    x, y = cand("x/S1"), cand("y/S1")
    p = make(x)
    p.begin_run()
    p.finish(p.claim_next(OUT, None).row_id, RowOutcome("failed", "Failed: The disk is full."))
    p.end_run()
    p.add_candidate(y)                                                # another 'S1': the started slide keeps 'S1'
    p.apply_output_index(index(["S1.ome.tif"]))
    assert (p.row(x.key).output_stem, p.row(y.key).output_stem, p.rows("note")) == ("S1", "S1__y", [])
    changed = p.apply_header(x.key, header(1, names=("Nucleus",)), None, 64 * GB)
    assert (p.row(x.key).status, p.row(x.key).output_stem) == ("unsupported", "S1__x")
    assert set(changed) == {x.key, "note|unmatched|s1.ome.tif"} and p.refresh() == []


def test_a_verdict_gives_a_slide_back_its_folder_file_at_once():
    """C3-V3 (DS-4): with slide naming, a verdict that found the folder's old file broken keeps it from the slide; a
    later verdict that finds it fine gives it back at once."""
    c = cand("SL01/Scan1", "SL01_Scan1")
    p = make(c, naming="slide")
    p.apply_output_index(index(["Scan1.ome.tif"]))
    p.apply_verification(f"{OUT}/Scan1.ome.tif", "unreadable", "not a TIFF file")
    assert st(p, c.key) == ("new", "new", True, True) and p.row(c.key).existing is None
    assert p.apply_verification(f"{OUT}/Scan1.ome.tif", "valid", "") == [c.key]
    assert st(p, c.key) == ("existing", "existing", False, True) and p.refresh() == []


def test_after_any_change_a_recompute_changes_nothing():
    """C3-V3 (MW-8): headers, ticks and verdicts derive only the rows they concern again. Over random steps with
    colliding names, both namings, old outputs, partials (some busy), verdicts and runs, refresh() must find nothing
    to change."""
    pool = [cand("SL01/Scan1", "SL01_Scan1"), cand("SL02/Scan1", "SL02_Scan1"), cand("old/SL02/Scan1", "SL02_Scan1"),
            cand("A"), cand("x/S5"), cand("y/S5", n=3), cand("exp", "S1_Scan1", multi=True),
            cand("exp2", "S1_Scan1", multi=True), cand("TS-07", "TS_07_Scan1")]
    files = ["Scan1.ome.tif", "SL02_Scan1.ome.tif", "A.ome.tif", "S5.ome.tif", "S1_Scan1.ome.tif", "TS-07.ome.tif"]
    for seed in range(30):
        rnd = random.Random(seed)
        p, outs = make(*pool, naming=rnd.choice(["folder", "slide"])), files
        for step in range(30):
            rid, op = rnd.choice(pool).key, rnd.randrange(5)
            if op == 0:
                h = rnd.choice([header(), header(1, names=("Nucleus",)), None])
                p.apply_header(rid, h, None if h else "Permission denied.", 64 * GB)
            elif op == 1:
                p.toggle([rid])
            elif op == 2:
                outs, parts = rnd.sample(files, 3), rnd.sample(files, 2)
                p.apply_output_index(index(outs, partials=parts, ready=rnd.sample(files, rnd.randint(0, 1)),
                                           busy=frozenset(norm_key(n) for n in parts[:rnd.randint(0, 2)])))
            elif op == 3:
                p.apply_verification(f"{OUT}/{rnd.choice(outs)}",
                                     rnd.choice(["valid", "incomplete", "unreadable", "unchecked"]), "")
            else:
                for _ in range(rnd.randint(0, p.begin_run())):
                    p.finish(p.claim_next(OUT, None).row_id, RowOutcome(rnd.choice(["done", "failed"]), "x"))
                p.end_run()
            assert p.refresh() == [], (seed, step, op)


def test_a_thousand_flat_slides_build_quickly():
    """MW-8/FS-16: 1,000 slide folders in one parent, each candidate and header applied on its own (MainWindow
    batches a tick's events, which is cheaper still): the sibling median is computed once per group and a header
    re-derives only its own row (was ~38 s of CPU)."""
    cands = [cand(f"S{i:04d}_Scan1", n=150 + i % 100) for i in range(1000)]
    start = time.process_time()
    p = Plan(SRC, OUT)
    for c in cands:
        p.add_candidate(c)
    for c in cands:
        p.apply_header(c.key, header(), None, 64 * GB)
    p.issues(env())
    p.totals()
    assert time.process_time() - start < 3.0
    assert p.totals().ticked == 1000 and len({r.output_stem for r in p.rows()}) == 1000


# ---- fix round 2: busy slides, bad first tiles, the partial folder, labels, flags of started slides
ROOT_FS = MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw")


def busy_index(*partials, **kw):
    """An index whose `partials` another stitcher is writing right now."""
    return index(partials=partials, busy=frozenset(norm_key(n) for n in partials), **kw)


def test_a_slide_another_stitcher_is_writing_is_busy():
    """A partial another stitcher is writing right now was 'Unfinished last time — will start again', ticked, while
    the runner skipped the slide. It is busy now: not ticked, not tickable, never planned or counted, also while its
    old file is still there (it is being replaced) and for an excluded kind."""
    a, b, seg = cand("A"), cand("B"), cand("x", "S1_Scan1", suffix="composite_image", kind="other_suffix")
    p = make(a, b, seg)
    live = busy_index("A.ome.tif", "B.ome.tif", p.row(seg.key).output_stem + ".ome.tif", outputs=["B.ome.tif"])
    assert set(p.apply_output_index(live)) == {a.key, b.key, seg.key}
    assert st(p, a.key) == st(p, b.key) == ("new", "busy", False, False)
    assert st(p, seg.key) == ("excluded", "busy", False, False)
    assert [row_text(p.row(k)) for k in (a.key, b.key)] == [
        "Another stitcher is writing it now — Rescan when it's done"] * 2
    assert p.toggle([a.key, b.key, seg.key], True) == p.select("all_new") == p.toggle_section("new") == []
    assert p.pending() == [] and p.totals().ticked == 0 and p.planned_output_keys() == frozenset()
    assert codes(p.issues(env(out_index=live))) == ["none_ticked"] and not p.row(b.key).replaces
    assert set(p.existing_counts().values()) == {0}
    assert p.begin_run() == 0 and p.claim_next(OUT, None) is None


def test_while_every_new_slide_is_busy_start_says_why_and_what_to_do():
    """'Tick at least one slide.' asked for what can't be done while another stitcher writes every new slide; the
    run bar says so now, with Rescan. Slides that can be ticked in Already stitched or Not included don't change
    that; a new slide that can be ticked brings the request back."""
    a, b, old, mono = cand("A"), cand("B"), cand("Old"), cand("Mono")
    seg = cand("x", "S1_Scan1", suffix="composite_image", kind="other_suffix")
    p = make(a, old, mono, seg)
    p.apply_header(mono.key, header(n_channels=1, names=("DAPI",)), None, 64 * GB)    # can't be ticked either
    live = busy_index("A.ome.tif", outputs=["Old.ome.tif"])
    p.apply_output_index(live)
    [issue] = p.issues(env(out_index=live))
    assert (issue.code, issue.severity, issue.action) == ("none_ticked", "block", "rescan")
    assert issue.text == "Nothing to stitch yet — another stitcher is writing A. Rescan when it's done."
    q = make(a, b)
    both = busy_index("A.ome.tif", "B.ome.tif")
    q.apply_output_index(both)
    assert q.issues(env(out_index=both))[0].text == ("Nothing to stitch yet — another stitcher is writing 2 slides "
                                                     "(A, B). Rescan when it's done.")
    few = cand("Few", n=3)                                            # unticked, but it can be ticked
    q.add_candidate(few)
    [issue] = q.issues(env(out_index=both))
    assert (issue.text, issue.action) == ("Tick at least one slide.", None)


@pytest.mark.parametrize("outputs, partials, ready, state", [
    (["A.ome.tif"], [], [], ("existing", "existing_checking", False, True)),   # it has finished
    ([], [], ["A.ome.tif"], ("new", "ready", True, True)),                      # ... but the old file was in use
    ([], ["A.ome.tif"], [], ("new", "unfinished", True, True))])                # it died (stale, or its owner exited)
def test_a_busy_slide_follows_the_other_stitcher(outputs, partials, ready, state):
    """The next index (a rescan, or the one after a run) shows what the other stitcher left. A tick from before the
    slide was busy doesn't come back to redo the other stitcher's file; an untick stays."""
    a, b = cand("A"), cand("B")
    p = make(a, b)
    p.toggle([a.key], True)                                          # the user ticks A and unticks B, then another
    p.toggle([b.key], False)                                         # stitcher starts on both
    p.apply_output_index(busy_index("A.ome.tif", "B.ome.tif"))
    assert st(p, a.key) == st(p, b.key) == ("new", "busy", False, False)
    rescan = make(a, b, carry=p.carry_state())
    for q in (p, rescan):
        q.apply_output_index(index(outputs, partials + ["B.ome.tif"], ready=ready))
        assert st(q, a.key) == state and st(q, b.key) == ("new", "unfinished", False, True)
        assert q.refresh() == []


def test_a_slide_skipped_as_busy_keeps_its_outcome_until_a_rescan():
    """Run statuses stay until a rescan: the runner's own reason stays on the row."""
    a = cand("A")
    p = make(a)
    p.begin_run()
    p.finish(p.claim_next(OUT, None).row_id, RowOutcome("skipped", "Skipped: another stitcher may be writing this "
                                                                   "slide"))
    p.end_run()
    p.apply_output_index(busy_index("A.ome.tif"))
    assert st(p, a.key) == ("new", "skipped", True, True)
    assert row_text(p.row(a.key)) == "Skipped: another stitcher may be writing this slide"


def test_busy_comes_from_the_save_folder(tmp_path, live_process, dead_process):
    """End to end with list_output_dir: a partial whose owner runs on this computer, or that changed in the last
    2 minutes with no owner to ask, is busy; a dead owner's is unfinished at once (CT-9), and so is a stale one; a
    complete one is ready, and once its final is in place the slide is already stitched."""
    out = tmp_path / "out"
    pdir = out / N.PARTIAL_DIR
    pdir.mkdir(parents=True)
    owners = {"Live": live_process, "Dead": dead_process, "Fresh": None}
    for name, owner in owners.items():
        (pdir / f"{name}.ome.tif").write_bytes(b"x")
        if owner:
            (pdir / f"{name}.ome.tif{N.OWNER_SUFFIX}").write_text(json.dumps(
                {"host": preflight.HOST, "boot": preflight._BOOT, "pid": owner[0], "started": owner[1], "token": "t"}))
    p = Plan(SRC, str(out))
    for name in owners:
        p.add_candidate(cand(name))

    def rescan():
        p.apply_output_index(preflight.list_output_dir(str(out), ROOT_FS, p.planned_output_keys()))
        return {r.cand.prefix: r.status for r in p.rows()}
    assert rescan() == {"Live": "busy", "Dead": "unfinished", "Fresh": "busy"}
    old = time.time() - 600
    os.utime(pdir / "Fresh.ome.tif", (old, old))                      # its stitcher stopped 10 minutes ago
    (pdir / f"Live.ome.tif{N.COMPLETE_SUFFIX}").write_text('{"gaps": 0}')   # written; it is about to be moved
    assert rescan() == {"Live": "ready", "Dead": "unfinished", "Fresh": "unfinished"}
    os.replace(pdir / "Live.ome.tif", out / "Live.ome.tif")
    for suffix in (N.OWNER_SUFFIX, N.COMPLETE_SUFFIX):
        os.remove(pdir / f"Live.ome.tif{suffix}")
    assert rescan()["Live"] == "existing_checking"


def test_an_unusable_partial_folder_blocks_start(tmp_path):
    """DS-10: the runner fails every slide of a save folder whose partial folder is a link or a file, so Start is
    blocked, with the reason and the way out (step 2 shows the same sentence)."""
    p = make(cand("A"))
    [issue] = [i for i in p.issues(env(out_index=index(pdir_problem=preflight.PDIR_LINK_TEXT)))
               if i.code == "pdir_problem"]
    assert (issue.severity, issue.action, issue.text) == (
        "block", "browse_output", "Can't save to out: its .stitcher-in-progress folder is a link to another place. "
                                  "Delete the link, or choose another save folder.")
    assert "pdir_problem" not in codes(p.issues(env()))
    out = tmp_path / "Stitched"
    out.mkdir()
    (out / N.PARTIAL_DIR).write_bytes(b"")
    idx = preflight.list_output_dir(str(out), ROOT_FS)
    q = Plan(SRC, str(out))
    q.add_candidate(cand("A"))
    q.apply_output_index(idx)
    assert [i.text for i in q.issues(env(out_dir=str(out), out_index=idx)) if i.severity == "block"] == [
        "Can't save to Stitched: it holds a file named .stitcher-in-progress, where unfinished images belong. "
        "Delete that file, or choose another save folder."]


def test_a_bad_first_tile_says_what_helps_instead_of_try_again():
    """UD4: reading a bad first tile again (no position tags, cut off, not a TIFF) can't help, so its row says what
    is wrong and what does help; an I/O error keeps Try again. Which of the two it is belongs to the row's state: a
    change is reported, and a rebuild keeps it."""
    a = cand("A")
    p = make(a)
    tags = "it has no position tags"
    assert p.apply_header(a.key, None, tags, 64 * GB, retryable=False) == [a.key]
    assert st(p, a.key) == ("new", "header_error", False, True)
    assert row_text(p.row(a.key)) == "Can't use the first tile (it has no position tags). Copy it again, then Rescan"
    assert p.apply_header(a.key, None, tags, 64 * GB) == [a.key]                 # only retryable changed
    assert row_text(p.row(a.key)) == f"Can't read the first tile ({tags}). Right-click → Try again"
    p.apply_header(a.key, None, "the file is cut off.", 64 * GB, retryable=False)
    q = p.rebuild("/data/out2")
    assert not q.row(a.key).header_retryable and row_text(q.row(a.key)) == (
        "Can't use the first tile (the file is cut off). Copy it again, then Rescan")


def test_label_is_cheap_and_follows_renames(monkeypatch):
    """Plan.label names a slide in run-bar texts, log lines and banners, for every event, so it never walks the
    rows: the display name, or the output name while other slides share the display name. Candidate rows are never
    removed from a plan (a rescan builds a new one); note rows that come and go don't count."""
    s1, s2, u = cand("batchA/Slide9", "Slide9_Scan1"), cand("batchB/Slide9", "Slide9_Scan1"), cand("Unique")
    p = make(s1, u)
    assert [p.label(p.row(k)) for k in (s1.key, u.key)] == ["Slide9", "Unique"]
    p.add_candidate(s2)
    assert [p.label(p.row(k)) for k in (s1.key, s2.key, u.key)] == ["Slide9__batchA", "Slide9__batchB", "Unique"]
    assert p.rename_output(s2.key, "Mine") is None
    assert [p.label(p.row(k)) for k in (s1.key, s2.key)] == ["Slide9__batchA", "Mine"]
    p.apply_output_index(index(["Slide9.ome.tif"]))                  # a row comes and goes again (only notes do)
    assert [r.row_id for r in p.rows("note")] == ["note|unmatched|slide9.ome.tif"]
    p.apply_output_index(index())
    assert p.rows("note") == [] and [p.label(p.row(k)) for k in (s1.key, s2.key)] == ["Slide9__batchA", "Mine"]
    q = p.rebuild()
    assert [q.label(q.row(k)) for k in (s1.key, s2.key, u.key)] == ["Slide9__batchA", "Mine", "Unique"]
    calls, real = [], PL.display_name
    monkeypatch.setattr(PL, "display_name", lambda c, full=False: (calls.append(c), real(c, full))[1])
    for r in q.rows():
        q.label(r)
    assert len(calls) == 3                                           # one name per call


def test_a_slide_that_started_keeps_its_renamed_flag():
    """Recomputes during and after a run (the minute timer, the end of the run) dropped the 'renamed' flag of the
    slides that had started, although they keep their names. A name the user gives a waiting slide gets no flag."""
    p = make(cand("batchA/Slide9", "Slide9_Scan1"), cand("batchB/Slide9", "Slide9_Scan1"))
    a, b = [r.row_id for r in p.rows("new")]

    def renamed():
        return [[f.text for f in p.row(k).flags if f.code == "renamed"] for k in (a, b)]
    before = renamed()
    assert before == [["Renamed so nothing is overwritten (…__batchA)"],
                      ["Renamed so nothing is overwritten (…__batchB)"]]
    p.begin_run()
    p.claim_next(OUT, None)
    p.refresh()
    assert renamed() == before
    assert p.rename_output(b, "Mine") is None and renamed() == [before[0], []]
    p.refresh()
    assert renamed() == [before[0], []]
    p.finish(a, RowOutcome("done", "Done"))
    p.end_run()
    assert renamed() == [before[0], []] and p.refresh() == []


def test_case_collision_outputs_never_transfer_on_add_remove_and_rescan(monkeypatch):
    monkeypatch.setattr(N, "_CASE_INSENSITIVE_FS", True)   # the historical Darwin branch
    tiles = [Entry(f"SlideA_[{i},0]_component_data.tif", False, False) for i in range(12)]
    candidates = [N.candidates_from_summary(N.summarize_listing(SRC + "/" + folder + "/SlideA", tiles),
                                            root=SRC, depth=2, dir_mtime=None)[0]
                  for folder in ("Run1", "run1", "RUN1")]
    original = make(*candidates[:2])
    outputs = {r.row_id: r.output_stem + ".ome.tif" for r in original.rows("new")}
    idx = index(list(outputs.values()) + ["SlideA__2.ome.tif", "SlideA__3.ome.tif"])
    for included in (candidates, candidates[1:], candidates[:2]):
        plan = make(*included)
        plan.apply_output_index(idx)
        for row in plan.rows():
            if row.section == "note":
                continue
            if row.row_id in outputs:
                assert row.existing.name == outputs[row.row_id]
            else:
                assert row.existing is None and row.checked
        assert {r.note.names[0] for r in plan.rows("note")} >= {"SlideA__2.ome.tif", "SlideA__3.ome.tif"}


def test_darwin_case_distinct_folders_and_prefixes_are_visible_and_idempotent(monkeypatch):
    monkeypatch.setattr(N, "_CASE_INSENSITIVE_FS", True)
    plan = make()
    entries = [Entry(f"{prefix}_[{i},0]_component_data.tif", False, False)
               for prefix in ("Slide1", "slide1") for i in range(12)]
    for folder in (SRC + "/Batch", SRC + "/batch"):
        found = N.candidates_from_summary(N.summarize_listing(folder, entries), root=SRC, depth=1, dir_mtime=None)
        for candidate in found:
            plan.add_candidate(candidate)
            plan.add_candidate(candidate)
    rows = plan.rows("new")
    assert len(rows) == 4 and len({N.norm_key(row.output_stem) for row in rows}) == 4


def test_approved_case_only_replacement_uses_indexed_basename():
    candidate = cand("SlideA")
    plan = make(candidate)
    plan.apply_output_index(index(["slidea.ome.tif"]))
    plan.toggle([candidate.key], True)
    plan.begin_run()
    job = plan.claim_next(OUT, None)
    assert job.replace and job.final_name == "slidea.ome.tif"


def test_legacy_numbered_output_cannot_be_claimed_by_a_new_same_named_folder():
    first, second, new = cand("BatchA/SlideA"), cand("BatchB/SlideA"), cand("SlideA__2")
    plan = make(first, second, new)
    plan.apply_output_index(index(["SlideA__2.ome.tif"]))
    row = plan.row(new.key)
    assert row.existing is None and row.checked and not row.replaces
    assert any(r.note.names == ("SlideA__2.ome.tif",) for r in plan.rows("note"))
