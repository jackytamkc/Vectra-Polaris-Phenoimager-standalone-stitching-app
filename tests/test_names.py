"""names.py: regexes, classification, listing summaries, candidates, notes, naming, de-duplication, estimates
(FINAL_SPEC §3.1-§3.4, §3.7, §3.9; acceptance B1). Pure: no files are touched."""
from __future__ import annotations

import json
import os

import pytest

from conftest import FIXTURES
from yfs.core import names as N
from yfs.core.model import Candidate, Entry, TileHeader


def entries(*specs):
    """'name' -> file, 'name/' -> folder, 'name@' -> symlink (file-like), 'name/@' -> junction-like folder link."""
    out = []
    for s in specs:
        link = s.endswith("@")
        s = s.rstrip("@")
        out.append(Entry(s.rstrip("/"), s.endswith("/"), link, hidden=s.startswith(".")))
    return out


def fixture_names(fname):
    return [ln.strip() for ln in (FIXTURES / fname).read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.parametrize("fname,prefix", [("names_741122_1_Scan1.txt", "741122_1_Scan1"),
                                          ("names_QA731104-12_Scan1.txt", "QA731104-12_Scan1")])
def test_full_slide_tile_names_parse(fname, prefix):
    names = fixture_names(fname)
    assert len(names) > 200
    for name in names:
        p, x, y, suffix = N.parse_tile_name(name)
        bx, by = name.rsplit("_[", 1)[1].split("]")[0].split(",")
        assert (p, suffix, x, y) == (prefix, "component_data", float(bx), float(by))
        assert N.classify_file(name) == "tile"


def test_edge_names():
    assert N.parse_tile_name("Slide [weird] (1)_[5230,15184]_component_data.tif") == \
        ("Slide [weird] (1)", 5230.0, 15184.0, "component_data")
    assert N.parse_tile_name("TMA_Core[1,2,A]_[100.5,-20.25]_Component_Data.TIFF") == \
        ("TMA_Core[1,2,A]", 100.5, -20.25, "component_data")
    assert N.parse_tile_name("S_[1, 2].tif") == ("S", 1.0, 2.0, "")
    assert N.parse_tile_name("S_[1,2]_binary_seg_maps.tif")[3] == "binary_seg_maps"
    assert N.parse_tile_name("._S_[1,2]_component_data.tif") is None          # AppleDouble
    assert N.parse_tile_name("S_[1,2]_component_data.ome.tif") is None         # an output, not a tile
    assert N.parse_tile_name("S_[1,2]_cell_seg_data.txt") is None


@pytest.mark.parametrize("name,kind", [
    ("._x.tif", "ignored"), (".DS_Store", "ignored"), ("Thumbs.db", "ignored"), ("desktop.ini", "ignored"),
    ("~$notes.docx", "ignored"), ("share (Z) - shortcut.lnk", "lnk"), ("S.ome.tmp.raw", "tmp_raw"),
    ("S.ome.tif", "output"), ("S.OME.TIFF", "output"), ("R1_Scan1_stitch.tif", "wsi"), ("x_stitched.tiff", "wsi"),
    ("S_Scan1.qptiff", "raw"), ("S.im3", "raw"), ("S_[1,2]_component_data.tif", "tile"), ("OverviewBF.tif", "scanner_aux"),
    ("Label.tif", "scanner_aux"), ("SampleMask.tif", "scanner_aux"), ("FocusMap.tif", "scanner_aux"),
    ("MarkupMarks.tif", "scanner_aux"), ("batch_procedure.ifp", "inform_marker"), ("Batch.log", "inform_marker"),
    ("MergeStatus.xml", "inform_marker"), ("project.ifp", "inform_marker"), ("case_001.tif", "tiff_other"),
    ("notes.txt", "other"), ("S_annotations.xml.lock", "other")])
def test_classify_file(name, kind):
    assert N.classify_file(name) == kind


def test_summarize_listing_and_candidates():
    tiles = [f"S1_[{x},{y}]_component_data.tif" for x in (10, 20) for y in (5, 6)]
    seg = [t.replace("component_data", "binary_seg_maps") for t in tiles]
    s = N.summarize_listing("/r/export", entries(*tiles, *seg, *[t.replace("S1", "S2") for t in tiles],
                                                  "batch_procedure.ifp", "x.ome.tif", "y_stitch.tif", "a.tif",
                                                  "notes.txt", "._x.tif", "sub/", ".hidden/", "$RECYCLE.BIN/",
                                                  ".stitcher-in-progress/", "j/@", "loop@", "S9.ome.tmp.raw",
                                                  "z.lnk"))
    assert [(g.prefix, g.suffix, len(g.names)) for g in s.groups] == \
        [("S1", "binary_seg_maps", 4), ("S1", "component_data", 4), ("S2", "component_data", 4)]
    assert list(s.groups[1].names) == sorted(tiles) and s.groups[1].coords[0] == (10.0, 5.0)
    assert s.subdirs == ("sub",) and s.links == 2 and s.inform_export and not s.raw_scan
    assert s.outputs == ("x.ome.tif",) and s.wsi == ("y_stitch.tif",) and s.other_tiffs == 1
    assert s.leftovers == ("S9.ome.tmp.raw", ".stitcher-in-progress") and s.other_files == 1   # ._x.tif: litter
    assert set(s.notable_files) == {"batch_procedure.ifp", "x.ome.tif", "y_stitch.tif", "S9.ome.tmp.raw", "z.lnk"}
    assert N.summarize_listing("/r", entries(".hidden/", "sub/"), show_hidden=True).subdirs == (".hidden", "sub")
    cands = N.candidates_from_summary(s, root="/r", depth=1, dir_mtime=123.0)
    assert [(c.prefix, c.kind, c.suffix) for c in cands] == [("S1", "component", "component_data"),
                                                            ("S1", "other_suffix", "binary_seg_maps"),
                                                            ("S2", "component", "component_data")]
    c = cands[0]
    assert c.multi_slide_folder and c.inform_export and c.rel_folder == "export" and c.folder_name == "export"
    key = "/r/export|s1|component_data" if N._CASE_INSENSITIVE_FS else "/r/export|S1|component_data"
    assert c.key == key and c.depth == 1 and c.dir_mtime == 123.0


def test_system_litter_is_not_counted_as_other_files():
    """BR-22: .DS_Store, Thumbs.db, desktop.ini, AppleDouble and Office lock files are not the folder's files."""
    s = N.summarize_listing("/r/x", entries(".DS_Store", "Thumbs.db", "desktop.ini", "._a.tif", "~$draft.docx",
                                            "notes.txt", ".nfs0001"))
    assert (s.other_files, s.n_entries) == (1, 7)


@pytest.mark.parametrize("hidden", [Entry(".notes", False, False), Entry("notes.txt", False, False, hidden=True)])
def test_hidden_ordinary_files_count_only_when_shown(hidden):
    """BR-22: hidden files are not visible files, but a hidden-only folder is not empty."""
    for shown in (False, True):
        s = N.summarize_listing("/data/HiddenOnly", [hidden], show_hidden=shown)
        assert s.n_entries == 1 and s.other_files == int(shown)


def test_suffix_preference_and_legacy():
    tiles = [f"S_[{i},0]" for i in range(3)]
    only_plain = N.summarize_listing("/r/a", entries(*[t + ".tif" for t in tiles]))
    assert [(c.kind, c.suffix) for c in N.candidates_from_summary(only_plain, root="/r", depth=1, dir_mtime=None)] \
        == [("unsuffixed", "")]
    both = N.summarize_listing("/r/a", entries(*[t + ".tif" for t in tiles], *[t + "_component_data.tif" for t in tiles]))
    assert [c.kind for c in N.candidates_from_summary(both, root="/r", depth=1, dir_mtime=None)] == ["component"]
    legacy = N.summarize_listing("/r/WSIs", entries("a.tif", "b.tiff", "notes.txt"))
    [c] = N.candidates_from_summary(legacy, root="/r", depth=1, dir_mtime=None)
    key = "/r/wsis|*|legacy" if N._CASE_INSENSITIVE_FS else "/r/WSIs|*|legacy"
    assert (c.kind, c.key, c.prefix, c.suffix, c.tile_names) == ("legacy_all", key, "", "legacy", ())
    raw = N.summarize_listing("/r/Scan1", entries("S.qptiff", "OverviewBF.tif", "Label.tif"))
    assert raw.raw_scan and N.candidates_from_summary(raw, root="/r", depth=1, dir_mtime=None) == []
    root = N.summarize_listing("/r", entries(*[t + "_component_data.tif" for t in tiles]))
    [c] = N.candidates_from_summary(root, root="/r", depth=0, dir_mtime=None)
    assert c.rel_folder == "." and c.folder_name == "r" and not c.multi_slide_folder


def test_notes_from_summary():
    def kinds(*specs, truncated=False):
        s = N.summarize_listing("/r/x", entries(*specs), truncated=truncated)
        return {n.kind: n for n in N.notes_from_summary(s)}

    k = kinds("a.ome.tif", "b.ome.tif", "w_stitch.tif", "s.qptiff", "z.lnk", "loop@", ".stitcher-in-progress/")
    assert k["outputs"].count == 2 and k["outputs"].names == ("a.ome.tif", "b.ome.tif")
    assert set(k) == {"outputs", "wsi", "raw_scan", "lnk", "symlink_skipped", "leftovers"}
    assert set(kinds()) == {"empty"}
    assert set(kinds("notes.txt", "data.csv")) == {"no_images"}
    assert set(kinds("notes.txt", "sub/")) == set()
    assert set(kinds(*[f"f{i}.txt" for i in range(5)], truncated=True)) == {"large_dir"}
    assert "raw_scan" not in kinds("s.qptiff", "S_[1,2]_component_data.tif")


def _fx_entries(fx):
    return [Entry(e["name"], e["dir"], False, e.get("size"), e.get("mtime"), e["name"].startswith("."))
            for e in fx["entries"]]


def _probe_entries(probe):
    """Rebuild a child folder's listing from the name-only probe facts of the fx_*.json fixtures."""
    out = [Entry(f"{p}_[{i * 900},{i * 700}]_{s}.tif", False, False) for p, s, n in probe["groups"] for i in range(n)]
    out += [Entry(f"d{i}", True, False) for i in range(probe["n_dirs"])]
    out += [Entry(name, False, False) for name in probe["others"]]
    out += [Entry(f"f{i}.dat", False, False) for i in range(probe["n_files"] - len(out) + probe["n_dirs"])]
    return out


def test_fx_disk_listing_summarises_like_the_spec():
    fx = json.loads((FIXTURES / "fx_disk.json").read_text(encoding="utf-8"))
    root = fx["root"]
    s = N.summarize_listing(root, _fx_entries(fx))
    assert len(s.outputs) == 2 and not s.groups                       # never a slide itself (§3.6)
    assert "$RECYCLE.BIN" not in s.subdirs and "System Volume Information" not in s.subdirs
    assert [n.kind for n in N.notes_from_summary(s)] == ["outputs"]
    slides = []
    for e in fx["entries"]:
        if e["dir"] and e["name"] in s.subdirs:
            child = N.summarize_listing(f"{root}/{e['name']}", _probe_entries(e["probe"]))
            slides += [c for c in N.candidates_from_summary(child, root=root, depth=1, dir_mtime=None)
                       if c.kind == "component"]
    assert [(c.folder_name, len(c.tile_names)) for c in slides] == [("741122_1_Scan1", 370), ("741122_2_Scan1", 365)]
    stems = [N.output_stem(c, "folder") + N.OME_SUFFIX for c in slides]
    assert {N.norm_key(x) for x in stems} <= {N.norm_key(x) for x in s.outputs}   # both already stitched


def test_fx_share_listing_summarises_like_the_spec():
    fx = json.loads((FIXTURES / "fx_share_user.json").read_text(encoding="utf-8"))
    s = N.summarize_listing(fx["root"], _fx_entries(fx))
    notes = {n.kind: n for n in N.notes_from_summary(s)}
    assert set(notes) == {"lnk"} and notes["lnk"].names[0].endswith(".lnk")
    assert not s.groups and s.other_tiffs == 0 and "seq_runs" in s.subdirs
    slides, export = [], None
    for e in fx["entries"]:
        if e["dir"]:
            child = N.summarize_listing(f"{fx['root']}/{e['name']}", _probe_entries(e["probe"]))
            slides += N.candidates_from_summary(child, root=fx["root"], depth=1, dir_mtime=None)
            export = child if e["name"] == "informexport" else export
    assert sorted((c.folder_name, len(c.tile_names)) for c in slides) == [
        ("kb07a1_Scan1", 173), ("kb07a2_Scan1", 64), ("kb07anc_Scan1", 178), ("kb07b1_Scan1", 237),
        ("kb07b2_Scan1", 204), ("kb07bnc_Scan1", 246)]
    assert export.inform_export and not export.groups


def test_natural_key_and_paths():
    names = ["QA731104-10", "qa731104-2", "QA731104-1", "a", "B"]
    assert sorted(names, key=N.natural_key) == ["a", "B", "QA731104-1", "qa731104-2", "QA731104-10"]
    assert sorted(["a1", "a"], key=N.natural_key) == ["a", "a1"]             # mixed shapes never raise
    assert N.norm_key("Ünï.OME.tif") == N.norm_key("ünï.ome.TIF")
    assert N.same_path("/a/b/", "/a/./b") and not N.same_path("/a/b", None)
    assert N.rel_folder("/r/a/b", "/r") == "a/b" and N.rel_folder("/r", "/r") == "."


@pytest.mark.parametrize("raw,clean", [
    ("Slide 1", "Slide 1"), ('a<b>c:d"e/f\\g|h?i*j', "a_b_c_d_e_f_g_h_i_j"), ("tab\there", "tab_here"),
    ("name. . ", "name"), ("CON", "_CON"), ("com1.backup", "_com1.backup"), ("lpt9", "_lpt9"), ("com10", "com10"),
    ("", "slide"), ("...", "slide")])
def test_sanitize_stem(raw, clean):
    assert N.sanitize_stem(raw) == clean


def test_sanitize_long_stem_is_hashed():
    s = N.sanitize_stem("x" * 300)
    assert len(s) == 179 and s[:172] == "x" * 172 and s[172] == "~" and len(s[173:]) == 6
    assert N.sanitize_stem("x" * 300) == s and N.sanitize_stem("x" * 299) != s


def test_sanitize_stem_replaces_undecodable_bytes():
    """FS-14: Linux names with a non-UTF-8 byte arrive surrogate-escaped; hashing a long one used to raise."""
    name = os.fsdecode(b"S" * 190 + b"\xe9")
    assert name.endswith("\udce9")
    s = N.sanitize_stem(name)
    assert len(s) == 179 and s.encode("utf-8") and N.sanitize_stem(name) == s
    assert N.sanitize_stem(os.fsdecode(b"Caf\xe9_Scan1")) == "Caf__Scan1"     # also portable to NTFS/SMB/APFS


@pytest.mark.parametrize("insensitive", [False, True])
def test_candidate_keys_follow_the_filesystem(monkeypatch, insensitive):
    """FS-12/PL-5: folders or prefixes that differ only in case are two slides where the filesystem says so."""
    monkeypatch.setattr(N, "_CASE_INSENSITIVE_FS", insensitive)
    tiles = lambda p: [f"{p}_[{i},0]_component_data.tif" for i in range(3)]          # noqa: E731
    keys = {c.key for folder in ("/r/Run1/S1_Scan1", "/r/run1/S1_Scan1")
            for c in N.candidates_from_summary(N.summarize_listing(folder, entries(*tiles("S1_Scan1"))),
                                               root="/r", depth=2, dir_mtime=None)}
    export = N.summarize_listing("/r/batch", entries(*tiles("S2_Scan1"), *tiles("s2_scan1")))
    assert len(keys) == 2 and len(export.groups) == 2
    assert len({c.key for c in N.candidates_from_summary(export, root="/r", depth=1, dir_mtime=None)}) == 2
    assert N.same_path("/r/Run1", "/r/run1/") == insensitive and N.same_path("/r/a/../b", "/r/b")
    assert N.path_key("/r/Run1/") == ("/r/run1" if insensitive else "/r/Run1")


def _cand(folder, prefix, kind="component", multi=False, suffix="component_data", n=12):
    tiles = tuple(f"{prefix}_[{i},0]_{suffix}.tif" for i in range(n))
    return Candidate(f"{N.norm_key(folder)}|{prefix.casefold()}|{suffix}", folder, folder.lstrip("/"),
                     os.path.basename(folder), prefix, suffix, kind, tiles, tuple((float(i), 0.0) for i in range(n)),
                     multi, False, 1)


def test_output_stem_and_engine_image_name():
    single = _cand("/r/TS4410-07", "TS4410_07_Scan1")
    assert N.output_stem(single, "folder") == "TS4410-07" and N.output_stem(single, "slide") == "TS4410_07_Scan1"
    assert N.engine_image_name(single) is None
    multi = _cand("/r/inform_export", "S1_Scan1", multi=True)
    assert N.output_stem(multi, "folder") == "S1_Scan1" and N.engine_image_name(multi) == "S1_Scan1"
    seg = _cand("/r/inform_export", "S1_Scan1", "other_suffix", True, "binary_seg_maps")
    assert N.output_stem(seg, "folder") == "S1_Scan1_binary_seg_maps"
    assert N.engine_image_name(seg) == "S1_Scan1_binary_seg_maps"
    legacy = Candidate("k", "/r/WSIs", "WSIs", "WSIs", "", "legacy", "legacy_all", (), (), False, False, 1)
    assert N.output_stem(legacy, "slide") == "WSIs" and N.engine_image_name(legacy) is None


def test_dedupe_stems():
    """§3.7 with D1: no member of a collision group keeps the bare name (DS-5)."""
    out = N.dedupe_stems([("a", "Slide9", "/s/duplicates/batchA/Slide9", "Slide9_Scan1"),
                          ("b", "Slide9", "/s/duplicates/batchB/Slide9", "Slide9_Scan1"),
                          ("c", "S1_Scan1", "/s/exportB", "S1_Scan1"), ("d", "s1_scan1", "/s/exportA", "s1_scan1"),
                          ("e", "Solo", "/s/x/Solo", "Solo_Scan1")])
    assert out == {"a": ("Slide9__batchA", "suffix"), "b": ("Slide9__batchB", "suffix"),
                   "c": ("S1_Scan1__exportB", "suffix"), "d": ("s1_scan1__exportA", "suffix"), "e": ("Solo", "")}
    # the common <SlideID>/Scan1 layout: every member is named after its tiles, the first one included
    ids = [(f"r{i}", "Scan1", f"/p/SL0{i}/Scan1", f"SL0{i}_Scan1") for i in (2, 3)]
    assert N.dedupe_stems(ids) == {"r2": ("SL02_Scan1", "tiles"), "r3": ("SL03_Scan1", "tiles")}
    # ... and a slide exported later (sorting first) changes nobody else's name
    more = N.dedupe_stems([("r1", "Scan1", "/p/SL01/Scan1", "SL01_Scan1")] + ids)
    assert more == {"r1": ("SL01_Scan1", "tiles"), "r2": ("SL02_Scan1", "tiles"), "r3": ("SL03_Scan1", "tiles")}
    # a copy of SL02 renames only itself and its twin
    dup = N.dedupe_stems(ids + [("c2", "Scan1", "/p/old/SL02/Scan1", "SL02_Scan1")])
    assert dup == {"r2": ("Scan1__SL02", "suffix"), "c2": ("Scan1__old", "suffix"), "r3": ("SL03_Scan1", "tiles")}
    # a tile name that is another row's own name is not used
    taken = N.dedupe_stems(ids + [("x", "SL03_Scan1", "/q/SL03_Scan1", "")])
    assert taken == {"r2": ("SL02_Scan1", "tiles"), "r3": ("Scan1__SL03", "suffix"), "x": ("SL03_Scan1", "")}
    # Case-only sources receive stable identity suffixes; none keeps the ambiguous bare name.
    same_case = N.dedupe_stems([("u", "S", "/p/Run1/S", ""), ("l", "S", "/p/run1/S", "")])
    assert len({stem for stem, _ in same_case.values()}) == 2
    assert all(stem.startswith("S__") and len(stem) == 19 and how == "suffix" for stem, how in same_case.values())


def test_dedupe_stems_with_fixed_names():
    """Typed names and the names of started slides don't change; others step around them."""
    clash = N.dedupe_stems([("a", "S", "/p/S", ""), ("b", "S", "/p/S", ""), ("c", "S__x", "/q/S__x", ""),
                            ("t", "T", "/q/T", "")], fixed={"t": "S__x"})
    assert clash["t"] == ("S__x", "") and len({stem for stem, _ in clash.values()}) == 4
    assert all(clash[k][1] == "suffix" for k in ("a", "b", "c"))
    # a slide that is running keeps its name; its waiting twin keeps the folder suffix it had before the run
    running = N.dedupe_stems([("a", "Slide9", "/s/batchA/Slide9", "Slide9_Scan1"),
                              ("b", "Slide9", "/s/batchB/Slide9", "Slide9_Scan1")], fixed={"a": "Slide9__batchA"})
    assert running == {"a": ("Slide9__batchA", ""), "b": ("Slide9__batchB", "suffix")}
    typed = N.dedupe_stems([("a", "Scan1", "/p/A/Scan1", "A_Scan1"), ("b", "Scan1", "/p/B/Scan1", "B_Scan1")],
                           fixed={"a": "Mine"})
    assert typed == {"a": ("Mine", "tiles"), "b": ("B_Scan1", "tiles")}       # still a group: 'tiles' -> Image@Name


def test_tile_name_and_long_dedupe_suffixes():
    seg = _cand("/r/x", "S1_Scan1", "other_suffix", False, "binary_seg_maps")
    assert N.tile_name(seg) == "S1_Scan1_binary_seg_maps" and N.engine_image_name(seg) is None
    legacy = Candidate("k", "/r/WSIs", "WSIs", "WSIs", "", "legacy", "legacy_all", (), (), True, False, 1)
    assert N.tile_name(legacy) == "" and N.engine_image_name(legacy) is None
    out = N.dedupe_stems([("a", "S", "/p/" + "A" * 200 + "/S", ""), ("b", "S", "/p/" + "B" * 200 + "/S", "")])
    assert all(len(stem) == 179 and how == "suffix" for stem, how in out.values())


def test_estimate_on_a_full_slide_listing():
    s = N.summarize_listing("/data/741122_1_Scan1", [Entry(n, False, False)
                                                     for n in fixture_names("names_741122_1_Scan1.txt")])
    [c] = N.candidates_from_summary(s, root="/data", depth=1, dir_mtime=None)
    um = 10000 / 20065.6
    h = TileHeader("f", (5, 1396, 1860), "float32", 5, 1396, 1860, um, um, True, ("DAPI",) * 5, "duplicate",
                   (None,) * 5, 5 * 1396 * 1860 * 4)
    e = N.estimate(c, h, 64 * 1024 ** 3)
    assert abs(e.canvas_w - 29231) <= 1860 and abs(e.canvas_h - 35598) <= 1396
    assert e.itemsize == 4 and e.n_channels == 5 and e.memory_mode == "ram"
    assert 19.0 < e.raw_bytes / 1024 ** 3 < 19.8 and e.out_typical == int(e.raw_bytes * N.OUT_TYPICAL_FACTOR)
    assert 0.5 < e.coverage <= 1.0
    assert N.estimate(c, h, 8 * 1024 ** 3).memory_mode == "disk"


def test_pyramid_levels_and_human_text():
    assert N.pyramid_levels(35598, 29231) == 8 and N.pyramid_levels(256, 256) == 0 and N.pyramid_levels(257, 3) == 1
    assert N.human_bytes(18960143992) == "17.7 GB" and N.human_bytes(10 * 1024 ** 3, approx=True) == "~10.0 GB"
    assert N.human_bytes(int(1.3 * 1024 ** 4)) == "1.3 TB" and N.human_bytes(512) == "512 B"
    assert N.human_bytes(600 * 1024 ** 2) == "600 MB" and N.human_bytes(None) == ""
    assert [N.human_duration(s) for s in (45, 420, 4320, 3599, 7200)] == ["45 s", "7 min", "1 h 12 min", "1 h", "2 h"]


@pytest.mark.parametrize("text", ["界" * 80, "𐐀" * 63, "a" * 179 + "界", "界" * 59 + "𐐀"])
def test_unicode_component_budget_includes_every_generated_suffix(tmp_path, text):
    stem = N.sanitize_stem(text)
    items = [("/Run1/" + text, stem, "/Run1/" + text, ""),
             ("/run1/" + text, stem, "/run1/" + text, "")]
    deduped = N.dedupe_stems(items)
    assert len({N.norm_key(name) for name, _ in deduped.values()}) == 2
    assert N.sanitize_stem(text) == stem
    for name in [stem] + [name for name, _ in deduped.values()]:
        assert len(name.encode("utf-8")) <= N.STEM_BYTES
        for suffix in (".ome.tif", ".ome.tif.owner", ".ome.tif.complete", ".ome.tif.stale-12345678", ".ome.tmp.raw"):
            path = tmp_path / (name + suffix)
            path.write_bytes(b"fits")
            assert path.read_bytes() == b"fits"


def test_collision_fallbacks_never_transfer_between_sources():
    items = [(name, "SlideA", "/root/" + name + "/SlideA", "SlideA") for name in ("Run1", "run1", "RUN1")]
    initial = N.dedupe_stems(items[:2])
    added = N.dedupe_stems(items)
    removed = N.dedupe_stems(items[1:])
    assert all(added[key] == value for key, value in initial.items())
    assert all(added[key] == value for key, value in removed.items())
    assert len({N.norm_key(stem) for stem, _ in added.values()}) == 3
