"""The identity gate (FINAL_SPEC §7.3, items 1-12): the v2 engine writes the same bytes as v1.

v1 is the verbatim copy in tests/fixtures/legacy_engine_v1.py (``legacy_engine`` fixture); v2 is the
repository's ``stitcher_RAM.py``. Both run with NUM_THREADS = 1, because v1 pastes overlapping tiles in a
timing-dependent order when it uses many threads (§5.7, "why identical" item 6). Item 7 is the exception:
it uses os.cpu_count() threads on tiles that do not overlap, where the result is deterministic.
Outputs are compared by SHA-256 of the files written to tmp_path.
"""
from __future__ import annotations

import gc
import hashlib
import os
import re
import shutil
import struct
import sys
import threading
import types
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import tifffile

import stitcher_RAM
from tests.conftest import FIXTURES, load_module, real_subset   # not "conftest": tests/gui/ has its own

BASELINE_A1_MD5 = "9e315913c16233a404f24579f8e95336"     # v1 on synth ProjectA/SlideA1_Scan1 (§7.3 item 1)

# Item 11: v1 + tifffile 2026.3.3 (numpy 2.4, imagecodecs 2026.3.6) on the 3-channel slide built below.
# If this fails after a tifffile/imagecodecs upgrade, the app would now write different bytes than v1 did.
# Review the tifffile release notes (photometric defaults for (H, W, 3) float data are announced to change),
# decide with the author whether the new output is acceptable, and only then update this value.
GOLDEN_3CH_MD5 = "067b20b2fecb5e0e45df0ba92b2886f8"

synth = load_module("yfs_test_make_synth_identity", FIXTURES / "make_synth.py")


@pytest.fixture
def engines(legacy_engine, monkeypatch):
    monkeypatch.setattr(legacy_engine, "NUM_THREADS", 1)
    monkeypatch.setattr(stitcher_RAM, "NUM_THREADS", 1)
    return legacy_engine, stitcher_RAM


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def stitch(mod, folder, out, mode="ram", ask=None, progress=None, **kw):
    """Run one engine end to end, as v1's run_process does. Returns (engine, write_to_disk result)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eng = mod.StitchingEngine(str(folder), str(out), progress, ask, **kw)
    assert eng.scan_metadata(), f"scan_metadata failed for {folder}"
    canvas = eng.stitch(use_ram=(mode == "ram"))
    assert canvas is not None, "stitch failed"
    result = eng.write_to_disk(canvas, mode == "ram")
    del canvas
    return eng, result


def levels(path):
    """Pixel arrays of level 0 and every pyramid level (SubIFDs)."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        return [page.asarray()] + [p.asarray() for p in (page.pages or [])]


def ome_xml(path) -> str:
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].description


def component_names(folder) -> list:
    return sorted(n for n in os.listdir(folder) if n.endswith("_component_data.tif") and not n.startswith("._"))


@pytest.fixture
def a1(synth_tree) -> Path:
    return synth_tree / "ProjectA" / "SlideA1_Scan1"


# ---------------------------------------------------------------------------------------------------- item 1
@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_01_legacy_mode_identical(engines, a1, tmp_path, mode):
    v1, v2 = engines
    e1, _ = stitch(v1, a1, tmp_path / "v1" / "SlideA1_Scan1.ome.tif", mode)
    e2, ok = stitch(v2, a1, tmp_path / "v2" / "SlideA1_Scan1.ome.tif", mode)
    assert md5(tmp_path / "v1" / "SlideA1_Scan1.ome.tif") == BASELINE_A1_MD5
    assert sha256(tmp_path / "v2" / "SlideA1_Scan1.ome.tif") == sha256(tmp_path / "v1" / "SlideA1_Scan1.ome.tif")
    assert ok is True and e2.failed_tiles == 0 and e2.last_error is None and e2.cancelled is False
    assert not e1.temp_path.exists() and not e2.temp_path.exists()


# ---------------------------------------------------------------------------------------------------- item 2
@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_02_tile_files_hidden_partial_then_replace(engines, a1, tmp_path, mode):
    v1, v2 = engines
    final_v1 = tmp_path / "v1" / "SlideA1_Scan1.ome.tif"
    stitch(v1, a1, final_v1, mode)
    out_dir = tmp_path / "v2"
    partial = out_dir / ".stitcher-in-progress" / "SlideA1_Scan1.ome.tif"
    kw = {"tile_files": component_names(a1)}
    seen_scratch = []
    if mode == "disk":                                   # also: temp_dir moves only the memmap backing file
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        kw = {"tile_files": [a1 / n for n in component_names(a1)], "temp_dir": str(scratch)}

        def progress(cur, tot, phase):
            if phase == "Stitching":
                seen_scratch.append((scratch / "SlideA1_Scan1.ome.tmp.raw").exists())
        kw["progress"] = progress
    e2, ok = stitch(v2, a1, partial, mode, **kw)
    assert ok is True
    os.replace(partial, out_dir / "SlideA1_Scan1.ome.tif")
    assert sha256(out_dir / "SlideA1_Scan1.ome.tif") == sha256(final_v1)
    if mode == "disk":
        assert e2.temp_path == tmp_path / "scratch" / "SlideA1_Scan1.ome.tmp.raw"
        assert seen_scratch and all(seen_scratch) and not e2.temp_path.exists()
        assert not list(partial.parent.glob("*.tmp.raw"))


# ---------------------------------------------------------------------------------------------------- item 3
def test_03_decoys_ignored_with_tile_files(engines, a1, tmp_path):
    v1, v2 = engines
    dec = tmp_path / "decoy" / "SlideA1_Scan1"
    shutil.copytree(a1, dec)
    first = component_names(a1)[0]
    (dec / ("._" + first)).write_bytes(b"\x00\x05\x16\x07" + bytes(4092))        # macOS AppleDouble file
    synth.write_tile(str(dec / first.replace("component_data", "binary_seg_maps")), 10000, 30000,
                     channels=["Nucleus"])                                       # 1-channel product, same place
    synth.plain_tif(str(dec / "SlideA1_Scan1.ome.tif"))                           # an earlier output
    stitch(v1, a1, tmp_path / "v1" / "SlideA1_Scan1.ome.tif")                    # v1 on a clean copy
    _, ok = stitch(v2, dec, tmp_path / "v2" / ".stitcher-in-progress" / "SlideA1_Scan1.ome.tif",
                   tile_files=component_names(dec))
    assert ok is True
    assert sha256(tmp_path / "v2" / ".stitcher-in-progress" / "SlideA1_Scan1.ome.tif") == \
        sha256(tmp_path / "v1" / "SlideA1_Scan1.ome.tif")
    # Why tile_files matters: v1 lists the folder itself, starts with the AppleDouble file and gives up.
    assert not v1.StitchingEngine(str(dec), str(tmp_path / "v1dec.ome.tif")).scan_metadata()


# ---------------------------------------------------------------------------------------------------- item 4
def test_04_one_slide_of_a_multi_slide_folder(engines, synth_tree, tmp_path):
    v1, v2 = engines
    inf = synth_tree / "inform_export"
    s1 = [n for n in component_names(inf) if n.startswith("S1_Scan1_")]
    assert len(s1) == 4
    only = tmp_path / "s1only" / "inform_export"                                 # same folder name as v2's input
    only.mkdir(parents=True)
    for n in s1:
        shutil.copy(inf / n, only / n)
    stitch(v1, only, tmp_path / "v1" / "S1_Scan1.ome.tif")
    stitch(v2, inf, tmp_path / "v2" / "S1_Scan1.ome.tif", tile_files=s1)
    assert sha256(tmp_path / "v2" / "S1_Scan1.ome.tif") == sha256(tmp_path / "v1" / "S1_Scan1.ome.tif")

    named, ok = stitch(v2, inf, tmp_path / "named" / "S1_Scan1.ome.tif", tile_files=s1, image_name="S1_Scan1")
    assert ok is True and named.image_name == "S1_Scan1"
    xml = ome_xml(tmp_path / "named" / "S1_Scan1.ome.tif")
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    assert ET.fromstring(xml).find("ome:Image", ns).get("Name") == "S1_Scan1"
    assert xml == ome_xml(tmp_path / "v1" / "S1_Scan1.ome.tif").replace('Name="inform_export"', 'Name="S1_Scan1"')
    for a, b in zip(levels(tmp_path / "named" / "S1_Scan1.ome.tif"), levels(tmp_path / "v1" / "S1_Scan1.ome.tif")):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------------- item 5
@pytest.mark.parametrize("names", [["DAPI", "Opal 520", "Opal 570", "Opal 690", "AF"], None], ids=["names", "generic"])
def test_05_suspicious_channels(engines, synth_tree, tmp_path, names):
    v1, v2 = engines
    src = synth_tree / "suspicious" / "SusSlide_Scan1"
    calls = {"v1": [], "v2": []}

    def resolver(tag):
        return lambda image_name, current: calls[tag].append((image_name, list(current))) or names
    stitch(v1, src, tmp_path / "v1" / "SusSlide_Scan1.ome.tif", ask=resolver("v1"))
    e2, _ = stitch(v2, src, tmp_path / "v2" / "SusSlide_Scan1.ome.tif", ask=resolver("v2"),
                   tile_files=component_names(src))
    assert sha256(tmp_path / "v2" / "SusSlide_Scan1.ome.tif") == sha256(tmp_path / "v1" / "SusSlide_Scan1.ome.tif")
    assert calls["v1"] == calls["v2"] == [("SusSlide_Scan1", ["DAPI"] * 5)]
    assert e2.channel_names == (names or [f"Channel {i}" for i in range(5)])


# ---------------------------------------------------------------------------------------------------- item 6
@pytest.mark.real
@pytest.mark.slow
def test_06_real_subset(engines, tmp_path):
    src = real_subset()
    if src is None:
        pytest.skip("set YFS_REAL_SUBSET to a folder holding one slide's real *_component_data.tif tiles")
    v1, v2 = engines
    tiffs = sorted(n for n in os.listdir(src) if n.lower().endswith((".tif", ".tiff")))
    tiles = component_names(src)
    if not tiles or tiles != tiffs:
        pytest.skip("YFS_REAL_SUBSET must hold only the component_data tiles of one slide")
    name = src.name + ".ome.tif"
    try:
        stitch(v1, src, tmp_path / "v1" / name)
        _, ok = stitch(v2, src, tmp_path / "v2" / ".stitcher-in-progress" / name, tile_files=tiles)
        assert ok is True
        os.replace(tmp_path / "v2" / ".stitcher-in-progress" / name, tmp_path / "v2" / name)
        assert sha256(tmp_path / "v2" / name) == sha256(tmp_path / "v1" / name)
    finally:                                                                       # ~420 MB each
        shutil.rmtree(tmp_path / "v1", ignore_errors=True)
        shutil.rmtree(tmp_path / "v2", ignore_errors=True)


# ---------------------------------------------------------------------------------------------------- item 7
def test_07_many_threads_without_overlap(legacy_engine, monkeypatch, tmp_path):
    folder = tmp_path / "Grid_Scan1"
    folder.mkdir()
    w, h = 64, 48
    for j in range(5):
        for i in range(6):                                                         # edge to edge: no overlap
            synth.write_tile(str(folder / f"Grid_Scan1_[{i * 100},{j * 100}]_component_data.tif"),
                             10000 + i * w, 30000 + j * h, w=w, h=h)
    outs = {}
    for label, threads in (("v1", os.cpu_count() or 16), ("v2", os.cpu_count() or 16), ("v2_one", 1)):
        mod = legacy_engine if label == "v1" else stitcher_RAM
        monkeypatch.setattr(mod, "NUM_THREADS", threads)
        e, _ = stitch(mod, folder, tmp_path / label / "Grid_Scan1.ome.tif")
        assert len(e.tiles) == 30
        outs[label] = sha256(tmp_path / label / "Grid_Scan1.ome.tif")
    assert outs["v1"] == outs["v2"] == outs["v2_one"]


# ---------------------------------------------------------------------------------------------------- item 8
def test_08_xml_failure_path(engines, a1, tmp_path):
    v1, v2 = engines
    outs = {"v1": tmp_path / "v1" / "SlideA1_Scan1.ome.tif",
            "v2": tmp_path / "v2" / ".stitcher-in-progress" / "SlideA1_Scan1.ome.tif"}
    normalised = {}
    for label, mod in (("v1", v1), ("v2", v2)):
        outs[label].parent.mkdir(parents=True)
        e = mod.StitchingEngine(str(a1), str(outs[label]))
        e.generate_visio_xml = lambda *a, **k: None
        assert e.scan_metadata()
        e.write_to_disk(e.stitch(True), True)
        with tifffile.TiffFile(outs[label]) as tif:
            assert tif.is_ome, "tifffile must still write its own OME-XML (the partial keeps the final name)"
        normalised[label] = re.sub(rb'UUID="urn:uuid:[0-9a-fA-F-]+"', b'UUID=""', outs[label].read_bytes())
    assert hashlib.sha256(normalised["v1"]).digest() == hashlib.sha256(normalised["v2"]).digest()


# ---------------------------------------------------------------------------------------------------- item 9
def _res_tile(path, unit, res, drop_resolution=False):
    """3 channels of 32x48 float32 with position tags; optionally without tags 282/283 (tifffile always writes
    them, so their codes are patched to private ones afterwards)."""
    positions = [(286, 5, 1, (100, 20000), True), (287, 5, 1, (200, 20000), True)]
    with tifffile.TiffWriter(path) as tif:
        for c, name in enumerate(["DAPI", "Opal 520", "Opal 570"]):
            tif.write(np.full((32, 48), c, np.float32), photometric="minisblack", resolution=(res, res),
                      resolutionunit=unit, description=synth.pe_xml(name), extratags=positions, metadata=None)
    if drop_resolution:
        with tifffile.TiffFile(path) as tif:
            spots = [(t.offset, 65000 + t.code) for p in tif.pages for t in p.tags.values() if t.code in (282, 283)]
            order = tif.byteorder
        with open(path, "r+b") as f:
            for offset, code in spots:
                f.seek(offset)
                f.write(struct.pack(order + "H", code))


@pytest.mark.parametrize("variant", ["inch", "cm", "none_unit", "no_resolution"])
def test_09_scan_metadata_attribute_parity(engines, tmp_path, variant):
    v1, v2 = engines
    unit, res = {"inch": ("INCH", 50800), "cm": ("CENTIMETER", 50800), "none_unit": ("NONE", 50800),
                 "no_resolution": ("CENTIMETER", 20000)}[variant]
    folder = tmp_path / variant
    folder.mkdir()
    for k in range(2):
        _res_tile(str(folder / f"T_[{k},0]_component_data.tif"), unit, res, variant == "no_resolution")
    seen = []
    for mod in (v1, v2):
        e = mod.StitchingEngine(str(folder), str(tmp_path / "unused.ome.tif"))
        ok = e.scan_metadata()
        seen.append(repr((ok, e.n_channels, e.tile_h, e.tile_w, e.dtype, e.channel_names, e.tiff_res_val,
                          e.tiff_res_unit, e.phys_size_x_um, e.phys_size_y_um, e.tiles, e.canvas_shape,
                          e.min_x, e.min_y)))
    assert seen[0] == seen[1]
    with tifffile.TiffFile(str(folder / "T_[0,0]_component_data.tif")) as tif:
        res_tags = stitcher_RAM.StitchingEngine.resolution_from_tags(tif.pages[0].tags)
    assert (res_tags is None) == (variant == "no_resolution")


def test_09b_static_helpers_match_v1_rules():
    E = stitcher_RAM.StitchingEngine
    assert E.normalize_shape((32, 48)) == (1, 32, 48)
    assert E.normalize_shape((5, 256, 320)) == (5, 256, 320)
    assert E.normalize_shape((256, 320, 5)) == (5, 256, 320)
    assert E.normalize_shape((2, 3, 4, 5)) is None
    assert E.assess_channels([], 5) == (None, "missing")
    assert E.assess_channels(["a", "b"], 3) == (None, "count")
    assert E.assess_channels(["DAPI", "DAPI"], 2) == (None, "duplicate")
    assert E.assess_channels(("a", "b"), 2) == (["a", "b"], None)
    assert E.get_rational(types.SimpleNamespace(value=(1, 0))) == 0


# --------------------------------------------------------------------------------------------------- item 10
def test_10_write_failure_record_and_no_pinned_canvas(engines, a1, tmp_path, monkeypatch):
    _, v2 = engines

    class NoSpaceAtFirstPyramidLevel(tifffile.TiffWriter):
        def write(self, data, *args, **kwargs):
            if kwargs.get("subifds") is None:            # level 0 is written with subifds=n_levels
                raise OSError(28, "No space left on device")
            return super().write(data, *args, **kwargs)
    monkeypatch.setattr(tifffile, "TiffWriter", NoSpaceAtFirstPyramidLevel)
    partial = tmp_path / ".stitcher-in-progress" / "SlideA1_Scan1.ome.tif"
    e = v2.StitchingEngine(str(a1), str(partial))
    assert e.scan_metadata()
    canvas = e.stitch(use_ram=True)
    ref = weakref.ref(canvas)
    assert e.write_to_disk(canvas, True) is False
    del canvas
    gc.collect()
    assert ref() is None, "the failed write must not keep the canvas alive"
    assert not partial.exists()
    assert e.last_error == {"type": "OSError", "message": "[Errno 28] No space left on device", "errno": 28,
                            "winerror": None, "oserror": True}
    assert e.cancelled is False


def test_10b_failure_record_from_scan_and_stitch(engines, a1, tmp_path):
    _, v2 = engines
    bad = tmp_path / "Bad_Scan1"
    bad.mkdir()
    (bad / "Bad_Scan1_[1,1]_component_data.tif").write_bytes(b"not a tiff at all")
    e = v2.StitchingEngine(str(bad), str(tmp_path / "bad.ome.tif"))
    assert e.scan_metadata() is False
    assert e.last_error["oserror"] is False and e.last_error["type"] and e.last_error["message"]

    e = v2.StitchingEngine(str(a1), str(tmp_path / "out" / "A1.ome.tif"), temp_dir=str(tmp_path / "missing"))
    assert e.scan_metadata()
    assert e.stitch(use_ram=False) is None                   # the scratch folder does not exist
    assert e.last_error["oserror"] is True and e.last_error["errno"] is not None
    assert not e.temp_path.exists()


# --------------------------------------------------------------------------------------------------- item 11
@pytest.mark.filterwarnings("ignore::DeprecationWarning")     # tifffile's announced RGB-default change
def test_11_tifffile_drift_guard(legacy_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(legacy_engine, "NUM_THREADS", 1)
    folder = tmp_path / "Tri_Scan1"
    synth.make_slide(str(folder), "Tri_Scan1", 2, 2, channels=["DAPI", "Opal 520", "Opal 570"])
    stitch(legacy_engine, folder, tmp_path / "out" / "Tri_Scan1.ome.tif")
    got = md5(tmp_path / "out" / "Tri_Scan1.ome.tif")
    assert got == GOLDEN_3CH_MD5, (
        f"tifffile {tifffile.__version__} writes a 3-channel slide differently from the reviewed golden output "
        f"({got} != {GOLDEN_3CH_MD5}). Read the tifffile release notes before updating GOLDEN_3CH_MD5.")


# --------------------------------------------------------------------------------------------------- item 12
def test_12_one_channel_slides_cannot_be_written(engines, tmp_path):
    folder = tmp_path / "Seg_Scan1"
    synth.make_slide(str(folder), "Seg_Scan1", 2, 1, suffix="binary_seg_maps", channels=["Nucleus"])
    results = []
    for label, mod in zip(("v1", "v2"), engines):
        out = tmp_path / label / "Seg_Scan1.ome.tif"
        out.parent.mkdir()
        e = mod.StitchingEngine(str(folder), str(out))
        assert e.scan_metadata() and e.n_channels == 1
        results.append(e.write_to_disk(e.stitch(True), True))
        assert not out.exists()
    # If this starts failing, the engine can write 1-channel slides: revisit ruling R24 ("unsupported").
    assert not results[0] and results[1] is False


# ------------------------------------------------------------------------------- cancellation (§5.7 e, f, h, j)
def test_cancel_before_scan_and_between_tiles(engines, a1, tmp_path):
    _, v2 = engines
    ev = threading.Event()
    ev.set()
    e = v2.StitchingEngine(str(a1), str(tmp_path / "x.ome.tif"), cancel_event=ev)
    assert e.scan_metadata() is False and e.cancelled is True and e.tiles == []
    assert e.dtype is None                                    # stopped before the first tile was read

    ev = threading.Event()
    count = [0]

    def progress(cur, tot, phase):
        if phase == "Stitching":
            count[0] += 1
            if count[0] == 2:
                ev.set()
    e = v2.StitchingEngine(str(a1), str(tmp_path / "y.ome.tif"), progress, cancel_event=ev)
    assert e.scan_metadata()
    assert e.stitch(use_ram=False) is None
    assert e.cancelled is True and not e.temp_path.exists() and e.last_error is None
    assert e.failed_tiles == 0                                # cancelled tiles are not failures
    assert e.cleanup_temp() is True                           # idempotent


def test_cancel_before_a_pyramid_level_removes_the_output(engines, a1, tmp_path):
    _, v2 = engines
    ev = threading.Event()
    out = tmp_path / "z.ome.tif"
    e = v2.StitchingEngine(str(a1), str(out), lambda c, t, p: p.startswith("Writing Pyramid") and ev.set(),
                           cancel_event=ev)
    assert e.scan_metadata()
    canvas = e.stitch(True)
    assert canvas is not None
    assert e.write_to_disk(canvas, True) is False
    assert e.cancelled is True and not out.exists() and e.last_error is None


def test_cancel_before_writing_creates_no_file(engines, a1, tmp_path):
    _, v2 = engines
    ev = threading.Event()
    out = tmp_path / "w.ome.tif"
    e = v2.StitchingEngine(str(a1), str(out), cancel_event=ev)
    assert e.scan_metadata()
    canvas = e.stitch(True)
    ev.set()
    assert e.write_to_disk(canvas, True) is False and not out.exists()


# ------------------------------------------------------------------------------------------ entry point (§5.7 l)
def test_constructor_defaults_reproduce_v1(legacy_engine):
    import inspect
    p1 = list(inspect.signature(legacy_engine.StitchingEngine.__init__).parameters.values())
    p2 = list(inspect.signature(stitcher_RAM.StitchingEngine.__init__).parameters.values())
    assert [(p.name, p.default) for p in p2[:len(p1)]] == [(p.name, p.default) for p in p1]
    assert all(p.kind is p.KEYWORD_ONLY and p.default is None for p in p2[len(p1):])
    e = stitcher_RAM.StitchingEngine("/data/SlideX", "/out/SlideX.ome.tif")
    assert (e.tile_files, e.image_name, e.cancel_event, e.cancelled, e.failed_tiles, e.last_error, e.n_listed) == \
        (None, "SlideX", None, False, 0, None, 0)
    assert e.temp_path == Path("/out/SlideX.ome.tmp.raw")


# ------------------------------------------------------------------------------------- n_listed (DS-1, D12)
def test_n_listed_counts_the_tiles_a_scan_dropped_and_changes_no_byte(engines, a1, tmp_path):
    """The engine's only addition for DS-1: n_listed, the files scan_metadata considered. A tile whose header can't
    be read (here 0 bytes) is dropped from e.tiles exactly as v1 drops it, and the output stays identical."""
    v1, v2 = engines
    folder = tmp_path / "SlideA1_Scan1"
    shutil.copytree(a1, folder)
    names = component_names(folder)
    (folder / names[3]).write_bytes(b"")                                           # an interrupted copy
    e1, _ = stitch(v1, folder, tmp_path / "v1" / "SlideA1_Scan1.ome.tif")
    e2, ok = stitch(v2, folder, tmp_path / "v2" / "SlideA1_Scan1.ome.tif", tile_files=names)
    assert ok is True and (e2.n_listed, len(e2.tiles), e2.failed_tiles) == (len(names), len(names) - 1, 0)
    assert len(e1.tiles) == len(e2.tiles) and not hasattr(e1, "n_listed")
    assert sha256(tmp_path / "v2" / "SlideA1_Scan1.ome.tif") == sha256(tmp_path / "v1" / "SlideA1_Scan1.ome.tif")
    legacy = v2.StitchingEngine(str(folder), str(tmp_path / "unused.ome.tif"))     # v1 mode lists the folder
    assert legacy.scan_metadata() and legacy.n_listed == len(names)


def test_classic_flag_starts_the_v1_ui(monkeypatch):
    started = []

    class FakeApp:
        def mainloop(self):
            started.append(True)
    monkeypatch.setattr(stitcher_RAM, "StitcherApp", FakeApp)
    assert stitcher_RAM._main(["--classic", "--input", "x"]) == 0 and started == [True]


def test_default_entry_hands_the_engine_to_the_v2_ui(monkeypatch):
    seen = {}
    fake = types.ModuleType("yfs.ui.app")
    fake.main = lambda engine_cls, argv: seen.update(engine_cls=engine_cls, argv=argv) or 7
    monkeypatch.setitem(sys.modules, "yfs.ui.app", fake)
    assert stitcher_RAM._main(["--smoke-test"]) == 7
    assert seen == {"engine_cls": stitcher_RAM.StitchingEngine, "argv": ["--smoke-test"]}
