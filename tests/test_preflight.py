"""Preflight I/O (FINAL_SPEC §3.8, §5.5.6): header prediction parity with the engine (acceptance A2), the
verification matrix of existing outputs (A3, D2), the output-folder index, the write probe, partial ownership
(D3), leftover deletion and finalizing a partial."""
from __future__ import annotations

import errno
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import tifffile

import stitcher_RAM
from tests.conftest import real_subset
from yfs.core import preflight
from yfs.core.fsservice import map_oserror
from yfs.core.names import COMPLETE_SUFFIX, OWNER_SUFFIX, PARTIAL_DIR, norm_key, temp_subdir
from yfs.core.places import MountTable
from yfs.core.model import OutputInfo

ENGINE = stitcher_RAM.StitchingEngine
ROOT_FS = MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw")
POSITIONS = [(286, 5, 1, (100, 20000), True), (287, 5, 1, (200, 20000), True)]
AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):
    monkeypatch.setattr(preflight, "PROBE_DELAY_S", 0.0)
    monkeypatch.setattr(preflight, "RETRY_DELAY_S", 0.0)


def age(*paths, seconds=200):
    for p in paths:
        os.utime(p, (time.time() - seconds, time.time() - seconds))


def pe_xml(name, color=None, image_type="FullResolution"):
    return ('<?xml version="1.0" encoding="utf-16"?>\n<PerkinElmer-QPI-ImageDescription>\n'
            f"  <ImageType>{image_type}</ImageType>\n" + (f"  <Name>{name}</Name>\n" if name else "")
            + (f"  <Color>{color}</Color>\n" if color else "") + "</PerkinElmer-QPI-ImageDescription>")


def write_tile(path, names, colors=None, unit="CENTIMETER", res=20000, drop=(), xml=True):
    """One page per channel (32 x 48 float32) plus a thumbnail page, like an inForm component tile. The tags in
    `drop` (e.g. 282/283, which tifffile always writes) are renamed to private codes afterwards."""
    with tifffile.TiffWriter(path) as tif:
        for c, name in enumerate(names):
            tif.write(np.full((32, 48), c, np.float32), photometric="minisblack", resolution=(res, res),
                      resolutionunit=unit, description=pe_xml(name, (colors or {}).get(c)) if xml else None,
                      extratags=POSITIONS, metadata=None)
        tif.write(np.zeros((8, 12, 3), np.uint8), photometric="rgb", description=pe_xml(None, None, "Thumbnail")
                  if xml else None, metadata=None)
    if drop:
        with tifffile.TiffFile(path) as tif:
            spots = [(t.offset, 65000 + t.code) for p in tif.pages for t in p.tags.values() if t.code in drop]
            order = tif.byteorder
        with open(path, "r+b") as f:
            for offset, code in spots:
                f.seek(offset)
                f.write(struct.pack(order + "H", code))
    return Path(path)


def engine_view(folder: Path, names):
    """What the v2 engine decides from the first of `names` (and whether it asked for channel names)."""
    asked = []
    eng = ENGINE(str(folder), str(folder / "unused.ome.tif"), None, lambda name, cur: asked.append(cur),
                 tile_files=names)
    eng.scan_metadata()
    return eng, asked


def assert_predicts(header, eng, asked):
    assert (header.n_channels, header.tile_h, header.tile_w) == (eng.n_channels, eng.tile_h, eng.tile_w)
    assert np.dtype(header.dtype) == eng.dtype
    assert (header.channel_problem is not None) == bool(asked)
    if header.channel_problem is None:
        assert list(header.channels_extracted) == eng.channel_names
    assert abs(header.um_per_px_x - eng.phys_size_x_um) < 1e-9 and abs(header.um_per_px_y - eng.phys_size_y_um) < 1e-9


# ------------------------------------------------------------------------------------------- A2: header parity
@pytest.mark.parametrize("folder, prefix, problem", [
    ("ProjectA/SlideA1_Scan1", "SlideA1_Scan1", None),
    ("suspicious/SusSlide_Scan1", "SusSlide_Scan1", "duplicate"),
    ("inform_export", "S1_Scan1_[", None),                       # one slide of a multi-slide folder
    ("round1_unmix/R1-02", "R1-02_Scan1", None),
    ("Spaces & ünïcödé/Slide [weird] (1)", "Slide [weird] (1)", None),
])
def test_header_predicts_the_engine_on_synthetic_tiles(synth_tree, folder, prefix, problem):
    d = synth_tree / folder
    names = sorted(n for n in os.listdir(d) if n.startswith(prefix) and n.endswith("_component_data.tif"))
    h = preflight.read_tile_header(str(d / names[0]), ENGINE)
    assert_predicts(h, *engine_view(d, names))
    assert (h.n_channels, h.tile_h, h.tile_w, h.dtype, h.channel_problem) == (5, 256, 320, "float32", problem)
    assert h.resolution_found and h.um_per_px_x == 0.5 and h.first_image_bytes == 5 * 256 * 320 * 4
    assert h.channel_colors == (None,) * 5 and not h.offline


def test_header_of_a_one_channel_product_is_unsupported_input(synth_tree):
    d = synth_tree / "inform_export"
    names = sorted(n for n in os.listdir(d) if n.endswith("_binary_seg_maps.tif"))
    h = preflight.read_tile_header(str(d / names[0]), ENGINE)
    assert_predicts(h, *engine_view(d, names))
    assert h.n_channels == 1 and h.channels_extracted == ("Nucleus",) and h.channel_problem is None


@pytest.mark.parametrize("unit, res, um", [
    ("INCH", 50800, 0.5), ("CENTIMETER", 50800, 10000 / 50800), ("NONE", 50800, 0.5)])
def test_resolution_units_match_the_engine(tmp_path, unit, res, um):
    write_tile(tmp_path / "T_[1,2]_component_data.tif", ["DAPI", "Opal 520", "Opal 570"], unit=unit, res=res)
    h = preflight.read_tile_header(str(tmp_path / "T_[1,2]_component_data.tif"), ENGINE)
    assert_predicts(h, *engine_view(tmp_path, ["T_[1,2]_component_data.tif"]))
    assert abs(h.um_per_px_x - um) < 1e-9 and h.resolution_found


# FS-15: the header read predicts that the engine will not place these tiles
@pytest.mark.parametrize("drop, kinds", [((282, 283), "resolution"), ((286, 287), "position"), ((287,), "position"),
                                         ((283, 286), "resolution or position")])
def test_a_first_tile_without_placement_tags_is_a_header_problem(tmp_path, drop, kinds):
    p = write_tile(tmp_path / "T_[1,2]_component_data.tif", ["DAPI", "Opal 520", "Opal 570"], drop=drop)
    with pytest.raises(ValueError, match=f"^it has no {kinds} tags$"):
        preflight.read_tile_header(str(p), ENGINE)
    eng, _ = engine_view(tmp_path, [p.name])
    assert eng.n_listed == 1 and eng.tiles == [] and eng.canvas_shape is None      # the engine drops it


def test_a_cut_off_first_tile_is_a_header_problem(synth_tree, tmp_path):
    src = synth_tree / "ProjectA" / "SlideA1_Scan1"
    first = sorted(n for n in os.listdir(src) if n.endswith("_component_data.tif"))[0]
    data = (src / first).read_bytes()
    (tmp_path / first).write_bytes(data[:len(data) // 3])          # an interrupted copy: the headers survive
    with pytest.raises(ValueError, match="^the file is cut off$"):
        preflight.read_tile_header(str(tmp_path / first), ENGINE)
    eng, _ = engine_view(tmp_path, [first])
    assert eng.tiles == [] and eng.last_error is not None             # the engine fails the whole slide


def test_a_damaged_first_tile_is_a_bad_tile_but_an_io_error_is_not(tmp_path, monkeypatch):
    """UD4: an empty file, or one that isn't a TIFF, raises a plain ValueError (a bad tile: reading it again can't
    help, and the engine fails the slide). An I/O error met inside the TIFF structure, which tifffile reports as a
    TiffFileError (a ValueError too), raises the OSError, so the row keeps Try again."""
    p = tmp_path / "T_[1,2]_component_data.tif"
    for data in (b"", b"not a tiff at all" * 10):
        p.write_bytes(data)
        with pytest.raises(ValueError, match="^it isn't a readable TIFF$"):
            preflight.read_tile_header(str(p), ENGINE)
        eng, _ = engine_view(tmp_path, [p.name])
        assert eng.tiles == [] and eng.last_error is not None
    write_tile(p, ["DAPI", "Opal 520"])
    real_read = tifffile.FileHandle.read

    def read(self, size=-1):                     # the share fails once the file header has been read
        if self.tell() >= 8:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_read(self, size)
    monkeypatch.setattr(tifffile.FileHandle, "read", read)
    with pytest.raises(OSError) as e:
        preflight.read_tile_header(str(p), ENGINE)
    assert e.value.errno == errno.ESTALE and not isinstance(e.value, ValueError)


def test_an_io_error_that_tifffile_swallows_is_still_an_io_error(tmp_path, monkeypatch):
    """UD4: tifffile swallows an I/O error met while it walks the pages (it logs 'corrupted tag list' and goes on
    with the pages it found), so a share hiccup at a later channel's page made a 5-channel tile look like a 2, 3 or
    4-channel one, with no Try again. Any read that fails now raises its OSError."""
    p = write_tile(tmp_path / "T_[1,2]_component_data.tif", ["DAPI", "Opal 520", "Opal 570", "Opal 620", "Opal 690"])
    with tifffile.TiffFile(p) as tif:
        offsets = [page.offset for page in tif.pages]
    gone = os.open(tmp_path / "gone", os.O_WRONLY | os.O_CREAT)    # the OS refuses to read it (EBADF)
    real_read = tifffile.FileHandle.read
    try:
        for at in offsets[1:]:
            def read(self, size=-1, at=at):          # the share goes away when the walk reaches page `at`
                if self.tell() >= at:
                    os.dup2(gone, self.fileno())
                return real_read(self, size)
            monkeypatch.setattr(tifffile.FileHandle, "read", read)
            with pytest.raises(OSError):
                preflight.read_tile_header(str(p), ENGINE)
    finally:
        os.close(gone)


@pytest.mark.parametrize("names, xml, problem", [
    (["DAPI", "DAPI", "DAPI"], True, "duplicate"), (["DAPI", None, "Opal 570"], True, "count"),
    (["a", "b", "c"], False, "missing")])
def test_channel_problems_match_the_engine(tmp_path, names, xml, problem):
    write_tile(tmp_path / "T_[1,2]_component_data.tif", names, xml=xml)
    h = preflight.read_tile_header(str(tmp_path / "T_[1,2]_component_data.tif"), ENGINE)
    assert h.channel_problem == problem
    assert_predicts(h, *engine_view(tmp_path, ["T_[1,2]_component_data.tif"]))


def test_interleaved_samples_use_the_engine_shape_rule(tmp_path):
    tifffile.imwrite(tmp_path / "rgb.tif", np.zeros((32, 48, 3), np.uint16), photometric="rgb",
                     resolution=(20000, 20000), resolutionunit="CENTIMETER", extratags=POSITIONS)
    h = preflight.read_tile_header(str(tmp_path / "rgb.tif"), ENGINE)
    assert (h.n_channels, h.tile_h, h.tile_w, h.shape) == (3, 32, 48, (32, 48, 3))
    assert_predicts(h, *engine_view(tmp_path, ["rgb.tif"]))


def test_channel_colours_come_from_the_tile_xml(tmp_path):
    p = write_tile(tmp_path / "T_[1,2]_component_data.tif", ["DAPI", "Opal 520", "Opal 570"],
                   colors={0: "0,0,255", 1: "0,255,0", 2: "bad"})
    assert preflight.read_tile_header(str(p), ENGINE).channel_colors == ("#0000ff", "#00ff00", None)


def test_header_never_reads_pixel_data(synth_tree, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("pixel data was read")
    for owner in (tifffile.TiffFile, tifffile.TiffPage, tifffile.TiffPageSeries):
        monkeypatch.setattr(owner, "asarray", forbidden)
    monkeypatch.setattr(tifffile.FileHandle, "read_segments", forbidden)
    d = synth_tree / "ProjectA" / "SlideA1_Scan1"
    assert preflight.read_tile_header(str(d / sorted(os.listdir(d))[0]), ENGINE).n_channels == 5


def test_windows_placeholder_is_not_opened(tmp_path, monkeypatch):
    p = write_tile(tmp_path / "T_[1,2]_component_data.tif", ["DAPI"])
    real_stat = os.stat

    class St:
        st_file_attributes = 0x400000            # RECALL_ON_DATA_ACCESS

    monkeypatch.setattr(preflight, "WINDOWS", True)
    monkeypatch.setattr(preflight.os, "stat", lambda path, *a, **k: St() if str(path) == str(p) else
                        real_stat(path, *a, **k))
    monkeypatch.setattr(tifffile, "TiffFile", None)          # opening it would fail
    h = preflight.read_tile_header(str(p), ENGINE)
    assert h.offline and h.n_channels == 0 and h.first_image_bytes == 0


def test_first_tiff_name_is_the_engine_choice(tmp_path):
    for name in ("b.tif", "A.TIFF", "._a.tif", "c.txt", "a.ome.tif"):
        (tmp_path / name).write_bytes(b"")
    engine_first = sorted(f for f in os.listdir(tmp_path) if f.lower().endswith((".tif", ".tiff")))[0]
    assert preflight.first_tiff_name(str(tmp_path)) == engine_first == "._a.tif"
    (tmp_path / "none").mkdir()
    assert preflight.first_tiff_name(str(tmp_path / "none")) is None


def test_legacy_gaps_count_only_the_tiles_left_out(tmp_path):
    """DS-1 for legacy_all rows (every TIFF in the folder): of the files the engine leaves out, a readable TIFF without
    XPosition/YPosition (an overview image, an earlier output) is no tile; an unreadable one, or one with positions
    but no resolution, is a gap."""
    write_tile(tmp_path / "a.tif", ["DAPI"])
    write_tile(tmp_path / "b.tif", ["DAPI"], drop=(282, 283))
    write_tile(tmp_path / "c.tif", ["DAPI"], drop=(286, 287))
    (tmp_path / "d.TIFF").write_bytes(b"")
    (tmp_path / "e.txt").write_bytes(b"")
    eng = ENGINE(str(tmp_path), str(tmp_path / "out" / "unused.ome.tif"))
    assert eng.scan_metadata() and eng.n_listed - len(eng.tiles) == 3      # v1's rule leaves out b, c and d
    assert preflight.legacy_gaps(str(tmp_path), [os.path.basename(t["path"]) for t in eng.tiles]) == 2


@pytest.mark.real
def test_header_predicts_the_engine_on_the_real_subset():
    d = real_subset()
    if d is None:
        pytest.skip("set YFS_REAL_SUBSET to a folder of real *_component_data.tif tiles")
    names = sorted(n for n in os.listdir(d) if n.endswith("_component_data.tif") and not n.startswith("._"))
    h = preflight.read_tile_header(str(d / names[0]), ENGINE)
    eng, asked = engine_view(d, names)
    assert_predicts(h, eng, asked)
    assert h.resolution_found and len(h.channel_colors) == h.n_channels
    assert all(c is None or (c.startswith("#") and len(c) == 7) for c in h.channel_colors)


# ------------------------------------------------------------------------------------- A3: verify_output
@pytest.fixture(scope="module")
def stitched(tmp_path_factory, synth_tree, legacy_engine):
    """SlideA1 written by v1 and by v2 (2 pyramid levels)."""
    d, out = tmp_path_factory.mktemp("verify"), {}
    for label, mod in (("v1", legacy_engine), ("v2", stitcher_RAM)):
        threads, mod.NUM_THREADS = mod.NUM_THREADS, 1
        try:
            eng = mod.StitchingEngine(str(synth_tree / "ProjectA" / "SlideA1_Scan1"), str(d / f"{label}.ome.tif"))
            assert eng.scan_metadata()
            canvas = eng.stitch(use_ram=True)
            eng.write_to_disk(canvas, True)
            del canvas
        finally:
            mod.NUM_THREADS = threads
        out[label] = d / f"{label}.ome.tif"
    return out


def copy(src, tmp_path, name="x.ome.tif") -> Path:
    dst = tmp_path / name
    shutil.copyfile(src, dst)
    return dst


def patch_tag(path, code, index, value, sub=None):
    """Overwrite value `index` of tag `code` in page 0 (or in its SubIFD `sub`) in place."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0] if sub is None else tif.pages[0].pages[sub]
        tag, order = page.tags[code], tif.byteorder
        fmt = {3: "H", 4: "I", 13: "I", 16: "Q", 18: "Q"}[int(tag.dtype)]
        offset = tag.valueoffset + index * struct.calcsize(fmt)
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(struct.pack(order + fmt, value))


@pytest.mark.parametrize("label", ["v1", "v2"])
def test_v1_and_v2_outputs_are_valid(stitched, label):
    assert preflight.verify_output(str(stitched[label])) == ("valid", "")


@pytest.mark.parametrize("fraction", [0.50, 0.90, 0.99])
def test_truncated_outputs_are_incomplete(stitched, tmp_path, fraction):
    p = copy(stitched["v2"], tmp_path)
    with open(p, "r+b") as f:
        f.truncate(int(os.path.getsize(p) * fraction))
    verdict, detail = preflight.verify_output(str(p))
    assert verdict == "incomplete" and detail


@pytest.mark.parametrize("sub", [None, 1])
def test_zeroed_bytecounts_are_incomplete(stitched, tmp_path, sub):
    p = copy(stitched["v2"], tmp_path)
    patch_tag(p, 325, 0, 0, sub)                 # TileByteCounts
    assert preflight.verify_output(str(p))[0] == "incomplete"


def test_a_subifd_never_written_is_incomplete(stitched, tmp_path):
    p = copy(stitched["v2"], tmp_path)
    patch_tag(p, 330, 1, 0)                      # SubIFDs: the second level's offset was never filled in
    assert preflight.verify_output(str(p)) == ("incomplete", "a pyramid level was never written")


def pyramid(path, levels, step=2, description=None, shape=(736, 920, 5)):
    """A tiled pyramid like the engine's (the engine would write 2 levels for this shape), with `levels` SubIFDs
    `step` times smaller each, and page 0's ImageDescription."""
    data = np.zeros(shape, np.float32)
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(data, subifds=levels, tile=(512, 512), planarconfig="CONTIG", description=description,
                  metadata=None)
        for k in range(1, levels + 1):
            tif.write(data[::step ** k, ::step ** k], tile=(512, 512), planarconfig="CONTIG", metadata=None)
    return Path(path)


def engine_xml(stitched) -> bytes:
    with tifffile.TiffFile(stitched["v2"]) as tif:
        return tif.pages[0].description.encode("utf-8")              # the engine writes UTF-8 bytes ("µm")


def test_an_engine_written_file_with_missing_pyramid_levels_is_incomplete(stitched, tmp_path):
    p = pyramid(tmp_path / "x.ome.tif", 1, description=engine_xml(stitched))
    assert preflight.verify_output(str(p)) == ("incomplete", "1 of 2 pyramid levels")


@pytest.mark.parametrize("levels, step, description", [
    (1, 4, None), (0, 2, None), (3, 2, None), (1, 2, "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" "
                                                    "Creator=\"OME Bio-Formats 6.12\"><Image ID=\"Image:0\"/></OME>")])
def test_another_programs_pyramid_is_valid(tmp_path, levels, step, description):
    """D2/FS-5: QuPath or bfconvert lay pyramids out differently (factor 4, other level counts): that is no sign of
    an interrupted write, so the file is valid, made by another program."""
    p = pyramid(tmp_path / "x.ome.tif", levels, step, description)
    assert preflight.verify_output(str(p)) == ("valid", "made by another program")


@pytest.mark.parametrize("label", ["v1", "v2"])
def test_engine_output_is_recognised_by_its_ome_xml(stitched, tmp_path, label):
    with tifffile.TiffFile(stitched[label]) as tif:
        assert preflight._engine_written(tif.pages[0])
    tifffile.imwrite(tmp_path / "ome.ome.tif", np.zeros((64, 80, 5), np.float32), ome=True)   # tifffile's OME-XML
    tifffile.imwrite(tmp_path / "plain.tif", np.zeros((64, 80, 5), np.float32), description="Slide 1")
    for name in ("ome.ome.tif", "plain.tif"):
        with tifffile.TiffFile(tmp_path / name) as tif:
            assert not preflight._engine_written(tif.pages[0])


def test_any_cut_of_an_engine_output_is_incomplete(stitched, tmp_path):
    """Wherever an interrupted copy or write cut the file once page 0's tags are complete, the verdict is
    'incomplete' (including inside a pyramid level's IFD), never 'valid' or 'unreadable'."""
    data = stitched["v2"].read_bytes()
    with tifffile.TiffFile(stitched["v2"]) as tif:
        subs = tif.pages[0].subifds
    cuts = sorted({int(len(data) * k / 97) for k in range(1, 97)} | {o + d for o in subs for d in (1, 8, 40, 120)})
    seen = {}
    for n in cuts:
        (tmp_path / "x.ome.tif").write_bytes(data[:n])
        seen[n] = preflight.verify_output(str(tmp_path / "x.ome.tif"))[0]
    assert {v for n, v in seen.items() if n > 2048} == {"incomplete"}
    assert "valid" not in seen.values()


def test_unreadable_files(tmp_path):
    (tmp_path / "junk.ome.tif").write_bytes(b"not a tiff at all" * 10)
    assert preflight.verify_output(str(tmp_path / "junk.ome.tif"))[0] == "unreadable"
    (tmp_path / "empty.ome.tif").write_bytes(b"")
    assert preflight.verify_output(str(tmp_path / "empty.ome.tif")) == ("incomplete", "the file is empty")


def test_io_errors_leave_the_file_unchecked(stitched, tmp_path, monkeypatch):
    """D2/DS-7/FS-5/MW-3: an I/O error is no evidence of anything: 'unchecked' (check again next time), never a
    verdict that ticks the row for replacement."""
    def text(eno):
        return map_oserror(OSError(eno, "x")).message.rstrip(".")
    assert preflight.verify_output(str(tmp_path / "missing.ome.tif")) == ("unchecked", text(errno.ENOENT))
    p = copy(stitched["v2"], tmp_path)
    if not AS_ROOT:
        p.chmod(0o000)
        try:
            assert preflight.verify_output(str(p)) == ("unchecked", text(errno.EACCES))
        finally:
            p.chmod(0o644)

    def eio(*a, **k):
        raise OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(preflight.tifffile, "TiffFile", eio)
    assert preflight.verify_output(str(p)) == ("unchecked", text(errno.EIO))


def test_an_io_error_inside_the_pyramid_is_unchecked(stitched, tmp_path, monkeypatch):
    p = copy(stitched["v2"], tmp_path)
    with tifffile.TiffFile(str(p)) as tif:
        first_level = tif.pages[0].subifds[0]
    real_read = tifffile.FileHandle.read

    def read(self, size=-1):                     # the share fails while the pyramid levels' IFDs are read
        if self.tell() >= first_level - 8:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_read(self, size)
    monkeypatch.setattr(tifffile.FileHandle, "read", read)
    assert preflight.verify_output(str(p)) == ("unchecked", map_oserror(OSError(errno.ESTALE, "")).message.rstrip("."))


def test_small_image_needs_no_pyramid(tmp_path):
    tifffile.imwrite(tmp_path / "s.ome.tif", np.zeros((200, 250, 5), np.float32), planarconfig="CONTIG")
    assert preflight.verify_output(str(tmp_path / "s.ome.tif")) == ("valid", "")




# ------------------------------------------------------------------------------------------ list_output_dir
def make_out(tmp_path) -> Path:
    out = tmp_path / "out"
    (out / PARTIAL_DIR).mkdir(parents=True)
    (out / "sub").mkdir()
    for name, size in (("A.ome.tif", 10), ("B.OME.TIFF", 20), ("notes.txt", 1), ("X.ome.tmp.raw", 30)):
        (out / name).write_bytes(b"x" * size)
    for name, size in (("C.ome.tif", 40), ("C.ome.tmp.raw", 50), ("D.ome.tif", 60)):
        (out / PARTIAL_DIR / name).write_bytes(b"x" * size)
    return out


def own(partial, pid, started, host=None, token="t0", boot=None) -> dict:
    """Write an owner record for `partial`, as another stitcher's claim would."""
    record = {"host": host or preflight.HOST, "boot": preflight._BOOT if boot is None else boot, "pid": pid,
              "started": started, "token": token}
    Path(str(partial) + OWNER_SUFFIX).write_text(json.dumps(record))
    return record


def test_output_index_lists_outputs_partials_and_leftovers(tmp_path):
    out = make_out(tmp_path)
    age(*out.iterdir(), *(out / PARTIAL_DIR).iterdir())
    idx = preflight.list_output_dir(str(out), ROOT_FS, planned=["c.OME.tif", "a.ome.tif"])
    assert idx.exists and idx.error is None and idx.out_dir == str(out)
    assert idx.names == {norm_key(n) for n in ("A.ome.tif", "B.OME.TIFF", "notes.txt", "X.ome.tmp.raw", "sub")}
    assert {k: (i.name, i.size) for k, i in idx.outputs.items()} == {"a.ome.tif": ("A.ome.tif", 10),
                                                                     "b.ome.tiff": ("B.OME.TIFF", 20)}
    assert idx.outputs["a.ome.tif"].mtime_ns == os.stat(out / "A.ome.tif").st_mtime_ns
    assert sorted(idx.partials) == ["c.ome.tif", "d.ome.tif"] and idx.ready == {}
    assert sorted((os.path.relpath(p, out), i.size) for p, i in idx.leftovers) == [
        (os.path.join(PARTIAL_DIR, "C.ome.tmp.raw"), 50), (os.path.join(PARTIAL_DIR, "D.ome.tif"), 60),
        ("X.ome.tmp.raw", 30)]
    assert idx.free_bytes and idx.free_bytes > 0
    assert (idx.lane, idx.lane_class, idx.is_system_volume) == ("/", "local", True)
    assert (idx.missing_levels, idx.nearest_existing, idx.pdir_problem, idx.cloud_synced) == (0, None, None, False)


def test_without_a_plan_no_partial_is_offered(tmp_path):
    """DS-9: at startup the save folder is indexed before the scan has built a plan: only scratch files can be
    leftovers then, never a partial (it may be a finished image waiting for Finalize)."""
    out = make_out(tmp_path)
    age(*out.iterdir(), *(out / PARTIAL_DIR).iterdir())
    for planned in (None, frozenset()):
        idx = preflight.list_output_dir(str(out), ROOT_FS, planned)
        offered = sorted(os.path.basename(p) for p, _ in idx.leftovers)
        assert offered == (["C.ome.tmp.raw", "X.ome.tmp.raw"] if planned is None else
                           ["C.ome.tif", "C.ome.tmp.raw", "D.ome.tif", "X.ome.tmp.raw"])


def test_complete_and_live_partials_are_never_leftovers(tmp_path, live_process, dead_process):
    """D3, DS-9, FS-13: a complete partial is 'ready'; a live one (fresh, or its owner runs) and its scratch file
    are not offered; a dead owner's or a stale one is."""
    out = make_out(tmp_path)
    pdir = out / PARTIAL_DIR
    for name in ("R.ome.tif", "R.ome.tmp.raw", "F.ome.tmp.raw", "O.ome.tif", "O.ome.tmp.raw", "Z.ome.tif",
                 "Z.ome.tmp.raw", "Y.ome.tif.stale-ab12"):
        (pdir / name).write_bytes(b"x" * 7)
    (pdir / ("R.ome.tif" + COMPLETE_SUFFIX)).write_text('{"gaps": 0}')
    (pdir / "F.ome.tif").write_bytes(b"fresh")                        # a run writing it just now (heartbeat)
    own(pdir / "O.ome.tif", *live_process)                            # its owner runs on this computer
    own(pdir / "Z.ome.tif", *dead_process)                            # its owner has exited
    age(*[p for p in pdir.iterdir() if p.name != "F.ome.tif"], seconds=600)
    idx = preflight.list_output_dir(str(out), ROOT_FS, planned=["a.ome.tif"])
    assert sorted(idx.ready) == ["r.ome.tif"] and "r.ome.tif" not in idx.partials
    assert sorted(idx.partials) == ["c.ome.tif", "d.ome.tif", "f.ome.tif", "o.ome.tif", "z.ome.tif"]
    offered = sorted(os.path.basename(p) for p, _ in idx.leftovers)
    assert offered == ["C.ome.tif", "C.ome.tmp.raw", "D.ome.tif", "R.ome.tmp.raw", "Y.ome.tif.stale-ab12",
                       "Z.ome.tif", "Z.ome.tmp.raw"]


def test_live_partials_are_busy(tmp_path, live_process, dead_process):
    """The plan's 'busy' rows: partials another stitcher is writing (its owner runs on this computer or, with no
    owner to ask, it changed in the last 2 minutes). A dead owner's, a stale and a complete partial are not busy."""
    out = make_out(tmp_path)
    pdir = out / PARTIAL_DIR
    for name in ("Live.ome.tif", "Fresh.ome.tif", "Dead.ome.tif", "Done.ome.tif"):
        (pdir / name).write_bytes(b"x")
    own(pdir / "Live.ome.tif", *live_process)
    own(pdir / "Dead.ome.tif", *dead_process)
    (pdir / ("Done.ome.tif" + COMPLETE_SUFFIX)).write_text('{"gaps": 0}')
    age(*[p for p in [*out.iterdir(), *pdir.iterdir()] if not p.name.startswith("Fresh")], seconds=600)
    idx = preflight.list_output_dir(str(out), ROOT_FS)
    assert idx.busy == {"live.ome.tif", "fresh.ome.tif"} and idx.busy <= set(idx.partials)
    assert sorted(idx.ready) == ["done.ome.tif"]


def test_partials_age_by_the_clock_of_their_folder(tmp_path, monkeypatch):
    """DS-3, as the runner tells it: a partial that another computer wrote a minute ago, by the clock of the server
    that stamps the save folder's files, is busy even when this computer's clock runs 4 minutes ahead."""
    out = tmp_path / "out"
    (out / PARTIAL_DIR).mkdir(parents=True)
    (out / PARTIAL_DIR / "A.ome.tif").write_bytes(b"x")
    age(out / PARTIAL_DIR / "A.ome.tif", seconds=300)                 # 5 minutes old by this computer's clock
    asked = []
    monkeypatch.setattr(preflight, "share_now", lambda folder: (asked.append(folder), time.time() - 240)[1])
    idx = preflight.list_output_dir(str(out), ROOT_FS, [])
    assert idx.busy == {"a.ome.tif"} and idx.leftovers == () and asked == [str(out / PARTIAL_DIR)]


def test_v1_scratch_file_is_live_while_its_output_grows(tmp_path):
    """FS-13: v1 writes X.ome.tif in place next to its X.ome.tmp.raw: no leftover while X.ome.tif is fresh."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "X.ome.tmp.raw").write_bytes(b"canvas")
    (out / "X.ome.tif").write_bytes(b"growing")
    age(out / "X.ome.tmp.raw", seconds=600)
    assert preflight.list_output_dir(str(out), ROOT_FS, []).leftovers == ()
    age(out / "X.ome.tif", seconds=600)
    assert [os.path.basename(p) for p, _ in preflight.list_output_dir(str(out), ROOT_FS, []).leftovers] == \
        ["X.ome.tmp.raw"]


def test_scratch_files_in_a_custom_temp_folder_are_leftovers_of_their_save_folder(tmp_path):
    """DS-16 / D4: a crash in disk mode leaves the canvas-sized scratch file in this save folder's subfolder of the
    custom temp folder; it is offered with the save folder's leftovers (another save folder's are not)."""
    out, other, temp = make_out(tmp_path), tmp_path / "other", tmp_path / "scratch"
    mine, theirs = Path(temp_subdir(str(temp), str(out))), Path(temp_subdir(str(temp), str(other)))
    for d in (mine, theirs):
        d.mkdir(parents=True)
        (d / "C.ome.tmp.raw").write_bytes(b"x" * 1000)
    (mine / "D.ome.tmp.raw").write_bytes(b"x" * 2000)
    (out / PARTIAL_DIR / "D.ome.tif").write_bytes(b"live")           # D's run is still going
    age(mine / "C.ome.tmp.raw", mine / "D.ome.tmp.raw", theirs / "C.ome.tmp.raw", out / PARTIAL_DIR / "C.ome.tif",
        seconds=600)
    idx = preflight.list_output_dir(str(out), ROOT_FS, ["c.ome.tif", "d.ome.tif"], temp_dir=str(temp))
    assert (str(mine / "C.ome.tmp.raw"), 1000) in [(p, i.size) for p, i in idx.leftovers]
    assert not any(str(theirs) in p or p.endswith("D.ome.tmp.raw") for p, _ in idx.leftovers)
    paths = [p for p, _ in idx.leftovers] + [str(mine / "D.ome.tmp.raw")]
    n, freed, errors = delete_confirmed(paths, out_dir=str(out))    # D's partial is checked at delete
    assert (n, freed) == (1, 1000) and errors == ["D.ome.tmp.raw: another stitcher is using it, kept."]
    assert not (mine / "C.ome.tmp.raw").exists() and (mine / "D.ome.tmp.raw").exists()
    age(out / PARTIAL_DIR / "D.ome.tif", seconds=600)                 # D's run crashed too
    assert delete_confirmed([str(mine / "D.ome.tmp.raw")], out_dir=str(out)) == (1, 2000, [])
    assert not mine.exists() and theirs.exists()                       # the emptied subfolder goes


def test_output_index_of_a_missing_folder_or_a_file(tmp_path):
    idx = preflight.list_output_dir(str(tmp_path / "new" / "deeper"), ROOT_FS)
    assert not idx.exists and idx.error is None and not idx.outputs and idx.free_bytes > 0
    assert (idx.missing_levels, idx.nearest_existing) == (2, str(tmp_path))          # MW-26
    (tmp_path / "file").write_bytes(b"")
    idx = preflight.list_output_dir(str(tmp_path / "file"), ROOT_FS)
    assert idx.error is not None and idx.error.code == "not_a_dir" and idx.missing_levels == 0


def test_output_index_reports_a_cloud_synced_save_folder(tmp_path):
    """PL-6: cloud_synced comes from the mount table (here an rclone mount)."""
    mounts = MountTable.from_mountinfo_text(f"1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
                                            f"2 1 0:50 / {tmp_path} rw - fuse.rclone labcloud: rw")
    assert preflight.list_output_dir(str(tmp_path), mounts).cloud_synced is mounts.cloud_synced(str(tmp_path))
    assert preflight.list_output_dir(str(tmp_path), mounts).cloud_synced
    assert not preflight.list_output_dir(str(tmp_path), ROOT_FS).cloud_synced


def test_a_linked_partial_folder_is_not_used(tmp_path):
    """DS-10: <out>/.stitcher-in-progress linked to a folder of real images: nothing in it is a partial or a
    leftover, step 2 gets the reason, and delete_leftovers refuses to delete through the link."""
    out, target = tmp_path / "out", tmp_path / "fastdisk"
    out.mkdir()
    target.mkdir()
    for name in ("OtherProject_Slide1.ome.tif", "Old.ome.tmp.raw"):
        (target / name).write_bytes(b"real image")
    age(*target.iterdir(), seconds=3600)
    (out / PARTIAL_DIR).symlink_to(target, target_is_directory=True)
    idx = preflight.list_output_dir(str(out), ROOT_FS, planned=[])
    assert idx.pdir_problem == preflight.PDIR_LINK_TEXT and idx.exists and idx.error is None
    assert not idx.partials and not idx.ready and idx.leftovers == ()
    paths = [str(out / PARTIAL_DIR / n) for n in ("OtherProject_Slide1.ome.tif", "Old.ome.tmp.raw")]
    n, freed, errors = delete_confirmed(paths)
    assert n == 0 and len(errors) == 2 and all("link" in e for e in errors)
    assert sorted(os.listdir(target)) == ["Old.ome.tmp.raw", "OtherProject_Slide1.ome.tif"]
    (out / PARTIAL_DIR).unlink()
    (out / PARTIAL_DIR).write_bytes(b"")
    assert preflight.list_output_dir(str(out), ROOT_FS).pdir_problem == preflight.PDIR_FILE_TEXT


def test_output_index_when_the_partial_folder_vanishes_meanwhile(tmp_path, monkeypatch):
    out, real_scandir = make_out(tmp_path), os.scandir

    def scandir(path):
        if str(path).endswith(PARTIAL_DIR):
            raise FileNotFoundError(2, "No such file or directory")
        return real_scandir(path)
    monkeypatch.setattr(preflight.os, "scandir", scandir)
    idx = preflight.list_output_dir(str(out), ROOT_FS)
    assert idx.exists and idx.error is None and not idx.partials and "a.ome.tif" in idx.outputs


@pytest.mark.skipif(AS_ROOT, reason="root ignores permissions")
def test_output_index_of_a_locked_folder(tmp_path):
    (tmp_path / "locked").mkdir(mode=0o000)
    try:
        idx = preflight.list_output_dir(str(tmp_path / "locked"), ROOT_FS)
    finally:
        (tmp_path / "locked").chmod(0o755)
    assert idx.error is not None and idx.error.code == "no_access"


# --------------------------------------------------------------------------------------------- write_probe
def test_write_probe_creates_the_folder_and_leaves_nothing(tmp_path):
    target = tmp_path / "a" / "b"
    preflight.write_probe(str(target))
    assert target.is_dir() and os.listdir(target) == []


def test_write_probe_without_create_never_makes_the_folder(tmp_path):
    """DS-8: the runner's per-slide check must fail for a vanished save folder, not re-create it."""
    with pytest.raises(FileNotFoundError):
        preflight.write_probe(str(tmp_path / "gone"), create=False)
    assert not (tmp_path / "gone").exists()


def test_write_probe_failures_raise_oserror(tmp_path):
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(OSError):
        preflight.write_probe(str(tmp_path / "file"))
    if not AS_ROOT:
        (tmp_path / "ro").mkdir(mode=0o555)
        try:
            with pytest.raises(PermissionError):
                preflight.write_probe(str(tmp_path / "ro"))
        finally:
            (tmp_path / "ro").chmod(0o755)


def test_write_probe_rides_out_a_brief_lock(tmp_path, monkeypatch):
    """FS-17: a create that fails once is retried under a new name; a delete that keeps failing (a virus scanner
    holds the new file) is no failure: the file is removed at the next probe."""
    real_open, real_remove, opened, removals = os.open, os.remove, [], []

    def flaky_open(path, *a, **k):
        if ".yfs-write-test-" in str(path):
            opened.append(path)
            if len(opened) == 1:
                raise PermissionError(errno.EACCES, "sharing violation")
        return real_open(path, *a, **k)

    def locked_remove(path, *a, **k):                # the scanner holds the written probe for a while
        if len(opened) == 2 and path == opened[1]:
            removals.append(path)
            if len(removals) <= preflight.PROBE_TRIES:
                raise PermissionError(errno.EACCES, "in use")
        return real_remove(path, *a, **k)
    monkeypatch.setattr(preflight.os, "open", flaky_open)
    monkeypatch.setattr(preflight.os, "remove", locked_remove)
    preflight.write_probe(str(tmp_path))
    assert len(opened) == 2 and opened[0] != opened[1]                  # a unique name for every attempt
    left = [n for n in os.listdir(tmp_path) if n.startswith(".yfs-write-test-")]
    assert len(left) == 1
    preflight.write_probe(str(tmp_path))
    assert os.listdir(tmp_path) == []


# ----------------------------------------------------------------------------------------- partial ownership
def test_partial_states(tmp_path, live_process, dead_process, monkeypatch):
    """D3: complete beats everything; an owner on this computer decides by its process; otherwise the age by the
    folder's clock does (another computer's owner, or none: v1 and older v2)."""
    p, now = tmp_path / "S.ome.tif", time.time()
    p.write_bytes(b"x")
    state = lambda mtime=now - 10: preflight.partial_state(str(p), mtime, now)   # noqa: E731
    assert state() == "live" and state(now - 121) == "stale"
    pid, started = live_process
    own(p, pid, started)
    assert state(now - 3600) == "live"
    own(p, pid, started - 50)                                            # its pid, but a later process
    assert state() == "dead"
    own(p, *dead_process)
    assert state() == "dead"                                             # Quit now, restart at once (CT-9)
    own(p, pid, started, host="otherhost.example.org")
    assert state() == "live" and state(now - 121) == "live"             # its owner record is fresh: by the age rule
    age(Path(str(p) + OWNER_SUFFIX), seconds=121)
    assert state() == "live" and state(now - 121) == "stale"
    own(p, 4242, 1.0, boot=preflight._BOOT - 86400)                     # a namesake computer (or before a restart)
    age(Path(str(p) + OWNER_SUFFIX), seconds=121)
    assert state() == "live" and state(now - 121) == "stale"
    token = preflight.write_owner(str(tmp_path / "T.ome.tif"))          # this process's own claim
    assert preflight.partial_state(str(tmp_path / "T.ome.tif"), now - 3600, now) == "live"
    preflight.release_owner(token)
    assert preflight.partial_state(str(tmp_path / "T.ome.tif"), now - 3600, now) == "dead"
    (tmp_path / ("S.ome.tif" + COMPLETE_SUFFIX)).write_text("{}")
    assert state(now - 3600) == "complete"
    real_stat = os.stat
    monkeypatch.setattr(preflight.os, "stat", lambda path, *a, **k: (_ for _ in ()).throw(
        OSError(errno.EIO, "Input/output error")) if str(path).endswith(COMPLETE_SUFFIX) else real_stat(path, *a, **k))
    assert state(now - 3600) == "live"                                   # can't tell: never taken over


def test_the_owner_record_is_created_exclusively(tmp_path):
    token = preflight.write_owner(str(tmp_path / "S.ome.tif"))
    try:
        record = preflight.read_owner(str(tmp_path / "S.ome.tif"))
        assert record["token"] == token and record["pid"] == os.getpid() and record["host"] == preflight.HOST
        with pytest.raises(FileExistsError):
            preflight.write_owner(str(tmp_path / "S.ome.tif"))
    finally:
        preflight.release_owner(token)
    (tmp_path / ("X.ome.tif" + OWNER_SUFFIX)).write_text('{"pid": ')   # half-written: no owner to ask
    assert preflight.read_owner(str(tmp_path / "X.ome.tif")) is None


def test_an_owner_record_that_cannot_be_read(tmp_path, monkeypatch):
    """read_owner reads an I/O error as 'no owner to ask' (the age rule decides); owner_token, which the runner uses to
    tell whether its claim is still its own, raises it instead: a share hiccup is not a takeover."""
    p = tmp_path / "S.ome.tif"
    own(p, 4242, 1.0, token="theirs")
    assert preflight.owner_token(str(p)) == "theirs" and preflight.owner_token(str(tmp_path / "X.ome.tif")) is None

    def hiccup(path, *a, **k):
        raise OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(preflight, "open", hiccup, raising=False)
    assert preflight.read_owner(str(p)) is None
    with pytest.raises(OSError):
        preflight.owner_token(str(p))


def test_the_share_clock_comes_from_a_probe_file(tmp_path, monkeypatch):
    """DS-3: staleness is judged by the clock that stamps the folder's files; this computer's only as a fallback."""
    real_stat = os.stat
    monkeypatch.setattr(preflight.os, "stat", lambda path, *a, **k: os.stat_result(
        (0o100644, 1, 1, 1, 0, 0, 0, 1000, 1000, 1000)) if ".yfs-clock-" in str(path) else real_stat(path, *a, **k))
    assert preflight.share_now(str(tmp_path)) == 1000 and os.listdir(tmp_path) == []
    assert abs(preflight.share_now(str(tmp_path / "missing")) - time.time()) < 5


def test_a_clock_probe_that_cannot_be_deleted_yet_is_deleted_later(tmp_path, monkeypatch):
    """FS-17 for share_now, which list_output_dir calls for every index of a save folder with partials: a virus
    scanner that holds the new probe file keeps its delete from working. The next write_probe deletes it, as it
    does its own probe files, so it doesn't stay in the partial folder for good."""
    real_remove = os.remove

    def held(path, *a, **k):
        if ".yfs-clock-" in str(path):
            raise PermissionError(errno.EACCES, "in use")
        return real_remove(path, *a, **k)
    monkeypatch.setattr(preflight.os, "remove", held)
    preflight.share_now(str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1
    monkeypatch.setattr(preflight.os, "remove", real_remove)            # the scanner lets go
    preflight.write_probe(str(tmp_path))
    assert os.listdir(tmp_path) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux moves create_time() with the wall clock")
@pytest.mark.parametrize("step", [3, -2, 60])
def test_a_step_of_the_wall_clock_never_makes_a_live_owner_dead(tmp_path, live_process, dead_process, monkeypatch,
                                                                step):
    """DS-3: psutil computes a Linux process's create_time() from the boot time as the wall clock shows it now
    (/proc/stat btime), so an NTP step moves a live owner's start time away from its record. It must stay live
    (taking its partial over would mix two writers' bytes), and a dead or reused pid must stay dead."""
    linux = pytest.importorskip("psutil._pslinux")
    p, now = tmp_path / "S.ome.tif", time.time()
    p.write_bytes(b"")
    pid, started = live_process
    real_boot_time = linux.boot_time
    monkeypatch.setattr(linux, "boot_time", lambda: real_boot_time() + step)
    for owner, state in (((pid, started), "live"), (dead_process, "dead"), ((pid, started - 50), "dead")):
        own(p, *owner)
        assert preflight.partial_state(str(p), now, now) == state


@pytest.mark.skipif(preflight.WINDOWS, reason="no hidepid; os.kill would end the process")
def test_an_owner_hidden_by_hidepid_is_judged_by_age(tmp_path, dead_process, monkeypatch):
    """/proc mounted with hidepid=2: psutil can't see another user's process (NoSuchProcess), yet it exists (signal 0
    is refused, not answered with ESRCH). Its owner can't be told dead, so the age rule decides."""
    def hidden(pid):
        raise psutil.NoSuchProcess(pid)
    p, now = tmp_path / "S.ome.tif", time.time()
    p.write_bytes(b"")
    monkeypatch.setattr(psutil, "Process", hidden)
    own(p, 1, 1.0)                                                       # pid 1: init, root's (or ours in a container)
    age(Path(str(p) + OWNER_SUFFIX), seconds=600)
    assert preflight.partial_state(str(p), now - 10, now) == "live"
    assert preflight.partial_state(str(p), now - 600, now) == "stale"
    own(p, *dead_process)                                                # really gone: dead, taken over at once
    assert preflight.partial_state(str(p), now - 600, now) == "dead"


def test_the_heartbeat_keeps_another_computers_claim_live_by_the_shares_clock(tmp_path):
    """DS-3 on SMB: the writer's os.utime() stamps its partial with the writer's clock, here 150 s behind the share's.
    The heartbeat also rewrites the owner record in place, a data write the server stamps with its own clock, and
    the newer of the two decides. The record keeps its content, and is never created by a heartbeat."""
    p = tmp_path / "S.ome.tif"
    p.write_bytes(b"")
    age(p, seconds=150)
    record = own(p, 4242, 1.0, host="otherhost.example.org")
    owner = Path(str(p) + OWNER_SUFFIX)
    age(owner, seconds=100)
    preflight.touch_owner(str(p))
    assert time.time() - owner.stat().st_mtime < 5 and json.loads(owner.read_text()) == record
    assert preflight.partial_state(str(p), p.stat().st_mtime, time.time()) == "live"
    age(owner, seconds=121)                                              # no heartbeat since: the writer crashed
    assert preflight.partial_state(str(p), p.stat().st_mtime, time.time()) == "stale"
    with pytest.raises(FileNotFoundError):
        preflight.touch_owner(str(tmp_path / "X.ome.tif"))
    assert sorted(os.listdir(tmp_path)) == ["S.ome.tif", "S.ome.tif" + OWNER_SUFFIX]


# ---------------------------------------------------------------------------------------- delete_leftovers
def test_delete_leftovers_respects_the_age_rule_and_the_kinds(tmp_path):
    out = make_out(tmp_path)
    age(out / "X.ome.tmp.raw", out / PARTIAL_DIR / "C.ome.tif", out / PARTIAL_DIR / "C.ome.tmp.raw", out / "A.ome.tif")
    paths = [str(out / "X.ome.tmp.raw"), str(out / PARTIAL_DIR / "C.ome.tif"), str(out / PARTIAL_DIR / "C.ome.tmp.raw"),
             str(out / PARTIAL_DIR / "D.ome.tif"), str(out / "A.ome.tif"), str(out / "gone.tmp.raw")]
    n, freed, errors = delete_confirmed(paths)
    assert (n, freed) == (3, 30 + 40 + 50)
    assert len(errors) == 2 and errors[0].startswith("D.ome.tif:") and errors[1].startswith("A.ome.tif:")
    assert (out / "A.ome.tif").exists() and (out / PARTIAL_DIR / "D.ome.tif").exists()
    age(out / PARTIAL_DIR / "D.ome.tif")
    assert delete_confirmed([str(out / PARTIAL_DIR / "D.ome.tif")]) == (1, 60, [])
    assert not (out / PARTIAL_DIR).exists()      # emptied, so removed


def test_delete_keeps_a_live_runs_scratch_file(tmp_path, live_process):
    """FS-13: another instance's disk-mode canvas is not touched while its partial is heartbeated or its owner runs,
    even though the canvas itself is old; a stale pair goes, with the partial's sidecars."""
    pdir = tmp_path / "out" / PARTIAL_DIR
    pdir.mkdir(parents=True)
    for stem in ("Live", "Owned", "Old"):
        (pdir / f"{stem}.ome.tif").write_bytes(b"partial")
        (pdir / f"{stem}.ome.tmp.raw").write_bytes(b"canvas")
    own(pdir / "Owned.ome.tif", *live_process)
    own(pdir / "Old.ome.tif", *live_process, host="otherhost.example.org")
    age(*[p for p in pdir.iterdir() if p.name != "Live.ome.tif"], seconds=600)
    idx = preflight.list_output_dir(str(pdir.parent), ROOT_FS, planned=[])
    assert sorted(os.path.basename(p) for p, _ in idx.leftovers) == ["Old.ome.tif", "Old.ome.tmp.raw"]
    n, _, errors = delete_confirmed([str(p) for p in pdir.iterdir() if OWNER_SUFFIX not in p.name])
    assert n == 2 and sorted(os.listdir(pdir)) == ["Live.ome.tif", "Live.ome.tmp.raw", "Owned.ome.tif",
                                                   "Owned.ome.tif.owner", "Owned.ome.tmp.raw"]
    assert sum("another stitcher" in e for e in errors) == 3 and sum("changed in the last" in e for e in errors) == 1


def test_delete_keeps_a_complete_partial_and_judges_age_by_the_share_clock(tmp_path, monkeypatch):
    pdir = tmp_path / "out" / PARTIAL_DIR
    pdir.mkdir(parents=True)
    (pdir / "Done.ome.tif").write_bytes(b"finished image")
    (pdir / ("Done.ome.tif" + COMPLETE_SUFFIX)).write_text('{"gaps": 0}')
    (pdir / "Skew.ome.tif").write_bytes(b"partial")
    age(pdir / "Done.ome.tif", seconds=3600)
    age(pdir / "Skew.ome.tif", seconds=300)              # the share's clock is 5 min behind: it is fresh there
    monkeypatch.setattr(preflight, "share_now", lambda folder: time.time() - 300)
    n, _, errors = delete_confirmed([str(pdir / "Done.ome.tif"), str(pdir / "Skew.ome.tif")])
    assert n == 0 and "waiting to be moved into place" in errors[0] and "changed in the last" in errors[1]


# ------------------------------------------------------------------------------------------ finalize_partial
@pytest.fixture
def partial(tmp_path):
    (tmp_path / PARTIAL_DIR).mkdir()
    p = tmp_path / PARTIAL_DIR / "S.ome.tif"
    p.write_bytes(b"new")
    (tmp_path / "S.ome.tif").write_bytes(b"old")
    return p


def flaky_replace(monkeypatch, failures, error=PermissionError(13, "Permission denied")):
    calls, real = [], os.replace

    def replace(a, b):
        calls.append((a, b))
        if len(calls) <= failures:
            raise error
        return real(a, b)
    monkeypatch.setattr(preflight.os, "replace", replace)
    monkeypatch.setattr(preflight, "RETRY_DELAY_S", 0.0)
    return calls


def test_finalize_replaces_the_old_file(partial):
    Path(str(partial) + OWNER_SUFFIX).write_text("{}")
    preflight.finalize_partial(str(partial), str(partial.parent.parent / "S.ome.tif"))
    assert (partial.parent.parent / "S.ome.tif").read_bytes() == b"new" and not partial.exists()
    assert os.listdir(partial.parent) == []                          # its sidecars went with it


def test_finalize_retries_permission_errors(partial, monkeypatch):
    calls = flaky_replace(monkeypatch, 2)
    preflight.finalize_partial(str(partial), str(partial.parent.parent / "S.ome.tif"), retries=5)
    assert len(calls) == 3 and (partial.parent.parent / "S.ome.tif").read_bytes() == b"new"


def test_finalize_gives_up_after_the_retries_and_the_partial_stays_complete(partial, monkeypatch):
    calls = flaky_replace(monkeypatch, 99)
    with pytest.raises(PermissionError):
        preflight.finalize_partial(str(partial), str(partial.parent.parent / "S.ome.tif"), retries=5, gaps=2)
    assert len(calls) == 5 and partial.read_bytes() == b"new"
    assert preflight.complete_gaps(str(partial)) == 2                # DS-9: remembered on disk, with its gaps
    assert preflight.list_output_dir(str(partial.parent.parent), ROOT_FS, []).ready.keys() == {"s.ome.tif"}


def test_finalize_does_not_retry_other_errors(partial, monkeypatch):
    calls = flaky_replace(monkeypatch, 99, OSError(5, "Input/output error"))
    with pytest.raises(OSError):
        preflight.finalize_partial(str(partial), str(partial.parent.parent / "S.ome.tif"))
    assert len(calls) == 1


@pytest.mark.parametrize("data, detail", [
    (b"II*\x00\x00\x00\x00\x00", "it holds no image"),
    (b"not an image", "it is not a TIFF image"),
    (b"II*\x00\x08\x00\x00\x00\xff\xff", "its image structure is damaged"),
])
def test_unreadable_outputs_have_plain_reasons(tmp_path, data, detail):
    path = tmp_path / "Slide1.ome.tif"
    path.write_bytes(data)
    assert preflight.verify_output(str(path)) == ("unreadable", detail)


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_a_damaged_finished_image_is_unfinished_on_rescan(tmp_path, data):
    pdir = tmp_path / PARTIAL_DIR
    pdir.mkdir()
    partial, final = pdir / "Slide1.ome.tif", tmp_path / "Slide1.ome.tif"
    partial.write_bytes(data)
    final.write_bytes(b"keep the existing image")
    complete = Path(str(partial) + COMPLETE_SUFFIX)
    complete.write_text('{"gaps": 0}')
    with pytest.raises(ValueError, match="damaged"):
        preflight.finalize_kept(str(partial), str(final))
    assert not complete.exists()
    assert partial.read_bytes() == data and final.read_bytes() == b"keep the existing image"
    idx = preflight.list_output_dir(str(tmp_path), ROOT_FS)
    assert not idx.ready and "slide1.ome.tif" in idx.partials


def test_finalize_kept_moves_a_complete_valid_image(stitched, tmp_path, monkeypatch):
    """D12: the UI's Finalize action. It needs the COMPLETE sidecar and a valid image; a claim that is not ours
    (DS-3: a 0-byte claim of another stitcher) or a damaged file is never moved, with a plain reason."""
    pdir, final = tmp_path / PARTIAL_DIR, tmp_path / "S.ome.tif"
    pdir.mkdir()
    final.write_bytes(b"the old file")
    (pdir / "S.ome.tif").write_bytes(b"")                            # another stitcher's fresh claim
    with pytest.raises(ValueError, match="isn't waiting to be moved into place"):
        preflight.finalize_kept(str(pdir / "S.ome.tif"), str(final))
    (pdir / ("S.ome.tif" + COMPLETE_SUFFIX)).write_text('{"gaps": 0}')
    with pytest.raises(ValueError, match="damaged"):
        preflight.finalize_kept(str(pdir / "S.ome.tif"), str(final))
    assert final.read_bytes() == b"the old file"
    shutil.copyfile(stitched["v2"], pdir / "S.ome.tif")
    (pdir / ("S.ome.tif" + COMPLETE_SUFFIX)).write_text('{"gaps": 0}')
    flaky_replace(monkeypatch, 99)
    with pytest.raises(PermissionError):                             # QuPath still has the old file open
        preflight.finalize_kept(str(pdir / "S.ome.tif"), str(final), retries=2)
    monkeypatch.undo()
    monkeypatch.setattr(preflight, "verify_output", lambda p: ("unchecked", "The disk reported a read error"))
    with pytest.raises(preflight.NotChecked, match="couldn't be checked"):     # a read error: Try again may help
        preflight.finalize_kept(str(pdir / "S.ome.tif"), str(final))
    monkeypatch.undo()
    size, mtime_ns = preflight.finalize_kept(str(pdir / "S.ome.tif"), str(final))
    assert (size, mtime_ns) == (os.stat(final).st_size, os.stat(final).st_mtime_ns) and size > 1000
    assert not pdir.exists()                                         # sidecars removed, empty folder removed



def delete_confirmed(paths, **kwargs):
    """Capture the same immutable snapshots the output index gives the confirmation dialog."""
    approved = []
    for path in paths:
        try:
            st = os.lstat(path)
            info = OutputInfo(os.path.basename(path), st.st_size, st.st_mtime_ns, (st.st_dev, st.st_ino),
                              preflight.path_id(os.path.dirname(path)))
        except FileNotFoundError:
            info = OutputInfo(os.path.basename(path), 0, 0, (-1, -1), preflight.path_id(os.path.dirname(path)))
        approved.append((path, info))
    return preflight.delete_leftovers(approved, **kwargs)

@pytest.mark.parametrize("damage", [None, "data", "plane", "subifd"])
def test_verifies_later_ome_planes_without_reading_pixels(tmp_path, monkeypatch, damage):
    path = tmp_path / "Slide1.ome.tif"
    data = np.zeros((2, 64, 64), np.uint16)
    if damage == "subifd":
        with tifffile.TiffWriter(path, ome=True) as writer:
            writer.write(data, photometric="minisblack", metadata={"axes": "CYX"}, subifds=1)
            writer.write(data[:, ::2, ::2], photometric="minisblack", subfiletype=1)
    else:
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)
    with tifffile.TiffFile(path) as tif:
        if damage in ("data", "subifd"):
            page = tif.pages[1].pages[0] if damage == "subifd" else tif.pages[1]
            cut = page.dataoffsets[0] + 100
        elif damage == "plane":
            page = tif.pages[0]
            next_ifd = page.offset + 2 + len(page.tags) * 12
    if damage in ("data", "subifd"):
        with open(path, "r+b") as stream:
            stream.truncate(cut)
    elif damage == "plane":
        with open(path, "r+b") as stream:
            stream.seek(next_ifd)
            stream.write(b"\0" * 4)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *a, **k: pytest.fail("verification read pixels"))
    verdict, detail = preflight.verify_output(str(path))
    assert verdict == ("valid" if damage is None else "incomplete"), detail


@pytest.mark.parametrize("swap", ["directory", "file", "size", "mtime"])
def test_leftover_confirmation_is_bound_to_indexed_objects(tmp_path, swap):
    out = tmp_path / "out"
    pdir = out / PARTIAL_DIR
    pdir.mkdir(parents=True)
    path = pdir / "Slide1.ome.tif"
    path.write_bytes(b"approved old file")
    age(path, seconds=600)
    idx = preflight.list_output_dir(str(out), ROOT_FS, planned=[])
    assert len(idx.leftovers) == 1
    if swap == "directory":
        out.rename(tmp_path / "original")
        pdir.mkdir(parents=True)
        path.write_bytes(b"unapproved file")
        age(path, seconds=600)
    elif swap == "file":
        replacement = pdir / "new"
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
        os.replace(replacement, path)
    elif swap == "size":
        path.write_bytes(b"changed contents and length")
        age(path, seconds=600)
    else:
        age(path, seconds=700)
    before = path.read_bytes()
    deleted, _, errors = preflight.delete_leftovers(idx.leftovers, out_dir=str(out), out_identity=idx.identity)
    assert deleted == 0 and errors and path.read_bytes() == before


def test_leftover_plain_paths_are_not_confirmation_snapshots(tmp_path):
    path = tmp_path / "Slide1.ome.tmp.raw"
    path.write_bytes(b"never approved")
    age(path)
    count, _, errors = preflight.delete_leftovers([str(path)])
    assert count == 0 and "refresh" in errors[0] and path.exists()


def test_swallowed_later_ifd_io_error_is_unchecked(tmp_path, monkeypatch):
    path = tmp_path / "Slide1.ome.tif"
    tifffile.imwrite(path, np.zeros((2, 128, 128), np.uint16), photometric="minisblack", ome=True,
                     metadata={"axes": "CYX"})
    with tifffile.TiffFile(path) as tif:
        later = tif.pages[1].offset
    original = preflight._ReadErrors
    class FailingFile(original):
        def readinto(self, buffer):
            if self.tell() >= later:
                self.error = OSError(errno.EIO, "Input/output error")
                raise self.error
            return super().readinto(buffer)
    monkeypatch.setattr(preflight, "_ReadErrors", FailingFile)
    assert preflight.verify_output(str(path)) == ("unchecked", "The disk or share reported a read error")


def test_finalize_preserves_a_changed_partial_and_its_owner(tmp_path, monkeypatch):
    pdir = tmp_path / PARTIAL_DIR
    pdir.mkdir()
    partial, final = pdir / "Slide1.ome.tif", tmp_path / "Slide1.ome.tif"
    tifffile.imwrite(partial, np.zeros((16, 32), np.uint16), metadata=None)
    final.write_bytes(b"previous good image")
    preflight.mark_complete(str(partial))
    verify = preflight.verify_output
    def changed(path):
        verdict = verify(path)
        replacement = pdir / "replacement"
        replacement.write_bytes(b"unverified replacement")
        os.replace(replacement, path)
        Path(path + OWNER_SUFFIX).write_text('{"token":"new-owner"}')
        return verdict
    monkeypatch.setattr(preflight, "verify_output", changed)
    with pytest.raises(ValueError, match="changed"):
        preflight.finalize_kept(str(partial), str(final))
    assert final.read_bytes() == b"previous good image" and partial.read_bytes() == b"unverified replacement"
    assert preflight.owner_token(str(partial)) == "new-owner"


def test_finalize_rechecks_confirmed_destination_after_verification(tmp_path, monkeypatch):
    pdir = tmp_path / PARTIAL_DIR
    pdir.mkdir()
    partial, final = pdir / "Slide1.ome.tif", tmp_path / "Slide1.ome.tif"
    tifffile.imwrite(partial, np.zeros((16, 32), np.uint16), metadata=None)
    final.write_bytes(b"approved old image")
    expected = preflight.destination_identity(str(final))
    preflight.mark_complete(str(partial))
    verify = preflight.verify_output
    def changed(path):
        verdict = verify(path)
        final.write_bytes(b"unapproved new destination")
        return verdict
    monkeypatch.setattr(preflight, "verify_output", changed)
    with pytest.raises(FileExistsError):
        preflight.finalize_kept(str(partial), str(final), expected_destination=expected)
    assert final.read_bytes() == b"unapproved new destination" and partial.exists()


def test_snapshot_deletes_custom_scratch_after_partial_folder_is_gone(tmp_path):
    out, temp = tmp_path / "out", tmp_path / "temp"
    out.mkdir()
    folder = Path(temp_subdir(str(temp), str(out)))
    folder.mkdir(parents=True)
    raw = folder / "Slide1.ome.tmp.raw"
    raw.write_bytes(b"abandoned canvas")
    age(raw, seconds=600)
    idx = preflight.list_output_dir(str(out), ROOT_FS, planned=[], temp_dir=str(temp))
    assert len(idx.leftovers) == 1
    assert preflight.delete_leftovers(idx.leftovers, out_dir=str(out), out_identity=idx.identity) == (1, 16, [])
    assert not raw.exists()
