"""Generate a synthetic Vectra/PhenoImager-like tile tree for testing the stitcher.

Pixel value of channel c at global pixel (gy, gx) = (gy * 4096 + gx) * (c + 1)  (float32),
so any correctly stitched canvas can be verified exactly, overlaps included.

    python tests/fixtures/make_synth.py [FOLDER] [--force]

FOLDER defaults to a new temporary folder. An existing folder is replaced only when it is empty or holds the
MARKER file of an earlier run; --force replaces any folder that is not a link.
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import tempfile
import numpy as np
import tifffile

RES_PX_PER_CM = 20000          # 0.5 um / px
TILE_W, TILE_H = 320, 256
STEP_X, STEP_Y = 300, 240      # 20 / 16 px overlap like real fields
CHANNELS = ["DAPI", "Opal 520", "Opal 570", "Opal 690", "Autofluorescence"]
MARKER = ".yfs-synth"          # written into every tree made here; only such a tree is replaced without --force


def pe_xml(name, image_type="FullResolution", slide_id="x"):
    return (
        '<?xml version="1.0" encoding="utf-16"?>\n'
        "<PerkinElmer-QPI-ImageDescription>\n"
        "  <DescriptionVersion>3</DescriptionVersion>\n"
        "  <AcquisitionSoftware>VectraPolaris</AcquisitionSoftware>\n"
        f"  <ImageType>{image_type}</ImageType>\n"
        f"  <SlideID>{slide_id}</SlideID>\n"
        "  <IsUnmixedComponent>True</IsUnmixedComponent>\n"
        + (f"  <Name>{name}</Name>\n" if name else "")
        + "</PerkinElmer-QPI-ImageDescription>"
    )


def write_tile(path, gx, gy, channels=CHANNELS, w=TILE_W, h=TILE_H, dtype=np.float32):
    yy, xx = np.mgrid[gy:gy + h, gx:gx + w]
    base = (yy.astype(np.float64) * 4096 + xx).astype(np.float64)
    # Positions are stored in cm (rational) such that x_res * x_pos == pixel offset.
    pos_x = (int(gx), RES_PX_PER_CM)
    pos_y = (int(gy), RES_PX_PER_CM)
    extratags = [(286, 5, 1, pos_x, True), (287, 5, 1, pos_y, True)]
    with tifffile.TiffWriter(path) as tif:
        for c, name in enumerate(channels):
            plane = (base * (c + 1)).astype(dtype)
            tif.write(plane, photometric="minisblack", compression="lzw",
                      resolution=(RES_PX_PER_CM, RES_PX_PER_CM), resolutionunit="CENTIMETER",
                      description=pe_xml(name, slide_id=os.path.basename(path)),
                      extratags=extratags, metadata=None)
        thumb = np.zeros((h // 4, w // 4, 3), np.uint8)
        tif.write(thumb, photometric="rgb", compression="lzw",
                  resolution=(RES_PX_PER_CM // 4, RES_PX_PER_CM // 4), resolutionunit="CENTIMETER",
                  description=pe_xml(None, "Thumbnail"), extratags=extratags, metadata=None)


def make_slide(folder, slide, nx, ny, origin=(10000, 30000), suffix="component_data", channels=CHANNELS):
    os.makedirs(folder, exist_ok=True)
    ox, oy = origin
    for j in range(ny):
        for i in range(nx):
            gx, gy = ox + i * STEP_X, oy + j * STEP_Y
            # Name uses tile-centre microns like real exports
            cx_um = int(round((gx + TILE_W / 2) * 10000 / RES_PX_PER_CM))
            cy_um = int(round((gy + TILE_H / 2) * 10000 / RES_PX_PER_CM))
            name = f"{slide}_[{cx_um},{cy_um}]_{suffix}.tif"
            write_tile(os.path.join(folder, name), gx, gy, channels=channels)


def plain_tif(path, shape=(64, 64)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.zeros(shape, np.uint8))


def prepare_root(root, force=False):
    """Make `root` an empty folder. A non-empty folder is emptied only when it holds MARKER (a tree made here) or
    `force` is set, and a link never; otherwise FileExistsError is raised and nothing is touched."""
    if os.path.islink(root):
        raise FileExistsError(f"{root} is a link: give the folder itself")
    if os.path.isdir(root) and os.listdir(root):
        if not (force or os.path.isfile(os.path.join(root, MARKER))):
            raise FileExistsError(f"{root} is not empty and was not made by make_synth.py: choose a new folder, "
                                  f"or pass --force to replace it")
        for dp, dn, fn in os.walk(root):
            for d in dn:                     # unlock ("locked" is chmod 000) before os.walk lists it
                if not os.path.islink(os.path.join(dp, d)):
                    os.chmod(os.path.join(dp, d), 0o755)
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)


def main(root, force=False):
    root = os.path.abspath(root)                                 # the symlink loop below needs an absolute target
    prepare_root(root, force)
    P = lambda *a: os.path.join(root, *a)
    open(P(MARKER), "w").write("Synthetic test data made by tests/fixtures/make_synth.py; safe to delete.\n")

    # 1. Clean project: one folder per slide, plus an existing output in the parent (like a data-disk root)
    make_slide(P("ProjectA", "SlideA1_Scan1"), "SlideA1_Scan1", 3, 3)
    make_slide(P("ProjectA", "SlideA2_Scan1"), "SlideA2_Scan1", 2, 2)
    plain_tif(P("ProjectA", "SlideA1_Scan1.ome.tif"))      # pretend previous output
    open(P("ProjectA", "notes.txt"), "w").write("lab notes\n")

    # 2. inForm batch export: two slides in ONE folder + other per-tile products
    make_slide(P("inform_export"), "S1_Scan1", 2, 2)
    make_slide(P("inform_export"), "S2_Scan1", 3, 2, origin=(50000, 10000))
    make_slide(P("inform_export"), "S1_Scan1", 2, 2, suffix="binary_seg_maps", channels=["Nucleus"])
    open(P("inform_export", "S1_Scan1_[1,1]_cell_seg_data.txt"), "w").write("x\n")
    open(P("inform_export", "batch_procedure.ifp"), "w").write("x\n")

    # 3. Root with whole-slide *_stitch.tif from another tool + per-slide subfolders
    plain_tif(P("round1_unmix", "R1-01_Scan1_stitch.tif"), (512, 512))
    make_slide(P("round1_unmix", "R1-01_Scan1"), "R1-01_Scan1", 2, 2)
    make_slide(P("round1_unmix", "R1-02"), "R1-02_Scan1", 2, 1)     # folder name != slide prefix
    os.makedirs(P("round1_unmix", "ImageAnalysisPipeline"))
    open(P("round1_unmix", "Batch.log"), "w").write("log\n")

    # 4. Duplicate basenames in different batches -> output-name collision
    make_slide(P("duplicates", "batchA", "Slide9"), "Slide9_Scan1", 2, 1)
    make_slide(P("duplicates", "batchB", "Slide9"), "Slide9_Scan1", 1, 2)

    # 5. Suspicious channel names (all DAPI) -> channel dialog
    make_slide(P("suspicious", "SusSlide_Scan1"), "SusSlide_Scan1", 2, 1, channels=["DAPI"] * 5)

    # 6. Unrelated content: non-tile TIFFs and a deep tree
    for k in range(5):
        plain_tif(P("unrelated", "WSIs", f"case_{k:03d}.tif"))
    deep = P("unrelated", *[f"level{i}" for i in range(9)])
    os.makedirs(deep)
    make_slide(os.path.join(deep, "DeepSlide_Scan1"), "DeepSlide_Scan1", 1, 1)
    os.makedirs(P("unrelated", "big_flat"), exist_ok=True)   # a big flat folder of non-tile files
    for k in range(1500):
        open(P("unrelated", "big_flat", f"file_{k:05d}.txt"), "w").close()

    # 7. Edge cases
    os.makedirs(P("empty_dir"))
    os.makedirs(P(".hidden_dir"))
    make_slide(P("locked"), "Locked_Scan1", 1, 1)
    os.chmod(P("locked"), 0o000)
    os.symlink(root, P("unrelated", "loop_to_root"))           # symlink loop
    make_slide(P("Spaces & ünïcödé", "Slide [weird] (1)"), "Slide [weird] (1)", 1, 1)
    print("synthetic tree at", root)


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", help="where to write the tree (default: a new temporary folder)")
    ap.add_argument("--force", action="store_true", help="replace FOLDER even if make_synth.py did not make it")
    args = ap.parse_args(argv)
    try:
        main(args.folder or os.path.join(tempfile.mkdtemp(prefix="yfs-synth-"), "synth"), force=args.force)
    except FileExistsError as exc:
        print(f"make_synth.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli())
