"""Headless engine regression: stitch a folder with a given stitcher_RAM.py and verify/hash the output.

usage: python regress.py <dir containing stitcher_RAM.py> <tile folder> <output .ome.tif> [ram|disk] [tile files... via @listfile]
Prints JSON: md5, pixel check, OME XML, pyramid shapes.
"""
import hashlib
import importlib.util
import json
import logging
import os
import sys

import numpy as np
import tifffile

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def load_engine(src_dir):
    spec = importlib.util.spec_from_file_location("stitcher_under_test", os.path.join(src_dir, "stitcher_RAM.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, src_dir)
    spec.loader.exec_module(mod)
    return mod


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pixels(path, synthetic=True):
    with tifffile.TiffFile(path) as tif:
        lvl0 = tif.pages[0].asarray()          # (H, W, C) interleaved
        shapes = [tif.pages[0].shape] + [p.shape for p in (tif.pages[0].pages or [])]
        xml = tif.pages[0].description
    res = {"shape": list(lvl0.shape), "dtype": str(lvl0.dtype), "pyramid": [list(s) for s in shapes], "xml": xml}
    if synthetic:
        h, w, c = lvl0.shape
        # Canvas origin is the minimum tile position; values encode global coords (gy*4096+gx)*(c+1)
        base = lvl0[..., 0].astype(np.float64)
        gy0 = int(base[0, 0] // 4096)
        gx0 = int(base[0, 0] - gy0 * 4096)
        yy, xx = np.mgrid[0:h, 0:w]
        expect0 = ((yy + gy0) * 4096.0 + (xx + gx0))
        ok = True
        for ch in range(c):
            exp = (expect0 * (ch + 1)).astype(lvl0.dtype)
            covered = lvl0[..., 0] != 0   # gaps (no tile) are zeros
            if not np.array_equal(lvl0[..., ch][covered], exp[covered]):
                ok = False
        res["pixels_ok"] = ok
        res["covered_fraction"] = float((lvl0[..., 0] != 0).mean())
    return res


def main():
    src_dir, folder, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "ram"
    mod = load_engine(src_dir)
    if os.path.exists(out):
        os.remove(out)
    eng = mod.StitchingEngine(folder, out, None, None)
    assert eng.scan_metadata(), "scan failed"
    canvas = eng.stitch(use_ram=(mode == "ram"))
    assert canvas is not None, "stitch failed"
    eng.write_to_disk(canvas, mode == "ram")
    info = check_pixels(out, synthetic="synth" in folder)
    info["md5"] = md5(out)
    info["size"] = os.path.getsize(out)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
