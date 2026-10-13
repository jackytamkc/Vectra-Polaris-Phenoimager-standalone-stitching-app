"""Core end-to-end check (FINAL_SPEC §6.3 B6, §7.3): discovery -> plan -> preflight -> BatchRunner with the real
engine at NUM_THREADS = 1 on a synthetic tile tree, then every stitched component slide is compared byte for byte
with the v1 engine (tests/fixtures/legacy_engine_v1.py) run on the same tiles.

    python tools/core_e2e.py [SYNTH_ROOT [OUT_DIR]]

SYNTH_ROOT is generated with tests/fixtures/make_synth.py when it does not exist; both default to new temporary
folders. Also checked: .stitcher-in-progress/ is gone afterwards and the summary CSV has one line per slide.
Exit status 0 when every check passes. Synthetic data only; nothing outside OUT_DIR and a temp folder is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import psutil  # noqa: E402

import stitcher_RAM  # noqa: E402
from yfs import __version__  # noqa: E402
from yfs.core import preflight  # noqa: E402
from yfs.core.discovery import Discovery, DiscoveryOptions  # noqa: E402
from yfs.core.events import CandidateFound, RunFinished, ScanFinished, ScanNote  # noqa: E402
from yfs.core.fsservice import FsService  # noqa: E402
from yfs.core.names import PARTIAL_DIR  # noqa: E402
from yfs.core.places import MountTable  # noqa: E402
from yfs.core.plan import Plan  # noqa: E402
from yfs.core.runner import CSV_COLUMNS, BatchRunner, RunOptions  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Inbox:
    """A thread-safe `post` that keeps every event."""

    def __init__(self) -> None:
        self.events: List[object] = []
        self._cond = threading.Condition()

    def post(self, ev: object) -> None:
        with self._cond:
            self.events.append(ev)
            self._cond.notify_all()

    def wait(self, kind: type, timeout: float) -> object:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                hit = next((e for e in self.events if isinstance(e, kind)), None)
                if hit is not None:
                    return hit
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no {kind.__name__} within {timeout} s")
                self._cond.wait(0.5)


def build_plan(root: str, out_dir: str, inbox: Inbox) -> Plan:
    """What MainWindow does before Start: discovery, then headers and the output index through FS lanes."""
    mounts = MountTable.load()
    fs = FsService(inbox.post, mounts)
    try:
        Discovery(1, root, fs, inbox.post, DiscoveryOptions(out_dir=out_dir, budget_dirs=10 ** 6,
                                                            budget_s=3600.0)).start()
        inbox.wait(ScanFinished, 600)
        plan = Plan(root, out_dir)
        for ev in list(inbox.events):
            if isinstance(ev, CandidateFound):
                plan.add_candidate(ev.candidate)
            elif isinstance(ev, ScanNote):
                plan.add_note(ev.note)
        ram = psutil.virtual_memory().available
        for r in plan.rows():
            if r.section != "note" and r.cand.kind != "legacy_all":
                first = os.path.join(r.cand.folder, sorted(r.cand.tile_names)[0])
                job = partial(preflight.read_tile_header, first, stitcher_RAM.StitchingEngine)
                try:
                    plan.apply_header(r.row_id, fs.submit(first, "header", job).result(120), None, ram)
                except Exception as e:
                    plan.apply_header(r.row_id, None, str(e), ram)
        index = partial(preflight.list_output_dir, out_dir, mounts, plan.planned_output_keys())
        plan.apply_output_index(fs.submit(out_dir, "index_output", index).result(120))
    finally:
        fs.shutdown()
    return plan


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def v1_output(v1, folder: str, tile_names: Sequence[str], image_name: str, work: Path) -> str:
    """v1 on a clean folder holding only these tiles, named like the slide (v1 names the image after it)."""
    clean = work / f"tiles{len(os.listdir(work))}" / image_name
    clean.mkdir(parents=True)
    for n in tile_names:
        try:
            os.symlink(os.path.join(folder, n), clean / n)
        except OSError:                                  # no symlinks (Windows without privilege): copy
            shutil.copy2(os.path.join(folder, n), clean / n)
    out = str(clean.parent / (image_name + ".ome.tif"))
    eng = v1.StitchingEngine(str(clean), out)
    if not eng.scan_metadata():
        raise RuntimeError(f"v1 could not scan {image_name}")
    canvas = eng.stitch(use_ram=True)
    eng.write_to_disk(canvas, True)
    del canvas
    return out


def run_e2e(root: str, out_dir: str, log: Callable[[str], None] = print) -> Dict[str, object]:
    """Runs the whole check; returns {'ok': bool, 'slides': {output name: status}, 'problems': [...]}."""
    v1 = _load("legacy_engine_v1_e2e", REPO / "tests" / "fixtures" / "legacy_engine_v1.py")
    v1.NUM_THREADS = stitcher_RAM.NUM_THREADS = 1        # v1 pastes overlaps in timing order otherwise (§5.7)
    inbox, problems = Inbox(), []
    plan = build_plan(root, out_dir, inbox)
    components = [r.row_id for r in plan.rows() if r.section != "note" and r.cand.kind == "component"
                  and r.checkable]
    plan.select("none")
    plan.toggle(components, True)
    log(f"discovery: {len(components)} component slides; running the batch ...")
    runner = BatchRunner(1, plan, stitcher_RAM.StitchingEngine, inbox.post, RunOptions(out_dir,
                                                                                         app_version=__version__))
    runner.start()
    summary = inbox.wait(RunFinished, 1800).summary
    runner.join(10)
    slides = {}
    with tempfile.TemporaryDirectory(prefix="yfs-e2e-v1-") as work:
        for row_id, o in summary.rows:
            r = plan.row(row_id)
            name = r.output_stem + ".ome.tif"
            slides[name] = o.status
            if not o.status.startswith("done") or not o.output_path:
                problems.append(f"{name}: {o.message}")
                continue
            ref = v1_output(v1, r.cand.folder, r.cand.tile_names, r.image_name or r.cand.folder_name, Path(work))
            same = sha256(ref) == sha256(o.output_path)
            log(f"  {'identical' if same else 'DIFFERENT'}  {o.status:12s} {name}")
            if not same:
                problems.append(f"{name}: differs from the v1 output")
    if len(summary.rows) != len(components) or not components:
        problems.append(f"{len(summary.rows)} of {len(components)} slides were run")
    if os.path.exists(os.path.join(out_dir, PARTIAL_DIR)):
        problems.append(f"{PARTIAL_DIR}/ was left behind")
    problems += _csv_problems(summary.csv_path, len(components))
    for p in problems:
        log(f"PROBLEM: {p}")
    return {"ok": not problems, "slides": slides, "problems": problems, "csv": summary.csv_path}


def _csv_problems(path: Optional[str], n: int) -> List[str]:
    if not path or not os.path.exists(path):
        return ["no summary CSV was written"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or tuple(rows[0]) != CSV_COLUMNS:
        return ["the summary CSV has unexpected columns"]
    return [] if len(rows) - 1 == n else [f"the summary CSV has {len(rows) - 1} lines for {n} slides"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("synth_root", nargs="?", help="synthetic tile tree (generated if missing)")
    ap.add_argument("out_dir", nargs="?", help="folder for the stitched images (default: a new temp folder)")
    args = ap.parse_args(argv)
    root = args.synth_root or os.path.join(tempfile.mkdtemp(prefix="yfs-e2e-"), "synth")
    if not os.path.isdir(root):
        _load("make_synth_e2e", REPO / "tests" / "fixtures" / "make_synth.py").main(root)
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="yfs-e2e-out-")
    try:
        report = run_e2e(os.path.abspath(root), os.path.abspath(out_dir))
    finally:
        if not args.synth_root:                          # a temporary tree: unlock its "locked" folder, remove it
            if os.path.isdir(os.path.join(root, "locked")):
                os.chmod(os.path.join(root, "locked"), 0o755)
            shutil.rmtree(os.path.dirname(root), ignore_errors=True)
    print(("CORE E2E OK" if report["ok"] else "CORE E2E FAILED") + f": {len(report['slides'])} slides, "
          f"output in {out_dir}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
