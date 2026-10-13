"""BatchRunner (FINAL_SPEC §2.4.4-§2.4.7, §5.5.7; acceptance A4): the partial-file protocol, heartbeat, space
checks, stop modes, failure classification (R28), the summary CSV and the event order. Most tests use FakeEngine
on folders of empty files that carry tile names; the last ones use the real engine (and B6, tools/core_e2e.py)."""
from __future__ import annotations

import csv
import dataclasses
import errno
import json
import logging
import os
import re
import sys
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
import tifffile

import stitcher_RAM
from tests.conftest import FIXTURES, REPO, load_module
from tests.fakes import CONTIG_ERROR, ENOSPC_ERROR, FakeClock, FakeEngine
from yfs.core import osutil, preflight, runner
from yfs.core.events import (OutputWritten, RowBytes, RowFinished, RowProgress, RowStarted, RunFinished,
                             RunStarted)
from yfs.core.fsservice import scan_dir
from yfs.core.model import Job, RowOutcome, TileHeader
from yfs.core.names import (COMPLETE_SUFFIX, HEARTBEAT_S, LEVEL0_FACTOR, OWNER_SUFFIX, PARTIAL_DIR, STALE_HEX,
                            STALE_INFIX, candidates_from_summary, summarize_listing, temp_subdir)
from yfs.core.places import MountTable
from yfs.core.plan import Plan
from yfs.core.runner import (APPEARED_TEXT, BUSY_TEXT, CANCELLED_TEXT, CSV_COLUMNS, FINALIZE_TEXT, TAKEN_OVER_TEXT,
                             WAITING_TEXT, BatchRunner, RunOptions)

AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
ROOT_FS = MountTable.from_mountinfo_text("1 0 8:1 / / rw - ext4 /dev/sda1 rw")
RUN_EVENTS = (RunStarted, RowStarted, RowProgress, RowBytes, RowFinished, OutputWritten, RunFinished)
synth = load_module("yfs_test_make_synth_runner", FIXTURES / "make_synth.py")


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):
    monkeypatch.setattr(preflight, "PROBE_DELAY_S", 0.0)
    monkeypatch.setattr(preflight, "RETRY_DELAY_S", 0.0)


def age(*paths, seconds=200):
    for p in paths:
        os.utime(p, (time.time() - seconds, time.time() - seconds))


def own(partial, pid, started, host=None, token="t0") -> None:
    """An owner record for `partial`, as another stitcher's claim writes it (D3)."""
    record = {"host": host or preflight.HOST, "boot": preflight._BOOT, "pid": pid, "started": started,
              "token": token}
    Path(str(partial) + OWNER_SUFFIX).write_text(json.dumps(record))


def make_tiles(folder: Path, prefix: str, n: int = 12) -> Path:
    """n empty files named like component tiles (FakeEngine never opens them)."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_[{1000 + 160 * (i % 4)},{2000 + 128 * (i // 4)}]_component_data.tif").write_bytes(b"")
    return folder


def make_plan(src: Path, out: Path, slides=("SlideA_Scan1", "SlideB_Scan1", "SlideC_Scan1"), n: int = 12) -> Plan:
    plan = Plan(str(src), str(out))
    for name in slides:
        folder = make_tiles(src / name, name, n)
        summary = summarize_listing(str(folder), scan_dir(str(folder), cap=10000).entries)
        for c in candidates_from_summary(summary, root=str(src), depth=1, dir_mtime=None):
            plan.add_candidate(c)
    assert len(plan.ticked()) == len(slides)
    return plan


def give_estimate(plan: Plan) -> None:
    header = TileHeader("t.tif", (5, 256, 320), "float32", 5, 256, 320, 0.5, 0.5, True, ("a", "b", "c", "d", "e"),
                        None, (None,) * 5, 5 * 256 * 320 * 4)
    for r in plan.ticked():
        plan.apply_header(r.row_id, header, None, 64 << 30)
        assert plan.row(r.row_id).est is not None


@pytest.fixture
def src(tmp_path) -> Path:
    return tmp_path / "src"


@pytest.fixture
def out(tmp_path) -> Path:
    return tmp_path / "out"


def run(plan, engine, out, collector, runner_box=None, **opts):
    """Run a batch to the end; returns (RunSummary, outcomes by row_id)."""
    opts.setdefault("poll_s", 0.01)
    r = BatchRunner(1, plan, engine, collector, RunOptions(str(out), **opts))
    if runner_box is not None:
        runner_box.append(r)
    before = len(collector.of_type(RunFinished))
    r.start()
    summary = collector.wait_type(RunFinished, count=before + 1, timeout=60)[-1].summary
    r.join(5)
    assert not r.is_alive()
    return summary, dict(summary.rows)


def check_event_order(events):
    """RunStarted (RowStarted (RowProgress|RowBytes|OutputWritten)* RowFinished)* RunFinished."""
    evs = [e for e in events if isinstance(e, RUN_EVENTS)]
    assert isinstance(evs[0], RunStarted) and isinstance(evs[-1], RunFinished)
    current = None
    for e in evs[1:-1]:
        assert not isinstance(e, (RunStarted, RunFinished))
        if isinstance(e, RowStarted):
            assert current is None
            current = e.row_id
        elif isinstance(e, RowFinished):
            assert e.row_id == current
            current = None
        elif isinstance(e, (RowProgress, RowBytes)):
            assert e.row_id == current
        else:
            assert current is not None
    assert current is None
    return evs


# ---------------------------------------------------------------------------------------------- happy path
def test_batch_summary_and_csv_include_a_finalize_completed_during_the_run(src, out, collector, monkeypatch):
    plan = make_plan(src, out, slides=("SlideA_Scan1", "SlideB_Scan1"))
    first = plan.rows()[0].row_id

    def stitch(_runner, job):
        if job.row_id == first:
            return RowOutcome("finalize", "Close the old image, then Finalize")
        plan.finish(first, RowOutcome("done_gaps", "Done with gaps", failed_tiles=2))
        return RowOutcome("done", "Done")

    monkeypatch.setattr(BatchRunner, "_run_job", stitch)
    summary, outcomes = run(plan, FakeEngine, out, collector)
    assert summary.counts == {"done_gaps": 1, "done": 1}
    assert outcomes[first].failed_tiles == 2
    with open(summary.csv_path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "done_gaps" and rows[0]["failed_tiles"] == "2"


def test_a_clean_batch(src, out, collector):
    """Partial only in .stitcher-in-progress, final only after success, events, OutputWritten, RowBytes, CSV."""
    seen, texts = [], []

    def hook(eng, phase, step):
        seen.append((phase, eng.output_path.parent.name, eng.output_path.exists(),
                     (out / eng.output_path.name).exists()))
        texts.extend(r.run_text for r in plan.rows() if r.status == "running")
    Engine = FakeEngine.configure(on_step=hook, durations={"write": 0.3}, steps={"write": 4}, needed_gb=0.5)
    plan = make_plan(src, out)
    summary, outcomes = run(plan, Engine, out, collector, app_version="2.0.0")
    assert summary.counts == {"done": 3} and summary.stop_reason is None and not summary.stopped_by_user
    assert {p for _, p, _, _ in seen} == {PARTIAL_DIR}          # the engine only ever wrote into the partial dir
    assert all(partial_there and not final_there for _, _, partial_there, final_there in seen)
    assert not (out / PARTIAL_DIR).exists()
    assert "Reading tile positions…" in texts and "Stitching tiles 12/12" in texts
    assert any(re.fullmatch(r"Writing full-resolution image( · \d+ KB|…)", t) for t in texts)
    for row_id, o in outcomes.items():
        final = out / (plan.row(row_id).output_stem + ".ome.tif")
        assert o.output_path == str(final) and final.stat().st_size == o.output_bytes == 64 * 1024 + 3 * 512
        assert re.fullmatch(r"Done · 66 KB · \d+ s", o.message) and plan.row(row_id).status == "done"
        assert (o.tiles_used, o.failed_tiles, o.memory_mode, o.um_per_px) == (12, 0, "ram", 0.5)
        assert o.channels == ("DAPI", "Opal 520", "Opal 570", "Opal 690", "Autofluorescence")
    written = {e.path: (e.size, e.mtime_ns) for e in collector.of_type(OutputWritten)}
    assert written == {o.output_path: (os.stat(o.output_path).st_size, os.stat(o.output_path).st_mtime_ns)
                       for o in outcomes.values()}
    evs = check_event_order(collector.events)
    assert evs[0].n_jobs == 3 and len(collector.of_type(RowFinished)) == 3
    phases = [e.phase for e in collector.of_type(RowProgress) if e.row_id == evs[1].row_id]
    assert phases[0] == "reading" and phases[-1] == "finishing"
    assert {"stitching", "preparing", "writing", "pyramid"} <= set(phases)
    sizes = [e for e in collector.of_type(RowBytes)]
    assert sizes and all(e.expected == int(int(0.5 * 1024 ** 3) * LEVEL0_FACTOR) for e in sizes)
    assert all(0 <= e.written <= 64 * 1024 for e in sizes if e.phase == "writing")
    assert all(e.phase in ("writing", "pyramid") and e.written <= 64 * 1024 + 3 * 512 for e in sizes)
    # the summary table (§2.4.7)
    assert summary.csv_path and re.fullmatch(r"YourFavouriteStitcher_summary_\d{8}-\d{6}\.csv",
                                             os.path.basename(summary.csv_path))
    assert Path(summary.csv_path).read_bytes().startswith(b"\xef\xbb\xbf")             # utf-8-sig
    with open(summary.csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(CSV_COLUMNS) and len(rows) == 3
    first = rows[0]
    assert (first["slide"], first["status"], first["tiles"], first["memory_mode"], first["app_version"]) == \
        ("SlideA_Scan1", "done", "12", "ram", "2.0.0")
    assert first["tifffile_version"] == tifffile.__version__ and first["channels"].startswith("DAPI; Opal 520")
    assert first["output_bytes"] == str(64 * 1024 + 3 * 512) and first["um_per_px"] == "0.5"


def test_done_variants_and_resolver(src, out, collector):
    plan = make_plan(src, out, slides=("Gaps_Scan1",))
    _, outcomes = run(plan, FakeEngine.configure(failed_tiles=3), out, collector, summary_csv=False)
    assert [o.message for o in outcomes.values()] == ["Done with gaps: 3 tiles couldn't be read"]
    plan = make_plan(src, out / "2", slides=("Dapi_Scan1", "Named_Scan1"))
    named = [r.row_id for r in plan.rows() if r.cand.folder_name == "Named_Scan1"]
    plan.row(named[0]).channel_override = ("A", "B", "C", "D", "E")      # as set_channels would (R26)
    _, outcomes = run(plan, FakeEngine.configure(names=["DAPI"] * 5), out / "2", collector, summary_csv=False)
    generic, own = [outcomes[r.row_id] for r in plan.rows()]
    assert generic.status == "done_generic" and generic.message.startswith("Done · generic channel names")
    assert own.status == "done" and own.channels == ("A", "B", "C", "D", "E")


def test_disk_mode_uses_the_temp_folder_and_cleans_it(src, out, tmp_path, collector, monkeypatch):
    monkeypatch.setattr(runner, "free_bytes", lambda path: 1 << 62)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen = []
    Engine = FakeEngine.configure(needed_gb=1e6, on_step=lambda eng, phase, i: seen.append(eng.temp_path.exists()))
    _, outcomes = run(make_plan(src, out, slides=("Big_Scan1",)), Engine, out, collector, temp_dir=str(scratch))
    (o,) = outcomes.values()
    assert o.status == "done" and o.memory_mode == "disk" and any(seen)
    assert os.listdir(scratch) == [] and not (out / PARTIAL_DIR).exists()


def test_tile_changes_since_planning(src, out, collector):
    plan = make_plan(src, out, slides=("Grow_Scan1", "Gone_Scan1"))
    make_tiles(src / "Grow_Scan1", "Grow_Scan1", 13)
    for f in (src / "Gone_Scan1").iterdir():
        f.unlink()
    _, outcomes = run(plan, FakeEngine, out, collector, summary_csv=False)
    gone, grow = [outcomes[r.row_id] for r in plan.rows()]                  # display (= run) order
    assert grow.status == "done" and grow.message.endswith("(tile count changed since planning: 12 → 13)")
    assert grow.tiles_used == 13
    assert (gone.status, gone.message) == ("skipped", "Skipped: its tiles are no longer there")


def test_an_output_that_appeared_is_not_overwritten(src, out, collector):
    plan = make_plan(src, out, slides=("Twice_Scan1",))
    out.mkdir()
    (out / "Twice_Scan1.ome.tif").write_bytes(b"someone else's")
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: the output appeared since planning"]
    assert (out / "Twice_Scan1.ome.tif").read_bytes() == b"someone else's"


def test_an_unfinished_replacement_is_finished(src, out, collector):
    """A crash while replacing S.ome.tif left the old file and a stale partial. The plan shows the row as
    'unfinished' (ticked, Job.replace False): the runner must redo it, not skip it as 'appeared since planning'."""
    plan = make_plan(src, out, slides=("Again_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    (out / "Again_Scan1.ome.tif").write_bytes(b"old")
    (out / PARTIAL_DIR / "Again_Scan1.ome.tif").write_bytes(b"crashed")
    os.utime(out / PARTIAL_DIR / "Again_Scan1.ome.tif", (time.time() - 300, time.time() - 300))
    plan.apply_output_index(preflight.list_output_dir(str(out), ROOT_FS))
    assert plan.rows()[0].status == "unfinished"
    _, outcomes = run(plan, FakeEngine, out, collector, summary_csv=False)
    assert [o.status for o in outcomes.values()] == ["done"] and (out / "Again_Scan1.ome.tif").stat().st_size > 3


# ------------------------------------------------------------------------------------- claims and partials
def test_o_excl_allows_one_claim_only(tmp_path):
    ident = runner.claim_partial(str(tmp_path / "p.ome.tif"))
    st = os.stat(tmp_path / "p.ome.tif")
    assert ident == (st.st_dev, st.st_ino)                        # D3: what we claimed, for the check before finalize
    assert runner.claim_partial(str(tmp_path / "p.ome.tif")) is None


def test_a_claim_lost_to_another_stitcher_is_skipped(src, out, collector, monkeypatch):
    """Another instance creates the partial between our check and our claim: O_EXCL refuses ours."""
    plan = make_plan(src, out, slides=("Race_Scan1",))
    real_partials_of = runner._partials_of

    def partials_of(pdir, final_name):          # the check misses it ...
        found = real_partials_of(pdir, final_name)
        Path(pdir, final_name).write_bytes(b"theirs")   # ... because the other stitcher claims it just now
        return found
    monkeypatch.setattr(runner, "_partials_of", partials_of)
    Engine = FakeEngine.configure()
    _, outcomes = run(plan, Engine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: another stitcher may be writing this slide"]
    assert (out / PARTIAL_DIR / "Race_Scan1.ome.tif").read_bytes() == b"theirs" and Engine.calls == []


@pytest.mark.parametrize("age, taken_over", [(100, False), (121, True)])
def test_partials_younger_than_120_s_are_respected(src, out, collector, age, taken_over):
    plan = make_plan(src, out, slides=("Old_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    leftover, raw = out / PARTIAL_DIR / "Old_Scan1.ome.tif", out / PARTIAL_DIR / "Old_Scan1.ome.tmp.raw"
    for p in (leftover, raw):
        p.write_bytes(b"crashed")
        os.utime(p, (time.time() - age, time.time() - age))
    _, outcomes = run(plan, FakeEngine, out, collector, summary_csv=False)
    (o,) = outcomes.values()
    if taken_over:
        assert o.status == "done" and (out / "Old_Scan1.ome.tif").exists() and not (out / PARTIAL_DIR).exists()
    else:
        assert o.message == "Skipped: another stitcher may be writing this slide"
        assert leftover.read_bytes() == b"crashed" and raw.exists()


def test_heartbeat_keeps_a_long_claim_fresh_and_a_second_runner_skips_it(src, out, collector, monkeypatch):
    """A4/D3: a tile phase of 5 min (fake monotonic clock). The partial and the scratch file are touched at least
    every HEARTBEAT_S + 2 s, without explicit times (so a share stamps them with its own clock), and a second
    BatchRunner on the same output folder skips the slide; after a crash (no heartbeat for 121 s by the share's
    clock) a third one takes it over."""
    clock = FakeClock(0.0)
    monkeypatch.setattr(runner, "_monotonic", clock.time)
    monitors, touched, real_utime = [], [], os.utime

    class Monitor(runner.PartialMonitor):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            monitors.append(self)
    monkeypatch.setattr(runner, "PartialMonitor", Monitor)

    def utime(path, *a, **k):
        if PARTIAL_DIR in str(path):
            touched.append((clock.time(), os.path.basename(path), a, k))
        return real_utime(path, *a, **k)
    monkeypatch.setattr(os, "utime", utime)
    partial, second = out / PARTIAL_DIR / "Long_Scan1.ome.tif", []

    def step(eng, phase, i):
        if phase != "stitch":
            return
        deadline = time.monotonic() + 10
        while monitors[0].polled_at < clock.time():          # let the monitor see this moment
            assert time.monotonic() < deadline, "the partial monitor stopped polling"
            time.sleep(0.0005)
        if i == 200:                                           # a second app instance starts the same batch
            col2 = type(collector)()
            r2 = BatchRunner(2, make_plan(src, out, slides=("Long_Scan1",)), FakeEngine, col2,
                             RunOptions(str(out), summary_csv=False, poll_s=0.001))
            r2.start()
            second.append(col2.wait_type(RowFinished, timeout=30)[0].outcome)
            r2.join(5)
    Engine = FakeEngine.configure(clock=clock, durations={"stitch": 300.0}, steps={"stitch": 300}, on_step=step)
    _, outcomes = run(make_plan(src, out, slides=("Long_Scan1",)), Engine, out, collector, poll_s=0.001)
    assert [o.status for o in outcomes.values()] == ["done"]
    beats = [t for t, name, _, _ in touched if name == "Long_Scan1.ome.tif"]
    assert max(b - a for a, b in zip([0.0] + beats, beats + [300.0])) <= HEARTBEAT_S + 2
    assert {name for _, name, _, _ in touched} == {"Long_Scan1.ome.tif", "Long_Scan1.ome.tmp.raw"}   # FS-13
    assert all(a == () and k == {} for _, _, a, k in touched)
    assert second[0].message == "Skipped: another stitcher may be writing this slide"
    # a crash: the partial is left behind and nobody touches it for 121 s
    os.remove(out / "Long_Scan1.ome.tif")
    partial.parent.mkdir()
    partial.write_bytes(b"crashed")
    age(partial, seconds=121)
    col3 = type(collector)()
    _, outcomes = run(make_plan(src, out, slides=("Long_Scan1",)), FakeEngine, out, col3, summary_csv=False)
    assert [o.status for o in outcomes.values()] == ["done"] and (out / "Long_Scan1.ome.tif").stat().st_size > 7


def test_the_monitor_posts_nothing_once_stopped(tmp_path, monkeypatch):
    """A size poll stuck on a slow share must not post RowBytes after stop() (it would follow RowFinished)."""
    p = tmp_path / "p.ome.tif"
    p.write_bytes(b"x")
    gate, inside, real_getsize = threading.Event(), threading.Event(), os.path.getsize

    def slow_getsize(path):
        if str(path) == str(p):
            inside.set()
            gate.wait(10)
        return real_getsize(path)
    monkeypatch.setattr(runner.os.path, "getsize", slow_getsize)
    job = Job("r", 1, str(tmp_path), "component", "S", "component_data", "S", (), str(tmp_path), "p.ome.tif",
              False, None, None, None, None)
    posted = []
    mon = runner.PartialMonitor(str(p), job, posted.append, poll_s=0.001)
    mon.phase = "writing"
    mon.start()
    assert inside.wait(10)
    mon.stop()                                   # gives up waiting after 1 s: the poll is still stuck
    gate.set()
    mon.join(10)
    assert posted == [] and not mon.is_alive()


def test_every_heartbeat_rewrites_the_owner_record(tmp_path, monkeypatch):
    """D3: a heartbeat also rewrites the owner record in place, a data write that every file server stamps with its
    own clock (SMB stamps os.utime() with the writer's). Its content stays; a record that is gone is not re-created."""
    clock, touches, real_touch = FakeClock(0.0), [], preflight.touch_owner
    monkeypatch.setattr(runner, "_monotonic", clock.time)

    def touch(partial):
        try:
            real_touch(partial)
        finally:
            touches.append(partial)
    monkeypatch.setattr(preflight, "touch_owner", touch)
    p = tmp_path / "p.ome.tif"
    p.write_bytes(b"")
    token, record = preflight.write_owner(str(p)), Path(str(p) + OWNER_SUFFIX)
    content = record.read_bytes()
    age(record, seconds=600)
    job = Job("r", 1, str(tmp_path), "component", "S", "component_data", "S", (), str(tmp_path), "p.ome.tif",
              False, None, None, None, None)
    mon = runner.PartialMonitor(str(p), job, lambda ev: None, poll_s=0.001)

    def beat():
        n, deadline = len(touches), time.monotonic() + 10
        while len(touches) == n:
            clock.advance(HEARTBEAT_S)
            assert time.monotonic() < deadline, "no heartbeat"
            time.sleep(0.005)
    try:
        mon.start()
        beat()
        assert time.time() - record.stat().st_mtime < 5 and record.read_bytes() == content
        record.unlink()
        beat()
        assert set(touches) == {str(p)} and os.listdir(tmp_path) == ["p.ome.tif"]
    finally:
        mon.stop()
        preflight.release_owner(token)


# ------------------------------------------------------------------------------------------- space checks
def test_space_check_1_skips_before_any_tile_is_read(src, out, collector, monkeypatch):
    plan = make_plan(src, out, slides=("Huge_Scan1",))
    give_estimate(plan)
    worst = plan.ticked()[0].est.out_worst
    monkeypatch.setattr(runner, "free_bytes", lambda path: worst - 1)
    Engine = FakeEngine.configure()
    _, outcomes = run(plan, Engine, out, collector)
    (o,) = outcomes.values()
    assert o.status == "skipped" and re.fullmatch(r"Skipped: not enough space \(needs ~[\d.]+ \w+, [\d.]+ \w+ free\)",
                                                  o.message)
    assert Engine.calls == [] and not (out / PARTIAL_DIR).exists()


def test_space_check_2_skips_disk_mode_before_stitch(src, out, collector):
    Engine = FakeEngine.configure(needed_gb=1e9)        # far more than RAM: v1's rule picks disk mode
    _, outcomes = run(make_plan(src, out, slides=("Disk_Scan1",)), Engine, out, collector, summary_csv=False)
    (o,) = outcomes.values()
    assert o.status == "skipped" and o.message.startswith("Skipped: not enough space (needs ~")
    assert [c for c, _ in Engine.calls] == ["scan_metadata"] and not (out / PARTIAL_DIR).exists()


# ---------------------------------------------------------------------------------------------- stopping
def test_stop_after_current(src, out, collector):
    box = []
    Engine = FakeEngine.configure(on_step=lambda eng, phase, i: box[0].stop_after_current())
    plan = make_plan(src, out)
    summary, outcomes = run(plan, Engine, out, collector, runner_box=box)
    assert list(summary.counts.items()) == [("done", 1)] and summary.stop_reason == "user"
    assert summary.stopped_by_user and box[0].stopping == "after_current"
    first, *rest = plan.rows()
    assert first.status == "done" and [(r.status, r.checked) for r in rest] == [("new", True), ("new", True)]
    check_event_order(collector.events)


@pytest.mark.parametrize("phase", ["stitch", "write"])
def test_stop_now_discards_the_current_slide(src, out, collector, monkeypatch, phase):
    box = []

    def hook(eng, ph, i):
        if ph == phase:
            box[0].stop_now()
    Engine = FakeEngine.configure(on_step=hook, needed_gb=1e6)          # disk mode: a temp file to clean
    monkeypatch.setattr(runner, "free_bytes", lambda path: 1 << 62)
    summary, outcomes = run(make_plan(src, out), Engine, out, collector, runner_box=box)
    (o,) = outcomes.values()
    assert (o.status, o.message) == ("cancelled", CANCELLED_TEXT) and summary.stop_reason == "user"
    assert box[0].stopping == "now" and not (out / PARTIAL_DIR).exists() and not list(out.glob("*.ome.tif"))
    assert [c for c, _ in Engine.calls][-1] == ("write_to_disk" if phase == "write" else "stitch")
    check_event_order(collector.events)


def test_stop_before_start_runs_nothing(src, out, collector):
    plan = make_plan(src, out)
    r = BatchRunner(1, plan, FakeEngine, collector, RunOptions(str(out)))
    r.stop_now()
    r.start()
    summary = collector.wait_type(RunFinished)[0].summary
    assert summary.rows == () and summary.stop_reason == "user" and summary.csv_path is None
    assert [r.status for r in plan.rows()] == ["new"] * 3


# ------------------------------------------------------------------------- failures and output-side stops
def test_two_output_side_failures_in_a_row_stop_the_batch(src, out, collector):
    Engine = FakeEngine.configure(fail="write", last_error=ENOSPC_ERROR)
    plan = make_plan(src, out)
    summary, outcomes = run(plan, Engine, out, collector)
    assert summary.stop_reason == "output_unavailable" and summary.counts == {"failed": 2}
    assert all(o.output_side and o.message == "Failed: The disk is full." for o in outcomes.values())
    assert plan.rows()[2].status == "new" and not list(out.glob("*.ome.tif"))
    assert not (out / PARTIAL_DIR).exists()


def test_a_failed_write_probe_stops_at_once(src, out, collector):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"a file, not a folder")
    Engine = FakeEngine.configure()
    summary, outcomes = run(make_plan(src, out), Engine, out, collector)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_unavailable and o.message.startswith(f"Failed: Can't save to {out}: ")
    assert summary.stop_reason == "output_unavailable" and Engine.calls == []


def test_a_failure_with_a_healthy_output_folder_is_not_output_side(src, out, collector):
    Engine = FakeEngine.configure(fail="write", last_error=CONTIG_ERROR, fail_if=lambda e: "SlideA" in e.image_name)
    summary, outcomes = run(make_plan(src, out), Engine, out, collector)
    assert [o.status for o in outcomes.values()] == ["failed", "done", "done"] and summary.stop_reason is None
    failed = list(outcomes.values())[0]
    assert not failed.output_side and failed.message.startswith("Failed: <PLANARCONFIG.CONTIG: 1> does not match")
    assert summary.counts == {"failed": 1, "done": 2}


@pytest.mark.skipif(AS_ROOT, reason="root ignores permissions")
def test_without_last_error_a_fresh_write_probe_decides(src, out, collector):
    def lock_output(eng):
        out.chmod(0o555)                         # the share went read-only while level 0 was written
        return True
    Engine = FakeEngine.configure(fail="write", fail_if=lock_output)
    try:
        summary, outcomes = run(make_plan(src, out, slides=("Lost_Scan1",)), Engine, out, collector)
    finally:
        out.chmod(0o755)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_side


@pytest.fixture
def slide_logging(tmp_path, collector):
    """osutil.setup_logging (it remembers each slide's last problem); undone afterwards."""
    root, level = logging.getLogger(), logging.getLogger().level
    listener = osutil.setup_logging(str(tmp_path / "logs"), collector)
    yield
    listener.stop()
    for h in listener.handlers:
        h.close()
    for h in [h for h in root.handlers if isinstance(h, osutil._SlideQueueHandler)]:
        root.removeHandler(h)
    root.setLevel(level)
    osutil.set_log_slide("")


def test_without_last_error_the_slide_log_gives_the_reason(src, out, collector, slide_logging):
    Engine = FakeEngine.configure(fail="scan", log_warning="⚠️ T_[1,2].tif: missing TIFF position tags [286, 287]")
    _, outcomes = run(make_plan(src, out, slides=("NoTags_Scan1",)), Engine, out, collector)
    (o,) = outcomes.values()
    assert o.message == "Failed: T_[1,2].tif: missing TIFF position tags [286, 287]" and not o.output_side


def test_an_engine_exception_becomes_text(src, out, collector):
    Engine = FakeEngine.configure(raise_in="scan")
    _, outcomes = run(make_plan(src, out, slides=("Boom_Scan1",)), Engine, out, collector)
    (o,) = outcomes.values()
    assert (o.status, o.message) == ("failed", "Failed: fake scan exception") and not (out / PARTIAL_DIR).exists()


def test_permission_error_on_replace_leaves_the_row_to_finalize(src, out, collector, monkeypatch):
    calls, real_replace = [], os.replace

    def replace(a, b):
        if PARTIAL_DIR in str(a):
            calls.append(a)
            raise PermissionError(13, "Permission denied")
        return real_replace(a, b)
    monkeypatch.setattr(preflight.os, "replace", replace)
    monkeypatch.setattr(preflight.os, "link", replace)
    monkeypatch.setattr(preflight, "RETRY_DELAY_S", 0.0)
    summary, outcomes = run(make_plan(src, out, slides=("Open_Scan1",)), FakeEngine, out, collector,
                            replace_retries=3)
    (o,) = outcomes.values()
    assert (o.status, o.message, len(calls)) == ("finalize", FINALIZE_TEXT, 3)
    assert o.partial_path == str(out / PARTIAL_DIR / "Open_Scan1.ome.tif") and Path(o.partial_path).exists()
    assert Path(o.partial_path + COMPLETE_SUFFIX).exists()          # DS-9: complete, remembered on disk
    assert o.output_path == str(out / "Open_Scan1.ome.tif") and not Path(o.output_path).exists()
    assert not collector.of_type(OutputWritten) and summary.counts == {"finalize": 1}
    monkeypatch.undo()
    preflight.move_into_place(o.partial_path, o.output_path)        # Finalize (finalize_kept: after verify_output)
    assert Path(o.output_path).exists() and os.listdir(out / PARTIAL_DIR) == []   # with its sidecars


def test_run_finished_is_posted_even_after_an_internal_error(src, out, collector, monkeypatch):
    plan = make_plan(src, out)
    monkeypatch.setattr(plan, "claim_next", lambda *a: 1 / 0)
    summary, _ = run(plan, FakeEngine, out, collector)
    assert summary.rows == () and [r.status for r in plan.rows()] == ["new"] * 3


# ------------------------------------------------------------------------------------- R28: no pinned canvas
@pytest.mark.parametrize("how", [dict(fail="write", last_error=ENOSPC_ERROR), dict(raise_in="write")])
def test_no_pinned_canvas_after_a_failed_write(src, out, collector, how):
    Engine = FakeEngine.configure(**how)
    plan = make_plan(src, out, slides=("Pin_Scan1",))
    out.mkdir()                                  # made by the final check at Start (DS-8: never by a slide)
    r = BatchRunner(1, plan, Engine, collector, RunOptions(str(out)))
    plan.begin_run()
    outcome = r._run_job(plan.claim_next(str(out), None))
    assert outcome.status == "failed" and len(Engine.canvas_refs) == 1
    assert Engine.canvas_refs[0]() is None                 # dead as soon as _run_job has returned


@pytest.fixture
def one_thread(monkeypatch):
    monkeypatch.setattr(stitcher_RAM, "NUM_THREADS", 1)


def test_real_engine_enospc_is_output_side_and_frees_the_canvas(synth_tree, out, collector, monkeypatch,
                                                                one_thread):
    """The verify2/last_error_leak.py injection: ENOSPC at the first pyramid level, through the runner."""
    refs = []

    class Spy(stitcher_RAM.StitchingEngine):
        def stitch(self, use_ram=True):
            canvas = super().stitch(use_ram)
            refs.append(weakref.ref(canvas))
            return canvas

    class Boom(tifffile.TiffWriter):
        def write(self, data, *a, **k):
            if k.get("subifds") is None:
                raise OSError(28, "No space left on device")
            return super().write(data, *a, **k)
    monkeypatch.setattr(stitcher_RAM.tifffile, "TiffWriter", Boom)
    plan = Plan(str(synth_tree / "ProjectA"), str(out))
    folder = synth_tree / "ProjectA" / "SlideA1_Scan1"
    summary = summarize_listing(str(folder), scan_dir(str(folder), cap=10000).entries)
    plan.add_candidate(candidates_from_summary(summary, root=str(folder.parent), depth=1, dir_mtime=None)[0])
    plan.toggle([plan.rows("new")[0].row_id], True)                  # 9 tiles: unticked by default
    out.mkdir()
    r = BatchRunner(1, plan, Spy, collector, RunOptions(str(out)))
    plan.begin_run()
    outcome = r._run_job(plan.claim_next(str(out), None))
    assert (outcome.status, outcome.message, outcome.output_side) == ("failed", "Failed: The disk is full.", True)
    assert len(refs) == 1 and refs[0]() is None
    assert not (out / PARTIAL_DIR / "SlideA1_Scan1.ome.tif").exists()


# ------------------------------------------------------------------------------- DS-1: dropped tiles are gaps
def test_tiles_dropped_by_the_scan_are_gaps(src, out, collector):
    """DS-1: tiles whose headers the metadata scan couldn't read leave the same hole as unreadable pixels:
    'Done with gaps', failed_tiles, the CSV and OutputWritten.gaps (for the ledger) count them."""
    Engine = FakeEngine.configure(dropped=2, failed_tiles=1)
    summary, outcomes = run(make_plan(src, out, slides=("Holes_Scan1",)), Engine, out, collector)
    (o,) = outcomes.values()
    assert (o.status, o.message, o.tiles_used, o.failed_tiles) == \
        ("done_gaps", "Done with gaps: 3 tiles couldn't be read", 10, 3)
    assert [e.gaps for e in collector.of_type(OutputWritten)] == [3]
    with open(summary.csv_path, encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert (row["status"], row["tiles"], row["failed_tiles"]) == ("done_gaps", "10", "3")


def test_real_engine_reports_a_tile_it_could_not_read_as_a_gap(tmp_path, collector, one_thread):
    """DS-1 with the real engine: a 0-byte tile (an interrupted copy; not the first, which the engine samples)."""
    src, out = tmp_path / "in", tmp_path / "out"
    synth.make_slide(str(src / "SlideX_Scan1"), "SlideX_Scan1", 4, 3)
    tiles = sorted(os.listdir(src / "SlideX_Scan1"))
    (src / "SlideX_Scan1" / tiles[5]).write_bytes(b"")
    plan = Plan(str(src), str(out))
    summary = summarize_listing(str(src / "SlideX_Scan1"), scan_dir(str(src / "SlideX_Scan1"), cap=10000).entries)
    plan.add_candidate(candidates_from_summary(summary, root=str(src), depth=1, dir_mtime=None)[0])
    plan.toggle([r.row_id for r in plan.rows("new")], True)
    _, outcomes = run(plan, stitcher_RAM.StitchingEngine, out, collector)
    (o,) = outcomes.values()
    assert (o.status, o.tiles_used, o.failed_tiles) == ("done_gaps", 11, 1)
    assert o.message == "Done with gaps: 1 tile couldn't be read"
    assert [e.gaps for e in collector.of_type(OutputWritten)] == [1]


def test_a_legacy_folder_counts_only_tiles_it_could_not_read_as_gaps(tmp_path, collector):
    """DS-1 on a legacy_all row, which takes every TIFF in its folder (v1's rule): an overview image without position
    tags is left out by design, no gap; a file that can't be read may have been a tile, so it is one."""
    src, out = tmp_path / "in", tmp_path / "out"
    folder = src / "OldExport"
    folder.mkdir(parents=True)
    for i in range(6):
        (folder / f"field_{i:02d}.tif").write_bytes(b"")      # FakeEngine never opens its tiles
    (folder / "zz_broken.tif").write_bytes(b"")
    tifffile.imwrite(folder / "zz_overview.tif", np.zeros((8, 8), np.uint8))
    plan = Plan(str(src), str(out))
    s = summarize_listing(str(folder), scan_dir(str(folder), cap=10000).entries)
    (cand,) = candidates_from_summary(s, root=str(src), depth=1, dir_mtime=None)
    assert cand.kind == "legacy_all"
    plan.add_candidate(cand)
    plan.toggle([r.row_id for r in plan.rows() if r.section != "note"], True)
    _, outcomes = run(plan, FakeEngine.configure(dropped=2), out, collector)       # the last two: zz_*
    (o,) = outcomes.values()
    assert (o.status, o.message, o.tiles_used, o.failed_tiles) == \
        ("done_gaps", "Done with gaps: 1 tile couldn't be read", 6, 1)
    assert [e.gaps for e in collector.of_type(OutputWritten)] == [1]


# ------------------------------------------------------------------------------- DS-2: the custom temp folder
def test_two_save_folders_never_share_a_scratch_file(tmp_path, collector, monkeypatch):
    """DS-2 / D4: two batches with the same custom temp folder and the same slide name ('Scan1') but different save
    folders: each save folder has its own scratch subfolder, so the two memmaps can never be one file. Empty
    subfolders go at the end of the run."""
    monkeypatch.setattr(runner, "free_bytes", lambda path: 1 << 62)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen = []
    Engine = FakeEngine.configure(needed_gb=1e6, on_step=lambda eng, phase, i: phase == "stitch" and i == 1 and
                                  seen.append((eng.temp_path, eng.temp_path.exists())))
    for name in ("outA", "outB"):
        out = tmp_path / name
        _, outcomes = run(make_plan(tmp_path / "in" / name, out, slides=("Scan1",)), Engine, out, collector,
                          temp_dir=str(scratch))
        assert [(o.status, o.memory_mode) for o in outcomes.values()] == [("done", "disk")]
    assert seen == [(Path(temp_subdir(str(scratch), str(tmp_path / n))) / "Scan1.ome.tmp.raw", True)
                    for n in ("outA", "outB")]
    assert os.listdir(scratch) == []


# ---------------------------------------------------------------------------------- DS-3: partial ownership
def test_a_skewed_clock_never_takes_over_a_heartbeated_partial(src, out, collector, monkeypatch):
    """DS-3: staleness is judged by the share's clock. This computer's clock runs 150 s ahead; another computer's
    partial that its heartbeat keeps fresh is still respected."""
    plan = make_plan(src, out, slides=("Shared_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Shared_Scan1.ome.tif"
    partial.write_bytes(b"theirs")
    own(partial, 4242, 1.0, host="otherhost.example.org")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 150)
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: " + BUSY_TEXT]
    assert partial.read_bytes() == b"theirs"


def test_a_writer_whose_clock_lags_the_share_keeps_its_live_partial(src, out, collector):
    """DS-3 on SMB: the writer's heartbeat os.utime() stamps its partial with the writer's own clock, 150 s behind the
    file server's; its rewrite of the owner record gets the server's time, so the claim stays live."""
    plan = make_plan(src, out, slides=("Lag_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Lag_Scan1.ome.tif"
    partial.write_bytes(b"")                                   # stitching in RAM: a 0-byte claim
    own(partial, 4242, 1.0, host="otherhost.example.org", token="theirs")
    age(partial, seconds=150)                                   # its latest heartbeat, by the writer's clock
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: " + BUSY_TEXT] and partial.exists()


@pytest.mark.parametrize("how, when", [("file", "stitch"), ("owner", "stitch"), ("owner", "write")])
def test_a_partial_taken_over_meanwhile_is_never_written_or_moved_into_place(src, out, collector, how, when):
    """DS-3: another stitcher took the partial over (a new file at that path, or its owner record). While ours was
    stitched (a laptop asleep past the stale age, say): the engine, which opens the partial by name, must not write
    into their claim. While ours was written: nothing is moved into place. Either way it is reported, and our owner
    record goes, so nobody takes their claim for our live work."""
    def take_over(eng, phase, i):
        if phase == when and i == 1:
            if how == "file":                    # (an older stitcher without owner records): ours moved aside
                os.rename(eng.output_path, str(eng.output_path) + ".aside")
                eng.output_path.write_bytes(b"their claim")
            else:
                own(eng.output_path, 4242, 1.0, host="otherhost.example.org", token="theirs")
    Engine = FakeEngine.configure(on_step=take_over)
    _, outcomes = run(make_plan(src, out, slides=("Swap_Scan1",)), Engine, out, collector)
    (o,) = outcomes.values()
    assert (o.status, o.message) == ("skipped", "Skipped: " + TAKEN_OVER_TEXT)
    assert not (out / "Swap_Scan1.ome.tif").exists() and not collector.of_type(OutputWritten)
    partial = out / PARTIAL_DIR / "Swap_Scan1.ome.tif"
    assert partial.exists()                                          # theirs: never ours to delete
    assert ("write_to_disk" in [c for c, _ in Engine.calls]) == (when == "write")
    if how == "file":
        assert partial.read_bytes() == b"their claim" and not Path(str(partial) + OWNER_SUFFIX).exists()
    else:
        assert json.loads(Path(str(partial) + OWNER_SUFFIX).read_text())["token"] == "theirs"


def test_a_takeover_never_removes_a_claim_made_since_it_was_judged(tmp_path):
    """D3: two stitchers judged the same partial stale; the other one has replaced it with its fresh claim before
    our rename: ours notices that it moved another file and puts it back."""
    p = tmp_path / "S.ome.tif"
    p.write_bytes(b"stale")
    age(p, seconds=3600)
    judged = os.stat(p)
    os.remove(p)
    p.write_bytes(b"")                                          # their claim
    assert runner._take_over(str(p), judged, None) is False
    assert visible_files(tmp_path) == ["S.ome.tif"] and p.read_bytes() == b""
    (tmp_path / "S.ome.tmp.raw").write_bytes(b"canvas")
    own(p, 4242, 1.0)
    assert runner._take_over(str(p), os.stat(p), None) is True  # the one judged: deleted with its sidecars
    assert visible_files(tmp_path) == []
    assert runner._take_over(str(p), judged, None) is False     # the second taker finds nothing to rename


def test_a_stale_partial_on_a_fat_drive_is_taken_over(src, out, collector, monkeypatch):
    """FAT and exFAT give a file renamed to a longer name another file id (its directory entry moves). The takeover's
    rename to '<partial>.stale-<hex>' must still recognise the partial it judged, or a crash leftover on a USB drive
    would block its slide for good."""
    real_stat, real_rename, renamed = os.stat, os.rename, set()

    def rename(a, b, *args, **kwargs):
        real_rename(a, b, *args, **kwargs)
        if len(os.path.basename(str(b))) > len(os.path.basename(str(a))):
            renamed.add(str(b))

    def stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if str(path) not in renamed:
            return st
        fields = list(st)
        fields[1] += 1000003                                    # st_ino
        return os.stat_result(fields, {"st_atime_ns": st.st_atime_ns, "st_mtime_ns": st.st_mtime_ns,
                                       "st_ctime_ns": st.st_ctime_ns})
    monkeypatch.setattr(runner.os, "rename", rename)
    monkeypatch.setattr(runner.os, "stat", stat)
    (out / PARTIAL_DIR).mkdir(parents=True)
    leftover = out / PARTIAL_DIR / "Usb_Scan1.ome.tif"
    leftover.write_bytes(b"crash leftover")
    age(leftover, seconds=3600)
    _, outcomes = run(make_plan(src, out, slides=("Usb_Scan1",)), FakeEngine, out, collector)
    assert [o.status for o in outcomes.values()] == ["done"] and renamed
    assert not (out / PARTIAL_DIR).exists() and (out / "Usb_Scan1.ome.tif").stat().st_size > 64 * 1024
    [aside] = renamed                         # the length the plan's long_path counts, a remnant the index knows
    assert len(aside) == len(str(leftover)) + len(STALE_INFIX) + STALE_HEX and preflight.STALE_RE.search(aside)


def test_giving_up_never_deletes_a_claim_that_reuses_our_file_id(tmp_path, monkeypatch):
    """D3: another stitcher took our claim over and claimed the name again; its new file got our old (st_dev, st_ino)
    (ext4 hands a freed inode number to the next new file). When our slide then fails or is stopped, nothing of
    theirs may go: only our owner record proves a claim ours."""
    claim = BatchRunner._claim(str(tmp_path / "S.ome.tif"))
    try:
        os.remove(claim.path)                                   # their takeover ...
        preflight.remove_sidecars(claim.path)
        Path(claim.path).write_bytes(b"")                       # ... and their claim, with our old file id
        own(claim.path, 4242, 1.0, host="otherhost.example.org", token="theirs")
        real_id = preflight.path_id
        monkeypatch.setattr(preflight, "path_id", lambda path: claim.ident if path == claim.path else real_id(path))
        BatchRunner._discard(claim)
        assert sorted(visible_files(tmp_path)) == ["S.ome.tif", "S.ome.tif" + OWNER_SUFFIX]
        os.remove(claim.path + OWNER_SUFFIX)
        BatchRunner._discard(claim)                             # no owner record: not provably ours either
        assert visible_files(tmp_path) == ["S.ome.tif"]
    finally:
        preflight.release_owner(claim.token)


# ------------------------------------------------------------------------------ CT-9: dead and live owners
def test_a_dead_owners_fresh_partial_is_taken_over_at_once(src, out, collector, dead_process):
    """CT-9: 'Quit now', then a restart within 2 minutes: the partial's owner on this computer has exited."""
    plan = make_plan(src, out, slides=("Quit_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Quit_Scan1.ome.tif"
    partial.write_bytes(b"quit")                               # 1 s old
    own(partial, *dead_process)
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.status for o in outcomes.values()] == ["done"] and not (out / PARTIAL_DIR).exists()


def test_a_live_owners_partial_is_skipped_however_old(src, out, collector, live_process):
    plan = make_plan(src, out, slides=("Busy_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Busy_Scan1.ome.tif"
    partial.write_bytes(b"busy")
    own(partial, *live_process)
    age(partial, seconds=3600)                                 # e.g. a heartbeat that can't touch it on Windows
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: " + BUSY_TEXT] and partial.read_bytes() == b"busy"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux moves create_time() with the wall clock")
def test_a_live_owner_stays_live_after_a_step_of_the_wall_clock(src, out, collector, live_process, monkeypatch):
    """DS-3: an NTP step of 3 s moves every create_time() psutil reports on Linux. The live owner's 0-byte claim must
    not be taken over: its engine would later open the same path and write into ours."""
    linux = pytest.importorskip("psutil._pslinux")
    plan = make_plan(src, out, slides=("Step_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Step_Scan1.ome.tif"
    partial.write_bytes(b"")
    own(partial, *live_process)
    real_boot_time = linux.boot_time
    monkeypatch.setattr(linux, "boot_time", lambda: real_boot_time() + 3)
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: " + BUSY_TEXT] and partial.read_bytes() == b""


# ------------------------------------------------------------------------------ DS-8: a vanished save folder
def mounted_share(tmp_path):
    """A share 'mounted' under $HOME (a symlink to the server's folder), as sshfs or rclone mounts are."""
    server, mnt = tmp_path / "server", tmp_path / "home" / "mnt" / "labshare"
    (server / "stitched").mkdir(parents=True)
    mnt.parent.mkdir(parents=True)
    mnt.symlink_to(server, target_is_directory=True)
    return server, mnt


def unmount(mnt):
    mnt.unlink()
    mnt.mkdir()                                                 # the empty mount-point folder is left


def test_a_save_folder_that_vanishes_between_slides_is_not_recreated(src, tmp_path, collector):
    """DS-8: the share drops after slide 1: slide 2 fails with output_unavailable, the batch stops, and nothing (no
    folder, no CSV) is written onto the local disk under the bare mount point."""
    server, mnt = mounted_share(tmp_path)
    out, plan = mnt / "stitched", make_plan(src, mnt / "stitched")

    def post(ev):
        collector(ev)
        if isinstance(ev, RowFinished) and len(collector.of_type(RowFinished)) == 1:
            unmount(mnt)
    r = BatchRunner(1, plan, FakeEngine, post, RunOptions(str(out), poll_s=0.01))
    r.start()
    summary = collector.wait_type(RunFinished, timeout=60)[0].summary
    r.join(5)
    first, second = [o for _, o in summary.rows]
    assert first.status == "done" and second.status == "failed" and second.output_unavailable
    assert second.message.startswith(f"Failed: Can't save to {out}: ")
    assert summary.stop_reason == "output_unavailable" and summary.csv_path is None
    assert os.listdir(mnt) == [] and sorted(visible_files(server / "stitched")) == [PARTIAL_DIR, "SlideA_Scan1.ome.tif"]
    assert os.listdir(server / "stitched" / PARTIAL_DIR) == []      # (unreachable at the end of the run: kept)


@pytest.mark.parametrize("when", ["stitching", "formatting"])
def test_a_save_folder_lost_mid_slide_gets_nothing(src, tmp_path, collector, when):
    """DS-8: the share drops while slide 1 is stitched: the check before the engine opens the partial stops it, and
    nothing at all, not even a folder, is written on the local disk under the bare mount point. If it drops just
    after that check, the engine's own makedirs puts the partial there: it is not moved into place but removed.
    Either way the batch stops."""
    server, mnt = mounted_share(tmp_path)
    out = mnt / "stitched"

    class Engine(FakeEngine.configure(on_step=lambda eng, phase, i: when == "stitching" and phase == "stitch" and
                                      i == 1 and unmount(mnt))):
        def _report(self, current, total, text):
            if when == "formatting" and text.startswith("Formatting"):
                unmount(mnt)
            super()._report(current, total, text)
    summary, outcomes = run(make_plan(src, out), Engine, out, collector)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_unavailable and o.message.startswith(f"Failed: Can't save to {out}: ")
    assert summary.stop_reason == "output_unavailable" and not collector.of_type(OutputWritten)
    if when == "stitching":
        assert [c for c, _ in Engine.calls] == ["scan_metadata", "stitch"]
        assert [files for _, _, files in os.walk(mnt)] == [[]]
    else:
        assert runner.FOLDER_CHANGED_TEXT in o.message
        assert (out / PARTIAL_DIR / "SlideA_Scan1.ome.tif").exists()   # unowned path: never delete it
    assert visible_files(server / "stitched") == [PARTIAL_DIR]                # no final; our claim: out of reach
    assert sorted(os.listdir(server / "stitched" / PARTIAL_DIR)) == ["SlideA_Scan1.ome.tif",
                                                                      "SlideA_Scan1.ome.tif" + OWNER_SUFFIX]


# ---------------------------------------------------------------- DS-9: a complete partial waits for Finalize
def test_a_kept_partial_is_never_restitched_and_is_finalized_later(tmp_path, collector, monkeypatch, one_thread):
    """DS-9 / D3 rule 1: the old file was in use, so the finished image stays complete in the partial folder. However
    old, it is no leftover and no other batch re-stitches or takes it over; a finalize_only job checks it and
    moves it into place with its gaps."""
    src, out = tmp_path / "in", tmp_path / "out"
    synth.make_slide(str(src / "Kept_Scan1"), "Kept_Scan1", 4, 3)
    out.mkdir()
    (out / "Kept_Scan1.ome.tif").write_bytes(b"the old file, open in QuPath")

    def job(**kw):
        plan = Plan(str(src), str(out))
        s = summarize_listing(str(src / "Kept_Scan1"), scan_dir(str(src / "Kept_Scan1"), cap=10000).entries)
        plan.add_candidate(candidates_from_summary(s, root=str(src), depth=1, dir_mtime=None)[0])
        plan.toggle([r.row_id for r in plan.rows() if r.section != "note"], True)
        plan.begin_run()
        return plan, dataclasses.replace(plan.claim_next(str(out), None), replace=True, **kw)
    real_replace = os.replace
    monkeypatch.setattr(preflight.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(13, "in use"))
                        if PARTIAL_DIR in str(a) else real_replace(a, b))
    plan, first = job()
    o = BatchRunner(1, plan, stitcher_RAM.StitchingEngine, collector, RunOptions(str(out)))._run_job(first)
    assert o.status == "finalize" and preflight.complete_gaps(o.partial_path) == 0
    kept, kept_bytes = Path(o.partial_path), Path(o.partial_path).read_bytes()
    monkeypatch.setattr(preflight.os, "replace", real_replace)
    age(kept, seconds=3600)
    idx = preflight.list_output_dir(str(out), ROOT_FS, frozenset())
    assert list(idx.ready) == ["kept_scan1.ome.tif"] and idx.leftovers == () and not idx.partials
    plan, again = job()                                           # a batch that plans to stitch it again
    o = BatchRunner(2, plan, FakeEngine, collector, RunOptions(str(out)))._run_job(again)
    assert o.message == "Skipped: " + WAITING_TEXT and kept.read_bytes() == kept_bytes
    plan, ready = job(finalize_only=True)
    col = type(collector)()
    done = BatchRunner(3, plan, FakeEngine, col, RunOptions(str(out)))._run_job(ready)
    assert (done.status, done.output_path, done.failed_tiles) == ("done", str(out / "Kept_Scan1.ome.tif"), 0)
    assert [(e.path, e.gaps) for e in col.of_type(OutputWritten)] == [(done.output_path, 0)]
    assert preflight.verify_output(done.output_path) == ("valid", "") and not (out / PARTIAL_DIR).exists()


def test_a_finalize_only_job_never_moves_a_damaged_image(src, out, collector):
    plan = make_plan(src, out, slides=("Bad_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    partial = out / PARTIAL_DIR / "Bad_Scan1.ome.tif"
    partial.write_bytes(b"II*\x00 cut short")
    Path(str(partial) + COMPLETE_SUFFIX).write_text('{"gaps": 2}')
    plan.begin_run()
    job = dataclasses.replace(plan.claim_next(str(out), None), finalize_only=True)
    o = BatchRunner(1, plan, FakeEngine, collector, RunOptions(str(out)))._run_job(job)
    assert o.status == "failed" and o.message.startswith("Failed: The finished image is damaged (")
    assert not (out / "Bad_Scan1.ome.tif").exists() and partial.exists()
    assert not Path(str(partial) + COMPLETE_SUFFIX).exists()      # an ordinary leftover from now on


# --------------------------------------------------------------------- DS-10, DS-12, DS-13: the partial folder
def test_a_linked_partial_folder_is_refused(src, tmp_path, collector):
    out, target = tmp_path / "out", tmp_path / "fastdisk"
    out.mkdir()
    target.mkdir()
    (target / "Link_Scan1.ome.tif").write_bytes(b"real image")
    age(target / "Link_Scan1.ome.tif", seconds=3600)
    (out / PARTIAL_DIR).symlink_to(target, target_is_directory=True)
    summary, outcomes = run(make_plan(src, out, slides=("Link_Scan1",)), FakeEngine, out, collector)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_unavailable and preflight.PDIR_LINK_TEXT in o.message
    assert os.listdir(target) == ["Link_Scan1.ome.tif"]
    assert (target / "Link_Scan1.ome.tif").read_bytes() == b"real image"
    (target / ("Link_Scan1.ome.tif" + COMPLETE_SUFFIX)).write_text("{}")      # looks finished, behind the link
    plan = make_plan(src, out, slides=("Link_Scan1",))
    plan.begin_run()
    job = dataclasses.replace(plan.claim_next(str(out), None), finalize_only=True)
    o = BatchRunner(2, plan, FakeEngine, collector, RunOptions(str(out)))._run_job(job)
    assert o.status == "failed" and o.output_unavailable and not (out / "Link_Scan1.ome.tif").exists()
    assert (target / "Link_Scan1.ome.tif").read_bytes() == b"real image"


def test_an_output_that_appeared_is_not_overwritten_even_with_a_stale_partial(src, out, collector):
    """DS-12: a 'new' row; after planning a colleague writes the final and an old crash partial lies around: the
    skip rule is just exists(final) and not job.replace."""
    plan = make_plan(src, out, slides=("Late_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    (out / PARTIAL_DIR / "Late_Scan1.ome.tif").write_bytes(b"crash leftover")
    age(out / PARTIAL_DIR / "Late_Scan1.ome.tif", seconds=600)
    (out / "Late_Scan1.ome.tif").write_bytes(b"COLLEAGUE'S OUTPUT")
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.message for o in outcomes.values()] == ["Skipped: the output appeared since planning"]
    assert (out / "Late_Scan1.ome.tif").read_bytes() == b"COLLEAGUE'S OUTPUT"


def test_an_output_that_appears_while_stitching_is_not_overwritten(src, out, collector):
    """DS-12 at the moment of the move: a colleague saved the final while ours was stitched. Nothing is
    overwritten; our finished image stays complete, so the next scan offers it ('ready') for the user to decide."""
    def colleague(eng, phase, i):
        if phase == "stitch" and i == 1:
            (out / "Late2_Scan1.ome.tif").write_bytes(b"COLLEAGUE'S OUTPUT")
    _, outcomes = run(make_plan(src, out, slides=("Late2_Scan1",)), FakeEngine.configure(on_step=colleague), out,
                      collector)
    (o,) = outcomes.values()
    assert (o.status, o.message) == ("skipped", "Skipped: " + APPEARED_TEXT)
    assert (out / "Late2_Scan1.ome.tif").read_bytes() == b"COLLEAGUE'S OUTPUT" and not collector.of_type(OutputWritten)
    assert preflight.list_output_dir(str(out), ROOT_FS, []).ready.keys() == {"late2_scan1.ome.tif"}


@pytest.mark.parametrize("when", ["stitching", "writing"])
def test_a_remounted_save_folder_leaves_no_live_looking_claim(src, out, collector, monkeypatch, when):
    """DS-8: an automount remounted the share while the slide was stitched or written: every st_dev changed. The
    batch stops and nothing is moved into place. Our owner record goes, so no other instance on this computer takes
    the partial for live work while this app runs; the partial, whose id no longer proves it ours, is left to the
    age rule."""
    real_id, changed = preflight.path_id, []

    def path_id(path):
        ident = real_id(path)
        return (ident[0] + 1, ident[1]) if changed and str(path).startswith(str(out)) else ident
    def remount(eng, phase, step):
        if phase == ("stitch" if when == "stitching" else "write") and step == 4:
            changed.append(True)
    monkeypatch.setattr(preflight, "path_id", path_id)
    Engine = FakeEngine.configure(on_step=remount)
    _, outcomes = run(make_plan(src, out, slides=("Remount_Scan1",)), Engine, out, collector, summary_csv=False)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_unavailable and runner.FOLDER_CHANGED_TEXT in o.message
    partial = out / PARTIAL_DIR / "Remount_Scan1.ome.tif"
    assert partial.exists() and preflight.owner_alive(preflight.read_owner(str(partial))) is False
    assert ("write_to_disk" in [c for c, _ in Engine.calls]) == (when == "writing")
    assert not (out / "Remount_Scan1.ome.tif").exists()


def test_an_owner_record_that_cannot_be_read_is_no_takeover(src, out, collector, monkeypatch):
    """A share hiccup just as the claim is checked before the write: whose the partial is can't be told, so the
    engine doesn't write, and the row says what went wrong rather than 'another stitcher took this slide over'."""
    def hiccup(partial):
        raise OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(preflight, "owner_token", hiccup)
    Engine = FakeEngine.configure()
    _, outcomes = run(make_plan(src, out, slides=("Hiccup_Scan1",)), Engine, out, collector, summary_csv=False)
    (o,) = outcomes.values()
    assert (o.status, o.message, o.output_side) == ("failed", "Failed: The disk or share reported a read error.", True)
    assert "write_to_disk" not in [c for c, _ in Engine.calls] and visible_files(out) == []


def test_a_partial_that_cannot_be_marked_complete_is_given_up(src, out, collector, monkeypatch):
    """D3: the share hiccups as the finished partial is to be marked complete, so it isn't trusted. It goes with our
    owner record: left behind, it would look live to every other instance on this computer (this app still runs),
    which would skip the slide until it quits."""
    def hiccup(partial, gaps=0):
        raise OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(preflight, "mark_complete", hiccup)
    _, outcomes = run(make_plan(src, out, slides=("Flaky_Scan1",)), FakeEngine, out, collector, summary_csv=False)
    (o,) = outcomes.values()
    assert o.status == "failed" and o.output_side and not collector.of_type(OutputWritten)
    assert visible_files(out) == []


def test_a_claim_that_cannot_be_recorded_is_an_output_side_failure(src, out, collector, monkeypatch):
    def full(partial):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(preflight, "write_owner", full)
    summary, outcomes = run(make_plan(src, out), FakeEngine, out, collector)
    assert [(o.status, o.message, o.output_side) for o in outcomes.values()] == \
        [("failed", "Failed: The disk is full.", True)] * 2 and summary.stop_reason == "output_unavailable"
    assert not (out / PARTIAL_DIR).exists()


def test_a_case_variant_partial_is_found_and_removed(src, out, collector):
    """DS-13: a crash partial whose name differs only in case (slide naming changed the stem) is this slide's
    partial: taken over and removed, so the slide isn't 'Unfinished' forever."""
    plan = make_plan(src, out, slides=("SlideA_Scan1",))
    (out / PARTIAL_DIR).mkdir(parents=True)
    variant = out / PARTIAL_DIR / "slidea_scan1.ome.tif"
    variant.write_bytes(b"crashed")
    age(variant, seconds=3600)
    _, outcomes = run(plan, FakeEngine, out, collector)
    assert [o.status for o in outcomes.values()] == ["done"] and not (out / PARTIAL_DIR).exists()
    variant.parent.mkdir()
    variant.write_bytes(b"live")                               # a fresh one is another stitcher's: respected
    os.remove(out / "SlideA_Scan1.ome.tif")
    _, outcomes = run(make_plan(src, out, slides=("SlideA_Scan1",)), FakeEngine, out, type(collector)())
    assert [o.message for o in outcomes.values()] == ["Skipped: " + BUSY_TEXT] and variant.read_bytes() == b"live"


# ------------------------------------------------------------------------- FS-17 and MW-9: probes and progress
def test_a_briefly_locked_probe_file_does_not_stop_the_batch(src, out, collector, monkeypatch):
    """FS-17: the delete of slide 2's probe file fails once (a virus scanner holds it): all 4 slides are stitched
    and no probe file is left."""
    real_remove, probes = os.remove, []

    def remove(path, *a, **k):
        if ".yfs-write-test-" in str(path):
            probes.append(path)
            if len(probes) == 2:
                raise PermissionError(13, "in use")
        return real_remove(path, *a, **k)
    monkeypatch.setattr(preflight.os, "remove", remove)
    summary, outcomes = run(make_plan(src, out, slides=("k1_Scan1", "k2_Scan1", "k3_Scan1", "k4_Scan1")),
                            FakeEngine, out, collector)
    assert summary.counts == {"done": 4} and summary.stop_reason is None
    assert not [n for n in os.listdir(out) if n.startswith(".yfs-") and not n.startswith(".yfs-lock-")]


def test_pyramid_writing_is_reported_as_activity(src, out, collector):
    """MW-9: a slow pyramid level still shows the partial growing (RowBytes with phase 'pyramid'), and the row keeps
    its 'Writing pyramid level' text instead of the level-0 wording."""
    texts = []

    class SlowPyramid(FakeEngine.configure()):
        def _report(self, current, total, text):
            super()._report(current, total, text)
            if text.startswith("Writing Pyramid"):
                time.sleep(0.05)
                texts.append(plan.rows()[0].run_text)
    plan = make_plan(src, out, slides=("Pyr_Scan1",))
    run(plan, SlowPyramid, out, collector)
    pyramid = [e for e in collector.of_type(RowBytes) if e.phase == "pyramid"]
    assert pyramid and all(e.written >= 64 * 1024 for e in pyramid)
    assert texts == [f"Writing pyramid level {k} of 3" for k in (1, 2, 3)]
    check_event_order(collector.events)


# ------------------------------------------------------------------------- B6: the whole core, real engine
@pytest.mark.slow
def test_core_e2e_on_the_synthetic_tree(synth_tree, tmp_path, monkeypatch):
    """B6: discovery -> plan -> runner, real engine; every component slide equals v1's output."""
    monkeypatch.setattr(stitcher_RAM, "NUM_THREADS", stitcher_RAM.NUM_THREADS)   # the tool sets it to 1
    e2e = load_module("yfs_tools_core_e2e", REPO / "tools" / "core_e2e.py")
    lines = []
    report = e2e.run_e2e(str(synth_tree), str(tmp_path / "out"), log=lines.append)
    assert report["ok"], "\n".join(lines)
    assert report["slides"] == {
        "SlideA1_Scan1.ome.tif": "done", "SlideA2_Scan1.ome.tif": "done", "S1_Scan1.ome.tif": "done",
        "S2_Scan1.ome.tif": "done", "R1-01_Scan1.ome.tif": "done", "R1-02.ome.tif": "done",
        "Slide9__batchA.ome.tif": "done", "Slide9__batchB.ome.tif": "done",              # D1: both suffixed
        "SusSlide_Scan1.ome.tif": "done_generic",
        "Slide [weird] (1).ome.tif": "done"}
    assert sum(ln.strip().startswith("identical") for ln in lines) == 10


@pytest.mark.parametrize("ready,arrival", [(True, "verification"), (True, "publication"), (False, "publication")])
def test_unapproved_publication_never_overwrites_an_arriving_destination(src, out, collector, monkeypatch,
                                                                       ready, arrival):
    plan = make_plan(src, out, slides=("Slide1",))
    plan.begin_run()
    job = dataclasses.replace(plan.claim_next(str(out), None), finalize_only=ready)
    pdir = out / PARTIAL_DIR
    pdir.mkdir(parents=True)
    partial, final = pdir / job.final_name, out / job.final_name
    if ready:
        tifffile.imwrite(partial, np.zeros((32, 48), np.uint16), metadata=None)
        preflight.mark_complete(str(partial))
    sentinel = b"another program's completed output"
    if arrival == "verification":
        verify = preflight.verify_output
        def check(path):
            verdict = verify(path)
            final.write_bytes(sentinel)
            return verdict
        monkeypatch.setattr(preflight, "verify_output", check)
    else:
        publish = preflight.move_into_place
        def late(*a, **kw):
            final.write_bytes(sentinel)
            return publish(*a, **kw)
        monkeypatch.setattr(preflight, "move_into_place", late)
    batch = BatchRunner(1, plan, FakeEngine, collector, RunOptions(str(out)))
    outcome = batch._run_job(job)
    assert outcome.status == "skipped" and APPEARED_TEXT in outcome.message
    assert final.read_bytes() == sentinel and partial.exists()
    assert Path(str(partial) + COMPLETE_SUFFIX).exists() and not collector.of_type(OutputWritten)


def test_changed_save_folder_keeps_unowned_partial_and_scratch(src, out, collector):
    plan = make_plan(src, out, slides=("Slide1",))
    def substitute(eng, phase, step):
        if phase == "stitch" and step == 4:
            out.rename(out.with_name("original"))
            eng.output_path.parent.mkdir(parents=True)
            eng.output_path.write_bytes(b"unowned partial")
            eng.temp_path.write_bytes(b"unowned scratch")
    Engine = FakeEngine.configure(on_step=substitute)
    _, outcomes = run(plan, Engine, out, collector, summary_csv=False)
    (outcome,) = outcomes.values()
    assert outcome.status == "failed" and outcome.output_unavailable
    assert not collector.of_type(OutputWritten)
    assert (out / PARTIAL_DIR / "Slide1.ome.tif").read_bytes() == b"unowned partial"
    assert (out / PARTIAL_DIR / "Slide1.ome.tmp.raw").read_bytes() == b"unowned scratch"
    assert "write_to_disk" not in [method for method, _ in Engine.calls]


@pytest.mark.parametrize("ready", [False, True])
def test_case_only_existing_name_is_replaced_and_reported(src, out, collector, ready):
    plan = make_plan(src, out, slides=("SlideA",))
    out.mkdir()
    final = out / "slidea.ome.tif"
    final.write_bytes(b"old output")
    if ready:
        partial = out / PARTIAL_DIR / "SlideA.ome.tif"
        partial.parent.mkdir()
        tifffile.imwrite(partial, np.ones((32, 48), np.uint16), metadata=None)
        preflight.mark_complete(str(partial))
    plan.apply_output_index(preflight.list_output_dir(str(out), ROOT_FS))
    row = plan.rows()[0]
    plan.toggle([row.row_id], True)
    assert row.replaces
    _, outcomes = run(plan, FakeEngine, out, collector, summary_csv=False)
    (outcome,) = outcomes.values()
    assert outcome.status == "done" and outcome.output_path == str(final)
    assert final.read_bytes() != b"old output" and not (out / "SlideA.ome.tif").exists()
    assert [ev.path for ev in collector.of_type(OutputWritten)] == [str(final)]


def test_concurrent_summary_writes_preserve_both_reports(src, out, collector, monkeypatch):
    plan = make_plan(src, out, slides=("Slide1",))
    plan.begin_run()
    job = plan.claim_next(str(out), None)
    batch = BatchRunner(1, plan, FakeEngine, collector, RunOptions(str(out)))
    batch._note_save_folder()
    monkeypatch.setattr(runner.time, "strftime", lambda *args: "20300101-010101")
    gate = threading.Barrier(2)
    results = []
    def write(message):
        gate.wait(2)
        results.append(batch._write_csv([(job, RowOutcome("done", message))]))
    threads = [threading.Thread(target=write, args=(message,)) for message in ("first report", "second report")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert len(set(results)) == 2 and None not in results
    reports = []
    for path in results:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and len(rows[0]) == len(CSV_COLUMNS)
        reports.append(rows[0]["message"])
    assert sorted(reports) == ["first report", "second report"]


def test_unicode_folder_run_keeps_valid_final_and_can_take_over_stale_scratch(tmp_path, collector, one_thread):
    src, out = tmp_path / "src", tmp_path / "out"
    folder = src / ("界" * 80)
    synth.make_slide(str(folder), "Slide1_Scan1", 4, 3)
    plan = Plan(str(src), str(out))
    summary = summarize_listing(str(folder), scan_dir(str(folder), cap=10000).entries)
    plan.add_candidate(candidates_from_summary(summary, root=str(src), depth=1, dir_mtime=None)[0])
    row = plan.rows("new")[0]
    pdir = out / PARTIAL_DIR
    pdir.mkdir(parents=True)
    partial = pdir / (row.output_stem + ".ome.tif")
    partial.write_bytes(b"abandoned")
    scratch = pdir / runner._raw_name(partial.name)
    scratch.write_bytes(b"old canvas")
    age(partial, scratch, seconds=600)
    _, outcomes = run(plan, stitcher_RAM.StitchingEngine, out, collector, summary_csv=False)
    (outcome,) = outcomes.values()
    assert outcome.status == "done" and preflight.verify_output(outcome.output_path)[0] == "valid"
    assert not scratch.exists() and not partial.exists()
    assert Path(outcome.output_path).stat().st_size > 1000


def test_runner_rejects_oversized_names_before_engine_work(src, out, collector):
    plan = make_plan(src, out, slides=("Slide1",))
    plan.begin_run()
    job = dataclasses.replace(plan.claim_next(str(out), None), final_name="界" * 80 + ".ome.tif")
    Engine = FakeEngine.configure()
    result = BatchRunner(1, plan, Engine, collector, RunOptions(str(out)))._run_job(job)
    assert result.status == "failed" and "shorter name" in result.message and Engine.calls == []


def visible_files(folder):
    """Reservation inodes intentionally persist; all other artifacts must still be accounted for."""
    return [name for name in os.listdir(folder) if not name.startswith(".yfs-lock-")]
