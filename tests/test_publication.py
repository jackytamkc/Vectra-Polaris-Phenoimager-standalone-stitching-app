"""Final-review publication races, using real reservations in separate processes."""
from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from yfs.core import preflight, runner
from yfs.core.events import OutputWritten
from yfs.core.model import Job
from yfs.core.names import PARTIAL_DIR
from yfs.core.plan import Plan


def job_for(out, ready=False):
    return Job("slide", 1, str(out), "component", "Slide1", "component_data", "Slide1", (),
               str(out), "Slide1.ome.tif", True, None, None, None, None, finalize_only=ready)


def finalizer(mode, out, paused, release, result):
    partial = str(Path(out) / PARTIAL_DIR / "Slide1.ome.tif")
    final = str(Path(out) / "Slide1.ome.tif")
    real_move = preflight.move_into_place
    def delayed(*args, **kwargs):
        paused.set()
        if not release.wait(4):
            raise TimeoutError("publication barrier")
        return real_move(*args, **kwargs)
    preflight.move_into_place = delayed
    events = []
    try:
        if mode == "ui":
            size, _ = preflight.finalize_kept(partial, final)
            result.put(("done", size))
            return
        job = job_for(out, ready=mode == "ready")
        batch = runner.BatchRunner(1, Plan(str(out), str(out)), None, events.append, runner.RunOptions(str(out)))
        batch._out_id = preflight.path_id(str(out))
        batch._phase = lambda *a: None
        if mode == "ready":
            outcome = batch._finalize_ready(job, time.time(), os.path.dirname(partial))
        else:
            token = preflight.write_owner(partial)
            claim = runner.Claim(partial, preflight.path_id(partial), token, preflight.path_id(os.path.dirname(partial)))
            eng = SimpleNamespace(failed_tiles=0, n_listed=0, tiles=[], channel_names=["DAPI"],
                                  phys_size_x_um=0.5, n_channels=1)
            outcome = batch._finalize(eng, job, time.time(), claim, True, "", SimpleNamespace(stop=lambda: None))
        written = [e.size for e in events if isinstance(e, OutputWritten)]
        result.put((outcome.status, written))
    except Exception as e:
        result.put((type(e).__name__, str(e)))


@pytest.mark.parametrize("mode", ["ui", "ready", "normal"])
def test_every_finalizer_reserves_verified_generation_across_processes(tmp_path, mode):
    out = tmp_path / "out"
    pdir = out / PARTIAL_DIR
    pdir.mkdir(parents=True)
    partial, final = pdir / "Slide1.ome.tif", out / "Slide1.ome.tif"
    tifffile.imwrite(partial, np.zeros((32, 48), np.uint16), metadata=None)
    verified = partial.read_bytes()
    tifffile.imwrite(final, np.ones((32, 48), np.uint16), metadata=None)
    previous = final.read_bytes()
    if mode != "normal":
        preflight.mark_complete(str(partial))
    ctx = multiprocessing.get_context("spawn")
    paused, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=finalizer, args=(mode, str(out), paused, release, result))
    process.start()
    try:
        assert paused.wait(3)
        with pytest.raises(preflight.ReservationBusy):
            preflight.finalize_kept(str(partial), str(final))
        assert runner.BatchRunner._claim(str(partial)) is None
        assert final.read_bytes() == previous and partial.read_bytes() == verified
        release.set()
        process.join(3)
        assert not process.is_alive() and process.exitcode == 0
        status, written = result.get(timeout=1)
        assert status == "done" and written in (len(verified), [len(verified)])
        pdir.mkdir(exist_ok=True)
        claim = runner.BatchRunner._claim(str(partial))
        assert claim is not None
        partial.write_bytes(b"competing writer's unfinished image")
        owner = Path(str(partial) + ".owner").read_bytes()
        assert final.read_bytes() == verified
        assert preflight.verify_output(str(final))[0] == "valid"
        assert partial.read_bytes() == b"competing writer's unfinished image"
        assert Path(str(partial) + ".owner").read_bytes() == owner
        preflight.release_owner(claim.token)
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(2)
        result.close()


def takeover(out, paused, release, result):
    partial = str(Path(out) / PARTIAL_DIR / "Slide1.ome.tif")
    real_rename = os.rename
    def delayed(a, b, *args, **kwargs):
        answer = real_rename(a, b, *args, **kwargs)
        paused.set()
        if not release.wait(4):
            raise TimeoutError("takeover barrier")
        return answer
    runner.os.rename = delayed
    batch = runner.BatchRunner(1, Plan(out, out), None, lambda ev: None, runner.RunOptions(out))
    try:
        with preflight.reserve_output(partial):
            busy = batch._clear_way(os.path.dirname(partial), os.path.basename(partial), None)
            claim = None if busy else batch._claim(partial)
            result.put(claim.token if claim else busy)
    except Exception as e:
        result.put(str(e))


def test_takeover_holds_reservation_until_replacement_claim(tmp_path):
    out = tmp_path / "out"
    pdir = out / PARTIAL_DIR
    pdir.mkdir(parents=True)
    partial = pdir / "Slide1.ome.tif"
    partial.write_bytes(b"abandoned")
    os.utime(partial, (time.time() - 600,) * 2)
    ctx = multiprocessing.get_context("spawn")
    paused, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=takeover, args=(str(out), paused, release, result))
    process.start()
    try:
        assert paused.wait(3)
        assert not partial.exists()
        assert runner.BatchRunner._claim(str(partial)) is None
        release.set()
        process.join(3)
        token = result.get(timeout=1)
        assert process.exitcode == 0 and preflight.owner_token(str(partial)) == token
        assert partial.exists()
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(2)
        result.close()
