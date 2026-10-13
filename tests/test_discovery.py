"""Discovery: the breadth-first walk of FINAL_SPEC §3.5 on the synthetic tree (acceptance B2), mount boundaries
(R27), wide folders, budgets, stop, Search-deeper seeds, the output folder, cached listings, the cloud depth and
the end-of-walk retry of folders that failed fast. Listings go through fake_fs so the tests can assert which
folders were (not) listed."""
from __future__ import annotations

import os
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from fake_fs import FakeOs
from yfs.core import discovery, fsservice
from yfs.core.discovery import Discovery, DiscoveryOptions, depth_of, seeds_for
from yfs.core.events import CandidateFound, FsDone, LaneHealthChanged, ScanFinished, ScanNote, ScanPaused, ScanProgress
from yfs.core.fsservice import FsFailure, FsService
from yfs.core.model import FsError
from yfs.core.places import MountTable


def table(*mounts) -> MountTable:
    lines = ["1 0 8:1 / / rw - ext4 /dev/sda1 rw"]
    for i, (mp, fstype, src) in enumerate(mounts, 2):
        lines.append(f"{i} 1 0:{i} / {str(mp).replace(' ', chr(92) + '040')} rw - {fstype} {src} rw")
    return MountTable.from_mountinfo_text("\n".join(lines))


def tiles(folder: Path, prefix: str, n: int) -> Path:
    """n empty tile files: names are all discovery looks at."""
    folder.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        (folder / f"{prefix}_[{1000 + 10 * t},2000]_component_data.tif").write_bytes(b"")
    return folder


def wait_until(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


class FlakyFs(FsService):
    """The first `times` listings of the `fail` folders fail at once with `code`, as on a lane that is not answering;
    health() then reports `lane_health` for every lane."""

    def __init__(self, post, mounts, *, os_api, fail, code="unresponsive", times=1, lane_health="ok"):
        super().__init__(post, mounts, os_api=os_api)
        self.left = {str(p): times for p in fail}
        self.code, self.lane_health = code, lane_health

    def submit(self, path, kind, fn, **kw):
        if self.left.get(path, 0) > 0:
            self.left[path] -= 1
            fut: Future = Future()
            fut.set_exception(FsFailure(FsError(self.code, "The location isn't answering.")))
            return fut
        return super().submit(path, kind, fn, **kw)

    def health(self, path_or_lane):
        return self.lane_health


class Unplugged(FsService):
    """auto_enter() says 'disconnected' for the `off` folders, as FsService does while it marks a drive disconnected."""

    def __init__(self, post, mounts, *, os_api, off):
        super().__init__(post, mounts, os_api=os_api)
        self.off = {str(p) for p in off}

    def auto_enter(self, path):
        return "disconnected" if path in self.off else super().auto_enter(path)


class Scan:
    """Runs one Discovery with its own FsService and fake os_api; collects what it posted."""

    def __init__(self, collector, root, mounts=None, fake=None, fs=None, scan_id=1, seeds=None, **opts):
        self.fake = fake or FakeOs()
        self.fs = fs or FsService(collector, mounts or table(), os_api=self.fake)
        self.col, self.root, self.scan_id = collector, str(root), scan_id
        self.d = Discovery(scan_id, self.root, self.fs, collector, DiscoveryOptions(**opts), seeds=seeds)

    def run(self, timeout=30.0) -> ScanFinished:
        self.d.start()
        return self.finished(timeout)

    def finished(self, timeout=30.0) -> ScanFinished:
        return self.col.wait_for(lambda evs: next((e for e in evs if isinstance(e, ScanFinished)
                                                   and e.scan_id == self.scan_id), None), timeout)

    def cands(self):
        return [e.candidate for e in self.col.of_type(CandidateFound) if e.scan_id == self.scan_id]

    def notes(self, kind=None):
        return [e.note for e in self.col.of_type(ScanNote) if e.scan_id == self.scan_id and kind in (None, e.note.kind)]

    def rel(self, path):
        return os.path.relpath(path, self.root)

    def listed(self):
        return {self.rel(p) for p in self.fake.paths("scandir")}


@pytest.fixture
def services():
    made = []
    yield made
    for fs in made:
        fs.shutdown()


def scan(collector, services, root, **kw) -> Scan:
    s = Scan(collector, root, **kw)
    services.append(s.fs)
    return s


def test_b2_synthetic_tree(synth_tree, collector, services):
    s = scan(collector, services, synth_tree)
    fin = s.run()
    got = sorted((c.rel_folder, c.prefix, c.kind, len(c.tile_names)) for c in s.cands())
    assert got == [
        ("ProjectA/SlideA1_Scan1", "SlideA1_Scan1", "component", 9), ("ProjectA/SlideA2_Scan1", "SlideA2_Scan1",
                                                                       "component", 4),
        ("Spaces & ünïcödé/Slide [weird] (1)", "Slide [weird] (1)", "component", 1),
        ("duplicates/batchA/Slide9", "Slide9_Scan1", "component", 2),
        ("duplicates/batchB/Slide9", "Slide9_Scan1", "component", 2),
        ("inform_export", "S1_Scan1", "component", 4), ("inform_export", "S1_Scan1", "other_suffix", 4),
        ("inform_export", "S2_Scan1", "component", 6),
        ("round1_unmix/R1-01_Scan1", "R1-01_Scan1", "component", 4), ("round1_unmix/R1-02", "R1-02_Scan1", "component", 2),
        ("suspicious/SusSlide_Scan1", "SusSlide_Scan1", "component", 2), ("unrelated/WSIs", "", "legacy_all", 0)]
    inform = [c for c in s.cands() if c.rel_folder == "inform_export"]
    assert all(c.multi_slide_folder and c.inform_export for c in inform)
    notes = sorted((n.kind, s.rel(n.path)) for n in s.notes())
    assert notes == [("empty", "empty_dir"), ("empty", "round1_unmix/ImageAnalysisPipeline"),
                     ("no_access", "locked"), ("no_images", "unrelated/big_flat"),
                     ("not_searched_depth", "unrelated/level0/level1"), ("outputs", "ProjectA"),
                     ("symlink_skipped", "unrelated"), ("wsi", "round1_unmix")]
    listed = s.listed()
    assert ".hidden_dir" not in listed and "unrelated/loop_to_root" not in listed
    assert "unrelated/level0/level1" in listed and "unrelated/level0/level1/level2" not in listed
    assert not any(p.startswith("ProjectA/SlideA1_Scan1/") for p in listed)       # slide folders have no subfolders
    assert (fin.slides_found, fin.stopped, fin.pruned, fin.max_depth) == (10, False, (), 3)
    assert [s.rel(p) for p in fin.frontier] == ["unrelated/level0/level1"]
    assert fin.dirs_done == len(listed) - 1 and [s.rel(p) for p in fin.failed] == ["locked"]
    assert not [e for e in collector.of_type(ScanPaused)]


def test_symlinked_tiles_are_a_slide(synth_tree, tmp_path, collector, services):
    slide = tmp_path / "linked" / "SlideL_Scan1"
    slide.mkdir(parents=True)
    for tile in (synth_tree / "ProjectA" / "SlideA2_Scan1").iterdir():
        os.symlink(tile, slide / tile.name)
    os.symlink(synth_tree / "ProjectA", tmp_path / "linked" / "folder_link")
    s = scan(collector, services, tmp_path / "linked")
    s.run()
    [c] = s.cands()
    assert (c.folder_name, c.prefix, len(c.tile_names)) == ("SlideL_Scan1", "SlideA2_Scan1", 4)
    assert [(n.kind, n.count) for n in s.notes()] == [("symlink_skipped", 1)]
    assert "folder_link" not in s.listed()


def test_mount_boundaries(synth_tree, collector, services):
    mounts = table((synth_tree / "ProjectA", "fuse.rclone", "remote:"), (synth_tree / "suspicious", "proc", "proc"),
                   (synth_tree / "round1_unmix", "cifs", "//fileserver.example.ac.uk/share"),
                   (synth_tree / "duplicates", "autofs", "systemd-1"))
    s = scan(collector, services, synth_tree, mounts=mounts)
    s.run()
    listed = s.listed()
    assert "ProjectA" not in listed and "suspicious" not in listed and "duplicates" not in listed
    assert "round1_unmix" in listed and "round1_unmix/R1-02" in listed                # network: entered
    assert [s.rel(n.path) for n in s.notes("cloud_skipped")] == ["ProjectA"]
    [skipped] = s.notes("not_mounted")                    # one note for the folder that holds them (FS-2)
    assert (s.rel(skipped.path), skipped.count, skipped.detail, skipped.names) == \
        (".", 2, "autofs, pseudo", ("duplicates", "suspicious"))
    folders = {c.rel_folder for c in s.cands()}
    assert "round1_unmix/R1-02" in folders and not any(f.startswith(("ProjectA", "suspicious", "duplicates"))
                                                       for f in folders)


def test_not_mounted_follows_the_live_auto_enter(synth_tree, collector, services):
    """FS-2: discovery asks fs.auto_enter() for every folder, so a drive that answers again is entered next time."""
    fake = FakeOs()
    fs = Unplugged(collector, table(), os_api=fake, off=[synth_tree / "ProjectA", synth_tree / "duplicates"])
    s = scan(collector, services, synth_tree, fs=fs, fake=fake)
    s.run()
    [n] = s.notes("not_mounted")
    assert (s.rel(n.path), n.count, n.detail, n.names) == (".", 2, "disconnected", ("duplicates", "ProjectA"))
    assert not any(p.startswith(("ProjectA", "duplicates")) for p in s.listed())
    fs.off.clear()                                        # reconnected; the root's listing comes from the cache
    s2 = scan(collector, services, synth_tree, fs=fs, fake=fake, scan_id=2)
    s2.run()
    assert not s2.notes("not_mounted") and {"ProjectA/SlideA1_Scan1", "duplicates/batchB/Slide9"} <= \
        {c.rel_folder for c in s2.cands()}


def test_cloud_root_is_searched_one_level(synth_tree, collector, services):
    s = scan(collector, services, synth_tree / "duplicates",
             mounts=table((synth_tree / "duplicates", "fuse.rclone", "remote:")))
    fin = s.run()
    assert s.fs.concurrency(s.root) == 1 and fin.max_depth == 1
    assert s.listed() == {".", "batchA", "batchB"} and s.cands() == []
    assert [s.rel(n.path) for n in s.notes("not_searched_depth")] == ["batchA", "batchB"]
    assert not s.notes("cloud_skipped")


@pytest.mark.parametrize("max_depth, extra_depth, searched", [(3, 0, 1), (6, 3, 4), (2, 3, 2)])
def test_search_deeper_goes_deeper_on_a_cloud_root(synth_tree, collector, services, max_depth, extra_depth,
                                                   searched):
    """PL-16 / FS-10: a cloud root searches min(max_depth, 1 + extra_depth) levels, and says so."""
    s = scan(collector, services, synth_tree / "duplicates", max_depth=max_depth, extra_depth=extra_depth,
             mounts=table((synth_tree / "duplicates", "fuse.rclone", "remote:")))
    fin = s.run()
    assert fin.max_depth == searched
    assert sorted(c.rel_folder for c in s.cands()) == ([] if searched < 2 else ["batchA/Slide9", "batchB/Slide9"])


def test_output_folder_listed_not_descended(synth_tree, collector, services):
    s = scan(collector, services, synth_tree, out_dir=str(synth_tree / "ProjectA"))
    s.run()
    assert "ProjectA" in s.listed() and "ProjectA/SlideA1_Scan1" not in s.listed()
    assert not [c for c in s.cands() if c.rel_folder.startswith("ProjectA")]
    assert [s.rel(n.path) for n in s.notes("outputs")] == ["ProjectA"]
    slide = synth_tree / "ProjectA" / "SlideA1_Scan1"
    s2 = scan(collector, services, synth_tree / "ProjectA", out_dir=str(slide), scan_id=2)
    s2.run()
    assert {c.rel_folder for c in s2.cands()} == {"SlideA1_Scan1", "SlideA2_Scan1"}    # tiles in it still found


def stray_tiles_tree(root: Path) -> Path:
    tiles(root, "test_Scan1", 3)                                    # a few test tiles in the chosen folder
    for i in (1, 2, 3):
        tiles(root / f"Slide{i}_Scan1", f"Slide{i}_Scan1", 12)
    tiles(root / "Slide1_Scan1" / "rerun", "Slide1r_Scan1", 12)     # a slide folder inside a slide folder
    (root / "Slide2_Scan1" / "stitched" / "old").mkdir(parents=True)
    return root


def test_subfolders_of_slide_folders_are_searched(tmp_path, collector, services):
    """FS-11: stray tiles do not hide the slide folders below them; the output folder is still only listed."""
    root = stray_tiles_tree(tmp_path / "Project")
    s = scan(collector, services, root, out_dir=str(root / "Slide2_Scan1" / "stitched"))
    fin = s.run()
    assert sorted((c.rel_folder, len(c.tile_names)) for c in s.cands()) == [
        (".", 3), ("Slide1_Scan1", 12), ("Slide1_Scan1/rerun", 12), ("Slide2_Scan1", 12), ("Slide3_Scan1", 12)]
    assert "Slide2_Scan1/stitched" in s.listed() and "Slide2_Scan1/stitched/old" not in s.listed()
    assert (fin.slides_found, fin.frontier, fin.failed) == (5, (), ())


def test_subfolders_of_a_slide_folder_at_the_depth_limit_are_the_frontier(tmp_path, collector, services):
    root = stray_tiles_tree(tmp_path / "Project")
    s = scan(collector, services, root, max_depth=1)
    fin = s.run()
    assert "Slide1_Scan1/rerun" not in s.listed() and len(s.cands()) == 4
    assert sorted(s.rel(p) for p in fin.frontier) == ["Slide1_Scan1", "Slide2_Scan1"]
    assert sorted((s.rel(n.path), n.count) for n in s.notes("not_searched_depth")) == [("Slide1_Scan1", 1),
                                                                                       ("Slide2_Scan1", 1)]


def make_wide(root: Path, n=250, slide_at=None) -> Path:
    for i in range(n):
        (root / f"case_{i:03d}" / "inner").mkdir(parents=True)
    if slide_at is not None:
        (root / f"case_{slide_at:03d}" / "X_[1,2]_component_data.tif").write_bytes(b"")
    return root


def test_wide_folder_is_pruned_and_no_prune_searches_it(tmp_path, collector, services):
    root = make_wide(tmp_path / "wide")
    s = scan(collector, services, tmp_path)
    fin = s.run()
    [note] = s.notes("wide_pruned")
    assert (s.rel(note.path), note.count, note.detail) == ("wide", 200, "")    # detail: no sample folder failed
    assert [s.rel(p) for p in fin.pruned] == ["wide"]
    cases = {p for p in s.listed() if p.startswith("wide/case_") and p.count("/") == 1}
    assert cases == {f"wide/case_{i:03d}" for i in range(50)}                    # the first 50, natural order
    s2 = scan(collector, services, tmp_path, scan_id=2, no_prune=frozenset({str(root)}))
    s2.run()
    assert not s2.notes("wide_pruned") and len([p for p in s2.listed() if p.count("/") == 1]) == 250


def test_wide_folder_with_tiles_in_the_sample_is_searched_fully(tmp_path, collector, services):
    make_wide(tmp_path / "wide", slide_at=7)
    s = scan(collector, services, tmp_path)
    s.run()
    assert not s.notes("wide_pruned") and len([p for p in s.listed() if p.count("/") == 1]) == 250
    assert [c.rel_folder for c in s.cands()] == ["wide/case_007"]


def cohort(root: Path, n=250) -> Path:
    """Study/<patient>/Scan1/tiles: n patient folders, the tiles one level below them."""
    for i in range(1, n + 1):
        tiles(root / f"P{i:03d}" / "Scan1", f"P{i:03d}_Scan1", 1)
    return root


def test_wide_folder_counts_tiles_anywhere_in_its_sample(tmp_path, collector, services):
    """FS-4: a cohort laid out as Study/<patient>/Scan1/tiles is searched fully: all 250 slides."""
    s = scan(collector, services, cohort(tmp_path / "Study"))
    fin = s.run()
    assert (fin.slides_found, len(s.cands()), fin.pruned) == (250, 250, ())
    assert not s.notes("wide_pruned")


@pytest.mark.parametrize("below, code", [("", "no_access"), ("Scan1", "no_access"), ("", "unresponsive")])
def test_wide_folder_whose_sample_could_not_be_searched_is_searched_fully(tmp_path, collector, services, below,
                                                                          code):
    """FS-4 with failures: none of the first 50 folders could be searched completely (they or the folders in them are
    locked, or their lane still isn't answering at the end of the walk). Nothing then says that the folder holds no
    slides, so the rest is searched too."""
    root = cohort(tmp_path / "Study")
    bad = [root / f"P{i:03d}" / below for i in range(1, 51)]
    fake = FakeOs()
    if code == "no_access":
        for p in bad:
            fake.fail(p, PermissionError(13, "Permission denied"))
        fs = FsService(collector, table(), os_api=fake)
    else:
        fs = FlakyFs(collector, table(), os_api=fake, fail=bad, lane_health="unresponsive")
    s = scan(collector, services, root, fs=fs, fake=fake)
    fin = s.run()
    assert (fin.slides_found, fin.pruned) == (200, ()) and not s.notes("wide_pruned")
    assert sorted(fin.failed) == sorted(map(str, bad)) and len(s.notes(code)) == 50


def test_wide_folder_is_still_pruned_when_part_of_its_sample_failed(tmp_path, collector, services):
    """FS-4: sample folders searched without tiles still prune the rest when other sample folders failed. The note's
    detail tells how many of the 50 couldn't be searched completely."""
    root = make_wide(tmp_path / "wide")
    fake = FakeOs()
    for p in (root / "case_003", root / "case_010" / "inner"):
        fake.fail(p, PermissionError(13, "Permission denied"))
    s = scan(collector, services, tmp_path, fake=fake)
    fin = s.run()
    [note] = s.notes("wide_pruned")
    assert (s.rel(note.path), note.count, note.detail) == ("wide", 200, "2")
    assert [s.rel(p) for p in fin.pruned] == ["wide"]
    assert sorted(s.rel(p) for p in fin.failed) == ["wide/case_003", "wide/case_010/inner"]


def nested_wide(root: Path) -> Path:
    """201 folders; f003, in the sample, holds 201 more."""
    for i in range(201):
        (root / f"f{i:03d}").mkdir(parents=True)
    for j in range(201):
        (root / "f003" / f"g{j:03d}").mkdir()
    return root


@pytest.mark.parametrize("slide_at", [None, "f003/g010"])
def test_wide_folder_inside_a_wide_folders_sample(tmp_path, collector, services, slide_at):
    """FS-4: the inner wide folder is decided first and counts for the outer one; tiles in it decide both."""
    root = nested_wide(tmp_path / "top")
    if slide_at:
        tiles(root / slide_at, "Deep_Scan1", 1)
    s = scan(collector, services, root)
    fin = s.run()
    if slide_at:
        assert fin.pruned == () and [c.rel_folder for c in s.cands()] == [slide_at] and len(s.listed()) == 403
    else:
        assert [s.rel(p) for p in fin.pruned] == ["f003", "."]
        assert [(s.rel(n.path), n.count) for n in s.notes("wide_pruned")] == [("f003", 151), (".", 151)]
        assert len(s.listed()) == 1 + 50 + 50


def test_wide_folder_inside_a_wide_folders_sample_that_could_not_be_searched(tmp_path, collector, services):
    """FS-4: the inner wide folder's own sample is locked, so its rest is searched too, as part of the outer one's
    sample. The outer one is pruned (its other sample folders were searched), noting the one sample folder that
    couldn't be searched completely."""
    root = nested_wide(tmp_path / "top")
    fake = FakeOs()
    for j in range(50):
        fake.fail(root / "f003" / f"g{j:03d}", PermissionError(13, "Permission denied"))
    s = scan(collector, services, root, fake=fake)
    fin = s.run()
    assert [s.rel(p) for p in fin.pruned] == ["."] and len(fin.failed) == 50
    assert [(s.rel(n.path), n.count, n.detail) for n in s.notes("wide_pruned")] == [(".", 151, "1")]
    assert {f"f003/g{j:03d}" for j in range(201)} <= s.listed() and len(s.listed()) == 1 + 50 + 201


def test_wide_folder_waits_for_a_sample_folder_that_failed_fast(tmp_path, collector, services):
    """FS-4 with FS-6: a sample folder that is listed again at the end still counts for the wide decision."""
    make_wide(tmp_path / "wide", slide_at=7)
    fake = FakeOs()
    fs = FlakyFs(collector, table(), os_api=fake, fail=[tmp_path / "wide" / "case_007"])
    s = scan(collector, services, tmp_path, fs=fs, fake=fake)
    fin = s.run()
    assert not s.notes("wide_pruned") and len([p for p in s.listed() if p.count("/") == 1]) == 250
    assert [c.rel_folder for c in s.cands()] == ["wide/case_007"] and fin.failed == ()


def test_budget_pause_resume_and_stop(synth_tree, collector, services):
    s = scan(collector, services, synth_tree, budget_dirs=10)
    s.d.start()
    paused = collector.wait_type(ScanPaused)[0]
    assert paused.reason == "budget" and paused.dirs_done >= 10 and s.d.is_alive()
    time.sleep(0.3)
    assert not collector.of_type(ScanFinished)                                  # really waiting
    s.d.resume()
    collector.wait_type(ScanPaused, count=2)
    s.d.resume()
    fin = s.finished()
    assert not fin.stopped and fin.slides_found == 10
    assert collector.of_type(ScanProgress) and collector.of_type(ScanProgress)[-1].scan_id == 1
    s2 = scan(collector, services, synth_tree, scan_id=2, budget_dirs=5)
    s2.d.start()
    collector.wait_for(lambda evs: [e for e in evs if isinstance(e, ScanPaused) and e.scan_id == 2])
    s2.d.stop()
    fin2 = s2.finished()
    assert fin2.stopped and fin2.dirs_done < 24 and fin2.max_depth == 3


@pytest.mark.parametrize("fstype, searched", [("ext4", 3), ("fuse.rclone", 1)])
def test_stop_while_listing(tmp_path, collector, services, fstype, searched):
    for n in ("a", "b", "c"):
        (tmp_path / n).mkdir()
    fake = FakeOs()
    gate = fake.block(tmp_path / "b")
    try:
        s = scan(collector, services, tmp_path, fake=fake, mounts=table((tmp_path, fstype, "src")))
        s.d.start()
        while not fake.inside:
            time.sleep(0.01)
        t0 = time.monotonic()
        s.d.stop()
        fin = s.finished(timeout=3)
        assert fin.stopped and time.monotonic() - t0 < 1.0
        assert fin.max_depth == searched                  # also a stopped scan tells the depth it searched
    finally:
        gate.set()


def test_stop_keeps_folders_that_failed_fast_as_failed(tmp_path, collector, services):
    """FS-6: stopped before the end-of-walk retry, a folder that failed fast is reported, not forgotten."""
    for n in ("a_flaky", "b_blocked"):
        (tmp_path / n).mkdir()
    fake = FakeOs()
    gate = fake.block(tmp_path / "b_blocked")
    try:
        fs = FlakyFs(collector, table(), os_api=fake, fail=[tmp_path / "a_flaky"])
        s = scan(collector, services, tmp_path, fs=fs, fake=fake)
        s.d.start()
        while not fake.inside:                            # a_flaky was refused before b_blocked was submitted
            time.sleep(0.01)
        s.d.stop()
        fin = s.finished(timeout=3)
        assert fin.stopped and [s.rel(p) for p in fin.failed] == ["a_flaky"]
        assert [s.rel(n.path) for n in s.notes("unresponsive")] == ["a_flaky"]
    finally:
        gate.set()


def test_stop_while_a_folder_is_listed_again_keeps_it_as_failed(tmp_path, collector, services):
    """FS-6: stopped while the end-of-walk retry lists a folder that failed fast (it hangs this time), the folder is
    still reported, with its first error."""
    for n in ("a_flaky", "b"):
        (tmp_path / n).mkdir()
    fake = FakeOs()
    gate = fake.block(tmp_path / "a_flaky")              # only the retry gets here: the first listing was refused
    try:
        fs = FlakyFs(collector, table(), os_api=fake, fail=[tmp_path / "a_flaky"])
        s = scan(collector, services, tmp_path, fs=fs, fake=fake)
        s.d.start()
        wait_until(lambda: fake.inside)
        s.d.stop()
        fin = s.finished(timeout=3)
        assert fin.stopped and [s.rel(p) for p in fin.failed] == ["a_flaky"]
        assert [(s.rel(n.path), n.detail) for n in s.notes("unresponsive")] == [("a_flaky",
                                                                                 "The location isn't answering.")]
    finally:
        gate.set()


def test_search_deeper_seeds(synth_tree, collector, services):
    s = scan(collector, services, synth_tree)
    fin = s.run()
    depth, seeds, rounds = 3, seeds_for(s.root, fin.frontier), 0
    assert seeds == [(str(synth_tree / "unrelated" / "level0" / "level1"), 3)]
    while seeds and rounds < 5:
        rounds += 1
        depth += 3
        sd = scan(collector, services, synth_tree, fs=s.fs, fake=s.fake, scan_id=1 + rounds, seeds=seeds,
                  max_depth=depth)
        seeds = seeds_for(s.root, sd.run().frontier)
        if any(c.prefix == "DeepSlide_Scan1" for c in sd.cands()):
            break
    deep = [c for c in sd.cands() if c.prefix == "DeepSlide_Scan1"]
    assert rounds == 3 and deep and deep[0].depth == 11
    assert depth_of(s.root, s.root) == 0 and depth_of(s.root, deep[0].folder) == 11


def test_warm_cache_and_unresponsive_folder(synth_tree, collector, services, monkeypatch):
    monkeypatch.setitem(fsservice.DEADLINE["list_discovery"], "network", 0.5)
    fake = FakeOs()
    gate = fake.block(synth_tree / "unrelated")
    try:
        mounts = table((synth_tree / "unrelated", "cifs", "//fileserver.example.ac.uk/unrelated"))
        s = scan(collector, services, synth_tree, mounts=mounts, fake=fake)
        fin = s.run(timeout=10)
        [note] = s.notes("unresponsive")
        assert s.rel(note.path) == "unrelated" and note.detail
        assert s.fs.health(str(synth_tree / "unrelated")) == "unresponsive"
        assert len(s.cands()) == 11 and not [c for c in s.cands() if c.kind == "legacy_all"]   # the rest was searched
        assert sorted(s.rel(p) for p in fin.failed) == ["locked", "unrelated"]
        calls = len(fake.paths("scandir"))
        s2 = scan(collector, services, synth_tree, fs=s.fs, fake=fake, scan_id=2)
        fin2 = s2.run(timeout=10)
        assert [s.rel(p) for p in fake.paths("scandir")[calls:]] == ["locked"]    # the rest from the warm cache
        assert len(s2.cands()) == 11 and s2.notes("unresponsive")                # fail-fast: same note at once
        assert sorted(s.rel(p) for p in fin2.failed) == ["locked", "unrelated"]  # no retry: still not answering
    finally:
        gate.set()


def test_capped_browser_listing_is_listed_again(tmp_path, collector, services):
    """FS-1: the browser caps its listings without looking at tile names; discovery must not use such a listing."""
    folder = tiles(tmp_path / "batch", "P1_Scan1", 300)
    fake = FakeOs()
    fs = FsService(collector, table(), os_api=fake)
    rid = fs.list_dir(str(folder), ticket=fs.new_ticket(), owner="browser", cap=100)
    ev = collector.wait_for(lambda evs: next((e for e in evs if isinstance(e, FsDone) and e.req_id == rid), None))
    assert ev.result.truncated and len(ev.result.entries) == 100 and fs.cached_listing(str(folder)) is ev.result
    s = scan(collector, services, folder, fs=fs, fake=fake)
    s.run()
    [c] = s.cands()
    assert len(c.tile_names) == 300 and len(fake.paths("scandir")) == 2


@pytest.mark.parametrize("tiles_in_it, browser_cap, reused", [(False, 25, True), (False, 15, False),
                                                               (True, 25, False)])
def test_capped_listing_is_reused_only_where_discovery_would_cap(tmp_path, collector, services, monkeypatch,
                                                                 tiles_in_it, browser_cap, reused):
    """FS-1: a capped cached listing is reused only if discovery's own cap would have stopped there too: at least
    LIST_CAP_NO_TILES entries and no tile name among them."""
    monkeypatch.setattr(discovery, "LIST_CAP_NO_TILES", 20)
    folder = tmp_path / "big"
    if tiles_in_it:
        tiles(folder, "P1_Scan1", 30)
    else:
        folder.mkdir()
        for k in range(30):
            (folder / f"patch_{k:03d}.png").write_bytes(b"")
    fake = FakeOs()
    fs = FsService(collector, table(), os_api=fake)
    rid = fs.list_dir(str(folder), ticket=fs.new_ticket(), owner="browser", cap=browser_cap)
    collector.wait_for(lambda evs: any(isinstance(e, FsDone) and e.req_id == rid for e in evs))
    s = scan(collector, services, folder, fs=fs, fake=fake)
    s.run()
    assert len(fake.paths("scandir")) == (1 if reused else 2)
    if tiles_in_it:
        assert [len(c.tile_names) for c in s.cands()] == [30]
    else:
        assert not s.cands() and [n.kind for n in s.notes()] == ["large_dir"]


def test_cache_age_zero_lists_every_folder_again(tmp_path, collector, services):
    """FS-3: a user's rescan (cache_age=0) sees new slide folders and new tiles at once."""
    tiles(tmp_path / "S1_Scan1", "S1_Scan1", 2)
    tiles(tmp_path / "S2_Scan1", "S2_Scan1", 2)
    s = scan(collector, services, tmp_path)
    s.run()
    tiles(tmp_path / "S2_Scan1", "S2_Scan1", 3)                     # the export went on
    tiles(tmp_path / "S3_Scan1", "S3_Scan1", 2)                     # a forgotten slide folder was copied in
    calls = len(s.fake.paths("scandir"))
    s2 = scan(collector, services, tmp_path, fs=s.fs, fake=s.fake, scan_id=2)
    s2.run()
    assert {c.prefix: len(c.tile_names) for c in s2.cands()} == {"S1_Scan1": 2, "S2_Scan1": 2}   # 120 s reuse
    assert len(s.fake.paths("scandir")) == calls
    s3 = scan(collector, services, tmp_path, fs=s.fs, fake=s.fake, scan_id=3, cache_age=0)
    s3.run()
    assert {c.prefix: len(c.tile_names) for c in s3.cands()} == {"S1_Scan1": 2, "S2_Scan1": 3, "S3_Scan1": 2}
    assert len(s.fake.paths("scandir")) == calls + 4                # the root and its three slide folders


def test_folders_that_failed_fast_are_listed_again_at_the_end(synth_tree, collector, services):
    """FS-6: folders refused by a lane that was not answering are listed once more when it answers again."""
    fake = FakeOs()
    fs = FlakyFs(collector, table(), os_api=fake, fail=[synth_tree / "ProjectA", synth_tree / "duplicates" / "batchB"])
    s = scan(collector, services, synth_tree, fs=fs, fake=fake)
    fin = s.run()
    assert fin.slides_found == 10 and {"ProjectA/SlideA2_Scan1", "duplicates/batchB/Slide9"} <= \
        {c.rel_folder for c in s.cands()}
    assert [s.rel(p) for p in fin.failed] == ["locked"] and not s.notes("unresponsive")


@pytest.mark.parametrize("code, times, lane_health", [("unresponsive", 1, "unresponsive"),
                                                      ("unresponsive", 2, "ok"), ("timeout", 1, "ok")])
def test_folders_that_cannot_be_listed_are_reported(synth_tree, collector, services, code, times, lane_health):
    """FS-6: no retry while the lane is still not answering; only one retry; a folder that timed out is not retried."""
    fake = FakeOs()
    fs = FlakyFs(collector, table(), os_api=fake, fail=[synth_tree / "ProjectA"], code=code, times=times,
                 lane_health=lane_health)
    s = scan(collector, services, synth_tree, fs=fs, fake=fake)
    fin = s.run()
    assert sorted(s.rel(p) for p in fin.failed) == ["ProjectA", "locked"]
    assert [(s.rel(n.path), n.detail) for n in s.notes("unresponsive")] == [("ProjectA",
                                                                             "The location isn't answering.")]
    assert not any(p.startswith("ProjectA") for p in s.listed()) and fin.slides_found == 8


def test_folders_refused_by_a_hung_lane_are_listed_once_it_answers(tmp_path, collector, services, monkeypatch):
    """FS-6 on real lanes: a1 and a2 stall past the deadline, so the share's lane stops answering and b1 and b2 are
    refused at once (a2 may be refused first too, while it waits behind a1, and then stall on its retry). They are
    listed at the end of the walk, after a1 answered, while zz1 and zz2 (on the local lane) keep the walk going."""
    monkeypatch.setitem(fsservice.DEADLINE["list_discovery"], "network", 0.5)
    share = tmp_path / "share"
    for b in ("b1", "b2"):
        tiles(share / b, f"{b}_Scan1", 2)
    for p in (share / "a1", share / "a2", tmp_path / "zz1", tmp_path / "zz2"):
        p.mkdir()
    fake = FakeOs()
    gates = {n: fake.block(p) for n, p in (("a1", share / "a1"), ("a2", share / "a2"), ("zz1", tmp_path / "zz1"),
                                          ("zz2", tmp_path / "zz2"))}
    try:
        s = scan(collector, services, tmp_path, fake=fake, mounts=table((share, "cifs", "//fileserver.example.org/x")))
        s.d.start()
        collector.wait_for(lambda evs: any(isinstance(e, LaneHealthChanged) and e.health == "unresponsive"
                                           for e in evs))
        time.sleep(0.2)                                   # b1 and b2 are refused meanwhile
        gates["a1"].set()
        collector.wait_for(lambda evs: s.fs.health(str(share)) in ("ok", "slow"))
        gates["zz1"].set()
        gates["zz2"].set()
        fin = s.finished()
        assert sorted(c.rel_folder for c in s.cands()) == ["share/b1", "share/b2"]
        assert sorted(s.rel(p) for p in fin.failed) == ["share/a1", "share/a2"]
        assert sorted(s.rel(n.path) for n in s.notes("unresponsive")) == ["share/a1", "share/a2"]
    finally:
        for g in gates.values():
            g.set()


def test_a_folder_that_hangs_does_not_hide_the_slides_beside_it(tmp_path, collector, services, monkeypatch):
    """A share's lane lists two folders at once: while a1 hangs past its deadline, its second thread lists the
    folders beside it, so only a1 is reported (the lane's idle thread must not keep a second one from starting)."""
    monkeypatch.setitem(fsservice.DEADLINE["list_discovery"], "network", 0.5)
    share = tmp_path / "share"
    (share / "a1").mkdir(parents=True)
    for k in range(1, 6):
        tiles(share / f"b{k}", f"b{k}_Scan1", 2)
    fake = FakeOs()
    gate = fake.block(share / "a1")
    try:
        s = scan(collector, services, share, fake=fake, mounts=table((share, "cifs", "//fileserver.example.org/x")))
        fin = s.run()
        assert sorted(c.rel_folder for c in s.cands()) == ["b1", "b2", "b3", "b4", "b5"]
        assert [s.rel(p) for p in fin.failed] == ["a1"]
    finally:
        gate.set()


def test_found_slides_never_open_a_file(synth_tree, collector, services):
    s = scan(collector, services, synth_tree)
    s.run()
    assert set(op for op, _ in s.fake.calls) <= {"scandir", "stat"}
    assert all(os.path.isdir(p) for op, p in s.fake.calls if op == "stat")      # stat_self of folders only


def test_stop_at_budget_pause_boundary_finishes_once(tmp_path, collector, services, monkeypatch):
    tiles(tmp_path / "Slide1", "Slide1", 2)
    s = scan(collector, services, tmp_path, budget_dirs=1)
    clear = s.d._wake.clear
    def stop_before_clear():
        s.d.stop()
        clear()
    monkeypatch.setattr(s.d._wake, "clear", stop_before_clear)
    try:
        result = s.run(timeout=1)
        s.d.join(1)
        assert result.stopped and not s.d.is_alive()
        assert len(collector.of_type(ScanFinished)) == 1
        assert not collector.of_type(ScanPaused)
    finally:
        s.d.stop()


def test_darwin_case_distinct_entries_survive_discovery(tmp_path, collector, services, monkeypatch):
    from yfs.core import names
    monkeypatch.setattr(names, "_CASE_INSENSITIVE_FS", True)
    for folder in ("Batch", "batch"):
        for prefix in ("Slide1", "slide1"):
            tiles(tmp_path / folder, prefix, 12)
    s = scan(collector, services, tmp_path)
    s.run()
    assert len(s.cands()) == len({candidate.key for candidate in s.cands()}) == 4
