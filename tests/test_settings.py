"""yfs.core.settings (FINAL_SPEC §5.5.8, §5.6; acceptance A5)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from yfs.core import settings as st
from yfs.core.settings import DEFAULTS, LEDGER_MAX, RECENT_MAX, Settings


def test_paths_follow_yfs_config_dir(cfg_dir):
    assert st.config_dir() == str(cfg_dir)
    assert st.log_dir() == os.path.join(str(cfg_dir), "logs")
    s = Settings.load()
    assert s.path == os.path.join(str(cfg_dir), "settings.json") and s.load_error is None


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_per_os_paths(monkeypatch, tmp_path, platform):
    monkeypatch.delenv("YFS_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    home, lad = str(tmp_path / "home"), str(tmp_path / "lad")
    expected = {
        "win32": (os.path.join(lad, "YourFavouriteStitcher"), os.path.join(lad, "YourFavouriteStitcher", "logs")),
        "darwin": (os.path.join(home, "Library", "Application Support", "YourFavouriteStitcher"),
                   os.path.join(home, "Library", "Logs", "YourFavouriteStitcher")),
        "linux": (os.path.join(home, ".config", "yourfavouritestitcher"),
                  os.path.join(home, ".local", "state", "yourfavouritestitcher", "logs")),
    }[platform]
    assert (st.config_dir(), st.log_dir()) == expected


def test_xdg_variables_are_honoured_when_absolute(monkeypatch, tmp_path):
    monkeypatch.delenv("YFS_CONFIG_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert st.config_dir() == os.path.join(str(tmp_path / "cfg"), "yourfavouritestitcher")
    assert st.log_dir() == os.path.join(str(tmp_path / "state"), "yourfavouritestitcher", "logs")
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")          # invalid per the XDG spec: ignored
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert st.config_dir() == os.path.join(str(tmp_path / "home"), ".config", "yourfavouritestitcher")


def test_missing_file_gives_defaults(cfg_dir):
    s = Settings.load()
    assert s.load_error is None and not s.dirty
    for key in ("schema", "paths.browser_last.output", "browser.sort", "discovery.max_depth", "run.temp_dir",
                "ui.scale", "ui.log_height", "last_run.state"):
        node = DEFAULTS
        for part in key.split("."):
            node = node[part]
        assert s.get(key) == node
    assert s.get("no.such.key", "fallback") == "fallback"


def test_deep_merge_keeps_unknown_keys_and_sections(cfg_dir):
    stored = {"schema": 1, "future_key": {"a": 1},
              "ui": {"scale": 1.5, "welcome_dismissed": True, "extra": "kept"},
              "browser": {"sash": 300, "sort": ["size", "desc"]},
              "discovery": {"max_depth": 5},
              "run": {"temp_dir": "/scratch/tmp", "keep_awake": False},
              "paths": "not a dict", "verified_outputs": {"not": "a list"},
              "last_run": {"state": "running", "planned": 4}}
    (cfg_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")
    s = Settings.load()
    assert s.load_error is None
    assert s.get("future_key") == {"a": 1} and s.get("ui.extra") == "kept"
    assert s.get("ui.scale") == 1.5 and s.get("ui.welcome_dismissed") is True and s.get("ui.log_open") is False
    assert s.get("browser.sash") == 300 and s.get("browser.show_hidden") is False
    assert s.get("browser.sort") == ["size", "desc"]
    assert s.get("discovery.naming") == "folder" and s.get("discovery.max_depth") == 5
    assert s.get("run.temp_dir") == "/scratch/tmp" and s.get("run.keep_awake") is False
    assert s.get("paths") == DEFAULTS["paths"]                   # a section never becomes a non-dict
    assert s.get("last_run.state") == "running" and s.get("last_run.done") == 0
    s.save_async()
    assert s.save_now()
    again = json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))
    assert again["future_key"] == {"a": 1} and again["ui"]["extra"] == "kept"


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe\x00garbage", b""])
def test_corrupt_file_becomes_bak(cfg_dir, content):
    (cfg_dir / "settings.json").write_bytes(content)
    s = Settings.load()
    assert s.load_error == st.DAMAGED_TEXT
    assert (cfg_dir / "settings.json.bak").read_bytes() == content
    assert not (cfg_dir / "settings.json").exists()
    assert s.get("ui.scale") == "auto"
    s.set("ui.scale", 1.25)
    assert s.save_now()
    assert json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["ui"]["scale"] == 1.25


def test_a_bom_from_a_windows_editor_is_not_damage(cfg_dir):
    (cfg_dir / "settings.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"ui": {"scale": 1.75}}).encode())
    s = Settings.load()
    assert s.load_error is None and s.get("ui.scale") == 1.75
    assert not (cfg_dir / "settings.json.bak").exists()


def test_get_returns_copies_and_set_creates_parents(cfg_dir):
    s = Settings.load()
    s.get("browser.sort").append("x")
    assert s.get("browser.sort") == ["name", "asc"]
    assert not s.dirty
    s.set("ui.log_open", False)                    # unchanged value: not dirty
    assert not s.dirty
    s.set("a.b.c", [1])
    assert s.dirty and s.get("a.b.c") == [1] and s.get("a") == {"b": {"c": [1]}}


def test_recents_newest_first_deduped_and_capped(cfg_dir, monkeypatch):
    s = Settings.load()
    for i in range(RECENT_MAX + 3):
        s.add_recent("input", f"/data/run{i}", slides=i)
    paths = [r["path"] for r in s.get("recent.inputs")]
    assert len(paths) == RECENT_MAX and paths[0] == f"/data/run{RECENT_MAX + 2}"
    s.add_recent("input", "/data/run5/")                       # same folder: moves to the front, keeps its count
    first = s.get("recent.inputs")[0]
    assert first["path"] == "/data/run5/" and first["slides"] == 5 and "T" in first["used"]
    assert [r["path"] for r in s.get("recent.inputs")].count("/data/run5") == 0
    monkeypatch.setattr(os.path, "normcase", lambda p: p.lower())   # Windows/macOS-like comparison
    s.add_recent("output", "/Out/Stitched")
    assert "slides" not in s.get("recent.outputs")[0]
    s.add_recent("output", "/out/stitched", slides=3)
    outs = s.get("recent.outputs")
    assert [r["path"] for r in outs] == ["/out/stitched"] and outs[0]["slides"] == 3
    s.add_recent("input", "")                                  # ignored
    assert len(s.get("recent.inputs")) == RECENT_MAX


def test_recent_entries_are_sanitised_on_load(cfg_dir):
    junk = {"recent": {"inputs": [{"path": "/a"}, "oops", {"nopath": 1}, {"path": ""}] +
                       [{"path": f"/p{i}"} for i in range(20)], "outputs": "bad"}}
    (cfg_dir / "settings.json").write_text(json.dumps(junk), encoding="utf-8")
    s = Settings.load()
    assert [r["path"] for r in s.get("recent.inputs")][:2] == ["/a", "/p0"]
    assert len(s.get("recent.inputs")) == RECENT_MAX and s.get("recent.outputs") == []


def test_relative_folder_paths_are_dropped_on_load(cfg_dir, monkeypatch):
    """UD6 (DS-11, PL-3): a relative folder would be found again from each working folder the app starts in."""
    stored = {"paths": {"input": "tiles", "output": "/data/Stitched",
                        "browser_last": {"input": "./ProjectC", "output": "/data", "temp": "scratch"}},
              "recent": {"inputs": [{"path": "tiles"}, {"path": "/data/ProjectC", "slides": 6}],
                         "outputs": [{"path": "Stitched"}]}}
    (cfg_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")
    s = Settings.load()
    assert s.get("paths") == {"input": "", "output": "/data/Stitched",
                              "browser_last": {"input": "", "output": "/data", "temp": ""}}
    assert s.get("recent.inputs") == [{"path": "/data/ProjectC", "slides": 6}] and s.get("recent.outputs") == []
    assert not s.dirty                                         # nothing is written back until something changes
    monkeypatch.setattr(sys, "platform", "win32")             # Windows: a drive or UNC part, and a root after it
    for path, kept in (("D:\\Stitched", True), ("\\\\labshare\\data", True), ("D:Stitched", False),
                       ("\\Stitched", False), ("Stitched", False)):
        assert st._full_path(path) is kept, path


def test_damaged_ledger_entries_are_dropped(cfg_dir):
    stored = {"verified_outputs": [["/out/a.ome.tif", 10, 20], ["/out/b.ome.tif", "x", 1], "junk", [1, 2],
                                   ["/out/c.ome.tif", 3, 4], ["/out/d.ome.tif", 5, 6, "many"], "abc",
                                   ["/out/e.ome.tif", 1, 2, 3, 4]]}
    (cfg_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")
    s = Settings.load()
    assert s.ledger_has("/out/a.ome.tif", 10, 20) and s.ledger_has("/out/c.ome.tif", 3, 4)
    assert not s.ledger_has("/out/b.ome.tif", 0, 1) and not s.ledger_has("/out/d.ome.tif", 5, 6)
    assert not s.ledger_has("/out/e.ome.tif", 1, 2) and len(s._ledger) == 2


def test_the_ledger_keeps_the_gaps_of_each_output(cfg_dir, tmp_path):
    """DS-1: an output made with missing tiles is ledgered with its gap count (ledger_get); entries without gaps
    keep the three-item form that older v2 versions read."""
    s = Settings.load()
    gaps, clean = str(tmp_path / "Gaps.ome.tif"), str(tmp_path / "Clean.ome.tif")
    assert s.ledger_get(gaps, 100, 5) is None
    s.ledger_add(gaps, 100, 5, gaps=3)
    s.ledger_add(clean, 7, 8)
    assert s.ledger_get(gaps, 100, 5) == 3 and s.ledger_has(gaps, 100, 5) and s.ledger_get(gaps, 101, 5) is None
    assert s.ledger_get(clean, 7, 8) == 0 and s.ledger_get(clean, 7, 9) is None
    assert s.save_now()
    stored = json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["verified_outputs"]
    assert stored == [[st._ledger_key(gaps), 100, 5, 3], [st._ledger_key(clean), 7, 8]]
    again = Settings.load()
    assert again.ledger_get(gaps, 100, 5) == 3 and again.ledger_get(clean, 7, 8) == 0


def test_ledger_has_add_cap_and_persistence(cfg_dir, tmp_path):
    s = Settings.load()
    out = str(tmp_path / "out" / "Slide.ome.tif")
    assert not s.ledger_has(out, 100, 5)
    s.ledger_add(out, 100, 5)
    assert s.ledger_has(out, 100, 5) and not s.ledger_has(out, 101, 5) and not s.ledger_has(out, 100, 6)
    for i in range(LEDGER_MAX + 10):
        s.ledger_add(str(tmp_path / f"f{i}.ome.tif"), i, i)
    assert not s.ledger_has(out, 100, 5)                       # oldest dropped
    assert not s.ledger_has(str(tmp_path / "f9.ome.tif"), 9, 9) and s.ledger_has(str(tmp_path / "f10.ome.tif"), 10, 10)
    s.ledger_add(str(tmp_path / "f10.ome.tif"), 10, 10)          # re-adding makes it the newest
    s.ledger_add(str(tmp_path / "new.ome.tif"), 1, 1)
    assert s.ledger_has(str(tmp_path / "f10.ome.tif"), 10, 10)
    assert s.save_now()
    stored = json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["verified_outputs"]
    assert len(stored) == LEDGER_MAX and stored[-1][1:] == [1, 1]
    again = Settings.load()
    assert again.ledger_has(str(tmp_path / "new.ome.tif"), 1, 1) and again.get("verified_outputs") is None


def test_writes_are_atomic(cfg_dir, monkeypatch):
    s = Settings.load()
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(st.os, "replace", lambda a, b: (replaced.append((str(a), str(b))), real_replace(a, b))[1])
    stop = threading.Event()
    seen = {"reads": 0, "errors": 0}

    def reader():
        path = cfg_dir / "settings.json"
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
                seen["reads"] += 1
            except FileNotFoundError:
                pass
            except ValueError:
                seen["errors"] += 1
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for i in range(150):
        s.set("big", "x" * (i * 997))
        s.save_async()
    assert s.save_now(timeout=10)
    stop.set()
    t.join(5)
    assert seen["errors"] == 0 and seen["reads"] > 0
    assert replaced and all(os.path.dirname(a) == str(cfg_dir) and os.path.basename(a).startswith(".settings.")
                            and a.endswith(".tmp") and b.endswith("settings.json") for a, b in replaced)
    assert len({a for a, _ in replaced}) == len(replaced)              # DS-14: a new temporary file every time
    assert [n for n in os.listdir(cfg_dir) if n.endswith(".tmp")] == []
    assert len(json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["big"]) == 149 * 997
    assert s._writer is not None and s._writer.daemon


def test_two_instances_saving_at_once_never_damage_the_file(cfg_dir):
    """DS-14: two app instances (two Settings on one file, each with its own writer thread) save at the same
    moments with payloads of different lengths; settings.json is always one complete JSON document."""
    a, b = Settings.load(), Settings.load()
    stop, bad, path = threading.Event(), [], cfg_dir / "settings.json"

    def reader():
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except ValueError:
                bad.append(1)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for i in range(60):
        a.set("last_run", {"state": "running", "pad": "a" * (40000 + i)})
        b.set("last_run", {"state": "running", "pad": "b" * (5000 + i)})
        a.save_async()
        b.save_async()
    assert a.save_now(timeout=20) and b.save_now(timeout=20)
    stop.set()
    t.join(5)
    assert bad == [] and json.loads(path.read_text(encoding="utf-8"))["last_run"]["state"] == "running"
    assert [n for n in os.listdir(cfg_dir) if n.endswith(".tmp")] == []


def test_replace_retries_briefly_when_the_file_is_held(cfg_dir, monkeypatch):
    """Windows: os.replace fails while another instance has settings.json open; a brief retry gets through, and
    a save that still fails leaves no temporary file behind."""
    monkeypatch.setattr(st, "REPLACE_DELAY_S", 0.0)
    real_replace, calls = os.replace, []

    def held(a, b, times):
        calls.append(a)
        if len(calls) <= times:
            raise PermissionError(13, "Access is denied")
        return real_replace(a, b)
    s = Settings.load()
    monkeypatch.setattr(st.os, "replace", lambda a, b: held(a, b, 2))
    s.set("ui.scale", 1.5)
    assert s.save_now(timeout=5) and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(st.os, "replace", lambda a, b: held(a, b, 99))
    s.set("ui.scale", 2.0)
    assert s.save_now(timeout=5) is False and len(calls) == st.REPLACE_TRIES
    assert [n for n in os.listdir(cfg_dir) if n.endswith(".tmp")] == []
    assert json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["ui"]["scale"] == 1.5


def test_save_now_without_changes_returns_at_once(cfg_dir):
    s = Settings.load()
    t0 = time.monotonic()
    assert s.save_now() is True and time.monotonic() - t0 < 0.5
    assert not (cfg_dir / "settings.json").exists()


def test_save_failure_is_reported_not_raised(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    s = Settings.load(str(blocker / "settings.json"))            # its folder is a file
    assert "could not be read" in s.load_error and s.get("ui.scale") == "auto"
    s.set("ui.scale", 2.0)
    assert s.save_now(timeout=5) is False


def test_save_now_times_out_when_the_disk_hangs(cfg_dir, monkeypatch):
    s = Settings.load()
    gate = threading.Event()
    monkeypatch.setattr(s, "_write", lambda text: gate.wait(10) or True)
    s.set("ui.scale", 1.5)
    t0 = time.monotonic()
    assert s.save_now(timeout=0.3) is False
    assert time.monotonic() - t0 < 2
    gate.set()
    assert s.save_now(timeout=5) is True


def test_unicode_and_surrogate_paths_round_trip(cfg_dir):
    s = Settings.load()
    odd = "/data/Spaces & ünïcödé/\udcff-raw-byte"
    s.add_recent("input", odd, slides=2)
    s.set("paths.input", odd)
    assert s.save_now()
    again = Settings.load()
    assert again.get("paths.input") == odd and again.get("recent.inputs")[0]["path"] == odd
