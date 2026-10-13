"""tests/fixtures/make_synth.py replaces only folders it made and writes to a temporary folder by default (PL-11)."""
from __future__ import annotations

import os
import stat
import tempfile

import pytest

from tests.conftest import FIXTURES, REPO, load_module   # not "conftest": tests/gui/ has its own

synth = load_module("yfs_test_make_synth_cli", FIXTURES / "make_synth.py")


@pytest.fixture
def quick(monkeypatch, tmp_path):
    """make_synth with empty tile files: the folder handling is under test, not the pixels."""
    monkeypatch.setattr(synth, "write_tile", lambda path, *a, **k: open(path, "wb").close())
    yield synth
    for locked in tmp_path.glob("**/locked"):        # let pytest clean the temp tree up afterwards
        os.chmod(locked, 0o755)


def test_a_folder_it_did_not_make_is_refused_and_left_alone(quick, tmp_path):
    root = tmp_path / "scans_test"
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "data.txt").write_text("irreplaceable\n")
    os.chmod(root / "keep", 0o555)                   # the old code chmod-ed every folder to 0755, then deleted it
    try:
        with pytest.raises(FileExistsError, match="not made by make_synth.py"):
            quick.main(str(root))
        assert [p.name for p in root.iterdir()] == ["keep"]
        assert (root / "keep" / "data.txt").read_text() == "irreplaceable\n"
        assert stat.S_IMODE((root / "keep").stat().st_mode) == 0o555
    finally:
        if (root / "keep").exists():
            os.chmod(root / "keep", 0o755)


def test_the_command_line_refuses_with_exit_code_2(quick, tmp_path, capsys):
    root = tmp_path / "scans_test"
    root.mkdir()
    (root / "data.txt").write_text("x")
    assert quick.cli([str(root)]) == 2
    assert "--force" in capsys.readouterr().err and [p.name for p in root.iterdir()] == ["data.txt"]


def test_a_link_is_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "data.txt").write_text("x")
    os.symlink(tmp_path / "real", tmp_path / "link")
    with pytest.raises(FileExistsError, match="is a link"):
        synth.main(str(tmp_path / "link"), force=True)
    assert (tmp_path / "real" / "data.txt").exists()


def test_a_tree_it_made_is_replaced_locked_folder_included(quick, tmp_path):
    root = tmp_path / "synth"
    quick.main(str(root))
    assert (root / quick.MARKER).is_file() and stat.S_IMODE((root / "locked").stat().st_mode) == 0
    (root / "ProjectA" / "stale.txt").write_text("from an earlier run")
    quick.main(str(root))                            # the old code failed here: it never unlocked "locked"
    assert not (root / "ProjectA" / "stale.txt").exists() and (root / "ProjectA" / "SlideA1_Scan1").is_dir()
    assert os.readlink(root / "unrelated" / "loop_to_root") == str(root)


def test_force_replaces_any_folder(quick, tmp_path):
    root = tmp_path / "scans_test"
    root.mkdir()
    (root / "data.txt").write_text("x")
    quick.main(str(root), force=True)
    assert not (root / "data.txt").exists() and (root / quick.MARKER).is_file()


def test_an_empty_folder_is_used_as_it_is(quick, tmp_path):
    quick.main(str(tmp_path))
    assert (tmp_path / quick.MARKER).is_file() and (tmp_path / "inform_export").is_dir()


def test_without_a_folder_the_tree_goes_to_a_new_temporary_folder(quick, tmp_path, monkeypatch):
    old_default = (FIXTURES / "synth").exists()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert quick.cli([]) == 0
    [made] = [p for p in tmp_path.iterdir() if p.name.startswith("yfs-synth-")]
    assert (made / "synth" / quick.MARKER).is_file() and (made / "synth" / "ProjectA").is_dir()
    assert (FIXTURES / "synth").exists() == old_default


def test_the_old_default_folder_is_git_ignored():
    assert "tests/fixtures/synth/" in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
