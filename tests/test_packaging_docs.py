"""The README's workflow, the requirements files and the build specs agree with the code and the release workflow
(PL-8 and its byte-identity wording, PL-10, PL-12). The specs run with stand-ins for PyInstaller's builders, so
PyInstaller is not needed."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import yfs
from tests.conftest import REPO   # not "conftest": tests/gui/ has its own
from yfs.core.osutil import LOG_KEEP

README = (REPO / "README.md").read_text(encoding="utf-8")


def requirements(name: str) -> dict:
    """{project: specifier} of one requirements file; comments and -r lines are skipped."""
    found = {}
    for line in (REPO / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*([^#]*)", line)
        if m:
            found[m.group(1).lower()] = m.group(2).replace(" ", "")
    return found


def run_spec(folder: str) -> dict:
    """{builder name: its keyword arguments} of <folder>/YourFavouriteStitcher.spec."""
    made = {}

    def builder(name):
        def build(*args, **kw):
            made[name] = kw
            return SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[])
        return build

    path = REPO / folder / "YourFavouriteStitcher.spec"
    namespace = {"SPECPATH": str(path.parent), **{n: builder(n) for n in ("Analysis", "PYZ", "EXE", "BUNDLE")}}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return made


def test_the_mac_app_reports_the_app_version():
    plist = run_spec("Mac")["BUNDLE"]["info_plist"]
    assert plist["CFBundleShortVersionString"] == plist["CFBundleVersion"] == yfs.__version__


def test_the_build_pins_pyinstaller_6_and_the_gated_tifffile_and_imagecodecs():
    specifiers = pytest.importorskip("packaging.specifiers")
    build = requirements("requirements-build.txt")
    pyinstaller = specifiers.SpecifierSet(build["pyinstaller"])
    assert "6.22.3" in pyinstaller and "7.0" not in pyinstaller      # 7.0 rejects the Mac spec's onefile .app
    assert re.fullmatch(r"==[\d.]+", build["tifffile"]) and re.fullmatch(r"==[\d.]+", build["imagecodecs"])


def test_the_readme_names_the_assets_the_release_workflow_uploads():
    workflow = (REPO / ".github" / "workflows" / "build-release.yml").read_text(encoding="utf-8")
    files = re.search(r"files: \|\n((?:[ \t]+\S+\n)+)", workflow).group(1).split()
    assets = [f.rsplit("/", 1)[1] for f in files]
    assert len(assets) == 3 and all(f"`{a}`" in README for a in assets), assets


def test_the_readme_says_the_mac_download_is_apple_silicon_only_and_how_to_open_it():
    assert run_spec("Mac")["EXE"]["target_arch"] is None              # host arch: the release runner is arm64
    assert "Apple Silicon Macs only" in README and "Intel Macs are both supported" not in README
    assert "Privacy & Security** and click **Open Anyway**" in README  # macOS 15 dropped right-click > Open


def test_the_readme_says_how_to_see_the_diag_output_on_windows():
    assert run_spec("Windows")["EXE"]["console"] is False             # no console: print() writes nothing
    assert "--diag > diag.txt" in README


def test_the_readme_log_count_and_the_dev_requirements_match_the_code():
    assert re.search(rf"the last {LOG_KEEP}\s+are kept", README)      # osutil prunes older session logs
    assert "from PIL import" in (REPO / "tests" / "gui" / "shotlib.py").read_text(encoding="utf-8")
    assert "pillow" in requirements("requirements-dev.txt")           # so tools/shots.py runs after the dev install


def test_the_readme_promises_v1_bytes_only_on_one_thread():
    # the engine pastes tiles from os.cpu_count() threads, so where tiles overlap the pixels follow thread timing and
    # two runs of v1 or v2 can differ there; the identity tests compare on one thread (FINAL_SPEC 5.7, item 6)
    engine = (REPO / "stitcher_RAM.py").read_text(encoding="utf-8")
    assert re.search(r"^NUM_THREADS = os\.cpu_count\(\)", engine, re.M)
    paragraphs = [" ".join(p.split()) for p in README.split("\n\n")]  # unwrapped
    claims = [p for p in paragraphs if re.search(r"byte[ -]for[ -]byte|same bytes", p)]
    assert claims and all("one thread" in p for p in claims), claims
    assert any("overlap" in p and "checksum" in p for p in claims), claims


def test_the_readme_documents_the_plan_menu_shortcut_after_rename():
    from yfs.ui.dialogs import _KEYS_MAIN

    keys = next(keys for keys, action in _KEYS_MAIN if "Finalize" in action)
    row = f"| {keys} | The slide's menu (Try again, Finalize, Search deeper…) |"
    assert row in README
    assert README.splitlines()[README.splitlines().index(row) - 1].startswith("| F2 |")


def test_the_readme_distinguishes_typed_paths_from_relative_command_line_paths(tmp_path, monkeypatch):
    from yfs.ui import app

    monkeypatch.chdir(tmp_path)
    args = []
    monkeypatch.setattr(app, "_run", lambda engine, parsed: args.append(parsed) or 0)
    assert app.main(object, ["--input", "ProjectC", "--output", "stitched"]) == 0
    assert Path(args[0].input) == tmp_path / "ProjectC"
    assert Path(args[0].output) == tmp_path / "stitched"
    assert not app._full_path("ProjectC", "posix") and not app._full_path("ProjectC", "win32")
    for flag in ("--input", "--output"):
        row = next(line for line in README.splitlines() if line.startswith(f"| `{flag} FOLDER` |"))
        assert "relative paths" in row and "folder the app starts in" in row
    assert "full path" in README


@pytest.mark.parametrize("verdict, checked, label", [
    ("valid", False, "Skipped: already stitched"),
    ("incomplete", True, "The existing file looks incomplete"),
    ("unchecked", False, "Couldn't check the existing file"),
    ("unreadable", False, "Couldn't check the existing file"),
])
def test_the_readme_distinguishes_existing_file_verdicts(verdict, checked, label):
    from yfs.core.model import Candidate, OutputInfo
    from yfs.core.plan import PlanRow, default_state, row_text

    cand = Candidate("slide", "/data/Slide1_Scan1", ".", "Slide1_Scan1", "Slide1", "component_data", "component",
                     (), (), False, False, 0)
    row = PlanRow(cand.key, cand, output_stem="Slide1_Scan1", existing=OutputInfo("Slide1_Scan1.ome.tif", 1024, 0),
                  existing_verdict=verdict)
    row.section, row.status, row.checked, row.checkable = default_state(row)
    assert row.checked is checked and row_text(row).startswith(label)
    documented = next((line for line in README.splitlines() if line.startswith(f"| *{label}")), "")
    assert ("Ticked by default" if checked else "Unticked by default") in documented


def test_the_readme_explains_browser_badges_for_unknown_contents():
    from yfs.core.model import CLOUD, NETWORK, FsError
    from yfs.ui.browser import describe_folder

    for lane, auto, error in ((CLOUD, "cloud", None), (NETWORK, "disconnected", None),
                              (NETWORK, None, FsError("timeout", "labshare didn't answer."))):
        badge = describe_folder(None, None, "input", frozenset(), lane, error, auto)
        assert badge.inside in README
    assert "Only folders with tiles" in README and "not checked" in README


@pytest.mark.parametrize("platform, config, logs", [
    ("win32", r"%LOCALAPPDATA%\YourFavouriteStitcher", r"%LOCALAPPDATA%\YourFavouriteStitcher\logs"),
    ("darwin", "~/Library/Application Support/YourFavouriteStitcher", "~/Library/Logs/YourFavouriteStitcher"),
    ("linux", "~/.config/yourfavouritestitcher", "~/.local/state/yourfavouritestitcher/logs"),
])
def test_the_readme_distinguishes_settings_and_log_folders(platform, config, logs, monkeypatch):
    import posixpath

    from yfs.core import settings

    paths = SimpleNamespace(join=posixpath.join, isabs=posixpath.isabs, expanduser=lambda _: "/home/example")
    monkeypatch.setattr(settings, "os", SimpleNamespace(path=paths, environ={"LOCALAPPDATA": "/appdata"}))
    monkeypatch.setattr(settings, "sys", SimpleNamespace(platform=platform))
    actual = settings.config_dir(), settings.log_dir()
    for documented, path in zip((config, logs), actual):
        expected = documented.replace("%LOCALAPPDATA%", "/appdata").replace("~", "/home/example").replace("\\", "/")
        assert expected == path.replace("\\", "/")
        assert f"`{documented}`" in README
