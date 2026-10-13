"""Nothing under yfs/core may import tkinter (FINAL_SPEC §5.1, acceptance A6). Every module is covered,
including modules added later: the list is read from the package folder, not hard-coded."""
from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import yfs.core

CORE = Path(yfs.core.__file__).resolve().parent
REPO = CORE.parents[1]
MODULES = sorted(m.name for m in pkgutil.iter_modules([str(CORE)]))
TK = ("tkinter", "_tkinter", "Tkinter")


def test_core_modules_are_found():
    assert {"model", "events", "names", "places", "fsservice", "discovery", "plan"} <= set(MODULES)


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_no_tkinter_in_the_source(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):    # __import__("tkinter") / import_module("tkinter")
            mods = [node.args[0].value] if getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "__import__", "import_module") else []
        else:
            continue
        assert not [m for m in mods if m.split(".")[0] in TK], f"{path.name}:{node.lineno} imports tkinter"


def test_every_core_module_imports_without_tkinter():
    code = "\n".join([
        "import importlib, sys, traceback",
        "for name in ('tkinter', '_tkinter', 'tkinter.ttk', 'tkinter.font'):",
        "    sys.modules[name] = None          # any import of tkinter now raises ImportError",
        f"for mod in {MODULES!r}:",
        "    try:",
        "        importlib.import_module('yfs.core.' + mod)",
        "        print('OK', mod)",
        "    except BaseException:",
        "        print('FAIL', mod, traceback.format_exc().strip().splitlines()[-1])",
    ])
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=120)
    failed = [ln for ln in out.stdout.splitlines() if ln.startswith("FAIL")]
    assert out.returncode == 0 and not failed, failed or out.stderr
    assert len([ln for ln in out.stdout.splitlines() if ln.startswith("OK")]) == len(MODULES)
