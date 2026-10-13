"""Folder browser: the in-app, keyboard-operable folder chooser (FINAL_SPEC §2.2, §5.5.14).

Every filesystem access runs on FsService lanes (or, for the places list, a one-shot worker thread); the Tk thread
only renders results, so a slow or hung mount can never freeze the dialog. describe_folder, describe_file,
describe_tiles, describe_target and browser_initial are pure and unit-tested in tests/test_describe.py.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import gc
import itertools
import logging
import os
import re
import stat
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from functools import partial
from tkinter import filedialog, font as tkfont, ttk
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from yfs.core import osutil
from yfs.core.events import FsDone, FsSlow, LaneHealthChanged, MountsChanged, PickerResult
from yfs.core.model import (CLOUD, H_DISCONNECTED, H_OK, H_SLOW, H_UNRESPONSIVE, LOCAL, NETWORK, DirSummary, FsError,
                            OutputIndex, OutputInfo, Rollup)
from yfs.core.names import (COMPONENT_SUFFIX, EXCLUDED_DIR_NAMES, ROLLUP_PER_CHILD, ROLLUP_PER_VIEW,
                            classify_file, human_bytes, natural_key, norm_key, same_path, stitchable_groups,
                            summarize_listing)
from yfs.core.places import (Place, breadcrumb_parts, containing_place, list_places, normalize_user_path,
                              tcc_protected, windows_volume_label)
from yfs.core.preflight import write_probe
from yfs.ui.dialogs import NewFolderDialog
from yfs.ui.theme import TOKENS as T, tag_tree, theme_of
from yfs.ui.widgets import SlideFacts, Tooltip, safe_grab

if TYPE_CHECKING:
    from yfs.core.settings import Settings
    from yfs.ui.app import AppContext

ELL = "…"
WINDOWS, MACOS = sys.platform == "win32", sys.platform == "darwin"
_NOTABLE = frozenset({"output", "wsi", "raw", "inform_marker", "lnk", "tmp_raw"})
_NO_ANSWER = frozenset({"timeout", "unresponsive", "unavailable", "disconnected"})
_STALL = frozenset({"timeout", "unresponsive"})              # asked again once the lane answers again (UD1)
_UNCHECKED = frozenset({"cloud", "autofs", "disconnected"})  # auto_enter() reasons that leave a folder unknown
_TITLE = {"input": "Choose the folder with your tiles", "output": "Choose where to save the stitched images",
          "temp": "Choose a folder for temporary files"}
_PRIMARY = {"input": "Use ", "output": "Save here: ", "temp": "Use for temporary files: "}
_WRITE = {"output": "save", "temp": "write"}                 # what the write check checks, in its texts
_SECTION = {"recent": "RECENT", "computer": "THIS MAC" if MACOS else "THIS COMPUTER", "network": "NETWORK",
            "cloud": "CLOUD", "bookmarks": "BOOKMARKS (FILES)"}
_DOT = {H_OK: "ok", H_SLOW: "slow", H_UNRESPONSIVE: "bad", H_DISCONNECTED: "off"}
_HEALTH = {H_SLOW: " (slow)", H_UNRESPONSIVE: " (not answering)", H_DISCONNECTED: " (disconnected)"}
_GROUP = {"dir": 0, "tiles": 1, "file": 2, "other": 3}
_SORT_DEFAULT = {"name": "asc", "tiles": "desc", "inside": "asc", "modified": "desc"}
# Control and Alt/Command (not NumLock) in Event.state; 0x20000 is Tk's own Alt bit (Windows, generated events)
_MODS = 0x20004 if WINDOWS else 0x1C if MACOS else 0x2000C
_BARE_SERVER = re.compile(r"^(?:\\\\|//|smb://)[^\\/]+[\\/]*$", re.IGNORECASE)
_TOKENS = itertools.count(1)
_LinkState = Tuple[str, Tuple[str, ...], int]             # resolved parent, remaining components, link hops
log = logging.getLogger(__name__)


# ============================================================================================ pure helpers
@dataclass(frozen=True)
class Badge:
    icon: str; tiles_text: str; inside: str; bold: bool; dim: bool; rank: int


def _n(k: int, one: str, many: Optional[str] = None) -> str:
    return f"{k:,} {one if k == 1 else (many or one + 's')}"


def _base(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or path


def _stitchable(summary: Optional[DirSummary]) -> Dict[str, int]:
    """prefix -> tile count of the group the engine would stitch: component_data, else the unsuffixed one."""
    return {p: len(g.names) for p, g in _slide_groups(summary).items()}


def _slide_groups(summary: Optional[DirSummary]) -> Dict[str, Any]:
    """prefix -> the TileGroup the engine would stitch (component_data, else unsuffixed)."""
    return stitchable_groups(summary)


def describe_folder(summary: Optional[DirSummary], rollup: Optional[Rollup], mode: str, out_names: FrozenSet[str],
                    lane_class: str, error: Optional[FsError], auto: Optional[str] = None, failed: int = 0) -> Badge:
    """§2.2.6 badge of a folder row (pure; the first matching rule wins). `auto` is MountTable.auto_enter() of the
    folder; texts use Unicode, so display them through Theme.text(). rank orders the *What's inside* sort.
    `out_names` remains for callers but cannot establish completion; `failed` counts unreadable descendants."""
    if error is not None and error.code != "cancelled":
        text = ("No access" if error.code == "no_access" else "Didn't answer — F5 to try again"
                if error.code in _NO_ANSWER else error.message.rstrip("."))
        return Badge("folder_error", "", text, False, True, 8)
    if auto == "cloud" or lane_class == CLOUD:
        return Badge("folder", "", "Cloud folder — open to look inside", False, False, 7)
    if auto in ("disconnected", "autofs"):
        return Badge("folder", "", "Network drive — open to connect", False, False, 7)
    if auto == "pseudo":
        return Badge("folder", "", "System folder", False, True, 7)
    if summary is None:
        return Badge("folder", "", ELL, False, True, 7)
    s, slides = summary, _stitchable(summary)
    tiles = f"{sum(slides.values()):,}"
    if slides and mode == "output":
        return Badge("folder_tiles", tiles, "Slide tiles — don't save here", False, True, 0)
    bold = mode != "temp"                                # temporary files: no folder stands out from the others
    if len(slides) == 1:
        # Output filenames alone establish neither slide identity nor verified completion (D1, D2).
        return Badge("folder_tiles", tiles, "Slide tiles", bold, False, 0)
    if slides:
        return Badge("folder_tiles", tiles, f"{len(slides)} slides (inForm batch export)", bold, False, 1)
    if rollup is not None and rollup.slide_folders:
        more = "+" if rollup.capped or failed else ""
        many = "s" if rollup.slide_folders > 1 or more else ""
        tail = f" · {_n(failed, 'folder')} couldn't be read" if failed else ""
        return Badge("folder_slides", "", f"{rollup.slide_folders:,}{more} slide folder{many} inside{tail}", bold, False, 2)
    if failed:
        return Badge("folder_error", "", f"Contents unknown · {_n(failed, 'folder')} couldn't be read", False, False, 7)
    if s.groups:                                         # the kind first: a narrow column keeps it (BR-26)
        sufs = ", ".join(sorted({g.suffix for g in s.groups}))
        return Badge("folder", f"{sum(len(g.names) for g in s.groups):,}",
                     f"{sufs} tiles, not unmixed components", False, False, 3)
    for cond, text, rank in ((s.raw_scan, "Raw scan — unmix in inForm first", 4),
                             (s.outputs, _n(len(s.outputs), "stitched image"), 5),
                             (s.wsi, _n(len(s.wsi), "whole-slide image") + " (another program)", 5),
                             (s.inform_export, "inForm export", 6),
                             (s.other_tiffs, _n(s.other_tiffs, "other TIFF image"), 6)):
        if cond:
            return Badge("folder", "", text, False, False, rank)
    if s.truncated:
        return Badge("folder", "", f"{s.n_entries:,}+ items, no tiles", False, True, 7)
    if not s.n_entries:
        return Badge("folder", "", "Empty", False, True, 7)
    files = s.other_files + len(s.notable_files)                # what is left here: .lnk and .tmp.raw are notable
    rest = (_n(len(s.subdirs), "folder") if s.subdirs else _n(files, "file") if files else
            _n(s.links, "link") if s.links else "only hidden items")
    return Badge("folder", "", "No tiles · " + rest, False, True, 7)


def describe_file(name: str, size: Optional[int]) -> Tuple[str, str, bool]:
    """(icon, *What's inside* text, dim) of a file row (§2.2.6); files that are not notable show their size."""
    kind = classify_file(name)
    if kind == "output":
        return "file_ome", "Stitched image" + (f" · {human_bytes(size)}" if size is not None else ""), False
    text, dim = {"wsi": ("Whole-slide image (another program)", False), "raw": ("Raw scan", False),
                 "inform_marker": ("inForm batch file", False),
                 "lnk": ("Windows shortcut (can't be followed)", True),
                 "tmp_raw": ("Temporary file from an unfinished run", True)}.get(kind, (human_bytes(size), True))
    return "file", text, dim


def describe_tiles(summary: DirSummary) -> Tuple[str, str]:
    """(name, pattern) of the one row that stands for every tile of a folder: '370 tile files',
    '…_[x,y]_component_data.tif'. It counts the tiles slides are stitched from, like the strip and the preview;
    other groups (inForm's binary_seg_maps...) follow the pattern: '· also 9 binary_seg_maps'."""
    main = list(_slide_groups(summary).values()) or list(summary.groups)
    keys = {(g.prefix, g.suffix) for g in main}
    rest = [g for g in summary.groups if (g.prefix, g.suffix) not in keys]
    sufs = sorted({g.suffix for g in main}, key=lambda s: (s != COMPONENT_SUFFIX, s))
    pattern = ", ".join(f"{ELL}_[x,y]{'_' + s if s else ''}.tif" for s in sufs[:2]) + (", " + ELL if sufs[2:] else "")
    if rest:
        kinds = {g.suffix for g in rest}
        what = kinds.pop() if len(kinds) == 1 and "" not in kinds else "other tiles"
        pattern += f" · also {sum(len(g.names) for g in rest):,} {what}"
    return _n(sum(len(g.names) for g in main), "tile file"), pattern


def describe_target(mode: str, name: str, summary: Optional[DirSummary], rollup: Optional[Rollup], *,
                    error: Optional[FsError] = None, auto: Optional[str] = None, save_folder: bool = False,
                    lane_class: str = LOCAL, free: Optional[int] = None, planned: FrozenSet[str] = frozenset(),
                    place: str = "", unchecked: int = 0, linked: str = "",
                    free_error: Optional[FsError] = None, failed: int = 0) -> Tuple[str, str]:
    """(style, text) of the target strip (pure; §2.2.3, output mode §2.2.1). style: Ok | Info | Warn | Error.
    rollup None means its probes are still running (or there is nothing below the folder). place: what holds the
    target, named when it doesn't answer; unchecked: subfolders whose contents can't be known yet (cloud folders,
    network drives, folders that didn't answer); linked: a subfolder that is a link to slide folders (the search
    doesn't follow links, so the rollup leaves them out). `failed` counts unreadable descendants; `free_error`
    describes only a capacity check, never write permission."""
    slides = _stitchable(summary)
    tiles = sum(slides.values())
    if error is not None and error.code in _STALL:     # the place once, then what the user can do (UD1)
        said = error.message if error.code == "timeout" else f"{place or name} isn't answering."
        if mode != "input":
            return "Warn", f"{said} Press F5 to try again."
        return "Warn", f"{said} Press F5 to try again, or use {'it' if name == place else name} anyway."
    if mode != "input" and error is None and free_error is not None:
        return "Warn", f"Couldn't check free space: {free_error.message} Press F5 to try again."
    if mode == "temp":                                   # only what matters for scratch files (BR-20)
        if error is not None:
            return "Error", f"Can't write here: {error.message}"
        free_text = f"Free space: {human_bytes(free)}. " if free is not None else ""
        if lane_class == CLOUD:
            return "Warn", f"{free_text}Cloud folder: temporary files would be uploaded too. A local disk is better."
        if lane_class == NETWORK:
            return "Info", f"{free_text}Network folder: temporary files are faster on a local disk."
        return ("Ok", free_text.rstrip()) if free is not None else ("Info", f"Checking {name}{ELL}")
    if mode == "output":
        if error is not None:
            return "Error", f"Can't save here: {error.message}"
        outs = summary.outputs if summary is not None else ()
        facts = [f"Free space: {human_bytes(free)}"] if free is not None else []
        facts += ["contains " + _n(len(outs), "stitched image")] if outs else []
        have = len(planned & {norm_key(o) for o in outs})
        facts += [f"{have} of your {_n(len(planned), 'slide')} already have stitched files here"] if have else []
        lines, style = ([" · ".join(facts) + "."] if facts else []), "Ok"
        if slides:
            lines.append("This folder holds tiles; stitched images would be mixed in.")
            style = "Warn"
        if lane_class == CLOUD:
            lines.append("Cloud folder: stitched files will upload in the background.")
            style = "Warn"
        elif lane_class == NETWORK:                  # advice, never in a green banner
            lines.append("Network folder: saving over the network is slower; a local disk is faster.")
            style = "Info" if style == "Ok" else style
        if summary is None and free is None:
            return "Info", f"Checking {name}{ELL}"
        return style, " ".join(lines)
    if save_folder:
        return "Warn", "This is your save folder. Stitched images are never used as tiles."
    if error is not None:
        return "Error", f"{name}: {error.message}"
    if auto == "cloud" or (summary is None and lane_class == CLOUD):
        return "Info", "Cloud folder: files download as they are read. Searching one level only."
    if auto in ("disconnected", "autofs"):
        return "Info", f"{name} is a network drive. Open it to connect."
    if summary is None:
        return "Info", f"Checking what's inside {name}{ELL}"
    full = (f" {_n(failed, 'folder')} couldn't be read. Press F5 to try again." if failed else
            " You'll see the full list before anything is stitched.")
    style = "Warn" if failed else "Ok"
    if len(slides) == 1:
        suffix = next(g.suffix for g in summary.groups if g.prefix in slides and g.suffix in (COMPONENT_SUFFIX, ""))
        return style, f"{name}: 1 slide, {tiles:,} tiles ({ELL}{'_' + suffix if suffix else '_[x,y]'}.tif).{full}"
    if slides:
        return style, f"{name}: {len(slides)} slides in one folder (inForm batch export), {tiles:,} tiles." + \
            (full if failed else "")
    if rollup is not None and rollup.slide_folders:
        more = "+" if rollup.capped or failed else ""
        folders = f"{rollup.slide_folders:,}{more} slide folder{'s' if rollup.slide_folders > 1 or more else ''}"
        but = f", not counting {linked}: the search doesn't follow links" if linked else ""
        return style, f"{name} contains {folders} ({rollup.tiles:,}{more} tiles){but}.{full}"
    if failed:
        return "Warn", f"Couldn't check every folder in {name}.{full}"
    if summary.raw_scan:
        return "Warn", f"{name} holds raw scans (.qptiff). Unmix and export component data in inForm first."
    if summary.groups:
        sufs = ", ".join(sorted({g.suffix for g in summary.groups}))
        return "Warn", f"{name} holds {sufs} tiles, not unmixed components. You can still tick them in the list."
    if rollup is None and summary.subdirs:
        return "Info", f"Checking what's inside {name}{ELL}"
    if lane_class == CLOUD:                          # its subfolders are never probed (§2.2.4): nothing is known
        return ("Info", "Cloud folder: subfolders aren't checked automatically. Open one to look inside, or use this "
                        "folder (the search looks one level deep here).") if summary.subdirs else \
            ("Warn", f"No tiles found in {name}.")
    if linked:
        return "Info", f"The search doesn't follow links: open {linked} to use the slide folders it leads to."
    if unchecked:                                    # (the search skips cloud folders and network drives too)
        one = unchecked == 1
        return "Info", (f"No tiles found so far: {_n(unchecked, 'folder')} here {'wasn' if one else 'weren'}'t "
                        f"checked. Open {'it' if one else 'one'} to look inside.")
    if not summary.n_entries:
        return "Warn", f"{name} is empty."
    if rollup is not None and rollup.capped:         # (40 subfolders per folder, 200 per view are looked at)
        return "Warn", f"No tiles found in {name} so far (not every subfolder was checked). The search will look " \
                       "3 levels deep."
    if not summary.subdirs:                          # (nothing below to look into)
        return "Warn", f"No tiles found in {name}."
    return "Warn", f"No tiles found in {name} within 2 levels. The search will look 3 levels deep."


def browser_initial(mode: str, field_text: str, input_path: Optional[str], settings: "Settings",
                    places: Sequence[Place] = (), input_is_slide: bool = False) -> Tuple[str, Optional[str]]:
    """(folder to open, name to select or None). Pure string work, the rules of §2.2.2 / R25. input_is_slide: the
    input folder is itself one slide, so the save folder is looked for next to it, with nothing selected: never in
    its folder of tiles (BR-8). The temporary-files chooser has no recent folders and no input to go by."""
    roots = [p.path for p in places if p.section in ("computer", "network", "cloud")]

    def split(path: str) -> Tuple[str, Optional[str]]:
        path = os.path.normpath(path)
        if os.path.dirname(path) == path or any(same_path(path, r) for r in roots):
            return path, None                           # a drive, share or mount point: open it, select nothing
        return os.path.dirname(path), _base(path)

    text = (field_text or "").strip()
    if text and os.path.isabs(text):
        return split(text)
    last = settings.get(f"paths.browser_last.{mode}") or ""
    if last:
        return os.path.normpath(last), None
    recent = settings.get(f"recent.{mode}s") or []      # (there is no recent.temps)
    if recent and isinstance(recent[0], dict) and recent[0].get("path"):
        return split(recent[0]["path"])
    if mode == "output" and input_path:
        return (split(input_path)[0], None) if input_is_slide else split(input_path)
    return os.path.expanduser("~"), None


def _middle(text: str, limit: int, ell: str) -> str:
    """Shorten in the middle (§2.2.3: names longer than 40 characters)."""
    if len(text) <= limit:
        return text
    head = (limit - len(ell)) // 2
    return text[:head] + ell + text[len(text) - (limit - len(ell) - head):]


class _Ruler:
    """Text widths in one font, summed from the widths of its characters, each measured once per font: a Tk
    measure costs about a millisecond, too slow for the thousands of rows of a listing (BR-26)."""
    _known: Dict[Tuple[Any, ...], Dict[str, int]] = {}

    def __init__(self, font: tkfont.Font) -> None:
        self.font = font
        self.chars = _Ruler._known.setdefault((font.cget("family"), font.cget("size"), font.cget("weight"),
                                               font.metrics("linespace"), font.measure("MW")), {})
        self.texts: Dict[str, int] = {}

    def width(self, text: str) -> int:
        total = self.texts.get(text)
        if total is None:
            chars, total = self.chars, 0
            for ch in text:
                w = chars.get(ch)
                if w is None:
                    w = chars[ch] = self.font.measure(ch)
                total += w
            if len(self.texts) > 50000:
                self.texts.clear()
            self.texts[text] = total
        return total

    def fit(self, text: str, width: int, ell: str, middle: bool = False) -> str:
        """`text`, or `text` shortened with `ell` to at most `width` px: in the middle (a name keeps its end, as
        on the primary button), else at the end, before a whole word when one fits (a badge)."""
        if self.width(text) <= width:
            return text
        if width < self.width(ell):
            return ""
        room, n, used = width - self.width(ell), len(text), 0
        head = tail = 0
        while head + tail < n:
            front = not middle or head <= tail
            w = self.width(text[head] if front else text[n - 1 - tail])
            if used + w > room:
                break
            used += w
            head, tail = (head + 1, tail) if front else (head, tail + 1)
        if middle:
            return text[:head] + ell + text[n - tail:]
        word = text.rfind(" ", 0, head + 1) if text[head:head + 1] not in (" ", "") else head
        kept = text[:word if word > 0 else head]
        if kept.count("(") > kept.count(")"):             # never a bracket left open: '(another…'
            kept = kept[:kept.rfind("(")]
        return kept.rstrip(" ,·—(-") + ell


def _unreadable(path: str, error: FsError) -> DirSummary:
    """A summary that only says why the folder can't be shown: the preview's message for errors and hangs."""
    return DirSummary(path, 0, False, (), (), (), (), False, False, 0, (), (), 0, error=error)


def _link_start(path: str, hops: int = 0, known: str = "") -> _LinkState:
    """Split a full path for component-by-component link resolution; string work only (also drive/UNC roots)."""
    parts = []
    if known:
        try:
            known = os.path.commonpath((path, known))      # ancestors of a resolved parent are resolved too
        except ValueError:                               # different Windows drives
            known = ""
    while os.path.dirname(path) != path and not (known and same_path(path, known)):
        path, name = os.path.split(path)
        parts.append(name)
    return path, tuple(reversed(parts)), hops


# ============================================================================================ lane calls
def _locate(path: str) -> Tuple[str, str, Optional[str]]:
    """Lane call: ('dir' | 'file' | 'missing', path, closest existing ancestor of a missing path)."""
    try:
        return ("dir" if stat.S_ISDIR(os.stat(path).st_mode) else "file"), path, None
    except (FileNotFoundError, NotADirectoryError):
        p = path
        while os.path.dirname(p) != p:
            p = os.path.dirname(p)
            if os.path.isdir(p):
                return "missing", path, p
        return "missing", path, None


@dataclass(frozen=True)
class _PlacesLoaded:
    """Private UiBus event: list_places() and system_picker_available() ran in a worker thread."""
    token: int; places: Tuple[Place, ...]; picker: bool


class _RowTip(Tooltip):
    """A Tooltip under the row of a tree that the pointer is on (a Tooltip shows under its whole widget): the
    whole name and badge of a shortened row, a place's full path."""

    def __init__(self, tree: ttk.Treeview) -> None:
        super().__init__(tree, lambda: self.full)
        self.tree, self.row, self.full = tree, "", ""
        tree.bind("<Leave>", lambda _e: self.point("", ""), add="+")

    def point(self, row: str, text: str) -> None:
        """The pointer is on `row` ('' for none): `text` shows there after the delay; nothing when it is empty."""
        if row != self.row:
            self.row, self.full = row, text
            self.hide()
            if text:
                self._schedule()

    def _show(self) -> None:
        super()._show()
        box = self.tree.bbox(self.row) if self._tip is not None and self.row else ""
        if box:
            tip, px = self._tip, theme_of(self.tree).px
            x = min(self.tree.winfo_rootx() + box[0] + px(24), tip.winfo_screenwidth() - tip.winfo_reqwidth() - 2)
            tip.wm_geometry(f"+{max(0, x)}+{self.tree.winfo_rooty() + box[1] + box[3] + px(2)}")


class _Row:
    """One listing row. kind: dir | tiles (one row for every tile) | file (notable) | other (shown on request).
    A link row becomes a dir once it is followed to a folder (target)."""
    __slots__ = ("key", "name", "kind", "hidden", "link", "size", "mtime", "summary", "error", "auto", "probing",
                 "roll", "target", "stalled", "date_error", "resolving", "roll_errors")

    def __init__(self, key: str, name: str, kind: str, hidden: bool, link: bool, size: Optional[int],
                 mtime: Optional[float]) -> None:
        self.key, self.name, self.kind, self.hidden, self.link = key, name, kind, hidden, link
        self.size, self.mtime = size, mtime
        self.summary: Optional[DirSummary] = None
        self.error: Optional[FsError] = None
        self.auto: Optional[str] = None
        self.probing = 0                                   # priority of the probe (or link lookup) in flight; 0 = none
        # [probed, slide folders, tiles, capped, pending, sample, paths that didn't answer, shares and cloud
        # folders inside that are never looked into]
        self.roll: Optional[List[Any]] = None
        self.roll_errors: Dict[str, FsError] = {}          # failed descendant checks; never evidence of no tiles
        self.target: Optional[str] = None                  # a link row: the folder it leads to
        self.stalled: Optional[str] = None                 # the path that didn't answer (asked again when it does)
        self.date_error: Optional[FsError] = None          # metadata failure never replaces valid tile facts
        self.resolving: Optional[_LinkState] = None


# ============================================================================================ the dialog
class FolderBrowser(tk.Toplevel):
    """§2.2. Consumes FsDone/FsSlow (owner 'browser'), LaneHealthChanged, PickerResult. Never calls os.* itself."""

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", *, mode: str, initial: Tuple[str, Optional[str]],
             on_done: Callable[[Optional[str]], None], title: Optional[str] = None) -> "FolderBrowser":
        """initial = browser_initial(...). title overrides the mode's title (Settings uses it for the temp folder)."""
        return cls(master, ctx, mode, initial, on_done, title)

    def __init__(self, master: tk.Misc, ctx: "AppContext", mode: str, initial: Tuple[str, Optional[str]],
                 on_done: Callable[[Optional[str]], None], title: Optional[str] = None) -> None:
        super().__init__(master)
        self.withdraw()
        self.ctx, self.mode, self._on_done = ctx, mode, on_done
        self.px, self.G, self.T = ctx.theme.px, ctx.theme.G, ctx.theme.text
        self._title = title or _TITLE[mode]
        self.title(self.T(self._title))
        self.transient(master.winfo_toplevel())
        if ctx.theme.ws == "x11":
            try:
                self.wm_attributes("-type", "dialog")
            except tk.TclError:
                pass
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.configure(background=T["bg"])
        s = ctx.settings
        self._path, self._ticket, self._t0 = "", 0, 0.0
        self._located_path = ""                           # a just-resolved target may be used before its list arrives
        # the loading request (a listing, or the locate of a typed or first folder): its FsSlow drive §2.2.8
        self._wait_rid, self._wait_path = 0, ""
        self._wait_again: Callable[[], None] = self._retry
        self._wait_back: Callable[[], None] = self._go_back
        self._no_answer: Optional[FsError] = None          # this folder didn't answer (the strip and preview say so)
        self._stall: Optional[Tuple[str, Callable[[], None]]] = None    # (lane, ask again once it answers)
        self._calls: Dict[int, Tuple[Optional[int], Callable[[FsDone], None]]] = {}
        self._back: List[str] = []
        self._fwd: List[str] = []
        self._rows: Dict[str, _Row] = {}
        self._names: Dict[str, str] = {}
        self._created: set = set()
        self._statted: set = set()
        self._seq = itertools.count()
        self._listing: Any = None
        self._summary: Optional[DirSummary] = None
        self._list_error: Optional[FsError] = None
        self._select: Optional[str] = None
        self._select_file = self._loading = self._keep_waiting = self._no_cache = self._editing = False
        self._use_cache = True
        self._intent_revision = 0
        self._writing: Optional[Tuple[int, str, int]] = None   # (token, folder, intent revision) of the write check
        self._dating: set = set()                           # stat requests that date every row (Modified sort)
        self._stat_batches: Dict[int, List[str]] = {}
        self._unresponsive_since: Dict[str, float] = {}
        self._recovery_seen: Dict[tuple, float] = {}
        self._layout_keys: set = set()
        self._pinned: Optional[str] = None                 # a row asked for, shown although a filter hides it (BR-2)
        self._keep_view: Optional[Tuple[str, float, Optional[str]]] = None   # F5: (folder, scroll, focus row) (BR-17)
        # BR-26: rows are drawn shortened to their columns. By row key: the full texts (name, tiles, inside,
        # modified, bold, where a link leads), the texts drawn (name, values), and a shortened row's tooltip
        self._full: Dict[str, Tuple[str, str, str, str, bool, str]] = {}
        self._shown: Dict[str, Tuple[str, Tuple[str, str, str]]] = {}
        self._cut: Dict[str, str] = {}
        self._fit_widths: Tuple[int, int] = (0, 0)
        self._refit_keys: List[str] = []                   # rows still to be fitted to the columns' new widths
        fonts = ctx.theme.fonts
        self._rulers = {False: _Ruler(fonts["YfsBody"]), True: _Ruler(fonts["YfsBold"])}
        self._small = _Ruler(fonts["YfsSmall"])
        self._places: Tuple[Place, ...] = ()
        self._place_items: Dict[str, Tuple[Place, str]] = {}
        self._picker_ok, self._kill = False, None
        self._token, self._picker_token, self._locate_token = next(_TOKENS), 0, 0
        self._comp_ticket = ctx.fs.new_ticket()
        self._comp_paths: Dict[str, str] = {}              # resolved autocomplete parents for this edit session
        self._jobs: Dict[str, str] = {}
        self._layout: List[str] = []
        self._layout_pos, self._roll_budget, self._probe_queue = 0, ROLLUP_PER_VIEW, []
        self._ov_kind = ""
        self._strip_msg: Optional[Tuple[str, str, Optional[Tuple[str, Callable[[], None]]]]] = None
        self._target_seen = ""
        self._free: Dict[str, int] = {}
        self._free_pending: set = set()
        self._free_errors: Dict[str, FsError] = {}
        self._capacity_ticket = ctx.fs.new_ticket()
        self._out_sizes: Dict[str, Dict[str, int]] = {}
        self._cycle: Optional[Tuple[str, str, List[str], int]] = None
        self._lb: Optional[ttk.Treeview] = None              # the suggestion list of path editing, in _lb_box
        self._lb_box: Optional[tk.Frame] = None
        self._lb_items: Tuple[str, List[str]] = ("", [])
        self._edit_text = ""                                 # the path text as last typed or set
        self._edit_error = False                             # the strip's message is about the typed path
        self._ta_buf = ""
        sort = s.get("browser.sort") or ["name", "asc"]
        self._sort = (sort[0], sort[1]) if len(sort) == 2 and sort[0] in _SORT_DEFAULT else ("name", "asc")
        self._v_tiles, self._v_hidden, self._v_all = (tk.BooleanVar(self, bool(s.get("browser." + k)))
                                                      for k in ("only_tiles", "show_hidden", "show_all_files"))
        self._build()
        self._bindings()
        bus = ctx.bus
        self._unsub = [bus.subscribe(FsDone, self._on_fs_done), bus.subscribe(FsSlow, self._on_slow),
                       bus.subscribe(LaneHealthChanged, self._on_health), bus.subscribe(PickerResult, self._on_picker),
                       bus.subscribe(_PlacesLoaded, self._on_places), bus.subscribe(MountsChanged, self._on_mounts)]
        self.bind("<Destroy>", lambda e: e.widget is self and self._cleanup(), add="+")
        ctx.fs.reload_mounts()                           # shares mounted since the table was read (UD2)
        self._load_places()
        self._place_window(master)
        self._first(*initial)
        self.deiconify()
        safe_grab(self)
        self.tree.focus_set()
        self._init_sashes()

    # ------------------------------------------------------------------------------------- public API
    def navigate(self, path: str, *, push: bool = True, select: Optional[str] = None) -> None:
        """Show `path`: crumbs at once, then the cached listing, then a fresh one (§2.2.7)."""
        self._new_intent()
        path = os.path.normpath(path) if path else path
        old, fs = self._ticket, self.ctx.fs
        self._ticket, self._locate_token = fs.new_ticket(), 0      # a pending locate's answer is obsolete (CT-4)
        if old:
            fs.cancel_ticket(old)
        self._calls = {k: v for k, v in self._calls.items() if v[0] is None}
        if push and self._path and not same_path(path, self._path):
            self._back.append(self._path)
            del self._fwd[:]
        if not same_path(path, self._path):
            self._filter_var.set("")                      # the filter belongs to the folder it was typed in
        self._set_location(path, select)
        self._focus_list()
        cached = fs.cached_listing(path) if self._use_cache else None
        if cached is not None and not cached.truncated:   # a cut-short listing may be discovery's (FS-1)
            self._render(cached)
            if time.monotonic() - cached.listed_at > self._fresh:
                self._list()
        else:
            self._list()

    def current(self) -> str:
        return self._path

    def target(self) -> str:
        """The selected folder row, else the folder named by a pending selection, else the current folder."""
        r = self._target_row()
        if r is not None:
            return self._row_path(r)
        if self._select and not self._select_file:
            return self._child(self._select)
        return self._path

    def choose(self) -> None:
        """The primary action: input mode returns the target; the save and temporary-files choosers first check
        that it can be written. A click on another folder while a check runs starts that folder's check instead
        (BR-7)."""
        path = self.target()
        row = self._target_row()
        if (self._listing is None and path != self._located_path) or (row is not None and row.link and row.target is None) \
                or (row is None and self._select and not self._select_file):
            state = row.resolving if row is not None else None
            self._locate(path, lambda *_a: self._show_error(FsError("not_found", "This folder no longer exists.")),
                         self._cancel_locate, then=self.choose, resolution=state)
            return
        if self.mode == "input":
            self._finish(path)
            return
        if self._writing is not None and same_path(self._writing[1], path):
            return                                        # already checking this folder
        token = next(_TOKENS)
        self._writing = (token, path, self._intent_revision)
        self._strip_msg = ("Info", f"Checking that you can {_WRITE[self.mode]} in {self._name_of(path)}{ELL}", None)
        self._update_target()
        self._call(path, "write_probe", partial(write_probe, path), lambda e: self._probed(token, path, e), user=True)

    def cancel(self) -> None:
        self._finish(None)

    def back(self) -> None:
        if self._back:
            self._fwd.append(self._path)
            self.navigate(self._back.pop(), push=False)

    def forward(self) -> None:
        if self._fwd:
            self._back.append(self._path)
            self.navigate(self._fwd.pop(), push=False)

    def up(self) -> None:
        parent = os.path.dirname(self._path)
        if parent and parent != self._path:
            self.navigate(parent, select=_base(self._path))

    def refresh(self) -> None:
        """F5: forget the cached listing and look again, keeping the selected row and the scroll position (BR-17).
        An unresponsive lane, of this folder or of a row that didn't answer, lets one request through."""
        fs = self.ctx.fs
        fs.cancel_ticket(self._capacity_ticket)
        self._capacity_ticket = fs.new_ticket()           # a refresh never joins an older usage request
        self._free.clear()
        self._free_pending.clear()
        self._free_errors.clear()
        if self._locate_token:
            self._wait_again()                            # resolving a typed/initial path must never list its alias
            return
        stalled = [p for r in self._rows.values() for p in [r.stalled] + (r.roll[6] if r.roll else []) if p]
        for lane in {fs.lane_key(p) for p in [self._path] + stalled}:
            if fs.health(lane) == H_UNRESPONSIVE:
                fs.retry(lane)
        fs.invalidate(self._path)
        sel, focus = self.tree.selection(), self._rows.get(self.tree.focus())
        r = self._rows.get(sel[0]) if sel else None
        self._keep_view = (self._path, self.tree.yview()[0], focus.name if focus is not None else None)
        self._no_cache = True
        self.navigate(self._path, push=False, select=r.name if r is not None and r.name else self._select)
        self._select_file = r is not None and r.kind != "dir"

    def edit_path(self, initial: Optional[str] = None) -> None:
        """Swap the crumbs for an Entry with the full path, all selected (or start it with `initial`); the strip
        says what the keys do there. Asked again while editing (Ctrl+L), it selects what is typed, as an address
        bar does."""
        self._new_intent()
        if not self._editing:
            self._comp_paths = {}
            self._editing = True
            self._crumb_box.pack_forget()
            self._edit.pack(side="left", fill="both", expand=True)
            self._set_edit(self._path if initial is None else initial)
        elif initial is not None:
            self._set_edit(initial)
        self._lb_hide()
        self._edit.focus_set()
        if initial is None:
            self._edit.select_range(0, "end")
        self._cycle = None
        self._update_target()

    def new_folder(self) -> None:
        """The save and temporary-files choosers: make a folder here (§2.2.9); 'stitched' is offered for saving."""
        if self.mode != "input" and self._listing is not None:
            self._new_intent()
            existing = frozenset(norm_key(e.name) for e in self._listing.entries)
            dlg = NewFolderDialog.ask(self, self.ctx, existing, self._make_folder)
            if self.mode == "temp":
                dlg.var.set("")

    # ------------------------------------------------------------------------------------- building
    def _build(self) -> None:
        px, ic = self.px, self.ctx.icons
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._bar = ttk.Progressbar(self, style="Thin.Horizontal.TProgressbar", mode="indeterminate")
        self._bar.update_idletasks()                      # its requested height is known after an idle pass
        # the loading bar is place()d in the toolbar's bottom padding, which is as tall as the bar
        self._bar_pad = max(px(6), self._bar.winfo_reqheight())
        bar = self._toolbar = ttk.Frame(self, style="Header.TFrame", padding=(px(8), px(6), px(8), self._bar_pad))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(3, weight=1)
        self._nav = {}
        for col, (name, cmd) in enumerate((("back", self.back), ("fwd", self.forward), ("up", self.up))):
            self._nav[name] = ttk.Button(bar, image=ic.get(name), style="Flat.TButton", command=cmd, takefocus=0)
            self._nav[name].grid(row=0, column=col)
        self._pathbar = ttk.Frame(bar, style="Card.TFrame", padding=(px(2), px(1)))
        self._pathbar.grid(row=0, column=3, sticky="ew", padx=px(6))
        self._crumb_box = ttk.Frame(self._pathbar, style="Surface.TFrame")
        self._crumb_box.pack(side="left", fill="y")
        # A Crumb.TButton that posts the menu: clam still draws a 1-px arrow on a Crumb.TMenubutton (arrowsize 0).
        self._more = ttk.Button(self._crumb_box, text=self.G["ell"], style="Crumb.TButton", takefocus=0,
                                command=lambda: self._more_menu.tk_popup(self._more.winfo_rootx(),
                                                                         self._more.winfo_rooty() +
                                                                         self._more.winfo_height()))
        self._more_menu = tk.Menu(self._more, tearoff=0)
        self._crumb_ws: List[Tuple[Optional[ttk.Label], ttk.Button, str, str]] = []
        self._crumb_hidden: List[Tuple[str, str]] = []
        self._edit_var = tk.StringVar(self)
        self._edit_var.trace_add("write", self._edit_changed)
        self._edit = ttk.Entry(self._pathbar, textvariable=self._edit_var)
        for w in (self._pathbar, self._crumb_box):
            w.bind("<Button-1>", lambda _e: self.edit_path())
        self._pathbar.bind("<Configure>", lambda _e: self._fit_crumbs())
        col = 4
        ttk.Button(bar, text="Refresh", image=ic.get("refresh"), compound="left", style="Flat.TButton",
                   command=self.refresh, takefocus=0).grid(row=0, column=col)
        if self.mode != "input":
            col += 1
            ttk.Button(bar, text="New folder", image=ic.get("folder_new"), compound="left", style="Flat.TButton",
                       command=self.new_folder, takefocus=0).grid(row=0, column=col)
        col += 1
        self._sort_button = ttk.Menubutton(bar, text="Sort", width=5, takefocus=True)
        self._sort_button.grid(row=0, column=col, padx=(px(4), 0))
        self._sort_column_var, self._sort_order_var = tk.StringVar(self), tk.StringVar(self)
        sort_menu = tk.Menu(self._sort_button, tearoff=0, postcommand=self._sort_menu_state)
        self._sort_button.configure(menu=sort_menu)
        for column, label in (("name", "Name"), ("tiles", "Tiles"), ("inside", "What's inside"),
                              ("modified", "Modified")):
            if column != "tiles" or self.mode == "input":
                sort_menu.add_radiobutton(label=label, value=column, variable=self._sort_column_var,
                                          command=lambda c=column: self._sort_by(c, _SORT_DEFAULT[c]))
        sort_menu.add_separator()
        for order, label in (("asc", "Ascending"), ("desc", "Descending")):
            sort_menu.add_radiobutton(label=label, value=order, variable=self._sort_order_var,
                                      command=lambda d=order: self._sort_by(self._sort[0], d))
        self._filter_var = tk.StringVar(self)
        self._filter = ttk.Entry(bar, textvariable=self._filter_var, width=14)
        self._filter.grid(row=0, column=col + 1, padx=(px(4), 0))
        hint = ttk.Label(self._filter, text="Filter", style="Placeholder.TLabel")
        hint.bind("<Button-1>", lambda _e: self._filter.focus_set())

        def show_hint(*_a: Any) -> None:
            if self._filter_var.get() or str(self.tk.call("focus")) == str(self._filter):
                hint.place_forget()
            else:
                hint.place(x=px(7), rely=0.5, anchor="w")
        for seq in ("<FocusIn>", "<FocusOut>"):
            self._filter.bind(seq, lambda _e: self.after_idle(show_hint), add="+")
        self._filter_var.trace_add("write",
                                   lambda *_: (show_hint(), self._later("filter", 150, self._filters_changed)))
        show_hint()
        ttk.Separator(self).grid(row=1, column=0, sticky="ew")
        pw = self._pw = ttk.Panedwindow(self, orient="horizontal")
        pw.grid(row=2, column=0, sticky="nsew")
        side_f = ttk.Frame(pw, style="Sidebar.TFrame", padding=(px(4), px(6), 0, px(4)))
        side_f.rowconfigure(0, weight=1)
        side_f.columnconfigure(0, weight=1)
        side = self._side = ttk.Treeview(side_f, show="tree", style="Sidebar.Treeview", selectmode="browse")
        side.grid(row=0, column=0, sticky="nsew")
        side.column("#0", width=px(200), stretch=True)
        side.bind("<Configure>", lambda _e: [self._paint_place(i) for i in self._place_items])
        side_bar = ttk.Scrollbar(side_f, orient="vertical", command=side.yview)

        def side_scroll(first: str, last: str) -> None:   # a scrollbar only when the places don't fit
            if float(first) <= 0.0 and float(last) >= 1.0:
                side_bar.grid_remove()
            else:
                side_bar.grid(row=0, column=1, sticky="ns")
            side_bar.set(first, last)
        side.configure(yscrollcommand=side_scroll)
        tag_tree(side, "Sidebar")
        pw.add(side_f, weight=0)
        lf = self._list_frame = ttk.Frame(pw, style="Surface.TFrame")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        tree = self.tree = ttk.Treeview(lf, columns=("tiles", "inside", "modified"), style="Listing.Treeview",
                                        selectmode="browse")
        tree.grid(row=0, column=0, sticky="nsew")
        self._vsb = ttk.Scrollbar(lf, orient="vertical", command=tree.yview)
        self._vsb.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=self._yscroll)
        font = self.ctx.theme.fonts["YfsBody"]
        width = lambda sample: font.measure(sample) + px(18)                       # noqa: E731
        for col_id, head, anchor, sample, stretch in (
                ("#0", "Name", "w", "Slide_folder_name_round1", True), ("tiles", "Tiles", "e", "00,000", False),
                ("inside", "What's inside", "w", "Raw scan - unmix in inForm first", True),
                ("modified", "Modified", "w", "08 Jan 2026", False)):
            tree.heading(col_id, text=head, anchor=anchor, command=lambda c=col_id: self._sort_by(c))
            tree.column(col_id, width=width(sample) + (self.ctx.icons.size if col_id == "#0" else 0), anchor=anchor,
                        stretch=stretch, minwidth=px(140) if stretch else px(40))
        tag_tree(tree, "Listing")
        lf.bind("<Configure>", lambda _e: self._columns())
        self._ov = ttk.Frame(lf, style="Card.TFrame", padding=px(16))
        pw.add(lf, weight=1)
        self._preview = ttk.Frame(pw, style="Sidebar.TFrame", padding=px(10))
        self._facts = SlideFacts(self._preview, self.ctx)
        self._facts.pack(fill="both", expand=True)
        self._preview_on, self._side_w = False, 0         # (_side_w: the sidebar's width while the dialog is open)
        foot = ttk.Frame(self, style="Header.TFrame", padding=(px(10), px(6), px(10), px(8)))
        foot.grid(row=3, column=0, sticky="ew")
        foot.columnconfigure(0, weight=1)
        opts = ttk.Frame(foot, style="Surface.TFrame")
        opts.grid(row=0, column=0, columnspan=3, sticky="ew")
        self._footer_options = [(text, var, under) for text, var, under in (
            ("Only folders with tiles", self._v_tiles, 18), ("Show hidden", self._v_hidden, 5),
            ("Show all files", self._v_all, 5)) if var is not self._v_tiles or self.mode == "input"]   # (BR-2)
        for text, var, under in self._footer_options:     # in the Tab order, with Alt+underlined letter (BR-13)
            ttk.Checkbutton(opts, text=text, variable=var, style="Surface.TCheckbutton", underline=under,
                            command=self._filters_changed).pack(side="left", padx=(0, px(14)))
        self._note = ttk.Label(opts, style="Surface.Small.TLabel")       # a momentary note, at the right; the
        self._note.pack(side="right")                                       # listing's notes take the room left
        self._others = ttk.Label(opts, style="Surface.Small.TLabel")
        self._others.pack(side="left")
        self._notes = {self._others: "", self._note: ""}                     # their whole texts (_fit_notes)
        for label in self._notes:
            Tooltip(label, lambda w=label: self._notes[w] if str(w.cget("text")) != self._notes[w] else "")
        opts.bind("<Configure>", lambda _e: self._fit_notes())
        self._strip = ttk.Frame(foot, style="Info.TFrame", padding=(px(8), px(5)))
        self._strip.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(px(6), px(8)))
        self._strip.columnconfigure(1, weight=1)
        self._strip_icon = ttk.Label(self._strip, style="Info.Banner.TLabel")
        self._strip_icon.grid(row=0, column=0, sticky="nw", padx=(0, px(6)))
        self._strip_text = ttk.Label(self._strip, style="Info.Banner.TLabel", justify="left")
        self._strip_text.grid(row=0, column=1, sticky="w")
        self._strip_btn = ttk.Button(self._strip, style="Info.Banner.TButton", text="Go there", underline=0)
        self._strip.rowconfigure(0, minsize=self._strip_btn.winfo_reqheight())   # no jump when an action shows
        self._strip.bind("<Configure>", lambda _e: self._wrap_strip())
        self._picker = ttk.Button(foot, text=self.T(f"Use the system folder picker{ELL}"), style="Link.TButton",
                                  command=self._system_picker, underline=22)
        self._cancel = ttk.Button(foot, text="Cancel", command=self.cancel)
        self._cancel.grid(row=2, column=1, padx=(0, px(8)), sticky="e")
        self._primary = ttk.Button(foot, style="Primary.TButton", command=self.choose)
        self._primary.grid(row=2, column=2, sticky="e")
        self._primary.bind("<Return>", lambda _e: self.choose())
        Tooltip(self._primary, lambda: self.T(_PRIMARY[self.mode] + self.q(self.target())))
        foot.bind("<Configure>", lambda _e: self._fit_primary())

    def _bindings(self) -> None:
        tree, side, edit = self.tree, self._side, self._edit
        cmd = ["Control", "Command"] if self.ctx.theme.ws == "aqua" else ["Control"]
        on = lambda seqs, fn: [self.bind(s, lambda e, f=fn: (f(), "break")[1]) for s in seqs]   # noqa: E731
        on([f"<{m}-Return>" for m in cmd] + [f"<{m}-KP_Enter>" for m in cmd], self.choose)
        on(["<Alt-Left>", "<Command-bracketleft>"] if "Command" in cmd else ["<Alt-Left>"], self.back)
        on(["<Alt-Right>", "<Command-bracketright>"] if "Command" in cmd else ["<Alt-Right>"], self.forward)
        on(["<Alt-Up>", "<Command-Up>"] if "Command" in cmd else ["<Alt-Up>"], self.up)
        on([f"<{m}-l>" for m in cmd] + ["<F6>"] + (["<Command-Shift-G>"] if "Command" in cmd else []), self.edit_path)
        on(["<F5>"] + [f"<{m}-r>" for m in cmd], self.refresh)
        on([f"<{m}-f>" for m in cmd], lambda: (self._filter.focus_set(), self._filter.select_range(0, "end")))
        on([f"<{m}-h>" for m in cmd], lambda: self._toggle(self._v_hidden))
        on([f"<{m}-N>" for m in cmd], self.new_folder)
        if "Command" not in cmd:                           # Alt+letter mnemonics (Option types characters on a Mac)
            for text, var, under in self._footer_options:
                on([f"<Alt-{text[under]}>"], lambda v=var: self._toggle(v))
            on(["<Alt-p>"], lambda: self._picker.winfo_manager() and self._system_picker())
            self.bind("<Alt-KeyPress>", self._strip_key)
        self.bind("<Escape>", self._escape)
        self.bind("<Configure>", self._on_configure)
        for w in (edit, self._filter):                     # Ctrl+H must not also delete a character in an entry
            for m in cmd:
                w.bind(f"<{m}-h>", lambda e: (self._toggle(self._v_hidden), "break")[1])
        for seq in ("Return", "KP_Enter"):                 # Ctrl+Enter is the primary action, not "open" (in path
            tree.bind(f"<{seq}>", lambda e: (self._open(tree.focus()), "break")[1])      # edit: the typed path's)
            edit.bind(f"<{seq}>", lambda e: (self._edit_enter(), "break")[1])
            for m in cmd:
                tree.bind(f"<{m}-{seq}>", lambda e: (self.choose(), "break")[1])
                edit.bind(f"<{m}-{seq}>", lambda e: (self._edit_enter(choose=True), "break")[1])
        tree.bind("<Double-1>", lambda e: self._open(tree.identify_row(e.y)))
        tree.bind("<ButtonRelease-1>", self._released)
        tree.bind(f"<{cmd[-1]}-Button-1>", self._ctrl_click)
        tree.bind("<Down>", lambda e: self._arrow(1))
        tree.bind("<Up>", lambda e: self._arrow(-1))
        self._list_tip = _RowTip(tree)
        tree.bind("<Motion>", self._tree_motion, add="+")
        tree.bind("<BackSpace>", lambda e: (self.up(), "break")[1])
        tree.bind("<<TreeviewSelect>>", lambda e: self._update_target())
        tree.bind("<Key>", self._type_ahead)
        for key, fn in (("<Home>", lambda k: k[:1]), ("<End>", lambda k: k[-1:]),
                        ("<Prior>", lambda k: k[max(0, self._focus_index(k) - self._page()):][:1]),
                        ("<Next>", lambda k: k[min(len(k) - 1, self._focus_index(k) + self._page()):][:1])):
            tree.bind(key, lambda e, f=fn: (self._select_key((f(self._keys()) or [None])[0]), "break")[1])
        edit.bind("<Tab>", lambda e: (self._tab(), "break")[1])
        edit.bind("<Down>", lambda e: (self._lb_move(1), "break")[1])
        edit.bind("<Up>", lambda e: (self._lb_move(-1), "break")[1])
        edit.bind("<KeyRelease>", self._edit_typed)
        edit.bind("<FocusOut>", lambda e: self._later("editfocus", 10, self._edit_focus_lost))
        side.bind("<ButtonRelease-1>", lambda e: self._open_place(side.identify_row(e.y)))
        side.bind("<Return>", lambda e: self._open_place(side.focus()))
        side.bind("<Down>", lambda e: self._side_move(1))
        side.bind("<Up>", lambda e: self._side_move(-1))
        side.bind("<FocusOut>", lambda e: self._mark_place())   # arrows alone don't move: show where we are
        side.bind("<<TreeviewSelect>>", lambda e: [side.selection_remove(i) for i in side.selection()
                                                   if i not in self._place_items])
        for seq in ["<Button-3>"] + (["<Button-2>", "<Control-Button-1>"] if "Command" in cmd else []):
            side.bind(seq, self._place_menu)
        self._side_tip = _RowTip(side)
        side.bind("<Motion>", self._side_motion, add="+")

    def _place_window(self, master: tk.Misc) -> None:
        """Remembered size (logical px), else px(1080) x px(680); clamped to 90 % of the screen; centred on the
        main window."""
        px, sw, sh = self.px, self.winfo_screenwidth(), self.winfo_screenheight()
        try:
            w, h = (px(int(v)) for v in str(self.ctx.settings.get("browser.geometry") or "").split("x"))
        except ValueError:
            w, h = px(1080), px(680)
        w, h = min(max(w, px(720)), int(sw * 0.9)), min(max(h, px(480)), int(sh * 0.9))
        self.minsize(min(px(720), int(sw * 0.9)), min(px(480), int(sh * 0.9)))
        top = master.winfo_toplevel()
        if top.winfo_ismapped():
            x, y = top.winfo_rootx() + (top.winfo_width() - w) // 2, top.winfo_rooty() + (top.winfo_height() - h) // 2
        else:
            x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{max(0, min(x, sw - w))}+{max(0, min(y, sh - h))}")
        self._show_preview(w >= px(1000))

    def _init_sashes(self) -> None:
        """Sidebar at its remembered width (or the width it has had since the dialog opened), preview at px(230);
        waits until the panes have a real size."""
        width = self._pw.winfo_width()
        if width <= 1:
            self._later("sash", 50, self._init_sashes)
            return
        try:
            side = self._side_w or self.px(int(self.ctx.settings.get("browser.sash") or 210))
        except (TypeError, ValueError):
            side = self.px(210)
        if self._preview_on:                              # the far sash first: a sash can't pass its neighbour
            self._pw.sashpos(1, width - self.px(236))
        self._pw.sashpos(0, max(self.px(120), min(side, width // 3)))

    # ------------------------------------------------------------------------------------- navigation
    def _first(self, folder: str, select: Optional[str]) -> None:
        """The first navigation goes through `locate`: a missing folder opens its closest existing ancestor."""
        folder = os.path.normpath(folder)
        self._ticket = self.ctx.fs.new_ticket()
        self._set_location(folder, select)

        def missing(closest: Optional[str], _found: str) -> None:
            self.navigate(closest or folder, push=False)
            if closest:
                self._note_set(f"{self.q(_base(folder))} no longer exists — showing the closest folder that does.")
        self._locate(folder, missing, self._go_back, select=select, push=False)

    def _locate(self, path: str, missing: Callable[[Optional[str], str], None], back: Callable[[], None],
                select: Optional[str] = None, push: bool = True, then: Optional[Callable[[], None]] = None,
                resolution: Optional[_LinkState] = None) -> None:
        """Ask the lane what `path` is: a folder opens, a file opens its folder with the file selected (then `then`
        runs), a missing path calls missing(closest existing ancestor, resolved path). It is the loading phase
        (§2.2.8): slow and not-answering states with `back` as Go back (UD1). Any navigation makes its answer
        obsolete (CT-4)."""
        self._new_intent()
        token = self._locate_token = next(_TOKENS)
        state = resolution or _link_start(path, known=self._path if self._listing is not None else "")

        def again() -> None:
            self.ctx.fs.retry(self._wait_path)
            self._locate(path, missing, back, select, push, then, state)

        def failed(err: FsError) -> None:
            if err.code in _STALL:
                self._unanswered(FsError(err.code, f"{self._place_name(self._wait_path)} didn't answer."))
                return
            self._locate_token = 0
            self._end_wait()
            if same_path(path, self._path):
                self._list_error = err
                if self._listing is None:
                    self._clear_tree()
                self._show_error(err)
            else:
                self._strip_msg, self._edit_error = ("Error", f"Can't open {self.q(_base(path))}: {err.message}", None), True
            self._update_target()

        def located(e: FsDone) -> None:
            if token != self._locate_token:
                return
            if e.error is not None:
                failed(e.error)
                return
            self._locate_token = 0
            self._end_wait()
            kind, found, closest = e.result
            if kind == "missing":
                missing(closest, found)
                return
            self._located_path = found if kind == "dir" else os.path.dirname(found)
            if kind == "dir":
                self.navigate(found, push=push, select=select)
            else:
                self.navigate(os.path.dirname(found), push=push, select=_base(found))
                self._select_file = True
            if then is not None:
                then()

        def watching(rid: int, where: str, pending: _LinkState) -> None:
            nonlocal state
            state = pending
            self._wait(rid, where, again, back)

        def resolved(found: str, err: Optional[FsError]) -> None:
            if err is not None and err.code not in ("not_found", "not_a_dir"):
                failed(err)
                return
            # Missing components cannot hide links below them. Locate finds the closest existing parent;
            # every component above it was resolved before any following stat or listing (FS-9).
            rid = self._call(found, "locate", partial(_locate, found), located, user=True)
            self._wait(rid, found, again, back)
        self._resolve_links(state, resolved, watching, lambda: token == self._locate_token, 0)

    def _resolve_links(self, state: _LinkState, done: Callable[[str, Optional[FsError]], None],
                       watching: Callable[[int, str, _LinkState], None], valid: Callable[[], bool], ticket: int) -> None:
        """Read each component before descending into it, in its parent's lane. Mount roots are never lstatted
        on their parent's lane. A saved state resumes a timed-out hop, including its remaining suffix (UD3)."""
        if not valid():
            return
        parent, parts, hops = state
        if not parts:
            done(parent, None)
            return
        fs, path = self.ctx.fs, os.path.join(parent, parts[0])
        if fs.lane_key(path) != fs.lane_key(parent) or any(same_path(path, m.mount_point) for m in fs.mounts.mounts()):
            self._resolve_links((path, parts[1:], hops), done, watching, valid, ticket)
            return

        def answered(e: FsDone) -> None:
            if not valid():
                return
            if e.error is not None:
                done(os.path.join(parent, *parts), e.error)
            elif e.result is None:
                self._resolve_links((path, parts[1:], hops), done, watching, valid, ticket)
            elif hops >= 8:
                done(os.path.join(parent, *parts), FsError("io", "It leads through too many links."))
            else:
                self._resolve_links(_link_start(os.path.join(e.result, *parts[1:]), hops + 1, parent),
                                    done, watching, valid, ticket)
        rid = fs.readlink(path, ticket=ticket, owner="browser")
        self._calls[rid] = (ticket if ticket == self._ticket else None, answered)
        watching(rid, path, state)

    def _cancel_locate(self) -> None:
        """Stop waiting for a typed path: stay in the folder shown."""
        self._locate_token, self._stall = 0, None
        self._end_wait()

    def _set_location(self, path: str, select: Optional[str]) -> None:
        """Everything about a new location that needs no I/O: state, crumbs, the 'Opening' row, the target."""
        self._path, self._select, self._select_file, self._pinned = path, select, False, None
        if self._keep_view is not None and not same_path(self._keep_view[0], path):
            self._keep_view = None                        # F5's scroll position is for its own folder
        self._use_cache, self._no_cache = not self._no_cache, False      # F5 ignores cached listings and probes
        self._listing = self._summary = self._list_error = self._no_answer = self._stall = None
        self._clear_tree()
        self._rows, self._names, self._statted, self._dating = {}, {}, set(), set()
        self._stat_batches = {}
        self._roll_budget, self._probe_queue, self._layout = ROLLUP_PER_VIEW, [], []
        self._keep_waiting, self._strip_msg, self._loading = False, None, False
        self._lane_class = self.ctx.fs.lane_class(path)
        self._fresh = 10.0 if self._lane_class == LOCAL else 60.0
        idx = self.ctx.output_index()
        self._out_names = frozenset(idx.outputs) if idx is not None else frozenset()
        for job in ("layout", "probe", "links", "empty", "bar", "slow", "visible", "dates", "again", "filter"):
            self._cancel_job(job)                         # (navigate() cleared the filter: nothing to apply)
        self._hide_bar()
        self._edit_off()
        self._overlay_hide()
        self._note_set("")
        self.tree.insert("", "end", iid="__opening", text=self.T(f"Opening {self.q(_base(path))}{ELL}"),
                         image=self.ctx.icons.tree("wait"), tags=("dim",))
        self._others_set("")
        self._crumbs()
        for name, on in (("back", self._back), ("fwd", self._fwd), ("up", os.path.dirname(path) != path)):
            self._nav[name].state(["!disabled"] if on else ["disabled"])
        self._mark_place()
        if self._preview_on:                              # never the last folder's slide while this one loads (BR-9)
            self._facts.show_folder(None, None, None, title=self._title_of(path))
        self._update_target()

    def _list(self) -> None:
        rid = self.ctx.fs.list_dir(self._path, ticket=self._ticket, owner="browser", priority=0)
        self._calls[rid] = (self._ticket, self._on_list)
        self._wait(rid, self._path, self._retry, self._go_back)

    def _wait(self, rid: int, path: str, again: Callable[[], None], back: Callable[[], None]) -> None:
        """`rid` (a listing, or a locate) is the loading request now: the bar after 300 ms, then its FsSlow drive
        the waiting and not-answering states; `again` is their Try again, `back` their Go back."""
        self._loading, self._keep_waiting, self._stall = True, False, None
        self._wait_rid, self._wait_path, self._wait_again, self._wait_back = rid, path, again, back
        if same_path(path, self._path):
            self._no_answer = None
        if self._ov_kind in ("wait", "unresponsive"):
            self._overlay_hide()
        self._t0 = time.monotonic()
        self._cancel_job("again")
        self._later("bar", 300, self._show_bar)

    def _end_wait(self) -> None:
        """The loading request answered: no bar, no waiting card, nothing to ask again."""
        self._loading = False
        self._hide_bar()
        self._cancel_job("slow")
        self._cancel_job("again")
        if self._ov_kind in ("wait", "unresponsive"):
            self._overlay_hide()
            self._empty_state()

    def _on_list(self, e: FsDone) -> None:
        if e.req_id == self._wait_rid:                    # (a typed path may be what loads now)
            self._end_wait()
        if e.error is not None:
            self._list_error = e.error
            if self._listing is None:
                self._clear_tree()
            self._show_error(e.error)
            self._update_target()
            return
        if self._ov_kind not in ("wait", "unresponsive"):
            self._overlay_hide()
        self._render(e.result)
        if e.elapsed >= self.ctx.fs.SLOW_S:
            self._note_set(f"Answered after {int(e.elapsed)} s")
        if self.ctx.fs.auto_enter(self._path) == "autofs":
            self.ctx.fs.reload_mounts()                   # opening it mounted a share: learn its lane (UD2)

    def _open(self, key: str) -> None:
        r = self._rows.get(key)
        if r is not None and r.kind == "dir":
            if r.link and r.target is None:
                self._locate(self._row_path(r), lambda *_a: self._show_error(FsError("not_found", "Link target missing.")),
                             self._cancel_locate, resolution=r.resolving)
            else:
                self.navigate(self._row_path(r))          # a link opens the real folder (UD3)

    def _child(self, name: str) -> str:
        return os.path.join(self._path, name)

    def _row_path(self, r: _Row) -> str:
        """A link's resolved target, or its last pending destination for display only until resolution finishes."""
        if r.resolving is not None and r.target is None:
            return os.path.join(r.resolving[0], *r.resolving[1])
        return r.target or self._child(r.name)

    def _focus_list(self, hiding: Optional[tk.Misc] = None) -> None:
        """The keyboard goes to the listing (BUX-M1): after a navigation, unless the user is typing a filter, or,
        with `hiding`, only if it is on that widget, which is about to disappear (BR-14). Never taken from
        another window."""
        focus = str(self.tk.call("focus"))
        if hiding is not None:
            if focus == str(hiding) or focus.startswith(str(hiding) + "."):
                self.tree.focus_set()
        elif focus in ("", str(self)) or (focus.startswith(str(self) + ".") and focus != str(self._filter)):
            self.tree.focus_set()

    def _go_back(self) -> None:
        """Back in history; without any (the first folder), up to the parent, or out of a share that is slow or
        isn't answering: its parent folder would only wait for it again."""
        if self._back:
            self.back()
            return
        path = self._wait_path if self._loading else self._path
        fs, parent = self.ctx.fs, os.path.dirname(path)
        lane = fs.lane_key(path)
        if fs.lane_class(path) != LOCAL and fs.health(lane) in (H_SLOW, H_UNRESPONSIVE):
            while fs.lane_key(parent) == lane and os.path.dirname(parent) != parent:
                parent = os.path.dirname(parent)
        if parent != self._path:
            self.navigate(parent, push=False)

    def _retry(self) -> None:
        self.ctx.fs.retry(self._path)
        self.navigate(self._path, push=False, select=self._select)

    # ------------------------------------------------------------------------------------- listing
    def _render(self, listing: Any) -> None:
        """Build the rows of a listing (names only; tiles collapse into one row), then show and probe them."""
        r0 = self._target_row()
        if self._select is None and r0 is not None:
            self._select = r0.name                       # keep the selection across a refresh
        old = {r.name: r for r in self._rows.values()}
        self._listing, self._list_error, self._no_answer = listing, None, None
        self._summary = summarize_listing(listing.path, listing.entries, truncated=listing.truncated)
        self._clear_tree()
        self._rows, self._names = {}, {}
        for e in listing.entries:
            low = e.name.casefold()
            kind = "dir" if e.is_dir else classify_file(e.name)
            if kind == "tile":
                continue
            hidden = (e.hidden or low.startswith(".") or kind == "ignored"
                      or (e.is_dir and (low in EXCLUDED_DIR_NAMES or low.startswith(".trash-"))))
            row_kind = "dir" if e.is_dir else "file" if kind in _NOTABLE else "other"
            r = old.get(e.name)
            if r is None or r.link != e.is_link or (r.kind != row_kind and not r.link):    # a followed link keeps
                r = _Row(f"r{next(self._seq)}", e.name, row_kind, hidden, e.is_link, e.size, e.mtime)
                if row_kind == "dir" and not e.is_link:  # cloud, disconnected, unmounted autofs, pseudo (§2.2.4)
                    r.auto = self.ctx.fs.auto_enter(self._child(e.name))
            self._rows[r.key], self._names[r.name] = r, r.key
        if self._summary.groups:
            r = _Row("tiles", "", "tiles", False, False, None, None)
            self._rows[r.key] = r
        self._follow_links()                             # first: the empty state waits for links that may be folders
        self._relayout()
        self._probe_children()
        self._update_target()

    def _clear_tree(self) -> None:
        """Delete every item, including the rows a filter detached (they would pile up otherwise)."""
        self.tree.delete(*(self._created | set(self.tree.get_children(""))))
        self._created, self._full, self._shown, self._cut = set(), {}, {}, {}

    def _passes(self, r: _Row) -> bool:
        return r.key == self._pinned or self._shows(r)

    def _only_tiles(self) -> bool:
        """'Only folders with tiles' is for finding tiles: the save and temporary-files choosers ignore it (BR-2)."""
        return self.mode == "input" and self._v_tiles.get()

    def _shows(self, r: _Row, tiles: bool = True) -> bool:
        """The filters: hidden items, other files, 'Only folders with tiles' (unless `tiles` is False) and the
        filter text."""
        if (r.hidden and not self._v_hidden.get()) or (r.kind == "other" and not self._v_all.get()):
            return False
        if tiles and r.kind == "dir" and self._only_tiles() and not self._maybe_tiles(r):
            return False
        f = self._filter_var.get().strip().casefold()
        if not f:
            return True
        name = (describe_tiles(self._summary)[0] if r.kind == "tiles" else r.name).casefold()
        return fnmatch.fnmatchcase(name, f) if ("*" in f or "?" in f) else f in name

    def _maybe_tiles(self, r: _Row) -> bool:
        """Only folders with tiles: tiles or slide folders found, still being looked at, or not known yet (§2.2.6,
        BUX-M2)."""
        if self._unknown(r):
            return True
        if r.auto or r.error is not None:
            return False
        return (r.summary is None or bool(r.summary.groups) or (r.roll is not None and (r.roll[1] or r.roll[4])))

    def _unknown(self, r: _Row) -> bool:
        """A folder whose contents can't be known yet: cloud, network drives, or it (or a folder in it) didn't
        answer or couldn't be read, or it holds shares or cloud folders, which are never looked into."""
        err = self._row_error(r)
        return (r.auto in _UNCHECKED or err is not None or bool(r.roll_errors)
                or bool(r.roll is not None and r.roll[7] and not r.roll[1]))

    def _filters_changed(self) -> None:
        """The user changed a filter: from now on it applies to the row they were shown although it hides it, and
        to the row they chose (BR-3)."""
        self._pinned = None
        self._relayout(refilter=True)

    def _toggle(self, var: tk.BooleanVar) -> None:
        var.set(not var.get())
        self._filters_changed()

    def _relayout(self, refilter: bool = False) -> None:
        """Show the rows that pass the filters, in the sort order; tree items are created 500 at a time. A row
        asked for (the field's folder, the folder just left, a picked or new folder) is shown and selected even
        when a filter hides it, and the footer says so: never silently dropped (BR-2). So is the row chosen when
        what its probe found would hide it: only the user's own filter change (`refilter`) unchooses it (BR-3)."""
        if self._listing is None:
            return
        sel = self.tree.selection()
        if self._select in self._names:                   # (also one that a probe still out may hide later)
            self._pinned = self._names[self._select]
        elif sel and not refilter and sel[0] in self._rows and not self._shows(self._rows[sel[0]]):
            self._pinned = sel[0]
        rows = [r for r in self._rows.values() if self._passes(r)]
        col, order = self._sort
        rows.sort(key=lambda r: natural_key(r.name), reverse=col == "name" and order == "desc")
        if col != "name":
            value = {"tiles": lambda r: self._tile_count(r), "modified": lambda r: r.mtime or 0.0,
                     "inside": lambda r: self._badge(r).rank if r.kind == "dir" else 0}[col]
            rows.sort(key=value, reverse=order == "desc")
        if col == "modified":
            rows.sort(key=lambda r: r.mtime is None)     # rows without a date last, newest or oldest first (BR-10)
        rows.sort(key=lambda r: _GROUP[r.kind])
        self._layout, self._layout_pos = [r.key for r in rows], 0
        self._layout_keys = set(self._layout)
        for c in ("#0", "tiles", "inside", "modified"):
            on = c == col or (c == "#0" and col == "name")
            self.tree.heading(c, image=self.ctx.icons.get("sort_down" if order == "desc" else "sort_up") if on else "")
        self._date_rows()
        pinned = self._rows.get(self._pinned or "")      # (first: it explains a row shown, and chosen)
        notes = [f"Also showing {self.q(pinned.name)}, which your filters hide"] \
            if pinned is not None and not self._shows(pinned) else []
        hidden = sum(1 for r in self._rows.values() if r.kind == "other" and not r.hidden) if not self._v_all.get() else 0
        notes += [_n(hidden, "other file") + " not shown"] if hidden else []
        bare = sum(1 for r in self._rows.values() if r.kind == "dir" and r.key not in self._layout_keys
                   and self._shows(r, tiles=False)) if self._only_tiles() else 0
        notes += [_n(bare, "folder") + " without tiles not shown"] if bare else []     # what the option hides
        notes += ["Showing the first 20,000 items"] if self._listing.truncated else []
        notes += [f"Sorting by date{ELL}"] if self._dating or "dates" in self._jobs else []
        failed_dates = sum(1 for r in rows if r.date_error is not None)
        if failed_dates:
            notes.append(f"Couldn't read {_n(failed_dates, 'date')} — F5 to try again")
        self._others_set(" · ".join(notes))
        self._continue_layout()

    def _continue_layout(self) -> None:
        gone = self._created - self._layout_keys if self._layout_pos == 0 else ()
        if gone:                                          # a hidden row can't stay chosen: Tk would keep it (BR-3)
            self.tree.selection_remove(*gone)
        made = 0
        for key in self._layout[self._layout_pos:]:
            if key not in self._created:
                if made == 500:
                    break
                self.tree.insert("", "end", iid=key, **self._item(self._rows[key]))
                self._created.add(key)
                made += 1
            self._layout_pos += 1
        self.tree.set_children("", *self._layout[:self._layout_pos])
        if self._layout_pos < len(self._layout):
            self._later("layout", 1, self._continue_layout)
            return
        if self._select is not None:
            key = self._names.get(self._select)
            self._select = None
            if key in self._layout_keys:
                self._select_key(key)
        if self._keep_view is not None and same_path(self._keep_view[0], self._path):
            _path, top, focus = self._keep_view           # F5 left the list where it was (BR-17)
            self._keep_view, key = None, self._names.get(focus or "")
            if key in self._layout_keys and not self.tree.selection():
                self.tree.focus(key)
            self.tree.yview_moveto(top)
        self._empty_state()
        self._later("visible", 50, self._visible_work)
        self._update_target()

    def _item(self, r: _Row) -> Dict[str, Any]:
        """Tree item options of a row: icon, name, Tiles, What's inside, Modified, tags; the name and the badge
        are shortened to their columns (_fitted)."""
        ic, mod = self.ctx.icons, time.strftime("%d %b %Y", time.localtime(r.mtime)) if r.mtime else ""
        if not mod and r.date_error is not None:
            mod = "No date"                               # the footer explains why; fits the Modified column
        lead = ""                                         # a link: where it leads (its tooltip)
        if r.kind == "dir":
            b = self._badge(r)
            image, tiles, inside = ic.tree(b.icon, dim=b.dim and b.icon != "folder_error"), b.tiles_text, b.inside
            tags: Tuple[str, ...] = ("slide",) if b.bold else ("dim",) if b.dim else ()
            name = r.name
            if r.link:                                    # it opens another place: say so
                inside, lead = inside + " · link", f" {self.G['arrow']} {r.target}" if r.target else ""
        elif r.kind == "tiles":
            (name, inside), image, tiles, tags = describe_tiles(self._summary), ic.tree("file_tiles"), "", ()
        else:
            icon, inside, dim = describe_file(r.name, r.size)
            image, tiles, tags, name = ic.tree(icon, dim=dim), "", ("dim",) if dim else (), r.name
        self._full[r.key] = (self.T(name), tiles, self.T(inside), mod, "slide" in tags, self.T(lead))
        text, values = self._shown[r.key] = self._fitted(r.key)
        return dict(text=text, image=image, values=values, tags=tags + (("hidden",) if r.hidden else ()))

    def _fitted(self, key: str) -> Tuple[str, Tuple[str, str, str]]:
        """(name, values) of a row with the name and the badge shortened, with an ellipsis, to the room their
        columns give; a shortened row (or a link) shows both in full as a tooltip (BR-26)."""
        name, tiles, inside, mod, bold, lead = self._full[key]
        ruler, ell = self._rulers[bold], self.G["ell"]
        short = ruler.fit(name, self._fit_widths[0], ell, middle=True)
        badge = ruler.fit(inside, self._fit_widths[1], ell)
        if (short, badge) != (name, inside) or lead:
            self._cut[key] = "\n".join(t for t in (name + lead, inside) if t)
        else:
            self._cut.pop(key, None)
        return short, (tiles, badge, mod)

    def _refit(self) -> None:
        """The columns changed width: draw every row's name and badge again in the room they have now, the rows in
        view first, 500 rows per event-loop turn (a folder of 20,000 rows never holds the window up)."""
        t, px = self.tree, self.px
        room = (t.column("#0", "width") - self.ctx.icons.size - px(14), t.column("inside", "width") - px(10))
        if room != self._fit_widths:
            self._fit_widths, self._refit_keys = room, self._visible_keys() + list(self._created)
        keys, self._refit_keys = self._refit_keys[:500], self._refit_keys[500:]
        for key in keys:
            shown = self._fitted(key) if key in self._created else None
            if shown is not None and shown != self._shown[key]:
                self._shown[key] = shown
                t.item(key, text=shown[0], values=shown[1])
        if self._refit_keys:
            self._later("refit", 1, self._refit)

    def _badge(self, r: _Row) -> Badge:
        looking = r.roll is not None and r.roll[4] and not r.roll[1]   # nothing found yet, still looking below
        if r.roll is not None and r.roll[7] and not r.roll[1] and not looking and not r.roll_errors \
                and self._row_error(r) is None:
            return Badge("folder", "", f"No tiles found so far · {_n(r.roll[7], 'folder')} not checked",
                         False, False, 7)
        return describe_folder(None if looking else r.summary, self._rollup(r), self.mode, self._out_names,
                               self._lane_class, self._row_error(r), r.auto, failed=len(r.roll_errors))

    @staticmethod
    def _rollup(r: _Row) -> Optional[Rollup]:
        """The row's roll-up; None until it has one, or while it found nothing and probes are still out."""
        if r.roll is None or (r.roll[4] and not r.roll[1]):
            return None
        return Rollup(r.roll[0], r.roll[1], r.roll[2], bool(r.roll[3] or r.roll[4] or r.roll[7] or r.roll_errors))

    def _row_error(self, r: _Row) -> Optional[FsError]:
        """The row's own error; a folder whose subfolders didn't answer, with no slide folder found among the rest,
        didn't answer either: nothing can be said about it yet."""
        if r.error is not None or r.roll is None or not r.roll[6] or r.roll[1] or r.roll[4]:
            return r.error
        return FsError("timeout", f"{self.ctx.fs.mounts.place_name(r.roll[6][0])} didn't answer.")

    def _tile_count(self, r: _Row) -> int:
        return sum(_stitchable(r.summary).values()) if r.summary is not None else -1

    def _refresh_row(self, r: _Row) -> None:
        if r.key in self._created:
            self.tree.item(r.key, **self._item(r))
        if self._sort[0] != "name" or self._only_tiles():
            self._later("resort", 400, self._relayout)
        if (r.key in self.tree.selection() or not self.tree.selection()) and "target" not in self._jobs:
            self._later("target", 50, self._update_target)    # throttled, like the preview (BR-9)

    def _sort_menu_state(self) -> None:
        self._sort_column_var.set(self._sort[0])
        self._sort_order_var.set(self._sort[1])

    def _sort_by(self, col: str, order: Optional[str] = None) -> None:
        col = "name" if col == "#0" else col
        if order is None:
            order = _SORT_DEFAULT[col] if col != self._sort[0] else ("asc" if self._sort[1] == "desc" else "desc")
        self._sort = (col, order)
        self._relayout()

    def _select_key(self, key: Optional[str]) -> None:
        if key:
            self.tree.selection_set(key)
            self.tree.focus(key)
            self.tree.see(key)

    def _keys(self) -> Tuple[str, ...]:
        """The rows the listing shows (none while it says 'Opening…')."""
        return self.tree.get_children("") if self._listing is not None else ()

    def _arrow(self, step: int) -> Optional[str]:
        """Up/Down with nothing selected (after every navigation, or Esc) select the row the keyboard is on, else
        the first (last) one; Tk's own key navigation would do nothing, or skip that row (BUX-M1)."""
        kids = self._keys()
        if self.tree.selection() or not kids:
            return None
        focus = self.tree.focus()
        self._select_key(focus if focus in kids else kids[0] if step > 0 else kids[-1])
        return "break"

    def _use_here(self) -> None:
        """The folder shown is the target again: nothing selected, nothing waiting to be selected (BR-18)."""
        self._select = None
        if self.tree.selection():
            self.tree.selection_set(())
        self._update_target()

    def _ctrl_click(self, e: Any) -> Optional[str]:
        """Ctrl+click (Cmd+click) on the selected row unselects it, as in a file manager (BR-18)."""
        if self.tree.identify_row(e.y) in self.tree.selection():
            self._use_here()
            return "break"
        return None

    def _tree_motion(self, e: Any) -> None:
        """The pointer on a row drawn shortened: its whole name and badge as a tooltip (BR-26)."""
        row = self.tree.identify_row(e.y)
        self._list_tip.point(row, self._cut.get(row, ""))

    def _released(self, e: Any) -> None:
        """A click on the listing's empty space chooses the folder shown, as in a file manager (BR-18); a column
        that was dragged wider or narrower fits its texts again (BR-26)."""
        region = self.tree.identify_region(e.x, e.y)
        if region == "nothing":
            self._use_here()
        elif region in ("heading", "separator"):
            self._later("fit", 50, self._refit)

    def _focus_index(self, kids: Sequence[str]) -> int:
        focus = self.tree.focus()
        return kids.index(focus) if focus in kids else 0

    def _page(self) -> int:
        top, bottom = self.tree.yview()
        return max(1, int((bottom - top) * len(self.tree.get_children(""))) - 1)

    def _visible_keys(self) -> List[str]:
        kids = self.tree.get_children("")
        top, bottom = self.tree.yview()
        return list(kids[int(top * len(kids)):int(bottom * len(kids)) + 1])

    def _yscroll(self, first: str, last: str) -> None:
        self._vsb.set(first, last)
        self._later("visible", 150, self._visible_work)

    def _columns(self) -> None:
        """Only the tiles chooser shows Tiles; a listing narrower than px(560) drops Modified (§2.2.2). The names and
        badges are fitted to the columns' new widths."""
        cols = [c for c in ("tiles", "inside", "modified") if not (c == "tiles" and self.mode != "input")]
        if self._list_frame.winfo_width() < self.px(560):
            cols.remove("modified")
        if tuple(self.tree.cget("displaycolumns")) != tuple(cols):
            self.tree.configure(displaycolumns=cols)
        self._later("fit", 50, self._refit)

    def _type_ahead(self, e: Any) -> Optional[str]:
        """Printable keys select the first row starting with what was typed in the last second; '/', '~' and
        '\\' (and a drive letter + ':' on Windows) start path editing."""
        ch = e.char
        if not ch or not ch.isprintable() or e.state & _MODS or (ch == " " and not self._ta_buf):
            return None
        if not self._ta_buf and ch in "/~\\":
            self.edit_path(ch)
            return "break"
        if WINDOWS and ch == ":" and len(self._ta_buf) == 1 and self._ta_buf.isalpha():
            self.edit_path(self._ta_buf + ":")
            return "break"
        self._ta_buf += ch
        self._later("typeahead", 1000, lambda: (setattr(self, "_ta_buf", ""), self._note_set("")))
        want = self._ta_buf.casefold()
        self._select_key(next((k for k in self._keys() if self._rows[k].name.casefold().startswith(want)), None))
        self._note_set(self._ta_buf)
        return "break"

    # ------------------------------------------------------------------------------------- probes, roll-ups, stat
    def _probe_children(self) -> None:
        """One probe per child folder, in display order: the first screenful at priority 4, the rest at 6; none
        where auto_enter() gives a reason (§2.2.7). Scrolling re-prioritises (_visible_work)."""
        fs, todo = self.ctx.fs, []
        order = {k: i for i, k in enumerate(self._layout)}
        for r in self._rows.values():
            if (r.kind != "dir" or r.auto or r.summary is not None or r.error is not None or r.probing
                    or (r.link and r.target is None)):     # a link is probed where it leads, once that is known
                continue
            path = self._row_path(r)
            s = fs.cached_summary(path, max_age=self._fresh) if self._use_cache else None
            if s is not None:
                self._apply_probe(r, s, None)
            else:
                todo.append(r)
        todo.sort(key=lambda r: order.get(r.key, len(order)))
        self._probe_queue = [(r, 4 if i < 60 else 6) for i, r in enumerate(todo)]
        self._probe_next()

    def _probe_next(self) -> None:
        """Submit probes 300 per event-loop turn, so a folder of 20,000 subfolders never blocks the UI."""
        for r, priority in self._probe_queue[:300]:
            if not r.probing and r.summary is None and r.error is None:
                self._probe(r, priority)
        del self._probe_queue[:300]
        if self._probe_queue:
            self._later("probe", 1, self._probe_next)

    def _probe(self, r: _Row, priority: int) -> None:
        rid = self.ctx.fs.probe(self._row_path(r), ticket=self._ticket, owner="browser", priority=priority)
        r.probing = priority
        self._calls[rid] = (self._ticket, lambda e, r=r: self._on_probe(r, e))

    def _on_probe(self, r: _Row, e: FsDone) -> None:
        if self._rows.get(r.key) is not r:
            return
        r.probing = 0
        if r.kind == "other":                            # a followed link: a folder, unless it led to a file or nowhere
            if e.error is not None and e.error.code in ("not_a_dir", "not_found"):
                self._later("empty", 1, self._empty_state)    # (it may have been the last link still followed)
                return
            r.kind = "dir"
            self._later("relayout", 1, self._relayout)
        self._apply_probe(r, e.result, e.error)

    def _apply_probe(self, r: _Row, s: Optional[DirSummary], err: Optional[FsError]) -> None:
        r.summary, r.error = s, err
        r.stalled = self._row_path(r) if err is not None and err.code in _STALL else None
        if s is not None and not s.groups and s.subdirs and r.roll is None:
            self._start_rollup(r)
        self._refresh_row(r)

    def _start_rollup(self, r: _Row) -> None:
        """Probe up to 40 grandchildren (200 per view) to count the slide folders one level down (§2.2.7 step 5)."""
        base, fs = self._row_path(r), self.ctx.fs
        r.roll = [0, 0, 0, len(r.summary.subdirs) > ROLLUP_PER_CHILD, 0, None, [], 0]
        for name in r.summary.subdirs[:ROLLUP_PER_CHILD]:
            path = os.path.join(base, name)
            reason = fs.auto_enter(path)
            if reason:
                r.roll[7] += reason in _UNCHECKED
                continue
            if self._roll_budget <= 0:
                r.roll[3] = True
                break
            self._roll_budget -= 1
            s = fs.cached_summary(path, max_age=self._fresh) if self._use_cache else None
            if s is not None:
                self._roll_add(r, s)
            else:
                self._roll_probe(r, path)

    def _roll_probe(self, r: _Row, path: str) -> None:
        r.roll_errors.pop(path, None)
        r.roll[4] += 1
        rid = self.ctx.fs.probe(path, ticket=self._ticket, owner="browser", priority=9)
        self._calls[rid] = (self._ticket, lambda e: self._on_roll(r, path, e))

    def _on_roll(self, r: _Row, path: str, e: FsDone) -> None:
        if self._rows.get(r.key) is r and r.roll is not None:
            r.roll[4] -= 1
            if e.error is not None:
                r.roll_errors[path] = e.error
                if e.error.code in _STALL:
                    r.roll[6].append(path)                # asked again once it answers (_answers_again)
            elif e.result is not None:
                self._roll_add(r, e.result)
            self._refresh_row(r)

    @staticmethod
    def _roll_add(r: _Row, s: DirSummary) -> None:
        groups = _slide_groups(s)
        r.roll[0] += 1
        if groups:
            r.roll[1] += 1
            r.roll[2] += sum(len(g.names) for g in groups.values())
            first = next(iter(groups.values()))
            if r.roll[5] is None or natural_key(s.path) < natural_key(r.roll[5][0]):
                r.roll[5] = (s.path, first.prefix, first.coords)   # the preview maps the first slide inside

    def _follow_links(self) -> None:
        """Links that may lead to folders (and Windows junctions) are followed before anything touches them (UD3);
        300 per event-loop turn, like the probes."""
        todo = [r for r in self._rows.values()
                if r.link and r.kind in ("dir", "other") and r.target is None and not r.probing and r.error is None]
        for r in todo[:300]:
            r.probing = 4
            self._readlink(r, self._child(r.name), 0)
        if len(todo) > 300:
            self._later("links", 1, self._follow_links)

    def _readlink(self, r: _Row, path: str, hops: int) -> None:
        # The listing has already resolved its own path, so the row starts at that parent.
        state = r.resolving or (os.path.dirname(path), (_base(path),), hops)

        def watching(_rid: int, _where: str, pending: _LinkState) -> None:
            r.resolving = pending

        def done(folder: str, err: Optional[FsError]) -> None:
            r.probing = 0
            if err is None:
                r.resolving, r.error, r.stalled = None, None, None
                self._resolved(r, folder)
            elif err.code in _STALL or r.kind == "dir":
                path = os.path.join(r.resolving[0], r.resolving[1][0])
                r.kind, r.error, r.stalled = "dir", err, path if err.code in _STALL else None
                self._later("relayout", 1, self._relayout)
                self._refresh_row(r)
            else:
                self._later("empty", 1, self._empty_state)
        self._resolve_links(state, done, watching, lambda: self._rows.get(r.key) is r, self._ticket)

    def _resolved(self, r: _Row, folder: str) -> None:
        """A link row leads to `folder`: from now on it is probed, badged, chosen and opened as that folder."""
        r.target, r.auto = folder, self.ctx.fs.auto_enter(folder)
        if not r.auto:
            self._probe(r, 4)                             # it shows as a folder once the probe says it is one
        elif r.kind != "dir":
            r.kind = "dir"
            self._later("relayout", 1, self._relayout)

    def _plain(self, r: _Row) -> bool:
        """The stat pass asks about plain entries of this folder's lane only: never a link or a mount point,
        whose lstat would go to (and wait for) another volume (UD3)."""
        fs = self.ctx.fs
        return r.kind != "tiles" and not r.link and fs.lane_key(self._child(r.name)) == fs.lane_key(self._path)

    def _stat(self, rows: Sequence[_Row], priority: int) -> None:
        """Sizes and dates (POSIX: one lstat each) of rows not asked about yet, 200 names per request."""
        rows = [r for r in rows if r.name not in self._statted]
        self._statted.update(r.name for r in rows)
        names = [r.name for r in rows if self._plain(r)]
        for i in range(0, len(names), 200):
            rid = self.ctx.fs.stat_names(self._path, names[i:i + 200], ticket=self._ticket, owner="browser",
                                         priority=priority)
            self._stat_batches[rid] = names[i:i + 200]
            self._calls[rid] = (self._ticket, self._on_stat)
            if priority == 6:
                self._dating.add(rid)

    def _date_rows(self) -> None:
        """Sorted by Modified: every shown row needs its date, not only the visible ones (BR-10); 1,000 per
        event-loop turn."""
        if WINDOWS or self._sort[0] != "modified":         # Windows: the dates came with the listing
            return
        todo = [self._rows[k] for k in self._layout if self._rows[k].name not in self._statted]
        self._stat(todo[:1000], 6)
        if len(todo) > 1000:
            self._later("dates", 1, self._date_rows)

    def _visible_work(self) -> None:
        """Visible rows first: re-prioritise their probes (FsService merges duplicates) and stat their names."""
        if self._listing is None:
            return
        visible = [self._rows[k] for k in self._visible_keys() if k in self._rows]
        for r in visible:
            if r.kind == "dir" and r.probing > 4 and r.summary is None:
                self._probe(r, 4)
        if not WINDOWS:                                   # Windows: sizes and dates came with the listing
            self._stat(visible, 8)

    def _on_stat(self, e: FsDone) -> None:
        self._dating.discard(e.req_id)
        result = e.result or {}
        for name in self._stat_batches.pop(e.req_id, []):
            r = self._rows.get(self._names.get(name, ""))
            if r is not None:
                r.date_error = e.error or (FsError("io", "Couldn't read its date.") if name not in result else None)
                if name in result:
                    r.size, r.mtime = result[name][:2]
                self._refresh_row(r)
        self._later("resort", 100, self._relayout)

    # ------------------------------------------------------------------------------------- target, strip, preview
    def _target_row(self) -> Optional[_Row]:
        sel = self.tree.selection()
        r = self._rows.get(sel[0]) if sel else None
        return r if r is not None and r.kind == "dir" else None

    def _label(self, path: str) -> str:
        """A drive or share root is named by its place label; long names are shortened in the middle."""
        place = next((p for p in self._places if p.section in ("computer", "network", "cloud")
                      and same_path(p.path, path)), None)
        return _middle(place.label if place else _base(path), 40, self.G["ell"])

    def _name_of(self, path: str) -> str:
        """The target's name in the button, strip and preview: a selected link by its own name (the row's)."""
        r = self._target_row()
        return _middle(r.name, 40, self.G["ell"]) if r is not None and r.link else self._label(path)

    def _title_of(self, path: str) -> str:
        """The preview's title: the target's name on one line, shortened in the middle like its row (BR-25)."""
        return self._rulers[True].fit(self.T(self._name_of(path)), self.px(200), self.G["ell"], middle=True)

    def q(self, s: str) -> str:
        return f"{self.G['lq']}{s}{self.G['rq']}"

    def _target_facts(self) -> Tuple[Optional[DirSummary], Optional[Rollup], Optional[FsError], Optional[str]]:
        r = self._target_row()
        if r is not None:
            return r.summary, self._rollup(r), self._row_error(r), r.auto
        if self.target() != self._path:                  # a pending selection: known once its folder answers
            return None, None, self._list_error or self._no_answer, self.ctx.fs.auto_enter(self.target())
        return self._summary, self._folder_rollup(), self._list_error or self._no_answer, None

    def _folder_rollup(self) -> Optional[Rollup]:
        """The current folder's slide folders within 2 levels: its slide subfolders plus what their roll-ups found
        (BR-11), without hidden folders and links (the search follows neither); None while nothing is found yet
        and probes are still out."""
        if self._summary is None:
            return None
        probed = slides = tiles = 0
        pending = capped = False
        for r in self._rows.values():
            if r.kind != "dir" or r.hidden:
                continue
            if r.auto or r.error is not None:
                capped = capped or (not r.link and (r.auto in _UNCHECKED or r.error is not None))
                continue
            # a link still being looked at keeps the strip waiting too: it may lead to slide folders (_linked)
            pending = pending or r.summary is None or bool(r.roll and r.roll[4])
            if r.summary is None or r.link:
                continue
            probed += 1
            if _stitchable(r.summary):
                slides, tiles = slides + 1, tiles + self._tile_count(r)
            if r.roll is not None:
                slides, tiles = slides + r.roll[1], tiles + r.roll[2]
                capped = capped or bool(r.roll[3] or r.roll[7] or r.roll_errors)
        return Rollup(probed, slides, tiles, pending or capped) if slides or not pending else None

    def _linked(self) -> str:
        """The first link here that leads to slide folders: the search doesn't follow links (§3.5), so the strip
        says where they are."""
        names = [r.name for r in self._rows.values() if r.link and r.kind == "dir" and not r.hidden
                 and (_stitchable(r.summary) or (r.roll is not None and r.roll[1]))]
        return _middle(min(names, key=natural_key), 40, self.G["ell"]) if names else ""

    def _unchecked(self) -> int:
        """Subfolders here whose contents can't be known yet (BR-12)."""
        return sum(1 for r in self._rows.values() if r.kind == "dir" and not r.hidden and self._unknown(r))

    def _failed_folders(self) -> int:
        """Failed descendants of the target; links outside it are not part of the search."""
        row = self._target_row()
        if row is not None:
            return len(row.roll_errors)
        return sum(1 if r.error is not None else len(r.roll_errors) for r in self._rows.values()
                   if r.kind == "dir" and not r.hidden and not r.link)

    def _update_target(self) -> None:
        """Primary button label, target strip and preview for the current target (§2.2.3)."""
        if not self.winfo_exists():
            return
        path = self.target()
        if path != self._target_seen:                     # a save check still out was for the old target (BR-7)
            self._new_intent()
            self._target_seen, self._strip_msg, self._writing, self._edit_error = path, None, None, False
            if self.mode != "input":
                self._later("facts", 200, self._output_facts)
        name = self._name_of(path)
        self._fit_primary()
        summary, roll, err, auto = self._target_facts()
        action = None
        if self._strip_msg is not None:
            style, text, action = self._strip_msg
        elif self._editing:                               # what the keys do in path editing (BR-29)
            style, text = "Info", ("Enter opens the folder you typed, Ctrl+Enter uses it. Tab completes a name; "
                                   "Down lists the folders that match.")
        else:
            fs, idx = self.ctx.fs, self.ctx.output_index()
            out_dir = idx.out_dir if idx is not None else self.ctx.settings.get("paths.output")
            here = path == self._path and self._target_row() is None
            row = self._target_row()
            unchecked = self._unchecked() if here else row.roll[7] if row is not None and row.roll else 0
            style, text = describe_target(
                self.mode, name, summary, roll, error=err, auto=auto, lane_class=fs.lane_class(path),
                save_folder=self.mode == "input" and bool(out_dir) and same_path(path, out_dir),
                free=self._free.get(path), planned=self.ctx.planned_output_keys() if self.mode == "output" else frozenset(),
                place=self._place_name(path), unchecked=unchecked,
                linked=self._linked() if here else "", free_error=self._free_errors.get(path), failed=self._failed_folders())
            if row is not None and row.link and row.target is None and err is not None and err.code in _STALL:
                style = "Warn"
                text = f"{self._said(err, row.stalled or path)} Press F5 to try again, or choose another folder."
            if self.mode == "output" and _stitchable(summary) and err is None:
                action = (f"Create {self.q('stitched')} here", lambda p=path, s=summary: self._create_stitched(p, s))
            if self.mode != "input" and path in self._free_errors and err is None:
                action = ("Try again", self.refresh)
        icon = {"Warn": "warn", "Error": "error", "Info": "info"}.get(style)
        if icon is None:
            icon = "folder_tiles" if _stitchable(summary) else "folder_slides" if self.mode == "input" else "ok"
        self._strip.configure(style=f"{style}.TFrame")
        self._strip_icon.configure(style=f"{style}.Banner.TLabel", image=self.ctx.icons.get(icon))
        self._strip_text.configure(style=f"{style}.Banner.TLabel", text=self.T(text))
        if action:
            self._strip_btn.configure(style=f"{style}.Banner.TButton", text=self.T(action[0]), command=action[1])
            self._strip_btn.grid(row=0, column=2, sticky="e", padx=(self.px(8), 0))
        else:
            self._focus_list(hiding=self._strip_btn)
            self._strip_btn.grid_remove()
        self._wrap_strip()
        self._picker_link()
        if self.mode != "input" and self._listing is not None and "facts" not in self._jobs \
                and path not in self._free and path not in self._free_pending and path not in self._free_errors:
            self._later("facts", 200, self._output_facts)  # an initial lookup may outlast the first facts timer
        if "preview" not in self._jobs:                   # throttled: streaming results never hold it back (BR-9)
            self._later("preview", 120, self._update_preview)

    def _strip_key(self, e: Any) -> Optional[str]:
        """Alt + the first letter of the strip's action runs it, wherever the keyboard is (BR-13): in path editing
        Tab completes names, so [Go there] after a mistyped path can't be reached by Tab."""
        text = str(self._strip_btn.cget("text"))
        if self._strip_btn.winfo_manager() and text[:1].casefold() == e.keysym.casefold():
            self._strip_btn.invoke()
            return "break"
        return None

    def _fit_primary(self) -> None:
        """Leave Cancel and the system picker readable; shorten only the target's name when space is tight."""
        ruler, px = self._rulers[True], self.px
        prefix, name = self.T(_PRIMARY[self.mode]), self.T(self._name_of(self.target()))
        width = self._primary.master.winfo_width()
        if width > 1:
            picker = self._picker.winfo_reqwidth() + px(8) if self._picker.winfo_manager() else 0
            room = width - px(60) - self._cancel.winfo_reqwidth() - picker - ruler.width(prefix + self.q(""))
            name = ruler.fit(name, room, self.G["ell"], middle=True)
        self._primary.configure(text=prefix + self.q(name))

    def _wrap_strip(self) -> None:
        """Wrap the strip text to the room left by its icon and (when shown) its action button."""
        button = self._strip_btn.winfo_reqwidth() + self.px(8) if self._strip_btn.winfo_manager() else 0
        used = self._strip_icon.winfo_reqwidth() + button + self.px(32)
        self._strip_text.configure(wraplength=max(self.px(200), self._strip.winfo_width() - used))

    def _update_preview(self) -> None:
        if not self._preview_on or not self.winfo_exists():
            return
        summary, roll, err, _auto = self._target_facts()
        path, fs = self.target(), self.ctx.fs
        if err is not None:                               # what went wrong, like the strip; no map (BR-27)
            self._facts.show_folder(_unreadable(path, FsError(err.code, self._said(err, path))), None, None,
                                    title=self._title_of(path))
            return
        if self.mode != "input":                          # free space (and the stitched images it holds)
            sizes = self._out_sizes.get(path, {}) if self.mode == "output" else {}
            if path == self._path and self.mode == "output":
                sizes = {r.name: r.size for r in self._rows.values() if r.kind == "file" and r.size is not None}
            outs = {norm_key(n): OutputInfo(n, sz, 0) for n, sz in sizes.items() if classify_file(n) == "output"}
            idx = OutputIndex(path, True, frozenset(), outs, {}, (), self._free.get(path), fs.lane_key(path),
                              fs.lane_class(path), fs.mounts.is_system_volume(path))
        else:
            idx = self.ctx.output_index()
        self._facts.show_folder(summary, roll, idx, title=self._title_of(path))
        if summary is not None and not summary.groups:   # a parent folder: map the first slide folder inside
            r = self._target_row()
            if r is not None:
                sample = r.roll[5] if r.roll is not None else None
            else:
                kids = [x for x in self._rows.values() if x.kind == "dir" and not x.hidden and _slide_groups(x.summary)]
                first = min(kids, key=lambda x: natural_key(x.name), default=None)
                g = next(iter(_slide_groups(first.summary).values())) if first is not None else None
                sample = (first.name, g.prefix, g.coords) if g is not None else None
            if sample is not None:                        # (show_folder hid the map: this folder has no tiles)
                self._facts.map.show(sample[2], None, None)
                name = _middle(sample[1], 24, self.G["ell"])     # (one line in the caption, on any font)
                self._facts.caption.configure(text=self.T(f"Tile layout of {name} (from file names)"))
                self._facts.drawing.grid()

    def _output_facts(self) -> None:
        """Save and temporary-files choosers: free space of the target, and the sizes of the stitched images it
        holds (preview)."""
        path, fs = self.target(), self.ctx.fs
        row = self._target_row()
        if self._listing is None or (row is not None and row.link and row.target is None) \
                or (row is None and path != self._path):
            return                                       # a destination must be resolved before any following I/O
        ticket = self._capacity_ticket

        def free(e: FsDone) -> None:
            if ticket != self._capacity_ticket:
                return
            self._free_pending.discard(path)
            if e.error is None:
                self._free[path] = e.result[2]
            else:
                self._free_errors[path] = e.error
            self._update_target()

        def sizes(e: FsDone) -> None:
            if e.error is None:
                self._out_sizes[path] = {name: facts[0] for name, facts in e.result.items()}
                self._later("preview", 50, self._update_preview)
        if path not in self._free and path not in self._free_pending and path not in self._free_errors:
            self._free_pending.add(path)
            self._calls[fs.disk_usage(path, ticket=ticket, owner="browser", priority=3)] = (None, free)
        summary = self._target_facts()[0]
        if self.mode == "output" and path != self._path and summary is not None and summary.outputs \
                and path not in self._out_sizes:
            self._calls[fs.stat_names(path, summary.outputs[:8], ticket=0, owner="browser", priority=3)] = (None, sizes)

    def _new_intent(self) -> None:
        """Editing or navigating makes any earlier write acceptance obsolete, before its destination resolves."""
        self._intent_revision += 1
        if self._writing is not None:
            self._writing, self._strip_msg = None, None

    def _probed(self, token: int, path: str, e: FsDone) -> None:
        """A write check answered: only the latest one, for the folder still targeted, may close the dialog."""
        if self._writing != (token, path, self._intent_revision) or not same_path(path, self.target()):
            return                                        # the user went on to another folder (CT-4, BR-7)
        self._writing = None
        if e.error is None:
            self._finish(path)
            return
        if e.error.code in _STALL:                        # nothing was refused: the check can be asked again
            self._strip_msg = ("Warn", f"Couldn't check that you can {_WRITE[self.mode]} in {self._name_of(path)}: "
                                       f"{self._said(e.error, path)}",
                               ("Try again", lambda: (self.ctx.fs.retry(path), self.choose())))
        else:
            self._strip_msg = ("Error", f"Can't {_WRITE[self.mode]} here: {e.error.message}", None)
        self._update_target()
        if self._strip_msg[2] is not None and str(self.tk.call("focus")) in (str(self._primary), str(self.tree)):
            self._strip_btn.focus_set()                   # Enter or Space tries again (BR-13)

    def _create_stitched(self, folder: str, summary: DirSummary) -> None:
        """Output mode, a folder of tiles: make (or pick) its 'stitched' subfolder and select it."""
        if "stitched" in summary.subdirs:
            self.navigate(folder, select="stitched")
            return
        path, where = os.path.join(folder, "stitched"), self._path
        self._call(path, "mkdir", partial(os.makedirs, path, exist_ok=True),
                   lambda e: self._made(where, folder, "stitched", e), user=True)

    def _make_folder(self, name: Optional[str]) -> None:
        safe_grab(self)
        if name:
            path = self._child(name)
            folder = self._path
            self._call(path, "mkdir", partial(os.mkdir, path), lambda e: self._made(folder, folder, name, e),
                       user=True)

    def _made(self, where: str, folder: str, name: str, e: FsDone) -> None:
        """`name` was made in `folder` while the browser showed `where`: show it selected, unless the user has
        gone elsewhere since (CT-4)."""
        if e.error is None:
            self.ctx.fs.invalidate(folder)
        if not same_path(where, self._path):
            return
        if e.error is not None and e.error.code in _STALL:   # it may have been made all the same
            self._strip_msg = ("Warn", f"Couldn't create {self.q(name)}: "
                                       f"{self._said(e.error, folder)} Press F5 to see whether it was made.", None)
        elif e.error is not None:
            self._strip_msg = ("Error", f"Can't create {self.q(name)}: {e.error.message}", None)
        else:
            self.navigate(folder, push=not same_path(folder, self._path), select=name)
            return
        self._update_target()

    def _finish(self, path: Optional[str]) -> None:
        """Remember the dialog's state, close it, then report the choice (None = cancelled)."""
        cb, self._on_done = self._on_done, None
        if cb is None:
            return
        s, scale = self.ctx.settings, self.ctx.theme.scale
        if self.winfo_width() > 1:
            s.set("browser.geometry", f"{round(self.winfo_width() / scale)}x{round(self.winfo_height() / scale)}")
            try:
                s.set("browser.sash", round(self._pw.sashpos(0) / scale))
            except tk.TclError:
                pass
        for key, var in (("only_tiles", self._v_tiles), ("show_hidden", self._v_hidden), ("show_all_files", self._v_all)):
            if key != "only_tiles" or self.mode == "input":    # (the other choosers don't offer it: BR-2)
                s.set("browser." + key, bool(var.get()))
        s.set("browser.sort", list(self._sort))
        if path is not None:
            s.set(f"paths.browser_last.{self.mode}", self._path)
        s.save_async()
        self.destroy()
        cb(path)

    # ------------------------------------------------------------------------------------- states (§2.2.8)
    def _show_bar(self) -> None:
        if self._loading:
            # place() works in the frame's interior, which excludes its padding
            self._bar.place(in_=self._toolbar, x=-self.px(8), y=self._bar_pad, relx=0, rely=1.0, relwidth=1.0,
                            width=2 * self.px(8), anchor="sw")
            self._bar.lift(self._toolbar)
            self._bar.start(15)

    def _hide_bar(self) -> None:
        self._cancel_job("bar")
        self._bar.stop()
        self._bar.place_forget()

    def _place_name(self, path: str) -> str:
        p = containing_place(path, self._places)
        return p.label if p is not None else (_base(self.ctx.fs.lane_key(path)) or _base(path))

    def _said(self, err: FsError, path: str) -> str:
        """An error as one sentence; a place that doesn't answer is named once (UD1)."""
        return f"{self._place_name(path)} isn't answering." if err.code == "unresponsive" else err.message

    def _overlay(self, kind: str, icon: Optional[str], title: str, text: str = "",
                 buttons: Sequence[Tuple[str, Callable[[], None]]] = ()) -> None:
        """A message card place()d over the listing (loading problems, empty folders, the system picker). The
        keyboard on the card it replaces moves to this card's first button (BR-14)."""
        px, ov = self.px, self._ov
        had_focus = str(self.tk.call("focus")).startswith(str(ov) + ".")
        for w in ov.winfo_children():
            w.destroy()
        self._ov_kind = kind
        if icon:
            ttk.Label(ov, image=self.ctx.icons.get(icon), style="Surface.TLabel").grid(
                row=0, column=0, rowspan=2, sticky="nw", padx=(0, px(10)))
        self._ov_title = ttk.Label(ov, text=self.T(title), style="Surface.TLabel", font="YfsBold",
                                   wraplength=px(420), justify="left")
        self._ov_title.grid(row=0, column=1, sticky="w")
        if text:
            ttk.Label(ov, text=self.T(text), style="Surface.Muted.TLabel", wraplength=px(420),
                      justify="left").grid(row=1, column=1, sticky="w", pady=(px(4), 0))
        made = []
        if buttons:
            row = ttk.Frame(ov, style="Surface.TFrame")
            row.grid(row=2, column=1, sticky="w", pady=(px(12), 0))
            for label, cmd in buttons:
                made.append(ttk.Button(row, text=self.T(label), command=cmd))
                made[-1].pack(side="left", padx=(0, px(8)))
        # below the headings and the first row: the "Opening 'X'" row stays readable over a waiting card
        ov.place(in_=self._list_frame, relx=0.5, y=2 * self.ctx.theme.rowheight + px(12), anchor="n")
        ov.lift()
        if had_focus:
            (made[0] if made else self.tree).focus_set()

    def _overlay_hide(self) -> None:
        """Hide the card; the keyboard on one of its buttons goes to the listing (BR-14)."""
        self._focus_list(hiding=self._ov)
        self._ov_kind = ""
        self._ov.place_forget()

    def _on_slow(self, e: FsSlow) -> None:
        if e.owner != "browser" or e.req_id != self._wait_rid or not self._loading:
            return
        name = self._place_name(self._wait_path)
        if e.kind == "list" and self._listing is not None:    # a cached view is shown: just say it may be old
            self._note_set(f"{name} isn't answering; this list may be out of date.")
        elif e.past_deadline:
            self._unanswered(FsError("timeout", f"{name} didn't answer."))
        elif self._ov_kind != "unresponsive":
            self._waiting_tick()

    def _unanswered(self, err: FsError) -> None:
        """The loading request is past its deadline (a listing still waits for its late answer) or failed with
        `err`: the not-answering card (§2.2.8, UD1). Its lane answering again asks once more by itself."""
        path = self._wait_path
        lane = self.ctx.fs.lane_key(path)
        self._stall = (lane, self._wait_again)
        if same_path(path, self._path) or self._listing is None:
            self._no_answer = err
        if not self._keep_waiting and self._ov_kind != "unresponsive":
            since = self._unresponsive_since.get(lane) if err.code == "unresponsive" else None
            when = f" (since {time.strftime('%H:%M', time.localtime(since))})" if since else ""
            self._cancel_job("slow")
            self._overlay("unresponsive", "warn", f"{self._place_name(path)} isn't answering{when}.",
                          f"Nothing else in the app is affected. {self._drives()} sometimes need a minute to wake up.",
                          [("Keep waiting", self._keep), ("Try again", self._wait_again), ("Go back", self._wait_back)])
        self._update_target()

    def _waiting_tick(self) -> None:
        if not self._loading or self._ov_kind == "unresponsive" or self._keep_waiting:
            return
        name, secs = self._place_name(self._wait_path), int(time.monotonic() - self._t0)
        title = f"Waiting for {name} to answer{ELL} {secs} s"
        if self._ov_kind == "wait":
            self._ov_title.configure(text=self.T(title))
        else:
            text = f"{self._drives()} can take a moment to wake up."
            if tcc_protected(self._wait_path):
                text += " macOS may be asking for permission in another window."
            self._overlay("wait", "wait", title, text, [("Stop waiting", self._wait_back)])
        self._later("slow", 1000, self._waiting_tick)

    def _drives(self) -> str:
        """What may be waking up: a local disk (a USB disk spinning up) is not a network drive."""
        return "Disks" if self.ctx.fs.lane_class(self._wait_path) == LOCAL else "Network drives"

    def _keep(self) -> None:
        self._keep_waiting = True
        self._overlay_hide()

    def _show_error(self, err: FsError, path: Optional[str] = None) -> None:
        """§2.2.8 texts for a listing that failed."""
        path = path or self._path
        name, q = self._place_name(path), self.q(self._label(path))
        back = ("Dismiss", self._dismiss_place_error) if not same_path(path, self._path) and self._listing is not None \
            else ("Go back", self._go_back)
        again = ("Try again", self._retry)
        if err.code == "no_access":
            text = "Check System Settings → Privacy & Security → Files and Folders." if MACOS else ""
            self._overlay("error", "error", f"You don't have permission to open {q}.", text, [back])
        elif err.code == "not_found":
            recent = [p for p in self._places if p.section == "recent" and same_path(p.path, path)]
            buttons = [back] + ([("Remove from recent", lambda p=path: (self._forget_recent(p), back[1]()))]
                                if recent else [])
            self._overlay("error", "error", f"{q} doesn't exist any more.", "", buttons)
        elif err.code == "disconnected":
            self._overlay("error", "error", f"{self.ctx.fs.lane_key(self._path)} is disconnected.",
                          "Open it once in File Explorer to reconnect, then Try again.", [again, back])
        elif err.code == "unavailable":
            self._overlay("error", "error", "The network location isn't available.", "", [again, back])
        elif err.code in _STALL:
            lane = self.ctx.fs.lane_key(self._path)
            since = self._unresponsive_since.get(lane)
            when = f" (since {time.strftime('%H:%M', time.localtime(since))})" if since else ""
            self._overlay("error", "warn", f"{name} isn't answering{when}.",
                          "Nothing else in the app is affected, and this folder opens as soon as it answers.",
                          [again, back])
            self._stall = (lane, self._retry)
        elif err.code == "not_a_dir":
            self._overlay("error", "error", f"{q} is a file, not a folder.", "", [back])
        else:
            self._overlay("error", "error", f"Can't open {q}.", err.message, [again, back])

    def _empty_state(self) -> None:
        if self._ov_kind in ("error", "unresponsive", "picker") or self._listing is None:
            return
        if self.tree.get_children(""):
            if self._ov_kind == "empty":
                self._overlay_hide()
            return
        f = self._filter_var.get().strip()
        shown = [r for r in self._rows.values() if not r.hidden]
        if not self._listing.entries:
            text = "This folder is empty."
        elif f:
            text = f"Nothing here matches {self.q(f)}."
        elif not shown:
            text = "Only hidden items here (Ctrl+H shows them)."
        elif any(r.link and r.probing for r in shown):   # links still being followed may lead to folders
            if self._ov_kind == "empty":
                self._overlay_hide()
            return
        elif all(r.kind == "other" for r in shown):
            text = f"No folders here. Tiles usually sit in a folder named like {self.q('<slide>_Scan1')}."
        else:                                             # only "Only folders with tiles" hides folders
            hidden = sum(1 for r in shown if r.kind == "dir")
            text = (f"No folders with tiles here. {self.q('Only folders with tiles')} is hiding "
                    f"{_n(hidden, 'other folder')}.")
        self._overlay("empty", "info", text)

    def _note_set(self, text: str) -> None:
        self._notes[self._note] = self.T(text)
        self._fit_notes()

    def _others_set(self, text: str) -> None:
        self._notes[self._others] = self.T(text)
        self._fit_notes()

    def _fit_notes(self) -> None:
        """The footer's notes in the room the options leave, the momentary one first: a note that doesn't fit ends
        in an ellipsis, with its whole text as a tooltip, never cut off mid-word (BR-26)."""
        ruler, ell = self._small, self.G["ell"]
        note, others = self._notes[self._note], self._notes[self._others]
        room = self._note.master.winfo_width() - self._others.winfo_x() - self.px(8)
        if room > 1:                                      # (once laid out)
            note = ruler.fit(note, room, ell)
            others = ruler.fit(others, max(2, room - (ruler.width(note) + self.px(14) if note else 0)), ell)
        self._note.configure(text=note)
        self._others.configure(text=others)

    # ------------------------------------------------------------------------------------- crumbs and path editing
    def _crumbs(self) -> None:
        """Breadcrumbs from the containing place (string work only); the … menu holds the ancestors above it."""
        for sep, button, _l, _p in self._crumb_ws:
            if sep is not None:
                sep.destroy()
            button.destroy()
        place = containing_place(self._path, self._places) if self._path else None
        crumbs, self._crumb_hidden = breadcrumb_parts(self._path, place) if self._path else ([], [])
        self._crumb_ws = []
        for i, (label, path) in enumerate(crumbs):
            sep = ttk.Label(self._crumb_box, text=self.G["sep"], style="Surface.Muted.TLabel") if i else None
            here = i == len(crumbs) - 1                   # the folder shown: a click makes it the target (BR-18)
            b = ttk.Button(self._crumb_box, text=self.T(label), takefocus=0,
                           style="CrumbCurrent.TButton" if here else "Crumb.TButton",
                           command=self._use_here if here else lambda p=path: self.navigate(p))
            if i == 0 and place is not None:
                b.configure(image=self.ctx.icons.get(place.icon), compound="left")
            self._crumb_ws.append((sep, b, label, path))
        self._fit_crumbs()

    def _fit_crumbs(self) -> None:
        """Collapse the oldest crumbs into the … menu until the bar fits; the current folder
        always stays visible, shortened in the middle when even it doesn't fit (BR-26)."""
        ws = self._crumb_ws
        if not ws:
            return
        last, full = ws[-1][1], self.T(ws[-1][2])
        last.configure(text=full)
        avail = self._pathbar.winfo_width() - self.px(16)
        widths = [(s.winfo_reqwidth() if s is not None else 0) + b.winfo_reqwidth() for s, b, _l, _p in ws]
        total, hidden, more = sum(widths), set(), self._more.winfo_reqwidth()
        for i in list(range(1, len(ws) - 1)) + ([0] if len(ws) > 1 else []):
            if avail <= 1 or total + (more if self._crumb_hidden or hidden else 0) <= avail:
                break
            hidden.add(i)
            total -= widths[i]
        over = total + (more if self._crumb_hidden or hidden else 0) - avail
        if avail > 1 and over > 0:
            ruler = self._rulers[True]
            last.configure(text=ruler.fit(full, ruler.width(full) - over, self.G["ell"], middle=True))
        for s, b, _l, _p in ws:
            b.pack_forget()
            if s is not None:
                s.pack_forget()
        self._more.pack_forget()
        items = self._crumb_hidden + [(label, path) for i, (_s, _b, label, path) in enumerate(ws) if i in hidden]
        self._more_menu.delete(0, "end")
        for label, path in items:
            self._more_menu.add_command(label=self.T(label), command=lambda p=path: self.navigate(p))
        if items:
            self._more.pack(side="left")
        for i, (s, b, _l, _p) in enumerate(ws):
            if i not in hidden:
                if s is not None:
                    s.pack(side="left")
                b.pack(side="left")

    def _edit_off(self) -> None:
        """Back to the crumbs: the keyboard goes to the listing, and a message about the typed path goes too."""
        if self._editing:
            self._editing = False
            self._lb_hide()
            self._focus_list(hiding=self._edit)
            self._edit.pack_forget()
            self._crumb_box.pack(side="left", fill="y")
            if self._edit_error:
                self._strip_msg, self._edit_error = None, False
            self._update_target()

    def _edit_focus_lost(self) -> None:
        """Leaving the Entry for another widget of this dialog ends path editing; another window does not, nor
        does the strip's action about the typed path: a click gives [Go there] the keyboard before it acts, and
        ending path editing then would take the button away under the pointer."""
        self._lb_hide()
        focus = str(self.tk.call("focus"))
        if self._editing and focus and focus not in (str(self._edit), str(self._strip_btn)) \
                and focus.startswith(str(self) + "."):
            self._edit_off()

    def _edit_typed(self, _e: Any) -> None:
        """The typed text changed: a completion cycle, the suggestion moved to and a message about the old text
        end; the suggestions follow the text."""
        text = self._edit_var.get()
        if text == self._edit_text:
            return
        self._edit_text, self._cycle = text, None
        if self._edit_error:
            self._strip_msg, self._edit_error = None, False
            self._update_target()
        if self._lb_visible():
            self._lb_pick(None)                           # Enter takes what is typed now (BR-6)
            self._later("suggest", 120, self._lb_show)

    def _edit_changed(self, *_a: Any) -> None:
        """An edit (including a paste before KeyRelease) makes the pending typed lookup obsolete."""
        if self._editing:
            self._new_intent()
            if self._locate_token:
                self._cancel_locate()

    def _edit_enter(self, choose: bool = False) -> None:
        """Enter opens the typed path, or the suggestion moved to with Down, Up or a click (BR-6); Ctrl+Enter
        also uses it (BR-5). A key typed since that move whose release hasn't come yet (a fast typist's keys
        overlap) is typing too: the text wins."""
        i = self._lb_index() if self._edit_var.get() == self._edit_text else None
        if i is not None:
            parent, matches = self._lb_items
            self._set_edit(os.path.join(parent, matches[i]))
        self._lb_hide()
        self._commit_path(self._edit_var.get(), choose)

    def _commit_path(self, raw: str, choose: bool = False) -> None:
        """Enter in path edit: clean the text (UNC/smb mapping included), then locate it (§2.2.5). Path editing
        goes on, with the text, until the path is found (BR-16); with `choose` (Ctrl+Enter) the folder found is
        then used (BR-5)."""
        text = normalize_user_path(raw, self.ctx.fs.mounts)
        if not text:
            self._edit_off()
            return
        error = None
        if _BARE_SERVER.match(text):
            error = "Type the full share path, for example \\\\server\\share."
        elif text.startswith(("\\\\", "//", "smb:")) and not os.path.isabs(text):
            error = f"{self.q(text)} isn't mounted on this computer."
        elif WINDOWS and not os.path.isabs(text) and os.path.splitdrive(text)[0]:
            error = f"Can't find {self.q(text)}. Type the full path of a folder."      # 'C:data': which folder of C:?
        elif not os.path.isabs(text):
            text = os.path.normpath(os.path.join(self._path, text))   # a name (or a/b, ..) in the folder shown
        if error:
            self._strip_msg, self._edit_error = ("Warn", error, None), True
            self._update_target()
            return
        if choose and same_path(text, self._path):       # the folder shown: use it as it is
            self._edit_off()
            self._use_here()
            self.choose()
            return

        def missing(closest: Optional[str], found: str) -> None:
            """The text stays, with its first part that doesn't exist selected; the folder it isn't in is named
            once, and offered (BR-16)."""
            if self._editing and self._edit_var.get() != raw:
                return                                    # the user has typed on since: this is about old text
            rest = found[len(closest):].lstrip("\\/") if closest else found
            part = re.split(r"[\\/]", rest)[0] or rest
            self.edit_path(found)
            start = len(found) - len(rest)
            self._edit.selection_range(start, start + len(part))
            self._edit.icursor(start + len(part))
            where = f" in {self.q(self._label(closest))}" if closest else ""
            go = ("Go there", lambda: self.navigate(closest)) if closest else None
            self._strip_msg, self._edit_error = ("Error", f"Can't find {self.q(part)}{where}.", go), True
            self._update_target()
        self._locate(text, missing, self._cancel_locate, then=self.choose if choose else None)

    def _matches(self, text: str, on_ready: Callable[[], None]) -> Tuple[str, Optional[List[str]]]:
        """(parent, sub-folder names matching the typed prefix case-insensitively); None while the parent is listed."""
        text = os.path.expanduser(text) if text.startswith("~") else text
        if text and not os.path.isabs(text) and not text.startswith(("\\\\", "//", "smb:")):
            text = os.path.join(self._path, text)         # a name in the folder shown, as Enter takes it
        if text.endswith(("/", "\\") if WINDOWS else "/"):
            parent, prefix = text, ""
        else:
            parent, prefix = os.path.dirname(text), os.path.basename(text)
        if not parent or not os.path.isabs(parent):
            return parent, []
        parent = os.path.normpath(parent)
        if (parent != self._path or self._listing is None) and parent not in self._comp_paths:
            asked = self._edit_var.get()

            def resolved(found: str, err: Optional[FsError]) -> None:
                if err is None:
                    self._comp_paths[parent] = found
                    on_ready()
            state = _link_start(parent, known=self._path if self._listing is not None else "")
            self._resolve_links(state, resolved, lambda *_a: None,
                                lambda: self._editing and self._edit_var.get() == asked, self._comp_ticket)
            return parent, None
        parent = self._comp_paths.get(parent, parent)
        listing = self.ctx.fs.cached_listing(parent)
        if listing is None:
            asked = self._edit_var.get()
            rid = self.ctx.fs.list_dir(parent, ticket=self._comp_ticket, owner="browser", priority=0)
            self._calls[rid] = (None, lambda e: e.error is None and self._edit_var.get() == asked and on_ready())
            return parent, None
        pre, show_hidden = prefix.casefold(), self._v_hidden.get()
        aliases = {r.name for r in self._rows.values() if r.link and r.kind == "dir" and r.target is not None
                   and r.summary is not None and r.error is None} if same_path(parent, self._path) else set()
        names = [e.name for e in listing.entries if (e.is_dir or e.name in aliases) and e.name.casefold().startswith(pre)
                 and (show_hidden or not (e.hidden or e.name.startswith(".")))]
        return parent, sorted(names, key=natural_key)

    def _tab(self) -> None:
        """Tab: complete the longest common prefix and, when several folders match, show them (BR-29); a second
        Tab cycles through them (§2.2.5)."""
        text = self._edit_var.get()
        if self._cycle is not None and self._cycle[0] == text:
            _t, parent, matches, i = self._cycle
            i = (i + 1) % len(matches)
            new = os.path.join(parent, matches[i])
            self._cycle = (new, parent, matches, i)
            self._set_edit(new)
            if self._lb_visible():
                self._lb_pick(i if i < len(self._lb_items[1]) else None)
            return
        parent, matches = self._matches(text, self._tab)
        if not matches:
            if matches is not None:
                self.bell()
            return
        if len(matches) == 1:
            new = os.path.join(parent, matches[0], "")
        else:
            low = [m.casefold() for m in matches]
            k = len(os.path.commonprefix(low))
            new = os.path.join(parent, matches[0][:k])
        self._cycle = (new, parent, matches, -1) if len(matches) > 1 else None
        self._set_edit(new)
        if len(matches) > 1 or self._lb_visible():
            self._lb_show()

    def _set_edit(self, text: str) -> None:
        self._edit_var.set(text)
        self._edit_text = text
        self._edit.icursor("end")
        self._edit.xview_moveto(1.0)

    def _lb_visible(self) -> bool:
        return self._lb is not None and bool(self._lb_box.winfo_manager())

    def _lb_index(self) -> Optional[int]:
        """The suggestion moved to with Down, Up or a click, while the list shows."""
        sel = self._lb.selection() if self._lb_visible() else ()
        return int(sel[0]) if sel else None

    def _lb_pick(self, i: Optional[int]) -> None:
        self._lb.selection_set((str(i),) if i is not None else ())
        if i is not None:
            self._lb.see(str(i))

    def _lb_show(self) -> None:
        """The suggestions: up to 10 folders whose names start with the typed text, drawn like the listing's rows
        (icon, name, badge from cached probes) in a list place()d under the Entry, inside the dialog (BR-23). The
        Entry keeps the keyboard, and no suggestion is chosen until Down, Up or a click moves to one (BR-6)."""
        if not self._editing:
            return
        parent, matches = self._matches(self._edit_var.get(), self._lb_show)
        if not matches:
            if matches is not None:
                self._lb_hide()
            return
        old, i = self._lb_items, self._lb_index()         # a move made since the last key stays (typing ends it)
        chosen = old[1][i] if i is not None and old[0] == parent else None
        lb = self._lb
        if lb is None:                                    # a 1-px frame sets it apart from the listing below
            self._lb_box = tk.Frame(self, background=T["border_strong"], borderwidth=0, highlightthickness=0)
            lb = self._lb = ttk.Treeview(self._lb_box, show="tree", columns=("badge",), selectmode="browse",
                                         takefocus=0, style="Listing.Treeview")
            lb.pack(fill="both", expand=True, padx=1, pady=1)
            tag_tree(lb, "Listing")
            lb.bind("<Button-1>", self._lb_click)
        shown, px, ell = matches[:10], self.px, self.G["ell"]
        badges = [self._cached_badge(parent, m) for m in shown]
        names = [self.T(m) for m in shown]
        room, icon = max(1, self._edit.winfo_width() - px(4) - 2), self.ctx.icons.size + px(14)
        name_w = min(max(self._rulers[True].width(n) for n in names) + icon + px(8), room * 3 // 5)
        lb.column("#0", width=name_w, minwidth=0, stretch=False)
        lb.column("badge", width=room - name_w, minwidth=0, stretch=False)
        lb.delete(*lb.get_children(""))
        for i, (name, b) in enumerate(zip(names, badges)):
            b = b or Badge("folder", "", "", False, True, 7)          # nothing known yet
            ruler = self._rulers[b.bold]
            lb.insert("", "end", iid=str(i), text=ruler.fit(name, name_w - icon, ell, middle=True),
                      image=self.ctx.icons.tree(b.icon, dim=b.dim and b.icon != "folder_error"),
                      values=(ruler.fit(self.T(b.inside), room - name_w - px(10), ell),),
                      tags=("slide",) if b.bold else ("dim",) if b.dim else ())
        lb.configure(height=len(shown))
        self._lb_items = (parent, shown)
        if chosen in shown:
            self._lb_pick(shown.index(chosen))
        self._lb_box.place(in_=self._edit, x=0, y=-1, rely=1.0, relwidth=1.0, anchor="nw")
        self._lb_box.lift()

    def _lb_click(self, e: Any) -> str:
        """A click on a suggestion opens it; the Entry keeps the keyboard (a Treeview takes it on a click)."""
        row = self._lb.identify_row(e.y)
        if row:
            self._lb_pick(int(row))
            self._edit_enter()
        return "break"

    def _cached_badge(self, parent: str, name: str) -> Optional[Badge]:
        """A suggestion's badge from cached probes only (no request): its own summary and, for a parent folder,
        the cached summaries of up to 40 of its subfolders; None when nothing is known."""
        key = self._names.get(name) if same_path(parent, self._path) else None
        if key in self._rows:
            return self._badge(self._rows[key])
        path, fs = os.path.join(parent, name), self.ctx.fs
        s = fs.cached_summary(path)
        if s is None:
            return None
        roll = None
        if not s.groups and s.subdirs:
            known = [x for x in (fs.cached_summary(os.path.join(path, d)) for d in s.subdirs[:ROLLUP_PER_CHILD]) if x]
            if not known:
                return Badge("folder", "", _n(len(s.subdirs), "folder"), False, False, 7)
            slides = [sum(_stitchable(x).values()) for x in known if _stitchable(x)]
            roll = Rollup(len(known), len(slides), sum(slides), len(known) < len(s.subdirs))
        return describe_folder(s, roll, self.mode, self._out_names, LOCAL, None, fs.auto_enter(path))

    def _lb_move(self, step: int) -> None:
        """Down opens the suggestions (none chosen yet), then moves down the list; Up moves up, and from the
        first suggestion back to the typed text."""
        if not self._lb_visible():
            if step > 0:
                self._lb_show()
            return
        i = self._lb_index()
        i = (0 if step > 0 else None) if i is None else i + step
        self._lb_pick(None if i is None or i < 0 else min(i, len(self._lb_items[1]) - 1))

    def _lb_hide(self) -> None:
        if self._lb_visible():
            self._lb_box.place_forget()

    def _escape(self, _e: Any = None) -> str:
        """Esc, in order: close the suggestions or leave path edit (giving up a typed path still being looked
        for), clear the filter, clear the selection, Cancel."""
        if self._lb_visible():
            self._lb_hide()
        elif self._editing:
            if self._locate_token and self._wait_back == self._cancel_locate:
                self._cancel_locate()
            self._edit_off()
        elif self._filter_var.get():
            self._filter_var.set("")
            self.tree.focus_set()
        elif self.tree.selection():
            self._use_here()
        else:
            self.cancel()
        return "break"

    # ------------------------------------------------------------------------------------- places sidebar
    def _load_places(self) -> None:
        """list_places() reads small config files and system_picker_available() searches PATH: a worker does both.
        A newer load (a newer mount table) makes an older answer obsolete."""
        ctx, mode, mounts, token = self.ctx, self.mode, self.ctx.fs.mounts, next(_TOKENS)
        self._token = token
        recent = ctx.settings.get(f"recent.{mode}s") or []      # (the temporary-files chooser has none)

        def work() -> None:
            try:
                places: Tuple[Place, ...] = tuple(list_places(mounts, recent, mode))
            except Exception:
                log.exception("Could not list the places")
                places = ()
            ctx.bus.post(_PlacesLoaded(token, places, osutil.system_picker_available()))
        threading.Thread(target=work, name="yfs-places", daemon=True).start()

    def _on_places(self, ev: _PlacesLoaded) -> None:
        if ev.token != self._token or not self.winfo_exists():
            return
        self._places, self._picker_ok = ev.places, ev.picker
        side = self._side
        side.delete(*side.get_children(""))
        self._place_items, section = {}, None
        for p in self._places:
            if p.section != section:
                section = p.section
                side.insert("", "end", text=_SECTION.get(section, section.upper()), tags=("section",))
            iid = side.insert("", "end")
            self._place_items[iid] = (p, self.ctx.fs.lane_key(p.path))
            self._paint_place(iid)
            if WINDOWS and p.section == "computer" and re.match(r"^[A-Za-z]:\\$", p.path):
                self._call(p.path, "call", partial(windows_volume_label, p.path),   # 'Local Disk (C:)' -> its label
                           lambda e, i=iid: self._volume_label(i, e), priority=7, deadline=5)
        self._crumbs()
        self._mark_place()
        self._update_target()

    def _volume_label(self, iid: str, e: FsDone) -> None:
        if e.error is None and e.result and iid in self._place_items:
            p, lane = self._place_items[iid]
            self._place_items[iid] = (dataclasses.replace(p, label=f"{e.result} ({p.path[:2]})"), lane)
            self._paint_place(iid)

    def _paint_place(self, iid: str) -> None:
        """Health dot (no I/O: lane state only) and label suffix (§2.2.4)."""
        p, lane = self._place_items[iid]
        health, recent = self.ctx.fs.health(lane), p.section == "recent"
        dot = _DOT.get(health) if health != H_OK or p.lane_class != LOCAL else None
        dim = recent and health in (H_UNRESPONSIVE, H_DISCONNECTED)     # Recent: dimmed, no suffix
        label = p.label.removesuffix(" (cloud)") if p.section == "cloud" else p.label   # the heading says it
        text = " " + self.T(label + ("" if recent else _HEALTH.get(health, "")))
        room = self._side.column("#0", "width") - self.ctx.icons.size - self.px(12)
        self._side.item(iid, text=self._rulers[False].fit(text, room, self.G["ell"]),
                        image=self.ctx.icons.badged(p.icon, dot), tags=("dim",) if dim else ())

    def _on_health(self, e: LaneHealthChanged) -> None:
        if self.ctx.fs.health_snapshot(e.lane) != (e.health, e.since):
            return
        if e.health == H_UNRESPONSIVE:
            self._unresponsive_since[e.lane] = e.since
        for iid, (_p, lane) in self._place_items.items():
            if lane == e.lane:
                self._paint_place(iid)
        if e.health in (H_OK, H_SLOW):
            self._answers_again(e.lane)

    def _answers_again(self, lane: str) -> None:
        """A lane answers again: ask once more what didn't answer in it (UD1). Nothing retries by itself
        otherwise, and nothing is asked while the lane still hangs."""
        fs = self.ctx.fs
        health, since = fs.health_snapshot(lane)
        if health not in (H_OK, H_SLOW):
            return

        def fresh(kind, path):
            key = (lane, kind, path)
            if self._recovery_seen.get(key) == since:
                return False
            self._recovery_seen[key] = since
            return True

        if fs.lane_key(self._path) == lane:
            dates = [r for r in self._rows.values() if r.date_error is not None and r.date_error.code in _STALL
                     and fresh("date", self._child(r.name))]
            self._statted.difference_update(r.name for r in dates)
            for r in dates:
                r.date_error = None
            self._stat(dates, 6)
        if self._stall is not None and self._stall[0] == lane and fresh("navigate", self._wait_path):
            again, self._stall = self._stall[1], None              # a late listing comes first (_end_wait)
            self._later("again", 500, again)
        for r in list(self._rows.values()):
            if r.stalled and fs.lane_key(r.stalled) == lane and fresh("probe", r.stalled):
                r.error = r.stalled = None
                if r.link and r.target is None:
                    r.probing = 4
                    self._readlink(r, self._child(r.name), 0)
                else:
                    self._probe(r, 4)
                self._refresh_row(r)
            elif r.roll is not None and any(fs.lane_key(p) == lane for p in r.roll[6]):
                again = [p for p in r.roll[6] if fs.lane_key(p) == lane and fresh("roll", p)]
                r.roll[6] = [p for p in r.roll[6] if p not in again]
                for path in again:                        # the subfolders that answered are counted already
                    self._roll_probe(r, path)
                self._refresh_row(r)

    def _on_mounts(self, _e: MountsChanged) -> None:
        """A newer mount table (UD2): the sidebar, each folder's reason not to be probed, and the probes that are
        allowed now."""
        fs = self.ctx.fs
        self.ctx.mounts = fs.mounts
        self._load_places()
        if not self._path:
            return
        self._lane_class = fs.lane_class(self._path)
        self._fresh = 10.0 if self._lane_class == LOCAL else 60.0
        for r in self._rows.values():
            if r.kind == "dir" and not (r.link and r.target is None):
                auto = fs.auto_enter(self._row_path(r))
                if auto != r.auto:
                    r.auto = auto
                    self._refresh_row(r)
        self._probe_children()
        self._update_target()

    def _mark_place(self) -> None:
        """Select the place the current folder is in; Up/Down in the sidebar start from there (BUX-M1)."""
        best = containing_place(self._path, self._places) if self._places else None
        exact = next((i for i, (p, _l) in self._place_items.items() if same_path(p.path, self._path)), None)
        hit = exact or next((i for i, (p, _l) in self._place_items.items() if p is best), None)
        self._side.selection_set((hit,) if hit else ())
        if hit:
            self._side.focus(hit)

    def _side_move(self, step: int) -> str:
        """Up/Down in the sidebar move between places, over the section titles; Enter opens one."""
        items = [i for i in self._side.get_children("") if i in self._place_items]
        if items:
            cur = self._side.focus()
            i = items.index(cur) + step if cur in items else 0 if step > 0 else len(items) - 1
            iid = items[max(0, min(i, len(items) - 1))]
            self._side.selection_set(iid)
            self._side.focus(iid)
            self._side.see(iid)
        return "break"

    def _open_place(self, iid: str) -> None:
        if iid in self._place_items:
            path = self._place_items[iid][0].path
            self._locate(path, lambda *_a: self._show_error(
                FsError("not_found", "This place no longer exists."), path), self._cancel_locate)

    def _dismiss_place_error(self) -> None:
        """A failed sidebar destination leaves the current listing and its selection intact."""
        self._overlay_hide()
        self._mark_place()
        self._empty_state()

    def _place_menu(self, e: Any) -> None:
        iid = self._side.identify_row(e.y)
        if iid not in self._place_items:
            return
        p = self._place_items[iid][0]
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Copy path", command=lambda: (self.clipboard_clear(), self.clipboard_append(p.path)))
        if p.section == "recent":
            menu.add_command(label="Remove from recent", command=lambda: self._forget_recent(p.path))
        menu.tk_popup(e.x_root, e.y_root)

    def _forget_recent(self, path: str) -> None:
        key = "recent.inputs" if self.mode == "input" else "recent.outputs"
        s = self.ctx.settings
        s.set(key, [r for r in s.get(key) or [] if not same_path(r.get("path", ""), path)])
        places = tuple(p for p in self._places if not (p.section == "recent" and same_path(p.path, path)))
        self._on_places(_PlacesLoaded(self._token, places, self._picker_ok))

    def _side_motion(self, e: Any) -> None:
        iid = self._side.identify_row(e.y)
        p = self._place_items[iid][0] if iid in self._place_items else None
        self._side_tip.point(iid, self.T(p.label) + "\n" + p.detail if p is not None else "")

    # ------------------------------------------------------------------------------------- system picker
    def _picker_link(self) -> None:
        if self._picker_ok and not self.ctx.running():
            self._picker.grid(row=2, column=0, sticky="w")
        else:
            self._picker.grid_remove()
        self._fit_primary()

    def _system_picker(self) -> None:
        """Windows/macOS: the native dialog (main thread, explicit click). Linux: zenity in a worker (§2.2.9).
        The result is not taken blindly: the browser shows it, selected, and Enter confirms."""
        self._new_intent()
        self._update_target()
        title = self.T(self._title)
        if WINDOWS or MACOS:
            path = filedialog.askdirectory(parent=self, initialdir=self._path, title=title, mustexist=True)
            safe_grab(self)
            if path:
                self._picked(path)
            return
        self._picker_token = next(_TOKENS) + 10 ** 9      # far from tokens other windows may use
        self._kill = osutil.run_zenity(self._path, title, self.ctx.bus.post, self._picker_token)
        self._overlay("picker", "wait", f"Waiting for the system folder picker{ELL}", "",
                      [("Cancel", self._kill_picker)])

    def _kill_picker(self) -> None:
        if self._kill is not None:
            self._kill()
            self._kill = None
        if self._ov_kind == "picker":
            self._overlay_hide()

    def _on_picker(self, e: PickerResult) -> None:
        if e.token != self._picker_token or not self.winfo_exists():
            return
        self._kill = None
        if self._ov_kind == "picker":
            self._overlay_hide()
            self._empty_state()
        if e.error:
            self._strip_msg = ("Error", e.error, None)
            self._update_target()
        elif e.path:
            self._picked(e.path)

    def _picked(self, path: str) -> None:
        path = os.path.normpath(path)
        self._locate(os.path.dirname(path) or path, lambda *_a: self._show_error(
            FsError("not_found", "This folder no longer exists.")), self._cancel_locate,
                     select=_base(path) if os.path.dirname(path) != path else None, then=self._primary.focus_set)

    # ------------------------------------------------------------------------------------- plumbing
    def _call(self, path: str, kind: str, fn: Callable[[], Any], cb: Callable[[FsDone], None], *,
              user: bool = False, priority: int = 0, deadline: Optional[float] = None) -> int:
        rid = self.ctx.fs.call(path, kind, fn, ticket=0, owner="browser", priority=priority, user_initiated=user,
                               deadline_s=deadline)
        self._calls[rid] = (None, cb)
        return rid

    def _on_fs_done(self, e: FsDone) -> None:
        if e.owner != "browser":
            return
        entry = self._calls.pop(e.req_id, None)
        if entry is not None and (entry[0] is None or entry[0] == self._ticket) and self.winfo_exists():
            entry[1](e)
            if e.error is not None and e.error.code in _STALL:
                lane = self.ctx.fs.lane_key(os.path.dirname(e.path) if e.kind == "readlink" else e.path)
                self._later("recover:" + lane, 0, lambda: self._answers_again(lane))

    def _later(self, name: str, ms: int, fn: Callable[[], Any]) -> None:
        """Run fn once after ms; a newer request with the same name replaces it (debounce)."""
        self._cancel_job(name)

        def run() -> None:
            self._jobs.pop(name, None)
            fn()
        self._jobs[name] = self.after(ms, run)

    def _cancel_job(self, name: str) -> None:
        job = self._jobs.pop(name, None)
        if job is not None:
            self.after_cancel(job)

    def _on_configure(self, e: Any) -> None:
        if e.widget is self:
            self._lb_hide()
            self._show_preview(e.width >= self.px(1000))

    def _show_preview(self, on: bool) -> None:
        """The preview pane exists only while the dialog is at least px(1000) wide."""
        if on == self._preview_on:
            return
        self._preview_on = on
        if on:
            if self._pw.winfo_width() > 1:               # a sidebar dragged wider or narrower stays so
                self._side_w = self._pw.sashpos(0)
            self._pw.add(self._preview, weight=0)
            self._later("sash", 50, self._init_sashes)
            self._later("preview", 50, self._update_preview)
        else:
            self._pw.forget(self._preview)

    def _cleanup(self) -> None:
        """<Destroy>: stop listening and drop queued work; widgets may already be gone, so none is touched. The Tk
        variables die here, on the Tk thread: a garbage collection on a worker thread must never be the one that
        frees them (UD12, CT-6)."""
        for unsubscribe in self._unsub:
            unsubscribe()
        for name in list(self._jobs):
            self._cancel_job(name)
        self.ctx.fs.cancel_ticket(self._ticket)
        self.ctx.fs.cancel_ticket(self._comp_ticket)
        self.ctx.fs.cancel_ticket(self._capacity_ticket)
        if self._kill is not None:
            self._kill()
            self._kill = None
        del self._v_tiles, self._v_hidden, self._v_all, self._edit_var, self._filter_var, self._footer_options
        del self._sort_column_var, self._sort_order_var
        gc.collect()
