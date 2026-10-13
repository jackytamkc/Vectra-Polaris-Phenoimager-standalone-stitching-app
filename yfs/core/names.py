"""Name rules: classify files, summarise listings, build candidates and notes, output names, estimates.

FINAL_SPEC §3.1-§3.4, §3.7 and §3.9. Pure: no I/O and no tkinter, so it is safe on any thread.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model import Candidate, DirSummary, Entry, Estimate, Note, TileGroup, TileHeader

# ---- §3.1 constants
MAX_DEPTH_DEFAULT = 3          # levels below the chosen folder (the root itself is depth 0)
DEEPER_STEP = 3                # "Search deeper" adds this many levels
WIDE_THRESHOLD = 200           # a folder with more subfolders than this is "wide"
WIDE_SAMPLE = 50               # ...and only its first 50 (natural order) are searched first
BUDGET_DIRS = 3000             # pause after this many folders ...
BUDGET_SECONDS = 60.0          # ... or this much time; each resume grants the same again
LIST_CAP_NO_TILES = 5000       # stop listing a folder after 5,000 entries if no tile name has been seen
PROBE_CAP = 5000               # browser badge probes
BROWSER_LIST_CAP = 20000       # browser listing of the current folder
ROLLUP_PER_CHILD = 40          # browser: grandchildren probed per child
ROLLUP_PER_VIEW = 200          # browser: roll-up probes per folder view
FEW_TILES = 10                 # < 10 tiles -> unticked ("stamps or a test?")
SPARSE_COVERAGE = 0.10         # coverage below 10 % ...
SPARSE_MIN_BYTES = 1 << 30     # ... and a raw canvas above 1 GiB -> unticked ("sparse")
RECENT_CHANGE_S = 300          # folder mtime newer than 5 min -> flag recent_change
SIBLING_MIN, SIBLING_RATIO = 3, 0.25   # >= 3 sibling slide folders and n < 0.25 x median -> flag sibling_low
DUPLICATE_RATIO = 4.0          # "smaller copy" = another copy has >= 4x the tiles
PARTIAL_DIR = ".stitcher-in-progress"
OWNER_SUFFIX = ".owner"        # <partial name>.owner: who claimed a partial (JSON: host, pid, started, token)
COMPLETE_SUFFIX = ".complete"  # <partial name>.complete: the partial is complete and fsynced; it only waits to be
                               # moved into place (the old file was in use). Never taken over, never a leftover.
STALE_INFIX = ".stale-"        # <partial name>.stale-<STALE_HEX hex digits>: a stale partial while it is taken over
STALE_HEX = 8                  # (D3); the longest name the runner gives a file in PARTIAL_DIR
TEMP_SUBDIR_PREFIX = ".yfs-"   # custom temp folder: one subfolder per save folder (see temp_subdir)
WINDOWS_MAX_PATH = 259         # longest path Windows accepts without LongPathsEnabled
STALE_PARTIAL_S = 120          # a partial untouched for 120 s belongs to nobody
HEARTBEAT_S = 30               # the runner touches its claimed partial this often (R29)
RECENT_OUTPUT_S = 120          # an output modified this recently is not verified yet
PSEUDO_FSTYPES = frozenset({"proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2", "securityfs",
                            "debugfs", "tracefs", "configfs", "fusectl", "pstore", "bpf", "mqueue", "hugetlbfs",
                            "overlay", "squashfs", "nsfs", "binfmt_misc", "efivarfs", "ramfs", "rpc_pipefs",
                            "fuse.portal"})
OUT_TYPICAL_FACTOR = 0.91      # typical output size / raw canvas bytes
OUT_WORST_FACTOR = 1.34        # worst case: uncompressed + 1/3 pyramid
LEVEL0_FACTOR = 0.68           # expected level-0 bytes / raw canvas bytes (write progress)
RAM_HEADROOM_GB = 2.0          # v1 rule: RAM mode iff needed + 2 GB < available
OME_SUFFIX = ".ome.tif"
STEM_BYTES = 180                  # UTF-8; leaves room within 255 bytes for every sidecar, scratch and takeover

# ---- §3.2 regular expressions (verbatim)
TILE_RE = re.compile(r"^(?P<prefix>.+)_\[(?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\]"
                     r"(?:_(?P<suffix>[^.\[\]]+))?\.tiff?$", re.IGNORECASE)
OUTPUT_RE = re.compile(r"\.ome\.tiff?$", re.IGNORECASE)
WSI_RE = re.compile(r"_stitch(?:ed)?\.tiff?$", re.IGNORECASE)
RAW_RE = re.compile(r"\.(?:qptiff|im3)$", re.IGNORECASE)
SCANNER_AUX_RE = re.compile(r"^(?:label|overview\w*|\w*mask|focusmap|markup\w*)\.tiff?$", re.IGNORECASE)
INFORM_MARKER_RE = re.compile(r"^(?:batch_procedure\.ifp|batch\.log|mergestatus\.xml|.+\.ifp)$", re.IGNORECASE)
TMP_RAW_RE = re.compile(r"\.tmp\.raw$", re.IGNORECASE)
TIFF_RE = re.compile(r"\.tiff?$", re.IGNORECASE)
IGNORED_FILE_RE = re.compile(r"^(?:\._.*|\.ds_store|thumbs\.db|desktop\.ini|~\$.*)$", re.IGNORECASE)
LNK_RE = re.compile(r"\.lnk$", re.IGNORECASE)
COMPONENT_SUFFIX = "component_data"
EXCLUDED_DIR_NAMES = frozenset({            # compared with name.casefold(); always skipped by discovery
    "$recycle.bin", "system volume information", "lost+found", "@eadir", "#recycle", ".snapshot",
    "~snapshot", "__macosx", "__pycache__", ".git", "node_modules", ".trash", ".trashes",
    ".spotlight-v100", ".fseventsd", ".documentrevisions-v100", ".temporaryitems", ".stitcher-in-progress"})

_RULES = ((IGNORED_FILE_RE, "ignored"), (LNK_RE, "lnk"), (TMP_RAW_RE, "tmp_raw"), (OUTPUT_RE, "output"),
          (WSI_RE, "wsi"), (RAW_RE, "raw"), (TILE_RE, "tile"), (SCANNER_AUX_RE, "scanner_aux"),
          (INFORM_MARKER_RE, "inform_marker"), (TIFF_RE, "tiff_other"))
_NOTABLE = frozenset({"output", "wsi", "raw", "inform_marker", "tmp_raw", "lnk"})
_DIGITS = re.compile(r"(\d+)")
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\ud800-\udfff]')   # surrogates: undecodable bytes of a Linux name
_RESERVED = frozenset(["con", "prn", "aux", "nul"] + [f"{d}{i}" for d in ("com", "lpt") for i in range(1, 10)])
_CASE_INSENSITIVE_FS = sys.platform in ("win32", "darwin")


def classify_file(name: str) -> str:
    """§3.3: the kind of a file name (the first rule that matches wins)."""
    return next((kind for rx, kind in _RULES if rx.search(name)), "other")


def parse_tile_name(name: str) -> Optional[Tuple[str, float, float, str]]:
    """(prefix, x_um, y_um, suffix_lower) of a tile name, or None if the name is not a tile."""
    if classify_file(name) != "tile":
        return None
    m = TILE_RE.match(name)
    return m["prefix"], float(m["x"]), float(m["y"]), (m["suffix"] or "").lower()


def is_excluded_dir(e: Entry, show_hidden: bool = False) -> bool:
    """System names and .trash-* always; hidden (dot or attribute) folders unless show_hidden."""
    n = e.name.casefold()
    if n in EXCLUDED_DIR_NAMES or n.startswith(".trash-"):
        return True
    return not show_hidden and (e.hidden or n.startswith("."))


def summarize_listing(path: str, entries: Sequence[Entry], *, show_hidden: bool = False,
                      truncated: bool = False) -> DirSummary:
    """§3.4: one listing -> tile groups, outputs, notable files and counts. Names only, nothing is opened."""
    groups = defaultdict(list)                       # (prefix, suffix_lower) -> [(name, x, y)]
    subdirs, notable = [], []
    lists = {"output": [], "wsi": [], "tmp_raw": []}
    raw = inform = partial_dir = False
    other_tiffs = other_files = links = 0
    for e in entries:
        if e.is_dir:
            if e.name == PARTIAL_DIR:
                partial_dir = True
            elif e.is_link:                          # Windows junction: counted, never followed
                links += 1
            elif not is_excluded_dir(e, show_hidden):
                subdirs.append(e.name)
            continue
        k = classify_file(e.name)
        if e.is_link and k == "other":               # probably a folder symlink: counted, never followed
            links += 1
            continue
        if k == "tile":
            m = TILE_RE.match(e.name)
            groups[(m["prefix"], (m["suffix"] or "").lower())].append((e.name, float(m["x"]), float(m["y"])))
        elif k in lists:
            lists[k].append(e.name)
        elif k in ("raw", "scanner_aux"):
            raw = True
        elif k == "inform_marker":
            inform = True
        elif k == "tiff_other":
            other_tiffs += 1
        elif k not in ("lnk", "ignored") and (show_hidden or not (e.hidden or e.name.startswith("."))):
            other_files += 1
        if k in _NOTABLE:
            notable.append(e.name)
    tile_groups = tuple(TileGroup(p, s, tuple(n for n, _, _ in g), tuple((x, y) for _, x, y in g))
                        for (p, s), g in ((key, sorted(v)) for key, v in sorted(groups.items())))  # engine order
    return DirSummary(path, len(entries), truncated, _nat(subdirs), tile_groups, _nat(lists["output"]),
                      _nat(lists["wsi"]), raw, inform, other_tiffs,
                      _nat(lists["tmp_raw"]) + ((PARTIAL_DIR,) if partial_dir else ()), _nat(notable)[:200],
                      other_files, links=links)


def _nat(names: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(names, key=natural_key))


def rel_folder(folder: str, root: str) -> str:
    """Path of `folder` relative to `root` with "/" separators ("." for the root itself)."""
    try:
        return os.path.relpath(folder, root).replace(os.sep, "/")
    except ValueError:                               # another Windows drive
        return folder.replace(os.sep, "/")


def stitchable_groups(summary: Optional[DirSummary]) -> Dict[str, TileGroup]:
    """One group per prefix, in listing order: prefer unmixed components over unsuffixed tiles."""
    groups: Dict[str, TileGroup] = {}
    for group in summary.groups if summary is not None else ():
        if group.suffix == COMPONENT_SUFFIX or (group.suffix == "" and group.prefix not in groups):
            groups[group.prefix] = group
    return groups


def candidates_from_summary(s: DirSummary, *, root: str, depth: int, dir_mtime: Optional[float]) -> List[Candidate]:
    """§3.4: one candidate per prefix (component, else unsuffixed) plus one per other suffix; else legacy_all."""
    by_prefix = defaultdict(dict)                    # prefix -> {suffix: TileGroup}
    for g in s.groups:
        by_prefix[g.prefix][g.suffix] = g
    stitchable = stitchable_groups(s)
    multi = len(stitchable) > 1
    rel, name = rel_folder(s.path, root), os.path.basename(os.path.normpath(s.path)) or s.path
    base = os.path.normpath(s.path)

    def cand(prefix: str, suffix: str, kind: str, g: Optional[TileGroup] = None) -> Candidate:
        # Discovered entries are identities, even on case-sensitive Windows/macOS volumes.
        key = f"{base}|*|legacy" if g is None else \
            f"{base}|{prefix}|{suffix}"
        return Candidate(key, s.path, rel, name, prefix, suffix, kind, g.names if g else (), g.coords if g else (),
                         multi, s.inform_export, depth, dir_mtime)

    out = []
    for p, d in sorted(by_prefix.items(), key=lambda kv: natural_key(kv[0])):
        if p in stitchable:
            g = stitchable[p]
            out.append(cand(p, g.suffix, "component" if g.suffix else "unsuffixed", g))
        out += [cand(p, suf, "other_suffix", g) for suf, g in sorted(d.items()) if suf not in (COMPONENT_SUFFIX, "")]
    if not s.groups and s.other_tiffs > 0 and not s.raw_scan:
        out.append(cand("", "legacy", "legacy_all"))   # "every TIFF in this folder" (v1 behaviour)
    return out


def notes_from_summary(s: DirSummary) -> List[Note]:
    """§3.4 notes table."""
    p, lnk, out = s.path, tuple(n for n in s.notable_files if LNK_RE.search(n)), []
    if s.outputs: out.append(Note("outputs", p, len(s.outputs), names=s.outputs[:5]))
    if s.wsi: out.append(Note("wsi", p, len(s.wsi), names=s.wsi[:5]))
    if s.raw_scan and not s.groups: out.append(Note("raw_scan", p))
    if lnk: out.append(Note("lnk", p, len(lnk), names=lnk[:5]))
    if s.n_entries == 0:
        out.append(Note("empty", p))
    elif not (s.groups or s.outputs or s.wsi or s.raw_scan or s.other_tiffs or s.subdirs or s.truncated):
        out.append(Note("no_images", p))
    if s.truncated and not s.groups: out.append(Note("large_dir", p, s.n_entries))
    if s.leftovers: out.append(Note("leftovers", p, len(s.leftovers), names=s.leftovers[:5]))
    if s.links: out.append(Note("symlink_skipped", p, s.links))
    return out


def natural_key(s: str) -> Tuple:
    """Case-insensitive natural sort key ("-2" before "-10"); the raw string breaks ties."""
    return tuple(int(t) if i % 2 else t.casefold() for i, t in enumerate(_DIGITS.split(s))), s


def norm_key(name: str) -> str:
    """NFC + casefold; used for every comparison of output names, on every OS."""
    return unicodedata.normalize("NFC", name).casefold()


def path_key(path: str) -> str:
    """A path as the filesystem compares it: normpath, casefolded (NFC) on Windows and macOS, exact elsewhere."""
    path = os.path.normpath(path)
    return norm_key(path) if _CASE_INSENSITIVE_FS else path


def same_path(a: str, b: Optional[str]) -> bool:
    """normpath equality; case-insensitive on Windows and macOS."""
    return bool(a and b) and path_key(a) == path_key(b)


def sanitize_stem(stem: str) -> str:
    """§3.7: a file-name stem that is legal on Windows, macOS and Linux."""
    s = _ILLEGAL.sub("_", unicodedata.normalize("NFC", stem)).rstrip(". ")
    s = ("_" + s if s.split(".", 1)[0].casefold() in _RESERVED else s) or "slide"
    data = s.encode("utf-8")
    return data[:STEM_BYTES - 8].decode("utf-8", errors="ignore") + "~" + hashlib.sha1(data).hexdigest()[:6] \
        if len(data) > STEM_BYTES else s


def output_stem(c: Candidate, naming: str) -> str:
    """§3.7. naming: "folder" (default, v1) or "slide"."""
    base = c.prefix if c.kind != "legacy_all" and (c.multi_slide_folder or naming == "slide") else c.folder_name
    return sanitize_stem(f"{base}_{c.suffix}" if c.kind == "other_suffix" else base)


def tile_name(c: Candidate) -> str:
    """The slide's name from its tiles: the prefix ('<prefix>_<suffix>' for other products); '' for legacy_all."""
    if c.kind == "legacy_all":
        return ""
    return c.prefix if c.kind != "other_suffix" else f"{c.prefix}_{c.suffix}"


def engine_image_name(c: Candidate) -> Optional[str]:
    """§3.7: OME Image@Name; None keeps the engine default (the folder name)."""
    return (tile_name(c) or None) if c.multi_slide_folder else None


def _stem_key(stem: str) -> str:
    return norm_key(stem + OME_SUFFIX)


def _levels(folder: str) -> List[str]:
    """Folder names from the folder itself up to the top."""
    out = []
    while os.path.basename(folder):
        out.append(os.path.basename(folder))
        folder = os.path.dirname(folder)
    return out


def _differing_folders(folders: Sequence[str]) -> List[str]:
    """For each folder of a group: the name of its nearest folder (its own first, then upwards) that no other
    member has at that level, else the nearest that some other member doesn't have; '' if none (the paths differ
    only in case). Folder-named rows have equal own folder names, so theirs is found one level up
    (batchA/Slide9, batchB/Slide9 -> 'batchA', 'batchB')."""
    levels = [[norm_key(n) for n in _levels(f)] for f in folders]
    counts = [Counter(ls[i] if i < len(ls) else None for ls in levels) for i in range(max(map(len, levels)))]
    out = []
    for f, ls in zip(folders, levels):
        names = _levels(f)
        i = next((i for i, k in enumerate(ls) if counts[i][k] == 1), None)
        i = next((i for i, k in enumerate(ls) if counts[i][k] < len(levels)), None) if i is None else i
        out.append("" if i is None else names[i])
    return out


def dedupe_stems(items: Sequence[Tuple[str, str, str, str]],
                 fixed: Optional[Mapping[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """items = (row_id, stem, folder, tile stem): a row's automatic output stem, its folder and its sanitised tile
    name ('' if none). fixed: row_id -> a name that must not change (typed by the user, or of a slide that has
    started). Returns row_id -> (stem, how) for every item; how is '' (kept), 'tiles' or 'suffix'. §3.7 and D1:
    - a stem that only one row has is kept, so the v1 names of unique folders keep matching;
    - rows whose stems collide (norm_key) form a group and NONE keeps the bare stem, so an old '<stem>.ome.tif' is
      never attributed to one of them. A member is named after its tiles when that name is its own (no other row's
      stem, no other member's tile name), else after its nearest differing folder ('Slide9__batchB'). Names
      therefore don't depend on which folders sort first, and a duplicate renames only itself and its twin;
    - whatever still clashes gets a stable suffix from the exact source identity, never a numbered position.
    A group's fixed members keep their names; 'tiles' then only says the group is named after its tiles."""
    fixed = fixed or {}
    groups: Dict[str, list] = defaultdict(list)
    for it in items:
        groups[_stem_key(it[1])].append(it)
    shared = {k: sorted(g, key=lambda it: (it[2], it[0])) for k, g in groups.items() if len(g) > 1}
    wanted = Counter(_stem_key(it[3]) for g in shared.values() for it in g if it[3])
    taken = {_stem_key(s) for s in fixed.values()}
    out: Dict[str, Tuple[str, str]] = {}

    sources = {it[0]: it[2] + "\0" + it[0] for it in items}

    def give(row_id: str, want: str, how: str) -> None:
        want = sanitize_stem(want)
        name, key, n = want, _stem_key(want), 0
        while key in taken:
            source = sources[row_id] + (f"\0{n}" if n else "")
            suffix = "__" + hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
            base = want.encode("utf-8")[:STEM_BYTES - len(suffix)].decode("utf-8", errors="ignore")
            name, n = base + suffix, n + 1
            key = _stem_key(name)
        taken.add(key)
        out[row_id] = (name, how or ("suffix" if name != want else ""))

    singles = [g[0] for g in groups.values() if len(g) == 1]
    clash = sorted((it for it in singles if it[0] not in fixed and _stem_key(it[1]) in taken),
                   key=lambda it: (it[2], it[0]))
    for it in singles:
        if it[0] in fixed:
            out[it[0]] = (fixed[it[0]], "")
        elif _stem_key(it[1]) not in taken:
            give(it[0], it[1], "")
    taken.update(shared)                             # a colliding stem is given to nobody
    for it in clash:
        give(it[0], it[1], "suffix")
    for key in sorted(shared):
        g = shared[key]
        own = [bool(it[3]) and wanted[_stem_key(it[3])] == 1 and _stem_key(it[3]) not in groups for it in g]
        folders = _differing_folders([it[2] for it in g]) if not all(own) else [""] * len(g)
        proposals = [sanitize_stem(f"{it[1]}__{sanitize_stem(d)}") if d else it[1]
                     for it, d in zip(g, folders)]
        counts = Counter(_stem_key(p) for p, mine in zip(proposals, own) if not mine)
        taken.update(k for k, count in counts.items() if count > 1)
        for it, mine, proposal in zip(g, own, proposals):
            if it[0] in fixed:
                out[it[0]] = (fixed[it[0]], "tiles" if mine else "")
            elif mine:
                give(it[0], it[3], "tiles")
            else:
                give(it[0], proposal, "suffix")
    return out


def temp_subdir(temp_dir: str, out_dir: str) -> str:
    """The custom temp folder's private subfolder for one save folder: '<temp_dir>/.yfs-<12 hex>'. The O_EXCL claim
    of '<out_dir>/.stitcher-in-progress/<name>' then also guards the scratch file of that name (no two runs or
    instances can share one memmap file)."""
    key = os.path.normcase(os.path.abspath(out_dir))
    return os.path.join(temp_dir, TEMP_SUBDIR_PREFIX + hashlib.sha1(key.encode("utf-8", "surrogateescape")).hexdigest()[:12])


def pyramid_levels(h: int, w: int) -> int:
    """Engine formula: halve until max(h, w) <= 256."""
    n = 0
    while max(h, w) > 256:
        h, w, n = h // 2, w // 2, n + 1
    return n


def estimate(c: Candidate, h: TileHeader, available_ram_bytes: int) -> Estimate:
    """§3.9: canvas size from the tile-centre coordinates in the names and the first tile's header."""
    xs, ys = [x for x, _ in c.coords], [y for _, y in c.coords]
    w = int(round((max(xs) - min(xs)) / (h.um_per_px_x or 0.5))) + h.tile_w
    hgt = int(round((max(ys) - min(ys)) / (h.um_per_px_y or 0.5))) + h.tile_h
    item = np.dtype(h.dtype).itemsize
    raw = h.n_channels * hgt * w * item
    cov = min(1.0, len(c.coords) * h.tile_w * h.tile_h / float(w * hgt))
    mode = "ram" if raw / 1024 ** 3 + RAM_HEADROOM_GB < available_ram_bytes / 1024 ** 3 else "disk"
    return Estimate(w, hgt, h.n_channels, item, raw, int(raw * OUT_TYPICAL_FACTOR), int(raw * OUT_WORST_FACTOR),
                    cov, mode)


def human_bytes(n: Optional[int], approx: bool = False) -> str:
    """'17.7 GB' (binary units labelled like Explorer), '~9.4 GB' when approx, '' for None."""
    if n is None:
        return ""
    units, v, i = ("B", "KB", "MB", "GB", "TB", "PB"), float(n), 0
    while v >= 1024 and i < len(units) - 1:
        v, i = v / 1024, i + 1
    return ("~" if approx else "") + (f"{v:.0f} {units[i]}" if i < 3 or v >= 100 else f"{v:.1f} {units[i]}")


def human_duration(seconds: float) -> str:
    """'45 s', '7 min', '1 h 12 min'."""
    s = max(0, int(round(seconds)))
    h, m = divmod(int(round(s / 60.0)), 60)
    return f"{s} s" if s < 60 else f"{m} min" if not h else f"{h} h {m} min" if m else f"{h} h"
