"""The plan: the single source of truth for what will be stitched (FINAL_SPEC §2.3, §2.4.1, §3.7-§3.10, §5.5.5).

Rows change only through Plan methods, under its RLock: the UI calls them on the main thread, the runner thread
calls begin_run / claim_next / set_run_text / finish / end_run. Mutators return the row_ids that changed. Inside
`with plan.batch() as changed:` they return [] and the whole block costs one recompute, after which `changed` holds
the row_ids. Facts that concern one row only (a header, a tick, a verdict) re-derive just that row.
Discovery notes become rows of section "note", shown at the end of *Not included* (§2.3.8).
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import APP_NAME
from .model import (CLOUD, NETWORK, Candidate, Estimate, Issue, Job, Note, OutputIndex, OutputInfo, RowOutcome,
                    TileHeader)
from .names import (COMPLETE_SUFFIX, DUPLICATE_RATIO, FEW_TILES, OME_SUFFIX, OWNER_SUFFIX, PARTIAL_DIR,
                    RECENT_CHANGE_S, SIBLING_MIN, SIBLING_RATIO, SPARSE_COVERAGE, SPARSE_MIN_BYTES, STALE_HEX,
                    STALE_INFIX, WIDE_SAMPLE, WINDOWS_MAX_PATH, dedupe_stems, engine_image_name, estimate, human_bytes,
                    natural_key, norm_key, output_stem, path_key, rel_folder, same_path, sanitize_stem, temp_subdir,
                    tile_name)

_log = logging.getLogger(__name__)
SECTIONS = ("new", "existing", "excluded")
RUN_STATUSES = frozenset({"waiting", "running", "done", "done_gaps", "done_generic", "failed", "skipped",
                          "cancelled", "finalize"})
STITCHED = frozenset({"done", "done_gaps", "done_generic", "finalize"})
GIB = 1024 ** 3
_AGGREGATED = {"raw_scan": "raw_scan", "not_searched_depth": "not_searched_depth", "system_skipped": "system_skipped",
               "symlink_skipped": "symlink_skipped", "no_images": "no_images", "empty": "no_images", "lnk": "lnk",
               "not_mounted": "not_mounted"}
_COUNT_SUMS = frozenset({"not_searched_depth", "system_skipped", "symlink_skipped", "lnk", "not_mounted"})  # else: 1
_NOTE_ORDER = ("outputs", "unmatched", "wsi", "raw_scan", "wide_pruned", "not_searched_depth", "large_dir",
               "system_skipped", "symlink_skipped", "cloud_skipped", "not_mounted", "no_images", "lnk", "no_access",
               "unresponsive", "error")
_PER_KIND_ROWS = 50
_UNMATCHED = "note|unmatched|"                  # D1 note rows: an old output that several slides could own
_RUN_TEXT = {"done": "Done", "done_gaps": "Done with gaps", "done_generic": "Done · generic channel names",
             "failed": "Failed", "skipped": "Skipped", "cancelled": "Stopped — will start again next time",
             "finalize": "Finished, but the old file is in use"}   # a run status that lost its outcome (MW-6)
_WONT = {"unfinished": "unfinished last time", "incomplete": "the existing file looks incomplete",
         "ready": "finished last time, not moved into place yet"}   # unticked: what would have happened (FS-18)
_SUMMARY = f"{APP_NAME}_summary_YYYYmmdd-HHMMSS.csv"   # the runner's summary table (§2.4.7), as long as a real one


@dataclass
class Flag:
    code: str; severity: str; text: str        # §2.3.4


@dataclass
class PlanRow:                                  # mutated only through Plan methods (lock held)
    row_id: str
    cand: Candidate
    section: str = "new"                        # new | existing | excluded | note
    status: str = "new"                         # §2.3.3 planning status, then run status
    checked: bool = False
    checkable: bool = True
    user_touched: bool = False
    output_stem: str = ""
    output_edited: bool = False
    image_name: Optional[str] = None
    header: Optional[TileHeader] = None
    header_error: Optional[str] = None
    est: Optional[Estimate] = None
    existing: Optional[OutputInfo] = None       # the final in out_dir (slide naming: maybe its v1 folder-named file)
    existing_verdict: Optional[str] = None      # None | checking | valid | incomplete | unreadable | unchecked
    unfinished: Optional[OutputInfo] = None
    channel_override: Optional[Tuple[str, ...]] = None
    generic_accepted: bool = False
    flags: List[Flag] = field(default_factory=list)
    run_text: str = ""                          # transient progress text while running
    outcome: Optional[RowOutcome] = None
    note: Optional[Note] = None                 # for section == "note" rows
    plan_status: str = "new"                    # the planning status; run statuses never overwrite it (D11)
    replaces: bool = False                      # stitching it replaces a final that existed at planning (Job.replace)
    existing_detail: str = ""                   # why the existing file couldn't be checked
    existing_gaps: int = 0                      # tiles that were missing when the existing file was made
    ready: Optional[OutputInfo] = None          # a complete partial that only waits to be moved into place (D3)
    name_taken: bool = False                    # the user typed the name of a file that exists (§2.3.5)
    header_retryable: bool = True               # a Try again can help: an I/O error or timeout, not a bad tile
    busy: bool = False                          # another stitcher is writing this slide's partial right now


@dataclass(frozen=True)
class PlanEnv:                                  # inputs to issues(); built by MainWindow
    source: Optional[str]; out_dir: Optional[str]; scanning: bool; paused: bool
    out_index: Optional[OutputIndex]; out_index_pending: bool; write_error: Optional[str]
    temp_free: Optional[int]; temp_lane_class: Optional[str]; temp_same_volume: bool
    windows: bool; available_ram: int
    unsearched: int = 0                         # folders the search couldn't list (ScanFinished.failed)
    windows_long_paths: bool = False            # LongPathsEnabled: Windows paths may be longer than 259 characters
    temp_dir: Optional[str] = None              # the custom temporary folder (its scratch files count for long_path)


@dataclass(frozen=True)
class Totals:                                   # of the rows Start would run (Plan.pending)
    ticked: int; new: int; replace: int; existing: int; excluded: int; tiles: int
    out_typical: int; out_worst: int; unknown_estimates: int; disk_mode: int; max_temp_bytes: int
    unsized: int = 0                            # rows that can't be estimated (legacy_all, placeholders, errors)


def default_state(r: PlanRow) -> Tuple[str, str, bool, bool]:
    """§3.10 with D2: (section, status, checked, checkable) from the facts known about a candidate row. Only an
    existing file that looks incomplete is redone by default; one that couldn't be checked stays, unticked, and so
    does a file whose name the user typed (§2.3.5), whatever its check found. A slide whose partial another
    stitcher is writing ('busy') can't be ticked: the runner would skip it."""
    c = r.cand
    sec = "excluded" if c.kind in ("other_suffix", "unsuffixed", "legacy_all") else "new"
    if r.header and r.header.n_channels == 1: return sec, "unsupported", False, False   # R24: beats user_touched
    if r.busy: return sec, "busy", False, False
    if sec == "excluded": return "excluded", c.kind, False, True
    if r.ready: return "new", "ready", True, True
    if r.unfinished: return "new", "unfinished", True, True
    if r.existing and r.existing_verdict == "incomplete" and not r.name_taken: return "new", "incomplete", True, True
    if r.existing and r.existing_verdict in ("unreadable", "unchecked"): return "existing", "unchecked", False, True
    if r.existing:
        known = r.existing_verdict in ("valid", "incomplete")
        return "existing", "existing" if known else "existing_checking", False, True
    if r.header_error: return "new", "header_error", False, True
    if len(c.tile_names) < FEW_TILES: return "new", "few_tiles", False, True
    if r.est and r.est.coverage < SPARSE_COVERAGE and r.est.raw_bytes > SPARSE_MIN_BYTES:
        return "new", "sparse", False, True
    return "new", "new", True, True


def display_name(c: Candidate, full: bool = False) -> str:
    """Folder name for single-slide folders, prefix for multi-slide folders (§2.3.1); `full` adds the tile hint."""
    name = c.prefix if c.multi_slide_folder else c.folder_name
    name += f" ({c.suffix})" if c.kind == "other_suffix" else ""
    hint = full and not c.multi_slide_folder and c.prefix and norm_key(c.prefix) != norm_key(c.folder_name)
    return f"{name} (tiles: {c.prefix})" if hint else name


def parent_rel(c: Candidate) -> str:
    """The *Folder* column: the parent folder relative to the source ("" for the top)."""
    return c.rel_folder if c.multi_slide_folder else ("" if c.rel_folder == "." else os.path.dirname(c.rel_folder))


def top_warning(r: PlanRow) -> Optional[Flag]:
    """The flag whose text replaces "New file" on a ticked new row (§2.3.3)."""
    return next((f for f in r.flags if f.severity == "warn"), None)


def _n(k: int, one: str, many: Optional[str] = None) -> str:
    return f"{k:,} {one if k == 1 else (many or one + 's')}"


def _examples(names: Sequence[str], total: int) -> str:
    return ", ".join(names[:3]) + (", …" if total > 3 else "")


def _day(mtime_ns: int) -> str:
    t = time.localtime(mtime_ns / 1e9)
    return f"{t.tm_mday} {time.strftime('%b', t)}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _final_key(stem: str) -> str:
    return norm_key(stem + OME_SUFFIX)


def _other_name(r: PlanRow) -> bool:
    """The existing file is the slide's old folder-named output, not its own final name (slide naming, DS-4)."""
    return r.existing is not None and norm_key(r.existing.name) != _final_key(r.output_stem)


def _place(path: str) -> str:
    """A short place name for messages: the folder's own name ('Stitched'), or the path of a drive root."""
    return os.path.basename(os.path.normpath(path)) or path


def _sibling_group(c: Candidate) -> str:
    return c.folder if c.multi_slide_folder else os.path.dirname(c.folder)


def _recent(c: Candidate, now: float) -> Optional[Flag]:
    if c.dir_mtime is None or now - c.dir_mtime >= RECENT_CHANGE_S:
        return None
    mins = max(1, int(round((now - c.dir_mtime) / 60)))
    return Flag("recent_change", "warn", f"Folder changed {mins} min ago — is inForm still exporting?")


def _note_examples(kind: str, n: Note, rel: str) -> Tuple[str, ...]:
    """The examples one more note adds to an aggregated row."""
    if kind == "not_mounted" and n.names:
        return tuple(name if rel == "." else f"{rel}/{name}" for name in n.names)
    if kind in ("system_skipped", "lnk"):
        return n.names
    return (rel.split("/")[0],) if kind == "no_images" else (rel,)


def row_text(r: PlanRow) -> str:
    """The *What will happen* text of a row (§2.3.3 planning and run statuses, §2.3.8 notes)."""
    st, c = r.status, r.cand
    if r.section == "note": return _note_text(r)
    if r.outcome is not None and st == r.outcome.status: return r.outcome.message
    if st == "running": return r.run_text or "Running…"
    if st == "waiting": return "Waiting" if r.checked else "Won't be stitched"
    if st in _RUN_TEXT: return _RUN_TEXT[st]
    if st in ("new", "few_tiles", "sparse") and r.checked:
        warn = top_warning(r)
        return warn.text if warn else f"New file ({r.existing.name} stays as it is)" if _other_name(r) else "New file"
    if st == "new": return "Won't be stitched"
    if st in _WONT and not r.checked: return f"Won't be stitched ({_WONT[st]})"
    if st == "few_tiles": return f"Only {_n(len(c.tile_names), 'tile')} — stamps or a test? Tick to include"
    if st == "sparse":
        return (f"Sparse: {len(c.tile_names)} fields cover {r.est.coverage:.0%} of a {human_bytes(r.est.raw_bytes)} "
                f"canvas. Tick to include")
    if st == "header_error":                     # UD4: reading a bad tile again won't help, a good copy will
        detail = (r.header_error or "").rstrip(".")
        return f"Can't read the first tile ({detail}). Right-click → Try again" if r.header_retryable else \
            f"Can't use the first tile ({detail}). Copy it again, then Rescan"
    if st == "ready":
        return f"Finished last time — {'the old file was in use; ' if r.existing else ''}will move the new image " \
               f"into place"
    if r.existing is not None and st in ("unchecked", "replace", "existing", "existing_checking"):
        return _existing_text(r)
    if r.checked and st in ("other_suffix", "unsuffixed", "legacy_all"):      # ticked by the user
        return {"other_suffix": f"Will be stitched ({c.suffix} tiles, not unmixed components)",
                "unsuffixed": 'Will be stitched (tiles without "_component_data" in the name)',
                "legacy_all": "Will be stitched (every TIFF in this folder, like the old version)"}[st]
    return {"unfinished": "Unfinished last time — will start again",
            "busy": "Another stitcher is writing it now — Rescan when it's done",
            "incomplete": "The existing file looks incomplete — will redo it",
            "unsupported": "1-channel images can't be stitched yet (engine limitation)",
            "other_suffix": f"{c.suffix} tiles, not unmixed components. Tick to stitch anyway",
            "unsuffixed": 'Tiles without "_component_data" in the name. Tick to stitch anyway',
            "legacy_all": "Every TIFF in this folder, like the old version. Tick to stitch anyway"}.get(st, st)


def _existing_text(r: PlanRow) -> str:
    """Rows of the *Already stitched* section: what is known about the file of their name."""
    e, st, other, gaps = r.existing, r.status, _other_name(r), r.existing_gaps
    if st == "unchecked":
        detail = r.existing_detail.strip().splitlines()[0].rstrip(".") if r.existing_detail.strip() else ""
        return (f"Couldn't check {e.name if other else 'the existing file'}" + (f" ({detail})" if detail else "") +
                f" — tick to {'stitch it again' if other else 'replace it'}")
    if st == "replace": return f"Will replace the file from {_day(e.mtime_ns)}"
    if r.name_taken: return "A file with this name already exists — tick to replace it"
    if st == "existing_checking":
        return f"Already stitched as {e.name} — checking it's complete…" if other else \
            "Already stitched — checking it's complete…"
    if other: return f"Already stitched as {e.name} ({human_bytes(e.size)}, {_day(e.mtime_ns)})"
    if gaps:
        return f"Already stitched — {_n(gaps, 'tile')} {'was' if gaps == 1 else 'were'} missing when it was made. " \
               f"Tick to redo it"
    return f"Skipped: already stitched ({human_bytes(e.size)}, {_day(e.mtime_ns)})"


def _note_text(r: PlanRow) -> str:
    n, c = r.note, r.cand
    where, ex = c.folder_name if c.rel_folder == "." else c.rel_folder, _examples(n.names, n.count)
    if r.row_id.endswith("|*"):
        return f"…and {_n(n.count, 'more folder')} like this ({ex})"
    if n.kind == "not_mounted" and n.detail == "pseudo":
        return f"{_n(n.count, 'system folder')} skipped ({ex})"
    failed = f"; {n.detail} of them couldn't be searched" if n.kind == "wide_pruned" and n.detail else ""
    return {
        "outputs": f"{_n(n.count, 'stitched image')} in {where} ({ex}) — never used as tiles",
        "unmatched": f"{n.names[0] if n.names else where}: may belong to one of {n.count:,} slides named "
                     f"'{n.detail}'; not matched",
        "wsi": f"{_n(n.count, 'whole-slide image')} from another program (…_stitch.tif) in {where}",
        "raw_scan": f"{_n(n.count, 'raw scan')} (.qptiff) — unmix in inForm and export component data first ({ex})",
        "wide_pruned": f"{where}: {n.count:,} subfolders not searched (no slides found in the first {WIDE_SAMPLE} "
                       f"folders{failed}). Right-click → Search these folders too",
        "not_searched_depth": f"{_n(n.count, 'folder')} below {c.depth} levels not searched. "
                              f"Right-click → Search deeper",
        "large_dir": f"{where}: {n.count:,}+ items without tiles — not searched further",
        "system_skipped": f"System folders skipped ({ex})",
        "symlink_skipped": f"{_n(n.count, 'shortcut or link', 'shortcuts or links')} not followed",
        "cloud_skipped": f"{c.folder_name} (cloud) was not searched — choose it directly to search it",
        "not_mounted": f"{_n(n.count, 'folder')} not searched: not mounted or not connected ({ex}). "
                       f"Open {'it' if n.count == 1 else 'them'}, then Rescan",
        "no_images": f"{_n(n.count, 'folder')} without images ({ex})",
        "lnk": f"Windows shortcut (can't be followed): {ex}",
        "no_access": f"No access: {where}",
        "unresponsive": f"Didn't answer: {where}. Right-click → Search this folder again",
    }.get(n.kind, f"Couldn't read {where}: {n.detail}")


def _channel_detail(h: TileHeader) -> str:
    if h.channel_problem == "missing" or not h.channels_extracted:
        return "no names in the tiles"
    if h.channel_problem == "count":
        return f"{len(h.channels_extracted)} names for {h.n_channels} channels"
    return "{} ×{}".format(*Counter(h.channels_extracted).most_common(1)[0])


def _snap(r: PlanRow) -> tuple:
    return (r.section, r.status, r.checked, r.checkable, r.output_stem, r.image_name, r.existing, r.existing_verdict,
            r.existing_detail, r.existing_gaps, r.unfinished, r.ready, r.busy, r.name_taken, r.plan_status, r.replaces,
            r.header, r.header_error, r.header_retryable, r.est, r.channel_override, r.generic_accepted, r.run_text,
            r.outcome, r.note, tuple(r.flags))


def _mutator(method: Callable) -> Callable:
    """Run a Plan method under the lock, recompute every row and return the row_ids whose state changed (or that
    appeared or went away). A method returns None when it changed nothing. Inside batch() the recompute waits for
    the end of the batch."""
    @functools.wraps(method)
    def wrapper(self: "Plan", *args: Any, **kwargs: Any) -> List[str]:
        with self.lock:
            if self._batch is not None:
                self._dirty = method(self, *args, **kwargs) is not None or self._dirty
                return []
            before = {k: _snap(r) for k, r in self._rows.items()}
            if method(self, *args, **kwargs) is None:
                return []
            self._recompute()
            return self._changed_since(before)
    return wrapper


class Plan:
    """Single source of truth for what will run. RLock-protected; UI mutates on the main thread,
    the runner calls claim_next/finish/set_run_text. All mutators return the row_ids that changed."""

    def __init__(self, source: str, out_dir: Optional[str], naming: str = "folder",
                 carry: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        """carry = §2.3.9 choices by row_id (from carry_state() of the plan this one replaces)."""
        self.lock = threading.RLock()
        self.source, self.out_dir, self.naming = source, out_dir, naming
        self._carry = {k: dict(v) for k, v in (carry or {}).items()}
        self._rows: Dict[str, PlanRow] = {}
        self._notes: Dict[str, Dict[tuple, Note]] = {}   # note row id -> the discovery notes it stands for
        self._per_kind: Counter = Counter()            # per-folder note rows of each kind
        self._shown: Counter = Counter()               # display name -> candidate rows that have it (label)
        self._order: Dict[str, tuple] = {}            # row_id -> its display-order key (§2.3.1), fixed per row
        self._index: Optional[OutputIndex] = None
        # (normcase path, size, mtime_ns) -> (verdict, detail, gaps)
        self._verdicts: Dict[tuple, Tuple[str, str, int]] = {}
        # while running: row_id -> (status, outcome) before Start
        self._run: Optional[Dict[str, Tuple[str, Optional[RowOutcome]]]] = None
        self._claimed = 0
        self._batch: Optional[Dict[str, tuple]] = None   # inside batch(): the row snapshots from its start
        self._dirty = False                             # inside batch(): a recompute is due at its end
        # what the last recompute found, so that a single row can be derived again on its own
        self._stems: Dict[str, Tuple[str, str, str]] = {}   # row_id -> (naming, automatic stem, tile stem)
        self._renamed: Dict[str, Tuple[str, str]] = {}      # row_id -> (automatic stem, how it was made unique)
        self._dups: Dict[str, List[PlanRow]] = {}
        self._medians: Dict[str, float] = {}               # sibling group -> median tile count
        self._fallbacks: Dict[str, str] = {}               # slide naming: row_id -> key of its v1 output (DS-4)
        self._unmatched: set = set()                      # ambiguous outputs: no source ownership is known

    # ---- building (main thread)
    @_mutator
    def add_candidate(self, c: Candidate) -> List[str]:
        """Ignores duplicate keys; recomputes names, de-duplication and flags."""
        if c.key in self._rows:
            return None
        r = self._rows[c.key] = PlanRow(c.key, c)
        self._order[c.key] = (natural_key(c.rel_folder), natural_key(c.prefix), c.suffix)
        self._shown[display_name(c)] += 1               # candidate rows are never removed
        cv = self._carry.get(c.key, {})
        if "checked" in cv:
            r.checked, r.user_touched = bool(cv["checked"]), True
        if cv.get("output_stem"):
            r.output_stem, r.output_edited = cv["output_stem"], True
            r.name_taken = bool(cv.get("name_taken"))
        r.channel_override = tuple(cv["channel_override"]) if cv.get("channel_override") else None
        r.generic_accepted = bool(cv.get("generic_accepted"))
        return [c.key]

    def add_note(self, n: Note) -> List[str]:
        """Group a discovery note into a *Not included* row (§2.3.8); leftovers are shown by step 2 instead.
        Aggregated rows carry a summary Note (total count, up to 5 examples), updated in O(1) per note."""
        if n.kind == "leftovers":
            return []
        with self.lock:
            agg = _AGGREGATED.get(n.kind)
            rid = f"note|{agg}" if agg else f"note|{n.kind}|{n.path}"
            if not agg and rid not in self._notes and self._per_kind[n.kind] >= _PER_KIND_ROWS:
                rid = f"note|{n.kind}|*"                 # more than 50 folders of one kind: one row for the rest
            seen = self._notes.setdefault(rid, {})     # (kind, path) -> Note, in arrival order
            if (n.kind, n.path) in seen:
                return []
            seen[(n.kind, n.path)] = n
            rel, old = rel_folder(n.path, self.source), self._rows.get(rid)
            depth = 0 if rel == "." else rel.count("/") + 1
            if agg or rid.endswith("|*"):              # one row for many folders: totals and examples
                kind, folder = agg or n.kind, self.source
                names = tuple(dict.fromkeys((old.note.names if old else ()) + _note_examples(kind, n, rel)))[:5]
                count = (old.note.count if old else 0) + (max(n.count, 1) if kind in _COUNT_SUMS else 1)
                detail = n.detail if old is None or old.note.detail == n.detail else ""   # kept while all agree
                n, depth = Note(kind, self.source, count, detail, names), old.cand.depth if old else depth
            else:
                self._per_kind[n.kind] += 1
                folder = n.path
            cand = Candidate(rid, folder, rel_folder(folder, self.source),
                             os.path.basename(os.path.normpath(folder)) or folder, "", "", "note", (), (), False,
                             False, depth)                 # depth: "below N levels" of Search deeper
            self._rows[rid] = PlanRow(rid, cand, "note", "note", False, False, note=n)
            self._order[rid] = (_NOTE_ORDER.index(n.kind) if n.kind in _NOTE_ORDER else 99, rid.endswith("|*"),
                                natural_key(cand.rel_folder))
            return [rid]

    def note_paths(self, row_id: str) -> List[str]:
        """The folders behind a note row (e.g. every folder for *Search deeper*)."""
        with self.lock:
            return [path for _, path in self._notes.get(row_id, {})]

    def apply_header(self, row_id: str, header: Optional[TileHeader], error: Optional[str],
                     available_ram: int, retryable: bool = True) -> List[str]:
        with self.lock:
            r = self._rows.get(row_id)
            if r is None or r.section == "note":
                return []
            with self._rederive([r]) as changed:
                r.header, r.header_error, r.header_retryable, r.est = header, error, retryable, None
                if header is not None and r.cand.coords and not header.offline:
                    try:
                        r.est = estimate(r.cand, header, available_ram)
                    except (ValueError, ZeroDivisionError, TypeError):
                        r.est = None                     # an odd header: the size stays unknown
            return changed

    @_mutator
    def apply_output_index(self, idx: OutputIndex) -> List[str]:
        """§3.8 matching (ready partials -> ready, partials -> unfinished, or busy while another stitcher writes them,
        outputs -> existing_checking until verified)."""
        if self.out_dir is not None and same_path(idx.out_dir, self.out_dir):   # else: a stale index
            self._index = idx
        return []

    def apply_verification(self, path: str, verdict: str, detail: str, gaps: int = 0, size: Optional[int] = None,
                           mtime_ns: Optional[int] = None) -> List[str]:
        """R4 / D2: what the check of an existing output found: valid | incomplete | unreadable | unchecked (an
        OSError stopped the check). gaps: tiles that were missing when the file was made (from the ledger). size and
        mtime_ns: the version that was checked; a verdict about another version than the indexed one is ignored."""
        with self.lock:
            idx, key = self._index, norm_key(os.path.basename(path))
            info = idx.outputs.get(key) if idx is not None and same_path(os.path.dirname(path), idx.out_dir) else None
            if info is None or (size is not None and (size, mtime_ns) != (info.size, info.mtime_ns)):
                return []
            # its rows, and slides whose old folder-named file it is even while a verdict keeps it from them (DS-4)
            rows = [r for r in self._candidate_rows() if r.existing is info or self._fallbacks.get(r.row_id) == key]
            with self._rederive(rows) as changed:
                self._verdicts[self._verdict_key(info)] = (verdict, detail, gaps)
            return changed

    @_mutator
    def set_out_dir(self, out_dir: Optional[str]) -> List[str]:
        if not (out_dir and self.out_dir and same_path(out_dir, self.out_dir)):
            self._index = None
        self.out_dir = out_dir
        return []

    @_mutator
    def set_naming(self, naming: str) -> List[str]:
        self.naming = naming
        return []

    @_mutator
    def refresh(self) -> List[str]:
        """Derive every row again now, e.g. once a minute for time-based flags (recent_change); issues() doesn't."""
        return []

    # ---- user actions (main thread)
    def toggle(self, row_ids: Iterable[str], value: Optional[bool] = None) -> List[str]:
        """Tick or untick rows (value None: tick all unless all are ticked). Sets user_touched.
        During a run only waiting rows can be unticked (R10)."""
        with self.lock:
            rows = [self._rows[i] for i in row_ids if i in self._rows]
            if self._run is not None:
                rows, value = [r for r in rows if r.status == "waiting" and r.checked], False
            rows = [r for r in rows if r.checkable and r.section != "note"]
            if not rows:
                return []
            with self._rederive(rows) as changed:
                value = (not all(r.checked for r in rows)) if value is None else value
                for r in rows:
                    r.checked, r.user_touched = value, True
            return changed

    def toggle_section(self, section: str) -> List[str]:
        """Only the *New slides* section has a tick box (§2.3.1). It does nothing during a run (MW-2): unticking
        every waiting slide is the explicit Select ▾ → Untick all waiting."""
        with self.lock:
            if self._run is not None:
                return []
            rows = [r.row_id for r in self.rows("new") if r.checkable] if section == "new" else []
            return self.toggle(rows, not all(self._rows[i].checked for i in rows)) if rows else []

    def select(self, mode: str) -> List[str]:
        """all_new | none | not_stitched | invert | untick_smaller_copies | failed | untick_waiting."""
        with self.lock:
            if (self._run is not None) != (mode == "untick_waiting"):
                return []
            rows = [r for r in self.rows() if r.checkable and r.section != "note"]
            new, bad = [r for r in rows if r.section == "new"], ("failed", "skipped", "cancelled")
            on: List[PlanRow] = []
            off: List[PlanRow] = []
            if mode == "all_new": on = new
            elif mode == "none": off = rows
            elif mode == "not_stitched": on, off = new, [r for r in rows if r.section == "existing"]
            elif mode == "invert": on, off = [r for r in new if not r.checked], [r for r in new if r.checked]
            elif mode == "untick_waiting": off = [r for r in rows if r.status == "waiting"]
            elif mode == "failed":
                on = [r for r in rows if r.outcome and r.outcome.status in bad]
                off = [r for r in rows if r.checked and not (r.outcome and r.outcome.status in bad)]
            elif mode == "untick_smaller_copies":
                for group in self._duplicates(rows).values():
                    most = max(len(r.cand.tile_names) for r in group)
                    off += [r for r in group if len(r.cand.tile_names) * DUPLICATE_RATIO <= most]
            with self._rederive(on + off) as changed:
                for rows_, value in ((on, True), (off, False)):
                    for r in rows_:
                        r.checked, r.user_touched = value, True
            return changed

    def rename_output(self, row_id: str, stem: str) -> Optional[str]:
        """Error text, or None if accepted (§2.3.5). An empty stem restores the automatic name."""
        with self.lock:
            r, text = self._rows[row_id], stem.strip()
            if r.status in RUN_STATUSES and r.status != "waiting":
                return "This slide has started, so its name can't change now."
            text = next((text[:-len(s)].rstrip() for s in (".ome.tiff", ".ome.tif") if text.lower().endswith(s)), text)
            clean = sanitize_stem(text)
            if text and clean != text:
                return f"Some characters can't be used in file names. Suggested name: {clean}"
            key = _final_key(clean)
            if text and any(o is not r and _final_key(o.output_stem) == key for o in self._candidate_rows()):
                return "Another slide already uses this name."
            # the name it has (Enter or clicking away commits the editor unchanged, or only the case differs): what is
            # known about the file of that name, and the tick, stay as they are
            same = bool(text) and key == _final_key(r.output_stem)
            r.output_stem, r.output_edited = (clean, True) if text else (r.output_stem, False)
            if not same:
                r.name_taken = bool(text) and self._index is not None and key in self._index.outputs
                if r.name_taken:
                    r.checked, r.user_touched = False, False   # a file with this name exists: skipped unless ticked
            self._recompute()
            return None

    def set_channels(self, row_ids: Iterable[str], names: Optional[Sequence[str]]) -> List[str]:
        """names=None -> generic_accepted=True. Rows whose header has no channel problem are ignored (R26)."""
        with self.lock:
            rows = [r for r in filter(None, (self._rows.get(i) for i in row_ids)) if r.header and
                    r.header.channel_problem and not (r.status in RUN_STATUSES and r.status != "waiting")]
            with self._rederive(rows) as changed:
                for r in rows:
                    if names is None:
                        r.channel_override, r.generic_accepted = None, True
                    elif len(names) == r.header.n_channels:
                        r.channel_override, r.generic_accepted = tuple(names), False
            return changed

    def carry_state(self) -> Dict[str, Dict[str, Any]]:
        """For rebuilding after a rescan (§2.3.9). Ticks of rows that were stitched are not carried."""
        with self.lock:
            out = {}
            for r in self._candidate_rows():
                stitched = r.outcome is not None and r.outcome.status in STITCHED
                d = {"checked": r.checked} if r.user_touched and not stitched else {}
                d.update({"output_stem": r.output_stem} if r.output_edited else {})
                d.update({"name_taken": True} if r.name_taken and not stitched else {})
                d.update({"channel_override": r.channel_override} if r.channel_override else {})
                d.update({"generic_accepted": True} if r.generic_accepted else {})
                if d:
                    out[r.row_id] = d
            return out

    def rebuild(self, out_dir: Optional[str] = None) -> "Plan":
        """A new plan of the same slides, e.g. for another save folder after a batch (DS-15, §2.3.9): the same
        candidates, notes, headers and estimates, the user's choices carried over and no run state. out_dir None
        keeps this plan's folder and output index; for another folder the caller applies that folder's index."""
        with self.lock:
            p = Plan(self.source, self.out_dir if out_dir is None else out_dir, self.naming, self.carry_state())
            p._verdicts = dict(self._verdicts)
            with p.batch():
                for r in self._candidate_rows():
                    p.add_candidate(r.cand)
                    q = p._rows[r.row_id]
                    q.header, q.header_error, q.header_retryable = r.header, r.header_error, r.header_retryable
                    q.est = r.est
                for notes in self._notes.values():
                    for n in notes.values():
                        p.add_note(n)
                if p.out_dir and self.out_dir and same_path(p.out_dir, self.out_dir):
                    p._index = self._index
            return p

    # ---- queries (any thread, lock held briefly)
    def rows(self, section: Optional[str] = None) -> List[PlanRow]:
        """Display order (§2.3.1). rows("excluded") is the whole *Not included* section: excluded candidates
        (tickable first), then note rows; rows("note") gives the note rows alone."""
        with self.lock:
            cands = sorted(self._candidate_rows(), key=lambda r: (r.section == "excluded" and not r.checkable,) +
                           self._order[r.row_id])
            notes = sorted((r for r in self._rows.values() if r.section == "note"), key=lambda r: self._order[r.row_id])
            if section in ("note", None):
                return notes if section else [r for s in SECTIONS for r in cands if r.section == s] + notes
            return [r for r in cands if r.section == section] + (notes if section == "excluded" else [])

    def row(self, row_id: str) -> PlanRow:
        with self.lock:
            return self._rows[row_id]

    def ticked(self) -> List[PlanRow]:
        with self.lock:
            return [r for r in self.rows() if r.checked and r.section != "note"]

    def pending(self) -> List[PlanRow]:
        """Ticked rows not stitched in this plan: what Start would run (MW-7), in display order."""
        with self.lock:
            return [r for r in self.ticked() if not (r.outcome and r.outcome.status in STITCHED)]

    def label(self, r: PlanRow) -> str:
        """The slide's name in messages: its display name, or its output name when other slides share that name.
        Cheap enough for every event: the names are counted as the rows come."""
        with self.lock:
            name = display_name(r.cand)
            return name if self._shown[name] < 2 else r.output_stem

    def totals(self) -> Totals:
        """The run-bar totals (§2.4.1) of the pending rows. unknown_estimates: sizes still to come (header not read
        yet); unsized: rows whose size can't be estimated at all."""
        with self.lock:
            t, sections = self.pending(), Counter(r.section for r in self._rows.values())
            ests = [r.est for r in t if r.est is not None]
            disk = [e for e in ests if e.memory_mode == "disk"]
            later = sum(r.est is None and r.header is None and r.header_error is None and bool(r.cand.coords)
                        for r in t)
            return Totals(len(t), sections["new"], sum(r.replaces for r in t), sections["existing"],
                          sections["excluded"] + sections["note"], sum(len(r.cand.tile_names) for r in t),
                          sum(e.out_typical for e in ests), sum(e.out_worst for e in ests), later,
                          len(disk), max((e.raw_bytes for e in disk), default=0), len(t) - len(ests) - later)

    def issues(self, env: PlanEnv) -> List[Issue]:
        """§2.4.1: blocking issues first, then warnings, then information, about the rows Start would run; one line
        per code. It doesn't recompute the plan (MW-8)."""
        with self.lock:
            t, idx, out = self.pending(), env.out_index, []
            free = idx.free_bytes if idx is not None else None
            vol = (os.path.basename(idx.lane.rstrip("/\\")) or idx.lane) if idx is not None else ""

            def add(sev: str, code: str, text: str, rows: Iterable[PlanRow] = (), action: Optional[str] = None):
                out.append(Issue(sev, code, text, tuple(r.row_id for r in rows), action))

            label = self.label                           # slides that share a name are told apart by their output

            def names(rows: List[PlanRow]) -> str:
                return _examples([label(r) for r in rows], len(rows))

            def flagged(code: str) -> List[Tuple[PlanRow, Flag]]:
                return [(r, f) for r in t for f in r.flags if f.code == code]

            if not env.source: add("block", "no_source", "Choose the folder with your tiles.", action="browse_input")
            if not env.out_dir:
                add("block", "no_output", "Choose where to save the stitched images.", action="browse_output")
            elif env.out_index_pending: add("block", "checking_output", "Checking the save folder…")
            if env.paused: add("block", "paused", "Search paused.", action="resume_scan")
            elif env.scanning: add("block", "scanning", "Still searching…", action="stop_scan")
            if not t:
                old, new = self.rows("existing"), self.rows("new")
                busy = [r for r in new if r.status == "busy"]
                if self.ticked():                     # ticked, but all stitched in this plan (MW-7)
                    add("block", "none_ticked", "All ticked slides are done — Start a new batch to stitch more.")
                elif old and not new:
                    add("block", "none_ticked", "Nothing new to stitch — the slide already has a stitched file. Tick "
                        "it to replace it." if len(old) == 1 else f"Nothing new to stitch — all {len(old):,} slides "
                        f"already have stitched files. Tick one to replace it.")
                elif busy and not any(r.checkable for r in new):   # no new slide can be ticked: why, and what helps
                    which = label(busy[0]) if len(busy) == 1 else f"{_n(len(busy), 'slide')} ({names(busy)})"
                    add("block", "none_ticked", f"Nothing to stitch yet — another stitcher is writing {which}. "
                        f"Rescan when it's done.", action="rescan")
                else:
                    add("block", "none_ticked", "Tick at least one slide.")
            err = env.write_error or (idx.error.message if idx is not None and idx.error else None)
            if env.out_dir and err:
                add("block", "output_error", f"Can't save to {_place(env.out_dir)}: {err.strip().rstrip('.')}.",
                    action="browse_output")
            if env.out_dir and idx is not None and idx.pdir_problem:   # DS-10: the runner would fail every slide
                add("block", "pdir_problem", f"Can't save to {_place(env.out_dir)}: {_lower_first(idx.pdir_problem)}",
                    action="browse_output")
            if t and t[0].est and free is not None and t[0].est.out_typical > free:
                add("block", "no_space_first", f"Not enough space for even the first slide: needs "
                    f"{human_bytes(t[0].est.out_typical, True)}, {human_bytes(free)} free on {vol}.",
                    action="browse_output")
            if env.windows and not env.windows_long_paths and env.out_dir:
                long = [(n, r) for r, n in ((r, _longest_path(env.out_dir, r, env.temp_dir)) for r in t)
                        if n > WINDOWS_MAX_PATH]
                if long:
                    add("block", "long_path", f"Paths too long for Windows ({max(n for n, _ in long)} characters, "
                        f"{WINDOWS_MAX_PATH} allowed) for {names([r for _, r in long])}. Choose a shorter save "
                        f"folder or output name.", [r for _, r in long], "browse_output")
            # ---- warnings (the *Ready to stitch?* sheet)
            ch = [r for r, f in flagged("channels") if f.severity == "warn"]
            if len(ch) == 1:
                flag = next(f for f in ch[0].flags if f.code == "channels")
                add("warn", "channels", f"{label(ch[0])} needs a look: {_lower_first(flag.text)}.", ch, "fix_channels")
            elif ch:
                add("warn", "channels", f"{len(ch)} slides have channel-name problems: {names(ch)}.", ch,
                    "fix_channels")
            rep = [r for r in t if r.replaces]               # D2: every overwrite is confirmed on the sheet
            if rep:
                which = [label(r) + (" (looks incomplete)" if r.plan_status == "incomplete" else "") for r in rep]
                add("warn", "replace", f"{_n(len(rep), 'file')} will be replaced: {_examples(which, len(rep))}.", rep)
            if env.unsearched:
                add("warn", "unsearched", f"{_n(env.unsearched, 'folder')} couldn't be searched — slides in "
                    f"{'it' if env.unsearched == 1 else 'them'} are missing from this list.")
            if idx is not None and idx.missing_levels > 1 and idx.nearest_existing:
                add("warn", "missing_levels", f"{idx.missing_levels} new folders will be created inside "
                    f"{idx.nearest_existing}: {rel_folder(idx.out_dir, idx.nearest_existing)}", action="browse_output")
            ests = [r.est for r in t if r.est is not None]
            worst, typical = sum(e.out_worst for e in ests), sum(e.out_typical for e in ests)
            disk = [r for r in t if r.est is not None and r.est.memory_mode == "disk"]
            if (free is not None and worst > free) or (env.temp_free is not None and disk and
                                                       max(r.est.raw_bytes for r in disk) > env.temp_free):
                add("warn", "space_some", f"Space is tight: up to {human_bytes(worst, True)} needed, "
                    f"{human_bytes(free)} free on {vol}. Slides that don't fit will be skipped.")
            if idx is not None and idx.is_system_volume and free is not None and free - typical < 20 * GIB:
                add("warn", "system_disk", "This would nearly fill the disk your computer runs on (less than 20 GB "
                    "would be left).")
            if disk and env.temp_lane_class == NETWORK:
                add("warn", "temp_on_network", f"{_n(len(disk), 'slide')} need "
                    f"{human_bytes(sum(r.est.raw_bytes for r in disk), True)} of temporary space on a network drive "
                    f"(slow).", disk, "settings")
            now = time.time()                          # (the rows' flags age only through refresh())
            recent = [(r, f) for r, f in ((r, _recent(r.cand, now)) for r in t) if f is not None]
            if len(recent) == 1:
                add("warn", "recent_change", f"{label(recent[0][0])}: {_lower_first(recent[0][1].text)}",
                    [recent[0][0]])
            elif recent:
                add("warn", "recent_change", f"{len(recent):,} slide folders changed in the last "
                    f"{RECENT_CHANGE_S // 60} min — is inForm still exporting? ({names([r for r, _ in recent])})",
                    [r for r, _ in recent])
            big = flagged("legacy_big")
            if len(big) == 1:
                add("warn", "legacy_big", f"{label(big[0][0])}: {_lower_first(big[0][1].text)}", [big[0][0]])
            elif big:
                add("warn", "legacy_big", f"{len(big):,} ticked folders start with images over 2 GB — probably not "
                    f"tile folders ({names([r for r, _ in big])})", [r for r, _ in big])
            dups = list(self._duplicates(t).values())
            if len(dups) == 1:
                g = dups[0]
                where = [r.cand.rel_folder if r.cand.rel_folder != "." else "(top)" for r in g]
                add("warn", "duplicates_ticked", f"{g[0].cand.prefix} is ticked in {len(g)} places "
                    f"({_examples(where, len(g))}) — you'll get {len(g)} files.", g)
            elif dups:
                add("warn", "duplicates_ticked", f"{len(dups)} slides are ticked in more than one place — you'll get "
                    f"{sum(map(len, dups))} files ({_examples([g[0].cand.prefix for g in dups], len(dups))}).",
                    [r for g in dups for r in g])
            if idx is not None and (idx.lane_class == CLOUD or idx.cloud_synced):
                add("warn", "cloud_output", "The save folder is a cloud folder: stitched files upload in the "
                    "background.")
            if idx is not None and idx.lane_class == NETWORK:
                add("info", "network_output", "Saving over the network is slower than to a local disk.")
            return out

    def planned_output_keys(self) -> FrozenSet[str]:
        """norm_key of the final names of the rows Start would run."""
        with self.lock:
            return frozenset(_final_key(r.output_stem) for r in self.pending())

    def multi_parent(self) -> bool:
        """Show the *Folder* column? (rows come from more than one parent folder)"""
        with self.lock:
            return len({parent_rel(r.cand) for r in self._candidate_rows()}) > 1

    def existing_counts(self) -> Dict[str, int]:
        """For step 2 (§2.1.3): the slides (component rows) that have a file of their name in the save folder, by
        what is known about it: 'valid' (checked, or stitched in this plan), 'checking' (no verdict yet),
        'incomplete' and 'unchecked' (couldn't be checked). Not counted: a name the user typed onto someone else's
        file, and slides that will be redone or moved into place (unfinished, ready), whose old file is never
        checked. 'gaps' counts the valid images known to lack tiles."""
        with self.lock:
            out = dict.fromkeys(("valid", "checking", "incomplete", "unchecked", "gaps"), 0)
            for r in self._candidate_rows():
                if r.cand.kind != "component":
                    continue
                if r.outcome is not None and r.outcome.status in STITCHED - {"finalize"}:
                    out["valid"] += 1
                    out["gaps"] += r.outcome.status == "done_gaps"
                elif r.existing is not None and not (r.name_taken or r.unfinished or r.ready):
                    v = r.existing_verdict
                    out["unchecked" if v in ("unreadable", "unchecked") else v if v in out else "checking"] += 1
                    out["gaps"] += v == "valid" and r.existing_gaps > 0
            return out

    # ---- runner (runner thread)
    def begin_run(self) -> int:
        """Marks the pending rows 'waiting' (their status and outcome are kept for end_run); returns the number of
        jobs."""
        with self.lock:
            self._run, self._claimed = {}, 0
            for r in self.pending():                     # not again before the next rescan once stitched
                self._run[r.row_id] = (r.status, r.outcome)
                r.status, r.outcome, r.run_text = "waiting", None, ""
            return len(self._run)

    def claim_next(self, out_dir: str, temp_dir: Optional[str]) -> Optional[Job]:
        """First row in display order that is checked and 'waiting' -> status 'running'; returns a Job snapshot.
        Job.replace and finalize_only come from the planning intent, which retries keep (D11)."""
        with self.lock:
            r = next((r for r in self.rows() if r.checked and r.status == "waiting"), None)
            if r is None:
                return None
            r.status, self._claimed, c = "running", self._claimed + 1, r.cand
            final_name = r.existing.name if r.replaces and r.existing else r.output_stem + OME_SUFFIX
            return Job(r.row_id, self._claimed, c.folder, c.kind, c.prefix, c.suffix, self.label(r), c.tile_names,
                       out_dir, final_name, r.replaces, r.image_name, r.channel_override, r.est,
                       temp_dir, finalize_only=r.plan_status == "ready")

    def set_run_text(self, row_id: str, text: str) -> None:
        with self.lock:
            self._rows[row_id].run_text = text

    def finish(self, row_id: str, outcome: RowOutcome) -> None:
        with self.lock:
            r = self._rows[row_id]
            r.outcome, r.status, r.run_text = outcome, outcome.status, ""

    def end_run(self) -> None:
        """Rows that were never claimed get back their status and outcome from before Start (MW-6)."""
        with self.lock:
            for r in self._candidate_rows():
                if r.status == "waiting":
                    r.status, r.outcome = (self._run or {}).get(r.row_id, ("new", None))
            self._run = None
            self._recompute()

    # ---- internals
    def _candidate_rows(self) -> List[PlanRow]:
        return [r for r in self._rows.values() if r.section != "note"]

    def _changed_since(self, before: Mapping[str, tuple]) -> List[str]:
        """Rows whose state differs from the snapshots `before`, rows that appeared and rows that went away."""
        return [k for k, r in self._rows.items() if before.get(k) != _snap(r)] + \
            [k for k in before if k not in self._rows]

    @contextlib.contextmanager
    def batch(self) -> Iterator[List[str]]:
        """`with plan.batch() as changed:` one recompute for every mutation in the block (e.g. a UI tick's
        candidates and headers, MW-8); mutators inside return [], and `changed` holds the row_ids that changed once
        the block has ended. Nested blocks belong to the outermost one."""
        changed: List[str] = []
        with self.lock:
            if self._batch is not None:
                yield changed
                return
            self._batch, self._dirty = {k: _snap(r) for k, r in self._rows.items()}, False
            try:
                yield changed
            finally:
                before, self._batch = self._batch, None
                if self._dirty:
                    self._recompute()
                changed += self._changed_since(before)

    @contextlib.contextmanager
    def _rederive(self, rows: Sequence[PlanRow]) -> Iterator[List[str]]:
        """For facts that concern `rows` only (a header, a tick, a verdict): after the block those rows alone are
        derived again; the list then holds the ones whose state changed. A slide that started keeps its name, so
        when one leaves its run status (R24) every row is derived again. Inside batch() the batch recomputes."""
        changed: List[str] = []
        before = [(r, _snap(r), r.status in RUN_STATUSES) for r in rows] if self._batch is None else []
        yield changed
        if self._batch is not None:
            self._dirty = True
            return
        now = time.time()
        for r, _, _ in before:
            self._derive(r, now)
        if any((r.status in RUN_STATUSES) != started for r, _, started in before):
            snaps = {k: _snap(r) for k, r in self._rows.items()}   # the other rows are still as they were
            snaps.update((r.row_id, snap) for r, snap, _ in before)
            self._recompute()
            changed += self._changed_since(snaps)
        else:
            changed += [r.row_id for r, snap, _ in before if _snap(r) != snap]

    @staticmethod
    def _duplicates(rows: Iterable[PlanRow]) -> Dict[str, List[PlanRow]]:
        """prefix.casefold() -> slides with that prefix in more than one folder."""
        groups = defaultdict(list)
        for r in (r for r in rows if r.cand.kind in ("component", "unsuffixed")):
            groups[r.cand.prefix.casefold()].append(r)
        return {k: g for k, g in groups.items() if len(g) > 1 and len({r.cand.folder for r in g}) > 1}

    def _recompute(self) -> None:
        """Names, de-duplication, output matching, default state and flags of every candidate row, then the notes on
        old outputs that no row may claim (D1)."""
        rows = self._candidate_rows()
        # a name the user typed, and the name of a slide that started, stay as they are
        fixed = {r.row_id: r.output_stem for r in rows if r.output_edited or r.status in RUN_STATUSES}
        auto = {r.row_id: self._auto_stems(r) for r in rows}
        names = dedupe_stems([(r.row_id, auto[r.row_id][0], r.cand.folder, auto[r.row_id][1]) for r in rows], fixed)
        sizes = defaultdict(list)
        for r in rows:
            r.output_stem, how = names[r.row_id]
            r.image_name = tile_name(r.cand) if how == "tiles" else engine_image_name(r.cand)   # D1
            # a slide that started keeps its name, and the flag that says how the plan made it
            kept = r.status in RUN_STATUSES and not r.output_edited and self._renamed.get(r.row_id)
            self._renamed[r.row_id] = kept or (auto[r.row_id][0], "" if r.row_id in fixed else how)
            if r.cand.kind == "component":
                sizes[_sibling_group(r.cand)].append(len(r.cand.tile_names))
        self._medians = {k: statistics.median(v) for k, v in sizes.items() if len(v) >= SIBLING_MIN}  # once per group
        self._dups = self._duplicates(rows)
        self._fallbacks = self._folder_fallbacks(rows) if self.naming == "slide" else {}
        self._unmatched_notes(rows, auto)
        now = time.time()
        for r in rows:
            self._derive(r, now)

    def _auto_stems(self, r: PlanRow) -> Tuple[str, str]:
        """(automatic output stem, sanitised tile name) for the current naming; computed once per row and naming.
        A name that can't be made falls back to 'slide' (de-duplicated), so one odd folder can't stop the plan."""
        hit = self._stems.get(r.row_id)
        if hit is None or hit[0] != self.naming:
            try:
                hit = (self.naming, output_stem(r.cand, self.naming), sanitize_stem(tile_name(r.cand)) if
                       tile_name(r.cand) else "")
            except (UnicodeError, ValueError) as e:       # FS-14
                _log.warning("No output name for %s: %s", r.cand.rel_folder, e)
                hit = (self.naming, "slide", "")
            self._stems[r.row_id] = hit
        return hit[1], hit[2]

    def _folder_fallbacks(self, rows: Sequence[PlanRow]) -> Dict[str, str]:
        """Slide naming (§3.8 step 3, DS-4): row_id -> key of '<folder name>.ome.tif', the name v1 gave the output,
        for a single-slide folder whose name no other folder in the plan has, and only when that file is no row's
        own name."""
        folders, out = defaultdict(set), {}
        for r in rows:
            folders[norm_key(r.cand.folder_name)].add(r.cand.folder)
        finals = {_final_key(r.output_stem) for r in rows}
        for r in rows:
            c, key = r.cand, _final_key(r.cand.folder_name)
            if c.kind in ("component", "unsuffixed") and not c.multi_slide_folder and \
                    len(folders[norm_key(c.folder_name)]) == 1 and key not in finals:
                out[r.row_id] = key
        return out

    def _unmatched_notes(self, rows: Sequence[PlanRow], auto: Mapping[str, Tuple[str, str]]) -> None:
        """D1: an old '<name>.ome.tif' that several slides could own (their names collide, or with slide naming
        their folder names) is matched to none of them; it gets an info row in *Not included* instead."""
        idx, want = self._index, {}
        if idx is not None:
            shared: Dict[str, Tuple[str, set]] = {}
            for r in rows:
                c = r.cand
                names = [auto[r.row_id][0]]
                if self.naming == "slide" and c.kind in ("component", "unsuffixed") and not c.multi_slide_folder:
                    names.append(c.folder_name)
                for name in names:
                    shared.setdefault(_final_key(name), (name, set()))[1].add(c.folder)
            finals = {_final_key(r.output_stem) for r in rows}
            for key, (name, folders) in shared.items():
                if len(folders) < 2:
                    continue
                legacy = re.compile(re.escape(norm_key(name)) + r"__\d+\.ome\.tiff?$")
                for old_key, info in idx.outputs.items():
                    if (old_key == key and old_key not in finals) or legacy.fullmatch(old_key):
                        want[_UNMATCHED + old_key] = Note("unmatched", idx.out_dir, len(folders), name, (info.name,))
        self._unmatched = {rid[len(_UNMATCHED):] for rid in want}
        for rid in [k for k in self._rows if k.startswith(_UNMATCHED) and k not in want]:
            del self._rows[rid], self._order[rid]
        for rid, n in want.items():
            if rid not in self._rows or self._rows[rid].note != n:
                cand = Candidate(rid, n.path, n.names[0], n.names[0], "", "", "note", (), (), False, False, 0)
                self._rows[rid] = PlanRow(rid, cand, "note", "note", False, False, note=n)
                self._order[rid] = (_NOTE_ORDER.index("unmatched"), False, natural_key(n.names[0]))

    def _derive(self, r: PlanRow, now: float) -> None:
        """Output matching, default state and flags of one row, with what the last recompute found."""
        self._match(r)
        self._state(r)
        r.flags = self._flags(r, now)

    def _verdict_key(self, info: OutputInfo) -> tuple:
        return os.path.normcase(os.path.join(self._index.out_dir, info.name)), info.size, info.mtime_ns

    def _match(self, r: PlanRow) -> None:
        """§3.8: the row's final name against the save folder's partials and outputs. An unfinished or ready row
        keeps its existing final (it will be replaced, D11); an unfinished one is busy while another stitcher writes
        its partial. With slide naming a unique folder's old '<folder>.ome.tif' counts as the slide's output unless
        the check found it broken (DS-4), or a partial of the slide's own name shows it is being stitched under that
        name (the old file is then no concern of it)."""
        idx = self._index
        r.existing = r.unfinished = r.ready = r.existing_verdict = None
        r.existing_detail, r.existing_gaps, r.busy = "", 0, False
        if idx is None:
            return
        key = _final_key(r.output_stem)
        if key in self._unmatched:
            return
        r.ready = idx.ready.get(key)
        r.unfinished = idx.partials.get(key) if r.ready is None else None
        r.busy = r.unfinished is not None and key in idx.busy
        r.existing = idx.outputs.get(key)
        old_key = self._fallbacks.get(r.row_id, "")
        old = idx.outputs.get(old_key) if old_key not in self._unmatched else None
        if r.existing is None and old is not None and r.unfinished is None and r.ready is None and \
                self._verdicts.get(self._verdict_key(old), ("",))[0] not in ("incomplete", "unreadable"):
            r.existing = old
        if r.existing is not None:
            r.existing_verdict, r.existing_detail, r.existing_gaps = \
                self._verdicts.get(self._verdict_key(r.existing), ("checking", "", 0))

    def _state(self, r: PlanRow) -> None:
        sec, status, checked, checkable = default_state(r)
        if status == "unsupported":                  # R24: unticks even a user-ticked or waiting row
            r.section, r.status, r.checked, r.checkable = sec, status, False, False
            r.plan_status, r.replaces = status, False
        elif r.status not in RUN_STATUSES:            # run statuses, and the intent they ran with, stay until a rescan
            if status == "busy" and r.checked:        # taken by another stitcher: an earlier tick must not come back
                r.user_touched = False                # later to redo the file it is making (an untick stays)
            checked = r.checked if r.user_touched else checked
            if sec == "existing" and checked:         # a ticked slide whose old file has another name is new (DS-4)
                status = "new" if _other_name(r) else "replace"
            r.section, r.status, r.checked, r.checkable = sec, status, checked, checkable
            # D11: Job.replace when its own final exists at planning (never for an old file of another name, DS-4)
            r.plan_status = status
            r.replaces = status in ("replace", "incomplete", "unfinished", "ready") and r.existing is not None and \
                not _other_name(r)

    def _flags(self, r: PlanRow, now: float) -> List[Flag]:
        """§2.3.4 / §3.10, highest priority first. Flags never untick a row."""
        c, h, out = r.cand, r.header, []
        if h is not None and h.channel_problem:
            out.append(Flag("channels", "info", "Channel names set by you") if r.channel_override else
                       Flag("channels", "info", "Generic channel names") if r.generic_accepted else
                       Flag("channels", "warn", f"Check channel names ({_channel_detail(h)})"))
        recent = _recent(c, now)
        if recent is not None:
            out.append(recent)
        base, how = self._renamed.get(r.row_id, (r.output_stem, ""))
        if how == "tiles":
            out.append(Flag("renamed", "info", f"Named after its tiles: other slide folders are also called {base}"))
        elif how:
            out.append(Flag("renamed", "info", f"Renamed so nothing is overwritten (…{r.output_stem[len(base):]})"))
        others = [o for o in self._dups.get(c.prefix.casefold(), []) if path_key(o.cand.folder) != path_key(c.folder)]
        if others and c.kind in ("component", "unsuffixed"):
            big = max(others, key=lambda o: len(o.cand.tile_names))
            out.append(Flag("duplicate", "info", f"The same slide is also in {big.cand.rel_folder} "
                                                 f"({_n(len(big.cand.tile_names), 'tile')})"))
        median = self._medians.get(_sibling_group(c), 0)
        if c.kind == "component" and len(c.tile_names) < SIBLING_RATIO * median:
            out.append(Flag("sibling_low", "info", f"{len(c.tile_names)} tiles — other folders here have about "
                                                   f"{int(median)}. Export incomplete?"))
        if r.est is not None and r.est.memory_mode == "disk":
            out.append(Flag("disk_mode", "info", f"Too big for memory: uses {human_bytes(r.est.raw_bytes, True)} of "
                                                 f"temporary disk space (slower)"))
        if h is not None and h.offline:
            out.append(Flag("offline", "info", "Cloud placeholder — details are read while stitching"))
        if c.kind == "legacy_all" and r.checked and h is not None and h.first_image_bytes > 2 * GIB:
            out.append(Flag("legacy_big", "warn", f"The first image is {human_bytes(h.first_image_bytes)} — this is "
                                                  f"probably not a tile folder"))
        return out


def _longest_path(out_dir: str, r: PlanRow, temp_dir: Optional[str]) -> int:
    """PL-7: the longest path the runner writes for a row: a name beside its partial in PARTIAL_DIR (a sidecar, or
    the name a stale partial gets while it is taken over), the run's summary table, or the disk-mode scratch file
    in the custom temp folder (the engine's '<partial stem>.tmp.raw')."""
    final = r.output_stem + OME_SUFFIX
    beside = max(len(OWNER_SUFFIX), len(COMPLETE_SUFFIX), len(STALE_INFIX) + STALE_HEX)
    n = max(len(os.path.join(out_dir, PARTIAL_DIR, final)) + beside, len(os.path.join(out_dir, _SUMMARY)))
    if temp_dir and (r.est is None or r.est.memory_mode == "disk"):
        n = max(n, len(os.path.join(temp_subdir(temp_dir, out_dir), os.path.splitext(final)[0] + ".tmp.raw")))
    return n
