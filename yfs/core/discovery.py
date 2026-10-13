"""Breadth-first search for slide folders (FINAL_SPEC §3.5).

Folders are listed through FsService lanes (names only: no file is ever opened) and results are posted as events.
The walk never descends into the output folder, a capped folder, a folder link, or a cloud, disconnected,
unmounted-autofs or pseudo mount met on the way (R27); the subfolders of a slide folder are searched too. Of a
folder with more than 200 subfolders only the first 50 are searched, unless tiles turn up below them or none of them
could be searched completely. Folders that failed fast on a lane that was not answering are listed once more at the
end. The walk pauses after a budget of folders or time.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from dataclasses import dataclass
from functools import partial
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .events import CandidateFound, Post, ScanFinished, ScanNote, ScanPaused, ScanProgress
from .fsservice import DEADLINE, FsFailure, FsService
from .model import CLOUD, H_DISCONNECTED, H_UNRESPONSIVE, FsError, Listing, Note
from .names import (BUDGET_DIRS, BUDGET_SECONDS, EXCLUDED_DIR_NAMES, LIST_CAP_NO_TILES, MAX_DEPTH_DEFAULT,
                    PARTIAL_DIR, TILE_RE, WIDE_SAMPLE, WIDE_THRESHOLD, candidates_from_summary, notes_from_summary,
                    rel_folder, same_path, summarize_listing)

_NOTE_FOR_ERROR = {"no_access": "no_access", "timeout": "unresponsive", "unresponsive": "unresponsive",
                   "disconnected": "unresponsive", "unavailable": "unresponsive"}
_FAILED_FAST = frozenset({"unresponsive", "disconnected"})     # refused at once: the lane was not answering


@dataclass(frozen=True)
class DiscoveryOptions:
    max_depth: int = MAX_DEPTH_DEFAULT
    out_dir: Optional[str] = None            # listed, never descended into
    no_prune: FrozenSet[str] = frozenset()   # wide folders the user asked to search fully
    budget_dirs: int = BUDGET_DIRS
    budget_s: float = BUDGET_SECONDS
    cache_age: float = 120.0                 # reuse cached listings up to this old (s); 0: list every folder again
    extra_depth: int = 0                     # levels added by Search deeper (included in max_depth); a cloud root
                                             # searches min(max_depth, 1 + extra_depth) levels


def depth_of(root: str, path: str) -> int:
    """Levels of `path` below `root` (0 for the root): the depth to pass with a seed."""
    rel = rel_folder(path, root)
    return 0 if rel == "." else rel.count("/") + 1


def seeds_for(root: str, paths: Sequence[str]) -> List[Tuple[str, int]]:
    """Seeds for Search deeper (ScanFinished.frontier), Search these folders too, or Search this folder again."""
    return [(p, depth_of(root, p)) for p in paths]


def _as_good_as_own(listing: Listing) -> bool:
    """A cached listing can stand in for discovery's own if it is complete, or was capped after LIST_CAP_NO_TILES
    entries or more without a tile name among them. The browser caps its listings without looking at names."""
    return not listing.truncated or (len(listing.entries) >= LIST_CAP_NO_TILES
                                     and not any(TILE_RE.match(e.name) for e in listing.entries))


class Discovery(threading.Thread):
    """Breadth-first walk of §3.5. Emits CandidateFound, ScanNote, ScanProgress (<= 5/s), ScanPaused, ScanFinished."""

    def __init__(self, scan_id: int, root: str, fs: FsService, post: Post, opts: DiscoveryOptions,
                 seeds: Optional[Sequence[Tuple[str, int]]] = None) -> None:
        super().__init__(name=f"yfs-discovery-{scan_id}", daemon=True)
        self.scan_id, self.root, self.fs, self.post, self.opts = scan_id, root, fs, post, opts
        self.seeds = list(seeds) if seeds else [(root, 0)]
        self.ticket = fs.new_ticket()
        self.dirs_done = self.slides_found = 0
        self.max_depth = opts.max_depth                # the depth searched; run() lowers it for a cloud root
        self.frontier: List[str] = []
        self.pruned: List[str] = []
        self.failed: List[str] = []                    # folders that could not be listed, also after the retry
        # wide folder -> [rest, depth, pending, the sample around it, its samples]; a sample is [wide folder, no
        # failure in it so far], shared by the queue items of one sample folder's subtree
        self._wide: Dict[str, list] = {}
        self._retry: Dict[str, Tuple[tuple, FsError]] = {}     # folder that failed fast -> (queue item, first error)
        self._retried = False
        self._halt, self._wake = threading.Event(), threading.Event()
        self._active_s = 0.0

    def stop(self) -> None:
        """Finish now; keep results; posts ScanFinished(stopped=True). Any thread."""
        self._halt.set()
        self._wake.set()
        self.fs.cancel_ticket(self.ticket)

    def resume(self) -> None:
        """After ScanPaused: grant another budget. Any thread."""
        self._wake.set()

    def run(self) -> None:
        try:
            if self.fs.lane_class(self.root) == CLOUD:     # cloud: one level, plus what Search deeper added
                self.max_depth = min(self.opts.max_depth, 1 + self.opts.extra_depth)
            self._walk()
        finally:                                       # the UI always learns that the search ended
            self.post(ScanFinished(self.scan_id, self.dirs_done, self.slides_found, self._active_s,
                                   self._halt.is_set(), tuple(self.frontier), tuple(self.pruned),
                                   max_depth=self.max_depth, failed=tuple(self.failed)))

    def _note(self, kind: str, path: str, count: int = 0, detail: str = "", names: Tuple[str, ...] = ()) -> None:
        self.post(ScanNote(self.scan_id, Note(kind, path, count, detail, names)))

    def _list(self, path: str) -> Future:
        age = self.opts.cache_age
        cached = self.fs.cached_listing(path, max_age=age) if age > 0 else None
        if cached is not None and _as_good_as_own(cached):    # warm cache from the browser or an earlier walk
            fut: Future = Future()
            fut.set_result(cached)
            return fut
        fn = partial(self.fs.scan, path, cap=LIST_CAP_NO_TILES, cap_only_without_tiles=True, stat_self=True)
        return self.fs.submit(path, "list", fn, ticket=self.ticket, owner="discovery", priority=1,
                              deadline_s=DEADLINE["list_discovery"][self.fs.lane_class(path)])

    def _walk(self) -> None:
        o = self.opts
        conc = self.fs.concurrency(self.root)
        queue = deque((p, d, None, False) for p, d in self.seeds)   # (path, depth, sample, list only)
        inflight: Dict[Future, tuple] = {}
        start = last = time.monotonic()
        paused, current = 0.0, self.root
        d_budget, t_budget = o.budget_dirs, start + o.budget_s
        while not self._halt.is_set() and (queue or inflight or self._retry_once(queue)):   # retry: at the end
            while queue and len(inflight) < conc:
                item = queue.popleft()
                inflight[self._list(item[0])] = item
            for fut in wait(list(inflight), timeout=0.25, return_when=FIRST_COMPLETED).done:
                item = inflight.pop(fut)
                current = item[0]
                try:
                    listing = fut.result()
                except CancelledError:
                    continue
                except FsFailure as e:
                    if not self._retried and e.error.code in _FAILED_FAST:
                        self._retry[item[0]] = (item, e.error)     # its lane was not answering: again at the end
                    else:
                        self._give_up(item, e.error, queue)
                else:
                    self._retry.pop(item[0], None)
                    self._visit(item, listing, queue)
            now = time.monotonic()
            self._active_s = now - start - paused
            if now - last >= 0.2:
                last = now
                self.post(ScanProgress(self.scan_id, self.dirs_done, self.slides_found, current, self._active_s))
            if (queue or inflight) and not self._halt.is_set() and (self.dirs_done >= d_budget or now >= t_budget):
                self._wake.clear()
                if self._halt.is_set():
                    break
                self.post(ScanPaused(self.scan_id, "budget", self.dirs_done, self._active_s))
                self._wake.wait()                      # resume() or stop() from the main thread
                paused += time.monotonic() - now
                d_budget, t_budget = self.dirs_done + o.budget_dirs, time.monotonic() + o.budget_s
        for item, err in self._retry.values():         # stopped before or during their retry: still unsearched
            self._failed(item[0], err)
        self._active_s = time.monotonic() - start - paused

    def _retry_once(self, queue: deque) -> bool:
        """The walk is done: queue the folders that failed fast once more, where their lane answers again; the
        others have failed for good. True if anything was queued."""
        if self._retried:
            return False
        self._retried = True
        for item, err in list(self._retry.values()):
            if self.fs.health(item[0]) in (H_UNRESPONSIVE, H_DISCONNECTED):
                self._give_up(item, err, queue)
            else:
                queue.append(item)                     # stays in _retry until it is listed or fails again
        return bool(queue)

    def _give_up(self, item: tuple, err: FsError, queue: deque) -> None:
        """`item` could not be listed, for good: report it. Its sample (if any) was not searched completely."""
        self._retry.pop(item[0], None)
        self._failed(item[0], err)
        if item[2] is not None:
            item[2][1] = False
        self._wide_done(item[2], queue)

    def _failed(self, path: str, err: FsError) -> None:
        self._note(_NOTE_FOR_ERROR.get(err.code, "error"), path, detail=err.message)
        self.failed.append(path)

    def _visit(self, item: tuple, listing: Listing, queue: deque) -> None:
        """Post what one listing holds and queue its subfolders (a slide folder's too)."""
        path, depth, sample, list_only = item
        o = self.opts
        s = summarize_listing(path, listing.entries, truncated=listing.truncated)
        for c in candidates_from_summary(s, root=self.root, depth=depth, dir_mtime=listing.mtime):
            self.post(CandidateFound(self.scan_id, c))
            self.slides_found += c.kind == "component"
        for n in notes_from_summary(s):
            self.post(ScanNote(self.scan_id, n))
        system = [e.name for e in listing.entries if e.is_dir and e.name != PARTIAL_DIR and
                  (e.name.casefold() in EXCLUDED_DIR_NAMES or e.name.casefold().startswith(".trash-"))]
        if system:
            self._note("system_skipped", path, len(system), names=tuple(system[:5]))
        self.dirs_done += 1
        if s.groups:                                   # tiles in a wide folder's sample: search all of it
            self._wide_found(sample, queue)
        sample = sample if sample is not None and sample[0] in self._wide else None
        subs = [] if s.truncated or list_only else self._enterable(path, s.subdirs)   # not into capped/output
        queue.extend((p, depth + 1, None, True) for p in subs if same_path(p, o.out_dir))
        subs = [p for p in subs if not same_path(p, o.out_dir)]
        if subs and depth >= self.max_depth:
            self.frontier.append(path)
            self._note("not_searched_depth", path, len(subs))
        elif len(subs) > WIDE_THRESHOLD and path not in o.no_prune:
            samples = [[path, True] for _ in subs[:WIDE_SAMPLE]]
            self._wide[path] = [subs[WIDE_SAMPLE:], depth + 1, len(samples), sample, samples]
            queue.extend((p, depth + 1, t, False) for p, t in zip(subs, samples))
            return                                     # done for its own sample once it is decided itself
        else:
            queue.extend((p, depth + 1, sample, False) for p in subs)
            if sample is not None:                     # the sample's whole subtree counts for the wide folder
                self._wide[sample[0]][2] += len(subs)
        self._wide_done(sample, queue)

    def _enterable(self, path: str, names: Sequence[str]) -> List[str]:
        """The subfolders to search (R27). A cloud mount gets a note each; unmounted automounts, disconnected drives
        and pseudo filesystems one not_mounted note per folder. fs.auto_enter reads the current mount table."""
        fs, subs, skipped = self.fs, [], []
        for d in names:
            p = os.path.join(path, d)
            why = fs.auto_enter(p)
            if why == "cloud" and fs.lane_key(p) == fs.lane_key(path):
                why = None                             # inside a cloud folder the user chose: not a boundary
            if why is None:
                subs.append(p)
            elif why == "cloud":
                self._note("cloud_skipped", p)
            else:
                skipped.append((d, why))
        if skipped:
            self._note("not_mounted", path, len(skipped), ", ".join(sorted({why for _, why in skipped})),
                       tuple(d for d, _ in skipped[:5]))
        return subs

    def _wide_found(self, sample: Optional[list], queue: deque) -> None:
        """Tiles in a sample of a wide folder, so also in those of the wide folders around it: search their rest."""
        while sample is not None and sample[0] in self._wide:
            rest, depth, _, sample, _ = self._wide.pop(sample[0])
            queue.extend((p, depth, None, False) for p in rest)

    def _wide_done(self, sample: Optional[list], queue: deque) -> None:
        """One folder of a sample subtree of a wide folder is done. Once all are, without tiles: prune the rest if a
        sample folder was searched without a failure; if none was, there is nothing to go by, so search it too."""
        while sample is not None and sample[0] in self._wide:
            w = sample[0]
            self._wide[w][2] -= 1
            if self._wide[w][2]:
                return
            rest, depth, _, sample, samples = self._wide.pop(w)     # now done for the sample around it, if any
            incomplete = sum(not t[1] for t in samples)             # sample folders with a failure in them
            if incomplete and sample is not None:
                sample[1] = False
            if incomplete < len(samples):
                self.pruned.append(w)
                self._note("wide_pruned", w, len(rest), str(incomplete) if incomplete else "")
            else:
                queue.extend((p, depth, sample, False) for p in rest)
                if sample is not None and sample[0] in self._wide:  # the rest belongs to the sample around it
                    self._wide[sample[0]][2] += len(rest)
