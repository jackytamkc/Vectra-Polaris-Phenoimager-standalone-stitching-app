"""Events posted from worker threads to the UI queue. FINAL_SPEC §5.4 (verbatim)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from .model import Candidate, FsError, Note, RowOutcome, RunSummary

Post = Callable[[object], None]          # thread-safe, never blocks (UiBus.post or a test collector)

@dataclass(frozen=True)
class FsDone:          # a filesystem request finished (result or error); posted only when notify=True
    req_id: int; kind: str; path: str; ticket: int; owner: str; tag: Any
    result: Any; error: Optional[FsError]; elapsed: float

@dataclass(frozen=True)
class FsSlow:          # still running after 2.5 s (past_deadline=False) or past its deadline (True)
    req_id: int; kind: str; path: str; ticket: int; owner: str; tag: Any; elapsed: float; past_deadline: bool

@dataclass(frozen=True)
class LaneHealthChanged:
    lane: str; health: str; since: float      # time.time() of the change

@dataclass(frozen=True)
class ScanProgress:
    scan_id: int; dirs_done: int; slides_found: int; current: str; elapsed: float

@dataclass(frozen=True)
class CandidateFound:
    scan_id: int; candidate: Candidate

@dataclass(frozen=True)
class ScanNote:
    scan_id: int; note: Note

@dataclass(frozen=True)
class ScanPaused:
    scan_id: int; reason: str; dirs_done: int; elapsed: float

@dataclass(frozen=True)
class ScanFinished:
    scan_id: int; dirs_done: int; slides_found: int; elapsed: float; stopped: bool
    frontier: Tuple[str, ...]; pruned: Tuple[str, ...]
    max_depth: int = 0                       # the depth actually searched (1 for a cloud root without Search deeper)
    failed: Tuple[str, ...] = ()             # folders that could not be searched (no access, error, didn't answer)

@dataclass(frozen=True)
class RunStarted:
    run_id: int; n_jobs: int; t: float

@dataclass(frozen=True)
class RowStarted:
    run_id: int; row_id: str; t: float

@dataclass(frozen=True)
class RowProgress:     # phase: reading|stitching|preparing|writing|pyramid|finishing
    run_id: int; row_id: str; phase: str; current: int; total: int; t: float

@dataclass(frozen=True)
class RowBytes:        # write progress from polling the partial file (phase: writing = level 0, pyramid = levels 1+)
    run_id: int; row_id: str; written: int; expected: int; t: float
    phase: str = "writing"

@dataclass(frozen=True)
class RowFinished:
    run_id: int; row_id: str; outcome: RowOutcome

@dataclass(frozen=True)
class OutputWritten:   # -> settings ledger
    path: str; size: int; mtime_ns: int
    gaps: int = 0                            # tiles that were missing or unreadable when it was made

@dataclass(frozen=True)
class RunFinished:
    run_id: int; summary: RunSummary

@dataclass(frozen=True)
class MountsChanged:   # FsService swapped in a newer mount table (fs.mounts); places and lanes may differ now
    t: float

@dataclass(frozen=True)
class LogLine:
    created: float; level: int; slide: str; text: str; tag: str   # tag: info|ok|warn|error|debug (§4.9)

@dataclass(frozen=True)
class PickerResult:
    token: int; path: Optional[str]; error: Optional[str]

@dataclass(frozen=True)
class BackgroundError:
    where: str; text: str

COALESCE_KEY: Dict[type, Callable[[Any], Any]] = {      # UiBus keeps only the latest event per key in each tick
    ScanProgress: lambda e: ("scan", e.scan_id),
    RowProgress: lambda e: ("rowp", e.run_id, e.row_id),
    RowBytes: lambda e: ("rowb", e.run_id, e.row_id),
    FsSlow: lambda e: ("slow", e.req_id, e.past_deadline),
}
