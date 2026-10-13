"""Shared data model. Pure data: no I/O, no tkinter. FINAL_SPEC §5.3 (verbatim)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

LOCAL, NETWORK, CLOUD = "local", "network", "cloud"                       # lane classes
H_UNKNOWN, H_OK, H_SLOW, H_UNRESPONSIVE, H_DISCONNECTED = "unknown", "ok", "slow", "unresponsive", "disconnected"

@dataclass(frozen=True)
class FsError:
    code: str                  # not_found|no_access|not_a_dir|no_space|read_only|unavailable|timeout|unresponsive|
                               # disconnected|io|cancelled|not_ready|path_too_long
    message: str               # one plain-language sentence for the UI
    errno: Optional[int] = None
    winerror: Optional[int] = None

@dataclass(frozen=True)
class Entry:
    name: str
    is_dir: bool               # DirEntry.is_dir(follow_symlinks=False)
    is_link: bool              # symlink, or Windows junction/symlink (name-surrogate reparse point only)
    size: Optional[int] = None # Windows: free from the listing; POSIX: filled by stat pass
    mtime: Optional[float] = None
    hidden: bool = False       # dot-name or Windows HIDDEN|SYSTEM
    offline: bool = False      # Windows OFFLINE|RECALL_ON_OPEN|RECALL_ON_DATA_ACCESS

@dataclass(frozen=True)
class Listing:
    path: str
    entries: Tuple[Entry, ...]
    truncated: bool
    elapsed: float
    listed_at: float           # time.monotonic()
    mtime: Optional[float] = None   # the folder's own mtime (scan_dir(stat_self=True))

@dataclass(frozen=True)
class TileGroup:
    prefix: str
    suffix: str                                  # lower-case; "" when absent
    names: Tuple[str, ...]                       # plain sorted() order == engine order
    coords: Tuple[Tuple[float, float], ...]      # tile centres in µm, aligned with names

@dataclass(frozen=True)
class DirSummary:
    path: str
    n_entries: int
    truncated: bool
    subdirs: Tuple[str, ...]                     # visible subfolder names, natural order
    groups: Tuple[TileGroup, ...]
    outputs: Tuple[str, ...]                     # *.ome.tif(f)
    wsi: Tuple[str, ...]                         # *_stitch(ed).tif(f)
    raw_scan: bool
    inform_export: bool
    other_tiffs: int
    leftovers: Tuple[str, ...]                   # *.tmp.raw names; ".stitcher-in-progress" if that folder exists
    notable_files: Tuple[str, ...]               # ≤ 200
    other_files: int
    error: Optional[FsError] = None
    links: int = 0                               # folder links / junctions seen and not followed (rev 2)

@dataclass(frozen=True)
class Rollup:
    probed: int
    slide_folders: int
    tiles: int
    capped: bool

@dataclass(frozen=True)
class Candidate:
    key: str                                     # norm_key(folder)|prefix.casefold()|suffix  (legacy: |*|legacy)
    folder: str
    rel_folder: str                              # relative to the source root, "/"-separated, "." for the root
    folder_name: str
    prefix: str                                  # "" for legacy_all
    suffix: str                                  # "component_data" | "" | other | "legacy"
    kind: str                                    # component | unsuffixed | other_suffix | legacy_all
    tile_names: Tuple[str, ...]                  # () for legacy_all
    coords: Tuple[Tuple[float, float], ...]
    multi_slide_folder: bool
    inform_export: bool
    depth: int
    dir_mtime: Optional[float] = None

@dataclass(frozen=True)
class Note:
    kind: str     # outputs|wsi|raw_scan|lnk|no_images|empty|large_dir|leftovers|system_skipped|symlink_skipped|
                  # cloud_skipped|not_searched_depth|wide_pruned|no_access|unresponsive|error|not_mounted
    path: str
    count: int = 0
    detail: str = ""
    names: Tuple[str, ...] = ()                  # up to 5 examples

@dataclass(frozen=True)
class TileHeader:
    file: str
    shape: Tuple[int, ...]                       # tif.series[0].shape
    dtype: str
    n_channels: int
    tile_h: int
    tile_w: int
    um_per_px_x: float                           # engine default 0.5 when tags are absent
    um_per_px_y: float
    resolution_found: bool
    channels_extracted: Tuple[str, ...]
    channel_problem: Optional[str]               # None | missing | count | duplicate  (engine.assess_channels)
    channel_colors: Tuple[Optional[str], ...]    # "#rrggbb" per extracted channel, or None
    first_image_bytes: int
    offline: bool = False                        # Windows placeholder: nothing was read

@dataclass(frozen=True)
class Estimate:
    canvas_w: int
    canvas_h: int
    n_channels: int
    itemsize: int
    raw_bytes: int
    out_typical: int
    out_worst: int
    coverage: float
    memory_mode: str                             # "ram" | "disk" (prediction with the v1 rule)

@dataclass(frozen=True)
class OutputInfo:
    name: str
    size: int
    mtime_ns: int
    identity: Optional[Tuple[int, int]] = None
    parent_identity: Optional[Tuple[int, int]] = None

@dataclass(frozen=True)
class OutputIndex:
    out_dir: str
    exists: bool
    names: FrozenSet[str]                        # norm_key of every file name in out_dir
    outputs: Dict[str, OutputInfo]               # norm_key -> *.ome.tif(f) in out_dir
    partials: Dict[str, OutputInfo]              # norm_key -> *.ome.tif(f) in out_dir/.stitcher-in-progress
    leftovers: Tuple[Tuple[str, OutputInfo], ...]    # (absolute path, info) of *.tmp.raw and orphan partials
    free_bytes: Optional[int]
    lane: str
    lane_class: str
    is_system_volume: bool
    error: Optional[FsError] = None
    ready: Dict[str, OutputInfo] = field(default_factory=dict)   # norm_key -> complete partials (COMPLETE_SUFFIX
                                                 # sidecar present) that only wait to be moved into place
    missing_levels: int = 0                      # folder levels of out_dir that don't exist yet (0: it exists)
    nearest_existing: Optional[str] = None       # closest existing ancestor when missing_levels > 0
    pdir_problem: Optional[str] = None           # PARTIAL_DIR exists but is unusable (a link/junction): UI text
    cloud_synced: bool = False                   # out_dir is in a cloud-synced folder (OneDrive, iCloud, rclone...)
    busy: FrozenSet[str] = frozenset()           # norm_key of partials another stitcher is writing right now
    identity: Optional[Tuple[int, int]] = None   # save folder at indexing, carried into leftover approval

@dataclass(frozen=True)
class Issue:
    severity: str                                # block | warn | info
    code: str                                    # §2.4.1
    text: str
    rows: Tuple[str, ...] = ()
    action: Optional[str] = None                 # browse_input|browse_output|stop_scan|resume_scan|fix_channels|settings|delete_leftovers

@dataclass(frozen=True)
class Job:
    row_id: str
    run_index: int
    folder: str
    kind: str
    prefix: str
    suffix: str
    display: str
    planned_tile_names: Tuple[str, ...]
    out_dir: str
    final_name: str                              # "<stem>.ome.tif"
    replace: bool
    image_name: Optional[str]
    channel_override: Optional[Tuple[str, ...]]
    est: Optional[Estimate]
    temp_dir: Optional[str]
    finalize_only: bool = False                  # a complete partial from an earlier run only needs moving into place

@dataclass(frozen=True)
class RowOutcome:
    status: str             # done|done_gaps|done_generic|failed|skipped|cancelled|finalize
    message: str            # the "What will happen" text (§2.3.3)
    output_path: Optional[str] = None
    output_bytes: Optional[int] = None
    started: float = 0.0    # time.time()
    duration_s: float = 0.0
    tiles_used: int = 0
    failed_tiles: int = 0
    channels: Tuple[str, ...] = ()
    um_per_px: Optional[float] = None
    memory_mode: Optional[str] = None
    output_unavailable: bool = False             # the write probe failed at the start of this slide
    output_side: bool = False                    # failed while writing, and the output location is the cause (R28)
    partial_path: Optional[str] = None           # kept for "finalize"

@dataclass(frozen=True)
class RunSummary:
    started: float
    finished: float
    counts: Dict[str, int]                        # status -> n
    rows: Tuple[Tuple[str, RowOutcome], ...]
    stopped_by_user: bool
    stop_reason: Optional[str]                   # None | "user" | "output_unavailable"
    csv_path: Optional[str]
