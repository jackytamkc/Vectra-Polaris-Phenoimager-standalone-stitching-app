"""Preflight I/O (FINAL_SPEC §3.8, §5.5.6): first-tile headers, the output-folder index, checks of existing outputs,
the write probe, who owns a partial (D3), leftover clean-up and moving a finished partial into place.

Every function does blocking I/O: call it inside an FsService lane (fs.call / fs.submit) or on the runner thread,
never on the Tk thread. Nothing here reads pixel data.
"""
from __future__ import annotations

import errno
import hashlib
import io
import itertools
import json
import logging
import os
import re
import shutil
import socket
import stat
import struct
import sys
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from concurrent.futures import Future
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import tifffile

from .fsservice import map_oserror
from .model import OutputIndex, OutputInfo, TileHeader
from .names import (COMPLETE_SUFFIX, OUTPUT_RE, OWNER_SUFFIX, PARTIAL_DIR, STALE_INFIX, STALE_PARTIAL_S,
                    TEMP_SUBDIR_PREFIX, TMP_RAW_RE, norm_key, pyramid_levels, temp_subdir)
from .places import MountTable

WINDOWS = sys.platform == "win32"
_OFFLINE_ATTRS = 0x1000 | 0x40000 | 0x400000     # Windows OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS
_REPARSE_POINT, _NAME_SURROGATE = 0x400, 0x20000000    # junctions and symlinks; OneDrive placeholders are other ones
RETRY_DELAY_S = 1.0                              # between os.replace attempts in move_into_place
PROBE_TRIES, PROBE_DELAY_S = 3, 0.5              # write_probe: a file held briefly by a virus scanner (FS-17)
STALE_RE = re.compile(r"\.ome\.tiff?" + re.escape(STALE_INFIX) + "[0-9a-f]+$", re.IGNORECASE)   # being taken over (D3)
HOST = socket.gethostname()
PDIR_LINK_TEXT = f"Its {PARTIAL_DIR} folder is a link to another place. Delete the link, or choose another save folder."
PDIR_FILE_TEXT = (f"It holds a file named {PARTIAL_DIR}, where unfinished images belong. Delete that file, or choose "
                  f"another save folder.")
_PLACEMENT_TAGS = ((282, "resolution"), (283, "resolution"), (286, "position"), (287, "position"))   # X and Y
_OME = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
_BOOT = psutil.boot_time()                       # with the host name, tells this computer from a namesake
_STARTED = psutil.Process().create_time()        # with the pid, tells this process from a later one with its pid
_ids = itertools.count(1)                        # unique probe names: Windows keeps a deleted name busy a while
_live: set = set()                               # owner tokens of the partials this process is writing (D3)
_live_lock = threading.Lock()
_stuck: set = set()                              # probe files whose delete failed: write_probe retries them
_log = logging.getLogger(__name__)
_reservations = threading.local()
_UNSPECIFIED = object()


def read_tile_header(path: str, engine_cls: type) -> TileHeader:
    """§3.8: the first tile as the engine will see it (shape, dtype, channels, resolution) from tags and XML only.
    The engine's own static helpers make the decisions, so the prediction is exact; a tile the engine would not
    place, or could not open at all, raises ValueError (a bad tile: a header_error row without Try again, FS-15,
    UD4); an I/O error raises OSError, even one met inside the TIFF structure. A Windows cloud placeholder is not
    opened at all: TileHeader(offline=True)."""
    if _offline(path):
        return TileHeader(path, (), "", 0, 0, 0, 0.5, 0.5, False, (), None, (), 0, offline=True)
    raw = _ReadErrors(path)
    try:
        with io.BufferedReader(raw) as fh, tifffile.TiffFile(fh) as tif:
            _check_placeable(tif)
            shape, dtype = tuple(tif.series[0].shape), np.dtype(tif.series[0].dtype)
            n, tile_h, tile_w = engine_cls.normalize_shape(shape) or (0, 0, 0)    # the engine keeps 0 then
            names = list(engine_cls.extract_perkin_elmer_channels(tif))
            problem = engine_cls.assess_channels(names, n)[1]
            res = engine_cls.resolution_from_tags(tif.pages[0].tags)
            colors = _channel_colors(tif, len(names))
    except tifffile.TiffFileError as e:              # (a ValueError) damaged, or a share hiccup while reading an IFD
        raise _io_error(e) or ValueError("it isn't a readable TIFF") from None
    finally:
        if raw.error is not None:                    # a failed read, whatever tifffile made of it
            raise raw.error
    um_x = res[2] if res is not None and res[2] is not None else 0.5          # the engine's 0.5 default
    um_y = res[3] if res is not None and res[3] is not None else 0.5
    return TileHeader(path, shape, dtype.name, n, tile_h, tile_w, um_x, um_y, res is not None, tuple(names), problem,
                      colors, int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)


def _check_placeable(tif: "tifffile.TiffFile") -> None:
    """FS-15: the engine drops a tile without its resolution and position tags (re-saved by another program?) and
    fails the slide when the first tile's image data runs past the end of the file (an interrupted copy)."""
    tags = tif.pages[0].tags
    missing = list(dict.fromkeys(kind for code, kind in _PLACEMENT_TAGS if code not in tags))
    if missing:
        raise ValueError(f"it has no {' or '.join(missing)} tags")
    if any(_data_end(page) > tif.filehandle.size for page in tif.series[0].pages if page is not None):
        raise ValueError("the file is cut off")


class _ReadErrors(io.FileIO):
    """A file that keeps the OSError of a failed read (io.BufferedReader reads through readinto). tifffile swallows
    one met while it walks the pages or reads a tag: it logs the error and goes on with what it found, so a share
    hiccup would make a 5-channel tile look like a 2-channel one, or like one with channel names missing."""
    error: Optional[OSError] = None

    def readinto(self, buffer: Any) -> Optional[int]:
        try:
            return super().readinto(buffer)
        except OSError as e:
            self.error = e
            raise

    def seek(self, offset: int, whence: int = 0) -> int:
        try:
            return super().seek(offset, whence)
        except OSError as e:
            self.error = e
            raise


def _offline(path: str) -> bool:
    if not WINDOWS:
        return False
    try:
        return bool(os.stat(path).st_file_attributes & _OFFLINE_ATTRS)
    except OSError:
        return False


def _channel_colors(tif: "tifffile.TiffFile", count: int) -> Tuple[Optional[str], ...]:
    """'#rrggbb' from <Color>R,G,B</Color> of the pages the engine takes names from (same filter), else None."""
    colors: List[Optional[str]] = []
    for page in tif.pages:
        xml = page.tags[270].value if 270 in page.tags else ""
        if not isinstance(xml, str) or "PerkinElmer-QPI-ImageDescription" not in xml:
            continue
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        kind, name, color = root.find("ImageType"), root.find("Name"), root.find("Color")
        if (kind is not None and kind.text == "Thumbnail") or name is None or not name.text:
            continue
        try:
            rgb = [int(v) for v in (color.text or "").split(",")] if color is not None else []
        except ValueError:
            rgb = []
        colors.append("#%02x%02x%02x" % tuple(rgb) if len(rgb) == 3 and all(0 <= v < 256 for v in rgb) else None)
    return tuple(colors) if len(colors) == count else (None,) * count


def first_tiff_name(folder: str) -> Optional[str]:
    """legacy_all rows: the engine's first file, i.e. the first *.tif/*.tiff in plain sorted() order."""
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff")))
    return names[0] if names else None


def legacy_gaps(folder: str, placed: Iterable[str]) -> int:
    """DS-1 for legacy_all rows, which take every *.tif/*.tiff of the folder (v1's rule): how many of the files the
    engine did not place (not in `placed`) were tiles. A readable TIFF without XPosition/YPosition (an overview or
    label image, an earlier output) is no tile, and v1 leaves it out by design. Raises OSError."""
    placed = set(placed)
    return sum(_is_tile(os.path.join(folder, f)) for f in os.listdir(folder)
               if f.lower().endswith((".tif", ".tiff")) and f not in placed)


def _is_tile(path: str) -> bool:
    """True unless `path` is a readable TIFF whose first page lacks the position tags: a file the engine couldn't read
    may have been a tile."""
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            return 286 in tags and 287 in tags
    except Exception:
        return True


# ------------------------------------------------------------------------------------------- the output folder
def list_output_dir(out_dir: str, mounts: MountTable, planned: Optional[Iterable[str]] = None,
                    temp_dir: Optional[str] = None) -> OutputIndex:
    """§3.8: one listing of the output folder, of its partial folder and, with a custom temp folder, of this save
    folder's scratch subfolder (D4); one stat per output; the free space.
    - partials: unfinished *.ome.tif(f) in PARTIAL_DIR; ready: complete ones that only wait to be moved into place;
      busy: the partials that are live (another stitcher is writing them; the runner skips their slides). Their
      age is told by the partial folder's own clock, as the runner tells it (DS-3).
    - leftovers: *.tmp.raw files not in use (FS-13: neither they nor their partial, or v1's output, changed in the
      last STALE_PARTIAL_S, and no running owner), and, when `planned` (norm_key of the plan's final names) is
      given, partials no planned row maps to that are neither complete nor live. Without a plan no partial is
      offered (DS-9).
    A PARTIAL_DIR that is a link, a junction or a file is not used: pdir_problem says why (DS-10)."""
    names, outputs, found, v1_raws, raws, remnants = set(), {}, {}, [], [], []
    exists, error, pdir, pdir_problem, identity = True, None, None, None, None
    try:
        identity = path_id(out_dir)
        with os.scandir(out_dir) as it:
            for de in it:
                if de.name == PARTIAL_DIR:
                    pdir_problem = partial_dir_problem(de.path)
                    pdir = None if pdir_problem else de.path
                    continue
                names.add(norm_key(de.name))
                _sort_file(de, outputs, v1_raws, [])
        if pdir is not None:
            with os.scandir(pdir) as it:
                for de in it:
                    _sort_file(de, found, raws, remnants)
    except FileNotFoundError:
        exists = pdir is not None                    # False, unless only the partial folder vanished meanwhile
    except OSError as e:
        error = map_oserror(e)
    now = time.time()
    pdir_now = share_now(pdir) if found else now     # a probe file: the server's clock stamps the partials
    states = {k: partial_state(os.path.join(pdir, i.name), i.mtime_ns / 1e9, pdir_now) for k, i in found.items()}
    ready = {k: i for k, i in found.items() if states[k] == "complete"}
    partials = {k: i for k, i in found.items() if k not in ready}
    leftovers = [(p, i) for p, i in v1_raws if not _in_use(i, now, lambda k: k in outputs and
                                                            now - outputs[k].mtime_ns / 1e9 < STALE_PARTIAL_S)]
    leftovers += [(p, i) for p, i in raws + _scratch_raws(temp_dir, out_dir)
                  if not _in_use(i, now, lambda k: states.get(k) == "live")]
    if planned is not None:
        keep = {norm_key(k) for k in planned}
        leftovers += [(os.path.join(pdir, i.name), i) for k, i in sorted(partials.items())
                      if k not in keep and states[k] != "live"] + remnants
    levels, nearest = (0, None) if exists or error else _missing_levels(out_dir)
    return OutputIndex(out_dir, exists, frozenset(names), outputs, partials, tuple(leftovers), _free(out_dir),
                       mounts.lane_key(out_dir), mounts.lane_class(out_dir), mounts.is_system_volume(out_dir), error,
                       ready=ready, missing_levels=levels, nearest_existing=nearest, pdir_problem=pdir_problem,
                       cloud_synced=mounts.cloud_synced(out_dir),
                       busy=frozenset(k for k, state in states.items() if state == "live"), identity=identity)


def _sort_file(de: "os.DirEntry", outputs: dict, raws: list, remnants: list) -> None:
    """*.ome.tif(f) -> outputs[norm_key]; *.tmp.raw -> raws; '<partial>.stale-<hex>' (a takeover that was cut
    short) -> remnants. One stat each (free on Windows)."""
    kind = "output" if OUTPUT_RE.search(de.name) else "raw" if TMP_RAW_RE.search(de.name) else \
        "remnant" if STALE_RE.search(de.name) else None
    if kind is None:
        return
    try:
        if de.is_dir(follow_symlinks=False) or de.is_symlink():
            return
        st = de.stat()
    except OSError:                                  # vanished or a dangling link
        return
    info = OutputInfo(de.name, st.st_size, st.st_mtime_ns, (st.st_dev, st.st_ino), path_id(os.path.dirname(de.path)))
    if kind == "output":
        outputs[norm_key(de.name)] = info
    else:
        (raws if kind == "raw" else remnants).append((de.path, info))


def _scratch_raws(temp_dir: Optional[str], out_dir: str) -> List[Tuple[str, OutputInfo]]:
    """*.tmp.raw in this save folder's subfolder of the custom temp folder (D4, DS-16)."""
    raws: list = []
    if temp_dir:
        try:
            with os.scandir(temp_subdir(temp_dir, out_dir)) as it:
                for de in it:
                    _sort_file(de, {}, raws, [])
        except OSError:                              # none yet, or the temp folder is unavailable: nothing to offer
            pass
    return raws


def list_scratch_dir(temp_dir: str, out_dir: str) -> Tuple[int, tuple]:
    """Temporary-volume facts only; combine with the output index after both lanes answer."""
    free, raws = shutil.disk_usage(temp_dir).free, []
    try:
        with os.scandir(temp_subdir(temp_dir, out_dir)) as entries:
            for entry in entries:
                _sort_file(entry, {}, raws, [])
    except FileNotFoundError:
        pass
    return free, tuple(raws)


def _partner_keys(raw_name: str) -> Tuple[str, str]:
    """norm_key of the partial (or v1 output) a scratch file belongs to: X.ome.tmp.raw -> X.ome.tif / X.ome.tiff."""
    stem = raw_name[:-len(".tmp.raw")]
    return norm_key(stem + ".tif"), norm_key(stem + ".tiff")


def _in_use(raw: OutputInfo, now: float, partner_live: Callable[[str], bool]) -> bool:
    """FS-13: a scratch file is in use while it changed recently (the heartbeat touches it) or its partial is live."""
    return now - raw.mtime_ns / 1e9 < STALE_PARTIAL_S or any(partner_live(k) for k in _partner_keys(raw.name))


def _missing_levels(path: str) -> Tuple[int, Optional[str]]:
    """MW-26: how many folder levels of a missing `path` would be created, and the closest existing folder."""
    path, levels = os.path.abspath(path), 0
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return levels, None
        path, levels = parent, levels + 1
    return levels, path


def _free(path: str) -> Optional[int]:
    """Free bytes on the volume of `path`, or of its closest existing ancestor when it doesn't exist yet."""
    while True:
        try:
            return shutil.disk_usage(path).free
        except FileNotFoundError:
            parent = os.path.dirname(path)
            if not parent or parent == path:
                return None
            path = parent
        except OSError:
            return None


def partial_dir_problem(pdir: str) -> Optional[str]:
    """Why an existing PARTIAL_DIR must not be used (DS-10): a link or junction (its target could hold anyone's
    files) or not a folder at all. None for a plain folder, or when there is none."""
    try:
        st = os.lstat(pdir)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or (getattr(st, "st_file_attributes", 0) & _REPARSE_POINT
                                    and getattr(st, "st_reparse_tag", 0) & _NAME_SURROGATE):
        return PDIR_LINK_TEXT
    return None if stat.S_ISDIR(st.st_mode) else PDIR_FILE_TEXT


def path_id(path: str) -> Tuple[int, int]:
    """(st_dev, st_ino) of what is at `path` now: tells the save folder from whatever appears there after an unmount
    (DS-8), and our claimed partial from another stitcher's (D3). Always os.stat by path, so that every comparison
    uses one source (on Windows os.fstat may report another form of the file id)."""
    st = os.stat(path)
    return st.st_dev, st.st_ino


class ReservationBusy(OSError):
    """Another process is changing the ownership or publication of this output."""


class OutputReservation:
    def __init__(self, partial: str, folder: str) -> None:
        self.folder, self.parent = folder, os.path.dirname(partial)
        self.folder_id, self.parent_id = path_id(folder), self._parent_id()
        self.fd: Optional[int] = None

    def close(self) -> None:
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)

    def _parent_id(self) -> Optional[Tuple[int, int]]:
        try:
            return path_id(self.parent)
        except FileNotFoundError:                     # a scratch leftover can outlive the partial directory
            return None

    def check(self) -> None:
        if path_id(self.folder) != self.folder_id or self._parent_id() != self.parent_id:
            raise OSError("The save folder changed. Rescan before trying again.")


@contextmanager
def reserve_output(partial: str):
    """Cross-process, non-blocking reservation shared by claims, takeover and publication.
    The tiny lock file stays in the save folder: unlinking it would let another process lock a different inode.
    OS locks are released on process exit. Unsupported locking fails closed; it never permits an unsafe move."""
    partial = os.path.abspath(partial)
    parent = os.path.dirname(partial)
    folder = os.path.dirname(parent) if os.path.basename(parent) == PARTIAL_DIR else parent
    key = (folder, norm_key(os.path.basename(partial)))
    held = getattr(_reservations, "held", None)
    if held is None:
        held = _reservations.held = {}
    if key in held:
        held[key].check()
        yield held[key]
        return
    reservation = _open_reservation(partial)
    held[key] = reservation
    try:
        yield reservation
    finally:
        del held[key]
        reservation.close()


def _open_reservation(partial: str) -> OutputReservation:
    """A detached reservation: acquired and closed on the output lane, held across scratch-lane work."""
    parent = os.path.dirname(partial)
    folder = os.path.dirname(parent) if os.path.basename(parent) == PARTIAL_DIR else parent
    reservation = OutputReservation(partial, folder)
    digest = hashlib.sha256(norm_key(os.path.basename(partial)).encode("utf-8")).hexdigest()[:32]
    lock_path = os.path.join(folder, ".yfs-lock-" + digest)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            if WINDOWS:
                import msvcrt
                if not os.fstat(fd).st_size:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ReservationBusy(errno.EBUSY, "Another stitcher is updating this slide. Try again.") from None
            raise
        reservation.check()
        st = os.fstat(fd)
        if not WINDOWS and path_id(lock_path) != (st.st_dev, st.st_ino):
            raise ReservationBusy(errno.EBUSY, "The slide reservation changed. Try again.")
        reservation.fd = fd
        return reservation
    except BaseException:
        os.close(fd)
        raise


# --------------------------------------------------------------------------------------- partial ownership (D3)
def partial_state(path: str, mtime: float, now: float, stale_s: float = STALE_PARTIAL_S) -> str:
    """'complete' (fsynced, waits to be moved into place: never taken over or deleted), 'live' (its owner on this
    computer is running or, with no owner to ask, it or its owner record changed less than stale_s before `now` by
    its folder's clock), 'dead' (its owner on this computer has exited: take it over at once, CT-9) or 'stale'.
    `mtime` is the partial's."""
    try:
        os.stat(path + COMPLETE_SUFFIX)
        return "complete"
    except FileNotFoundError:
        pass
    except OSError:                                  # can't tell: treat it as in use, never take it over
        return "live"
    alive = owner_alive(read_owner(path))
    if alive is not None:
        return "live" if alive else "dead"
    # the owner record counts too: the heartbeat rewrites it (touch_owner), so a file server stamps it with its own
    # clock even where it stamps os.utime() with the writer's (SMB)
    return "live" if now - max(mtime, _mtime(path + OWNER_SUFFIX)) < stale_s else "stale"


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def read_owner(partial: str) -> Optional[dict]:
    """The '<partial>.owner' record, or None: there is none (v1, older v2), or it is unreadable or half-written."""
    try:
        return _load_owner(partial)
    except OSError:
        return None


def owner_token(partial: str) -> Optional[str]:
    """The token in '<partial>.owner', or None: there is none, or it is half-written. Raises OSError when it can't be
    read (a share hiccup): then whose the partial is can't be told."""
    return (_load_owner(partial) or {}).get("token")


def _load_owner(partial: str) -> Optional[dict]:
    try:
        with open(partial + OWNER_SUFFIX, "rb") as f:
            record = json.loads(f.read(4096).decode("utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def owner_alive(owner: Optional[dict]) -> Optional[bool]:
    """Is the process that claimed a partial still running? None when that can't be told: no record, a record from
    another computer (another host name, or the same name with another boot time), or a process this user may not
    see (/proc mounted with hidepid)."""
    if not owner or owner.get("host") != HOST:
        return None
    try:
        pid, started, boot = int(owner["pid"]), float(owner["started"]), float(owner["boot"])
    except (KeyError, TypeError, ValueError):
        return None
    if pid <= 0 or abs(boot - _BOOT) > 10:           # two computers of one name (a default Mac name?), or a restart
        return None
    if pid == os.getpid():                           # this process: live while one of its runs holds the claim
        with _live_lock:
            return owner.get("token") in _live
    boot_now = psutil.boot_time()                    # first: psutil 5's create_time() adds the boot time it read last
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        created = proc.create_time()
    except psutil.ZombieProcess:
        return False
    except psutil.NoSuchProcess:
        return None if _pid_exists(pid) else False
    except psutil.Error:                             # no access to it: it runs, so assume it is the owner
        return True
    # The owner if it started when its record says: by the clock, or by the time since boot. Linux computes
    # create_time() from the boot time as the wall clock shows it now, so a step of the clock (an NTP step) moves
    # it; Windows and macOS keep the start time.
    return abs(created - started) < 1.0 or abs((created - boot_now) - (started - boot)) < 1.0


def _pid_exists(pid: int) -> bool:
    """psutil can't see `pid`: does it exist anyway? /proc mounted with hidepid hides other users' processes; signal 0
    then fails with EPERM instead of ESRCH. Not on Windows, where os.kill would end the process."""
    if WINDOWS:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def write_owner(partial: str) -> str:
    """Create '<partial>.owner' with O_EXCL right after claiming `partial`: this computer (name and boot time), this
    process and its start time, and a random token, which stays live in this process until release_owner().
    Raises OSError."""
    token = os.urandom(8).hex()
    with _live_lock:
        _live.add(token)
    record = {"host": HOST, "boot": _BOOT, "pid": os.getpid(), "started": _STARTED, "token": token}
    try:
        fd = _create(partial + OWNER_SUFFIX)
        try:
            os.write(fd, json.dumps(record).encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        release_owner(token)
        raise
    return token


def release_owner(token: str) -> None:
    """The run that holds this token has ended: its partial (if one is left) no longer counts as live."""
    with _live_lock:
        _live.discard(token)


def touch_owner(partial: str) -> None:
    """The heartbeat's data write (D3): rewrite the first byte of '<partial>.owner' in place. A file server stamps a
    write with its own clock, while an SMB client stamps os.utime() with its own, which may be minutes off the share's
    (DS-3). Never creates the record. Raises OSError."""
    with open(partial + OWNER_SUFFIX, "r+b") as f:
        first = f.read(1)
        f.seek(0)
        f.write(first)


def remove_sidecars(partial: str) -> None:
    """Delete '<partial>.owner' and '<partial>.complete', quietly."""
    for suffix in (OWNER_SUFFIX, COMPLETE_SUFFIX):
        try:
            os.remove(partial + suffix)
        except OSError:
            pass


def share_now(folder: str) -> float:
    """The current time by the clock that stamps files in `folder` (a network share's server, DS-3): the mtime of a
    probe file created there just now. This computer's clock if the probe can't be written. A probe whose delete
    fails (a virus scanner holds it) is deleted by the next write_probe."""
    probe = os.path.join(folder, f".yfs-clock-{os.getpid()}-{next(_ids)}.tmp")
    try:
        os.close(_create(probe))
        return os.stat(probe).st_mtime
    except OSError:
        return time.time()
    finally:
        if not _remove_quietly(probe):
            _stuck.add(probe)


def _create(path: str) -> int:
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o644)


def _remove_quietly(path: str) -> bool:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


# ------------------------------------------------------------------------------------- existing outputs (D2)
def verify_output(path: str) -> Tuple[str, str]:
    """D2: the verdict on an existing output, from its headers and IFDs only (R4), and a detail for the UI:
    - 'valid': complete; the detail is 'made by another program' for a pyramid laid out unlike the engine's;
    - 'incomplete': evidence of an interrupted write: an empty file, missing image data, data or a pyramid level
      past the end of the file, a SubIFD never filled in, or an engine-written file with too few levels;
    - 'unreadable': not a TIFF, or a corrupt TIFF structure;
    - 'unchecked': an I/O error (no access, a share hiccup, the file vanished) kept it from being checked."""
    size, raw = -1, None
    try:
        size = os.path.getsize(path)
        raw = _ReadErrors(path)
        with io.BufferedReader(raw) as fh, tifffile.TiffFile(fh) as tif:
            problem = _ifd_chain_problem(tif, size)
            if problem:
                return "incomplete", problem
            if not tif.pages:
                return "unreadable", "it holds no image"
            pages = list(tif.pages)
            verdicts = [_verdict(page, size) for page in pages]
            result = next((v for v in verdicts if v[0] != "valid"), None)
            if result is None:
                problem = _ome_plane_problem(pages)
                result = ("incomplete", problem) if problem else ("valid", next((d for _, d in verdicts if d), ""))
            return result
    except Exception as e:
        os_error = _io_error(e)
        if os_error is not None:
            return "unchecked", map_oserror(os_error).message.rstrip(".")
        return ("incomplete", "the file is empty") if size == 0 else ("unreadable", _text(e))
    finally:
        if raw is not None and raw.error is not None:
            return "unchecked", map_oserror(raw.error).message.rstrip(".")


def _ifd_chain_problem(tif, size: int) -> str:
    """tifffile can log and omit an invalid later IFD. Check the on-disk chain before trusting its page list."""
    fh, big, order = tif.filehandle, tif.is_bigtiff, tif.byteorder
    width, count_width, entry_width = (8, 8, 20) if big else (4, 2, 12)
    offset_format, count_format = order + ("Q" if big else "I"), order + ("Q" if big else "H")
    fh.seek(8 if big else 4)
    raw = fh.read(width)
    if len(raw) != width:
        return "the image directory is cut off"
    offset, seen = struct.unpack(offset_format, raw)[0], set()
    first = True
    while offset:
        if offset in seen:
            return "the image directory chain is damaged"
        seen.add(offset)
        if offset + count_width > size:
            return "" if first else "an image directory is cut off"  # existing unreadable-first-IFD verdict
        first = False
        fh.seek(offset)
        raw = fh.read(count_width)
        if len(raw) != count_width:
            return "an image directory is cut off"
        count = struct.unpack(count_format, raw)[0]
        next_offset = offset + count_width + count * entry_width
        if next_offset + width > size:
            return "an image directory is cut off"
        fh.seek(next_offset)
        raw = fh.read(width)
        if len(raw) != width:
            return "an image directory is cut off"
        offset = struct.unpack(offset_format, raw)[0]
    return ""


def _ome_plane_problem(pages: list) -> str:
    """OME's SizeZ/C/T describes the required planes; packed samples share one IFD. No pixels are read."""
    try:
        root = ET.fromstring(pages[0].description)
    except (ET.ParseError, TypeError, ValueError):
        return ""
    required = 0
    for pixels in root.findall(f"{_OME}Image/{_OME}Pixels"):
        try:
            z, c, t = (int(pixels.get("Size" + axis, "1")) for axis in "ZCT")
            channels = pixels.findall(f"{_OME}Channel")
            samples = [int(ch.get("SamplesPerPixel", "1")) for ch in channels]
            if _engine_written(pages[0]):
                planes = z * t                         # v1 repeats SizeC on every Channel of its packed IFD
            else:
                planes = z * t * (len(samples) if samples and sum(samples) == c else c)
            required += planes
            for data in pixels.findall(f"{_OME}TiffData"):
                start = int(data.get("IFD", "0"))
                count = int(data.get("PlaneCount", "1" if "IFD" in data.attrib else str(planes)))
                if start < 0 or count < 0 or start + count > len(pages):
                    return "an OME image plane is missing"
        except (TypeError, ValueError):
            return "the OME plane description is damaged"
    return "an OME image plane is missing" if required > len(pages) else ""


def _io_error(e: Optional[BaseException]) -> Optional[OSError]:
    """The OSError behind `e`: tifffile turns one met while reading an IFD into TiffFileError('corrupted ...')."""
    for _ in range(8):
        if e is None or isinstance(e, OSError):
            return e
        e = e.__cause__ or e.__context__
    return None


def _verdict(page: "tifffile.TiffPage", size: int) -> Tuple[str, str]:
    problem = _data_problem(page, size)
    if problem:
        return "incomplete", problem
    subs, ours = tuple(page.subifds or ()), _engine_written(page)
    need = pyramid_levels(page.imagelength, page.imagewidth)
    if any(o == 0 or o >= size for o in subs):
        return "incomplete", "a pyramid level was never written"
    if ours and len(subs) < need:
        return "incomplete", f"{len(subs)} of {need} pyramid levels"
    try:
        levels = list(page.pages or ())
    except Exception as e:                           # a pyramid level's IFD is cut off or damaged
        if _io_error(e) is not None:
            raise
        return ("incomplete", "a pyramid level is cut off") if ours else ("unreadable", _text(e))
    if len(levels) != len(subs):
        return "incomplete", "a pyramid level is missing"
    for level, sub in enumerate(levels, 1):
        verdict, problem = _verdict(sub, size)
        if verdict != "valid":
            return verdict, f"pyramid level {level}: {problem}"
    return "valid", "" if ours or len(subs) == need else "made by another program"


def _engine_written(page: "tifffile.TiffPage") -> bool:
    """Is this level 0 as stitcher_RAM.write_to_disk writes it (v1 and v2 alike)? Its OME-XML (generate_visio_xml)
    has quirks no other writer shares: no Creator, and an Interleaved Pixels whose every Channel has
    SamplesPerPixel equal to SizeC."""
    try:
        root = ET.fromstring(page.description)
    except (ET.ParseError, TypeError, ValueError):
        return False
    pixels = root.find(f"{_OME}Image/{_OME}Pixels")
    channels = [] if pixels is None else pixels.findall(f"{_OME}Channel")
    return (root.tag == f"{_OME}OME" and "Creator" not in root.attrib and bool(channels)
            and pixels.get("Interleaved") == "true"
            and all(c.get("SamplesPerPixel") == pixels.get("SizeC") for c in channels))


def _data_problem(page: "tifffile.TiffPage", size: int) -> str:
    counts, offsets = page.databytecounts, page.dataoffsets
    if not counts or len(offsets) != len(counts) or any(c <= 0 for c in counts) or any(o <= 0 for o in offsets):
        return "image data missing"
    if _data_end(page) > size:
        return "the file is cut off"
    return ""


def _data_end(page: "tifffile.TiffPage") -> int:
    return max((o + c for o, c in zip(page.dataoffsets, page.databytecounts)), default=0)


def _text(e: BaseException) -> str:
    if "not a TIFF file" in str(e):
        return "it is not a TIFF image"
    return "its image structure is damaged"


# ------------------------------------------------------------------------------------------ the write probe
def write_probe(out_dir: str, create: bool = True) -> None:
    """Can we write in out_dir? With create, make it first if it is missing (the final check at Start, the browser);
    the runner's per-slide check passes create=False, so a share that vanished mid-run never gets its folder
    re-created on the bare mount point (DS-8). Then create, write and delete a uniquely named
    '.yfs-write-test-<pid>-<n>.tmp'. Creating or writing is tried PROBE_TRIES times, PROBE_DELAY_S apart, before
    the OSError is raised (FS-17); a delete that keeps failing proves nothing (a virus scanner may hold the new
    file): it is retried at the next probe. (os.access is not used: it is unreliable on CIFS ACLs.)"""
    if create:
        os.makedirs(out_dir, exist_ok=True)
    for old in list(_stuck):
        if _remove_quietly(old):
            _stuck.discard(old)
    for attempt in range(1, PROBE_TRIES + 1):
        probe = os.path.join(out_dir, f".yfs-write-test-{os.getpid()}-{next(_ids)}.tmp")
        try:
            fd = _create(probe)
            try:
                os.write(fd, b"yfs")
            finally:
                os.close(fd)
        except OSError:
            _remove_quietly(probe)
            if attempt == PROBE_TRIES:
                raise
            time.sleep(PROBE_DELAY_S)
            continue
        for attempt in range(1, PROBE_TRIES + 1):
            if _remove_quietly(probe):
                return
            if attempt < PROBE_TRIES:
                time.sleep(PROBE_DELAY_S)
        _log.warning("The test file %s could not be deleted yet (it is in use); it will be later.", probe)
        _stuck.add(probe)
        return


# ------------------------------------------------------------------------------------------------ leftovers
class _ScratchCleanup:
    def __init__(self, paths, out_dir, out_identity) -> None:
        self.paths, self.out_dir, self.out_identity = paths, out_dir, out_identity
        self.locks, self.allowed, self.errors = {}, [], []

    def prepare(self) -> None:
        """Output lane only: reserve both possible partial names before checking their ownership."""
        if self.out_identity is None or path_id(self.out_dir) != self.out_identity:
            raise OSError("The save folder changed; refresh the leftover list.")
        pdir = os.path.join(self.out_dir, PARTIAL_DIR)
        problem = partial_dir_problem(pdir)
        if problem:
            raise OSError(problem)
        for path, info in self.paths:
            if not TMP_RAW_RE.search(info.name) or not os.path.basename(os.path.dirname(path)).startswith(TEMP_SUBDIR_PREFIX):
                self.errors.append(f"{info.name}: not a temporary file for this save folder, kept.")
                continue
            partners = _partners(path, self.out_dir)
            try:
                for partner in partners:
                    key = norm_key(partner)
                    if key not in self.locks:
                        self.locks[key] = _open_reservation(partner)
                actual = {}
                try:
                    with os.scandir(pdir) as entries:
                        for entry in entries:
                            actual.setdefault(norm_key(entry.name), []).append(entry.path)
                except FileNotFoundError:
                    pass
                matches = [p for partner in partners for p in
                           actual.get(norm_key(os.path.basename(partner)), [partner])]
                for partner in matches:
                    try:
                        mtime = os.stat(partner).st_mtime
                    except FileNotFoundError:
                        continue
                    if partial_state(partner, mtime, share_now(os.path.dirname(partner))) in ("live", "complete"):
                        raise OSError("another stitcher is using it or a finished image is waiting, kept.")
                self.allowed.append((path, info))
            except OSError as e:
                self.errors.append(f"{info.name}: {map_oserror(e).message}")
        for reservation in self.locks.values():
            reservation.check()
        if path_id(self.out_dir) != self.out_identity:
            raise OSError("The save folder changed; refresh the leftover list.")

    def delete(self) -> Tuple[int, int, List[str]]:
        """Scratch lane only: output reservations stay held; file and directory approval is checked here."""
        count, freed, clocks, folders = 0, 0, {}, {}

        def now(folder):
            if folder not in clocks:
                clocks[folder] = share_now(folder)
            return clocks[folder]

        for path, info in self.allowed:
            try:
                _delete_snapshot(path, info, STALE_PARTIAL_S, None, now, check_partners=False)
                count, freed = count + 1, freed + info.size
                folders[os.path.dirname(path)] = info.parent_identity
            except FileNotFoundError:
                pass
            except OSError as e:
                self.errors.append(f"{info.name}: {map_oserror(e).message}")
        for folder, identity in folders.items():
            try:
                if path_id(folder) == identity:
                    os.rmdir(folder)
            except OSError:
                pass
        return count, freed, self.errors

    def close(self) -> None:
        for reservation in self.locks.values():
            reservation.close()


def delete_scratch(fs, paths: Sequence[Tuple[str, OutputInfo]], *, out_dir: str,
                   out_identity: Optional[Tuple[int, int]]) -> Future:
    """Schedule reservation, deletion and release on their own lanes; no worker waits for another lane.
    A timeout cancels work that has not started. A running delete keeps its reservations until it actually ends,
    even after a terminal timeout, so another stitcher cannot claim a scratch file still being inspected."""
    result, cancelled, lock = Future(), threading.Event(), threading.Lock()
    if not paths:
        result.set_result((0, 0, []))
        return result
    cleanup, started = _ScratchCleanup(paths, out_dir, out_identity), False

    def deliver(future):
        try:
            value = future.result()                     # callback: the future is already finished
        except Exception as e:
            cancelled.set()
            with lock:
                if not result.done():
                    result.set_exception(e)
        else:
            with lock:
                if not result.done():
                    result.set_result(value)

    def release():
        fs.retry(out_dir)
        fs.submit(out_dir, "call", cleanup.close, priority=0)

    def delete():
        nonlocal started
        with lock:
            if cancelled.is_set():
                return 0, 0, []
            started = True
        try:
            return cleanup.delete()
        finally:
            release()

    def deleted(future):
        deliver(future)
        with lock:
            pending = not started
        if pending:                                     # refused or expired in the queue: no worker owns release
            release()

    def prepare():
        handed_off = False
        try:
            cleanup.prepare()
            if not cancelled.is_set():
                future = fs.submit(os.path.dirname(paths[0][0]), "delete", delete, priority=0)
                future.add_done_callback(deleted)
                handed_off = True
        finally:
            if not handed_off:
                cleanup.close()

    def prepared(future):
        if future.cancelled() or future.exception() is not None:
            deliver(future)

    fs.submit(out_dir, "delete", prepare, priority=0).add_done_callback(prepared)
    return result


def delete_leftovers(paths: Sequence[Union[str, Tuple[str, OutputInfo]]], min_age_s: float = STALE_PARTIAL_S,
                     out_dir: Optional[str] = None,
                     out_identity: Optional[Tuple[int, int]] = None) -> Tuple[int, int, List[str]]:
    """Delete leftover files: *.tmp.raw, and inside a real PARTIAL_DIR (not a link or junction, DS-10) partial
    *.ome.tif(f) with their sidecars and '.stale-' remnants of a takeover. Kept: files changed in the last
    min_age_s seconds by their folder's own clock (DS-3), complete partials, and partials or scratch files whose
    partial is live (FS-13, D3); `out_dir` (the save folder) lets that check find the partial of a scratch file
    in a custom temp folder. Emptied PARTIAL_DIRs and scratch subfolders are removed. Returns (n_deleted,
    bytes_freed, error texts). Pass the indexed (path, OutputInfo) pairs and OutputIndex.identity as out_identity.
    Bare path strings have no approval snapshot and are refused."""
    deleted, freed, errors, folders, clocks = 0, 0, [], {}, {}

    def now(folder: str) -> float:
        if folder not in clocks:
            clocks[folder] = share_now(folder)
        return clocks[folder]

    for item in paths:
        path, approved = (item, None) if isinstance(item, str) else item
        name, parent = os.path.basename(path), os.path.dirname(path)
        in_pdir = os.path.basename(parent) == PARTIAL_DIR
        if not (TMP_RAW_RE.search(name) or (in_pdir and (OUTPUT_RE.search(name) or STALE_RE.search(name)))):
            errors.append(f"{name}: not a leftover file, kept.")
            continue
        try:
            if approved is None or approved.identity is None or approved.parent_identity is None:
                errors.append(f"{name}: refresh the leftover list before deleting it, kept.")
                continue
            if (out_dir and out_identity is not None and path_id(out_dir) != out_identity) or \
                    path_id(parent) != approved.parent_identity:
                errors.append(f"{name}: its folder changed; refresh the leftover list, kept.")
                continue
            if in_pdir and partial_dir_problem(parent):
                errors.append(f"{name}: its folder is a link to another place, kept.")
                continue
            st = os.lstat(path)
            if not _same_snapshot(st, approved):
                errors.append(f"{name}: it changed; refresh the leftover list, kept.")
                continue
            why = _why_kept(path, st, min_age_s, out_dir, now)
            if why:
                errors.append(f"{name}: {why}, kept.")
                continue
            partners = _partners(path, out_dir) if TMP_RAW_RE.search(name) else []
            partner = partners[0] if partners else path
            with reserve_output(partner):
                _delete_snapshot(path, approved, min_age_s, out_dir, now)
                if OUTPUT_RE.search(name) and path_id(parent) == approved.parent_identity:
                    remove_sidecars(path)
            deleted, freed = deleted + 1, freed + st.st_size
        except FileNotFoundError:
            pass
        except OSError as e:
            errors.append(f"{name}: {map_oserror(e).message}")
        if in_pdir or os.path.basename(parent).startswith(TEMP_SUBDIR_PREFIX):
            folders[parent] = approved.parent_identity
    for d, expected in folders.items():
        try:
            if path_id(d) != expected:
                continue
            os.rmdir(d)
        except OSError:                              # not empty: other partials are still there
            pass
    return deleted, freed, errors


def _same_snapshot(st: os.stat_result, info: OutputInfo) -> bool:
    return (stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == info.identity
            and st.st_size == info.size and st.st_mtime_ns == info.mtime_ns)


def _delete_snapshot(path: str, info: OutputInfo, min_age_s: float, out_dir: Optional[str], now,
                     check_partners: bool = True) -> None:
    """Recheck at deletion, anchored to the approved directory on systems with dir_fd support."""
    parent, name = os.path.dirname(path), os.path.basename(path)
    fd = None
    try:
        if os.unlink in os.supports_dir_fd:
            fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != info.parent_identity:
                raise OSError("Its folder changed; refresh the leftover list.")
        if path_id(parent) != info.parent_identity or not _same_snapshot(os.lstat(path), info):
            raise OSError("It changed; refresh the leftover list.")
        why = _why_kept(path, os.lstat(path), min_age_s, out_dir, now, check_partners=check_partners)
        if why:
            raise OSError(why + ", kept.")
        if fd is None:
            os.remove(path)
        else:
            if not _same_snapshot(os.stat(name, dir_fd=fd, follow_symlinks=False), info):
                raise OSError("It changed; refresh the leftover list.")
            os.unlink(name, dir_fd=fd)
    finally:
        if fd is not None:
            os.close(fd)


def _why_kept(path: str, st: os.stat_result, min_age_s: float, out_dir: Optional[str],
              now: Callable[[str], float], check_partners: bool = True) -> str:
    """Why a leftover candidate must stay ('' if it may go)."""
    if now(os.path.dirname(path)) - st.st_mtime < min_age_s:
        return f"changed in the last {int(min_age_s)} s (still in use?)"
    if OUTPUT_RE.search(path):
        state = partial_state(path, st.st_mtime, now(os.path.dirname(path)))
        return {"complete": "a finished image waiting to be moved into place",
                "live": "another stitcher is writing it"}.get(state, "")
    if not TMP_RAW_RE.search(path) or not check_partners:
        return ""
    for partner in _partners(path, out_dir):         # a scratch file: is its partial (or v1's output) live?
        try:
            mtime = os.stat(partner).st_mtime
        except OSError:
            continue
        if partial_state(partner, mtime, now(os.path.dirname(partner))) == "live":
            return "another stitcher is using it"
    return ""


def _partners(raw: str, out_dir: Optional[str]) -> List[str]:
    """The partial a scratch file belongs to: next to it (the default location; for v1, its output written in
    place), or in the save folder's PARTIAL_DIR when it lives in a custom temp folder's subfolder (D4)."""
    folder, stem = os.path.dirname(raw), os.path.basename(raw)[:-len(".tmp.raw")]
    if os.path.basename(folder).startswith(TEMP_SUBDIR_PREFIX):
        if not out_dir:
            return []
        folder = os.path.join(out_dir, PARTIAL_DIR)
    return [os.path.join(folder, stem + ext) for ext in (".tif", ".tiff")]


# ------------------------------------------------------------------------------------ moving into place (D3)
def fsync_path(path: str) -> None:
    with open(path, "rb+") as f:                     # writable: Windows can only flush a handle opened for writing
        os.fsync(f.fileno())


def mark_complete(partial: str, gaps: int = 0) -> None:
    """Write '<partial>.complete' once the partial is fsynced (and still ours): from then on it is never taken over
    or deleted as a leftover, only moved into place. It keeps the slide's gaps for the ledger."""
    with open(partial + COMPLETE_SUFFIX, "w", encoding="utf-8") as f:
        json.dump({"gaps": int(gaps)}, f)


def complete_gaps(partial: str) -> int:
    """The gaps recorded by mark_complete (0 if unknown)."""
    try:
        with open(partial + COMPLETE_SUFFIX, encoding="utf-8") as f:
            return int(json.load(f).get("gaps", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def move_into_place(partial: str, final: str, retries: int = 5, *, replace: bool = True) -> None:
    """os.replace the partial onto `final`, then delete its sidecars. A PermissionError (the old file is open in
    QuPath, or a virus scanner holds it) is retried `retries` times, RETRY_DELAY_S apart, then raised. Raises
    OSError."""
    with reserve_output(partial) as reservation:
        for attempt in range(1, max(1, retries) + 1):
            reservation.check()
            try:
                if replace:
                    os.replace(partial, final)
                elif WINDOWS:
                    os.rename(partial, final)         # Windows rename refuses an existing destination
                else:
                    os.link(partial, final)           # atomic no-clobber publication on the same filesystem
                    os.unlink(partial)
                break
            except PermissionError:
                if attempt >= retries:
                    raise
            time.sleep(RETRY_DELAY_S)
        reservation.check()
        remove_sidecars(partial)


def finalize_partial(partial: str, final: str, retries: int = 5, gaps: int = 0) -> None:
    """fsync the partial, mark it complete, then move it into place (above). When the move keeps failing the
    partial stays complete, to be moved later (finalize_kept). Raises OSError."""
    with reserve_output(partial):
        fsync_path(partial)
        mark_complete(partial, gaps)
        move_into_place(partial, final, retries)


class NotChecked(ValueError):
    """finalize_kept couldn't check the finished image (a read error): unlike the other ValueErrors, Try again
    may help."""


def finalize_kept(partial: str, final: str, retries: int = 5, *, replace: bool = True,
                  expected_destination: Any = _UNSPECIFIED) -> Tuple[int, int]:
    """The *Finalize* action (D12): move a complete partial that a run had to keep (the old file was in use) onto
    `final`; returns the final file's (size, mtime_ns) for the ledger. Raises ValueError when it is no longer a
    complete, valid image waiting here (NotChecked, a ValueError, when it couldn't be read: Try again may help),
    and OSError (PermissionError: the old file is still in use) when the move fails; all with a plain-language
    message."""
    with reserve_output(partial) as reservation:
        return _finalize_kept(partial, final, retries, replace, reservation, expected_destination)


def destination_identity(path: str) -> Optional[tuple]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _publication_identity(partial: str) -> tuple:
    st = os.stat(partial)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, owner_token(partial)


def _finalize_kept(partial: str, final: str, retries: int, replace: bool,
                   reservation: OutputReservation, expected_destination: Any) -> Tuple[int, int]:
    try:
        os.stat(partial + COMPLETE_SUFFIX)
    except FileNotFoundError:
        raise ValueError("The finished image isn't waiting to be moved into place any more (another stitcher may "
                         "have moved it, or it was not finished).") from None
    ident = _publication_identity(partial)
    verdict, detail = verify_output(partial)
    reservation.check()
    if _publication_identity(partial) != ident:
        raise ValueError("The finished image changed while it was checked. Rescan before trying again.")
    if verdict == "unchecked":
        raise NotChecked(f"The finished image couldn't be checked ({detail.rstrip('.')}). Try again.")
    if verdict != "valid":
        _remove_quietly(partial + COMPLETE_SUFFIX)     # a rescan must offer to stitch this image again
        raise ValueError(f"The finished image is damaged ({detail}), so it was not moved into place.")
    if expected_destination is not _UNSPECIFIED and destination_identity(final) != expected_destination:
        raise FileExistsError("The destination changed. Both images were kept; review the destination again.")
    move_into_place(partial, final, retries, replace=replace)
    st = os.stat(final)
    reservation.check()
    try:
        os.rmdir(os.path.dirname(partial))
    except OSError:
        pass
    return st.st_size, st.st_mtime_ns
