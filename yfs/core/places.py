"""Mounted volumes (lane keys and classes), the places sidebar, breadcrumbs and cleaning of pasted paths.

FINAL_SPEC §2.2.4, §2.2.5 and §5.5.2. Building the mount table reads only system tables, never user paths.
"""
from __future__ import annotations

import logging
import ntpath
import os
import posixpath
import re
import string
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import psutil

from .model import CLOUD, LOCAL, NETWORK
from .names import PSEUDO_FSTYPES, natural_key

_FLAVOR = "win32" if sys.platform == "win32" else "darwin" if sys.platform == "darwin" else "linux"
NETWORK_FSTYPES = frozenset({"cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs", "sshfs", "9p", "davfs",
                             "afpfs", "webdav", "autofs"})   # autofs: only when nothing is mounted on top
CLOUD_FSTYPES = frozenset({"fuse.rclone", "rclone", "fuse.gvfsd-fuse", "gvfsd-fuse"})
GVFS_FSTYPES = frozenset({"fuse.gvfsd-fuse", "gvfsd-fuse"})   # one folder per backend, classed by its name
GVFS_NETWORK = ("smb-share:", "sftp:", "dav:", "davs:", "afp-volume:", "nfs:", "ftp:")   # other backends: cloud
MACFUSE_FSTYPES = frozenset({"macfuse", "osxfuse", "osxfusefs"})   # macOS FUSE: rclone is cloud, sshfs network
LOCAL_DISK_FSTYPES = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "ntfs", "ntfs3", "fuseblk", "exfat", "vfat"})
_LOOSE_ROOTS = (("Volumes",), ("mnt",), ("media", ""), ("run", "media", ""))   # where volumes get mounted; "": any
_ONEDRIVE_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")
_DRIVE_LABELS = {2: "Removable Disk", 3: "Local Disk", 5: "CD Drive", 6: "RAM Disk"}
_KNOWN_FOLDERS = (("Desktop", "desktop", "B4BFCC3A-DB2C-424C-B029-7FE99A87C641"),
                  ("Documents", "documents", "FDD39AD0-238F-46AF-ADB4-6C85480369C7"),
                  ("Downloads", "downloads", "374DE290-123F-4565-9164-39C4925E467B"))
_XDG_ICONS = {"DESKTOP": "desktop", "DOCUMENTS": "documents", "DOWNLOAD": "downloads"}
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mount:
    mount_point: str; fstype: str; source: str; cls: str; label: str
    indirect: bool = False  # autofs indirect map (/net): its direct children are the automount points


@dataclass(frozen=True)
class Place:
    section: str; label: str; path: str; icon: str; lane_class: str   # section: recent|computer|network|cloud|bookmarks
    auto_probe: bool        # False for cloud, disconnected drives, macOS TCC folders, autofs not mounted
    detail: str = ""        # tooltip (UNC, source, fstype)


def _mod(flavor: str):
    return ntpath if flavor == "win32" else posixpath


def _norm(path: str, flavor: str) -> str:
    path = _windows_volume_path(path) if flavor == "win32" else path
    p = _mod(flavor).normpath(path) + ("\\" if flavor == "win32" and re.match(r"^[A-Za-z]:$", path) else "")
    return p.casefold() if flavor != "linux" else p


def _windows_volume_path(path: str) -> str:
    """Ordinary spelling for volume comparisons only; callers keep the original path for I/O."""
    p = path.replace("/", "\\")
    if p[:8].casefold() == "\\\\?\\unc\\":
        return "\\\\" + p[8:]
    return p[4:] if re.match(r"\\\\\?\\[A-Za-z]:", p) else p


def _under(path: str, root: str, flavor: str) -> bool:
    """True if `path` is `root` or inside it (string work only)."""
    a, r = _norm(path, flavor), _norm(root, flavor)
    sep = "\\" if flavor == "win32" else "/"
    return a == r or a.startswith(r if r.endswith(sep) else r + sep)


def _below(path: str, root: str, flavor: str) -> List[str]:
    """The components of `path` below `root`, which holds it ([] for root itself), as _norm spells them."""
    sep = "\\" if flavor == "win32" else "/"
    return [p for p in _norm(path, flavor)[len(_norm(root, flavor)):].split(sep) if p]


def _parts(path: str) -> List[str]:
    """The folder names of a POSIX path."""
    return [p for p in posixpath.normpath(path).split("/") if p]


def _unescape(s: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def _make_mount(mp: str, fstype: str, source: str, flavor: str = "linux", options: str = "") -> Mount:
    base, cls = _mod(flavor).basename(mp.rstrip("/")) or "Computer", _fs_class(fstype, source)
    if cls == CLOUD:
        label = base + " (cloud)"
    elif cls == NETWORK:
        label = base + (" (ssh)" if "sshfs" in fstype else "")
    else:
        label = base if mp == "/" else base + " disk"
    indirect = fstype == "autofs" and (flavor == "darwin" or "indirect" in options.split(","))   # macOS: /net, /home
    return Mount(mp, fstype, source, cls, label, indirect)


def _fs_class(fstype: str, source: str) -> str:
    if fstype in MACFUSE_FSTYPES:                    # a disk, an rclone remote, or sshfs and the like
        return LOCAL if source.startswith("/dev/") else CLOUD if _rclone_source(source) else NETWORK
    if fstype in CLOUD_FSTYPES:
        return CLOUD
    return NETWORK if fstype in NETWORK_FSTYPES or fstype.startswith("nfs") else LOCAL


def _rclone_source(source: str) -> bool:
    """An rclone remote ('teamdocs:', 'remote:Lab') rather than an sshfs host ('user@host:/x', 'host:/x')."""
    return "rclone" in source.casefold() or re.match(r"^[^/@:]+:(?!/)", source) is not None


def _unc_parts(text: str) -> Optional[List[str]]:
    """['host', 'share', ...] for \\\\host\\share\\x, //host/share/x or smb://[user@]host[:port]/share/x, else None.
    Only smb:// is a URI: its parts are percent-decoded and its host loses user@ and :port. UNC parts are kept
    as typed ('Cells_10%FBS', 'tenant.example.org@SSL')."""
    t = _windows_volume_path(text.strip()) if not text.lower().startswith("smb://") else text.strip()
    if t.startswith("\\\\?\\"):
        return None
    uri = t.lower().startswith("smb://")
    t = ("//" + t[6:] if uri else t).replace("\\", "/")
    if not t.startswith("//") or t.startswith("///"):
        return None
    parts = [x for x in t[2:].split("/") if x]
    if uri and parts:
        parts[0] = parts[0].rsplit("@", 1)[-1].split(":")[0]     # drop user[:pass]@ and :port
        parts = [unquote(x) for x in parts]
    return parts or None


def _loose_root(path: str) -> Optional[str]:
    """'/mnt/<x>', '/Volumes/<x>', '/media/<u>/<x>' or '/run/media/<u>/<x>' holding `path` (POSIX), else None."""
    parts = _parts(path)
    for pattern in _LOOSE_ROOTS:
        n = len(pattern) + 1
        if len(parts) >= n and all(not want or want == got for want, got in zip(pattern, parts)):
            return "/" + "/".join(parts[:n])
    return None


def _gvfs_label(name: str) -> str:
    """'labshare on fileserver' for a gvfs folder 'smb-share:server=fileserver,share=labshare'; 'user@host' or
    the host for backends without a share or volume."""
    kind, _, rest = name.partition(":")
    spec = {}
    for item in rest.split(","):
        key, _, value = item.partition("=")
        spec[key] = unquote(value)
    host, share = spec.get("server") or spec.get("host") or kind, spec.get("share") or spec.get("volume")
    return f"{share} on {host}" if share else f"{spec['user']}@{host}" if spec.get("user") else host


def _list_gvfs(root: str, timeout_s: float = 1.0) -> List[str]:
    """The backend folders in the gvfs root, which the local gvfs daemon serves; [] if it takes too long."""
    box: List[List[str]] = []

    def work() -> None:
        try:
            box.append(sorted(os.listdir(root)))
        except OSError:
            pass

    t = threading.Thread(target=work, name="yfs-gvfs", daemon=True)
    t.start()
    t.join(timeout_s)
    return box[0] if box else []


class MountTable:
    """Snapshot of mounted volumes. Building it does no I/O on user paths."""

    def __init__(self, mounts: Iterable[Mount] = (), *, flavor: Optional[str] = None,
                 disconnected: Iterable[str] = (), gvfs: Optional[Iterable[Tuple[str, str]]] = None) -> None:
        """flavor: 'linux' | 'darwin' | 'win32' (default: this OS); disconnected: Windows drives ('Y:')."""
        self.flavor = flavor or _FLAVOR
        top = {_norm(m.mount_point, self.flavor): m for m in mounts}   # the last mount on a point is on top
        self._mounts = tuple(top.values())
        self._by_len = sorted(top.items(), key=lambda kv: -len(kv[0]))
        self._disconnected = frozenset(d.casefold() for d in disconnected)
        self._gvfs = None if gvfs is None else tuple(sorted(gvfs))   # (root, backend name), read off the UI thread
        home = os.path.expanduser("~")
        self._cloud = [os.path.join(home, "Library", "Mobile Documents", "com~apple~CloudDocs"),
                       os.path.join(home, "Library", "CloudStorage")] if self.flavor == "darwin" else []
        # OneDrive (with Known Folder Move also Desktop and Documents) keeps its files on the local disk: a lane of
        # its own with the local class, and cloud_synced() for the output warning
        self._onedrive = [os.environ[v] for v in _ONEDRIVE_VARS if os.environ.get(v)] if self.flavor == "win32" else []

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MountTable) and self._state() == other._state()

    def __hash__(self) -> int:
        return hash(self._state())

    def _state(self) -> tuple:
        return self.flavor, self._mounts, self._disconnected, tuple(self._cloud), tuple(self._onedrive), self._gvfs

    def with_gvfs(self, timeout_s: float = 0.5) -> "MountTable":
        """Worker only: collect the mounted backend names without visiting their destinations."""
        roots = [m.mount_point for m in self._mounts if m.fstype in GVFS_FSTYPES]
        backends = [(root, name) for root in roots
                    for name in _list_gvfs(root, timeout_s / max(1, len(roots)))]
        return MountTable(self._mounts, flavor=self.flavor, disconnected=self._disconnected, gvfs=backends)

    @classmethod
    def load(cls, timeout_s: float = 1.0) -> "MountTable":
        """Linux /proc/self/mountinfo; macOS psutil.disk_partitions(all=True) (getfsstat MNT_NOWAIT); Windows
        GetLogicalDrives/GetDriveTypeW/WNetGetConnectionW. Any thread; an empty table if it takes too long."""
        box: List[MountTable] = []

        def work() -> None:
            try:
                if _FLAVOR == "linux":
                    with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
                        box.append(cls.from_mountinfo_text(f.read()))
                    box[0] = box[0].with_gvfs(timeout_s / 2)
                elif _FLAVOR == "darwin":
                    box.append(cls([_make_mount(p.mountpoint, p.fstype, p.device, "darwin")
                                    for p in psutil.disk_partitions(all=True)], flavor="darwin"))
                else:
                    box.append(cls._load_windows())
            except Exception as e:                   # best effort: an empty table treats everything as local
                _log.warning("Could not read the mount table: %s", e)

        t = threading.Thread(target=work, name="yfs-mounts", daemon=True)
        t.start()
        t.join(timeout_s)
        return box[0] if box else cls()

    @classmethod
    def _load_windows(cls) -> "MountTable":
        import ctypes
        k32, mpr = ctypes.windll.kernel32, ctypes.windll.mpr
        mounts, disconnected, mask = [], [], k32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            kind = k32.GetDriveTypeW(f"{letter}:\\") if mask >> i & 1 else 0
            if kind == 4:                             # DRIVE_REMOTE; 1201 = remembered but disconnected
                buf, size = ctypes.create_unicode_buffer(1024), ctypes.c_ulong(1024)
                if mpr.WNetGetConnectionW(f"{letter}:", buf, ctypes.byref(size)) == 1201:
                    disconnected.append(f"{letter}:")
                unc = _unc_parts(buf.value)                   # 'Z: labshare' (the share, or the mapped folder)
                label = f"{letter}: {unc[-1]}" if unc else f"{letter}:"
                mounts.append(Mount(f"{letter}:\\", "remote", buf.value, NETWORK, label))
            elif kind in _DRIVE_LABELS:
                mounts.append(Mount(f"{letter}:\\", "local", "", LOCAL, f"{_DRIVE_LABELS[kind]} ({letter}:)"))
        return cls(mounts, flavor="win32", disconnected=disconnected)

    @classmethod
    def from_mountinfo_text(cls, text: str) -> "MountTable":
        """Parse /proc/self/mountinfo (octal escapes such as \\040 decoded). For tests and the Linux loader."""
        mounts = []
        for line in text.splitlines():
            left, sep, right = line.partition(" - ")
            f, r = left.split(), right.split()
            if sep and len(f) >= 5 and len(r) >= 2:
                mounts.append(_make_mount(_unescape(f[4]), r[0], _unescape(r[1]), options=r[2] if len(r) > 2 else ""))
        return cls(mounts, flavor="linux")

    def mounts(self) -> Tuple[Mount, ...]:
        return self._mounts

    def mount_for(self, path: str) -> Optional[Mount]:
        """Longest-prefix match on the string path (no realpath)."""
        return next((m for mp, m in self._by_len if _under(path, mp, self.flavor)), None)

    def _cloud_root(self, path: str) -> Optional[str]:
        root = next((r for r in self._cloud if _under(path, r, self.flavor)), None)
        if root is None or not root.endswith("CloudStorage"):
            return root
        rest = _below(path, root, self.flavor)
        return os.path.join(root, rest[0]) if rest else root    # one lane per provider folder

    def _volume(self, path: str) -> Tuple[str, str]:
        """POSIX: (lane key, lane class) of `path`. A gvfs backend folder is a volume of its own; so is a folder
        where volumes get mounted (/mnt/<x>, /Volumes/<x>...) that this table doesn't know yet, on a local disk or
        an automount map, so that a share mounted after the table was read never shares the lane of the disk it is
        mounted on. Inside a known share or cloud mount, such folders are just folders of that volume."""
        cloud = self._cloud_root(path)
        if cloud:
            return cloud, CLOUD
        m = self.mount_for(path)
        root, cls = (m.mount_point, m.cls) if m else ("/", LOCAL)
        backend = _below(path, root, self.flavor)[:1] if m is not None and m.fstype in GVFS_FSTYPES else []
        if backend:
            return posixpath.join(root, backend[0]), NETWORK if backend[0].startswith(GVFS_NETWORK) else CLOUD
        loose = _loose_root(path) if cls == LOCAL or m is not None and m.indirect else None
        if loose and _under(loose, root, self.flavor) and _below(loose, root, self.flavor):   # below the mount
            return loose, cls                         # an automount map's share keeps the map's class
        return root, cls

    def lane_key(self, path: str) -> str:
        """Mount point | 'X:' | '\\\\server\\share' (casefolded) | 'onedrive' | a gvfs backend folder | the
        /mnt/<x>, /Volumes/<x>, /media/<u>/<x> or /run/media/<u>/<x> folder of a path the table doesn't know, on a
        local disk or an automount map."""
        if self.flavor == "win32":
            path = _windows_volume_path(path)
            unc = _unc_parts(path) if path.startswith(("\\\\", "//")) else None
            if any(_under(path, r, self.flavor) for r in self._onedrive):
                return "onedrive"
            return ("\\\\" + "\\".join(unc[:2])).casefold() if unc else ntpath.splitdrive(path)[0].upper() or "local"
        return self._volume(path)[0]

    def lane_class(self, path: str) -> str:
        """network (cifs, nfs, sshfs, mapped drive, UNC, gvfs smb/sftp/dav...), cloud (rclone, iCloud, CloudStorage,
        other gvfs backends) or local (also OneDrive, whose files are on the local disk)."""
        if self.flavor != "win32":
            return self._volume(path)[1]
        path = _windows_volume_path(path)
        if path.startswith(("\\\\", "//")):
            return NETWORK
        m = self.mount_for(path)
        return m.cls if m else LOCAL

    def is_system_volume(self, path: str) -> bool:
        if self.flavor == "win32":
            return ntpath.splitdrive(_windows_volume_path(path))[0].upper() == os.environ.get("SystemDrive", "C:").upper()
        m = self.mount_for(path)
        return m is not None and m.mount_point == "/"

    def cloud_synced(self, path: str) -> bool:
        """True for folders whose files sync to a cloud service (OneDrive, iCloud, CloudStorage, rclone, gvfs cloud
        backends): the output warning. Unlike the lane class, this includes OneDrive."""
        return self.lane_class(path) == CLOUD or any(_under(path, r, self.flavor) for r in self._onedrive)

    def place_name(self, path: str) -> str:
        """What holds `path`, for messages: the share or cloud folder of a network or cloud path ('labshare',
        'labshare on fileserver', 'Z:'), else the folder's own name."""
        mod = _mod(self.flavor)
        if self.lane_class(path) == LOCAL:
            return mod.basename(mod.normpath(path).rstrip("/\\")) or path
        if self.flavor == "win32":                    # a UNC share, or a mapped drive ('Z:')
            unc = _unc_parts(path)
            return (unc[1] if len(unc) > 1 else unc[0]) if unc else self.lane_key(path)
        key, m = self.lane_key(path), self.mount_for(path)
        depth, parts = len(_parts(key)), _parts(path)   # the key's last folder as the path spells it: macOS
        name = parts[depth - 1] if 0 < depth <= len(parts) else key   # keys of cloud folders are casefolded
        if m is not None and m.fstype in GVFS_FSTYPES and key != m.mount_point:
            return _gvfs_label(name)
        return "iCloud Drive" if name.casefold() == "com~apple~clouddocs" else name

    def unc_to_local(self, text: str) -> Optional[str]:
        """\\\\host\\share\\x or smb://host/share/x -> local mount path via the mount source (case-insensitive)."""
        want, best = _unc_parts(text), None
        scheme = "smb:" if self.flavor == "darwin" else ""    # macOS smbfs sources are URLs: //user@host/My%20Share
        sources = [(_unc_parts(scheme + m.source), m.mount_point) for m in self._mounts if m.cls == NETWORK]
        for root, name in self._gvfs or ():
            if name.startswith("smb-share:"):
                spec = dict(item.split("=", 1) for item in name.partition(":")[2].split(",") if "=" in item)
                if spec.get("server") and spec.get("share"):
                    sources.append(([unquote(spec["server"]), unquote(spec["share"])], posixpath.join(root, name)))
        for src, mount_point in sources:
            if want and src and [s.casefold() for s in src] == [w.casefold() for w in want[:len(src)]] \
                    and len(src) <= len(want) and (best is None or len(src) > len(best[0])):
                best = (src, mount_point)
        return _mod(self.flavor).join(best[1], *want[len(best[0]):]) if best else None

    def disconnected(self, path: str) -> bool:
        """Windows: a mapped drive whose WNetGetConnectionW returned 1201."""
        return self.flavor == "win32" and self.lane_key(path).casefold() in self._disconnected

    def auto_enter(self, path: str) -> Optional[str]:
        """Why `path` must NOT be listed or probed automatically, or None. String work only; a user's explicit
        navigation ignores it. 'disconnected': a Windows 1201 drive of this snapshot (FsService.auto_enter follows
        the live lane instead); 'cloud': a cloud lane; 'autofs': an automount point with nothing mounted on it, or
        a direct child of an indirect map's root (anything deeper is reached through a mount that already
        happened); 'pseudo': the mount point of a pseudo filesystem."""
        if self.disconnected(path):
            return "disconnected"
        if self.lane_class(path) == CLOUD:
            return "cloud"
        m = self.mount_for(path)
        if m is None:
            return None
        depth = len(_below(path, m.mount_point, self.flavor))
        if m.fstype == "autofs" and depth <= (1 if m.indirect else 0):
            return "autofs"
        return "pseudo" if m.fstype in PSEUDO_FSTYPES and depth == 0 else None


def _place(m: MountTable, section: str, label: str, path: str, icon: str, detail: str = "") -> Place:
    probe = m.auto_enter(path) is None and not (m.flavor == "darwin" and _tcc(path))
    return Place(section, label, path, icon, m.lane_class(path), probe, detail)


def list_places(mounts: MountTable, recent: Sequence[Mapping[str, Any]], mode: str) -> List[Place]:
    """Worker thread. §2.2.4 sections; reads only local tables, small config files and the gvfs root (listed by
    the local gvfs daemon, at most 1 s), never stats a network or cloud mount point. `recent` is the mode's list
    of {"path", "slides"} entries (newest first)."""
    out, home, fl = [], os.path.expanduser("~"), mounts.flavor
    for r in [r for r in recent if r.get("path")][:8]:
        p, n = r["path"], r.get("slides") if mode == "input" else None
        name = _mod(fl).basename(p.rstrip("/\\")) or p
        label = f"{name} · {n} slide{'s' if n != 1 else ''}" if n else name
        out.append(_place(mounts, "recent", label, p, "recent", p))
    computer, others = [_place(mounts, "computer", "Home", home, "home")], []
    if fl == "win32":
        computer += [_place(mounts, "computer", name, p, icon) for name, icon, p in _windows_known_folders()]
        for mt in mounts.mounts():
            if mt.cls == NETWORK:
                others.append(Place("network", mt.label, mt.mount_point, "drive_net", NETWORK,
                                    not mounts.disconnected(mt.mount_point), mt.source))
            else:
                computer.append(_place(mounts, "computer", mt.label, mt.mount_point, "drive"))
        for unc in filter(None, (_unc_parts(r.get("path") or "") for r in recent)):   # UNC roots used before
            if len(unc) >= 2:
                others.append(_place(mounts, "network", f"{unc[1]} on {unc[0]}", "\\\\" + "\\".join(unc[:2]),
                                     "drive_net"))
        if os.environ.get("OneDrive"):
            others.append(_place(mounts, "cloud", "OneDrive", os.environ["OneDrive"], "drive_cloud"))
    elif fl == "darwin":
        computer += [_place(mounts, "computer", n, os.path.join(home, n), n.lower())
                     for n in ("Desktop", "Documents", "Downloads")]
        for mt in mounts.mounts():
            if mt.mount_point.startswith("/Volumes/"):
                section, icon = {NETWORK: (NETWORK, "drive_net"), CLOUD: (CLOUD, "drive_cloud")}.get(
                    mt.cls, ("computer", "drive"))
                (computer if section == "computer" else others).append(_place(mounts, section, mt.label,
                                                                              mt.mount_point, icon, mt.source))
        others += [Place("cloud", label, p, "drive_cloud", CLOUD, False) for label, p in
                   zip(("iCloud Drive", "Cloud storage"), mounts._cloud) if os.path.isdir(p)]
    else:
        computer += _xdg_places(mounts, home) + [_place(mounts, "computer", "Computer", "/", "drive")]
        roots = ("/mnt/", "/media/", "/run/media/", "/home/", home.rstrip("/") + "/")
        for mt in mounts.mounts():
            mp, detail = mt.mount_point, f"{mt.source} · {mt.fstype}"
            if mt.fstype in LOCAL_DISK_FSTYPES and mp.startswith(roots) and mp != home:
                computer.append(_place(mounts, "computer", mt.label, mp, "drive", detail))
            elif mt.cls != LOCAL and not mp.startswith(("/proc/", "/sys/", "/dev/")):
                label = mt.label + (" (not mounted)" if mt.fstype == "autofs" else "")
                others.append(_place(mounts, mt.cls, label, mp, "drive_cloud" if mt.cls == CLOUD else "drive_net",
                                     detail))
            if mt.fstype in GVFS_FSTYPES:              # shares connected in the file manager (Files / Nautilus)
                others += [_place(mounts, NETWORK, _gvfs_label(n), posixpath.join(mp, n), "drive_net", n)
                           for n in (_list_gvfs(mp) if mounts._gvfs is None else
                                     [n for root, n in mounts._gvfs if root == mp]) if n.startswith(GVFS_NETWORK)]
        others += _gtk_bookmarks(mounts, home)
    others.sort(key=lambda p: (("network", "cloud", "bookmarks").index(p.section),
                               () if p.section == "bookmarks" else natural_key(p.label)))
    return out + computer + others


def _config_lines(home: str, *parts: str) -> List[str]:
    """Lines of a small file under $XDG_CONFIG_HOME (or ~/.config); [] if it is missing."""
    try:
        with open(os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"), *parts),
                  encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def _xdg_places(m: MountTable, home: str) -> List[Place]:
    """Configured XDG folders except $HOME. Destinations are checked only on navigation."""
    out = []
    for hit in filter(None, (re.match(r'^\s*XDG_(\w+)_DIR\s*=\s*"(.*)"\s*$', ln)
                             for ln in _config_lines(home, "user-dirs.dirs"))):
        p = os.path.normpath(re.sub(r"\\(.)", r"\1", hit.group(2)).replace("$HOME", home))
        if p != os.path.normpath(home) and os.path.isabs(p):
            out.append(_place(m, "computer", os.path.basename(p), p, _XDG_ICONS.get(hit.group(1), "folder")))
    return out


def _gtk_bookmarks(m: MountTable, home: str) -> List[Place]:
    """Read-only import of gtk-3.0/bookmarks: file:// URIs only, with their labels."""
    out = []
    for uri, _, label in (ln.strip().partition(" ") for ln in _config_lines(home, "gtk-3.0", "bookmarks")):
        if uri.startswith("file://"):
            p = unquote(urlparse(uri).path)
            out.append(_place(m, "bookmarks", label.strip() or os.path.basename(p.rstrip("/")) or p, p, "bookmark"))
    return out


def _windows_known_folders() -> List[Tuple[str, str, str]]:
    """Desktop, Documents and Downloads via SHGetKnownFolderPath (follows OneDrive redirection)."""
    out = []
    try:
        import ctypes

        class GUID(ctypes.Structure):
            _fields_ = [("data", ctypes.c_ubyte * 16)]

        for name, icon, guid in _KNOWN_FOLDERS:
            g, p = GUID.from_buffer_copy(uuid.UUID(guid).bytes_le), ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) == 0:
                out.append((name, icon, p.value))
                ctypes.windll.ole32.CoTaskMemFree(p)
    except Exception as e:                           # a missing or odd API must not break the sidebar
        _log.warning("Known folders unavailable: %s", e)
    return out


def windows_volume_label(root: str) -> str:
    """Windows: the volume label of a drive root ('' if none). Blocking: run it as a lane call (deadline 5 s)."""
    import ctypes
    buf = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(root, buf, 261, None, None, None, None, 0)
    return buf.value if ok else ""


def _looks_windows(path: str) -> bool:
    return bool(re.match(r"^(?:[A-Za-z]:|\\\\)", path))


def containing_place(path: str, places: Sequence[Place]) -> Optional[Place]:
    """The computer/network/cloud place with the longest path that contains `path`."""
    fl = "win32" if _looks_windows(path) else _FLAVOR
    hits = [p for p in places if p.section in ("computer", "network", "cloud") and _under(path, p.path, fl)]
    return max(hits, key=lambda p: len(p.path), default=None)


def breadcrumb_parts(path: str, place: Optional[Place]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """(crumbs [(label, path)] starting at the place, hidden ancestors [(label, path)]). Pure string work."""
    fl = "win32" if _looks_windows(path) else _FLAVOR
    p, chain, cur = _mod(fl), [], _mod(fl).normpath(path)
    while True:
        head, tail = p.split(cur)
        chain.insert(0, (tail or cur, cur))
        if not tail or head == cur:
            break
        cur = head
    i = next((k for k, (_, q) in enumerate(chain) if place and _norm(q, fl) == _norm(place.path, fl)), None)
    return (chain, []) if i is None else ([(place.label, chain[i][1])] + chain[i + 1:], chain[:i])


def normalize_user_path(text: str, mounts: MountTable) -> str:
    """Clean a typed or pasted path (§5.5.2): first non-empty line, quotes, file:// URIs, ~ and variables,
    UNC/smb:// -> local mount (POSIX) or \\\\host\\share (Windows), trailing separators; Windows 'D:' -> 'D:\\'.
    Pure (no I/O). A second pass leaves ordinary paths, UNC and smb:// text unchanged; odd text may change again
    (nested quotes, URIs or variables, mixed separators, spaces decoded at the ends). Relative results stay
    relative: callers refuse them."""
    win, s = mounts.flavor == "win32", next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.lower().startswith("file:"):
        u = urlparse(s)
        p = unquote(u.path)
        if u.netloc and u.netloc.lower() != "localhost":
            s = "//" + u.netloc + p                   # file://host/share/x is a UNC path
        else:
            s = p[1:] if re.match(r"^/[A-Za-z]:", p) else p   # file:///C:/x
    s = re.sub(r"%([^%/\\]+)%", lambda m: os.environ.get(m.group(1), m.group(0)), s)
    s = os.path.expanduser(os.path.expandvars(s))
    unc = _unc_parts(s) if s.lower().startswith(("smb://", "\\\\", "//")) else None
    if unc and not (win and s.startswith("\\\\?\\")):
        s = "\\\\" + "\\".join(unc) if win else mounts.unc_to_local(s) or s
    s, seps = s.replace("/", "\\") if win else s, "\\/" if win else "/"
    while len(s) > 1 and s[-1] in seps and _mod(mounts.flavor).splitdrive(s)[1].rstrip(seps):
        s = s[:-1]
    if win and re.match(r"^[A-Za-z]:$", s):
        s += "\\"                                     # the drive, not the current folder of that drive
    return s


def tcc_protected(path: str) -> bool:
    """macOS: Desktop, Documents, Downloads, iCloud, CloudStorage and /Volumes (removable and network)."""
    return _FLAVOR == "darwin" and _tcc(path)


def _tcc(path: str) -> bool:
    home = os.path.expanduser("~")
    roots = [os.path.join(home, n) for n in ("Desktop", "Documents", "Downloads")] + [
        os.path.join(home, "Library", "Mobile Documents"), os.path.join(home, "Library", "CloudStorage"), "/Volumes"]
    return any(_under(path, r, "darwin") for r in roots)
