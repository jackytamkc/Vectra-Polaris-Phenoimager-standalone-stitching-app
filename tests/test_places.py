"""places.py: mount tables, lane keys/classes, auto_enter (R27), UNC mapping, the places list, breadcrumbs and
pasted-path cleaning (FINAL_SPEC §2.2.4, §2.2.5, §5.5.2; acceptance B4). Fixtures are synthetic."""
from __future__ import annotations

import collections
import os
import threading
import time

import pytest

from conftest import FIXTURES
from yfs.core import places as P
from yfs.core.model import CLOUD, LOCAL, NETWORK
from yfs.core.places import Mount, MountTable, Place

IMG = "/home/labuser/mnt/Imaging Share"


@pytest.fixture
def ws():
    return MountTable.from_mountinfo_text((FIXTURES / "mountinfo_workstation.txt").read_text(encoding="utf-8"))


@pytest.fixture
def laptop():
    return MountTable.from_mountinfo_text((FIXTURES / "mountinfo_laptop.txt").read_text(encoding="utf-8"))


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setenv("OneDrive", r"C:\Users\labuser\OneDrive - Example University")
    monkeypatch.setenv("SystemDrive", "C:")
    return MountTable([Mount("C:\\", "local", "", LOCAL, "Local Disk (C:)"),
                       Mount("Z:\\", "remote", r"\\fileserver.example.ac.uk\groups\labshare-GROUP", NETWORK,
                             "Z: labshare-GROUP"),
                       Mount("Y:\\", "remote", r"\\nas.example.org\old", NETWORK, "Y: old")],
                      flavor="win32", disconnected=["Y:"])


def test_mountinfo_classes_labels_and_lane_keys(ws):
    by = {m.mount_point: m for m in ws.mounts()}
    expect = {"/": (LOCAL, "Computer"), "/mnt/labshare": (NETWORK, "labshare"), "/mnt/nas10g": (NETWORK, "nas10g"),
              "/mnt/hpc": (NETWORK, "hpc (ssh)"), "/mnt/scratch": (NETWORK, "scratch"), IMG: (NETWORK, "Imaging Share"),
              "/home/labuser/mnt/clouddrive": (CLOUD, "clouddrive (cloud)"), "/home/labuser/mnt/laptop":
                  (NETWORK, "laptop (ssh)"), "/home/Data8TB": (LOCAL, "Data8TB disk"), "/run/user/1000/gvfs":
                  (CLOUD, "gvfs (cloud)"), "/mnt/archive": (NETWORK, "archive"), "/media/labuser/USB STICK":
                  (LOCAL, "USB STICK disk")}
    for mp, (cls, label) in expect.items():
        assert (by[mp].cls, by[mp].label) == (cls, label), mp
    assert by["/mnt/labshare"].fstype == "cifs" and by["/mnt/scratch"].fstype == "autofs"   # mount on top wins
    assert by[IMG].source == "//fileserver.example.ac.uk/groups/Imaging Share"             # \040 decoded
    assert by["/proc/sys/fs/binfmt_misc"].fstype == "binfmt_misc"
    assert ws.lane_key(IMG + "/Group/Project") == IMG and ws.lane_class(IMG + "/Group") == NETWORK
    assert ws.lane_key("/home/labuser/docs") == "/" and ws.lane_class("/home/labuser/docs") == LOCAL
    assert ws.mount_for("/mnt/labshare/UserA").mount_point == "/mnt/labshare"
    assert ws.lane_key("/home/Data8TB/WSIs") == "/home/Data8TB"
    assert ws.is_system_volume("/home/labuser") and not ws.is_system_volume("/home/Data8TB/x")


@pytest.mark.parametrize("path,why", [
    ("/home/labuser/mnt/clouddrive", "cloud"), ("/home/labuser/mnt/clouddrive/a/b", "cloud"),
    ("/run/user/1000/gvfs", "cloud"), ("/mnt/scratch", "autofs"), ("/proc", "pseudo"), ("/sys", "pseudo"),
    ("/proc/sys", None), ("/run/user/1000/doc", "pseudo"), ("/snap/core24/1643", "pseudo"),
    ("/var/lib/docker/overlay2/3f2a9c/merged", "pseudo"), ("/run/netns/ns1", "pseudo"), ("/mnt/labshare", None),
    (IMG, None), ("/mnt/hpc/x", None), ("/home/Data8TB", None), ("/home/labuser", None),
    ("/mnt/scratch/Project", None), ("/mnt/scratch/Project/Slide1_Scan1", None)])   # below it: mounted by then
def test_auto_enter_workstation(ws, path, why):
    assert ws.auto_enter(path) == why


def test_laptop_fixture(laptop):
    assert laptop.lane_class("/mnt/smbhome/x") == NETWORK and laptop.lane_class("/mnt/dav") == NETWORK
    assert laptop.lane_class("/mnt/wsl/c") == NETWORK and laptop.lane_class("/home/labuser") == LOCAL
    assert laptop.auto_enter("/net/somehost") == "autofs" and laptop.auto_enter("/tmp") == "pseudo"
    assert laptop.auto_enter("/tmp/project") is None and laptop.lane_key("/home/labuser/x") == "/home"
    assert laptop.mount_for("/run/media/labuser/Backup Disk/a").fstype == "exfat"
    # an indirect map (/net): its root and the host folders are automount points, what is below them is mounted
    assert laptop.mount_for("/net").indirect and laptop.auto_enter("/net") == "autofs"
    assert laptop.auto_enter("/net/somehost/export") is None and laptop.auto_enter("/net/somehost/export/a") is None


def test_windows_table(win, monkeypatch):
    assert win.lane_key(r"z:\Lab\x") == "Z:" and win.lane_class(r"Z:\Lab") == NETWORK
    assert win.lane_class(r"C:\Users\labuser\Documents") == LOCAL and win.is_system_volume(r"C:\data")
    assert win.auto_enter(r"Y:\anything") == "disconnected" and win.disconnected("Y:") and not win.disconnected("Z:")
    assert win.lane_key(r"\\NAS.example.org\Share\x") == r"\\nas.example.org\share"
    assert win.lane_class(r"\\nas.example.org\share") == NETWORK
    assert win.auto_enter(r"Z:\Lab") is None and win.unc_to_local(r"\\fileserver.example.ac.uk\groups"
                                                                   r"\labshare-GROUP\UserA") == r"Z:\UserA"


def test_onedrive_is_local_but_cloud_synced(win, monkeypatch):
    """PL-6 / D6: Known Folder Move puts Desktop and Documents in OneDrive. Its files are on the local disk, so it
    is listed, probed and searched like a local folder (a lane of its own); only the output warning knows."""
    for p in (r"C:\Users\labuser\OneDrive - Example University\Desktop",
              r"c:\users\labuser\onedrive - example university\Documents\Vectra\ProjectC\Batch1"):
        assert (win.lane_class(p), win.lane_key(p), win.auto_enter(p), win.cloud_synced(p)) == (
            LOCAL, "onedrive", None, True)
    assert not win.cloud_synced(r"C:\Users\labuser\Documents") and not win.cloud_synced(r"Z:\Lab")
    monkeypatch.setenv("OneDriveConsumer", r"D:\OneDrive")
    assert MountTable(win.mounts(), flavor="win32").cloud_synced(r"D:\OneDrive\Scans")
    monkeypatch.setattr(P, "_windows_known_folders", lambda: [
        ("Documents", "documents", r"C:\Users\labuser\OneDrive - Example University\Documents")])
    sec = {(p.section, p.label): p for p in P.list_places(win, [], "input")}
    assert sec[("computer", "Documents")].auto_probe and sec[("computer", "Documents")].lane_class == LOCAL
    assert sec[("cloud", "OneDrive")].auto_probe and sec[("cloud", "OneDrive")].icon == "drive_cloud"


def test_cloud_synced_elsewhere(ws, monkeypatch, tmp_path):
    gvfs = "/run/user/1000/gvfs/"
    assert ws.cloud_synced("/home/labuser/mnt/clouddrive/out") and ws.cloud_synced(gvfs + "google-drive:host=x/a")
    assert not ws.cloud_synced(gvfs + "smb-share:server=fileserver.example.org,share=lab/out")
    assert not ws.cloud_synced("/mnt/labshare/out") and not ws.cloud_synced("/home/labuser/out")
    monkeypatch.setenv("HOME", str(tmp_path))
    mac = MountTable([], flavor="darwin")
    assert mac.cloud_synced(str(tmp_path / "Library" / "CloudStorage" / "OneDrive-Example" / "out"))
    assert mac.cloud_synced(str(tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "out"))
    assert not mac.cloud_synced(str(tmp_path / "Documents"))


def table(*lines: str) -> MountTable:
    """The workstation's root plus mountinfo lines 'mount_point fstype source [super options]'."""
    text = ["32 2 259:2 / / rw - ext4 /dev/nvme1n1p2 rw"]
    for i, line in enumerate(lines, 100):
        mp, fstype, source, *opts = line.split(" ")
        text.append(f"{i} 32 0:{i} / {mp} rw - {fstype} {source} {opts[0] if opts else 'rw'}")
    return MountTable.from_mountinfo_text("\n".join(text))


def test_an_automount_mounted_after_the_table_was_read():
    """CT-11 / FS-2 / PL-15 / D5: the table still says 'autofs' for a share the user's own access has mounted
    since. Only the automount point itself is off limits; inside it, folders are searched and probed, on the
    lane and with the class the mounted share gets."""
    stale = table("/mnt/labshare autofs systemd-1 rw,fd=60,pgrp=1,timeout=0,direct")
    fresh = table("/mnt/labshare autofs systemd-1 rw,fd=60,pgrp=1,timeout=0,direct",
                  "/mnt/labshare cifs //fileserver.example.ac.uk/labshare rw")
    inside = "/mnt/labshare/Project/Slide1_Scan1"
    assert stale.auto_enter("/mnt/labshare") == "autofs" and fresh.auto_enter("/mnt/labshare") is None
    assert stale.auto_enter("/mnt/labshare/Project") is None and stale.auto_enter(inside) is None
    assert (stale.lane_key(inside), stale.lane_class(inside)) == (fresh.lane_key(inside), fresh.lane_class(inside)) \
        == ("/mnt/labshare", NETWORK)
    indirect = table("/misc autofs /etc/auto.misc rw,fd=7,timeout=300,indirect")
    assert [indirect.auto_enter(p) for p in ("/misc", "/misc/share", "/misc/share/Project")] == ["autofs", "autofs",
                                                                                               None]


@pytest.mark.parametrize("path,key", [
    ("/mnt/latecomer/Project/x", "/mnt/latecomer"), ("/mnt/latecomer", "/mnt/latecomer"), ("/mnt", "/"),
    ("/media/labuser/NEWDISK/a", "/media/labuser/NEWDISK"), ("/media/labuser", "/"),
    ("/run/media/labuser/Backup/a/b", "/run/media/labuser/Backup"), ("/Volumes/lab/2026", "/Volumes/lab"),
    ("/mnt/labshare/UserA", "/mnt/labshare"), ("/mnt/labshare/inner/x", "/mnt/labshare/inner"),
    ("/mnt/labshare/inner2/x", "/mnt/labshare"), ("/home/labuser/mnt/x", "/")])
def test_volumes_the_table_does_not_know_get_their_own_lane(path, key):
    """CT-7 / FS-9 / D5: a share mounted after the table was read, in a place where volumes get mounted, never
    shares the lane of the disk below it (its hang would make every local folder 'not answering')."""
    t = table("/mnt/labshare cifs //fileserver.example.ac.uk/labshare", "/mnt/labshare/inner nfs4 nas:/inner")
    assert t.lane_key(path) == key
    if key.startswith(("/mnt/latecomer", "/media/", "/run/", "/Volumes/")):
        assert t.lane_class(path) == LOCAL and t.auto_enter(path) is None


@pytest.mark.parametrize("mount,path,key,cls,why", [
    ("/mnt cifs //fileserver.example.org/labshare", "/mnt/Project/Slide1_Scan1", "/mnt", NETWORK, None),
    ("/mnt nfs4 nas.example.org:/export", "/mnt/Project", "/mnt", NETWORK, None),
    ("/media/labuser fuse.rclone labdrive:", "/media/labuser/Lab/Scans", "/media/labuser", CLOUD, "cloud"),
    ("/mnt autofs /etc/auto.mnt rw,fd=7,timeout=300,indirect", "/mnt/labshare", "/mnt/labshare", NETWORK, "autofs"),
    ("/mnt autofs /etc/auto.mnt rw,fd=7,timeout=300,indirect", "/mnt/labshare/Project", "/mnt/labshare", NETWORK,
     None),
    ("/mnt ext4 /dev/sdb1", "/mnt/latecomer/Project", "/mnt/latecomer", LOCAL, None)])
def test_a_share_mounted_on_a_mount_folder_keeps_its_lane_and_class(mount, path, key, cls, why):
    """D5: an unknown /mnt/<x> or /media/<user>/<x> folder is a lane of its own only on a local disk. A share or
    cloud mount on /mnt or /media/<user> itself keeps its class (deadlines, warnings, search depth, no probing of
    cloud folders); each share of an automount map is a lane of its own, with the map's class."""
    t = table(mount)
    assert (t.lane_key(path), t.lane_class(path), t.auto_enter(path)) == (key, cls, why)
    assert t.cloud_synced(path) == (cls == CLOUD)


def test_gvfs_backends_are_classed_by_their_folder(ws, monkeypatch, tmp_path):
    """FS-10 / PL-17 / D7: shares connected in the file manager (gvfs) are network places of their own."""
    root = "/run/user/1000/gvfs"
    smb = root + "/smb-share:server=fileserver.example.org,share=labshare"
    sftp = root + "/sftp:host=hpc.example.org,user=labuser"
    drive = root + "/google-drive:host=example.org,user=someone"
    for p, cls in ((smb + "/ProjectC/Batch1", NETWORK), (sftp + "/data", NETWORK), (drive + "/x", CLOUD),
                   (root + "/davs:host=webdav.example.org,ssl=true/a", NETWORK), (root + "/mtp:host=Phone/x", CLOUD),
                   (root + "/unknown-backend:x=y", CLOUD)):
        assert ws.lane_class(p) == cls and ws.lane_key(p) == "/".join(p.split("/")[:6]), p
        assert ws.auto_enter(p) == (None if cls == NETWORK else "cloud")
    assert ws.lane_class(root) == CLOUD and ws.lane_key(root) == root and ws.auto_enter(root) == "cloud"
    assert ws.place_name(smb + "/ProjectC") == "labshare on fileserver.example.org"
    assert ws.place_name(sftp) == "labuser@hpc.example.org" and ws.place_name(drive) == "someone@example.org"
    names = [smb.split("/")[-1], sftp.split("/")[-1], drive.split("/")[-1], "dav:host=webdav.example.org,prefix=%2Fdav"]
    seen = []
    monkeypatch.setattr(P, "_list_gvfs", lambda r: seen.append(r) or names)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    network = [p for p in P.list_places(ws, [], "input") if p.section == "network"]
    gv = [(p.label, p.path, p.lane_class, p.auto_probe) for p in network if p.path.startswith(root)]
    assert seen == [root] and sorted(gv) == sorted([("labshare on fileserver.example.org", smb, NETWORK, True),
                                                    ("labuser@hpc.example.org", sftp, NETWORK, True),
                                                    ("webdav.example.org", root + "/" + names[3], NETWORK, True)])


def test_listing_the_gvfs_root_gives_up_when_the_daemon_hangs(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(P.os, "listdir", lambda p: gate.wait(5) and [])
    t0 = time.monotonic()
    try:
        assert P._list_gvfs("/run/user/1000/gvfs", timeout_s=0.2) == []
        assert time.monotonic() - t0 < 1.0
    finally:
        gate.set()


def test_macos_fuse_volumes(monkeypatch, tmp_path):
    """FS-10 / D7: macFUSE reports one fstype for every FUSE filesystem: an rclone remote is cloud, a disk is
    local, sshfs and the like are network."""
    part = collections.namedtuple("sdiskpart", "device mountpoint fstype opts")
    parts = [part("/dev/disk3s1s1", "/", "apfs", "ro"), part("teamdocs:", "/Volumes/teamdocs", "macfuse", "rw"),
             part("labremote:Lab/Scans", "/Users/labuser/scans", "osxfuse", "rw"),
             part("rclone-mount", "/Users/labuser/r2", "macfuse", "rw"),
             part("labuser@hpc.example.org:/data", "/Volumes/hpc", "macfuse", "rw"),
             part("hpc.example.org:/export", "/Users/labuser/hpc2", "osxfusefs", "rw"),
             part("/dev/disk5s1", "/Volumes/NTFS", "macfuse", "rw")]
    monkeypatch.setattr(P, "_FLAVOR", "darwin")
    monkeypatch.setattr(P.psutil, "disk_partitions", lambda all=False: parts)
    monkeypatch.setenv("HOME", str(tmp_path))
    m = MountTable.load()
    assert [m.lane_class(p.mountpoint + "/x") for p in parts[1:]] == [CLOUD, CLOUD, CLOUD, NETWORK, NETWORK, LOCAL]
    assert m.auto_enter("/Volumes/teamdocs/x") == "cloud" and m.cloud_synced("/Volumes/teamdocs/x")
    sec = {(p.section, p.label) for p in P.list_places(m, [], "input")}
    assert {("cloud", "teamdocs (cloud)"), ("network", "hpc"), ("computer", "NTFS disk")} <= sec


@pytest.mark.parametrize("text,local", [
    (r"\\fileserver.example.ac.uk\groups\labshare-GROUP\UserA", "/mnt/labshare/UserA"),
    ("smb://FILESERVER.example.ac.uk/groups/LABSHARE-group/UserA/x y", "/mnt/labshare/UserA/x y"),
    ("smb://labuser@fileserver.example.ac.uk/groups/Imaging%20Share/z", IMG + "/z"),
    ("//192.0.2.2/LabData/a", "/mnt/nas10g/a"), (r"\\192.0.2.2\LabData", "/mnt/nas10g"),
    (r"\\other.example.org\groups\labshare-GROUP", None), (r"\\fileserver.example.ac.uk", None), ("/mnt/x", None)])
def test_unc_to_local(ws, text, local):
    assert ws.unc_to_local(text) == local


def test_macos_loader_from_psutil(monkeypatch, tmp_path):
    part = collections.namedtuple("sdiskpart", "device mountpoint fstype opts")
    parts = [part("/dev/disk3s1s1", "/", "apfs", "ro"), part("/dev/disk3s5", "/System/Volumes/Data", "apfs", "rw"),
             part("//labuser@nas.example.org/lab", "/Volumes/lab", "smbfs", "rw"),
             part("afp_0RnnT0000002-1.2000000c", "/Volumes/archive", "afpfs", "rw"),
             part("/dev/disk4s1", "/Volumes/USB", "msdos", "rw"), part("map auto_home", "/System/Volumes/Data/home",
                                                                      "autofs", "rw")]
    monkeypatch.setattr(P, "_FLAVOR", "darwin")
    monkeypatch.setattr(P.psutil, "disk_partitions", lambda all=False: parts if all else parts[:1])
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Library" / "CloudStorage").mkdir(parents=True)
    m = MountTable.load()
    assert m.flavor == "darwin" and m.lane_class("/Volumes/lab/x") == NETWORK
    assert m.lane_class("/Volumes/archive") == NETWORK and m.lane_class("/Volumes/USB") == LOCAL
    assert m.lane_class("/Users/labuser/Desktop") == LOCAL and m.lane_key("/Volumes/lab/x") == "/Volumes/lab"
    assert m.unc_to_local("smb://nas.example.org/lab/2026") == "/Volumes/lab/2026"
    gd = str(tmp_path / "Library" / "CloudStorage" / "GoogleDrive-user" / "My Drive")
    assert m.lane_class(gd) == CLOUD and m.lane_key(gd).endswith("googledrive-user")
    pl = P.list_places(m, [], "input")
    sec = {(p.section, p.label): p for p in pl}
    assert not sec[("computer", "Desktop")].auto_probe and sec[("computer", "Home")].auto_probe
    assert ("computer", "USB disk") in sec and ("network", "lab") in sec and ("network", "archive") in sec
    assert ("cloud", "Cloud storage") in sec and P.tcc_protected(str(tmp_path / "Documents" / "x"))
    assert not P.tcc_protected(str(tmp_path / "Projects"))


def test_list_places_linux(ws, monkeypatch, tmp_path):
    home = tmp_path / "home" / "labuser"
    for d in ("Desktop", "Documents", "My Music", "Videos"):
        (home / d).mkdir(parents=True)
    cfg = tmp_path / "cfg"
    (cfg / "gtk-3.0").mkdir(parents=True)
    (cfg / "user-dirs.dirs").write_text('# comment\nXDG_DESKTOP_DIR="$HOME/Desktop"\nXDG_DOCUMENTS_DIR="$HOME/Documents"\n'
                                        'XDG_MUSIC_DIR="$HOME/My Music"\nXDG_VIDEOS_DIR="$HOME/"\n'
                                        'XDG_PICTURES_DIR="$HOME/Pictures"\n', encoding="utf-8")
    (cfg / "gtk-3.0" / "bookmarks").write_text("file:///mnt/labshare/UserA Lab data\nfile:///tmp/x%20y\n"
                                               "sftp://hpc.example.org/home/labuser\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    monkeypatch.setattr(P, "_list_gvfs", lambda root: [])         # never this machine's own gvfs folder
    pl = P.list_places(ws, [{"path": "/mnt/labshare/UserA", "slides": 6}, {"path": "/home/Data8TB", "slides": 1},
                            {"path": "/home/labuser/mnt/clouddrive/x"}], "input")
    got = [(p.section, p.label, p.icon, p.auto_probe) for p in pl]
    assert got[:3] == [("recent", "UserA · 6 slides", "recent", True), ("recent", "Data8TB · 1 slide", "recent", True),
                       ("recent", "x", "recent", False)]
    computer = [g[1] for g in got if g[0] == "computer"]
    assert computer == ["Home", "Desktop", "Documents", "My Music", "Pictures", "Computer", "Data8TB disk", "USB STICK disk"]
    assert [(g[1], g[3]) for g in got if g[0] == "network"] == [
        ("archive", True), ("hpc (ssh)", True), ("Imaging Share", True), ("labshare", True), ("laptop (ssh)", True),
        ("nas10g", True), ("scratch (not mounted)", False)]
    assert [(g[1], g[3]) for g in got if g[0] == "cloud"] == [("clouddrive (cloud)", False), ("gvfs (cloud)", False)]
    marks = [p for p in pl if p.section == "bookmarks"]
    assert [(p.label, p.path, p.lane_class) for p in marks] == [("Lab data", "/mnt/labshare/UserA", NETWORK),
                                                               ("x y", "/tmp/x y", LOCAL)]
    labshare = next(p for p in pl if p.label == "labshare")
    assert labshare.detail == "//fileserver.example.ac.uk/groups/labshare-GROUP · cifs" and labshare.icon == "drive_net"
    assert P.list_places(ws, [{"path": "/out", "slides": 3}], "output")[0].label == "out"


def test_containing_place_and_breadcrumbs():
    places = [Place("computer", "Computer", "/", "drive", LOCAL, True),
              Place("computer", "Home", "/home/labuser", "home", LOCAL, True),
              Place("network", "Imaging Share", IMG, "drive_net", NETWORK, True),
              Place("recent", "Group · 3 slides", IMG + "/Group", "recent", NETWORK, True)]
    path = IMG + "/Group/Project"
    pl = P.containing_place(path, places)
    assert pl.label == "Imaging Share"                            # recent entries are not crumb roots
    crumbs, hidden = P.breadcrumb_parts(path, pl)
    assert crumbs == [("Imaging Share", IMG), ("Group", IMG + "/Group"), ("Project", path)]
    assert hidden == [("/", "/"), ("home", "/home"), ("labuser", "/home/labuser"), ("mnt", "/home/labuser/mnt")]
    assert P.breadcrumb_parts("/a/b", None) == ([("/", "/"), ("a", "/a"), ("b", "/a/b")], [])
    assert P.containing_place("/srv/x", places).label == "Computer"
    win_crumbs, _ = P.breadcrumb_parts(r"C:\Users\x", None)
    assert win_crumbs == [("C:\\", "C:\\"), ("Users", r"C:\Users"), ("x", r"C:\Users\x")]


def test_normalize_user_path_posix(ws, monkeypatch):
    monkeypatch.setenv("HOME", "/home/labuser")
    monkeypatch.setenv("PROJ", "projects")
    cases = {
        '  "/mnt/labshare/UserA/"  ': "/mnt/labshare/UserA",
        "\n\n'/mnt/x y'\nsecond line": "/mnt/x y",
        "file:///home/labuser/My%20Files/": "/home/labuser/My Files",
        "~/data": "/home/labuser/data", "$HOME/a": "/home/labuser/a", "${HOME}/b": "/home/labuser/b",
        "%HOME%/c": "/home/labuser/c", "/mnt/$PROJ/%PROJ%": "/mnt/projects/projects",
        "/mnt/labshare/UserA/S_[1,2]_component_data.tif\r\n": "/mnt/labshare/UserA/S_[1,2]_component_data.tif",
        r"\\fileserver.example.ac.uk\groups\labshare-GROUP\UserA": "/mnt/labshare/UserA",
        "file://fileserver.example.ac.uk/groups/labshare-GROUP/UserA": "/mnt/labshare/UserA",
        "smb://unknown.example.org/share/x": "smb://unknown.example.org/share/x",
        "/": "/", "": ""}
    for text, want in cases.items():
        assert P.normalize_user_path(text, ws) == want, text


def test_normalize_user_path_windows(win, monkeypatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\labuser")
    cases = {"C:/data/x/": r"C:\data\x", "C:\\": "C:\\", '"C:\\Users\\labuser\\Desktop\\"': r"C:\Users\labuser\Desktop",
             "smb://labuser@fileserver.example.ac.uk/groups/x": r"\\fileserver.example.ac.uk\groups\x",
             "//nas.example.org/old/": r"\\nas.example.org\old", "file:///C:/Lab%20Data/a": r"C:\Lab Data\a",
             "%USERPROFILE%\\Documents": r"C:\Users\labuser\Documents"}
    for text, want in cases.items():
        assert P.normalize_user_path(text, win) == want, text
    for text, want in cases.items():                   # a second pass (MainWindow commits it again) is harmless
        assert P.normalize_user_path(P.normalize_user_path(text, win), win) == want, text


def test_a_bare_drive_letter_means_the_drive(win):
    """PL-3: 'D:' alone is the drive's root, not the process's current folder on it; 'Z:Data' stays relative
    (the main window refuses relative paths)."""
    for text, want in (("D:", "D:\\"), ("d:", "d:\\"), ("D:\\", "D:\\"), ("  'Z:'  ", "Z:\\"), ("Z:Data", "Z:Data")):
        once = P.normalize_user_path(text, win)
        assert once == want and P.normalize_user_path(once, win) == want, text
    assert win.lane_class("Z:") == win.lane_class("Z:\\") == NETWORK and win.mount_for("z:").mount_point == "Z:\\"


@pytest.mark.parametrize("text,want", [
    (r"\\files.example.org\labshare\Cells_10%FBS\Slide1", r"\\files.example.org\labshare\Cells_10%FBS\Slide1"),
    (r"\\fileserver.example.org\labshare\Run%201", r"\\fileserver.example.org\labshare\Run%201"),
    (r"\\fileserver.example.org\labshare\Run%2541", r"\\fileserver.example.org\labshare\Run%2541"),
    ("//fileserver.example.org/labshare/IHC_0.5%DAB/", r"\\fileserver.example.org\labshare\IHC_0.5%DAB"),
    (r"\\tenant.example.org@SSL\DavWWWRoot\sites\lab\Scans", r"\\tenant.example.org@SSL\DavWWWRoot\sites\lab\Scans"),
    (r"\\fileserver.example.org@8443\share\x", r"\\fileserver.example.org@8443\share\x"),
    ("smb://labuser@fileserver.example.org:445/labshare/Run%201", r"\\fileserver.example.org\labshare\Run 1"),
    ("file://fileserver.example.org/labshare/Cells_10%25FBS", r"\\fileserver.example.org\labshare\Cells_10%FBS")])
def test_unc_paths_are_kept_as_typed(win, text, want):
    """PL-4: only smb:// is a URI (decoded once, user@ and :port dropped); UNC folder names such as '10%FBS' and
    WebDAV hosts such as 'tenant@SSL' are used as they are, also when the path is cleaned a second time."""
    once = P.normalize_user_path(text, win)
    assert once == want and P.normalize_user_path(once, win) == want


def test_unc_paths_map_to_a_mount_as_typed(ws):
    assert P.normalize_user_path(r"\\192.0.2.2\LabData\Cells_10%FBS", ws) == "/mnt/nas10g/Cells_10%FBS"
    assert P.normalize_user_path("smb://labuser@192.0.2.2/LabData/Run%201", ws) == "/mnt/nas10g/Run 1"
    assert P.normalize_user_path("/mnt/nas10g/Run%201", ws) == "/mnt/nas10g/Run%201"
    assert ws.lane_key(r"\\tenant.example.org@SSL\x") == "/" and MountTable([], flavor="win32").lane_key(
        r"\\tenant.example.org@SSL\DavWWWRoot\x") == r"\\tenant.example.org@ssl\davwwwroot"


def test_mapped_drives_are_labelled_with_their_share(monkeypatch):
    """PL-9: 'Z: labshare', not 'Z:' (a UNC root has no basename). Runs _load_windows on a fake windll."""
    remotes = {"Z:": (0, r"\\fileserver.example.org\labshare"), "Y:": (1201, r"\\nas.example.org\archive"),
               "X:": (0, r"\\nas.example.org\data\ProjectC"), "W:": (2250, "")}

    class Kernel32:
        @staticmethod
        def GetLogicalDrives():
            return sum(1 << (ord(c) - 65) for c in "CWXYZ")

        @staticmethod
        def GetDriveTypeW(root):
            return 3 if root.startswith("C") else 4

    class Mpr:
        @staticmethod
        def WNetGetConnectionW(drive, buf, size_ref):
            code, remote = remotes[drive]
            buf.value = remote
            return code

    import ctypes
    monkeypatch.setattr(ctypes, "windll", collections.namedtuple("W", "kernel32 mpr")(Kernel32, Mpr), raising=False)
    t = MountTable._load_windows()
    assert {m.mount_point: m.label for m in t.mounts()} == {"C:\\": "Local Disk (C:)", "W:\\": "W:",
                                                            "X:\\": "X: ProjectC", "Y:\\": "Y: archive",
                                                            "Z:\\": "Z: labshare"}
    assert t.disconnected(r"Y:\x") and not t.disconnected(r"Z:\x")


@pytest.mark.parametrize("path,name", [
    ("/mnt/labshare/UserA/Project", "labshare"), (IMG + "/Group", "Imaging Share"),
    ("/home/labuser/mnt/clouddrive/x", "clouddrive"), ("/home/labuser/Data/Slide1_Scan1", "Slide1_Scan1"),
    ("/home/labuser/Data/", "Data"), ("/", "/")])
def test_place_name(ws, path, name):
    assert ws.place_name(path) == name


def test_place_name_windows(win):
    assert win.place_name(r"Z:\Lab\x") == "Z:" and win.place_name(r"\\nas.example.org\Share\x") == "Share"
    assert win.place_name(r"C:\Users\labuser\Data") == "Data" and win.place_name("C:\\") == "C:\\"


def test_place_names_are_spelled_as_in_the_path(monkeypatch, tmp_path):
    """The "<place> didn't answer." text names the folder the way the user sees it, although the lane key of a
    macOS cloud folder is casefolded (one lane however the path is spelled)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    t = MountTable([Mount("/", "apfs", "/dev/disk3s1s1", LOCAL, "Computer"),
                    Mount("/Volumes/Lab", "smbfs", "//labuser@nas.example.org/Lab", NETWORK, "Lab")], flavor="darwin")
    drive = str(tmp_path / "Library" / "CloudStorage" / "OneDrive-ExampleUniversity" / "Scans")
    assert t.lane_key(drive).endswith("onedrive-exampleuniversity") and t.lane_key(drive.lower()) == t.lane_key(drive)
    assert t.place_name(drive) == t.place_name(drive + "/x/") == "OneDrive-ExampleUniversity"
    icloud = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Lab"
    assert t.place_name(str(icloud)) == "iCloud Drive" and t.place_name("/Volumes/Lab/2026/Slide1_Scan1") == "Lab"


def test_mount_tables_compare_by_content(ws, laptop):
    again = MountTable.from_mountinfo_text((FIXTURES / "mountinfo_workstation.txt").read_text(encoding="utf-8"))
    assert ws == again and hash(ws) == hash(again) and ws != laptop and ws != MountTable(ws.mounts(), flavor="darwin")


def test_xdg_places_never_stat_configured_destinations(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_config_lines", lambda h, *p: [
        'XDG_DOCUMENTS_DIR="/shares/unresponsive/Documents"',
        'XDG_DOWNLOAD_DIR="$HOME/Downloads"'] if p == ("user-dirs.dirs",) else [])
    def forbidden(path):
        raise AssertionError("A configured destination must not be statted")
    monkeypatch.setattr(P.os.path, "isdir", forbidden)
    places = P.list_places(MountTable(flavor="linux"), [], "input")
    assert {p.label for p in places} >= {"Home", "Computer", "Documents", "Downloads"}


def test_gvfs_snapshot_maps_smb_and_unc_and_tracks_backend_changes(monkeypatch, tmp_path):
    root = str(tmp_path / "gvfs")
    names = ["smb-share:server=nas.example.org,share=Cells%2520A"]
    monkeypatch.setattr(P, "_list_gvfs", lambda *a: list(names))
    base = MountTable([Mount(root, "fuse.gvfsd-fuse", "gvfsd-fuse", CLOUD, "gvfs")])
    connected = base.with_gvfs()
    local = root + "/" + names[0] + "/Folder%20B"
    assert connected.unc_to_local("smb://nas.example.org/Cells%2520A/Folder%2520B") == local
    assert connected.unc_to_local(r"\\nas.example.org\Cells%20A\Folder%20B") == local
    assert P.normalize_user_path("smb://nas.example.org/Cells%2520A/Folder%2520B", connected) == local
    names.clear()
    disconnected = base.with_gvfs()
    assert disconnected != connected
    assert disconnected.unc_to_local(r"\\nas.example.org\Cells%20A") is None
    assert connected == MountTable(connected.mounts(), gvfs=connected._gvfs)


def test_extended_windows_paths_keep_io_spelling_and_volume_identity(win):
    pairs = [(r"C:\Data", r"\\?\C:\Data", LOCAL),
             (r"Z:\ProjectC", r"\\?\Z:\ProjectC", NETWORK),
             (r"\\nas-a.example.org\labshare\ProjectC", r"\\?\UNC\nas-a.example.org\labshare\ProjectC", NETWORK)]
    for ordinary, extended, cls in pairs:
        assert win.lane_key(ordinary) == win.lane_key(extended)
        assert win.lane_class(extended) == cls
        assert win.mount_for(ordinary) == win.mount_for(extended)
        assert P.normalize_user_path(extended, win) == extended
    assert win.is_system_volume(r"\\?\C:\Data")
    assert win.lane_key(r"\\?\UNC\nas-a.example.org\labshare") != win.lane_key(r"\\?\UNC\nas-b.example.org\labshare")
    assert win.lane_key(r"\\?\UNC\nas-a.example.org\labshare") != win.lane_key(r"\\?\UNC\nas-a.example.org\other")
