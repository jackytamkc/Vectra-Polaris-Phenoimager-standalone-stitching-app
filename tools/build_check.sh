#!/usr/bin/env bash
# PyInstaller build check (FINAL_SPEC §7.4; Linux, local only: CI is unchanged).
#
#   PYTHON=/path/to/python tools/build_check.sh [OUT_DIR]
#
# 1. Builds linux/YourFavouriteStitcher.spec exactly as CI does (cd linux && pyinstaller ...), but into
#    OUT_DIR/build and OUT_DIR/dist: the repository's linux/build and linux/dist are never touched.
# 2. On DISPLAY, or a private Xvfb when unset: --smoke-test must exit 0 and print "SMOKE OK <version>"; --classic must still be
#    running after 3 s (it is then killed); --diag must exit 0 and show the output of `zenity --version`, run by
#    the frozen app with its cleaned environment (proof that LD_LIBRARY_PATH is restored for child processes).
# 3. Every module of the yfs package must be inside the binary's PYZ (pyi-archive_viewer).
#
# PYTHON (default python3) must have PyInstaller and the app's requirements. OUT_DIR defaults to a new temporary
# folder. DISPLAY reuses an existing server. Otherwise XVFB_DISPLAY picks a private Xvfb display, or the first
# free one from :150 is used.
# Settings and session logs of the frozen runs go to OUT_DIR/config unless YFS_CONFIG_DIR is set.
# Exit status 0 when every check passes; each check prints PASS, FAIL or SKIP.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/yfs-build-check.XXXXXX")}"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
export YFS_CONFIG_DIR="${YFS_CONFIG_DIR:-$OUT/config}"   # the frozen runs keep their settings and logs in OUT
BIN="$OUT/dist/YourFavouriteStitcher"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$REPO/yfs/__init__.py")"
failed=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; failed=1; }
skip() { echo "SKIP  $*"; }
show() { sed 's/^/      | /' "$1" | tail -n "${2:-15}"; }

# ---- 1. build
echo "== build: PyInstaller $("$PYTHON" -m PyInstaller --version 2>/dev/null || echo '(missing)'), output in $OUT"
rm -f "$BIN"                                   # a failed build must not pass on an older binary
if (cd "$REPO/linux" && "$PYTHON" -m PyInstaller --noconfirm --distpath "$OUT/dist" --workpath "$OUT/build" \
        YourFavouriteStitcher.spec) > "$OUT/pyinstaller.log" 2>&1 && [ -x "$BIN" ]; then
    pass "built $BIN ($(du -h "$BIN" | cut -f1))"
else
    fail "PyInstaller build (log: $OUT/pyinstaller.log)"; show "$OUT/pyinstaller.log"; exit 1
fi

# ---- 2. run the binary on the supplied display, or a private Xvfb
XPID=""
cleanup() { [ -n "$XPID" ] && kill "$XPID" 2>/dev/null; }
trap cleanup EXIT
if [ -n "${DISPLAY:-}" ]; then
    echo "== run: existing display $DISPLAY"
elif ! command -v Xvfb >/dev/null; then
    fail "Xvfb is not installed: the frozen app could not be run"
    exit 1
else
    D="${XVFB_DISPLAY:-}"
    if [ -z "$D" ]; then
        for n in $(seq 150 199); do
            if [ ! -e "/tmp/.X11-unix/X$n" ] && [ ! -e "/tmp/.X$n-lock" ]; then D=":$n"; break; fi
        done
    fi
    [ -n "$D" ] || { fail "no free X display between :150 and :199"; exit 1; }
    Xvfb "$D" -screen 0 1600x1000x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
    XPID=$!
    for _ in $(seq 50); do [ -e "/tmp/.X11-unix/X${D#:}" ] && break; sleep 0.1; done
    kill -0 "$XPID" 2>/dev/null || { fail "Xvfb did not start on $D (log: $OUT/xvfb.log)"; XPID=""; exit 1; }
    export DISPLAY="$D"
    echo "== run: Xvfb $D"
fi

    timeout 120 "$BIN" --smoke-test > "$OUT/smoke.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "SMOKE OK $VERSION" "$OUT/smoke.log"; then
        pass "--smoke-test: exit 0, $(grep -o "SMOKE OK $VERSION" "$OUT/smoke.log")"
    else
        fail "--smoke-test: exit $rc, expected 'SMOKE OK $VERSION'"; show "$OUT/smoke.log"
    fi

    "$BIN" --classic > "$OUT/classic.log" 2>&1 &
    CPID=$!
    sleep 3
    if kill -0 "$CPID" 2>/dev/null; then
        pass "--classic: still running after 3 s"
        kill "$CPID" 2>/dev/null; sleep 0.5; kill -9 "$CPID" 2>/dev/null
    else
        wait "$CPID"; fail "--classic: exited after less than 3 s (status $?)"; show "$OUT/classic.log"
    fi
    wait "$CPID" 2>/dev/null

    timeout 120 "$BIN" --diag > "$OUT/diag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        fail "--diag: exit $rc"; show "$OUT/diag.log"
    elif ! grep -q "$VERSION" "$OUT/diag.log"; then
        fail "--diag: the version $VERSION is not in the output"; show "$OUT/diag.log"
    elif ! command -v zenity >/dev/null; then
        skip "--diag: exit 0; zenity is not installed, so the child-process environment was not checked"
    else
        ZV="$(env -u LD_LIBRARY_PATH zenity --version 2>/dev/null | head -n 1)"
        if [ -n "$ZV" ] && grep -i "zenity" "$OUT/diag.log" | grep -qF "$ZV"; then
            pass "--diag: exit 0, the frozen app ran zenity $ZV"
        else
            fail "--diag: no 'zenity ... $ZV' line (zenity failed inside the frozen app?)"; show "$OUT/diag.log"
        fi
    fi
# ---- 3. the yfs package inside the PYZ
"$PYTHON" -c 'import sys; from PyInstaller.utils.cliutils.archive_viewer import run; sys.argv[0] = "pyi-archive_viewer"; run()' \
    -l -r -b "$BIN" > "$OUT/archive.txt" 2>&1
missing=""
for f in $(cd "$REPO" && find yfs -name '*.py' | sort); do
    mod="${f%.py}"; mod="${mod%/__init__}"; mod="${mod//\//.}"
    grep -qE "(^|[[:space:]])$mod\$" "$OUT/archive.txt" || missing="$missing $mod"
done
if grep -qE '(^|[[:space:]])yfs\.ui\.app$' "$OUT/archive.txt" && [ -z "$missing" ]; then
    pass "PYZ holds every yfs module ($(grep -cE '(^|[[:space:]])yfs(\.|$)' "$OUT/archive.txt") entries)"
else
    fail "PYZ is missing:${missing:- yfs.ui.app} (listing: $OUT/archive.txt)"
fi

echo "== $([ $failed -eq 0 ] && echo "BUILD CHECK OK" || echo "BUILD CHECK FAILED") (logs in $OUT)"
exit $failed
