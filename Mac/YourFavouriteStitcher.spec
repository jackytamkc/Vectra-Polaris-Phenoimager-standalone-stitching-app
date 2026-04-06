# -*- mode: python ; coding: utf-8 -*-
# macOS build spec. Produces a .app bundle in dist/ plus the raw binary.
# Build from this directory with: pyinstaller YourFavouriteStitcher.spec
#
# target_arch='universal2' builds a fat binary supporting both Apple Silicon
# (arm64) and Intel (x86_64). Requires a universal2 Python and universal2
# wheels for numpy/tifffile/psutil/imagecodecs. If those aren't available,
# remove target_arch (or set it to 'arm64' or 'x86_64') to build for the
# host architecture only.


a = Analysis(
    ['../stitcher_RAM.py'],
    pathex=['..'],
    binaries=[],
    datas=[],
    hiddenimports=['psutil', 'imagecodecs._shared', 'imagecodecs._imcd'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='YourFavouriteStitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # UPX is unreliable on modern macOS codesign
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,               # 'universal2' if you have universal wheels; None = host arch
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='YourFavouriteStitcher.app',
    icon=None,
    bundle_identifier='uk.ed.ramachandran.favouritestitcher',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.1.0',
        'CFBundleVersion': '1.1.0',
        'LSMinimumSystemVersion': '11.0',
    },
)
