# Stitcher for Vectra Polaris / PhenoImager

A standalone GUI tool that sequentially stitches multi-channel TIFF tiles exported from Akoya Vectra Polaris / PhenoImager scanners into pyramidal OME-TIFFs readable by QuPath and Visiopharm.

Available for **Windows, Linux, and macOS** (Apple Silicon + Intel).

> Logic adapted from Pete Bankhead (QuPath founder). Created by Jacky @ Ramachandran Lab.

## Features

- Stitches multiple input folders in sequence — run it and walk away
- Automatic RAM vs disk mode based on canvas size and free memory
- Extracts PerkinElmer channel names and physical resolution from TIFF metadata
- Prompts interactively when channel names look wrong (e.g. all "DAPI")
- Writes pyramidal OME-TIFF that Visiopharm/QuPath open instantly
- Skips folders whose output already exists (safe to re-run)

## Download (pre-built binaries)

The fastest way to use this tool is to grab a pre-built binary from the **[Releases](https://github.com/jackytamkc/Vectra-Polaris-Phenoimager-standalone-stitching-app/releases)** page.

- **Windows**: `YourFavouriteStitcher-windows.exe` — double-click to run
- **Linux**: `YourFavouriteStitcher-linux` — make executable first: `chmod +x YourFavouriteStitcher-linux`
- **macOS**: `YourFavouriteStitcher-macos.app.zip` — unzip and move to `/Applications`. On first launch you may need to right-click → Open (Gatekeeper requires approval for unsigned apps). Apple Silicon and Intel Macs are both supported.

## Run from source

```bash
git clone https://github.com/jackytamkc/Vectra-Polaris-Phenoimager-standalone-stitching-app.git
cd Vectra-Polaris-Phenoimager-standalone-stitching-app
pip install -r requirements.txt
python stitcher_RAM.py
```

**Requirements**: Python ≥ 3.9, plus the packages pinned in `requirements.txt` (numpy, tifffile, imagecodecs, psutil).

## Build your own binary

```bash
pip install -r requirements-build.txt
# Linux
cd linux && pyinstaller YourFavouriteStitcher.spec
# Windows
cd Windows && pyinstaller YourFavouriteStitcher.spec
# macOS
cd Mac && pyinstaller YourFavouriteStitcher.spec
```

The binary will appear in the per-OS `dist/` subfolder (`.exe` on Windows, bare executable on Linux, `.app` bundle on macOS).

### Automated builds

The repository includes a GitHub Actions workflow (`.github/workflows/build-release.yml`) that builds binaries for all three platforms whenever you push a version tag (e.g. `git tag v1.2.0 && git push --tags`). The workflow uploads all three binaries as assets to the corresponding GitHub Release automatically.

## Performance notes

- Speed is usually **bound by network bandwidth** when reading tiles from a network drive — local SSD is dramatically faster.
- Make sure you have enough **free disk space** — the tool may create temporary files when the stitched canvas exceeds RAM.
- For canvases larger than available RAM, the tool automatically falls back to a memory-mapped disk mode (slower but unlimited size).
