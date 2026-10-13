# Stitcher for Vectra Polaris / PhenoImager

A standalone GUI tool that sequentially stitches multi-channel TIFF tiles exported from Akoya Vectra Polaris / PhenoImager scanners into pyramidal OME-TIFFs readable by QuPath and Visiopharm.

Available for **Windows, Linux, and macOS** (Apple Silicon; Intel Macs can run it from source).

> Logic adapted from Pete Bankhead (QuPath founder). Created by Jacky @ Ramachandran Lab.

## Features

- Stitches multiple slides in sequence — run it and walk away
- Automatic RAM vs disk mode based on canvas size and free memory
- Extracts PerkinElmer channel names and physical resolution from TIFF metadata
- Checks channel names before stitching and lets you fix them when they look wrong (e.g. all "DAPI")
- Writes pyramidal OME-TIFFs for Visiopharm/QuPath
- Checks existing stitched files before offering to skip or replace them

## What's new in v2

Version 2 keeps the stitching engine and replaces the interface:

- **One window, three steps:** *1 Tiles to stitch*, *2 Save stitched images to*, *3 Check the list, then stitch*.
- **A folder browser made for lab shares** (below): it previews which folders hold tiles and keeps other places
  usable when a share is slow or not answering.
- **You see the plan before anything runs.** The app searches the folder you chose (normally 3 levels deep, more on
  request) and lists what it found under *New slides*, *Already stitched* or *Not included*, each with a reason.
  Review the ticks and rename output files inline (F2).
- **Several slides in one folder** (an inForm batch export) become one stitched file per slide.
- **Channel names are settled before stitching** (*Check channel names (DAPI ×5)*), with *Copy names from* another slide
  and *Apply to every slide with the same problem*. Nothing asks you a question in the middle of the night.
- **Safer files.** A slide is written into a hidden `.stitcher-in-progress` folder and moved into place only when it
  is complete. Existing outputs are checked for completeness; interrupted slides restart from their tiles.
- **Progress you can read:** per-slide phases, time left, a stall notice, *Stop* (finish this slide or stop now),
  a completion banner and a summary table (CSV) next to the stitched files.
- **A log drawer** (Ctrl+J) with warnings and errors counted; every session also writes a full log file (the last 30
  are kept).
- **Interface size** (Settings) for HiDPI screens, and keyboard shortcuts for browsing and reviewing slides.

The stitching engine is still v1's: for a folder that holds one slide's tiles, the tests check that v2 writes the same
bytes as v1 when both read the tiles on one thread. The app, like v1, reads and pastes tiles on many threads at once,
so where neighbouring tiles overlap, each pixel of that thin strip comes from whichever tile happened to be pasted
there last. Two runs, of v1 or of v2, can therefore differ in those strips, and so in their checksums. By design,
slides that share a folder (an inForm batch export) and slides whose folder names collide are named per slide, so
their file names, and the image name stored inside, can differ from v1's.

The old interface is still available for one release: `python stitcher_RAM.py --classic` (or the binary with
`--classic`).

## Download (pre-built binaries)

The fastest way to use this tool is to grab a pre-built binary from the **[Releases](https://github.com/jackytamkc/Vectra-Polaris-Phenoimager-standalone-stitching-app/releases)** page.

- **Windows**: `YourFavouriteStitcher.exe` — double-click to run
- **Linux**: `YourFavouriteStitcher-linux` — make executable first: `chmod +x YourFavouriteStitcher-linux`
- **macOS**: `YourFavouriteStitcher-macos.zip` — unzip and move the app to `/Applications`. The app is not notarized, so Gatekeeper blocks its first launch: open it once, then go to **System Settings → Privacy & Security** and click **Open Anyway** (on macOS 14 and earlier, right-click → Open also works). This build runs on Apple Silicon Macs only; on an Intel Mac, run from source or build the app on that Mac (below).

## Run from source

```bash
git clone https://github.com/jackytamkc/Vectra-Polaris-Phenoimager-standalone-stitching-app.git
cd Vectra-Polaris-Phenoimager-standalone-stitching-app
pip install -r requirements.txt
python stitcher_RAM.py
```

**Requirements**: Python ≥ 3.9, plus the packages listed in `requirements.txt` (numpy, tifffile, imagecodecs, psutil).

Command-line options (all optional):

| Option | Effect |
|---|---|
| `--input FOLDER` | Search this tile folder at once; relative paths use the folder the app starts in |
| `--output FOLDER` | Save to this folder; relative paths use the folder the app starts in |
| `--scale 1.5` | Interface size for this session (1.0 = 100 %); Settings → *Interface size* sets it for good |
| `--classic` | The v1 interface |
| `--diag` | Print versions, font system, settings and log paths, then exit |
| `--smoke-test` | Build the window, run for 1.5 s, print `SMOKE OK <version>` (used by the build check) |

The Windows binary has no console window: `--diag` and `--smoke-test` print only when their output is redirected,
e.g. `YourFavouriteStitcher.exe --diag > diag.txt` (the file can take a few seconds to fill). *About* shows the same
diagnostics.

Default settings and session log folders (*Log folder* in the app opens the latter):

| System | Settings | Logs |
|---|---|---|
| Windows | `%LOCALAPPDATA%\YourFavouriteStitcher` | `%LOCALAPPDATA%\YourFavouriteStitcher\logs` |
| macOS | `~/Library/Application Support/YourFavouriteStitcher` | `~/Library/Logs/YourFavouriteStitcher` |
| Linux | `~/.config/yourfavouritestitcher` | `~/.local/state/yourfavouritestitcher/logs` |

Linux also honours `XDG_CONFIG_HOME` and `XDG_STATE_HOME`. `YFS_CONFIG_DIR`, when set, puts settings in that folder
and logs in its `logs` subfolder. *About* shows the paths actually in use.

## Choosing folders: the folder browser

In **1 Tiles to stitch**, choose the folder holding your `…_component_data.tif` tiles, or a parent folder holding
several slides. In **2 Save stitched images to**, choose where the OME-TIFFs should go. You can paste a full path
into either field and press Enter, or use *Browse…* (Ctrl+O / Ctrl+Shift+O).

The folder browser works the same way for both steps:

- **Places** on the left: *Recent* folders with their slide counts, this computer's folders and disks, network
  shares and cloud folders. A dot and text mark slow, not-answering or disconnected shares. Click a place, or use
  Up/Down then Enter in the sidebar.
- **Open or choose:** click a folder once to select it; double-click or Enter opens it. The button names the folder
  it will choose (*Use 'ProjectC'*, *Save here: 'stitched'*); Ctrl+Enter does the same. With no folder selected, it
  chooses the folder you are viewing. Click empty space in the list, or clear the filter and selection with Esc,
  to use that folder. Browse normally opens beside the folder already in the main-window field, with it selected;
  drive and share roots open directly.
- **What's inside** gives a quick preview: *Slide tiles*, *12 slide folders inside*, *2 slides (inForm batch
  export)*, *Raw scan — unmix in inForm first*, *3 stitched images*, or *No tiles*. A `+` after a count means the
  preview is partial. These badges describe files found; the plan checks existing images for completeness.
  Tile files collapse into one line; *Show all files* reveals other files.
- The **preview** draws tile positions from file names when there is room. The **bottom strip** explains the
  current choice or next action. When saving, it reports free space and existing images, and warns about mixing
  outputs with tiles. *Create 'stitched' here* makes or selects a separate subfolder; *New folder* lets you name one.
  *Save here* checks that the folder can be written before accepting it.
- The **path bar** (Ctrl+L or F6) accepts typed or pasted paths. Tab completes a name, Down shows suggestions,
  Enter opens the path, and Ctrl+Enter chooses it. A relative path here starts from the folder being viewed.
  Share paths such as `\\fileserver.example.org\labshare\ProjectC` or `smb://fileserver.example.org/labshare/ProjectC`
  are accepted. On Linux and macOS, they are mapped to an existing local mount; connect the share in your file
  manager first if no mount is found.
- **Folders not checked yet** stay visible with *Only folders with tiles*: *Cloud folder — open to look inside*
  and *Network drive — open to connect* are not empty-folder verdicts. Cloud subfolders are not checked
  automatically; open one to inspect it. Choosing a folder marked as cloud starts a search one level deep.
- **Slow or disconnected shares:** you can keep browsing other places. A stalled opening offers *Keep waiting*,
  *Try again* and *Go back*. A row marked *Didn't answer — F5 to try again* can be refreshed with F5. If a drive is
  disconnected, reconnect it in your file manager, then try again.
- **Keyboard navigation:** Tab / Shift+Tab moves between the sidebar, list, filter, footer options and actions
  (Tab completes names while editing a path). Up/Down moves in the list after opening a folder.
  *Use the system folder picker…* is also available when a system picker is installed.

## Check the list, then stitch

Nothing is stitched when you choose a folder. Review **3 Check the list, then stitch**: *New slides* contains work
to do, *Already stitched* contains existing outputs and their checks, and *Not included* explains exclusions and
folders not searched. Small or sparse tile sets start unticked; read the reason before including them. If step 1
says folders could not be searched, *Show* reveals them. Use *Search deeper* for folders beyond the search limit,
or F5 / *Rescan* to look again.

Read **What will happen**, especially when files already exist:

| What will happen | Meaning |
|---|---|
| *Already stitched — checking it's complete…* | Wait for the check; the file has not yet been verified. |
| *Skipped: already stitched…* | Unticked by default. The output passed the completeness check; tick only to replace it. |
| *Couldn't check the existing file…* | Unticked by default. Check the connection or access and Rescan; tick only if you intend to replace it. |
| *The existing file looks incomplete — will redo it* | Ticked by default. A replacement will be confirmed before stitching. |
| *Unfinished last time — will start again* | The interrupted slide will be restarted from its tiles. |
| *Finished last time… will move the new image into place* | The kept image will be checked and moved into place without restitching. |
| *Another stitcher is writing it now…* | No tick box. Rescan when the other stitcher has finished. |

Click tick boxes or use Space on selected slides; Delete unticks. F2 renames the output file. Enter opens slide
details. Right-click, Shift+F10 or the Menu key opens actions for that row, such as *Try again*, *Finalize* or
*Search deeper*. A damaged first tile needs a good copy followed by *Rescan*; retryable read errors offer *Try again*.

Press **Stitch Stitch!** (Ctrl+Enter) to run the ticked slides. The app checks the save folder again, then shows
*Ready to stitch?* if there are warnings, including any planned replacements. Resolve channel-name problems there,
or explicitly accept generic names. Blocking errors must be resolved first. During a run, the folders are locked;
you can untick a waiting slide to skip it. The list shows each slide's phase and the run bar estimates time left.

## Stopping and finishing a batch

**Stop…** offers *Finish this slide first, then stop* or *Stop now*. Finished slides are kept either way. Stop now
discards the current slide's unfinished work; a write step may need to finish before it can stop. That slide starts
again from its tiles next time. Once stopping begins, waiting slides are not stitched.

Closing the window during a run asks whether to finish the current slide then quit, or quit now. While a quit is
pending, the title and run bar say so. Close again and choose *Don't quit* to keep the app open; stitching still
stops. *Quit now* may leave unfinished files for the next run and may exit before a summary CSV is written.

The completion banner separates stitched, failed, skipped, stopped and waiting-to-finalize slides. **Done with
gaps** means tiles could not be read; review the row and log before using that image. Use *Open output folder* to
see the results, *Continue* after stopping, *Retry failed* after failures, or *Retry remaining* after a save-folder
failure. *Start a new batch* searches the folders again.

**Finished images waiting to be moved into place:** if an old output is still open in QuPath or another program,
the new image stays in `.stitcher-in-progress`. Close the old image, then choose **Finalize** in the banner or the
slide's menu. This checks and moves the kept image without restitching. A failure shows the reason and *Try again*
or *Rescan*. Quitting with such images from the run asks first; they remain available next time you stitch the same
slides into that save folder.

**Log and summary:** click *Log* or press Ctrl+J for the session messages, with *Warnings and errors* as a filter.
*Log folder* opens the full `session-YYYYmmdd-HHMMSS.log` files in the location listed above. By default, a batch
with results also saves `YourFavouriteStitcher_summary_YYYYmmdd-HHMMSS.csv` in the save folder, including after a
normal stop. It records each slide's status, message, failed-tile count and output path. Look for *Summary saved*
in the banner; a failure to save it is reported in the log. Settings can turn summary saving off. The CSV records
the batch as it ended; a later Finalize updates the window and log, but does not rewrite that CSV.

## Keyboard shortcuts

Main window (on macOS, Command replaces Control):

| Keys | Action |
|---|---|
| Ctrl+O / Ctrl+Shift+O | Browse for tiles / for the output folder |
| F5 | Search the folder again |
| Ctrl+Enter | Stitch Stitch! |
| Space / Delete | Toggle / untick the selected slides |
| F2 | Rename the output file of the slide |
| Shift+F10 or the Menu key | The slide's menu (Try again, Finalize, Search deeper…) |
| Enter or double-click | Slide details (double-clicking Output file renames instead) |
| Ctrl+A (in the plan) | Select all rows without changing their ticks |
| Ctrl+J | Show or hide the log |
| Ctrl+, | Settings |
| F1 | How it works |

On macOS, the Delete key (Backspace) also unticks slides.

Folder browser (on macOS, Command replaces Control):

| Keys | Action |
|---|---|
| Enter / double-click | Open the folder |
| Ctrl+Enter | Use (or Save here) |
| Tab / Shift+Tab | Move between controls (Tab completes names in the path field) |
| Up / Down | Move in the list or Places; Enter opens the focused folder |
| Backspace, Alt+Up | Parent folder |
| Alt+Left / Alt+Right | Back / forward |
| Ctrl+L, F6, or typing `/` `~` `\` | Type a path (Tab completes, Down shows suggestions) |
| F5 / Ctrl+R | Refresh, including folders that did not answer |
| Ctrl+F / Ctrl+H | Filter / show hidden items |
| Ctrl+Shift+N | New folder (when choosing where to save or keep temporary files) |
| Letters | Jump to a folder by its name |
| Esc | Leave the path field, clear the filter or selection, then cancel |

## Build your own binary

```bash
pip install -r requirements-build.txt    # Python 3.11 or later, as in the release workflow
# Linux
cd linux && pyinstaller YourFavouriteStitcher.spec
# Windows
cd Windows && pyinstaller YourFavouriteStitcher.spec
# macOS
cd Mac && pyinstaller YourFavouriteStitcher.spec
```

The binary will appear in the per-OS `dist/` subfolder (`.exe` on Windows, bare executable on Linux, `.app` bundle on macOS). A Mac build runs on the kind of Mac that built it (Apple Silicon or Intel).

`requirements-build.txt` pins PyInstaller 6 and the tifffile and imagecodecs versions that
`tests/test_engine_identity.py` passed with, so a binary writes its files with the same libraries as the tested code.

`tools/build_check.sh` builds the Linux binary into a separate folder and runs its `--smoke-test`, `--classic` and
`--diag` checks under Xvfb.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest            # GUI tests start a private Xvfb on Linux (skipped without Xvfb or a display)
python tools/shots.py --scale 1.0 --out tests/_shots/py-1.0    # screenshots of every screen (synthetic data)
python tests/fixtures/make_synth.py    # the synthetic tile tree the tests use, in a new temporary folder
```

`tests/test_engine_identity.py` compares the engine's output with v1's byte for byte, on one thread where tiles
overlap. Optional read-only checks on real data run when `YFS_REAL_SUBSET` (a folder of one slide's tiles) or
`YFS_REAL_ROOTS` is set.

### Automated builds

The repository includes a GitHub Actions workflow (`.github/workflows/build-release.yml`) that builds binaries for all three platforms whenever you push a version tag (e.g. `git tag v1.2.0 && git push --tags`). The workflow uploads all three binaries as assets to the corresponding GitHub Release automatically.

## Performance notes

- Speed is usually **bound by network bandwidth** when reading tiles from a network drive — local SSD is dramatically faster.
- Make sure you have enough **free disk space** — the tool may create temporary files when the stitched canvas exceeds RAM.
- For canvases larger than available RAM, the tool automatically falls back to a memory-mapped disk mode (slower and needs enough temporary disk space).
