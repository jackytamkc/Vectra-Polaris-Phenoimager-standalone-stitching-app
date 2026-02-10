"""
QuPath Component Stitcher
=========================
A high-performance tool to stitch tiled TIFFs (PerkinElmer/Vectra/General)
into Pyramidal OME-TIFFs compatible with QuPath.

Features:
- Automatic Hardware Profiling (CPU & Disk Benchmarking)
- Multi-Process & Multi-Threaded parallelization
- Preserves Channel Names (DAPI, Opal, etc.)
- GUI for folder selection

Author: [Your Name]
License: MIT
"""

import os
import time
import shutil
import logging
import multiprocessing
import tempfile
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 🧠 HARDWARE PROFILER
# ==========================================
class HardwareOptimizer:
    @staticmethod
    def benchmark_disk_speed(target_dir):
        """
        Writes a temporary 512MB file to check write speeds.
        Returns: 'NVMe', 'SSD', or 'HDD'
        """
        print("⚡ Benchmarking Disk Speed (this takes 1-2s)...")
        test_file = os.path.join(target_dir, 'speed_test.tmp')
        data = os.urandom(1024 * 1024 * 100)  # 100 MB chunk

        start = time.time()
        try:
            # Write 500MB total
            with open(test_file, 'wb') as f:
                for _ in range(5):
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

            duration = time.time() - start
            speed_mb_s = 500 / duration

            os.remove(test_file)

            if speed_mb_s > 400:
                return 'NVMe', speed_mb_s
            elif speed_mb_s > 100:
                return 'SSD', speed_mb_s
            else:
                return 'HDD', speed_mb_s

        except Exception as e:
            logger.warning(f"Could not benchmark disk: {e}. Assuming HDD.")
            return 'HDD', 50.0

    @staticmethod
    def get_optimal_config(target_dir):
        cpu_count = os.cpu_count() or 4
        disk_type, speed = HardwareOptimizer.benchmark_disk_speed(target_dir)

        print(f"   ► Detected: {cpu_count} Cores | {disk_type} ({speed:.1f} MB/s)")

        if disk_type == 'NVMe':
            # Fast disk: We can saturate CPUs
            parallel_folders = max(1, cpu_count // 4)
            threads_per_job = 4
        elif disk_type == 'SSD':
            # Decent disk: Moderate parallelism
            parallel_folders = max(1, cpu_count // 8)
            threads_per_job = 4
        else:
            # HDD or Network: Serial processing is safer to avoid thrashing
            parallel_folders = 1
            threads_per_job = max(4, cpu_count // 2)

        # Safety caps
        parallel_folders = min(parallel_folders, 8)

        return parallel_folders, threads_per_job


# ==========================================
# 🧵 STITCHING LOGIC
# ==========================================
class StitcherEngine:
    def __init__(self, input_dir, output_file, num_threads):
        self.input_dir = Path(input_dir)
        self.output_path = output_file
        self.num_threads = num_threads

        self.tiles = []
        self.canvas_shape = None
        self.dtype = None
        self.channel_names = []
        self.min_x, self.min_y = 0, 0
        self.tile_w, self.tile_h = 0, 0
        self.n_channels = 0

    def get_rational(self, tag):
        val = tag.value
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0] / val[1]
        return float(val)

    def extract_channel_names(self, tif):
        names = []
        try:
            for page in tif.pages:
                if 270 in page.tags:
                    xml_str = page.tags[270].value
                    if "PerkinElmer-QPI-ImageDescription" in xml_str:
                        try:
                            root = ET.fromstring(xml_str)
                            if root.find("ImageType").text == "Thumbnail": continue
                            name = root.find("Name")
                            if name is not None: names.append(name.text)
                        except:
                            continue
        except:
            pass
        return names

    def scan(self):
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.tif', '.tiff'))])
        if not files: return False

        TAG_X_RES, TAG_Y_RES = 282, 283
        TAG_X_POS, TAG_Y_POS = 286, 287

        for i, f in enumerate(files):
            path = self.input_dir / f

            # Analyze first tile for dimensions/channels
            if i == 0:
                img = tifffile.imread(path)
                self.dtype = img.dtype
                if img.ndim == 2:
                    self.n_channels, self.tile_h, self.tile_w = 1, *img.shape
                elif img.ndim == 3:
                    s = img.shape
                    if s[0] < s[1]:
                        self.n_channels, self.tile_h, self.tile_w = s
                    else:
                        self.tile_h, self.tile_w, self.n_channels = s[0], s[1], s[2]

                with tifffile.TiffFile(path) as tif:
                    names = self.extract_channel_names(tif)
                    self.channel_names = names if len(names) == self.n_channels else [f"Channel {x}" for x in
                                                                                      range(self.n_channels)]

            # Parse coordinates
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                if not (TAG_X_POS in tags and TAG_X_RES in tags): continue

                self.tiles.append({
                    'path': path,
                    'abs_x': int(round(self.get_rational(tags[TAG_X_RES]) * self.get_rational(tags[TAG_X_POS]))),
                    'abs_y': int(round(self.get_rational(tags[TAG_Y_RES]) * self.get_rational(tags[TAG_Y_POS])))
                })

        if not self.tiles: return False

        # Calculate Canvas
        xs = [t['abs_x'] for t in self.tiles]
        ys = [t['abs_y'] for t in self.tiles]
        self.min_x, max_x = min(xs), max(xs)
        self.min_y, max_y = min(ys), max(ys)
        total_w = (max_x + self.tile_w) - self.min_x
        total_h = (max_y + self.tile_h) - self.min_y
        self.canvas_shape = (self.n_channels, total_h, total_w)
        return True

    def stitch(self):
        # Temp file for raw data
        temp_raw = self.output_path + ".raw.tmp"
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        mem_img = tifffile.memmap(temp_raw, shape=self.canvas_shape, dtype=self.dtype, bigtiff=True)

        # Worker for threading
        def _copy_tile(tile):
            try:
                img = tifffile.imread(tile['path'])
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
                elif img.ndim == 3 and img.shape[2] == self.n_channels:
                    img = np.moveaxis(img, -1, 0)

                y = tile['abs_y'] - self.min_y
                x = tile['abs_x'] - self.min_x
                _, h, w = img.shape

                h_fit = min(h, self.canvas_shape[1] - y)
                w_fit = min(w, self.canvas_shape[2] - x)

                if h_fit > 0 and w_fit > 0:
                    mem_img[:, y:y + h_fit, x:x + w_fit] = img[:, :h_fit, :w_fit]
                return True
            except:
                return False

        # Run Stitching
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = [pool.submit(_copy_tile, t) for t in self.tiles]
            for _ in as_completed(futures): pass

        mem_img.flush()

        # Write Pyramid
        metadata = {'axes': 'CYX', 'Channel': {'Name': self.channel_names}}
        with tifffile.TiffWriter(self.output_path, bigtiff=True) as tif:
            opts = dict(tile=(256, 256), compression='zlib', metadata=metadata)
            tif.write(mem_img, subifds=3, **opts)
            prev = mem_img
            for _ in range(3):
                curr = prev[:, ::2, ::2]
                tif.write(curr, **opts)
                prev = curr

        del mem_img
        try:
            os.remove(temp_raw)
        except:
            pass


# Wrapper for Multiprocessing
def run_job(args):
    folder, out_dir, threads = args
    name = os.path.basename(folder)
    out_file = os.path.join(out_dir, f"{name}.ome.tif")

    if os.path.exists(out_file):
        print(f"⏩ [{name}] Skipped (Exists)")
        return

    try:
        print(f"⏳ [{name}] Processing...")
        stitcher = StitcherEngine(folder, out_file, threads)
        if stitcher.scan():
            stitcher.stitch()
            print(f"✅ [{name}] Done")
        else:
            print(f"⚠️ [{name}] No valid tiles found")
    except Exception as e:
        print(f"❌ [{name}] Failed: {e}")


# ==========================================
# 🖥️ USER INTERFACE
# ==========================================
def gui_mode():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Your Favourite Stitcher", "Select the ROOT INPUT FOLDER containing your slide subfolders.")
    input_dir = filedialog.askdirectory(title="Select Input Root Folder")
    if not input_dir: return

    messagebox.showinfo("Your Favourite Stitcher", "Select the OUTPUT FOLDER where stitched images will be saved.")
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir: return

    run_batch(input_dir, output_dir)


def run_batch(input_root, output_root):
    # 1. Gather Folders
    folders = [f.path for f in os.scandir(input_root) if f.is_dir()]
    valid_folders = []
    for f in folders:
        if any(file.endswith('.tif') for file in os.listdir(f)):
            valid_folders.append(f)

    print(f"\n🎯 Found {len(valid_folders)} folders containing .tif images.")
    if not valid_folders: return

    # 2. Auto-Optimize
    parallel_jobs, threads_per_job = HardwareOptimizer.get_optimal_config(output_root)
    print(f"🚀 Optimization: Running {parallel_jobs} folders in parallel ({threads_per_job} threads each).\n")

    # 3. Execute
    tasks = [(f, output_root, threads_per_job) for f in valid_folders]

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=parallel_jobs) as pool:
        pool.map(run_job, tasks)

    print(f"\n🎉 Batch processing complete in {time.time() - start_time:.2f}s")
    print(f"📂 Output location: {output_root}")


if __name__ == '__main__':
    # If users just double-click, open GUI.
    # If they use command line, use args.
    import sys

    if len(sys.argv) > 1:
        # Simple CLI argument handling
        if len(sys.argv) < 3:
            print("Usage: python qupath_stitcher.py <input_dir> <output_dir>")
        else:
            run_batch(sys.argv[1], sys.argv[2])
    else:
        gui_mode()