import os
import time
import shutil
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 1. NETWORK LOCATIONS
ROOT_INPUT_DIR = '/mnt/HCS_Users/Ravi Parhar/OLR1 unmixed images 5.2.26/remaining'  # Your Network Input
FINAL_OUTPUT_DIR = '/mnt/pramacha/Ravi_Parhar'  # Your Network Output

# 2. LOCAL FAST STORAGE (Crucial for Speed)
# The script will copy files here, stitch them, then delete them.
# Ensure this drive has enough space (e.g., 200GB+ free).
LOCAL_TEMP_DIR = '/home/jacky/stitch/'

# 3. PERFORMANCE
MAX_PARALLEL_JOBS = 4  # Process 2 folders at a time
NUM_THREADS_PER_JOB = 10  # Threads per folder


# ==========================================

class QuPathMetadataStitcher:
    # ... [Keep your exact Stitcher Class code from before] ...
    # ... [Paste the entire class QuPathMetadataStitcher here] ...
    def __init__(self, input_dir, output_path, num_threads=4):
        self.input_dir = Path(input_dir)
        self.output_path = output_path
        self.num_threads = num_threads
        self.tiles = []
        self.tile_h = 0
        self.tile_w = 0
        self.n_channels = 0
        self.dtype = None
        self.canvas_shape = None
        self.min_x = 0
        self.min_y = 0
        self.channel_names = []

    def get_rational(self, tag):
        val = tag.value
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0] / val[1]
        return float(val)

    def extract_perkin_elmer_channels(self, tif):
        names = []
        try:
            for i, page in enumerate(tif.pages):
                if 270 in page.tags:
                    xml_str = page.tags[270].value
                    if "PerkinElmer-QPI-ImageDescription" in xml_str:
                        try:
                            root = ET.fromstring(xml_str)
                            img_type = root.find("ImageType")
                            if img_type is not None and img_type.text == "Thumbnail": continue
                            name_node = root.find("Name")
                            if name_node is not None: names.append(name_node.text)
                        except ET.ParseError:
                            continue
        except Exception:
            pass
        return names

    def scan_metadata(self):
        print(f"   Scanning metadata...")
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.tif', '.tiff'))])
        if not files: raise ValueError("No TIFFs found.")

        TAG_X_RES, TAG_Y_RES = 282, 283
        TAG_X_POS, TAG_Y_POS = 286, 287

        for i, f in enumerate(files):
            file_path = self.input_dir / f
            if i == 0:
                sample = tifffile.imread(file_path)
                self.dtype = sample.dtype
                if sample.ndim == 2:
                    self.n_channels, self.tile_h, self.tile_w = 1, *sample.shape
                elif sample.ndim == 3:
                    s = sample.shape
                    if s[0] < s[1]:
                        self.n_channels, self.tile_h, self.tile_w = s
                    else:
                        self.tile_h, self.tile_w, self.n_channels = s[0], s[1], s[2]

                with tifffile.TiffFile(file_path) as tif:
                    found_names = self.extract_perkin_elmer_channels(tif)
                    if found_names and len(found_names) == self.n_channels:
                        self.channel_names = found_names
                    else:
                        self.channel_names = [f"Channel {x}" for x in range(self.n_channels)]

            with tifffile.TiffFile(file_path) as tif:
                page = tif.pages[0]
                tags = page.tags
                if not (TAG_X_POS in tags and TAG_X_RES in tags): continue
                self.tiles.append({
                    'path': file_path,
                    'abs_x': int(round(self.get_rational(tags[TAG_X_RES]) * self.get_rational(tags[TAG_X_POS]))),
                    'abs_y': int(round(self.get_rational(tags[TAG_Y_RES]) * self.get_rational(tags[TAG_Y_POS])))
                })

        xs = [t['abs_x'] for t in self.tiles]
        ys = [t['abs_y'] for t in self.tiles]
        self.min_x, max_x = min(xs), max(xs)
        self.min_y, max_y = min(ys), max(ys)
        total_w = (max_x + self.tile_w) - self.min_x
        total_h = (max_y + self.tile_h) - self.min_y
        self.canvas_shape = (self.n_channels, total_h, total_w)
        print(f"   Canvas: {total_w}x{total_h} | Ch: {self.channel_names}")

    def _process_single_tile(self, tile_info):
        try:
            img = tifffile.imread(tile_info['path'])
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim == 3 and img.shape[2] == self.n_channels:
                img = np.moveaxis(img, -1, 0)
            y = tile_info['abs_y'] - self.min_y
            x = tile_info['abs_x'] - self.min_x
            return (img, y, x)
        except:
            return None

    def create_pyramidal_tif(self):
        temp_raw = Path(self.output_path).with_suffix('.raw.tmp')
        mem_img = tifffile.memmap(temp_raw, shape=self.canvas_shape, dtype=self.dtype, bigtiff=True)

        print(f"   Stitching to local temp file...")
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_tile = {executor.submit(self._process_single_tile, t): t for t in self.tiles}
            completed = 0
            for future in as_completed(future_to_tile):
                res = future.result()
                if res:
                    img, y, x = res
                    _, h, w = img.shape
                    h_fit = min(h, self.canvas_shape[1] - y)
                    w_fit = min(w, self.canvas_shape[2] - x)
                    if h_fit > 0 and w_fit > 0:
                        mem_img[:, y:y + h_fit, x:x + w_fit] = img[:, :h_fit, :w_fit]

        mem_img.flush()

        print(f"   Genering Pyramid...")
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


# ==========================================
# 🚀 NETWORK OPTIMIZED WORKFLOW
# ==========================================

def process_one_folder(network_input_folder, network_output_root, local_root):
    folder_name = os.path.basename(network_input_folder)

    # 1. Check if output exists on network
    final_network_path = os.path.join(network_output_root, f"{folder_name}.ome.tif")
    if os.path.exists(final_network_path):
        print(f"⏩ Skipping {folder_name} (Already done)")
        return

    # 2. Define Local Paths
    local_job_dir = os.path.join(local_root, folder_name)
    local_output_file = os.path.join(local_root, f"{folder_name}.ome.tif")

    try:
        # --- A. COPY TO LOCAL (The Speedup) ---
        print(f"⬇️ [{folder_name}] Copying from Network to SSD...")
        if os.path.exists(local_job_dir): shutil.rmtree(local_job_dir)
        shutil.copytree(network_input_folder, local_job_dir)

        # --- B. STITCH LOCALLY ---
        print(f"🧵 [{folder_name}] Stitching on SSD...")
        stitcher = QuPathMetadataStitcher(local_job_dir, local_output_file, num_threads=NUM_THREADS_PER_JOB)
        stitcher.scan_metadata()
        stitcher.create_pyramidal_tif()

        # --- C. UPLOAD TO NETWORK ---
        print(f"⬆️ [{folder_name}] Uploading result to Network...")
        os.makedirs(network_output_root, exist_ok=True)
        shutil.move(local_output_file, final_network_path)
        print(f"✅ [{folder_name}] Finished!")

    except Exception as e:
        print(f"❌ [{folder_name}] Error: {e}")

    finally:
        # --- D. CLEANUP ---
        # Always clean up local temp files to free up SSD space for the next job
        if os.path.exists(local_job_dir): shutil.rmtree(local_job_dir)
        if os.path.exists(local_output_file): os.remove(local_output_file)


def main():
    start = time.time()
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    # Scan network for folders
    print("🔍 Scanning network for folders...")
    all_folders = [f.path for f in os.scandir(ROOT_INPUT_DIR) if f.is_dir()]
    print(f"Found {len(all_folders)} folders.")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as pool:
        futures = []
        for folder in all_folders:
            futures.append(pool.submit(process_one_folder, folder, FINAL_OUTPUT_DIR, LOCAL_TEMP_DIR))

        for _ in as_completed(futures): pass

    print(f"🎉 All done in {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()