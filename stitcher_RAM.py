import os
import time
import psutil
import logging
import threading
import shutil
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
NUM_THREADS = os.cpu_count() or 16
MAX_SAFE_RAM_PERCENT = 95
TILE_SIZE = 512  # Visiopharm Standard


# ==========================================

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


class StitchingEngine:
    def __init__(self, input_dir, output_path, progress_callback=None):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.progress_callback = progress_callback
        self.temp_path = self.output_path.with_suffix('.tmp.raw')

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

    def extract_perkin_elmer_channels(self, tif):
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

    def _read_tile_meta_worker(self, path):
        try:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                if not (286 in tags and 282 in tags): return None
                x_res = self.get_rational(tags[282])
                y_res = self.get_rational(tags[283])
                x_pos = self.get_rational(tags[286])
                y_pos = self.get_rational(tags[287])
                return {
                    'path': path,
                    'abs_x': int(round(x_res * x_pos)),
                    'abs_y': int(round(y_res * y_pos))
                }
        except:
            return None

    def scan_metadata(self):
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.tif', '.tiff'))])
        if not files: return False

        logging.info(f"🔍 [{self.input_dir.name}] Quick-Scanning {len(files)} tiles...")

        first_file = self.input_dir / files[0]
        try:
            sample = tifffile.imread(first_file)
            self.dtype = sample.dtype

            if sample.ndim == 2:
                self.n_channels, self.tile_h, self.tile_w = 1, *sample.shape
            elif sample.ndim == 3:
                s = sample.shape
                if s[0] < s[1] and s[0] < s[2]:
                    self.n_channels, self.tile_h, self.tile_w = s
                else:
                    self.tile_h, self.tile_w, self.n_channels = s[0], s[1], s[2]

            with tifffile.TiffFile(first_file) as tif:
                names = self.extract_perkin_elmer_channels(tif)
                self.channel_names = names if len(names) == self.n_channels else [f"Channel {x}" for x in
                                                                                            range(self.n_channels)]

        except Exception as e:
            logging.error(f"❌ Error reading first tile header: {e}")
            return False

        file_paths = [self.input_dir / f for f in files]
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            results = list(executor.map(self._read_tile_meta_worker, file_paths))
        self.tiles = [r for r in results if r is not None]

        if not self.tiles: return False

        xs = [t['abs_x'] for t in self.tiles]
        ys = [t['abs_y'] for t in self.tiles]
        self.min_x, max_x = min(xs), max(xs)
        self.min_y, max_y = min(ys), max(ys)

        total_w = (max_x + self.tile_w) - self.min_x
        total_h = (max_y + self.tile_h) - self.min_y
        self.canvas_shape = (self.n_channels, total_h, total_w)

        logging.info(f"📊 [{self.input_dir.name}] Canvas: {total_w}x{total_h} pixels | Channels: {self.channel_names}")
        return True

    def calculate_ram_needed_gb(self):
        dtype_size = np.dtype(self.dtype).itemsize
        total_pixels = self.canvas_shape[0] * self.canvas_shape[1] * self.canvas_shape[2]
        return (total_pixels * dtype_size) / (1024 ** 3)

    def stitch(self, use_ram=True):
        target = None
        mode_name = "RAM" if use_ram else "DISK (Fallback)"

        try:
            if use_ram:
                logging.info(f"⬇️  [{self.input_dir.name}] Downloading & Stitching to RAM...")
                target = np.zeros(self.canvas_shape, dtype=self.dtype)
            else:
                logging.info(f"⚠️  [{self.input_dir.name}] Not enough RAM. Using DISK Mode (Slower)...")
                target = tifffile.memmap(self.temp_path, shape=self.canvas_shape, dtype=self.dtype, bigtiff=True)

            completed_tiles = 0
            total_tiles = len(self.tiles)

            def _load_and_paste(tile):
                nonlocal completed_tiles
                try:
                    img = tifffile.imread(tile['path'])
                    if img.ndim == 2:
                        img = img[np.newaxis, :, :]
                    elif img.ndim == 3 and img.shape[2] == self.n_channels and img.shape[0] != self.n_channels:
                        img = np.moveaxis(img, -1, 0)

                    if img.shape[0] != self.n_channels: return False

                    y = tile['abs_y'] - self.min_y
                    x = tile['abs_x'] - self.min_x
                    _, h, w = img.shape

                    h_fit = min(h, self.canvas_shape[1] - y)
                    w_fit = min(w, self.canvas_shape[2] - x)

                    if h_fit > 0 and w_fit > 0:
                        target[:, y:y + h_fit, x:x + w_fit] = img[:, :h_fit, :w_fit]

                    completed_tiles += 1
                    if self.progress_callback:
                        self.progress_callback(completed_tiles, total_tiles, "Stitching")
                    return True
                except:
                    return False

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
                list(pool.map(_load_and_paste, self.tiles))

            if not use_ram: target.flush()
            if self.progress_callback: self.progress_callback(total_tiles, total_tiles, "Stitching")

            logging.info(f"✅ [{self.input_dir.name}] Stitching ({mode_name}) Complete.")
            return target

        except MemoryError:
            logging.error(f"❌ [{self.input_dir.name}] Out of Memory!")
            return None
        except Exception as e:
            logging.error(f"❌ Error during stitch: {e}")
            return None

    def write_to_disk(self, canvas, is_ram_mode):
        logging.info(f"💾 [{self.input_dir.name}] Formatting & Saving (Visiopharm Replica Mode)...")
        if self.progress_callback:
            self.progress_callback(0, 0, "Formatting...")

        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            # --- 1. DATA TRANSFORMATION (Crucial Step) ---
            # Current: (Channels, Height, Width) -> Planar (Slow on Network)
            # Target:  (Height, Width, Channels) -> Contiguous (Fast on Network)
            logging.info(f"   [{self.input_dir.name}] Converting to Pixel-Interleaved (H, W, C)...")

            # Use moveaxis to create a view (fast, no copy if possible)
            # FROM: (C, Y, X) -> TO: (Y, X, C)
            interleaved_data = np.moveaxis(canvas, 0, -1)

            # --- 2. CONFIGURATION ---
            # Match Visiopharm: LZW, 512px Tiles, Contiguous, OME-XML
            metadata = {'axes': 'YXC', 'Channel': {'Name': self.channel_names}}

            opts = dict(
                tile=(TILE_SIZE, TILE_SIZE),
                compression='lzw',
                planarconfig='CONTIG',  # <--- THE MAGIC KEY
                metadata=metadata
            )

            # --- 3. AUTO-CALCULATE PYRAMID LEVELS ---
            # Generate levels until dimension < 256 (matches Visiopharm 8 levels)
            h, w = interleaved_data.shape[:2]
            n_levels = 0
            while max(h, w) > 256:
                h //= 2
                w //= 2
                n_levels += 1

            logging.info(f"   [{self.input_dir.name}] Will generate {n_levels} Sub-resolutions.")

            with tifffile.TiffWriter(self.output_path, bigtiff=True) as tif:

                # Write Level 0 (Full Res)
                if self.progress_callback: self.progress_callback(0, 0, "Writing Full Res (Heavy)...")
                logging.info(f"   [{self.input_dir.name}] Writing Level 0 (Contiguous)...")
                tif.write(interleaved_data, subifds=n_levels, **opts)

                # Write Pyramids
                prev = interleaved_data
                for level in range(1, n_levels + 1):
                    if self.progress_callback:
                        self.progress_callback(level, n_levels, f"Writing Pyramid {level}/{n_levels}")

                    logging.info(f"   [{self.input_dir.name}] Downsampling Level {level}...")

                    # Downsample (Average pooling for smoother look)
                    # We stride only on H and W (indices 0 and 1), keeping C (index 2) intact
                    curr = prev[::2, ::2, :]

                    logging.info(f"   [{self.input_dir.name}] Writing Level {level}...")
                    tif.write(curr, **opts)
                    prev = curr

            final_size = os.path.getsize(self.output_path) / (1024 ** 2)
            logging.info(f"🎉 [{self.input_dir.name}] Success! (Size: {final_size:.1f} MB)")
            if self.progress_callback: self.progress_callback(100, 100, "Finished")

        except Exception as e:
            logging.error(f"❌ [{self.input_dir.name}] Write Failed: {e}")
            if os.path.exists(self.output_path) and os.path.getsize(self.output_path) < 50000:
                try:
                    os.remove(self.output_path)
                except:
                    pass
        finally:
            if not is_ram_mode:
                del canvas
                if self.temp_path.exists():
                    try:
                        os.remove(self.temp_path)
                    except:
                        pass
            else:
                del canvas

            # ==========================================


# 🛑 GUI LOGIC
# ==========================================

class StitcherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("It's your Favourite Stitcher")
        self.geometry("750x700")

        # Directories
        tk.Label(self, text="Input Directory (Root Folder):").pack(pady=(10, 0))
        self.input_entry = tk.Entry(self, width=60)
        self.input_entry.pack(pady=5)
        tk.Button(self, text="Browse...", command=self.browse_input).pack(pady=5)

        tk.Label(self, text="Output Directory:").pack(pady=(10, 0))
        self.output_entry = tk.Entry(self, width=60)
        self.output_entry.pack(pady=5)
        tk.Button(self, text="Browse...", command=self.browse_output).pack(pady=5)

        self.start_btn = tk.Button(self, text="Stitch Stitch!", command=self.start_thread, bg="#4CAF50", fg="white",
                                   font=("Arial", 12, "bold"))
        self.start_btn.pack(pady=20)

        # Progress
        self.progress_label = tk.Label(self, text="Idle")
        self.progress_label.pack(pady=(10, 0))
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', mode='determinate', length=500)
        self.progress_bar.pack(pady=10)

        # Logs
        tk.Label(self, text="Real-time Logs:").pack(pady=(10, 0), anchor="w", padx=20)
        self.log_area = scrolledtext.ScrolledText(self, state='disabled', height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        handler = TextHandler(self.log_area)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.after(200, self.show_welcome_message)

    def show_welcome_message(self):
        welcome_title = "HELLO WORLD!"
        welcome_text = (
            "Welcome to your FAVOURITE Stitcher!\n\n"
            "It is important to note that this Stitcher only works with AKOYA VECTRA POLARIS/PHENOIMAGER unmixed tiles\n"
            "Reason to use this:\n"
            "1. You can run this in background \n"
            "2. You can stitched multiple folders sequentially without you keep clicking \n"
            "3. You don't have to pay £5/hr\n"
            "Note: This app uses your RAM/disk to store tmp files, make sure you have some space left \n"
            "If your input/output file is in network drive, the stitching speed is 99% depends on your network speed \n"
            "App developed by Jacky@Ramachandran Lab\n\n"
            "Logic adopted from Pete Bankhead AKA Qupath Founder\n"
            "Enjoy!"
        )
        messagebox.showinfo(welcome_title, welcome_text)

    def browse_input(self):
        d = filedialog.askdirectory();
        self.input_entry.delete(0, tk.END);
        self.input_entry.insert(0, d)

    def browse_output(self):
        d = filedialog.askdirectory();
        self.output_entry.delete(0, tk.END);
        self.output_entry.insert(0, d)

    def update_progress(self, current, total, phase_text):
        if total > 0:
            percent = (current / total) * 100
            self.progress_bar.config(mode='determinate')
            self.progress_bar['value'] = percent
            self.progress_label.config(text=f"{phase_text}: {current}/{total} ({percent:.1f}%)")
        else:
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start(10)
            self.progress_label.config(text=f"{phase_text}")
        self.update_idletasks()

    def start_thread(self):
        inp, out = self.input_entry.get(), self.output_entry.get()
        if not inp or not out: messagebox.showerror("Error", "Select paths."); return
        self.start_btn.config(state="disabled", text="Processing...")
        threading.Thread(target=self.run_process, args=(inp, out), daemon=True).start()

    def run_process(self, input_root, output_root):
        valid_folders = [root for root, _, files in os.walk(input_root) if any(f.endswith('.tif') for f in files)]

        if not valid_folders:
            messagebox.showwarning("No Images", "No input folders found!")
            self.start_btn.after(0, lambda: self.start_btn.config(state="normal", text="Start Processing"))
            return

        for folder in valid_folders:
            folder_name = os.path.basename(folder)
            out_file = os.path.join(output_root, f"{folder_name}.ome.tif")

            if os.path.exists(out_file):
                logging.info(f"⏩ Skipping {folder_name} (Exists)")
                continue

            stitcher = StitchingEngine(folder, out_file, self.update_progress)
            if not stitcher.scan_metadata(): continue

            needed_gb = stitcher.calculate_ram_needed_gb()
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            use_ram = (needed_gb + 2.0 < available_gb)

            while use_ram and (psutil.virtual_memory().available / (1024 ** 3) < needed_gb + 2.0):
                self.progress_label.config(text="Waiting for RAM...")
                time.sleep(5)

            result_data = stitcher.stitch(use_ram=use_ram)
            if result_data is not None:
                stitcher.write_to_disk(result_data, use_ram)
                del result_data

        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate', value=100)
        self.progress_label.config(text="Finished")
        self.start_btn.after(0, lambda: self.start_btn.config(state="normal", text="Start Processing"))
        messagebox.showinfo("Done", "Processing Complete!")


if __name__ == "__main__":
    StitcherApp().mainloop()