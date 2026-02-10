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


class ChannelDialog(tk.Toplevel):
    def __init__(self, parent, image_name, current_names):
        super().__init__(parent)
        self.title("Fix Channel Names")
        self.geometry("600x450")
        self.result = None
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text=f"⚠️ Suspicious Channels Detected!", font=("Arial", 12, "bold"), fg="#e74c3c").pack(
            pady=(15, 5))
        tk.Label(self, text=f"File: {image_name}", font=("Arial", 10, "bold")).pack()
        tk.Label(self, text="Detected duplicates (e.g. all 'DAPI').\nEnter correct names (comma separated):",
                 justify=tk.CENTER).pack(pady=10)

        self.text_area = tk.Text(self, height=5, width=60, font=("Arial", 10))
        self.text_area.insert(tk.END, ", ".join(current_names))
        self.text_area.pack(pady=5, padx=20)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="✅ Use These Names", command=self.on_ok, bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ Ignore", command=self.on_cancel, padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    def on_ok(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Names cannot be empty!")
            return
        self.result = [x.strip() for x in text.split(',')]
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


class StitchingEngine:
    def __init__(self, input_dir, output_path, progress_callback=None, ask_user_callback=None):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.progress_callback = progress_callback
        self.ask_user_callback = ask_user_callback
        self.temp_path = self.output_path.with_suffix('.tmp.raw')

        self.tiles = []
        self.canvas_shape = None
        self.dtype = None
        self.channel_names = []
        self.min_x, self.min_y = 0, 0
        self.tile_w, self.tile_h = 0, 0
        self.n_channels = 0

        # Default Resolution (Fallback = 0.5um/pixel if metadata is missing)
        self.phys_size_x_um = 0.5
        self.phys_size_y_um = 0.5
        self.tiff_res_val = (20000, 20000)  # Pixels per Unit
        self.tiff_res_unit = 3  # 3 = Centimeter

    def get_rational(self, tag):
        try:
            val = tag.value
            if isinstance(val, (tuple, list)) and len(val) == 2:
                # Handle (Numerator, Denominator)
                if val[1] == 0: return 0
                return val[0] / val[1]
            return float(val)
        except:
            return 0

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
                # Normalize to (C, H, W)
                if s[0] < s[1] and s[0] < s[2]:
                    self.n_channels, self.tile_h, self.tile_w = s
                else:
                    self.tile_h, self.tile_w, self.n_channels = s[0], s[1], s[2]

            with tifffile.TiffFile(first_file) as tif:
                # 1. Extract Channels
                extracted = self.extract_perkin_elmer_channels(tif)
                is_suspicious = False
                if extracted and len(extracted) == self.n_channels:
                    unique = set(extracted)
                    if len(unique) < len(extracted): is_suspicious = True

                if extracted and len(extracted) == self.n_channels and not is_suspicious:
                    self.channel_names = extracted
                else:
                    logging.warning(f"⚠️ Channel Name Issue: {extracted}")
                    if self.ask_user_callback:
                        user_names = self.ask_user_callback(self.input_dir.name,
                                                            extracted if extracted else ["?"] * self.n_channels)
                        if user_names and len(user_names) == self.n_channels:
                            self.channel_names = user_names
                        else:
                            self.channel_names = [f"Channel {x}" for x in range(self.n_channels)]
                    else:
                        self.channel_names = [f"Channel {x}" for x in range(self.n_channels)]

                # 2. Extract Resolution (The Smart Fix)
                page = tif.pages[0]
                tags = page.tags

                if 282 in tags and 283 in tags:
                    x_res_raw = self.get_rational(tags[282])  # Pixels per Unit
                    y_res_raw = self.get_rational(tags[283])
                    unit = tags[296].value if 296 in tags else 2  # Default to Inch (2) if missing

                    # Normalize to Pixels per Centimeter
                    if unit == 2:  # Inch
                        res_cm_x = x_res_raw / 2.54
                        res_cm_y = y_res_raw / 2.54
                    elif unit == 3:  # Centimeter
                        res_cm_x = x_res_raw
                        res_cm_y = y_res_raw
                    else:  # No unit/Unknown
                        res_cm_x = 20000  # Assume 0.5um
                        res_cm_y = 20000

                    # Store for TIFF writing (Pixels per CM)
                    self.tiff_res_val = (res_cm_x, res_cm_y)
                    self.tiff_res_unit = 3  # Centimeter

                    # Store for OME-XML (Microns per Pixel)
                    # 1 cm = 10,000 microns.  Microns/Pixel = 10,000 / (Pixels/CM)
                    if res_cm_x > 0: self.phys_size_x_um = 10000 / res_cm_x
                    if res_cm_y > 0: self.phys_size_y_um = 10000 / res_cm_y

                    logging.info(f"📏 Resolution Found: {self.phys_size_x_um:.4f} µm/pixel")
                else:
                    logging.warning("⚠️ No resolution tags found. Defaulting to 0.5 µm/pixel.")

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
                logging.info(f"⚠️  [{self.input_dir.name}] Using DISK Mode (Slower)...")
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
                    if self.progress_callback: self.progress_callback(completed_tiles, total_tiles, "Stitching")
                    return True
                except:
                    return False

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
                list(pool.map(_load_and_paste, self.tiles))

            if not use_ram: target.flush()
            if self.progress_callback: self.progress_callback(total_tiles, total_tiles, "Stitching")

            logging.info(f"✅ [{self.input_dir.name}] Stitching ({mode_name}) Complete.")
            return target
        except Exception as e:
            logging.error(f"❌ Stitch Failed: {e}")
            return None

    # --- 🛠️ VISIOPHARM-COMPATIBLE XML GENERATOR (FIXED + DYNAMIC RESOLUTION) ---
    def generate_visio_xml(self, size_y, size_x, size_c, dtype):
        try:
            ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
            root = ET.Element("OME", xmlns=ns)

            pixel_type = str(dtype)
            if 'uint8' in pixel_type:
                pixel_type = 'uint8'
            elif 'uint16' in pixel_type:
                pixel_type = 'uint16'
            elif 'float' in pixel_type:
                pixel_type = 'float'

            image = ET.SubElement(root, "Image", ID="Image:0", Name=self.input_dir.name)

            # Use the calculated Physical Size
            pixels = ET.SubElement(image, "Pixels",
                                   ID="Pixels:0",
                                   DimensionOrder="XYCZT",
                                   Type=pixel_type,
                                   SizeX=str(size_x),
                                   SizeY=str(size_y),
                                   SizeC=str(size_c),
                                   SizeZ="1",
                                   SizeT="1",
                                   PhysicalSizeX=str(self.phys_size_x_um),
                                   PhysicalSizeXUnit="µm",
                                   PhysicalSizeY=str(self.phys_size_y_um),
                                   PhysicalSizeYUnit="µm",
                                   Interleaved="true")

            for i, name in enumerate(self.channel_names):
                ET.SubElement(pixels, "Channel",
                              ID=f"Channel:0:{i}",
                              Name=name,
                              SamplesPerPixel=str(size_c))  # SamplesPerPixel = Total Channels

            ET.SubElement(pixels, "TiffData", FirstZ="0", IFD="0")

            return ET.tostring(root, encoding='utf-8')
        except Exception as e:
            logging.error(f"XML Gen Error: {e}")
            return None

    def write_to_disk(self, canvas, is_ram_mode):
        logging.info(f"💾 [{self.input_dir.name}] Formatting & Saving (Visiopharm Replica)...")
        if self.progress_callback: self.progress_callback(0, 0, "Formatting...")

        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            logging.info(f"   [{self.input_dir.name}] Converting to Interleaved (H, W, C)...")
            interleaved_data = np.moveaxis(canvas, 0, -1)
            h, w, c = interleaved_data.shape

            xml_bytes = self.generate_visio_xml(h, w, c, self.dtype)

            # Use the detected resolution
            opts_main = dict(
                tile=(TILE_SIZE, TILE_SIZE),
                compression='lzw',
                planarconfig='CONTIG',
                resolution=self.tiff_res_val,
                resolutionunit=self.tiff_res_unit,
                description=xml_bytes,
                metadata=None
            )

            opts_pyramid = dict(
                tile=(TILE_SIZE, TILE_SIZE),
                compression='lzw',
                planarconfig='CONTIG',
                resolution=self.tiff_res_val,
                resolutionunit=self.tiff_res_unit,
                metadata=None
            )

            n_levels = 0
            temp_h, temp_w = h, w
            while max(temp_h, temp_w) > 256:
                temp_h //= 2
                temp_w //= 2
                n_levels += 1

            logging.info(f"   [{self.input_dir.name}] Generating {n_levels} Pyramid Levels.")

            with tifffile.TiffWriter(self.output_path, bigtiff=True) as tif:

                if self.progress_callback: self.progress_callback(0, 0, "Writing Full Res...")
                logging.info(f"   [{self.input_dir.name}] Writing Level 0...")
                tif.write(interleaved_data, subifds=n_levels, **opts_main)

                prev = interleaved_data
                for level in range(1, n_levels + 1):
                    if self.progress_callback:
                        self.progress_callback(level, n_levels, f"Writing Pyramid {level}/{n_levels}")

                    logging.info(f"   [{self.input_dir.name}] Downsampling Level {level}...")

                    curr = prev[::2, ::2, :]

                    logging.info(f"   [{self.input_dir.name}] Writing Level {level}...")
                    tif.write(curr, **opts_pyramid)
                    prev = curr

            final_size = os.path.getsize(self.output_path) / (1024 ** 2)
            logging.info(f"🎉 [{self.input_dir.name}] Success! (Size: {final_size:.1f} MB)")
            if self.progress_callback: self.progress_callback(100, 100, "Finished")

        except Exception as e:
            logging.error(f"❌ [{self.input_dir.name}] Write Failed: {e}")
            if os.path.exists(self.output_path):
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

    def ask_user_for_channels(self, image_name, current_names):
        result_container = {}

        def _popup():
            dialog = ChannelDialog(self, image_name, current_names)
            self.wait_window(dialog)
            result_container['names'] = dialog.result

        self.after(0, _popup)
        while 'names' not in result_container: time.sleep(0.5)
        return result_container['names']

    def start_thread(self):
        inp, out = self.input_entry.get(), self.output_entry.get()
        if not inp or not out: messagebox.showerror("Error", "Select paths."); return
        self.start_btn.config(state="disabled", text="Processing...")
        threading.Thread(target=self.run_process, args=(inp, out), daemon=True).start()

    def run_process(self, input_root, output_root):
        valid_folders = [root for root, _, files in os.walk(input_root) if any(f.endswith('.tif') for f in files)]
        if not valid_folders:
            messagebox.showwarning("No Images", "No input folders found!")
            self.start_btn.after(0, lambda: self.start_btn.config(state="normal", text="Stitch Stitch!"))
            return

        for folder in valid_folders:
            folder_name = os.path.basename(folder)
            out_file = os.path.join(output_root, f"{folder_name}.ome.tif")
            if os.path.exists(out_file):
                logging.info(f"⏩ Skipping {folder_name} (Exists)")
                continue

            # Pass both callbacks here
            stitcher = StitchingEngine(folder, out_file, self.update_progress, self.ask_user_for_channels)
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
        self.start_btn.after(0, lambda: self.start_btn.config(state="normal", text="Stitch Stitch!"))
        messagebox.showinfo("Done", "Processing Complete!")


if __name__ == "__main__":
    StitcherApp().mainloop()