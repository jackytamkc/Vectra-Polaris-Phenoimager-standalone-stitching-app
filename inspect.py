import tifffile
import tkinter as tk
from tkinter import filedialog
import os


def analyze_tiff(path):
    print(f"\n🔍 ANALYZING: {os.path.basename(path)}")
    print("=" * 60)

    try:
        with tifffile.TiffFile(path) as tif:
            # 1. Check Global Headers
            print(f"📂 File Size:       {os.path.getsize(path) / (1024 ** 3):.2f} GB")
            print(f"⚙️  Byte Order:      {tif.byteorder}")
            print(f"📑 Total Pages:     {len(tif.pages)}")
            print(f"📚 Series Count:    {len(tif.series)}")

            # 2. Check OME-XML
            is_ome = tif.is_ome
            print(f"🧬 OME-XML Detect:  {is_ome}")

            # 3. Analyze Pyramid Structure
            print("\n📐 PAGE LAYOUT & PYRAMID STRUCTURE:")
            print("-" * 60)
            print(f"{'ID':<4} | {'Level':<10} | {'Dimensions (WxH)':<20} | {'Tile':<10} | {'Comp.':<8} | {'SubIFDs'}")
            print("-" * 60)

            for i, page in enumerate(tif.pages):
                # Basic Dims
                h, w = page.shape[:2] if page.ndim >= 2 else (0, 0)
                tile = page.chunks if page.is_tiled else "STRIP"

                # Compression
                comp = page.compression.name if page.compression else "NONE"

                # Check for Sub-resolutions (SubIFDs)
                subifds = len(page.subifds) if page.subifds else 0

                # Attempt to guess pyramid level based on size relative to Page 0
                if i == 0:
                    base_w, base_h = w, h
                    level = "FULL RES"
                else:
                    ratio = base_w / w
                    level = f"1/{int(ratio)}x"

                print(f"{i:<4} | {level:<10} | {w}x{h:<13} | {str(tile):<10} | {comp:<8} | {subifds}")

                # Limit output if there are hundreds of pages (e.g. z-stacks)
                if i > 10:
                    print("... (Stopping scan to avoid spam)")
                    break

            # 4. Deep Dive into Page 0 (The Main Image)
            p0 = tif.pages[0]
            print("\n🔬 PAGE 0 DEEP DIVE (Performance Factors):")
            print("-" * 60)
            print(f"🔹 Planar Config:   {p0.planarconfig.name} (Separated vs Contiguous)")
            print(f"🔹 Photometric:     {p0.photometric.name}")
            print(f"🔹 Predictor:       {p0.predictor.name} (Helps compression?)")
            print(f"🔹 Sample Format:   {p0.dtype}")

            # 5. Metadata Peek
            print("\n📝 METADATA SNIPPET (First 300 chars):")
            print("-" * 60)
            if p0.description:
                print(p0.description[:300] + "...")
            else:
                print("[No ImageDescription Tag]")

    except Exception as e:
        print(f"❌ Error reading file: {e}")


# --- GUI RUNNER ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window

    print("Select the Visiopharm stitched image...")
    file_path = filedialog.askopenfilename(
        title="Select Visiopharm Image",
        filetypes=[("TIFF Files", "*.tif *.tiff *.ome.tif")]
    )

    if file_path:
        analyze_tiff(file_path)
        input("\nPress Enter to exit...")
    else:
        print("No file selected.")