import os
import time
import math
import tifffile  # Used only for fast metadata reading
from pathlib import Path
import pyvips
os.chdir('/home/jacky/stitch/')
# Try importing pyvips (Handle error if missing)
try:
    import pyvips
except OSError:
    print("❌ CRITICAL ERROR: Pyvips library not found.")
    print("   On Windows? You MUST download libvips binaries and add the 'bin' folder to your PATH.")
    print("   Download: https://github.com/libvips/libvips/releases")
    exit(1)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_DIRECTORY = '.'
OUTPUT_FILENAME = './output/stitched_commercial.ome.tif'

# Tunable Performance Settings
BATCH_SIZE = 100  # Group 100 tiles at a time (Balances graph depth vs RAM)
TILE_SIZE = 512  # Output tile size (standard for QuPath)
COMPRESSION = 'lzw'  # 'jpeg' (fastest/lossy) or 'lzw' (lossless/standard)
QUALITY = 90  # Only used if compression is 'jpeg'


# ==========================================

class CommercialStitcher:
    def __init__(self, input_dir, output_path):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.tiles = []
        self.min_x = 0
        self.min_y = 0

    def get_rational(self, tag):
        val = tag.value
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0] / val[1]
        return float(val)

    def scan_metadata(self):
        """
        Fast scan to get coordinates using tifffile (lighter than loading vips objects).
        """
        print(f"📂 Scanning metadata in {self.input_dir}...")
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.tif', '.tiff'))])

        if not files: raise ValueError("No TIFFs found.")

        TAG_X_RES, TAG_Y_RES = 282, 283
        TAG_X_POS, TAG_Y_POS = 286, 287

        for f in files:
            file_path = self.input_dir / f
            try:
                with tifffile.TiffFile(file_path) as tif:
                    page = tif.pages[0]
                    tags = page.tags

                    if not (TAG_X_POS in tags and TAG_X_RES in tags): continue

                    x_res = self.get_rational(tags[TAG_X_RES])
                    y_res = self.get_rational(tags[TAG_Y_RES])
                    x_pos = self.get_rational(tags[TAG_X_POS])
                    y_pos = self.get_rational(tags[TAG_Y_POS])

                    self.tiles.append({
                        'path': str(file_path),
                        'abs_x': int(round(x_res * x_pos)),
                        'abs_y': int(round(y_res * y_pos))
                    })
            except:
                pass

        if not self.tiles: raise ValueError("No tiles with metadata found.")

        # Normalize Coordinates
        xs = [t['abs_x'] for t in self.tiles]
        ys = [t['abs_y'] for t in self.tiles]
        self.min_x = min(xs)
        self.min_y = min(ys)

        print(f"✅ Metadata parsed for {len(self.tiles)} tiles.")
        print(f"   Origin Offset: X={self.min_x}, Y={self.min_y}")

    def stitch_batch(self, batch_tiles):
        """
        Merges a small list of tiles into one intermediate Vips image.
        Uses 'insert' with expand=True to grow the canvas dynamically.
        """
        # Start with the first tile in this batch
        first = batch_tiles[0]
        # access='sequential' tells VIPS we will read this file once, from top to bottom
        canvas = pyvips.Image.new_from_file(first['path'], access='sequential')

        # We must shift this first tile to its relative position in the batch context?
        # Actually, easiest way is to treat the first tile's position as the 'batch origin'
        # but simpler is to stitch them all relative to the global origin and crop later
        # OR just insert them into the canvas relative to the first image.

        # Better approach for VIPS:
        # Create a blank canvas? No, VIPS hates blank canvases (infinite size).
        # We start with the first image placed at its GLOBAL position?
        # No, 'insert' places 'canvas' onto 'base'.

        # CORRECT VIPS LOGIC:
        # 1. Start with the first image.
        # 2. Insert the second image at (x2 - x1, y2 - y1).

        base_x = first['abs_x'] - self.min_x
        base_y = first['abs_y'] - self.min_y

        # Embed the first image into a transparent black box if needed,
        # but 'insert' handles expansion automatically.

        for i in range(1, len(batch_tiles)):
            t = batch_tiles[i]

            # Load next tile
            overlay = pyvips.Image.new_from_file(t['path'], access='sequential')

            # Calculate relative position
            rel_x = (t['abs_x'] - self.min_x) - base_x
            rel_y = (t['abs_y'] - self.min_y) - base_y

            # Insert (expand=True automatically grows the image bounds)
            canvas = canvas.insert(overlay, rel_x, rel_y, expand=True)

        return canvas, base_x, base_y

    def run(self):
        start_time = time.time()

        # 1. Sort tiles (Spatial locality improves stitching speed)
        # Sorting by Y then X helps usually
        self.tiles.sort(key=lambda t: (t['abs_y'], t['abs_x']))

        # 2. Process in Batches (Map-Reduce style)
        # Splitting 10k images into chunks of 100 prevents building a graph 10k nodes deep.
        total_batches = math.ceil(len(self.tiles) / BATCH_SIZE)
        print(f"🚀 Processing {len(self.tiles)} tiles in {total_batches} batches...")

        batch_images = []

        for i in range(0, len(self.tiles), BATCH_SIZE):
            batch = self.tiles[i: i + BATCH_SIZE]
            print(f"   Stitching Batch {i // BATCH_SIZE + 1}/{total_batches}...", end='\r')

            # Create a mini-stitch of this batch
            img_chunk, chunk_x, chunk_y = self.stitch_batch(batch)
            batch_images.append((img_chunk, chunk_x, chunk_y))

        print("\n⚡ Batches prepared. Assembling final image...")

        # 3. Assemble Batches (The Final Merge)
        # Start with the first batch
        final_image, final_x, final_y = batch_images[0]

        # To avoid graph depth issues on the final merge if batches are huge,
        # we can just loop.
        for i in range(1, len(batch_images)):
            overlay, ox, oy = batch_images[i]

            # Calculate position relative to the main canvas start
            rel_x = ox - final_x
            rel_y = oy - final_y

            final_image = final_image.insert(overlay, rel_x, rel_y, expand=True)

        # 4. Write Pyramidal OME-TIFF
        print(f"💾 Saving to {self.output_path} (This leverages multi-core writing)...")

        # Ensure output dir exists
        os.makedirs(self.output_path.parent, exist_ok=True)

        # Set OME Metadata standard for QuPath
        final_image.set_type(pyvips.GValue.gint_type, "page-height", final_image.height)
        final_image.set_type(pyvips.GValue.gstr_type, "image-description",
                             f"""<?xml version="1.0" encoding="UTF-8"?>
                             <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" 
                                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                                  xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
                                 <Image ID="Image:0" Name="Stitched Image">
                                     <Pixels DimensionOrder="XYCZT" ID="Pixels:0" 
                                             SizeC="{final_image.bands}" SizeT="1" SizeX="{final_image.width}" SizeY="{final_image.height}" SizeZ="1" 
                                             Type="uint8">
                                     </Pixels>
                                 </Image>
                             </OME>""")

        # SAVE
        # tile=True: Saves as tiles (fast access)
        # pyramid=True: Generates zoom levels automatically
        # bigtiff=True: Allows > 4GB files
        final_image.tiffsave(
            self.output_path,
            compression=COMPRESSION,
            tile=True,
            tile_width=TILE_SIZE,
            tile_height=TILE_SIZE,
            pyramid=True,
            bigtiff=True,
            properties=True  # Keeps metadata
        )

        print(f"\n🎉 DONE! Total time: {time.time() - start_time:.2f}s")


# ==========================================
if __name__ == '__main__':
    stitcher = CommercialStitcher(INPUT_DIRECTORY, OUTPUT_FILENAME)
    stitcher.scan_metadata()
    stitcher.run()