import tifffile
import numpy as np
import xml.etree.ElementTree as ET
import os

filename = "Visio_Replica_Test.ome.tif"
# Exact channel names from your XML file
channel_names = ["DAPI", "Opal 480", "Opal 520", "Opal 570", "Opal 620", "Opal 690", "Opal 780", "Autofluorescence"]

print(f"🧪 Creating Visiopharm Replica with {len(channel_names)} channels...")

# 1. Create Data (512x512, 8 Channels, Contiguous)
# We use (Height, Width, Channel) shape for Visiopharm speed
data = np.zeros((512, 512, len(channel_names)), dtype=np.uint8)

# Add stripes so you can see data
for i in range(len(channel_names)):
    data[i * 50:(i + 1) * 50, :, i] = 255


# 2. GENERATE EXACT VISIOPHARM XML
def get_visio_xml():
    ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    root = ET.Element("OME", xmlns=ns)
    image = ET.SubElement(root, "Image", ID="Image:0", Name="Visio_Replica")

    # Header tags strictly matching your file
    pixels = ET.SubElement(image, "Pixels",
                           ID="Pixels:0",
                           DimensionOrder="XYCZT",
                           Type="uint8",
                           SizeX="512",
                           SizeY="512",
                           SizeC=str(len(channel_names)),
                           SizeZ="1",
                           SizeT="1",
                           Interleaved="true")  # Crucial tag

    # The Fix: Set SamplesPerPixel equal to Total Channels (8)
    for i, name in enumerate(channel_names):
        ET.SubElement(pixels, "Channel",
                      ID=f"Channel:0:{i}",
                      Name=name,
                      SamplesPerPixel=str(len(channel_names)))  # <--- THE KEY FIX: "8" not "1"

    # Link to TIFF Data (Explicitly says "Start at IFD 0")
    ET.SubElement(pixels, "TiffData", FirstZ="0", IFD="0")

    return ET.tostring(root, encoding='utf-8')


# 3. WRITE TO DISK
try:
    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        tif.write(
            data,
            tile=(512, 512),
            compression='lzw',
            planarconfig='CONTIG',  # Fast Visiopharm Style
            photometric='minisblack',
            description=get_visio_xml(),  # Force our Correct Metadata
            metadata=None  # Disable auto-generation
        )

    print(f"✅ Created {filename}")
    print("ACTION: Drag this into QuPath.")
    print("1. It should load instantly.")
    print("2. It should show exactly 8 channels (DAPI, Opal 480...).")

except Exception as e:
    print(f"❌ Failed: {e}")