import os
import glob
import shutil
import numpy as np
from astropy.io import fits
from PIL import Image
import cv2

# --------------------------
# Paths
# --------------------------
src_dir = "/data4/rajit/DATA_LEGACY_DIR"
dst_dir_linear = "/data4/rajit/DATA_LEGACY_CONVERTED_BW/linear"
best_dir = "/data4/rajit/DATA_LEGACY_CONVERTED_BW/BEST"

os.makedirs(dst_dir_linear, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

# --------------------------
# Functions
# --------------------------
def safe_normalize(arr):
    arr = np.nan_to_num(arr)
    arr -= np.min(arr)
    max_val = np.max(arr)
    return arr / max_val if max_val > 0 else np.zeros_like(arr)

def enhance_bar(arr, clipLimit, tileGridSize):
    arr = safe_normalize(arr)
    arr_uint8 = (arr * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(arr_uint8)

def compute_bar_strength(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    height, width = arr.shape
    cy, cx = height // 2, width // 2
    max_r = min(cx, cy, 100)
    dr = 2
    r_bins = np.arange(dr, max_r, step=dr)
    a2_values = []

    yy, xx = np.meshgrid(np.arange(height), np.arange(width))
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    thetas = np.arctan2(dy, dx)

    for r in r_bins:
        mask = (dist >= r - dr/2) & (dist < r + dr/2)
        if np.sum(mask) < 10:
            continue
        intensities = arr[mask]
        theta_bin = thetas[mask]
        total_intensity = np.sum(intensities)
        if total_intensity == 0:
            continue
        cos_sum = np.sum(intensities * np.cos(2 * theta_bin))
        sin_sum = np.sum(intensities * np.sin(2 * theta_bin))
        a2 = np.sqrt(cos_sum**2 + sin_sum**2) / total_intensity
        a2_values.append(a2)

    return np.max(a2_values) if a2_values else 0.0

# --------------------------
# Parameters
# --------------------------
clipLimit_range = np.linspace(1.0, 5.0, 10)
tileGridSize_range = [(i, i) for i in range(3, 8, 1)]
MAX_VARIANTS = 50

# --------------------------
# Main Loop
# --------------------------

# Track processing stats
fits_files = glob.glob(os.path.join(src_dir, "*.fits")) \
           + glob.glob(os.path.join(src_dir, "*.fit")) \
           + glob.glob(os.path.join(src_dir, "*.fz"))
total_files = len(fits_files)
processed_count = 0
skipped_count = 0
error_files = []

for f in fits_files:
    try:
        with fits.open(f) as hdul:
            data = hdul[0].data
            if data is None or data.ndim < 3 or data.shape[0] < 3:
                print(f"Skipping {f}: Not a multi-band FITS file with at least 3 layers.")
                skipped_count += 1
                error_files.append(f)
                continue

            g_band, r_band, z_band = data[0], data[1], data[2]
            combined_data = (g_band + r_band + z_band) / 3.0
            base_name = os.path.splitext(os.path.basename(f))[0]

            best_strength = 0.0
            best_img = None
            best_info = {"clip": None, "tile": None}

            count = 0
            for clipLimit in clipLimit_range:
                for tileGridSize in tileGridSize_range:
                    if count >= MAX_VARIANTS:
                        break
                    bw_enhanced = enhance_bar(combined_data, clipLimit, tileGridSize)

                    temp_img = Image.fromarray(bw_enhanced)
                    temp_path = f"/tmp/{base_name}_temp.png"
                    temp_img.save(temp_path)
                    strength = compute_bar_strength(temp_path)
                    os.remove(temp_path)

                    if strength > best_strength:
                        best_strength = strength
                        best_img = bw_enhanced.copy()
                        best_info = {"clip": clipLimit, "tile": tileGridSize}

                    count += 1
                if count >= MAX_VARIANTS:
                    break

            if best_img is not None:
                best_name = f"{base_name}__BEST__clip_{best_info['clip']:.1f}__tile_{best_info['tile'][0]}x{best_info['tile'][1]}.png"
                Image.fromarray(best_img).save(os.path.join(best_dir, best_name))

            # Update progress bar
            processed_count += 1
            bar_length = 40
            percent = processed_count / total_files
            filled_length = int(bar_length * percent)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: |{bar}| {processed_count}/{total_files} completed", end='')

    except Exception as e:
        print(f"Error processing {f}: {e}")
        skipped_count += 1
        error_files.append(f)

print("Linear scale processing complete.")
print(f"\nSummary:")
print(f"Total FITS files found: {total_files}")
print(f"Successfully processed: {processed_count}")
print(f"Skipped/corrupted: {skipped_count}")
if error_files:
    print("Files with errors/skipped:")
    for ef in error_files:
        print(f"  {ef}")