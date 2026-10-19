import os
import glob
import cv2
import pandas as pd
from openpyxl import Workbook
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = "/data/rajit"
MASKS_DIR = os.path.join(BASE_DIR, 'masks')
BEST_DIR = os.path.join(BASE_DIR, 'DATA_LEGACY_CONVERTED_BW', 'BEST')
OUT_DIR = os.path.join(BASE_DIR, 'legacy_with_bounding_box')
NO_BBOX_DIR = os.path.join(OUT_DIR, 'images_no_bbox')
WITH_BBOX_DIR = os.path.join(OUT_DIR, 'images_with_bbox')
WITH_BBOX_LABELS_DIR = os.path.join(WITH_BBOX_DIR, 'labels')
# Ensure output directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(NO_BBOX_DIR, exist_ok=True)
os.makedirs(WITH_BBOX_DIR, exist_ok=True)
os.makedirs(WITH_BBOX_LABELS_DIR, exist_ok=True)
META_CSV = os.path.join(BASE_DIR, 'gz3d_metadata.csv')
EXCEL_OUT = os.path.join(OUT_DIR, 'legacy_bounding_boxes.xlsx')

# Helper to normalize IDs for robust matching
def normalize_id(s):
    return s.replace('-', '').replace('_', '').lower()

# Load metadata
meta_df = pd.read_csv(META_CSV)

# Normalize MANGAID column in metadata for robust lookup
meta_df['MANGAID_NORM'] = meta_df['MANGAID'].astype(str).apply(normalize_id)

# Helper to find legacy image by MANGAID (improved)
def find_legacy_image_by_mangaid(mangaid):
    norm_mangaid = normalize_id(mangaid)
    # Try direct pattern matches first
    pattern1 = os.path.join(BEST_DIR, f"{mangaid}_*.png")
    pattern2 = os.path.join(BEST_DIR, f"{mangaid}-*.png")
    matches = glob.glob(pattern1) + glob.glob(pattern2)
    if matches:
        return matches[0]
    # Fallback: normalize all filenames and compare
    for img in glob.glob(os.path.join(BEST_DIR, "*.png")):
        base = os.path.splitext(os.path.basename(img))[0]
        norm_base = normalize_id(base)
        if norm_mangaid in norm_base:
            return img
    return None

# Prepare Excel log
wb = Workbook()
ws = wb.active
ws.append(['legacy_image', 'annotated_image', 'bbox_txt', 'ra', 'dec'])

# Track images with and without bounding boxes
images_with_bbox = set()
all_best_images = set([os.path.splitext(f)[0] for f in os.listdir(BEST_DIR) if f.endswith('.png')])

skipped_folders = []

mask_folders = os.listdir(MASKS_DIR)
for mask_folder in tqdm(mask_folders, desc="Processing masks"):
    folder_path = os.path.join(MASKS_DIR, mask_folder)
    bbox_path = os.path.join(folder_path, 'minarea_calibration_570r_500n_box.txt')
    if not os.path.isfile(bbox_path):
        skipped_folders.append((mask_folder, "No bbox file"))
        continue
    # Extract MANGAID from folder name (e.g., gz3d_1-100017_61_14715415)
    try:
        parts = mask_folder.split('_')
        mangaid = parts[1]  # '1-100017'
    except Exception:
        skipped_folders.append((mask_folder, "Failed to extract mangaid"))
        continue
    mangaid_norm = normalize_id(mangaid)
    # Lookup RA/DEC in metadata using normalized ID
    row = meta_df[meta_df['MANGAID_NORM'] == mangaid_norm]
    if row.empty:
        skipped_folders.append((mask_folder, f"MANGAID {mangaid} not found in metadata"))
        continue
    ra = str(row.iloc[0]['ra'])
    dec = str(row.iloc[0]['dec'])
    # Find legacy image by MANGAID
    legacy_img = find_legacy_image_by_mangaid(mangaid)
    if not legacy_img:
        skipped_folders.append((mask_folder, f"No matching image for MANGAID {mangaid}"))
        continue
    # Read bounding box
    with open(bbox_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Parse as float, then convert to int for drawing
    bbox = [tuple(map(float, l.split())) for l in lines]
    bbox_int = [tuple(map(int, map(round, pt))) for pt in bbox]
    # Draw bbox as polygon (not convex hull)
    img = cv2.imread(legacy_img)
    if img is None:
        continue
    pts = np.array(bbox_int, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
    # Save outputs
    base_name = os.path.splitext(os.path.basename(legacy_img))[0]
    out_img = os.path.join(OUT_DIR, base_name + '_bbox.png')
    out_txt = os.path.join(OUT_DIR, base_name + '_bbox.txt')
    cv2.imwrite(out_img, img)
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    # Log to Excel
    ws.append([legacy_img, out_img, out_txt, ra, dec])
    images_with_bbox.add(base_name)
    # Also save the original image to images_with_bbox
    orig_img_dst = os.path.join(WITH_BBOX_DIR, os.path.basename(legacy_img))
    cv2.imwrite(orig_img_dst, cv2.imread(legacy_img))
    # Also save the bbox txt to images_with_bbox/labels
    label_dst = os.path.join(WITH_BBOX_LABELS_DIR, base_name + '.txt')
    with open(label_dst, 'w') as f:
        f.write('\n'.join(lines))

# Save images from BEST with no bounding box (no corresponding bbox txt in OUT_DIR)
for img_file in os.listdir(BEST_DIR):
    if img_file.endswith('.png'):
        base = os.path.splitext(img_file)[0]
        if base not in images_with_bbox:
            src = os.path.join(BEST_DIR, img_file)
            dst = os.path.join(NO_BBOX_DIR, img_file)
            cv2.imwrite(dst, cv2.imread(src))

# Calculate and print completion percentage
total_galaxies = len([f for f in os.listdir(BEST_DIR) if f.endswith('.png')])
completed_galaxies = len(images_with_bbox)
if total_galaxies > 0:
    percent_completed = (completed_galaxies / total_galaxies) * 100
else:
    percent_completed = 0.0
print(f"Completed: {completed_galaxies} / {total_galaxies} galaxies ({percent_completed:.2f}%)")

wb.save(EXCEL_OUT)
print('Processing complete. Results saved in', OUT_DIR)
print(f'Images without bounding boxes saved in {NO_BBOX_DIR}')
print(f'Images with bounding boxes saved in {WITH_BBOX_DIR}')
print(f'Labels for images with bounding boxes saved in {WITH_BBOX_LABELS_DIR}')

# At the end, print summary of skipped folders
if skipped_folders:
    print("\nSkipped mask folders and reasons:")
    for folder, reason in skipped_folders:
        print(f"{folder}: {reason}")
else:
    print("\nAll mask folders processed successfully.")
