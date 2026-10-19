# === CONFIGURATION ===
# Specify which box file to read and what to save as
INPUT_BOX_FILENAME = 'minarea_040_box.txt'           # <-- Change this as needed
OUTPUT_BOX_FILENAME = 'minarea_calibration_570r_500n_box.txt'  # <-- Change this as needed

import os
import glob
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

def read_coords(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    coords = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    return np.array(coords)

def write_coords(filepath, coords):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")

def get_length_and_width(coords):
    # Assume coords are 4 points of a rectangle in order
    dists = [np.linalg.norm(coords[i] - coords[(i+1)%4]) for i in range(4)]
    length = max(dists)
    width = min(dists)
    # Find indices for length endpoints
    idx = np.argmax(dists)
    p1, p2 = coords[idx], coords[(idx+1)%4]
    return length, width, p1, p2

def adjust_length(coords, new_length):
    # Find length endpoints
    length, width, p1, p2 = get_length_and_width(coords)
    center = (p1 + p2) / 2
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    new_p1 = center - direction * new_length / 2
    new_p2 = center + direction * new_length / 2
    # Find the other two points (width direction)
    # Find the vector perpendicular to direction
    perp = np.array([-direction[1], direction[0]])
    # Find width endpoints from original box
    # Compute which original points are closest to p1 and p2
    idxs = np.argsort([np.linalg.norm(coords[i] - p1) for i in range(4)])[:2]
    # Use width from original box
    new_p3 = new_p2 + perp * width
    new_p4 = new_p1 + perp * width
    # Reorder to match original order as much as possible
    new_coords = np.array([new_p1, new_p2, new_p3, new_p4])
    return new_coords

def get_folder_id_from_path(path):
    # Extract folder name, e.g., gz3d_1-103960_127_14715653
    return os.path.basename(os.path.dirname(path))

def get_zooniverse_subject_id_from_folder(folder_id):
    # Extract the last part after the last underscore
    # e.g., gz3d_1-1038_61_14709525 -> 14709525
    return folder_id.split('_')[-1]

def get_ra_dec_from_gz3d(folder_id, gz3d_df):
    # Use zooniverse_subject_id for matching
    subject_id = get_zooniverse_subject_id_from_folder(folder_id)
    id_col = None
    for candidate in ['zooniverse_subject_id', 'subject_id', 'subjectid']:
        if candidate in gz3d_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"Could not find zooniverse_subject_id column in gz3d_metadata.csv. Columns: {gz3d_df.columns.tolist()}")
    row = gz3d_df[gz3d_df[id_col] == subject_id]
    if row.empty:
        raise ValueError(f"zooniverse_subject_id {subject_id} (from folder {folder_id}) not found in gz3d_metadata.csv. Example values: {gz3d_df[id_col].head(5).tolist()}")
    # Find RA/DEC column names (case-insensitive)
    ra_col = next((c for c in gz3d_df.columns if c.lower() == 'ra'), None)
    dec_col = next((c for c in gz3d_df.columns if c.lower() == 'dec'), None)
    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not find RA/DEC columns in gz3d_metadata.csv. Columns: {gz3d_df.columns.tolist()}")
    return float(row.iloc[0][ra_col]), float(row.iloc[0][dec_col])

def get_redshift_from_nsa(ra, dec, nsa_df, tol=1e-4):
    # Find RA/DEC column names (case-insensitive)
    ra_col = next((c for c in nsa_df.columns if c.lower() in ['ra', 'ra_nsa']), None)
    dec_col = next((c for c in nsa_df.columns if c.lower() in ['dec', 'dec_nsa']), None)
    redshift_col = next((c for c in nsa_df.columns if 'redshift' in c.lower()), None)
    if ra_col is None or dec_col is None or redshift_col is None:
        raise ValueError(f"Could not find RA/DEC/redshift columns in NSA catalogue. Columns: {nsa_df.columns.tolist()}")
    # Find NSA row with closest ra/dec within tolerance
    diff = np.abs(nsa_df[ra_col].astype(float) - ra) + np.abs(nsa_df[dec_col].astype(float) - dec)
    min_idx = diff.idxmin()
    if diff[min_idx] > tol:
        raise ValueError(f"No NSA match for ra={ra}, dec={dec} (min diff={diff[min_idx]})")
    return float(nsa_df.loc[min_idx, redshift_col])

def pixel_to_kpc(length_px, redshift, cosmo):
    # 1 px = 0.099 arcsec
    bar_length_arcsec = length_px * 0.099
    ang_diam_dist = cosmo.angular_diameter_distance(redshift).value  # in Mpc
    # Convert to kpc (as in snippet): arcsec * D_A / 117
    bar_length_kpc = bar_length_arcsec * ang_diam_dist * 1000 / 117  # Mpc to kpc
    return bar_length_kpc

def kpc_to_pixel(length_kpc, redshift, cosmo):
    ang_diam_dist = cosmo.angular_diameter_distance(redshift).value  # in Mpc
    bar_length_arcsec = length_kpc * 117 / (ang_diam_dist * 1000)
    length_px = bar_length_arcsec / 0.099
    return length_px

def process_box_file(box_file, gz3d_df, nsa_df, cosmo):
    coords = read_coords(box_file)
    length, width, p1, p2 = get_length_and_width(coords)
    folder_id = get_folder_id_from_path(box_file)
    ra, dec = get_ra_dec_from_gz3d(folder_id, gz3d_df)
    redshift = get_redshift_from_nsa(ra, dec, nsa_df)
    # Convert pixel length to kpc
    length_kpc = pixel_to_kpc(length, redshift, cosmo)
    # Calibration equation in kpc
    y_kpc = 0.570 * length_kpc + 0.500
    # Convert calibrated length back to pixels
    y_px = kpc_to_pixel(y_kpc, redshift, cosmo)
    new_coords = adjust_length(coords, y_px)
    # Save using OUTPUT_BOX_FILENAME in the same folder as input
    out_file = os.path.join(os.path.dirname(box_file), OUTPUT_BOX_FILENAME)
    write_coords(out_file, new_coords)
    print(f"Saved calibrated box to: {out_file}")  # Add this line for confirmation

def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'masks')
    # Use INPUT_BOX_FILENAME for pattern
    pattern = os.path.join(base_dir, '**', INPUT_BOX_FILENAME)
    print(f"Searching for files with pattern: {pattern}")  # Debug print
    print(f"Output files will be saved as: {OUTPUT_BOX_FILENAME}")  # Debug print

    # Directory: /data/rajit/masks
    #   minarea_035_box.txt
    #   minarea_040_box.txt
    #   ...
    # List contents of base_dir and subdirectories for debugging
    for root, dirs, files_in_dir in os.walk(base_dir):
        print(f"Directory: {root}")
        for file in files_in_dir:
            print(f"  {file}")

    files = glob.glob(pattern, recursive=True)
    print(f"Files found: {files}")  # Debug print
    print(f"Found {len(files)} files to process.")
    # Load gz3d_metadata and NSA catalogue
    gz3d_csv = os.path.join(os.path.dirname(__file__), 'gz3d_metadata.csv')
    nsa_csv = os.path.join(os.path.dirname(__file__), 'nsa_catalog.csv')
    gz3d_df = pd.read_csv(gz3d_csv)
    nsa_df = pd.read_csv(nsa_csv)
    print("gz3d_metadata.csv columns:", gz3d_df.columns.tolist())  # Debug printī
    print("nsa_catalog.csv columns:", nsa_df.columns.tolist())     # Debug print
    for col in gz3d_df.columns:
        gz3d_df[col] = gz3d_df[col].astype(str)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    processed = 0
    skipped = 0
    for idx, box_file in enumerate(files, 1):
        try:
            process_box_file(box_file, gz3d_df, nsa_df, cosmo)
            print(f"[{idx}/{len(files)}] Processed: {box_file}")
            processed += 1
        except Exception as e:
            print(f"[{idx}/{len(files)}] Skipped: {box_file} | Reason: {e}")
            skipped += 1
    print(f"\nProcessing complete. {processed} files processed, {skipped} files skipped.")
    print(f"\nProcessing complete. {processed} files processed, {skipped} files skipped.")

if __name__ == "__main__":
    main()
    # No additional code needed here. All logic is handled in main().