import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Directory containing the FITS files (server paths)
data_dir = '/data/rajit/DATA_GZ3D_DIR/'
output_dir = '/data/rajit/masks'
os.makedirs(output_dir, exist_ok=True)

# Get all FITS files (process all)
fits_files = [f for f in os.listdir(data_dir) if f.endswith('.fits.gz')]

def process_file(fname):
    base = os.path.splitext(os.path.splitext(fname)[0])[0]
    fits_path = os.path.join(data_dir, fname)
    subfolder = os.path.join(output_dir, base)
    galaxy_img_path = os.path.join(subfolder, 'galaxy.png')
    bar_mask_img_path = os.path.join(subfolder, 'bar_mask.png')
    superimposed_img_path = os.path.join(subfolder, 'superimposed.png')

    # Skip if all output images already exist
    if os.path.exists(galaxy_img_path) and os.path.exists(bar_mask_img_path) and os.path.exists(superimposed_img_path):
        return f"Skipped {fname}: Already processed."

    # Try to open FITS file, skip if corrupted
    try:
        hdu_list = fits.open(fits_path)
    except Exception:
        return f"Skipped {fname}: FITS file is empty or corrupt."

    # Check for bar mask (hdu_list[4])
    bar_mask_valid = (
        len(hdu_list) >= 5 and
        hdu_list[4].data is not None and
        np.any(np.isfinite(hdu_list[4].data)) and
        np.any(hdu_list[4].data != 0) and
        not np.all(np.isnan(hdu_list[4].data))
    )
    if not bar_mask_valid:
        hdu_list.close()
        return f'Skipped {fname}: No valid bar mask present.'

    # Create a subfolder for this dataset
    os.makedirs(subfolder, exist_ok=True)

    # Save galaxy image
    galaxy_data = hdu_list[0].data
    plt.imsave(galaxy_img_path, galaxy_data, origin='lower')

    # Save bar mask image
    bar_mask_data = hdu_list[4].data
    plt.imsave(bar_mask_img_path, bar_mask_data, origin='lower', cmap='viridis')

    # Load as RGBA
    galaxy_img = np.array(Image.open(galaxy_img_path).convert('RGBA'))
    bar_img = np.array(Image.open(bar_mask_img_path).convert('RGBA'))
    if bar_img.shape != galaxy_img.shape:
        bar_img = np.array(Image.fromarray(bar_img).resize((galaxy_img.shape[1], galaxy_img.shape[0]), Image.BILINEAR))

    # Remove purple from bar_img
    purple_rgb = np.array([68, 1, 84])
    purple_tol = 10
    mask_purple = np.all(np.abs(bar_img[..., :3] - purple_rgb) < purple_tol, axis=-1)
    bar_gray = np.array(Image.open(bar_mask_img_path).convert('L'))
    alpha_mask = (bar_gray > 10).astype(np.uint8) * 180
    bar_img[..., 3] = np.where(mask_purple, 0, alpha_mask)

    # Superimpose
    superimposed = galaxy_img.copy()
    bar_alpha = bar_img[..., 3:4] / 255.0
    superimposed[..., :3] = (1 - bar_alpha) * superimposed[..., :3] + bar_alpha * bar_img[..., :3]
    Image.fromarray(superimposed.astype(np.uint8)).save(superimposed_img_path)
    hdu_list.close()
    return f'Saved: {galaxy_img_path}, {bar_mask_img_path}, {superimposed_img_path}'

with ThreadPoolExecutor() as executor:
    for msg in tqdm(executor.map(process_file, fits_files), total=len(fits_files), desc="Processing FITS files"):
        print(msg)

print('Batch processing complete.')
