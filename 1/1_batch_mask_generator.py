import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from tqdm import tqdm  # Add tqdm for progress bar

# Directory containing the FITS files (server paths)
data_dir = '/data4/rajit/DATA_GZ3D_DIR'
output_dir = '/data4/rajit/masks'
os.makedirs(output_dir, exist_ok=True)

# Get all FITS files (process all)
fits_files = [f for f in os.listdir(data_dir) if f.endswith('.fits.gz')]

# Use tqdm to wrap the file list for progress bar
for fname in tqdm(fits_files, desc="Processing FITS files"):
    base = os.path.splitext(os.path.splitext(fname)[0])[0]
    fits_path = os.path.join(data_dir, fname)
    hdu_list = fits.open(fits_path)

    # Check for bar mask (hdu_list[4])
    bar_mask_valid = (
        len(hdu_list) >= 5 and
        hdu_list[4].data is not None and
        np.any(np.isfinite(hdu_list[4].data)) and
        np.any(hdu_list[4].data != 0) and
        not np.all(np.isnan(hdu_list[4].data))
    )
    if not bar_mask_valid:
        print(f'Skipped {fname}: No valid bar mask present.')
        hdu_list.close()
        continue

    # Create a subfolder for this dataset
    subfolder = os.path.join(output_dir, base)
    os.makedirs(subfolder, exist_ok=True)

    # Save galaxy image
    galaxy_data = hdu_list[0].data
    galaxy_img_path = os.path.join(subfolder, 'galaxy.png')
    plt.imsave(galaxy_img_path, galaxy_data, origin='lower')

    # Save bar mask image
    bar_mask_data = hdu_list[4].data
    bar_mask_img_path = os.path.join(subfolder, 'bar_mask.png')
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
    superimposed_img_path = os.path.join(subfolder, 'superimposed.png')
    Image.fromarray(superimposed.astype(np.uint8)).save(superimposed_img_path)

    hdu_list.close()
    print(f'Saved: {galaxy_img_path}, {bar_mask_img_path}, {superimposed_img_path}')

print('Batch processing complete.')
