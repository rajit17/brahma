import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

data_dir = '/data/rajit/DATA_GZ3D_DIR/'
fits_files = [f for f in os.listdir(data_dir) if f.endswith('.fits.gz')]

def check_mask(fname):
    fits_path = os.path.join(data_dir, fname)
    try:
        hdu_list = fits.open(fits_path)
    except Exception:
        return 0
    result = 0
    if len(hdu_list) >= 5:
        mask = hdu_list[4].data
        if np.count_nonzero(mask) >= 5:
            result = 1
    hdu_list.close()
    return result

valid_count = 0
with ThreadPoolExecutor(max_workers=500) as executor:
    results = list(tqdm(executor.map(check_mask, fits_files), total=len(fits_files), desc="Checking FITS files"))
    valid_count = sum(results)

print(f"Number of valid datasets: {valid_count}")
