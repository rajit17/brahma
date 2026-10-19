import requests, os
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_legacy_fits(ra, dec, bands='grz', outfn='cutout.fits'):
    """
    Downloads a FITS file cutout from the Legacy Survey viewer using fixed parameters.

    Args:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        bands (str): The photometric bands to include.
        outfn (str): The output filename for the FITS file.
    """
    # Fixed parameters as per the user's request
    layer = 'ls-dr10'
    pixscale = 0.099
    size_px = 525

    # Construct the URL with the fixed parameters.
    # The URL directly requests a 525x525 pixel image with a 0.099 arcsec/pixel scale.
    url = (f'https://www.legacysurvey.org/viewer/cutout.fits'
           f'?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size_px}&bands={bands}')
    
    # Send the request and save the FITS file.
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status() # Raise an exception for bad status codes
    with open(outfn, 'wb') as f:
        for chunk in r.iter_content(1024*32):
            f.write(chunk)

# Update paths for server environment
csv_path = '/data/rajit/gz3d_metadata.csv'
out_dir = '/data/rajit/DATA_LEGACY_DIR'
os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(csv_path)

# Filter for galaxies with at least 5 bar votes (remove .head(10) to process all)
df = df[df['GZ_bar_votes'] >= 5]

# Prepare list of files to download
download_tasks = []
for idx, row in df.iterrows():
    ra = row['ra']
    dec = row['dec']
    obj_id = row['MANGAID'] if 'MANGAID' in row else f"obj_{idx}"
    outfn = os.path.join(out_dir, f"{obj_id}_zoomed_ra{ra}_dec{dec}.fits")
    if not os.path.exists(outfn):
        download_tasks.append((ra, dec, outfn, obj_id))

total = len(download_tasks)

def show_progress(downloaded, total):
    bar_len = 40
    filled_len = int(bar_len * downloaded // total) if total else 0
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    left = total - downloaded
    sys.stdout.write(
        f"\r{left} galaxies left, {downloaded} downloaded [{bar}]"
    )
    sys.stdout.flush()

def download_task(args):
    ra, dec, outfn, obj_id = args
    try:
        download_legacy_fits(ra=ra, dec=dec, outfn=outfn)
        return True
    except Exception:
        return False

downloaded = 0
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(download_task, task) for task in download_tasks]
    for future in as_completed(futures):
        if future.result():
            downloaded += 1
        show_progress(downloaded, total)
print()  # Move to next line after progress bar