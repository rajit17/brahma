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
    layer = 'ls-dr10'
    pixscale = 0.099
    size_px = 525
    url = (f'https://www.legacysurvey.org/viewer/cutout.fits'
           f'?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size_px}&bands={bands}')
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(outfn, 'wb') as f:
        for chunk in r.iter_content(1024*32):
            f.write(chunk)

# Update paths for local environment
csv_path = '/data/rajit/Hoyle_bar_compare/hoyle_barlengths.csv'
out_dir = '/data/rajit/Hoyle_bar_compare/legacy_fits'
os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(csv_path)

# Update these column names as per your CSV
# Example: 'RA', 'DEC', 'objid' (change if needed)
ra_col = 'RA' if 'RA' in df.columns else 'ra'
dec_col = 'DEC' if 'DEC' in df.columns else 'dec'
id_col = 'objid' if 'objid' in df.columns else None

# Prepare list of files to download (avoid duplicates)
download_tasks = []
seen_outfns = set()
for idx, row in df.iterrows():
    ra = row[ra_col]
    dec = row[dec_col]
    obj_id = row[id_col] if id_col else f"obj_{idx}"
    outfn = os.path.join(out_dir, f"{obj_id}_zoomed_ra{ra}_dec{dec}.fits")
    if not os.path.exists(outfn) and outfn not in seen_outfns:
        download_tasks.append((ra, dec, outfn, obj_id))
        seen_outfns.add(outfn)

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
with ThreadPoolExecutor(max_workers=50) as executor:  # Increased to 50 workers
    futures = [executor.submit(download_task, task) for task in download_tasks]
    for future in as_completed(futures):
        if future.result():
            downloaded += 1
        show_progress(downloaded, total)
print()
import pandas as pd
import requests

# Read your CSV
df = pd.read_csv('/data/rajit/Hoyle_bar_compare/hoyle_barlengths.csv')

# Example columns: 'ra', 'dec', 'objid'
bands = 'grz'
size_px = 525
pixscale = 0.099

def download_sdss_image(ra, dec, band, filename):
    url = (
        f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale={pixscale}&width={size_px}&height={size_px}&opt={band}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed for {filename}")

# SDSS image output directory
sdss_out_dir = '/data/rajit/Hoyle_bar_compare/sdss_images'
os.makedirs(sdss_out_dir, exist_ok=True)

for idx, row in df.iterrows():
    ra = row['ra']  # update with your column name
    dec = row['dec']  # update with your column name
    for band in bands:
        filename = os.path.join(sdss_out_dir, f"galaxy_{idx}_{band}.jpg")
        download_sdss_image(ra, dec, band, filename)