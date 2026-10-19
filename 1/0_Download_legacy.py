import requests
import os
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_legacy_fits(ra, dec, bands='grz', outfn='cutout.fits'):
    """
    Downloads a FITS file cutout from the Legacy Survey viewer.
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

# --- Path Configurations ---
csv_path = '/data4/rajit/gz3d_metadata.csv'
# Replace 'plateifu_list.xlsx' with the actual name of your excel file
excel_path = '/data4/rajit/plateifu_mapping.xlsx' 
out_dir = '/data4/rajit/DATA_LEGACY_DIR'
os.makedirs(out_dir, exist_ok=True)

# --- Data Loading and Merging ---
print("Loading data...")
df_meta = pd.read_csv(csv_path)
df_excel = pd.read_excel(excel_path)

# Merge the dataframes on MANGAID to get the plateifu for each galaxy
# We use an 'inner' join to ensure we only download if we have both coordinates and the ID
df = pd.merge(df_meta, df_excel[['MANGAID', 'plateifu']], on='MANGAID', how='inner')

# Filter for galaxies with at least 5 bar votes
df = df[df['GZ_bar_votes'] >= 5]

# --- Task Preparation ---
download_tasks = []
for idx, row in df.iterrows():
    ra = row['ra']
    dec = row['dec']
    p_ifu = row['plateifu']
    
    # Using plateifu in the filename for better organization
    outfn = os.path.join(out_dir, f"{p_ifu}_ra{ra}_dec{dec}.fits")
    
    if not os.path.exists(outfn):
        download_tasks.append((ra, dec, outfn, p_ifu))

total = len(download_tasks)
print(f"Total files to download: {total}")

# --- Execution Logic ---
def show_progress(downloaded, total):
    bar_len = 40
    filled_len = int(bar_len * downloaded // total) if total else 0
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    left = total - downloaded
    sys.stdout.write(
        f"\r{left} left, {downloaded} done [{bar}]"
    )
    sys.stdout.flush()

def download_task(args):
    ra, dec, outfn, p_ifu = args
    try:
        download_legacy_fits(ra=ra, dec=dec, outfn=outfn)
        return True
    except Exception as e:
        # Logging errors can be helpful for debugging failed downloads
        # print(f"Error downloading {p_ifu}: {e}")
        return False

downloaded = 0
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(download_task, task) for task in download_tasks]
    for future in as_completed(futures):
        if future.result():
            downloaded += 1
        show_progress(downloaded, total)

print("\nProcess complete.")