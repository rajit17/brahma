# /data4/rajit/download_manga.py

import os
import pandas as pd
from astropy.io import fits
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u  # <-- Add this import
from tqdm import tqdm  # Add tqdm for progress bars

# Load the attached table (assuming it's a CSV file with commas)
df = pd.read_csv('/data4/rajit/Tobias Géron compare/Table1.csv', sep=',')
print("Columns in CSV:", df.columns.tolist())  # Debug: print columns

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Function to download MaNGA data cubes
def download_manga_cube(plateifu, outdir):
    # MaNGA DRP cube URL pattern (MPL-11/DR17)
    url = f"https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/{plateifu.split('-')[0]}/stack/manga-{plateifu}-LOGCUBE.fits.gz"
    outpath = os.path.join(outdir, f"{plateifu}-LOGCUBE.fits.gz")
    if not os.path.exists(outpath):
        os.system(f"wget -q -O {outpath} {url}")
    return outpath

def download_manga_band_fits(plateifu, band, outdir):
    # MaNGA DRP image URL pattern for single band (MPL-11/DR17)
    url = f"https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/{plateifu.split('-')[0]}/stack/manga-{plateifu}-{band}.fits.gz"
    outpath = os.path.join(outdir, f"{plateifu}-{band}.fits.gz")
    if not os.path.exists(outpath):
        os.system(f"wget -q -O {outpath} {url}")
    return outpath

def get_nearest_plateifu(ra, dec):
    pos = coords.SkyCoord(ra, dec, unit="deg")
    xid = SDSS.query_region(pos, spectro=True, radius=2 * u.arcsec, data_release=17)
    if xid is not None and 'plateifu' in xid.colnames:
        return xid['plateifu'][0]
    return None

# Output directory for cubes
outdir = '/data4/rajit/Tobias Géron compare/manga_cubes_fits'
os.makedirs(outdir, exist_ok=True)

# Prepare output DataFrame
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying by RA/DEC"):
    ra = row['RA']
    dec = row['DEC']
    plateifu = get_nearest_plateifu(ra, dec)
    if plateifu is None:
        continue
    fits_path = download_manga_cube(plateifu, outdir)
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            pixscal = hdr.get('PIXSCALE', None)
            size_px = hdr.get('NAXIS1', None)
    except Exception as e:
        pixscal = None
        size_px = None
    results.append({
        'PLATEIFU': plateifu,
        'pixscal': pixscal,
        'size_px': size_px,
        'RA': ra,
        'DEC': dec
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/data4/rajit/Tobias Géron compare/manga_cubes_fits', index=False)

# Filter for required values
filtered = results_df[(results_df['pixscal'] == 0.099) & (results_df['size_px'] == 525)]

bands = ['g', 'r', 'z']
for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Downloading cubes and grz bands"):
    plateifu = row['PLATEIFU']
    # Download cube (again, but only for filtered galaxies)
    download_manga_cube(plateifu, outdir)
    # Download bands
    for band in bands:
        download_manga_band_fits(plateifu, band, outdir)