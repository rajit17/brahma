import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from matplotlib.ticker import MaxNLocator
import os
from astropy.table import Table
import requests  # For downloading NSA FITS
from tqdm import tqdm  # Progress bar
import sys

# Check for required CSV files before loading
if not os.path.exists('/data4/rajit/ Recheck/bar_measurements.csv'):
    print("ERROR: '/data4/rajit/ Recheck/bar_measurements.csv' not found.")
    sys.exit(1)
if not os.path.exists('/data4/rajit/ Recheck/hoyle_barlengths.csv'):
    print("ERROR: '/data4/rajit/ Recheck/hoyle_barlengths.csv' not found.")
    sys.exit(1)

# Load the user's bar measurements CSV (pixel lengths)
print("Loading user bar measurements...")
user_measurements = pd.read_csv('/data4/rajit/ Recheck/bar_measurements.csv')

# Load Hoyle's bar lengths CSV
print("Loading Hoyle bar measurements...")
hoyle_measurements = pd.read_csv('/data4/rajit/ Recheck/hoyle_barlengths.csv')

# Parse objid, RA, and Dec from filenames in user_measurements for merging
print("Parsing filenames for objid, RA, Dec...")
def parse_info(filename):
    parts = filename.split('_')
    objid_str = parts[0]  # e.g., '5.877229817479825e+17'
    objid = int(float(objid_str))  # Convert scientific notation to full integer objid
    ra_part = next(p for p in parts if p.startswith('ra'))
    dec_part = next(p for p in parts if p.startswith('dec'))
    ra = float(ra_part[2:])
    # Remove '.fits' from dec_part if present
    dec_str = dec_part[3:]
    if dec_str.endswith('.fits'):
        dec_str = dec_str[:-5]
    dec = float(dec_str)
    return objid, ra, dec

# Use tqdm for progress bar on parsing
user_measurements[['objid', 'ra_user', 'dec_user']] = list(
    tqdm(user_measurements['filename'].apply(parse_info), desc="Parsing filenames")
)
user_measurements[['objid', 'ra_user', 'dec_user']] = user_measurements[['objid', 'ra_user', 'dec_user']].apply(pd.Series)

# Merge user measurements with Hoyle on objid (exact match)
print("Merging user and Hoyle measurements...")
matched = pd.merge(user_measurements, hoyle_measurements, on='objid', how='inner')

# Use RA/Dec from Hoyle for consistency, round to 6 decimal places for merging with NSA
matched['ra_rounded'] = matched['ra'].round(6)
matched['dec_rounded'] = matched['dec'].round(6)

# Set the pixel scale from the download parameters and FITS headers (0.099 arcsec/pixel)
matched['est_dr5_pixscale'] = 0.099

# Compute arcsec from pixels (using pixel_length as the measured bar length in pixels)
print("Converting pixel lengths to arcsec...")
matched['bar_length_arcsec'] = matched['pixel_length'] * matched['est_dr5_pixscale']

# Load NSA for redshift: Download FITS if not present, then extract key columns
nsa_fits_path = 'nsa_v1_0_1.fits'
if not os.path.exists(nsa_fits_path):
    print("Downloading NSA v1.0.1 FITS catalog...")
    url = 'https://data.sdss.org/sas/dr17/sdss/atlas/v1/nsa_v1_0_1.fits'
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    with open(nsa_fits_path, 'wb') as f, tqdm(
        desc="Downloading NSA FITS",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))
    print("Download complete.")

print("Reading NSA FITS catalog...")
nsa_table = Table.read(nsa_fits_path)
# Select relevant columns: IAUNAME, RA, DEC, Z (redshift)
nsa = nsa_table['IAUNAME', 'RA', 'DEC', 'Z'].to_pandas()
nsa.columns = ['iauname', 'ra_nsa', 'dec_nsa', 'redshift']

# Round NSA RA/DEC for matching
nsa['ra_rounded'] = nsa['ra_nsa'].round(6)
nsa['dec_rounded'] = nsa['dec_nsa'].round(6)

# Merge with NSA on rounded RA/DEC
print("Merging with NSA catalog...")
matched = pd.merge(matched, nsa[['ra_rounded', 'dec_rounded', 'redshift']], on=['ra_rounded', 'dec_rounded'], how='inner')

# Compute angular diameter distance
print("Computing angular diameter distances...")
matched['angular_diameter_distance'] = [
    cosmo.angular_diameter_distance(z) for z in tqdm(matched['redshift'], desc="Angular diameter distance")
]

# Convert to kpc (following the original logic: arcsec * d_A / 117 for h^{-1} kpc units)
print("Converting arcsec to kpc...")
matched['bar_length_kpc'] = [
    arcsec * d_A / 117 for arcsec, d_A in tqdm(zip(matched['bar_length_arcsec'], matched['angular_diameter_distance']),
                                               total=len(matched), desc="Arcsec to kpc")
]
matched['bar_length_kpc'] = np.array(matched['bar_length_kpc']).astype(float)

# Compute correlation and mean absolute error (using length_avg from Hoyle as reference)
print("Computing statistics...")
r_corr = pd.Series(matched['bar_length_kpc']).corr(matched['length_avg'])
mean_abs_error = np.abs(matched['bar_length_kpc'] - matched['length_avg']).mean()

print(f"Number of matched galaxies: {len(matched)}")
print(f"Correlation (r): {r_corr:.2f}")
print(f"Mean Absolute Error (L1): {mean_abs_error:.1f} kpc")

# Plot comparison
print("Plotting comparison...")
sns.set_context('paper')
fig, ax = plt.subplots(figsize=(3, 3))
sns.regplot(x=matched['bar_length_kpc'].astype(float), y=matched['length_avg'].astype(float), 
            marker='+', robust=True, ax=ax)
ax.set_xlabel('Your bar length [h$^{-1}$ kpc]')
ax.set_ylabel('Hoyle (2011) bar length [h$^{-1}$ kpc]')
ax.set_xlim([0., 22.])
ax.set_ylim([0., 22.])
ax.plot([0, 22], [0, 22], color='k', alpha=0.5, linestyle='--')
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax.text(1.1, 20, "$r={:1.2f}$".format(r_corr))
ax.text(1.1, 18, "L1={:1.1f} kpc".format(mean_abs_error))
plt.tight_layout()
plt.savefig('your_hoyle_bar_comparison.pdf')
plt.show()