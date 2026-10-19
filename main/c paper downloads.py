#!/usr/bin/env python3
"""
download_na10_sami_hsc.py

- Downloads NA10 (Nair & Abraham 2010) catalogue from VizieR (astroquery)
  and saves a CSV.
- Downloads FITS cutouts for NA10 positions using the DESI Legacy Survey cutout service.
- Optionally downloads SAMI DR3 catalogue positions (tries DataCentral REST).
- Optionally fetches HSC cutouts using HSC `downloadCutout.py` helper (if available).
"""

import os
import sys
import math
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Optional: astroquery to fetch NA10
try:
    from astroquery.vizier import Vizier
    from astropy import units as u
    from astropy.table import Table
except Exception as e:
    print("astroquery / astropy not installed. Install with: pip install astroquery astropy")
    raise

# ---------- User config ----------
OUT_ROOT = "/data/rajit/astro_downloads"   # change as needed
NA10_DIR = os.path.join(OUT_ROOT, "NA10")
SAMI_DIR = os.path.join(OUT_ROOT, "SAMI_DR3")
HSC_DIR = os.path.join(OUT_ROOT, "HSC_cutouts")

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(NA10_DIR, exist_ok=True)
os.makedirs(SAMI_DIR, exist_ok=True)
os.makedirs(HSC_DIR, exist_ok=True)

# Legacy Survey cutout parameters (used as default, robust and public)
LEGACY_LAYER = 'ls-dr10'    # change if needed
LEGACY_PIXSCALE = 0.099     # arcsec/pixel
LEGACY_SIZE_PX = 525        # pixels (square)
LEGACY_BANDS = 'grz'

# Concurrency
MAX_WORKERS = 24

# ---------- Helper functions ----------
def legacy_cutout_fits_url(ra, dec, layer=LEGACY_LAYER, pixscale=LEGACY_PIXSCALE, size_px=LEGACY_SIZE_PX, bands=LEGACY_BANDS):
    """Return a DESI Legacy cutout URL that returns a FITS file."""
    return (f'https://www.legacysurvey.org/viewer/cutout.fits'
            f'?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size_px}&bands={bands}')

def download_url_to_file(url, outfn, timeout=60):
    """Download with streaming and basic retry."""
    tmp = outfn + ".part"
    tries = 3
    for attempt in range(tries):
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(1024*32):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, outfn)
            return True
        except Exception as e:
            print(f"Download failed (attempt {attempt+1}/{tries}) for {url}: {e}")
            time.sleep(2 + attempt*2)
    if os.path.exists(tmp):
        os.remove(tmp)
    return False

# ---------- NA10: fetch catalog from VizieR ----------
def fetch_na10_vizier(save_csv=True, vizier_rows=50000):
    """
    Download NA10 (Nair & Abraham 2010) from VizieR.
    VizieR identifier: J/ApJS/186/427  (II/267)
    """
    print("Querying VizieR for NA10 (Nair & Abraham 2010)...")
    Vizier.ROW_LIMIT = vizier_rows
    # two ways: query by catalog ID or bibliographic code
    try:
        cat = Vizier.get_catalogs("II/267")   # common VizieR ID for NA10
    except Exception:
        # fallback to bibliographic code
        cat = Vizier.get_catalogs("J/ApJS/186/427")
    if len(cat) == 0:
        raise RuntimeError("No table returned from Vizier for NA10. Check network or table ID.")
    t = cat[0]
    df = t.to_pandas()
    print(f"NA10 rows downloaded: {len(df)}. Columns: {list(df.columns)[:10]} ...")
    # save CSV
    if save_csv:
        out_csv = os.path.join(NA10_DIR, "NA10_vizier.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved NA10 catalogue to {out_csv}")
    return df

def detect_ra_dec_columns(df):
    """Return names of RA and DEC columns in the DataFrame, robustly."""
    candidates_ra = ['RAJ2000', 'RAJ2000d', 'RAICRS', 'RA', 'ra', 'RAJ2000.0', 'RAJ2000_']
    candidates_dec = ['DEJ2000', 'DEJ2000d', 'DEC', 'dec', 'DEJ2000.0','DEJ2000_']
    ra_col = None
    dec_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    # try exact candidates
    for c in df.columns:
        low = c.lower()
        if low in ['raj2000','raj2000.0','raj2000d','ra','ra_icrs','ra_j2000']:
            ra_col = c
        if low in ['dej2000','dec','dec_j2000','dec_j2000.0','dej2000.0']:
            dec_col = c
    # try general fallback: column names that contain 'ra' and 'dec'
    if ra_col is None:
        for c in df.columns:
            if 'ra' in c.lower() and 'err' not in c.lower():
                ra_col = c
                break
    if dec_col is None:
        for c in df.columns:
            if 'dec' in c.lower() and 'err' not in c.lower():
                dec_col = c
                break
    return ra_col, dec_col

# ---------- Download cutouts for a table ----------
def prepare_download_tasks_from_df(df, ra_col, dec_col, out_dir, id_col=None, filter_mask=None, prefix="obj"):
    """
    Build a list of (ra, dec, outfn, obj_id) tasks for download.
    df: pandas.DataFrame
    filter_mask: optional boolean mask or lambda to select rows
    """
    if filter_mask is not None:
        df = df.loc[filter_mask].reset_index(drop=True)
    tasks = []
    for idx, row in df.iterrows():
        try:
            ra = float(row[ra_col])
            dec = float(row[dec_col])
        except Exception:
            continue
        obj_id = None
        if id_col and id_col in df.columns:
            obj_id = str(row[id_col])
        else:
            # try some common id fields
            for cand in ['objID','obj_id','ID','name','ObjID','plateifu','MANGAID']:
                if cand in df.columns:
                    obj_id = str(row[cand]); break
        if obj_id is None:
            obj_id = f"{prefix}_{idx}"
        # sanitize obj_id
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in obj_id)
        outfn = os.path.join(out_dir, f"{safe_id}_ra{ra:.6f}_dec{dec:.6f}.fits")
        tasks.append((ra, dec, outfn, safe_id))
    return tasks

def download_tasks_legacy(tasks, max_workers=MAX_WORKERS):
    total = len(tasks)
    print(f"Starting download of {total} cutouts (Legacy Survey FITS)...")
    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_task_legacy, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=total):
            ok, t = fut.result()
            if ok:
                downloaded += 1
            else:
                failed += 1
    print(f"Done. downloaded={downloaded}, failed={failed}")
    return downloaded, failed

def download_task_legacy(task):
    ra, dec, outfn, obj_id = task
    if os.path.exists(outfn):
        return True, task
    url = legacy_cutout_fits_url(ra, dec)
    ok = download_url_to_file(url, outfn)
    return ok, task

# ---------- Simple SAMI DR3 fetch helper ----------
def try_fetch_sami_catalog_datacentral(save_csv=True):
    """
    Try to download SAMI DR3 master catalog from Data Central using a known endpoint.
    If this fails, instruct user to download manually and place CSV in SAMI_DIR.
    """
    print("Attempting to fetch SAMI DR3 catalogue from Data Central (best-effort)...")
    # There is no single guaranteed URL; Data Central usually exposes a REST table download.
    # We'll try a common location used by authors (may change). This is a 'best-effort' step.
    possible_urls = [
        "https://data.researchdata.net.au/sami/dr3/complete_catalogue.csv",  # hypothetical
        "https://datacentral.org.au/download/sami-dr3.csv"                  # hypothetical
    ]
    for url in possible_urls:
        try:
            r = requests.get(url, timeout=20)
            if r.ok and len(r.content) > 100:
                outfn = os.path.join(SAMI_DIR, "sami_dr3_catalog.csv")
                with open(outfn, "wb") as f:
                    f.write(r.content)
                print(f"Saved SAMI catalog to {outfn}")
                return pd.read_csv(outfn)
        except Exception:
            continue
    # fallback: return None and instruct user
    print("Could not automatically download SAMI DR3 catalogue. Please download it from https://datacentral.org.au/ or https://sami-survey.org/data and place a CSV at:")
    print(f"    {os.path.join(SAMI_DIR, 'sami_dr3_catalog.csv')}")
    return None

# ---------- HSC cutouts using HSC helper script (optional) ----------
def call_hsc_downloadcutout(ra, dec, outfn_prefix, size_arcsec=60, bands="g,r,i"):
    """
    This function expects the HSC downloadCutout.py helper (from https://hsc-release... repo)
    to be installed on the server and callable as `downloadCutout.py`.
    It's optional. If not present, this function returns False.
    """
    # try to find script in path
    import shutil, subprocess
    exe = shutil.which("downloadCutout.py")  # user can clone the HSC repo and put script in PATH
    if exe is None:
        return False
    outfn = f"{outfn_prefix}_hsc.fits"
    cmd = [exe, "--ra", str(ra), "--dec", str(dec), "--out", outfn, "--size", str(size_arcsec), "--bands", bands]
    try:
        subprocess.check_call(cmd, timeout=120)
        return True
    except Exception as e:
        print("HSC cutout helper failed:", e)
        return False

# ---------- MAIN orchestration ----------
def main_download_na10_legacy():
    # 1) get NA10 table
    df_na10 = fetch_na10_vizier()
    ra_col, dec_col = detect_ra_dec_columns(df_na10)
    if ra_col is None or dec_col is None:
        raise RuntimeError("Cannot detect RA/DEC columns in NA10 table. Columns: " + ", ".join(df_na10.columns))
    print(f"Detected RA column = {ra_col}, DEC column = {dec_col}")

    # 2) try to detect a bar column (common names)
    bar_col_candidates = ['Bar','bar','B','bars','Barred','bar_flag']
    chosen_bar_col = None
    for c in bar_col_candidates:
        if c in df_na10.columns:
            chosen_bar_col = c; break
    if chosen_bar_col:
        print(f"Found candidate column for bar presence: {chosen_bar_col}")
        df_subset = df_na10[df_na10[chosen_bar_col].notnull()].copy()
        # for many NA10 variants the bar column may use integers (e.g., 0/1 or flags). We'll not coerce now.
    else:
        print("No explicit 'bar' column detected in NA10 table. Using full table (you can filter manually).")
        df_subset = df_na10

    # 3) prepare tasks and download using Legacy cutout
    tasks = prepare_download_tasks_from_df(df_subset, ra_col, dec_col, NA10_DIR, id_col=None, prefix="NA10")
    print(f"{len(tasks)} NA10 cutout tasks prepared (check NA10_DIR = {NA10_DIR}).")
    if len(tasks) == 0:
        print("No tasks to download. Exiting.")
        return
    download_tasks_legacy(tasks, max_workers=MAX_WORKERS)

def main_download_sami_hsc():
    # try to auto-fetch SAMI DR3
    df_sami = try_fetch_sami_catalog_datacentral()
    if df_sami is None:
        print("No SAMI table found automatically. If you have a SAMI CSV with RA/DEC, place it at:")
        print(f"   {os.path.join(SAMI_DIR, 'sami_dr3_catalog.csv')}")
        return
    ra_col, dec_col = detect_ra_dec_columns(df_sami)
    if ra_col is None or dec_col is None:
        print("Could not detect RA/DEC columns in SAMI table. Columns are:", df_sami.columns)
        return
    tasks = prepare_download_tasks_from_df(df_sami, ra_col, dec_col, SAMI_DIR, id_col=None, prefix="SAMI")
    print(f"Prepared {len(tasks)} SAMI cutouts. Downloading via Legacy Survey (fallback).")
    download_tasks_legacy(tasks, max_workers=MAX_WORKERS)
    # Try optional HSC via helper (if available)
    print("Attempting optional HSC cutouts using local HSC helper (if installed)...")
    for ra, dec, outfn, obj in tasks[:50]:   # sample / demo limit
        prefix = os.path.join(HSC_DIR, obj)
        ok = call_hsc_downloadcutout(ra, dec, prefix, size_arcsec=90)
        if ok:
            print(f"HSC saved for {obj}")

if __name__ == "__main__":
    # Example usage: run the NA10 downloader first
    main_download_na10_legacy()
    # Try to auto-download SAMI (best-effort)
    main_download_sami_hsc()
    print("All done. Check the directories for FITS files.")
