#!/usr/bin/env python3
"""
All-in-one FITS re-download and repair script.

What it does:
 1. Scans a source folder (or reads an errors-list file) for corrupted/invalid .fits files.
 2. Re-downloads each cutout from the Legacy Survey viewer using RA/DEC parsed from filenames.
 3. Saves fresh files into a NEW output folder (default: <src>_redownloaded) to keep originals safe.
 4. Validates downloads with astropy, retries with exponential backoff, writes atomic .part files.
 5. Produces `redownload_summary.csv` (success/fail messages) and `errors_retry.txt` (files that failed).

Usage:
  python3 0_redownload.py
  python3 0_redownload.py --src /data4/rajit/DATA_LEGACY_DIR --out /data4/rajit/DATA_LEGACY_DIR_redownloaded --workers 12
  python3 0_redownload.py --errors-list /data4/rajit/errors.txt --out /data4/rajit/DATA_LEGACY_DIR_redownloaded

Defaults are set to match your environment seen in the conversation, but you can override them.
"""
from __future__ import annotations
import argparse
import os
import re
import time
import csv
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits

# ------------------------
# Defaults & config
# ------------------------
DEFAULT_SRC = "/data4/rajit/DATA_LEGACY_DIR"
DEFAULT_OUT_SUFFIX = "_redownloaded"
DEFAULT_LAYER = "ls-dr10"
DEFAULT_PXSCALE = 0.099
DEFAULT_SIZE_PX = 525
DEFAULT_BANDS = "grz"
DEFAULT_WORKERS = 12
MIN_FILESIZE_BYTES = 5_000        # treat smaller files as invalid
RETRIES = 3
RETRY_BACKOFF_BASE = 3            # seconds
CHUNK_SIZE = 32 * 1024
USER_AGENT = "redownload_legacy_fits/1.0"

# Example filename pattern in your listing:
# 1-112213_zoomed_ra341.87308_dec-8.037169.fits
FILENAME_RA_DEC_RE = re.compile(
    r"_ra(?P<ra>-?\d+(?:\.\d+)?)_dec(?P<dec>-?\d+(?:\.\d+)?)(?:\.fits|\.FITS)$"
)

# ------------------------
# Utilities
# ------------------------
def parse_ra_dec_from_filename(basename: str):
    m = FILENAME_RA_DEC_RE.search(basename)
    if not m:
        return None, None
    try:
        return float(m.group("ra")), float(m.group("dec"))
    except Exception:
        return None, None


def is_fits_valid(path: str, min_bytes: int = MIN_FILESIZE_BYTES) -> bool:
    """Basic validity: exists, size, and astropy can open it."""
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < min_bytes:
            return False
        # Try opening with astropy
        with fits.open(path, memmap=False) as hdul:
            return len(hdul) >= 1
    except Exception:
        return False


def build_cutout_url(ra: float, dec: float, bands: str = DEFAULT_BANDS,
                     layer: str = DEFAULT_LAYER, pixscale: float = DEFAULT_PXSCALE,
                     size_px: int = DEFAULT_SIZE_PX) -> str:
    return (
        f"https://www.legacysurvey.org/viewer/cutout.fits"
        f"?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size_px}&bands={bands}"
    )


def download_stream(session: requests.Session, url: str, tmp_outfn: str, timeout: int = 60):
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_outfn, "wb") as fh:
            for chunk in r.iter_content(CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)


def atomic_move(tmp_path: str, final_path: str):
    os.replace(tmp_path, final_path)  # atomic on same fs


# ------------------------
# Worker
# ------------------------
def worker(task: dict):
    """
    task keys:
      - fname (basename)
      - ra, dec
      - outdir (str)
      - src_path (original path) [optional]
      - bands, layer, pixscale, size_px (optional)
    returns (task, success, message)
    """
    fname = task["fname"]
    ra = task["ra"]
    dec = task["dec"]
    outdir = task["outdir"]
    bands = task.get("bands", DEFAULT_BANDS)
    layer = task.get("layer", DEFAULT_LAYER)
    pixscale = task.get("pixscale", DEFAULT_PXSCALE)
    size_px = task.get("size_px", DEFAULT_SIZE_PX)
    final_path = os.path.join(outdir, fname)
    tmp_path = final_path + ".part"

    url = build_cutout_url(ra, dec, bands=bands, layer=layer, pixscale=pixscale, size_px=size_px)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            download_stream(session, url, tmp_path)
            # quick size sanity check
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < MIN_FILESIZE_BYTES:
                raise IOError(f"downloaded file too small ({os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 'missing'})")
            # validate FITS
            if not is_fits_valid(tmp_path):
                raise IOError("astropy validation failed for downloaded file")
            # ensure outdir exists and move atomically
            os.makedirs(outdir, exist_ok=True)
            atomic_move(tmp_path, final_path)
            return (task, True, f"OK (attempt {attempt})")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # cleanup tmp
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt < RETRIES:
                wait = RETRY_BACKOFF_BASE * attempt
                time.sleep(wait)
    return (task, False, f"FAILED after {RETRIES} attempts: {last_err}")


# ------------------------
# Main CLI
# ------------------------
def main():
    p = argparse.ArgumentParser(description="Scan for bad FITS and re-download cutouts into a new folder.")
    p.add_argument("--src", help="Source folder with .fits files", default=DEFAULT_SRC)
    p.add_argument("--errors-list", help="Text file with one full path per line (if you already have error list)")
    p.add_argument("--out", help="Output folder (default: <src>%s)" % DEFAULT_OUT_SUFFIX)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Thread workers")
    p.add_argument("--bands", default=DEFAULT_BANDS, help="Bands parameter for cutout (default: grz)")
    p.add_argument("--replace-original", action="store_true", help="If set, overwrite original files on successful redownload (USE CAREFULLY)")
    args = p.parse_args()

    src = args.src
    out = args.out or (src.rstrip("/") + DEFAULT_OUT_SUFFIX)
    workers = max(1, int(args.workers))
    bands = args.bands

    # Build list of bad files
    candidates = []
    if args.errors_list:
        if not os.path.exists(args.errors_list):
            print(f"errors-list file not found: {args.errors_list}")
            return
        with open(args.errors_list, "r") as fh:
            for line in fh:
                pth = line.strip()
                if not pth:
                    continue
                base = os.path.basename(pth)
                ra, dec = parse_ra_dec_from_filename(base)
                if ra is None:
                    print(f"Skipping (can't parse RA/DEC): {pth}")
                    continue
                candidates.append({
                    "fname": base, "ra": ra, "dec": dec, "outdir": out, "src_path": pth, "bands": bands
                })
    else:
        if not os.path.isdir(src):
            print(f"Source folder does not exist: {src}")
            return
        # scan src for .fits and select invalid ones
        all_files = sorted(os.listdir(src))
        for fname in all_files:
            if not fname.lower().endswith(".fits"):
                continue
            full = os.path.join(src, fname)
            if not is_fits_valid(full):
                ra, dec = parse_ra_dec_from_filename(fname)
                if ra is None:
                    print(f"Warning: cannot parse RA/DEC from filename (skipping): {fname}")
                    continue
                candidates.append({
                    "fname": fname, "ra": ra, "dec": dec, "outdir": out, "src_path": full, "bands": bands
                })

    if not candidates:
        print("No corrupted/invalid FITS found to redownload. Exiting.")
        return

    print(f"Found {len(candidates)} files to re-download. Output folder: {out}")
    os.makedirs(out, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, c): c for c in candidates}
        for fut in as_completed(futures):
            task, success, msg = fut.result()
            results.append({
                "fname": task["fname"],
                "ra": task["ra"],
                "dec": task["dec"],
                "src_path": task.get("src_path", ""),
                "out_path": os.path.join(out, task["fname"]),
                "success": success,
                "message": msg
            })
            status = "OK" if success else "FAIL"
            print(f"[{status}] {task['fname']} -> {msg}")
            # optionally replace original
            if args.replace_original and success and task.get("src_path"):
                try:
                    shutil.copy2(os.path.join(out, task["fname"]), task["src_path"])
                    print(f"  -> original replaced: {task['src_path']}")
                except Exception as e:
                    print(f"  -> warning: couldn't replace original: {e}")

    # write summary CSV and failed list
    summary_csv = os.path.join(out, "redownload_summary.csv")
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fname", "ra", "dec", "src_path", "out_path", "success", "message"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    failed = [r for r in results if not r["success"]]
    if failed:
        errfile = os.path.join(out, "errors_retry.txt")
        with open(errfile, "w") as fh:
            for r in failed:
                srcp = r["src_path"] or r["out_path"]
                fh.write(srcp + "\n")
        print(f"\n{len(failed)} files failed. See {errfile} and {summary_csv}")
    else:
        print(f"\nAll {len(results)} files re-downloaded successfully. Summary: {summary_csv}")

if __name__ == "__main__":
    main()
