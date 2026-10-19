# bar_measurement_pipeline_fixed.py
"""
Improved/robust bar measurement pipeline
- Accepts header pixscale (including 0.099) if present
- Robust objid extraction (handles scientific-notation filenames)
- Attempts to get redshift (z) from Hoyle CSV -> SDSS -> NED (if astroquery installed)
- Converts pixels -> arcsec -> kpc (requires z)
- Logs missing redshifts to missing_redshifts.csv
- Saves results to results/bar_measurements.csv
"""
import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pi
from PIL import Image
from astropy.io import fits
import warnings
from astropy.io.fits.verify import VerifyWarning
import scipy.stats
from tqdm import tqdm

warnings.simplefilter('ignore', VerifyWarning)

# Directories (keep your layout)
FITS_DIR = 'Hoyle_bar_compare/legacy_fits'
HOYLE_CSV = 'Hoyle_bar_compare/hoyle_barlengths.csv'
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
BAR_CSV = os.path.join(RESULTS_DIR, 'bar_measurements.csv')
LOG_FILE = os.path.join(RESULTS_DIR, 'log.txt')
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'summary.txt')
MISSING_Z_CSV = os.path.join(RESULTS_DIR, 'missing_redshifts.csv')
PNG_DIR = '/data4/rajit/Hoyle_bar_compare/Hoyle_BEST'
DETECTED_IMG_DIR = os.path.join(RESULTS_DIR, 'detected_hoyle_images')
BAR_COMPARISON_CSV = os.path.join(RESULTS_DIR, 'bar_length_comparison.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DETECTED_IMG_DIR, exist_ok=True)

# Logging
logging.basicConfig(filename=os.path.join(RESULTS_DIR, 'pipeline_debug.log'),
                    level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# YOLO weights - keep your path
YOLO_WEIGHTS = '/data4/rajit/Hoyle_bar_compare/best.pt'
if not os.path.isfile(YOLO_WEIGHTS):
    raise SystemExit(f"ERROR: YOLO weights file '{YOLO_WEIGHTS}' not found. Provide the correct path.")
from ultralytics import YOLO
model = YOLO(YOLO_WEIGHTS)

# Read Hoyle CSV
hoyle_df = pd.read_csv(HOYLE_CSV, dtype=str)  # read as strings for safe objid matching
# Normalize objid to string (strip spaces)
hoyle_df['objid'] = hoyle_df['objid'].astype(str).str.strip()

# Cosmology (astropy if available)
try:
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    use_astropy = True
except Exception:
    use_astropy = False
    logging.warning("astropy not found: install astropy for accurate DA computations.")

c_km_s = 299792.458
def angular_diameter_distance_mpc(z):
    if use_astropy:
        return float(cosmo.angular_diameter_distance(z).value)
    else:
        H0 = 70.0
        return (c_km_s / H0) * z / (1.0 + z)

def arcsec_to_kpc(arcsec, distance_mpc):
    return (arcsec / 206265.0) * (distance_mpc * 1e3)

# Try astroquery for redshift lookup (optional)
try:
    from astroquery.sdss import SDSS
    from astroquery.ned import Ned
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    have_astroquery = True
    logging.info("astroquery available: automatic SDSS/NED lookups enabled.")
except Exception:
    have_astroquery = False
    logging.warning("astroquery not available: SDSS/NED lookups disabled. Fill missing_redshifts.csv manually.")

# Helper: extract objid
def extract_objid_from_filename(filename):
    base = os.path.basename(filename)
    first = base.split('_')[0].replace('.fits','')
    # handle scientific notation -> integer-like
    try:
        if ('e' in first) or ('E' in first):
            val = float(first)
            first = "{:.0f}".format(val)
        else:
            # convert float string '1234.0' -> '1234'
            if re.fullmatch(r'\d+\.\d+', first):
                first = str(int(float(first)))
    except Exception:
        pass
    return str(first).strip()

# Helper: get pixscale and try to obtain distance/z
def get_conversion_info(header, filename, ra_cat=None, dec_cat=None):
    """
    Returns (arcsec_per_pixel (float), distance_mpc (float or None), z (float or None))
    Accept header pixscale values (including e.g. 0.099) if present.
    If no distance found in header or hoyle table, try astroquery (SDSS -> NED).
    """
    arcsec_per_pixel = None
    distance_mpc = None
    z = None

    # Look for pixscale keys
    for key in ['PIXSCALE', 'PIXSCL', 'PIXSCL1', 'PIXSCL2', 'PIXSCL_RA', 'PIXSCL_DEC',
                'CDELT1', 'CDELT2', 'CD1_1', 'CD2_2']:
        if key in header:
            try:
                val = float(header[key])
                # if CDELT/CD usually in degrees/pix -> convert to arcsec
                if key.startswith('CDELT') or key.startswith('CD'):
                    val = abs(val) * 3600.0
                arcsec_per_pixel = abs(val)
                logging.info(f"{filename}: header {key} -> arcsec_per_pixel={arcsec_per_pixel}")
                break
            except Exception:
                continue

    # If no header pixscale, try to take from Hoyle CSV row (some releases include 'pixscale' column)
    if arcsec_per_pixel is None:
        objid = extract_objid_from_filename(filename)
        row = hoyle_df[hoyle_df['objid'] == objid]
        if not row.empty:
            for col in ['pixscale','pixel_scale','pix_scale','pixel_scale_arcsec']:
                if col in row.columns and pd.notna(row.iloc[0].get(col)):
                    try:
                        arcsec_per_pixel = float(row.iloc[0][col])
                        logging.info(f"{filename}: found pixscale in Hoyle CSV column {col}: {arcsec_per_pixel}")
                        break
                    except Exception:
                        pass

    # If still None -> fallback to None (we accept header pixscale being present; user said pixscale exists)
    # Do NOT force a default assumption here (user insisted 0.099 is valid)

    # Try to get distance / redshift from hoyle_df if available
    row = hoyle_df[hoyle_df['objid'] == extract_objid_from_filename(filename)]
    if not row.empty:
        # if distance in Mpc in some column
        for dcol in ['distance_mpc', 'D_MPC', 'distance', 'DISTANCE', 'dist_mpc']:
            if dcol in row.columns and pd.notna(row.iloc[0].get(dcol)):
                try:
                    distance_mpc = float(row.iloc[0][dcol])
                    logging.info(f"{filename}: found distance in Hoyle CSV column {dcol}: {distance_mpc} Mpc")
                    break
                except Exception:
                    pass
        # if redshift present
        for zcol in ['z','redshift']:
            if zcol in row.columns and pd.notna(row.iloc[0].get(zcol)):
                try:
                    z = float(row.iloc[0][zcol])
                    distance_mpc = angular_diameter_distance_mpc(z)
                    logging.info(f"{filename}: found z in Hoyle CSV -> z={z}, DA={distance_mpc} Mpc")
                    break
                except Exception:
                    pass

    # If no distance yet and astroquery available, try SDSS spec then NED
    if distance_mpc is None and have_astroquery:
        # prefer catalog RA/DEC if present (passed from main loop)
        ra = ra_cat; dec = dec_cat
        if ra is None or dec is None:
            # fallback to header keys
            for rakey in ['RA', 'OBJRA', 'CRVAL1', 'RA_OBJ']:
                if rakey in header:
                    try:
                        ra = float(header[rakey]); break
                    except Exception:
                        pass
            for dek in ['DEC', 'OBJDEC', 'CRVAL2', 'DEC_OBJ']:
                if dek in header:
                    try:
                        dec = float(header[dek]); break
                    except Exception:
                        pass
        if ra is not None and dec is not None:
            coord = SkyCoord(ra, dec, unit=u.deg)
            # SDSS spectroscopic
            try:
                sdss_tbl = SDSS.query_region(coord, spectro=True, radius=3*u.arcsec)
                if sdss_tbl is not None and 'z' in sdss_tbl.colnames:
                    zvals = np.array(sdss_tbl['z'], dtype=float)
                    zvals = zvals[(zvals > 0) & (zvals < 1)]
                    if len(zvals) > 0:
                        z = float(zvals[0]); distance_mpc = angular_diameter_distance_mpc(z)
                        logging.info(f"{filename}: SDSS spec z={z}, DA={distance_mpc:.3f} Mpc")
            except Exception as e:
                logging.debug(f"{filename}: SDSS query failed: {e}")
            # If SDSS didn't find z, try NED
            if distance_mpc is None:
                try:
                    ned_tbl = Ned.query_region(coord, radius=5*u.arcsec)
                    if ned_tbl is not None and 'Redshift' in ned_tbl.colnames:
                        for zval in ned_tbl['Redshift']:
                            try:
                                zv = float(zval)
                                if zv > 0 and zv < 1:
                                    z = zv; distance_mpc = angular_diameter_distance_mpc(z)
                                    logging.info(f"{filename}: NED z={z}, DA={distance_mpc:.3f} Mpc")
                                    break
                            except Exception:
                                continue
                except Exception as e:
                    logging.debug(f"{filename}: NED query failed: {e}")

    return arcsec_per_pixel, distance_mpc, z


# Helper: convert xywh-like to pixel units robustly
def xywh_to_pixels(xywh, img_w, img_h):
    """
    xywh can be either normalized (0..1) or in pixels. Decide by checking typical range.
    Return (x_pix, y_pix, w_pix, h_pix)
    """
    x, y, w, h = [float(v) for v in xywh[:4]]
    # If w <= 1.0 assume normalized relative to image width
    if w <= 1.0 and h <= 1.0:
        x_pix = x * img_w
        y_pix = y * img_h
        w_pix = w * img_w
        h_pix = h * img_h
    else:
        # assume already in pixels
        x_pix, y_pix, w_pix, h_pix = x, y, w, h
    return x_pix, y_pix, w_pix, h_pix

def get_max_intensity_center(img):
    try:
        gray = np.array(img.convert('L'))
        y, x = np.unravel_index(np.nanargmax(gray), gray.shape)
        return int(x), int(y)
    except Exception:
        return img.width//2, img.height//2

# Main loop
results = []
skipped = []
fits_files = [f for f in os.listdir(FITS_DIR) if f.lower().endswith('.fits')]  # keep limit
for fn in tqdm(sorted(fits_files), desc="Processing FITS files", unit="file"):
    print(f"Processing {fn}...")
    path = os.path.join(FITS_DIR, fn)
    objid = extract_objid_from_filename(fn)
    # Match PNG by FITS prefix (everything before .fits)
    fits_prefix = fn.replace('.fits', '')
    png_candidates = [f for f in os.listdir(PNG_DIR) if f.startswith(fits_prefix) and f.lower().endswith('.png')]
    if not png_candidates:
        skipped.append(f"{fn}: no matching PNG found for prefix {fits_prefix}")
        print(f"Skipping {fn}: no matching PNG found for prefix {fits_prefix}")
        continue
    png_fn = png_candidates[0]
    png_path = os.path.join(PNG_DIR, png_fn)
    # catalog RA/DEC if present
    ra_cat = dec_cat = None
    row = hoyle_df[hoyle_df['objid'] == objid]
    if not row.empty:
        try:
            ra_cat = float(row.iloc[0]['ra'])
            dec_cat = float(row.iloc[0]['dec'])
        except Exception:
            ra_cat = dec_cat = None
    try:
        hdul = fits.open(path)
        header = hdul[0].header
        # Only use FITS for header/meta, not image data
        arcsec_per_pixel, distance_mpc, z_used = get_conversion_info(header, fn, ra_cat, dec_cat)
        if arcsec_per_pixel is None:
            skipped.append(f"{fn}: missing pixscale in header and Hoyle CSV")
            print(f"Skipping {fn}: missing pixscale (no header/Hoyle pixscale)")
            continue
        # Load PNG image for detection
        try:
            img = Image.open(png_path).convert('RGB')
        except Exception as e:
            skipped.append(f"{fn}: error loading PNG {png_fn}: {str(e)}")
            print(f"Skipping {fn}: error loading PNG {png_fn}: {str(e)}")
            continue
        img_w, img_h = img.width, img.height
        # Run detection
        preds = model.predict(img, conf=0.25, iou=0.45, verbose=False)
        if preds is None or len(preds) == 0:
            skipped.append(f"{fn}: model returned no predictions")
            print(f"Skipping {fn}: model returned no predictions")
            continue
        res = preds[0]

        # Choose detection set: prefer oriented bounding boxes (obb), else standard boxes
        detections = []
        # Try obb first
        if hasattr(res, 'obb') and res.obb is not None and len(res.obb) > 0:
            try:
                xywhrs = res.obb.xywhr.tolist()
                confs = res.obb.conf.tolist()
                for xywhr, conf in zip(xywhrs, confs):
                    x_pix, y_pix, w_pix, h_pix = xywh_to_pixels(xywhr[:4], img_w, img_h)
                    detections.append({'x': x_pix, 'y': y_pix, 'w': w_pix, 'h': h_pix, 'r': float(xywhr[4]), 'conf': float(conf)})
            except Exception as e:
                logging.debug(f"{fn}: obb parsing issue: {e}")
        # Fallback to boxes (xywh)
        if len(detections) == 0 and hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
            try:
                if hasattr(res.boxes, 'xywh') and res.boxes.xywh is not None:
                    xywhs = res.boxes.xywh.tolist()
                    confs = res.boxes.conf.tolist()
                    for xywh, conf in zip(xywhs, confs):
                        x_pix, y_pix, w_pix, h_pix = xywh_to_pixels(xywh, img_w, img_h)
                        detections.append({'x': x_pix, 'y': y_pix, 'w': w_pix, 'h': h_pix, 'r': 0.0, 'conf': float(conf)})
                else:
                    xyxys = res.boxes.xyxy.tolist()
                    confs = res.boxes.conf.tolist()
                    for xyxy, conf in zip(xyxys, confs):
                        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
                        w_pix = abs(x2 - x1); h_pix = abs(y2 - y1)
                        x_pix = (x1 + x2) / 2.0; y_pix = (y1 + y2) / 2.0
                        detections.append({'x': x_pix, 'y': y_pix, 'w': w_pix, 'h': h_pix, 'r': 0.0, 'conf': float(conf)})
            except Exception as e:
                logging.debug(f"{fn}: boxes parsing issue: {e}")
        if len(detections) == 0:
            skipped.append(f"{fn}: no bars detected")
            print(f"Skipping {fn}: no bars detected")
            continue
        max_x, max_y = get_max_intensity_center(img)
        best_idx = None; best_score = -1.0
        for i, d in enumerate(detections):
            conf = d['conf']
            x = d['x']; y = d['y']
            dist = sqrt((x - max_x)**2 + (y - max_y)**2)
            score = conf / (dist + 1e-9)
            if score > best_score:
                best_idx = i; best_score = score
        if best_idx is None:
            skipped.append(f"{fn}: no valid bar after scoring")
            print(f"Skipping {fn}: no valid bar after scoring")
            continue
        d = detections[best_idx]
        w_pix = float(d['w']); h_pix = float(d['h'])
        pixel_length = max(w_pix, h_pix)
        pixel_width = min(w_pix, h_pix)
        arcsec_length = pixel_length * float(arcsec_per_pixel)
        arcsec_width = pixel_width * float(arcsec_per_pixel)
        physical_length_kpc = physical_width_kpc = None
        if distance_mpc is not None:
            physical_length_kpc = arcsec_to_kpc(arcsec_length, distance_mpc)
            physical_width_kpc = arcsec_to_kpc(arcsec_width, distance_mpc)
        else:
            skipped.append(f"{fn}: missing distance/z")
            print(f"Skipping {fn}: missing distance (no z found). Logged to missing_redshifts.csv later.")
            continue
        angle_rad = float(d.get('r', 0.0)) % pi
        orientation_deg = (180 - np.degrees(angle_rad)) % 180 if w_pix > h_pix else (90 - np.degrees(angle_rad)) % 180
        results.append({
            'filename': fn,
            'objid': objid,
            'pixel_length': pixel_length,
            'pixel_width': pixel_width,
            'orientation_deg': orientation_deg,
            'arcsec_length': arcsec_length,
            'arcsec_width': arcsec_width,
            'physical_length_kpc': physical_length_kpc,
            'physical_width_kpc': physical_width_kpc,
            'distance_Mpc': distance_mpc,
            'z_used': z_used
        })
        # Save detected bounding box overlay image
        try:
            # Use res.plot() if available, else draw manually
            if hasattr(res, 'plot') and callable(res.plot):
                detected_img = res.plot()
                Image.fromarray(detected_img).save(os.path.join(DETECTED_IMG_DIR, png_fn))
            else:
                # Manual draw: overlay rectangle on img
                from PIL import ImageDraw
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                x0 = d['x'] - d['w']/2; y0 = d['y'] - d['h']/2
                x1 = d['x'] + d['w']/2; y1 = d['y'] + d['h']/2
                draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
                draw_img.save(os.path.join(DETECTED_IMG_DIR, png_fn))
        except Exception as e:
            logging.warning(f"Could not save detected image for {fn}: {e}")
    except Exception as e:
        logging.exception(f"Error processing {fn}: {e}")
        skipped.append(f"{fn}: error {str(e)}")
        print(f"Skipping {fn}: error {str(e)}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(BAR_CSV, index=False)
with open(LOG_FILE, 'w') as f:
    for line in skipped:
        f.write(line + '\n')

# Write missing redshifts file (unique objid/ra/dec) for manual crossmatch
missing_objs = []
for s in skipped:
    if 'missing distance' in s or 'missing pixscale' in s:
        # attempt extract filename from string
        parts = s.split(':',1)[0]
        fn = parts.strip()
        objid = extract_objid_from_filename(fn)
        row = hoyle_df[hoyle_df['objid'] == objid]
        if not row.empty:
            try:
                missing_objs.append({'objid': objid, 'ra': float(row.iloc[0]['ra']), 'dec': float(row.iloc[0]['dec'])})
            except Exception:
                missing_objs.append({'objid': objid, 'ra': None, 'dec': None})
        else:
            missing_objs.append({'objid': objid, 'ra': None, 'dec': None})
if missing_objs:
    pd.DataFrame(missing_objs).drop_duplicates('objid').to_csv(MISSING_Z_CSV, index=False)

# Comparison plots (robust handling)
if not results_df.empty:
    # Merge with Hoyle using objid
    merge_df = results_df.merge(hoyle_df, on='objid', how='left', suffixes=('_ours', '_hoyle'))
    # Compute Hoyle physical length if length_avg present and we have distance
    hoyle_kpc = []
    for i, row in merge_df.iterrows():
        try:
            if pd.notna(row.get('length_avg')) and pd.notna(row.get('distance_Mpc')):
                length_avg = float(row['length_avg'])
                distance_mpc = float(row['distance_Mpc'])
                hoy_kpc = arcsec_to_kpc(length_avg, distance_mpc)
                hoyle_kpc.append(hoy_kpc)
            else:
                hoyle_kpc.append(np.nan)
        except Exception:
            hoyle_kpc.append(np.nan)
    merge_df['hoyle_length_kpc'] = hoyle_kpc

    # Plot all measured bar lengths
    # Prepare data for plotting
    matched_mask = ~merge_df['hoyle_length_kpc'].isna()
    unmatched_mask = merge_df['hoyle_length_kpc'].isna()
    # Scatter plot: blue for matched, red for unmatched
    plt.figure()
    plt.scatter(merge_df.loc[matched_mask, 'physical_length_kpc'], merge_df.loc[matched_mask, 'hoyle_length_kpc'], s=10, c='blue', label='Matched Hoyle')
    plt.scatter(merge_df.loc[unmatched_mask, 'physical_length_kpc'], [0]*unmatched_mask.sum(), s=10, c='red', label='No Hoyle match')
    mx = max(np.nanmax(merge_df['physical_length_kpc']), np.nanmax(merge_df['hoyle_length_kpc'])) * 1.05
    plt.plot([0, mx], [0, mx], 'r--', label='1:1 line')
    plt.xlabel('Measured bar length (kpc)')
    plt.ylabel('Hoyle bar length (kpc)')
    plt.title('Bar Length Comparison (All Measured)')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'bar_length_comparison_all.png'))

    # Residuals for matched only
    comp_df = merge_df.dropna(subset=['physical_length_kpc','hoyle_length_kpc'])
    if not comp_df.empty:
        measured = comp_df['physical_length_kpc'].values
        hoyle = comp_df['hoyle_length_kpc'].values
        residuals = measured - hoyle
        pearson_r = scipy.stats.pearsonr(measured, hoyle)[0] if len(measured) > 1 else np.nan
        rmse = np.sqrt(np.mean(residuals**2))
        l1 = np.mean(np.abs(residuals))
        n_matched = len(comp_df)
        # Residuals histogram
        plt.figure()
        plt.hist(residuals, bins=20)
        plt.xlabel('Measured - Hoyle (kpc)')
        plt.ylabel('Count')
        plt.title('Residuals Histogram (Matched Only)')
        plt.savefig(os.path.join(PLOTS_DIR, 'residuals_histogram_matched.png'))
        with open(SUMMARY_FILE, 'w') as f:
            f.write(f"Pearson correlation (ρ): {pearson_r:.3f}\n")
            f.write(f"RMSE: {rmse:.3f} kpc\n")
            f.write(f"Linear correlation (r): {pearson_r:.3f}\n")
            f.write(f"Mean Absolute Error (L1): {l1:.3f} kpc\n")
            f.write(f"Number of matched objects: {n_matched}\n")
            f.write(f"Number of measured objects: {len(results_df)}\n")
    else:
        with open(SUMMARY_FILE, 'w') as f:
            f.write("No matched objects with both measured and Hoyle kpc lengths.\n")
            f.write(f"Number of measured objects: {len(results_df)}\n")
else:
    with open(SUMMARY_FILE, 'w') as f:
        f.write("No valid measurements found.\n")

# Prepare comparison CSV for matched objects
comp_df = merge_df.dropna(subset=['physical_length_kpc','hoyle_length_kpc'])
if not comp_df.empty:
    comparison_df = comp_df[['objid', 'hoyle_length_kpc', 'physical_length_kpc']].copy()
    comparison_df.rename(columns={'hoyle_length_kpc': 'hoyle_bar_length_kpc', 'physical_length_kpc': 'model_bar_length_kpc'}, inplace=True)
    comparison_df.to_csv(BAR_COMPARISON_CSV, index=False)
    # Plot as in attached image
    measured = comparison_df['model_bar_length_kpc'].values
    hoyle = comparison_df['hoyle_bar_length_kpc'].values
    plt.figure(figsize=(4,4))
    plt.scatter(measured, hoyle, s=20, c='dodgerblue', marker='+')
    mx = max(np.nanmax(measured), np.nanmax(hoyle)) * 1.05
    plt.plot([0, mx], [0, mx], 'k--', alpha=0.5)
    pearson_r = scipy.stats.pearsonr(measured, hoyle)[0] if len(measured) > 1 else np.nan
    l1 = np.mean(np.abs(measured - hoyle))
    plt.text(0.05*mx, 0.9*mx, f"r = {pearson_r:.2f}\nL1={l1:.2f} kpc", fontsize=10)
    plt.xlabel('Predicted bar length [kpc]')
    plt.ylabel('Hoyle (2011) bar length [kpc]')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bar_length_comparison_matched.png'))

print("Done. Results written to:", BAR_CSV)
print("Missing redshifts (if any) written to:", MISSING_Z_CSV)
print("Logs written to:", LOG_FILE)
