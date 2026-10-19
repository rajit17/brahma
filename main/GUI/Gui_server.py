# server_fixed_preview.py
# BRAHMa server that produces:
# - BEST: clean FITS to PNG
# - predictions: YOLO overlay
# - preview: thumbnails (not used by relay but kept)

import os, time, traceback, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import numpy as np
from PIL import Image
from astropy.io import fits

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------- CONFIG ----------------
API_TOKEN = "my_server_token_123"
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "best.pt"
yolo_model = YOLO(str(MODEL_PATH)) if (MODEL_PATH.exists() and YOLO is not None) else None

app = Flask(__name__)
CORS(app)

# ---------------- UTILITIES ----------------
def make_job_folder():
    jid = f"job_{int(time.time())}"
    folder = JOBS_DIR / jid
    (folder / "BEST").mkdir(parents=True, exist_ok=True)
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    (folder / "preview").mkdir(parents=True, exist_ok=True)
    (folder / "raw").mkdir(parents=True, exist_ok=True)
    return jid, folder

def fits_to_png(fits_path, out_path):
    """Convert FITS file to PNG with simple normalization."""
    try:
        with fits.open(fits_path) as hdul:
            arr = hdul[0].data
            if arr is None:
                Image.new("L", (64, 64), 0).save(out_path)
                return
            if getattr(arr, "ndim", 2) != 2:
                arr = arr[0]
            
            # --- FIX START: Correct FITS orientation ---
            # FITS origin is bottom-left; images are top-left. 
            # Flip up-down to fix the "upside down" text.
            arr = np.flipud(arr)
            # --- FIX END ---

            a = np.nan_to_num(arr.astype("float64"))
            a -= np.nanmin(a)
            mx = np.nanmax(a) or 1.0
            a = (a / mx * 255.0).clip(0, 255).astype("uint8")
            Image.fromarray(a).save(out_path)
    except Exception as e:
        print("FITS->PNG error:", e)
        Image.new("L", (64, 64), 0).save(out_path)

def make_preview(src_path, out_path, size=(256, 256)):
    try:
        img = Image.open(src_path)
        img.thumbnail(size)
        img.save(out_path)
    except Exception:
        Image.new("L", size, 0).save(out_path)

def process_one(fits_path, best_dir, pred_dir, preview_dir):
    """Process one FITS file into BEST (clean) and prediction images."""
    original_name = os.path.basename(fits_path)
    png_name = f"{original_name}.png"
    best_png = best_dir / png_name
    pred_png = pred_dir / png_name
    preview_png = preview_dir / png_name

    # Create BEST (clean) image from FITS
    fits_to_png(fits_path, best_png)

    result = {
        "Filename": png_name,
        "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "r": 0.0,
        "conf": 0.0, "info": "no-detection"
    }

    if yolo_model is None:
        # No model: prediction is just a copy of BEST
        try:
            shutil.copy(best_png, pred_png)
        except Exception:
            Image.new("L", (64, 64), 0).save(pred_png)
        make_preview(pred_png, preview_png)
        return result

    try:
        preds = yolo_model.predict(source=str(best_png), conf=0.25, iou=0.45, imgsz=640)
        r = preds[0]
        obb = getattr(r, "obb", None)

        best_conf = 0.0
        best_idx = -1
        best_obb = None

        if obb is not None and len(obb) > 0:
            for i in range(len(obb)):
                try:
                    c = float(obb.conf[i].item())
                except Exception:
                    c = float(obb.conf[i])
                if c > best_conf:
                    best_conf = c
                    best_obb = obb.xywhr[i]
                    best_idx = i

        if best_obb is not None and best_conf > 0.0:
            try:
                cx, cy, w_box, h_box, r_rad = [float(x.item()) for x in best_obb]
            except Exception:
                cx, cy, w_box, h_box, r_rad = [float(x) for x in best_obb]
            result.update({
                "x": cx, "y": cy, "w": w_box, "h": h_box, "r": r_rad,
                "conf": float(best_conf), "info": "detected"
            })
            try:
                r.obb = obb[best_idx:best_idx+1]
            except Exception:
                pass

        # Save prediction image (with boxes if any)
        # Save prediction image (with boxes if any)
        try:
            # --- FIX START: Hide labels and confidence ---
            # labels=False hides "galaxy"
            # conf=False hides the probability score
            arr = r.plot(labels=False, conf=False)
            # --- FIX END ---
            
            if arr is not None:
                Image.fromarray(arr).save(pred_png)
            else:
                shutil.copy(best_png, pred_png)
        except Exception:
             # ... rest of your error handling ...
            try:
                shutil.copy(best_png, pred_png)
            except Exception:
                Image.new("L", (64, 64), 0).save(pred_png)

        make_preview(pred_png, preview_png)

    except Exception as e:
        print("YOLO predict failed:", e)
        traceback.print_exc()
        try:
            shutil.copy(best_png, pred_png)
        except Exception:
            Image.new("L", (64, 64), 0).save(pred_png)
        make_preview(pred_png, preview_png)
        result["info"] = f"error: {e}"

    return result

def is_image_format(filename):
    """Check if file is a common image format (not FITS)."""
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
    return Path(filename).suffix.lower() in image_exts

def image_to_fits(image_path, fits_path):
    """Convert image file (PNG, JPG, etc.) to FITS format."""
    try:
        img = Image.open(image_path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        arr = np.array(img, dtype='float32')
        
        # --- FIX START: Correct image orientation for FITS ---
        # FITS uses bottom-left origin; PIL images use top-left.
        # Only flip vertically (up-down) to convert PIL to FITS coordinates.
        # Do NOT flip horizontally as that mirrors the image.
        arr = np.flipud(arr)
        # --- FIX END ---
        
        # Create FITS HDU and save
        hdu = fits.PrimaryHDU(data=arr)
        hdu.writeto(fits_path, overwrite=True)
        return True
    except Exception as e:
        print(f"Image to FITS conversion error: {e}")
        return False

# ---------------- ROUTES ----------------
@app.route("/compute", methods=["POST"])
def compute():
    token = request.form.get("token")
    if token != API_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    if not request.files:
        return jsonify({"error": "no files uploaded"}), 400

    job_id, job_folder = make_job_folder()
    best_dir = job_folder / "BEST"
    pred_dir = job_folder / "predictions"
    preview_dir = job_folder / "preview"
    raw_dir = job_folder / "raw"

    fits_files = []
    
    # Use getlist to retrieve all files
    files_list = request.files.getlist('files')
    if not files_list:
        for k, f in request.files.items():
            files_list.append(f)

    for f in files_list:
        fname = f.filename or "file.fits"
        outp = raw_dir / fname
        f.save(outp)
        
        # Check if file is non-FITS image and convert it
        if is_image_format(fname):
            fits_path = raw_dir / f"{Path(fname).stem}.fits"
            if image_to_fits(outp, fits_path):
                fits_files.append(fits_path)
                # Optionally remove the original image file
                try:
                    os.remove(outp)
                except Exception:
                    pass
            else:
                # Conversion failed, use original
                fits_files.append(outp)
        else:
            fits_files.append(outp)

    results = []
    for fp in fits_files:
        try:
            r = process_one(fp, best_dir, pred_dir, preview_dir)
            results.append(r)
        except Exception as e:
            traceback.print_exc()
            fname = os.path.basename(fp)
            png_name = f"{fname}.png"
            results.append({
                "Filename": png_name,
                "x": 0, "y": 0, "w": 0, "h": 0, "r": 0,
                "conf": 0, "info": f"proc-error:{e}"
            })

    confs = [float(r.get("conf", 0.0)) for r in results]
    detected = sum(1 for r in results if str(r.get("info", "")).startswith("detected"))
    avg_conf = float(np.mean(confs)) if confs else 0.0

    base = f"{request.url_root.rstrip('/')}/jobs/{job_id}"
    pred_base = f"{base}/predictions/"
    preview_base = f"{base}/preview/"

    return jsonify({
        "job_id": job_id,
        "summary": {"total": len(results), "detected": detected, "avg_conf": avg_conf},
        "results": results,
        "best_url": f"{base}/BEST/",
        "pred_url": pred_base,
        "preview_url": preview_base
    })

@app.route("/jobs/<job_id>/<folder>/<path:filename>")
def serve_file(job_id, folder, filename):
    folder_path = JOBS_DIR / job_id / folder
    if not folder_path.exists():
        abort(404)
    return send_from_directory(str(folder_path), filename)

@app.route("/")
def root():
    return jsonify({"status": "server running", "model_loaded": yolo_model is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)