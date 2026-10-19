import os
import pandas as pd
import shutil
import random
import time
from glob import glob
from pathlib import Path
import requests
from urllib.parse import urljoin
from tenacity import retry, stop_after_attempt, wait_exponential

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------- CONFIG ----------
BEST_DIR = Path("/home/rajit/DATA_LEGACY_CONVERTED_BW/BEST")  # images collected here (png/jpg)
DATA_DIR = Path("/home/rajit/legacy_with_bounding_box")
IMG_DIR = DATA_DIR / "images_with_bbox"                 # if you already have images here
LBL_DIR = DATA_DIR / "images_with_bbox" / "labels"     # original labels
OUT_IMAGES_NO_BBOX = DATA_DIR / "images_no_bbox"
OUT_PRED_DIR = DATA_DIR / "val_preds"
WEIGHTS_DIR = Path("weights")  # Directory to store downloaded model weights

SPLIT_RATIOS = (0.8, 0.1, 0.1)   # train, val, test
RANDOM_SEED = 42

MODEL_PRETRAIN = "yolo11x-obb.pt"  # YOLO11 extra-large for higher accuracy
EPOCHS = 30
BATCH = 32
IMG_SIZE = 512

# ---------- UTIL ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_image_files(d: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in exts and p.is_file()]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_yolo_model(model_name: str, weights_dir: Path):
    """Download YOLO11 model weights if not already present."""
    ensure_dir(weights_dir)
    model_path = weights_dir / model_name
    if model_path.exists():
        print(f"Model weights already exist at {model_path}")
        return model_path
    
    # Updated Ultralytics YOLO11 weights URL
    base_url = "https://github.com/ultralytics/assets/releases/download/v9.0.0/"
    url = urljoin(base_url, model_name)
    
    try:
        print(f"Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model weights downloaded to {model_path}")
            return model_path
        else:
            raise Exception(f"Failed to download {model_name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print("Falling back to Ultralytics auto-download during model initialization.")
        return model_name  # Let YOLO handle downloading if this fails

# ---------- STEP 1: Convert label files ----------
def convert_labels_to_one_line(label_dir: Path, image_dir: Path):
    """Convert label files to YOLO OBB single-line format: '0 x1 y1 x2 y2 x3 y3 x4 y4' with normalization.
    Handles single-line input or skips invalid files.
    Returns: (converted_list, skipped_list)
    """
    converted = []
    skipped = []

    ensure_dir(label_dir)
    for txt in sorted(label_dir.glob("*.txt")):
        try:
            # Find corresponding image
            stem = txt.stem
            img_candidates = [image_dir / (stem + ext) for ext in (".png", ".jpg", ".jpeg")]
            img_path = next((p for p in img_candidates if p.exists()), None)
            if img_path is None:
                skipped.append((txt.name, "no image found"))
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append((txt.name, "unable to read image"))
                continue
            h, w = img.shape[:2]

            with open(txt, "r") as f:
                raw_lines = [ln.strip() for ln in f if ln.strip()]

            if len(raw_lines) == 1:
                # Check if already in YOLO OBB format (class + 8 coords)
                parts = raw_lines[0].split()
                if len(parts) == 9 and parts[0].isdigit():
                    try:
                        coords = [float(x) for x in parts[1:]]
                        if all(0 <= x <= 1 for x in coords):  # Already normalized
                            converted.append(txt.name)
                            continue
                    except ValueError:
                        skipped.append((txt.name, "invalid single-line format"))
                        continue
                else:
                    skipped.append((txt.name, f"expected 9 values for single-line OBB, got {len(parts)}"))
                    continue
            elif len(raw_lines) != 4:
                skipped.append((txt.name, f"expected 4 lines or 1 valid OBB line, got {len(raw_lines)}"))
                continue

            # Convert 4-line format to single-line YOLO OBB
            coords = []
            valid = True
            for ln in raw_lines:
                parts = ln.split()
                if len(parts) < 2:
                    valid = False
                    break
                x = float(parts[0])
                y = float(parts[1])
                x = max(0.0, min(x, w))
                y = max(0.0, min(y, h))
                coords.extend([x / w, y / h])
            if not valid:
                skipped.append((txt.name, "bad coordinate line"))
                continue

            # Write single-line label with class 0
            new_line = "0 " + " ".join(f"{v:.6f}" for v in coords)
            with open(txt, "w") as f:
                f.write(new_line + "\n")
            converted.append(txt.name)
        except Exception as e:
            skipped.append((txt.name, f"exception: {e}"))
    return converted, skipped

# ---------- STEP 2: Collect images that have labels, and save no-bbox images ----------
def collect_images(best_dir: Path, label_dir: Path, images_with_bbox_out: Path, images_no_bbox_out: Path):
    """
    From BEST_DIR, check which images have labels in label_dir.
    Copy images with labels to images_with_bbox_out and images without labels to images_no_bbox_out.
    Returns two lists: (images_with_bbox_list, images_no_bbox_list) as Path objects.
    """
    ensure_dir(images_with_bbox_out)
    ensure_dir(images_no_bbox_out)

    img_files = [p for p in sorted(best_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    with_bbox = []
    no_bbox = []

    for img in img_files:
        stem = img.stem
        label_path = label_dir / (stem + ".txt")
        if label_path.exists():
            shutil.copy(img, images_with_bbox_out / img.name)
            with_bbox.append(images_with_bbox_out / img.name)
        else:
            shutil.copy(img, images_no_bbox_out / img.name)
            no_bbox.append(images_no_bbox_out / img.name)

    return with_bbox, no_bbox

# ---------- STEP 3: Split into train/val/test and copy images+labels ----------
def split_and_copy(with_bbox_imgs, label_dir: Path, data_dir: Path, split_ratios=SPLIT_RATIOS):
    random.seed(RANDOM_SEED)
    N = len(with_bbox_imgs)
    files = [p.name for p in with_bbox_imgs]
    random.shuffle(files)
    n1 = int(split_ratios[0] * N)
    n2 = n1 + int(split_ratios[1] * N)
    splits = {
        "train": files[:n1],
        "val": files[n1:n2],
        "test": files[n2:],
    }

    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            ensure_dir(data_dir / sub / split)

    for split, names in splits.items():
        for name in names:
            img_src = with_bbox_imgs[0].parent / name
            if not img_src.exists():
                possible = list(BEST_DIR.glob(name))
                if possible:
                    img_src = possible[0]
                else:
                    print(f"Warning: cannot find image source for {name}")
                    continue
            img_dst = data_dir / "images" / split / name
            shutil.copy(img_src, img_dst)

            label_src = label_dir / (Path(name).stem + ".txt")
            label_dst = data_dir / "labels" / split / (Path(name).stem + ".txt")
            if label_src.exists():
                shutil.copy(label_src, label_dst)
            else:
                print(f"Warning: missing label for {name}, expected {label_src}")

    return splits

# ---------- STEP 4: Write YAML ----------
def write_yaml(data_dir: Path, out_name="galaxy_obb.yaml"):
    yaml_text = f"""task: obb
path: {data_dir}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['galaxy']
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
mosaic: 0.5
"""
    yaml_path = data_dir / out_name
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    return yaml_path

# ---------- TRAIN / VALIDATE ----------
def train_and_validate(yaml_path: Path, model_pretrain=MODEL_PRETRAIN, epochs=EPOCHS, batch=BATCH, imgsz=IMG_SIZE):
    print("Starting training...")
    try:
        model_path = download_yolo_model(model_pretrain, WEIGHTS_DIR)
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to initialize model with {model_pretrain}: {e}")
        print("Exiting due to model initialization failure.")
        return None

    try:
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device='cpu',
            amp=True,
            cache='disk',
            workers=8,
            patience=10,
            box=10.0,
            dfl=2.0,
            label_smoothing=0.1,
            mixup=0.1,
            save_period=10,
            optimizer='AdamW',
            lr0=0.001,
            weight_decay=0.0005
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return None

    best_path = Path("runs/obb/train/weights/best.pt")
    if best_path.exists():
        print(f"=== VALIDATE {best_path} ===")
        model = YOLO(str(best_path))
        model.val(data=str(yaml_path), batch=1, project="runs/obb/val_single", name="run1", save=True)
    else:
        print("Warning: best.pt not found. Check training output directory.")
        best_path = None

    return best_path

# ---------- POLLING INFERENCE ----------
def polling_inference(model, val_dir: Path, out_dir: Path, poll_interval=1.0, conf=0.25, iou=0.45):
    ensure_dir(out_dir)
    processed = set()
    print(f"Monitoring {val_dir} for new image files (press Ctrl+C to stop)...\n")
    def get_max_intensity_center(img):
        gray = np.array(img.convert('L'))
        y, x = np.unravel_index(np.argmax(gray), gray.shape)
        return int(x), int(y)

    from math import sqrt

    try:
        while True:
            files = [f for f in sorted(val_dir.iterdir()) if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.is_file()]
            new_files = [f for f in files if f.name not in processed]
            if new_files:
                print(f"Found {len(new_files)} new image(s): {[f.name for f in new_files]}")
            for path in new_files:
                print(f"Processing {path.name} ...")
                try:
                    img = Image.open(path)
                    preds = model.predict(source=str(path), conf=conf, iou=iou)
                    if not preds:
                        print(" → No predictions returned, saving original image.")
                        img_arr = np.array(img)
                        out = img_arr
                    else:
                        res0 = preds[0]
                        if hasattr(res0, 'obb') and res0.obb is not None and len(res0.obb) > 0:
                            max_x, max_y = get_max_intensity_center(img)
                            best_idx, best_score = None, -1
                            for i in range(len(res0.obb)):
                                obb = res0.obb.xywhr[i]
                                conf_val = res0.obb.conf[i].item() if hasattr(res0.obb.conf[i], 'item') else float(res0.obb.conf[i])
                                if obb is None or conf_val < conf:
                                    continue
                                x = obb[0].item() * img.width
                                y = obb[1].item() * img.height
                                dist = sqrt((x - max_x)**2 + (y - max_y)**2)
                                score = conf_val / (dist + 1e-9)
                                if score > best_score:
                                    best_idx, best_score = i, score
                            if best_idx is not None:
                                res0.obb = res0.obb[best_idx:best_idx+1]
                                out = res0.plot() if hasattr(res0, "plot") else np.array(img)
                            else:
                                out = res0.plot() if hasattr(res0, "plot") else np.array(img)
                        else:
                            out = res0.plot() if hasattr(res0, "plot") else np.array(img)
                    Image.fromarray(out).save(out_dir / path.name)
                    print(f" → Saved annotated output to {out_dir / path.name}")
                except Exception as e:
                    print(f"Error processing {path.name}: {e}")
                processed.add(path.name)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nPolling stopped by user.")

# ---------- MAIN ----------
def main():
    print("=== START ===")
    ensure_dir(DATA_DIR)
    ensure_dir(BEST_DIR)
    ensure_dir(WEIGHTS_DIR)

    # 1) Convert labels
    print("Converting label files to YOLO OBB single-line format...")
    converted, skipped = convert_labels_to_one_line(LBL_DIR, IMG_DIR)
    print(f"Converted: {len(converted)}, Skipped: {len(skipped)}")
    if skipped:
        print("Skipped examples (name, reason):")
        for s in skipped[:20]:
            print(" ", s)
    if not converted:
        print("Error: No valid label files converted. Check label format and try again.")
        return

    # 2) Collect images from BEST_DIR (copy)
    images_with_bbox_out = DATA_DIR / "images_with_bbox_out"
    with_bbox, no_bbox = collect_images(BEST_DIR, LBL_DIR, images_with_bbox_out, OUT_IMAGES_NO_BBOX)
    print(f"Images with bbox: {len(with_bbox)}, images without bbox saved to {OUT_IMAGES_NO_BBOX} ({len(no_bbox)})")

    if not with_bbox:
        print("No images with bbox found in BEST_DIR. Exiting.")
        return

    # 3) Split and copy into data/images/{train,val,test} and data/labels/{...}
    splits = split_and_copy(with_bbox, LBL_DIR, DATA_DIR)
    print("Data split complete. Counts by split:")
    for sp, names in splits.items():
        print(f"  {sp}: {len(names)}")

    # 4) Write YAML
    yaml_path = write_yaml(DATA_DIR)
    print(f"Config YAML written to {yaml_path}")

    # 5) Train & validate
    best_path = train_and_validate(yaml_path)
    if best_path is None:
        print("Training failed. Skipping inference.")
        return

    # Save metrics to Excel after training
    results_dir = os.path.join('runs', 'obb', 'train')
    metrics_file = os.path.join(results_dir, 'results.csv')
    excel_file = os.path.join(results_dir, 'training_metrics.xlsx')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        columns_map = {
            'epoch': 'Epoch',
            'time': 'Time (s)',
            'train/box_loss': 'Train Box Loss',
            'train/cls_loss': 'Train Cls Loss',
            'train/dfl_loss': 'Train Dfl Loss',
            'val/box_loss': 'Val Box Loss',
            'val/cls_loss': 'Val Cls Loss',
            'val/dfl_loss': 'Val Dfl Loss',
            'metrics/precision(B)': 'Precision(B)',
            'metrics/recall(B)': 'Recall(B)',
            'metrics/mAP50(B)': 'MAP50(B)',
            'metrics/mAP50-95(B)': 'MAP50-95(B)',
            'lr/pg0': 'LR'
        }
        selected_cols = [c for c in columns_map if c in df.columns]
        df_out = df[selected_cols].rename(columns=columns_map)
        df_out.to_excel(excel_file, index=False)
        print(f"Epoch-wise metrics saved to {excel_file}")
    else:
        print(f"Metrics file not found: {metrics_file}. Excel export skipped.")

    # 6) Polling inference on val set using best_path if available else last model
    if best_path:
        inference_model = YOLO(str(best_path))
    else:
        try:
            inference_model = YOLO(download_yolo_model(MODEL_PRETRAIN, WEIGHTS_DIR))
        except Exception as e:
            print(f"Failed to load inference model: {e}")
            return

    val_dir = DATA_DIR / "images" / "val"
    polling_inference(inference_model, val_dir, OUT_PRED_DIR)

if __name__ == "__main__":
    main()