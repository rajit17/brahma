#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import shutil
import random
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

"""
prepare_and_train_obb.py

What this script does:
1. Convert 4-line 'x y' label files (in images_with_bbox/labels) into single-line
   YOLO-OBB style: "0 x1 y1 x2 y2 x3 y3 x4 y4" with coords normalized to [0,1].
2. Collect images from BEST/ that have corresponding labels; save images without
   labels into data/images_no_bbox/.
3. Create train/val/test splits and copy images+labels into data/images/{train,val,test}
   and data/labels/{train,val,test}.
4. Write a dataset YAML for ultralytics YOLO obb task.
5. Launch training (YOLOv8 obb pretrained specified).
6. Validate the trained model.
7. Start a polling loop that watches val images and saves annotated predictions
   into data/val_preds/ (press Ctrl+C to stop).

Notes & assumptions:
- Label input format expected: four lines, each with "x y" (pixel coords).
  The code tolerates extra whitespace or extra columns per line (takes first two tokens).
- Uses copies (shutil.copy) to avoid losing originals.
- Adjust MODEL_PRETRAIN, EPOCHS, BATCH, IMG_SIZE as needed.
"""

# ---------- CONFIG (set by argparse) ----------
def get_config():
    parser = argparse.ArgumentParser(description="Prepare YOLO OBB data and train model. All paths should be absolute or relative to the script location.")
    parser.add_argument('--best_dir', type=str, default='BEST', help='Directory with images (BEST)')
    parser.add_argument('--data_dir', type=str, default='legacy_with_bounding_box', help='Main data directory')
    parser.add_argument('--img_dir', type=str, default=None, help='Directory with images with bbox (default: <data_dir>/images_with_bbox)')
    parser.add_argument('--lbl_dir', type=str, default=None, help='Directory with label txts (default: <data_dir>/images_with_bbox/labels)')
    parser.add_argument('--out_images_no_bbox', type=str, default=None, help='Output dir for images without bbox (default: <data_dir>/images_no_bbox)')
    parser.add_argument('--out_pred_dir', type=str, default=None, help='Output dir for val predictions (default: <data_dir>/val_preds)')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=(0.8,0.1,0.1), help='Train/val/test split ratios')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_pretrain', type=str, default='yolov8s-obb.pt', help='Pretrained YOLO OBB model')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for training')
    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).resolve().parent
    best_dir = Path(args.best_dir) if os.path.isabs(args.best_dir) else base_dir / args.best_dir
    data_dir = Path(args.data_dir) if os.path.isabs(args.data_dir) else base_dir / args.data_dir
    img_dir = Path(args.img_dir) if args.img_dir else data_dir / "images_with_bbox"
    lbl_dir = Path(args.lbl_dir) if args.lbl_dir else data_dir / "images_with_bbox" / "labels"
    out_images_no_bbox = Path(args.out_images_no_bbox) if args.out_images_no_bbox else data_dir / "images_no_bbox"
    out_pred_dir = Path(args.out_pred_dir) if args.out_pred_dir else data_dir / "val_preds"

    return dict(
        BASE_DIR=base_dir,
        BEST_DIR=best_dir,
        DATA_DIR=data_dir,
        IMG_DIR=img_dir,
        LBL_DIR=lbl_dir,
        OUT_IMAGES_NO_BBOX=out_images_no_bbox,
        OUT_PRED_DIR=out_pred_dir,
        SPLIT_RATIOS=args.split_ratios,
        RANDOM_SEED=args.random_seed,
        MODEL_PRETRAIN=args.model_pretrain,
        EPOCHS=args.epochs,
        BATCH=args.batch,
        IMG_SIZE=args.img_size
    )

# ---------- UTIL ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_image_files(d: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in exts and p.is_file()]

# ---------- STEP 1: Convert label files ----------
def convert_labels_to_one_line(label_dir: Path, image_dir: Path):
    """Convert label files where each file has 4 lines of 'x y' into a single
    YOLO OBB line: '0 x1 y1 x2 y2 x3 y3 x4 y4' with normalization.
    Returns: (converted_list, skipped_list)
    """
    converted = []
    skipped = []

    ensure_dir(label_dir)  # if it doesn't exist, will be empty and we skip
    for txt in sorted(label_dir.glob("*.txt")):
        try:
            # Try to find the corresponding image to know width/height
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

            if len(raw_lines) != 4:
                skipped.append((txt.name, f"expected 4 lines, got {len(raw_lines)}"))
                continue

            coords = []
            valid = True
            for ln in raw_lines:
                parts = ln.split()
                if len(parts) < 2:
                    valid = False
                    break
                x = float(parts[0])
                y = float(parts[1])
                # clamp to image bounds
                x = max(0.0, min(x, w))
                y = max(0.0, min(y, h))
                coords.extend([x / w, y / h])
            if not valid:
                skipped.append((txt.name, "bad coordinate line"))
                continue

            # Write back single-line label with class 0
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
def split_and_copy(with_bbox_imgs, label_dir: Path, data_dir: Path, split_ratios, random_seed, best_dir):
    random.seed(random_seed)
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
            img_src = (with_bbox_imgs[0].parent / name) if with_bbox_imgs else None
            img_src = Path(with_bbox_imgs[0].parent / name) if with_bbox_imgs else None
            if img_src is None or not img_src.exists():
                # try best_dir fallback
                possible = list(best_dir.glob(name))
                if possible:
                    img_src = possible[0]
                else:
                    print(f"Warning: cannot find image source for {name}")
                    continue
            img_dst = data_dir / "images" / split / name
            shutil.copy(img_src, img_dst)

            # label
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
"""
    yaml_path = data_dir / out_name
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    return yaml_path

# ---------- TRAIN / VALIDATE ----------
def train_and_validate(yaml_path: Path, model_pretrain, epochs, batch, imgsz):
    print("Starting training...")
    model = YOLO(model_pretrain)
    model.train(data=str(yaml_path), epochs=epochs, batch=batch, imgsz=imgsz)
    # find best.pt
    run_dirs = sorted(glob("runs/obb/train/weights/*"), reverse=True)
    best_path = None
    possible = list(Path("runs/obb/train/weights").glob("best*.pt")) + list(Path("runs/obb/train/weights").glob("*.pt"))
    if possible:
        # choose the first best.pt-like or fallback to the newest .pt
        possible_sorted = sorted(possible, key=lambda p: p.stat().st_mtime, reverse=True)
        best_path = possible_sorted[0]
    else:
        # fallback default location
        candidate = Path("runs/obb/train/weights/best.pt")
        if candidate.exists():
            best_path = candidate
    if best_path:
        print(f"=== VALIDATE {best_path} ===")
        model = YOLO(str(best_path))
        model.val(data=str(yaml_path), batch=1, project="runs/obb/val_single", name="run1", save=True)
    else:
        print("Warning: could not find best.pt for validation. Check training output directory.")

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
                        # Score concept: select best bar by confidence/distance to center
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
                                # Keep only the best OBB
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
    """
    Usage (on server):
      python prepare_yolo_data.py --best_dir /path/to/BEST --data_dir /path/to/legacy_with_bounding_box --model_pretrain /path/to/yolov8s-obb.pt
    All other arguments are optional and have sensible defaults.
    """
    config = get_config()
    BASE_DIR = config['BASE_DIR']
    BEST_DIR = config['BEST_DIR']
    DATA_DIR = config['DATA_DIR']
    IMG_DIR = config['IMG_DIR']
    LBL_DIR = config['LBL_DIR']
    OUT_IMAGES_NO_BBOX = config['OUT_IMAGES_NO_BBOX']
    OUT_PRED_DIR = config['OUT_PRED_DIR']
    SPLIT_RATIOS = config['SPLIT_RATIOS']
    RANDOM_SEED = config['RANDOM_SEED']
    MODEL_PRETRAIN = config['MODEL_PRETRAIN']
    EPOCHS = config['EPOCHS']
    BATCH = config['BATCH']
    IMG_SIZE = config['IMG_SIZE']

    print("=== START ===")
    ensure_dir(DATA_DIR)
    ensure_dir(BEST_DIR)

    # 1) Convert labels
    print("Converting label files to YOLO OBB single-line format...")
    converted, skipped = convert_labels_to_one_line(LBL_DIR, IMG_DIR)
    print(f"Converted: {len(converted)}, Skipped: {len(skipped)}")
    if skipped:
        print("Skipped examples (name, reason):")
        for s in skipped[:20]:
            print(" ", s)

    # 2) Collect images from BEST_DIR (copy)
    images_with_bbox_out = DATA_DIR / "images_with_bbox_out"
    with_bbox, no_bbox = collect_images(BEST_DIR, LBL_DIR, images_with_bbox_out, OUT_IMAGES_NO_BBOX)
    print(f"Images with bbox: {len(with_bbox)}, images without bbox saved to {OUT_IMAGES_NO_BBOX} ({len(no_bbox)})")

    if not with_bbox:
        print("No images with bbox found in BEST_DIR. Exiting.")
        return  # Exit early if no valid images found

    # 3) Split and copy into data/images/{train,val,test} and data/labels/{...}
    splits = split_and_copy(with_bbox, LBL_DIR, DATA_DIR, split_ratios=SPLIT_RATIOS, random_seed=RANDOM_SEED, best_dir=BEST_DIR)
    print("Data split complete. Counts by split:")
    for sp, names in splits.items():
        print(f"  {sp}: {len(names)}")

    # 4) Write YAML
    yaml_path = write_yaml(DATA_DIR)
    print(f"Config YAML written to {yaml_path}")

    # 5) Train & validate
    best_path = train_and_validate(yaml_path, model_pretrain=MODEL_PRETRAIN, epochs=EPOCHS, batch=BATCH, imgsz=IMG_SIZE)

    # 6) Polling inference on val set using best_path if available else last model
    if best_path:
        inference_model = YOLO(str(best_path))
    else:
        inference_model = YOLO(MODEL_PRETRAIN)

    val_dir = DATA_DIR / "images" / "val"
    polling_inference(inference_model, val_dir, OUT_PRED_DIR)

if __name__ == "__main__":
    main()
