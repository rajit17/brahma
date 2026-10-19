
import os
try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def get_min_area_box(mask):
    """Get minimum area rectangle from binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.intp(box)

def get_max_iou_box(mask):
    """Get bounding box that maximizes IoU with the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

# -----------------------------
# Main code
# -----------------------------

# Directory containing all mask subfolders
masks_dir = r'/home/rajit/masks'


for subfolder in os.listdir(masks_dir):
    subfolder_path = os.path.join(masks_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    bar_mask_path = os.path.join(subfolder_path, 'bar_mask.png')
    galaxy_path = os.path.join(subfolder_path, 'galaxy.png')
    if not (os.path.exists(bar_mask_path) and os.path.exists(galaxy_path)):
        continue

    # Load probability map (bar mask)
    prob_map = cv2.imread(bar_mask_path, cv2.IMREAD_GRAYSCALE)
    if prob_map is None:
        continue
    prob_map = prob_map.astype(np.float32)/255.0

    # Only process thresholds 0.5 and 0.7 for Min Area
    for label, t in {"0.5": 0.5, "0.7": 0.7}.items():
        mask = (prob_map >= t).astype(np.uint8)*255
        min_area_box = get_min_area_box(mask)
        img = cv2.imread(galaxy_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if min_area_box is not None:
            cv2.drawContours(img, [min_area_box], -1, (255,0,0), 2)
            # Save bounding box coordinates to a text file
            box_txt_path = os.path.join(subfolder_path, f'minarea_{label.replace(".","")}_box.txt')
            with open(box_txt_path, 'w') as f:
                for pt in min_area_box:
                    f.write(f"{int(pt[0])} {int(pt[1])}\n")
            print(f'Saved: {box_txt_path}')
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(f"{label} - Min Area")
        plt.axis("off")
        save_path = os.path.join(subfolder_path, f'minarea_{label.replace(".","")}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Saved: {save_path}')
