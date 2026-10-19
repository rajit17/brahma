import os
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from math import pi, sqrt

# Load the trained model
model = YOLO('best.pt')

# Get user input for conversion parameters
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive number")

PIXSCALE = get_float_input("Enter PIXSCALE (arcsec/pixel): ")
DISTANCE_MPC = get_float_input("Enter galaxy distance (MPC): ")

# CHANGED: Direct pixel-to-kpc conversion factor
PIXEL_TO_KPC = (PIXSCALE * DISTANCE_MPC) / 206.265  # kpc/pixel

# Define directories (remain unchanged)
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'predictions'

os.makedirs(OUTPUT_DIR, exist_ok=True)
processed = set()

print(f"\nMonitoring {INPUT_DIR} for new image files...")

def get_max_intensity_center(img):
    """Find center of maximum intensity region"""
    try:
        gray = np.array(img.convert('L'))
        y, x = np.unravel_index(np.argmax(gray), gray.shape)
        return int(x), int(y)
    except:
        return img.width//2, img.height//2

def pixels_to_kpc(pixels):
    """Convert pixel measurements to kiloparsecs (SIMPLIFIED)"""
    return pixels * PIXEL_TO_KPC  # Direct multiplication

while True:
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    new_files = set(files) - processed
    
    if new_files:
        print(f"\nFound {len(new_files)} new image(s): {sorted(new_files)}")
        
        for fn in sorted(new_files):
            path = os.path.join(INPUT_DIR, fn)
            print(f"\nProcessing {fn}...")
            
            try:
                img = Image.open(path)
                preds = model.predict(img, conf=0.25, iou=0.45)
                res = preds[0]
                
                info = "No bars detected"
                out = np.array(img)

                if res.obb is not None and len(res.obb) > 0:
                    max_x, max_y = get_max_intensity_center(img)
                    
                    best_idx, best_score = None, -1
                    for i in range(len(res.obb)):
                        obb = res.obb.xywhr[i]
                        conf = res.obb.conf[i].item()
                        
                        if obb is None or conf < 0.25:
                            continue
                            
                        x = obb[0].item() * img.width
                        y = obb[1].item() * img.height
                        dist = sqrt((x - max_x)**2 + (y - max_y)**2)
                        score = conf / (dist + 1e-9)
                        
                        if score > best_score:
                            best_idx, best_score = i, score

                    if best_idx is not None:
                        res.obb = res.obb[best_idx:best_idx+1]
                        
                        obb = res.obb.xywhr[0].tolist()
                        x, y, w, h, r = obb
                        w_pix = w 
                        h_pix = h 
                        
                        # Convert to kpc
                        length_kpc = pixels_to_kpc(max(w_pix, h_pix))
                        breadth_kpc = pixels_to_kpc(min(w_pix, h_pix))
                        
                        # Angle calculation
                        angle_rad = r % pi
                        angle = (180 - np.degrees(angle_rad)) % 180 if w_pix > h_pix else (90 - np.degrees(angle_rad)) % 180
                        
                        info = (f"Bar Dimensions: {length_kpc:.2f} kpc × {breadth_kpc:.2f} kpc\n"
                                f"Orientation: {angle:.2f}° from positive x-axis (anti-clockwise)\n"
                                f"Detection Confidence: {res.obb.conf[0].item():.2f}")

                        # Plot image with only the selected bar
                        out = res.plot()

                # Save results
                with open(os.path.join(OUTPUT_DIR, f"{os.path.splitext(fn)[0]}.txt"), 'w') as f:
                    f.write(info + f"\nConversion Parameters:\nPIXSCALE: {PIXSCALE} arcsec/px\nDistance: {DISTANCE_MPC} MPC")
                
                Image.fromarray(out).save(os.path.join(OUTPUT_DIR, fn))
                print(f"Results:\n{info}\nSaved to {OUTPUT_DIR}/{fn}")
                
            except Exception as e:
                print(f"Error processing {fn}: {str(e)}")
        
        processed.update(new_files)
    
    time.sleep(1)