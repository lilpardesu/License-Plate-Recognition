import os
import cv2
import numpy as np
from ultralytics import YOLO
from preprocess_ocr import preprocess_plate

YOLO_WEIGHTS = "ir_plate_detector/weights/best.pt"
INPUT_DIR = "data/test"
OUTPUT_DIR = "outputs/preprocess_demo"
MAX_IMAGES = 20  # number of test images to demo

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def make_panel(images, titles, target_h=180):
    # Resize each image to same height, keep aspect, stack horizontally
    resized = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, title in zip(images, titles):
        if img is None:
            img = np.zeros((target_h, target_h, 3), dtype=np.uint8)

        if len(img.shape) == 2:  # gray -> bgr for text
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        scale = target_h / max(h, 1)
        nw = max(1, int(w * scale))
        img_r = cv2.resize(img, (nw, target_h))
        cv2.putText(img_r, title, (8, 22), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        resized.append(img_r)

    return cv2.hconcat(resized)

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input folder not found: {INPUT_DIR}")
        return

    ensure_dir(OUTPUT_DIR)

    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    files = list_images(INPUT_DIR)
    if len(files) == 0:
        print(f"❌ No images found in {INPUT_DIR}")
        return

    files = files[:MAX_IMAGES]
    print(f"Processing {len(files)} images...")

    saved = 0
    for name in files:
        img_path = os.path.join(INPUT_DIR, name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # YOLO detect plate
        results = yolo(img, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        confs = results.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)

        plate = safe_crop(img, x1, y1, x2, y2)
        if plate is None:
            continue

        # Full preprocessing with debug outputs
        processed, dbg = preprocess_plate(plate, return_debug=True)

        gray = dbg.get("gray", None)
        th = dbg.get("adaptive_thresh", None)
        morph = dbg.get("morph", None)
        final_img = dbg.get("final", processed)

        # Build and save panel
        panel = make_panel(
            images=[plate, gray, th, morph, final_img],
            titles=["plate_crop", "grayscale", "adaptive_thresh", "morph", "final_for_ocr"],
            target_h=180
        )

        out_name = os.path.splitext(name)[0] + "_pipeline.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, panel)
        saved += 1

    print(f"✅ Saved {saved} preprocessing demo panels to: {OUTPUT_DIR}")
    print("You can include these images in your report as proof of preprocessing steps.")

if __name__ == "__main__":
    main()
