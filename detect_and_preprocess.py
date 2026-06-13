import os
import cv2
import numpy as np
from ultralytics import YOLO


def run_lpr():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best.pt")
    input_path = os.path.abspath(os.path.join(current_dir, "license plate dataset", "train", "images"))
    output_path = os.path.abspath(os.path.join(current_dir, "data", "results"))

    if not os.path.exists(model_path) or not os.path.exists(input_path):
        print("Paths not found, check folder structure")
        return

    os.makedirs(output_path, exist_ok=True)
    model = YOLO(model_path)

    imgs = [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in imgs:
        full_path = os.path.join(input_path, img_name)
        img = cv2.imread(full_path)
        if img is None:
            continue

        # conf=0.5 to reduce false positives
        results = model(full_path, conf=0.5, verbose=False)

        for r in results:
            for j, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # skip detections that are too small to be a plate
                if w < 30 or h < 10:
                    continue

                crop = img[y1:y2, x1:x2]

                # upscale before processing
                resized = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray, h=10)

                # local contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)

                # mild sharpening
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)

                base = os.path.splitext(img_name)[0]
                out_file = os.path.join(output_path, f"{base}_plate_{j}.jpg")
                cv2.imwrite(out_file, sharpened)
                print(f"saved: {base}_plate_{j}.jpg")


if __name__ == "__main__":
    run_lpr()
