import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from ultralytics import YOLO

# ==================== PATHS (defaults) ====================
DEFAULT_YOLO_WEIGHTS = "ir_plate_detector/weights/best.pt"
DEFAULT_CRNN_WEIGHTS = "models/crnn/crnn_best.pt"
DEFAULT_OUTPUT_PATH = "outputs/one_image_result.jpg"

# ==================== CRNN ====================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)  # B,T,C
        rnn_out, _ = self.rnn(conv)
        return torch.log_softmax(self.fc(rnn_out), dim=2)

def load_crnn(path, device):
    ckpt = torch.load(path, map_location=device)
    idx_to_char = {v: k for k, v in ckpt["chars"].items()}
    model = CRNN(num_classes=len(ckpt["chars"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, idx_to_char

def decode(logits, idx_to_char):
    pred = torch.argmax(logits[0], dim=1).cpu().numpy()
    text, prev = [], -1
    for p in pred:
        if p != prev and p != 0:  # 0 is CTC blank
            text.append(idx_to_char.get(int(p), ""))
        prev = p
    return "".join(text)

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def main():
    parser = argparse.ArgumentParser(description="Test LPR on one image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--yolo", default=DEFAULT_YOLO_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--crnn", default=DEFAULT_CRNN_WEIGHTS, help="CRNN weights path")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Output visualization image path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Input image not found: {args.image}")
        return
    if not os.path.exists(args.yolo):
        print(f"❌ YOLO weights not found: {args.yolo}")
        return
    if not os.path.exists(args.crnn):
        print(f"❌ CRNN weights not found: {args.crnn}")
        return

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    yolo = YOLO(args.yolo)
    crnn, idx_to_char = load_crnn(args.crnn, device)

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Failed to read image: {args.image}")
        return

    # Detect plate
    results = yolo(img, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        print("⚠️ No plate detected.")
        cv2.imwrite(args.out, img)
        print(f"Saved unchanged image to: {args.out}")
        return

    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    det_conf = float(confs[best_idx])

    x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    plate_img = safe_crop(img, x1, y1, x2, y2)

    if plate_img is None:
        print("⚠️ Invalid plate crop.")
        cv2.imwrite(args.out, img)
        print(f"Saved image to: {args.out}")
        return

    # OCR preprocessing (same as your best pipeline)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    processed = cv2.resize(gray, (128, 32))

    x = processed.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = crnn(x)

    pred_text = decode(logits, idx_to_char)

    # Draw visualization
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # NOTE: OpenCV text rendering may not display Persian perfectly.
    label = f"plate: {pred_text} | conf: {det_conf:.2f}"
    y_text = max(20, y1 - 10)
    cv2.putText(vis, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Save outputs
    cv2.imwrite(args.out, vis)
    cv2.imwrite("outputs/one_image_crop.jpg", plate_img)
    cv2.imwrite("outputs/one_image_preprocessed.jpg", processed)

    print("\n✅ Prediction complete")
    print(f"Predicted plate text: {pred_text}")
    print(f"Detection confidence: {det_conf:.4f}")
    print(f"Saved visualization: {args.out}")
    print("Saved crop: outputs/one_image_crop.jpg")
    print("Saved preprocessed: outputs/one_image_preprocessed.jpg")

if __name__ == "__main__":
    main()
