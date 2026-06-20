import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import editdistance

# ==================== PATHS ====================
YOLO_WEIGHTS = 'ir_plate_detector/weights/best.pt'
CRNN_WEIGHTS = 'models/crnn/crnn_best.pt'
TEST_CSV = 'data/ocr/test_labels.csv'
TEST_IMAGE_DIR = 'data/test'

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
        conv = conv.squeeze(2).permute(0, 2, 1)  # B, T, C
        rnn_out, _ = self.rnn(conv)
        return torch.log_softmax(self.fc(rnn_out), dim=2)

def load_crnn(path, device):
    ckpt = torch.load(path, map_location=device)
    idx_to_char = {v: k for k, v in ckpt['chars'].items()}
    model = CRNN(num_classes=len(ckpt['chars'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, idx_to_char

def decode(logits, idx_to_char):
    """Greedy CTC decode."""
    pred = torch.argmax(logits[0], dim=1).cpu().numpy()
    text, prev = [], -1
    for p in pred:
        if p != prev and p != 0:  # 0 = CTC blank
            text.append(idx_to_char.get(int(p), ''))
        prev = p
    return ''.join(text)

def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load models
    yolo = YOLO(YOLO_WEIGHTS)
    crnn, idx_to_char = load_crnn(CRNN_WEIGHTS, device)

    # Load test labels
    if not os.path.exists(TEST_CSV):
        print(f"❌ Not found: {TEST_CSV}")
        return
    test_df = pd.read_csv(TEST_CSV)

    predictions = []
    ground_truths = []

    print(f"Evaluating {len(test_df)} test samples...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred_text = ""
        gt_text = str(row['text'])

        # Convert crop filename back to original image filename
        plate_file = str(row['filename'])
        orig_name = plate_file.replace('_plate.jpg', '.jpg').replace('_plate.png', '.png')
        img_path = os.path.join(TEST_IMAGE_DIR, orig_name)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)

            if img is not None:
                # YOLO Detection
                results = yolo(img, verbose=False)[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    confs = results.boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)

                    # Safe crop
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    plate_img = img[y1:y2, x1:x2]

                    if plate_img.size > 0:
                        # EXACT preprocessing used in your good run
                        processed = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                        processed = cv2.resize(processed, (128, 32))

                        x = processed.astype(np.float32) / 255.0
                        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = crnn(x)

                        pred_text = decode(logits, idx_to_char)

        predictions.append(pred_text)
        ground_truths.append(gt_text)

    # Metrics
    n = len(predictions)
    if n == 0:
        print("❌ No samples evaluated.")
        return

    exact = sum(p == g for p, g in zip(predictions, ground_truths))
    total_chars = sum(len(g) for g in ground_truths)
    edits = sum(editdistance.eval(p, g) for p, g in zip(predictions, ground_truths))

    exact_acc = (exact / n) * 100
    cer = (edits / total_chars) * 100 if total_chars > 0 else 0.0
    wer = ((n - exact) / n) * 100

    print("\n" + "=" * 50)
    print(f"Exact Match Accuracy: {exact_acc:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print("=" * 50)

    pd.DataFrame({
        'predicted': predictions,
        'ground_truth': ground_truths,
        'correct': [p == g for p, g in zip(predictions, ground_truths)]
    }).to_csv('evaluation_results.csv', index=False)

    print("✅ Saved: evaluation_results.csv")

if __name__ == "__main__":
    evaluate()
