import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import editdistance

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
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)  # B, T, C
        rnn_out, _ = self.rnn(conv)
        return torch.log_softmax(self.fc(rnn_out), dim=2)

def load_crnn(path, device):
    ckpt = torch.load(path, map_location=device)
    idx_to_char = {v: k for k, v in ckpt['chars'].items()}  # index -> char
    model = CRNN(num_classes=len(ckpt['chars'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, idx_to_char

def decode(logits, idx_to_char):
    pred = torch.argmax(logits[0], dim=1).cpu().numpy()
    text, prev = [], -1
    for p in pred:
        if p != prev and p != 0:  # CTC blank = 0
            text.append(idx_to_char.get(int(p), ''))
        prev = p
    return ''.join(text)

def build_filename_index(search_roots):
    """
    Build map: basename -> full path
    If duplicates exist, first one wins (prints warning).
    """
    index = {}
    duplicates = 0
    total_files = 0

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    total_files += 1
                    full_path = os.path.join(dirpath, fn)
                    if fn not in index:
                        index[fn] = full_path
                    else:
                        duplicates += 1

    print(f"Indexed {len(index)} unique image basenames from {total_files} files.")
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate basenames found (kept first occurrence).")
    return index

def evaluate_ocr_only():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load CRNN
    crnn, idx_to_char = load_crnn('models/crnn/crnn_best.pt', device)

    # Load labels
    csv_path = 'data/ocr/test_labels.csv'
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    test_df = pd.read_csv(csv_path)
    print(f"Evaluating OCR-only on {len(test_df)} GT crops...")
    print("CSV columns:", list(test_df.columns))

    # Search likely roots (add/remove if needed)
    search_roots = [
        'data/ocr/test',
        'data/ocr/validation',
        'data/ocr/train',
        'data/ocr',
        'data'
    ]
    print("Searching image files in roots:", search_roots)
    file_index = build_filename_index(search_roots)

    predictions, gts = [], []
    found_count = 0
    missing_count = 0
    infer_count = 0

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = str(row['filename']).strip()
        gt = str(row['text'])
        pred_text = ""

        crop_path = file_index.get(filename, None)

        if i < 5:
            print(f"[DEBUG] filename={filename}")
            print(f"[DEBUG] matched_path={crop_path}")

        if crop_path is None:
            missing_count += 1
            predictions.append(pred_text)
            gts.append(gt)
            continue

        found_count += 1

        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing_count += 1
            predictions.append(pred_text)
            gts.append(gt)
            continue

        img = cv2.resize(img, (128, 32))
        x = img.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = crnn(x)

        pred_text = decode(logits, idx_to_char)
        infer_count += 1

        predictions.append(pred_text)
        gts.append(gt)

    print(f"\nFound crops: {found_count}")
    print(f"Successful OCR inferences: {infer_count}")
    print(f"Missing/failed crops: {missing_count}")

    total = len(predictions)
    if total == 0:
        print("❌ No samples evaluated.")
        return

    exact = sum(p == g for p, g in zip(predictions, gts))
    total_chars = sum(len(g) for g in gts)
    edits = sum(editdistance.eval(p, g) for p, g in zip(predictions, gts))

    exact_acc = (exact / total) * 100
    cer = (edits / total_chars) * 100 if total_chars > 0 else 0.0
    wer = ((total - exact) / total) * 100

    print("\n" + "=" * 50)
    print(f"OCR-only Exact Match Accuracy: {exact_acc:.2f}%")
    print(f"OCR-only CER: {cer:.2f}%")
    print(f"OCR-only WER: {wer:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_ocr_only()
