import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =========================================================
# PATHS + LAYOUT DETECTION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def first_existing_dir(cands):
    for p in cands:
        if p.exists() and p.is_dir():
            return p
    return None


def detect_layout():
    # OCR pipeline
    ocr_train_csv = DATA_DIR / "ocr" / "train_labels.csv"
    ocr_val_csv = DATA_DIR / "ocr" / "validation_labels.csv"
    ocr_train_img = first_existing_dir([
        DATA_DIR / "ocr" / "train" / "plates_gray",
        DATA_DIR / "ocr" / "train" / "plates",
        DATA_DIR / "ocr" / "train",
    ])
    ocr_val_img = first_existing_dir([
        DATA_DIR / "ocr" / "validation" / "plates_gray",
        DATA_DIR / "ocr" / "validation" / "plates",
        DATA_DIR / "ocr" / "validation",
    ])

    ocr_ok = ocr_train_csv.exists() and ocr_val_csv.exists() and ocr_train_img and ocr_val_img


    if ocr_ok:
        return {
            "name": "OCR",
            "train_csv": ocr_train_csv,
            "val_csv": ocr_val_csv,
            "train_img_root": ocr_train_img,
            "val_img_root": ocr_val_img,
        }

    raise FileNotFoundError(
        "No complete layout found.\n"
        "Need:\n"
        " data/ocr/train_labels.csv + data/ocr/train/(plates_gray|plates|...)\n"
    )


LAYOUT = detect_layout()


# =========================================================
# CONFIG
# =========================================================
class Config:
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_CSV = LAYOUT["train_csv"]
    VAL_CSV = LAYOUT["val_csv"]
    TRAIN_IMG_ROOT = LAYOUT["train_img_root"]
    VAL_IMG_ROOT = LAYOUT["val_img_root"]

    SAVE_DIR = BASE_DIR / "models" / "crnn"
    RESUME_PATH = SAVE_DIR / "crnn_best.pt"


os.makedirs(Config.SAVE_DIR, exist_ok=True)

print(f"Selected layout: {LAYOUT['name']}")
print("Using paths:")
print("  TRAIN_CSV      =", Config.TRAIN_CSV)
print("  VAL_CSV        =", Config.VAL_CSV)
print("  TRAIN_IMG_ROOT =", Config.TRAIN_IMG_ROOT)
print("  VAL_IMG_ROOT   =", Config.VAL_IMG_ROOT)
print("  SAVE_DIR       =", Config.SAVE_DIR)
print("  DEVICE         =", Config.DEVICE)


# =========================================================
# CHARSET
# =========================================================
CHARS = ['-'] + [str(i) for i in range(10)] + [
    'ب', 'ج', 'د', 'س', 'ص', 'ط', 'ق', 'ل', 'م', 'ن', 'و', 'ه', 'ی',
    'ا', 'ت', 'پ', 'ث', 'چ', 'ح', 'خ', 'ذ', 'ر', 'ز', 'ژ', 'ش', 'ض',
    'ظ', 'ع', 'غ', 'ف', 'ک', 'گ'
]
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

DIGIT_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_text(s: str) -> str:
    s = str(s).strip().replace(" ", "").replace("\u200c", "").replace("\ufeff", "")
    out = []
    for ch in s:
        ch = DIGIT_MAP.get(ch, ch)
        out.append(ch)
    return "".join(out)


# =========================================================
# DATASET
# =========================================================
def build_image_index(img_root: Path):
    by_name = {}
    by_stem = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            if p.name not in by_name:
                by_name[p.name] = str(p)
            if p.stem not in by_stem:
                by_stem[p.stem] = str(p)
    return by_name, by_stem


def resolve_img_path(filename: str, by_name, by_stem):
    f = str(filename).replace("\\", "/")
    base = os.path.basename(f)
    stem = Path(base).stem
    if base in by_name:
        return by_name[base]
    if stem in by_stem:
        return by_stem[stem]
    return None


class PlateDataset(Dataset):
    def __init__(self, csv_path: Path, img_root: Path):
        df = pd.read_csv(csv_path)

        # flexible column names
        filename_col, text_col = None, None
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ["filename", "file", "image", "img", "name"]:
                filename_col = c
            if cl in ["text", "label", "plate", "target", "ocr"]:
                text_col = c

        if filename_col is None or text_col is None:
            raise ValueError(f"{csv_path} missing filename/text columns. Found: {list(df.columns)}")

        by_name, by_stem = build_image_index(img_root)

        samples = []
        missing = 0
        bad_chars = 0
        missing_examples = []

        for _, row in df.iterrows():
            fname = str(row[filename_col]).strip()
            text = normalize_text(row[text_col])

            if not fname or not text:
                continue

            unknown = [ch for ch in text if ch not in CHAR_TO_IDX]
            if unknown:
                bad_chars += 1
                continue

            img_path = resolve_img_path(fname, by_name, by_stem)
            if img_path is None:
                missing += 1
                if len(missing_examples) < 8:
                    missing_examples.append(fname)
                continue

            samples.append({"img_path": img_path, "text": text})

        self.samples = samples

        print(f"\nDataset: {csv_path.name}")
        print(f"  valid={len(self.samples)} | missing={missing} | bad_chars={bad_chars}")
        if missing_examples:
            print("  missing examples:", missing_examples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        text = item["text"]
        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return img, target, len(target)


def collate_fn(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.cat(targets, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, lengths


# =========================================================
# MODEL
# =========================================================
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
        x = self.cnn(x)                # B,512,1,T
        x = x.squeeze(2).permute(0, 2, 1)  # B,T,512
        x, _ = self.rnn(x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=2)


# =========================================================
# TRAIN
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    n_batches = 0

    for imgs, texts, text_lens in loader:
        imgs = imgs.to(Config.DEVICE)
        texts = texts.to(Config.DEVICE)
        text_lens = text_lens.to(Config.DEVICE)

        out = model(imgs).permute(1, 0, 2)  # T,B,C
        T, B, _ = out.size()
        input_lens = torch.full((B,), T, dtype=torch.long, device=Config.DEVICE)

        loss = criterion(out, texts, input_lens, text_lens)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    train_ds = PlateDataset(Config.TRAIN_CSV, Config.TRAIN_IMG_ROOT)
    val_ds = PlateDataset(Config.VAL_CSV, Config.VAL_IMG_ROOT)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset is empty after matching CSV filenames to image files.")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    if Config.RESUME_PATH.exists():
        ckpt = torch.load(Config.RESUME_PATH, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"\n🔁 Resumed from: {Config.RESUME_PATH}")
    else:
        print("\n🆕 Training from scratch")

    best_val = float("inf")
    print("\n🚀 Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Epoch {epoch:02d}/{Config.EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_path = Config.SAVE_DIR / "crnn_best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "chars": CHAR_TO_IDX,
                    "config": {
                        "IMG_HEIGHT": Config.IMG_HEIGHT,
                        "IMG_WIDTH": Config.IMG_WIDTH,
                        "BATCH_SIZE": Config.BATCH_SIZE,
                        "EPOCHS": Config.EPOCHS,
                        "LR": Config.LR,
                    },
                },
                save_path,
            )
            print(f"💾 Saved best: {save_path}")

    print("\n✅ Done")


if __name__ == "__main__":
    main()
