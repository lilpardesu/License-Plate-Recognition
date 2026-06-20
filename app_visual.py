import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ----------------- Config -----------------
YOLO_WEIGHTS = "ir_plate_detector/weights/best.pt"
CRNN_WEIGHTS = "models/crnn/crnn_best.pt"
OCR_W, OCR_H = 128, 32

# ----------------- CRNN -----------------
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
            512, 256, 2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)                        # [B, C, 1, T]
        conv = conv.squeeze(2).permute(0, 2, 1)  # [B, T, C]
        rnn_out, _ = self.rnn(conv)
        return torch.log_softmax(self.fc(rnn_out), dim=2)

def ctc_decode(logits, idx_to_char):
    pred = torch.argmax(logits[0], dim=1).cpu().numpy()
    text = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:  # 0 = CTC blank
            text.append(idx_to_char.get(int(p), ""))
        prev = p
    return "".join(text)

# ----------------- Helpers -----------------
def is_digit_any(ch: str) -> bool:
    return ch.isdigit() or ch in "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"

def format_plate_segments(raw_text: str):
    """
    Output target:
    [2 digits] [letter] [3 digits] [2 digits]
    Example: 47 ط 738 19
    """
    s = (raw_text or "").strip().replace(" ", "")
    digits = [c for c in s if is_digit_any(c)]
    letters = [c for c in s if (not is_digit_any(c)) and c.strip()]

    if len(digits) >= 7 and len(letters) >= 1:
        d = "".join(digits[:7])
        l = letters[0]
        return d[:2], l, d[2:5], d[5:7]

    # fallback (if OCR output incomplete)
    return s, "", "", ""

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

# ----------------- Load models -----------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
    if not os.path.exists(CRNN_WEIGHTS):
        raise FileNotFoundError(f"CRNN weights not found: {CRNN_WEIGHTS}")

    yolo = YOLO(YOLO_WEIGHTS)

    ckpt = torch.load(CRNN_WEIGHTS, map_location=device)
    chars = ckpt["chars"]  # char -> idx
    idx_to_char = {v: k for k, v in chars.items()}

    crnn = CRNN(num_classes=len(chars)).to(device)
    crnn.load_state_dict(ckpt["model"])
    crnn.eval()

    return yolo, crnn, idx_to_char, device

# ----------------- Inference -----------------
def run_lpr(img_bgr, yolo, crnn, idx_to_char, device):
    results = yolo(img_bgr, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None, None, None, None, "No plate detected"

    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    conf = float(confs[best_idx])

    x1, y1, x2, y2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    plate = safe_crop(img_bgr, x1, y1, x2, y2)
    if plate is None:
        return None, None, None, None, "Invalid crop"

    # OCR preprocessing (your stable version)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    pre = cv2.resize(gray, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)

    x = pre.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = crnn(x)

    raw_text = ctc_decode(logits, idx_to_char)
    seg1, seg2, seg3, seg4 = format_plate_segments(raw_text)

    vis = img_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return vis, plate, pre, conf, (raw_text, seg1, seg2, seg3, seg4)

# ----------------- UI -----------------
st.set_page_config(page_title="LPR Visual Demo", layout="wide")
st.title("🚘 License Plate Recognition (Visual Demo)")
st.write("Upload image → detect plate → OCR")

try:
    yolo, crnn, idx_to_char, device = load_models()
    st.success(f"Models loaded on: {device}")
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read image file.")
        st.stop()

    st.subheader("Original")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Run LPR", type="primary"):
        vis, plate, pre, conf, out = run_lpr(img_bgr, yolo, crnn, idx_to_char, device)

        if vis is None:
            st.warning(out)
        else:
            raw_text, s1, s2, s3, s4 = out

            st.subheader("Prediction")

            # Fixed visual order using flex (prevents RTL/LTR reordering issue)
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    gap:12px;
                    align-items:center;
                    font-size:40px;
                    font-weight:700;
                    font-family: Arial, Tahoma, sans-serif;
                    background:#0f172a;
                    color:#86efac;
                    padding:10px 14px;
                    border-radius:10px;
                    width:fit-content;
                ">
                    <span dir="ltr">{s1}</span>
                    <span dir="rtl">{s2}</span>
                    <span dir="ltr">{s3}</span>
                    <span dir="ltr">{s4}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"**Detection confidence:** `{conf:.4f}`")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detected plate box", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption="Plate crop", use_container_width=True)
            with c3:
                st.image(pre, caption="OCR input (grayscale 128x32)", use_container_width=True)
