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

def ctc_decode_tokens(logits, idx_to_char):
    """CTC greedy decode, returns token list and joined raw string."""
    pred = torch.argmax(logits[0], dim=1).cpu().numpy()
    tokens, prev = [], -1
    for p in pred:
        if p != prev and p != 0:  # 0 = CTC blank
            tokens.append(idx_to_char.get(int(p), ""))
        prev = p
    raw = "".join(tokens)
    return tokens, raw

# ----------------- Helpers -----------------
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
DIGIT_MAP = {**{p: e for p, e in zip(PERSIAN_DIGITS, EN_DIGITS)},
             **{a: e for a, e in zip(ARABIC_DIGITS, EN_DIGITS)}}

# common IR plate letters (plus multi-char token "الف")
VALID_LETTERS = {
    "الف", "ب", "ج", "د", "س", "ص", "ط", "ق", "ل", "م",
    "ن", "و", "ه", "ی", "ک", "ع", "پ", "ت"
}

def to_en_digit(ch: str) -> str:
    return DIGIT_MAP.get(ch, ch)

def is_any_digit(ch: str) -> bool:
    c = to_en_digit(ch)
    return c.isdigit()

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().replace(" ", "")
    s = "".join(to_en_digit(ch) for ch in s)
    return s

def extract_letter_and_digits(raw_text: str):
    """
    Robust extraction:
    - prioritizes 'الف' as one letter token
    - otherwise picks first valid non-digit letter
    """
    s = normalize_text(raw_text)

    # 1) explicit 'الف'
    if "الف" in s:
        s_wo = s.replace("الف", "", 1)
        digits = "".join(ch for ch in s_wo if ch.isdigit())
        return "الف", digits

    # 2) other letters
    letter = ""
    for ch in s:
        if not ch.isdigit() and ch in VALID_LETTERS:
            letter = ch
            break

    # remove first occurrence of selected letter from s
    s_wo = s
    if letter:
        idx = s_wo.find(letter)
        if idx != -1:
            s_wo = s_wo[:idx] + s_wo[idx + len(letter):]

    digits = "".join(ch for ch in s_wo if ch.isdigit())
    return letter, digits

def format_plate_segments(raw_text: str):
    """
    Target display: [2 digits] [letter] [3 digits] [2 digits]
    For imperfect OCR (e.g., 6 digits), keep best-effort but preserve letter.
    """
    letter, digits = extract_letter_and_digits(raw_text)

    # perfect case
    if len(digits) >= 7:
        d = digits[:7]
        return d[:2], letter if letter else "?", d[2:5], d[5:7]

    # best-effort for missing digits
    if len(digits) >= 5:
        first2 = digits[:2]
        last2 = digits[-2:]
        mid = digits[2:-2]  # may be 1..3 chars
        return first2, letter if letter else "?", mid, last2

    if len(digits) >= 2:
        return digits[:2], letter if letter else "?", digits[2:], ""

    # fallback
    return raw_text, "", "", ""

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
    chars = ckpt["chars"]  # token -> idx
    idx_to_char = {v: k for k, v in chars.items()}

    crnn = CRNN(num_classes=len(chars)).to(device)
    crnn.load_state_dict(ckpt["model"])
    crnn.eval()

    return yolo, crnn, idx_to_char, device

# ----------------- Multi-plate inference -----------------
def run_lpr_multi(img_bgr, yolo, crnn, idx_to_char, device, conf_thresh=0.25):
    results = yolo(img_bgr, verbose=False)[0]
    vis = img_bgr.copy()

    if results.boxes is None or len(results.boxes) == 0:
        return vis, []

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    valid_idxs = [i for i, c in enumerate(confs) if float(c) >= conf_thresh]
    if len(valid_idxs) == 0:
        return vis, []

    valid_idxs.sort(key=lambda i: boxes[i][0])  # left-to-right

    outputs = []
    plate_id = 1

    for i in valid_idxs:
        x1, y1, x2, y2 = boxes[i]
        conf = float(confs[i])

        plate = safe_crop(img_bgr, x1, y1, x2, y2)
        if plate is None:
            continue

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        pre = cv2.resize(gray, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)

        x = pre.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = crnn(x)

        tokens, raw_text = ctc_decode_tokens(logits, idx_to_char)
        s1, s2, s3, s4 = format_plate_segments(raw_text)
        formatted = " ".join([p for p in [s1, s2, s3, s4] if p != ""]).strip()

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"#{plate_id}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        outputs.append({
            "id": plate_id,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "tokens": tokens,
            "raw_text": raw_text,
            "seg1": s1,
            "seg2": s2,
            "seg3": s3,
            "seg4": s4,
            "formatted": formatted,
            "crop": plate,
            "pre": pre
        })
        plate_id += 1

    return vis, outputs

# ----------------- UI -----------------
st.set_page_config(page_title="LPR Visual Demo (Multi-Plate)", layout="wide")
st.title("🚘 License Plate Recognition (Multi-Plate Visual Demo)")
st.write("Upload image → detect all plates → OCR each plate")

try:
    yolo, crnn, idx_to_char, device = load_models()
    st.success(f"Models loaded on: {device}")
except Exception as e:
    st.error(str(e))
    st.stop()

conf_thresh = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)

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
        vis, outputs = run_lpr_multi(img_bgr, yolo, crnn, idx_to_char, device, conf_thresh=conf_thresh)

        if len(outputs) == 0:
            st.warning("No plates detected above threshold.")
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)
        else:
            st.subheader(f"Detected plates: {len(outputs)}")
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="All detected plates", use_container_width=True)

            for out in outputs:
                st.markdown(f"### Plate #{out['id']}")

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        gap:12px;
                        align-items:center;
                        font-size:36px;
                        font-weight:700;
                        font-family: Arial, Tahoma, sans-serif;
                        background:#0f172a;
                        color:#86efac;
                        padding:10px 14px;
                        border-radius:10px;
                        width:fit-content;
                        margin-bottom:8px;
                    ">
                        <span dir="ltr">{out['seg1']}</span>
                        <span dir="rtl">{out['seg2']}</span>
                        <span dir="ltr">{out['seg3']}</span>
                        <span dir="ltr">{out['seg4']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(f"Raw OCR: {out['raw_text']}")
                st.caption(f"Tokens: {out['tokens']}")
                st.write(f"**Detection confidence:** `{out['conf']:.4f}`")
                x1, y1, x2, y2 = out["bbox"]
                st.write(f"**BBox:** `({x1}, {y1}, {x2}, {y2})`")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(cv2.cvtColor(out["crop"], cv2.COLOR_BGR2RGB), caption=f"Plate #{out['id']} crop", use_container_width=True)
                with c2:
                    st.image(out["pre"], caption=f"Plate #{out['id']} OCR input (128x32)", use_container_width=True)

                st.divider()
