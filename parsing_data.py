import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
DATA_ROOT = "data"
SPLITS = ["train", "validation", "test"]
PLATE_LABEL = "کل ناحیه پلاک"
OCR_ROOT = os.path.join(DATA_ROOT, "ocr")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Persian + Arabic digits -> English
DIGIT_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

# Arabic forms -> Persian canonical
CHAR_MAP = {
    "ي": "ی",
    "ك": "ک",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
}

# Tokens that should become single Alef class
ALEF_WORD_TOKENS = {"الف", "آلف", "ألف", "إلف"}  # robust variants


# =====================================================
# HELPERS
# =====================================================
def canon(s: str) -> str:
    """Canonical string for robust comparisons."""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(" ", "").replace("\u200c", "").replace("\ufeff", "")
    for a, b in CHAR_MAP.items():
        s = s.replace(a, b)
    return s


PLATE_LABEL_CANON = canon(PLATE_LABEL)


def normalize_label_token(raw: str) -> str:
    """
    Normalize one XML object label into one OCR token.
    Returns '' if token should be ignored.
    """
    t = canon(raw)

    # map Alef word token to single-char Alef
    if t in ALEF_WORD_TOKENS:
        return "ا"

    # map digits to English
    t = "".join(DIGIT_MAP.get(c, c) for c in t)

    return t


def parse_bbox(obj):
    b = obj.find("bndbox")
    if b is None:
        return None
    try:
        xmin = int(float(b.findtext("xmin", "0")))
        ymin = int(float(b.findtext("ymin", "0")))
        xmax = int(float(b.findtext("xmax", "0")))
        ymax = int(float(b.findtext("ymax", "0")))
        return xmin, ymin, xmax, ymax
    except Exception:
        return None


def center_of(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def index_images(split_root):
    by_filename = {}
    by_stem = {}
    for p in Path(split_root).rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            if p.name not in by_filename:
                by_filename[p.name] = str(p)
            if p.stem not in by_stem:
                by_stem[p.stem] = str(p)
    return by_filename, by_stem


def find_image_for_xml(xml_path, xml_root, by_filename, by_stem):
    xml_fname = (xml_root.findtext("filename", default="") or "").strip()
    if xml_fname and xml_fname in by_filename:
        return by_filename[xml_fname]

    stem = Path(xml_path).stem
    if stem in by_stem:
        return by_stem[stem]

    return None


# =====================================================
# MAIN
# =====================================================
def process_split(split):
    split_root = os.path.join(DATA_ROOT, split)
    out_csv = os.path.join(OCR_ROOT, f"{split}_labels.csv")

    if not os.path.isdir(split_root):
        print(f"❌ Missing split folder: {split_root}")
        return 0

    xml_files = sorted(Path(split_root).rglob("*.xml"))
    if not xml_files:
        print(f"⚠️ No XML files found in: {split_root}")
        return 0

    by_filename, by_stem = index_images(split_root)

    rows = []
    parse_fail = 0
    alef_rows = 0

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            parse_fail += 1
            continue

        img_path = find_image_for_xml(xml_path, root, by_filename, by_stem)
        if img_path is None:
            continue

        stem = Path(img_path).stem

        # plate objects
        plates = []
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", default="")
            if canon(raw_name) == PLATE_LABEL_CANON:
                box = parse_bbox(obj)
                if box is not None:
                    plates.append({"bbox": box, "chars": []})

        if not plates:
            continue

        plates.sort(key=lambda p: p["bbox"][0])

        # char objects
        chars = []
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", default="")
            name_can = canon(raw_name)

            if name_can == PLATE_LABEL_CANON:
                continue

            tok = normalize_label_token(raw_name)

            # keep only single-char OCR tokens
            if len(tok) != 1:
                continue

            box = parse_bbox(obj)
            if box is None:
                continue

            cx, cy = center_of(box)
            chars.append({"ch": tok, "bbox": box, "cx": cx, "cy": cy})

        # assign chars to corresponding plate
        for c in chars:
            for p in plates:
                if point_in_box(c["cx"], c["cy"], p["bbox"]):
                    p["chars"].append(c)
                    break

        for i, p in enumerate(plates):
            if not p["chars"]:
                continue

            p["chars"].sort(key=lambda t: t["cx"])
            text = "".join(t["ch"] for t in p["chars"]).strip()
            if not text:
                continue

            if "ا" in text:
                alef_rows += 1

            crop_name = f"{stem}_p{i}.jpg"
            rows.append({"filename": crop_name, "text": text})

    os.makedirs(OCR_ROOT, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "text"])
        w.writeheader()
        w.writerows(rows)

    print(
        f"✅ {split}: rows={len(rows)} | alef_rows={alef_rows} | "
        f"xml={len(xml_files)} | parse_fail={parse_fail} -> {out_csv}"
    )
    return len(rows)


def main():
    total = 0
    for split in SPLITS:
        total += process_split(split)
    print(f"\n🎉 Done. Total OCR rows: {total}")


if __name__ == "__main__":
    main()
