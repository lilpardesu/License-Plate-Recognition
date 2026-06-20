import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
DATA_ROOT = "data"
SPLITS = ["train", "validation", "test"]

PLATE_LABEL = "کل ناحیه پلاک"

# output:
# data/ocr/{split}/plates
# data/ocr/{split}/plates_gray
OCR_ROOT = os.path.join(DATA_ROOT, "ocr")

# small padding around plate bbox
PAD_X_RATIO = 0.03
PAD_Y_RATIO = 0.08

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================
# HELPERS
# =====================================================
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


def safe_crop_with_padding(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * PAD_X_RATIO)
    py = int(bh * PAD_Y_RATIO)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def index_images(split_root):
    """
    Build:
      1) by exact filename (e.g., 10030.jpg)
      2) by stem (e.g., 10030) fallback
    """
    by_filename = {}
    by_stem = {}

    for p in Path(split_root).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_IMAGE_EXTS:
            continue

        fname = p.name
        stem = p.stem

        # keep first if duplicates
        if fname not in by_filename:
            by_filename[fname] = str(p)
        if stem not in by_stem:
            by_stem[stem] = str(p)

    return by_filename, by_stem


def find_image_for_xml(xml_path, xml_root, by_filename, by_stem):
    # try <filename> tag
    xml_fname = (xml_root.findtext("filename", default="") or "").strip()
    if xml_fname and xml_fname in by_filename:
        return by_filename[xml_fname]

    # fallback xml stem
    stem = Path(xml_path).stem
    if stem in by_stem:
        return by_stem[stem]

    return None


# =====================================================
# MAIN
# =====================================================
def process_split(split):
    split_root = os.path.join(DATA_ROOT, split)

    if not os.path.isdir(split_root):
        print(f"❌ Missing split folder: {split_root}")
        return 0

    out_color_dir = os.path.join(OCR_ROOT, split, "plates")
    out_gray_dir = os.path.join(OCR_ROOT, split, "plates_gray")
    os.makedirs(out_color_dir, exist_ok=True)
    os.makedirs(out_gray_dir, exist_ok=True)

    xml_files = sorted(Path(split_root).rglob("*.xml"))
    if len(xml_files) == 0:
        print(f"⚠️ No XML files found in: {split_root}")
        return 0

    by_filename, by_stem = index_images(split_root)

    saved = 0
    missed_image = 0
    parse_fail = 0

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            parse_fail += 1
            continue

        img_path = find_image_for_xml(xml_path, root, by_filename, by_stem)
        if img_path is None:
            missed_image += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            missed_image += 1
            continue

        # collect all plate boxes
        plate_boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name", default="") or "").strip()
            if name == PLATE_LABEL:
                box = parse_bbox(obj)
                if box is not None:
                    plate_boxes.append(box)

        if not plate_boxes:
            continue

        # left->right ordering for deterministic p0,p1...
        plate_boxes.sort(key=lambda b: b[0])

        stem = Path(img_path).stem
        for i, box in enumerate(plate_boxes):
            crop = safe_crop_with_padding(img, box)
            if crop is None:
                continue

            out_name = f"{stem}_p{i}.jpg"

            color_path = os.path.join(out_color_dir, out_name)
            gray_path = os.path.join(out_gray_dir, out_name)

            cv2.imwrite(color_path, crop)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(gray_path, gray)
            saved += 1

    print(f"✅ {split}: saved={saved} | xml={len(xml_files)} | parse_fail={parse_fail} | missed_image={missed_image}")
    return saved


def main():
    os.makedirs(OCR_ROOT, exist_ok=True)

    total = 0
    for split in SPLITS:
        total += process_split(split)

    print(f"\n🎉 Done. Total saved crops: {total}")


if __name__ == "__main__":
    main()
