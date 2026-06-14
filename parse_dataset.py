import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
from pathlib import Path
import shutil

def parse_xml(xml_path):
    """Parse XML and return plate box + text"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    plate_box = None
    chars = []  # (xmin, char)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        
        if name == "کل ناحیه پلاک":  # Full plate
            plate_box = {
                'xmin': xmin,
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
        else:
            # Individual character
            chars.append((xmin, name))
    
    # Sort characters by x-position (left-to-right)
    chars.sort(key=lambda x: x[0])
    text = ''.join([c[1] for c in chars])
    
    # Convert Western digits to Persian if needed
    # text = text.translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹'))
    
    return plate_box, text

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert to YOLO normalized format"""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def process_split(split_name):
    """Process train/validation/test folders"""
    input_dir = f"data/{split_name}"
    output_img_dir = f"data/processed/{split_name}/images"
    output_label_dir = f"data/processed/{split_name}/labels"
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    ocr_data = []  # For CRNN training
    
    xml_files = list(Path(input_dir).glob("*.xml"))
    
    for xml_path in xml_files:
        img_path = xml_path.with_suffix('.jpg')
        if not img_path.exists():
            img_path = xml_path.with_suffix('.png')
        
        if not img_path.exists():
            continue
            
        # Parse annotations
        plate_box, text = parse_xml(xml_path)
        if not plate_box or not text:
            continue
        
        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Save YOLO label (plate detection)
        x, y, bw, bh = convert_to_yolo_format(
            plate_box['xmin'], plate_box['ymin'], 
            plate_box['xmax'], plate_box['ymax'], w, h
        )
        
        label_file = os.path.join(output_label_dir, f"{xml_path.stem}.txt")
        with open(label_file, 'w') as f:
            f.write(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
        
        # Copy image to processed folder
        shutil.copy(img_path, os.path.join(output_img_dir, img_path.name))
        
        # For OCR: Store metadata (you'll crop later during training or now)
        ocr_data.append({
            'filename': img_path.name,
            'text': text,
            'xmin': plate_box['xmin'],
            'ymin': plate_box['ymin'],
            'xmax': plate_box['xmax'],
            'ymax': plate_box['ymax']
        })
    
    # Save OCR labels
    df = pd.DataFrame(ocr_data)
    df.to_csv(f"data/processed/{split_name}_ocr_labels.csv", index=False)
    print(f"{split_name}: {len(ocr_data)} samples")

# Run for all splits
for split in ['train', 'validation', 'test']:
    if os.path.exists(f"data/{split}"):
        process_split(split)

print("Done! Check data/processed/")
print("YOLO format: images/ + labels/ folders")
print("OCR format: *_ocr_labels.csv files")
