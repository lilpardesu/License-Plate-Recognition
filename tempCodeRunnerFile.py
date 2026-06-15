import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    plate_box = None
    chars = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        
        if name == "کل ناحیه پلاک":  # Full plate box
            plate_box = {
                'xmin': xmin,
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
        else:  # Individual character
            chars.append((xmin, name))
    
    # Sort chars left-to-right
    chars.sort(key=lambda x: x[0])
    text = ''.join([c[1] for c in chars])
    
    return plate_box, text

def process_split(split_name):
    input_dir = f"data/{split_name}"
    if not os.path.exists(input_dir):
        print(f"❌ Folder not found: {input_dir}")
        return 0
    
    os.makedirs('data/processed', exist_ok=True)
    
    ocr_data = []
    xml_files = list(Path(input_dir).glob("*.xml"))
    print(f"📁 Processing {split_name}: {len(xml_files)} XML files found")
    
    for xml_path in xml_files:
        try:
            plate_box, text = parse_xml(xml_path)
            if plate_box and text:
                # Find matching image (jpg or png)
                img_name = xml_path.with_suffix('.jpg').name
                img_path = xml_path.with_suffix('.jpg')
                if not img_path.exists():
                    img_path = xml_path.with_suffix('.png')
                    img_name = xml_path.with_suffix('.png').name
                
                if img_path.exists():
                    ocr_data.append({
                        'filename': img_name,
                        'text': text,
                        'xmin': plate_box['xmin'],
                        'ymin': plate_box['ymin'],
                        'xmax': plate_box['xmax'],
                        'ymax': plate_box['ymax']
                    })
        except Exception as e:
            print(f"   Error parsing {xml_path}: {e}")
    
    if ocr_data:
        df = pd.DataFrame(ocr_data)
        csv_path = f"data/processed/{split_name}_ocr_labels.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Created: {csv_path} ({len(ocr_data)} plates)")
        return len(ocr_data)
    return 0

# Run for all splits
total = 0
for split in ['train', 'validation', 'test']:
    count = process_split(split)
    total += count

print(f"\n🎉 Total plates parsed: {total}")
if total > 0:
    print("👉 Now run: python extract_plates.py")
else:
    print("❌ Check that data/train, data/validation, data/test exist with XML files")
