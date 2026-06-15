import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os

def extract_plates():
    # Load from the actual training output folder
    model = YOLO('ir_plate_detector/weights/best.pt')
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        csv_path = f'data/processed/{split}_ocr_labels.csv'
        if not os.path.exists(csv_path):
            print(f"Skipping {split}: CSV not found")
            continue
            
        df = pd.read_csv(csv_path)
        output_dir = f'data/ocr/{split}/plates'
        os.makedirs(output_dir, exist_ok=True)
        
        ocr_labels = []
        
        for _, row in df.iterrows():
            img_path = f'data/processed/{split}/images/{row["filename"]}'
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # Get coordinates from CSV
            xmin = max(0, int(row['xmin']) - 2)
            ymin = max(0, int(row['ymin']) - 2)
            xmax = min(w, int(row['xmax']) + 2)
            ymax = min(h, int(row['ymax']) + 2)
            
            plate_img = img[ymin:ymax, xmin:xmax]
            
            if plate_img.size == 0:
                continue
            
            plate_name = f"{Path(row['filename']).stem}_plate.jpg"
            plate_path = os.path.join(output_dir, plate_name)
            cv2.imwrite(plate_path, plate_img)
            
            ocr_labels.append({
                'filename': plate_name,
                'text': str(row['text'])
            })
        
        ocr_df = pd.DataFrame(ocr_labels)
        ocr_df.to_csv(f'data/ocr/{split}_labels.csv', index=False)
        print(f"{split}: {len(ocr_labels)} plates extracted")

if __name__ == "__main__":
    extract_plates()
    print("\nDone! Cropped plates saved to data/ocr/")
