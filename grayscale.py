import cv2
import os
from pathlib import Path

def convert_plates_to_grayscale():
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        input_dir = f'data/ocr/{split}/plates'
        output_dir = f'data/ocr/{split}/plates_gray'
        
        if not os.path.exists(input_dir):
            print(f"⚠️  Skipping {split}: {input_dir} not found")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        image_files = list(Path(input_dir).glob('*.jpg'))
        print(f"🔄 Processing {split}: {len(image_files)} images...")
        
        converted = 0
        for img_path in image_files:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ❌ Failed to read: {img_path.name}")
                continue
            
            # Convert to grayscale (Professor requirement)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Save
            output_path = os.path.join(output_dir, img_path.name)
            cv2.imwrite(output_path, gray)
            converted += 1
        
        print(f"✅ {split}: {converted} images converted to grayscale")
        print(f"   Saved to: {output_dir}")
    
    print("\n🎉 Grayscale conversion complete!")
    print("Next step: python train_crnn.py")

if __name__ == "__main__":
    convert_plates_to_grayscale()
