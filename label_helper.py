import os
import cv2
import pandas as pd

def label_plates():
    results_dir = 'data/results'
    csv_file = 'labels.csv'
    
    # Load existing labels
    labeled = set()
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        labeled = set(df['filename'].tolist())
        print(f"Resuming: {len(labeled)} images already labeled")
    
    # Get remaining images
    all_images = sorted([f for f in os.listdir(results_dir) if f.endswith('.jpg')])
    remaining = [f for f in all_images if f not in labeled]
    
    if not remaining:
        print("All images labeled!")
        return
    
    print(f"\nLabeling {len(remaining)} remaining images...")
    print("Commands: type plate text | 'skip' to skip | 'quit' to save and exit\n")
    
    new_labels = []
    
    for i, img_file in enumerate(remaining, 1):
        img_path = os.path.join(results_dir, img_file)
        img = cv2.imread(img_path)
        
        # Show image
        cv2.imshow(f'Labeling ({i}/{len(remaining)})', img)
        key = cv2.waitKey(1)  # Required to refresh window
        
        # Get user input
        user_input = input(f"[{i}/{len(remaining)}] {img_file}: ").strip()
        
        # Handle commands
        if user_input.lower() == 'quit':
            print("Saving and quitting...")
            break
            
        elif user_input.lower() == 'skip':
            print(f"  -> Skipped {img_file}")
            cv2.destroyAllWindows()
            continue  # Move to next image
            
        elif user_input:  # If not empty
            # Clean: remove spaces, keep Persian letters and digits
            clean_text = user_input.replace(' ', '')
            new_labels.append({'filename': img_file, 'text': clean_text})
            print(f"  -> Saved: {clean_text}")
            
            # Save every 5 labels to prevent data loss
            if len(new_labels) % 5 == 0:
                save_to_csv(new_labels, csv_file, labeled)
                print(f"  (Auto-saved {len(new_labels)} new labels)")
        
        cv2.destroyAllWindows()
    
    # Final save
    if new_labels:
        save_to_csv(new_labels, csv_file, labeled)
        total = len(labeled) + len(new_labels)
        print(f"\nDone! Total labeled: {total}")
    
    cv2.destroyAllWindows()

def save_to_csv(new_labels, csv_file, existing_set):
    """Append new labels to existing CSV"""
    df_new = pd.DataFrame(new_labels)
    
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_file, index=False)

if __name__ == '__main__':
    try:
        label_plates()
    except KeyboardInterrupt:
        print("\nInterrupted! Progress saved.")
        cv2.destroyAllWindows()
