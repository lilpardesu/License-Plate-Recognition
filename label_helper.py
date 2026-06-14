import os
import pandas as pd
import shutil

def label_plates():
    results_dir = 'data/results'
    csv_file = 'labels.csv'
    

    start_index = 1032

    if not os.path.exists(results_dir):
        print("Folder not found:", results_dir)
        return


    all_images = sorted([
        f for f in os.listdir(results_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    if start_index >= len(all_images):
        print(f"Index {start_index} is out of range. Total: {len(all_images)}")
        return

    remaining = all_images[start_index:]

    print(f"Total images: {len(all_images)}")
    print(f"Starting tomorrow from index: {start_index} (File: {remaining[0]})")

    for i, img_file in enumerate(remaining, start=start_index + 1):
        img_path = os.path.abspath(os.path.join(results_dir, img_file))

        os.system(f'start "" "{img_path}"')

        prompt = f"[{i}/{len(all_images)}] Plate for {img_file} (skip/quit): "
        text = input(prompt).strip()

        if text.lower() == "quit":
            break
        if text.lower() == "skip":
            continue


        save_to_csv_safe([{"filename": img_file, "text": text.replace(" ", "")}], csv_file)
        print(f"Saved: {text}")

def save_to_csv_safe(new_labels, csv_file):

    if os.path.exists(csv_file):
        shutil.copy(csv_file, csv_file + ".bak")

    df_new = pd.DataFrame(new_labels)
    if os.path.exists(csv_file):
        df_old = None
        for enc in ['utf-8-sig', 'cp1252', 'latin-1']:
            try:
                df_old = pd.read_csv(csv_file, encoding=enc)
                break
            except:
                continue
        
        if df_old is not None:
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=['filename'], keep='last')
        else:
            df = df_new
    else:
        df = df_new

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    label_plates()
