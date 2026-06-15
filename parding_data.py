import os
import xml.etree.ElementTree as ET
import cv2
import shutil

def convert_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    return (x * dw, y * dh, w * dw, h * dh)


def process_labels(source_base="data", target_base="data/processed"):

    for split in ["train", "validation", "test"]:

        img_target = f"{target_base}/{split}/images"
        lbl_target = f"{target_base}/{split}/labels"

        os.makedirs(img_target, exist_ok=True)
        os.makedirs(lbl_target, exist_ok=True)

        search_path = f"{source_base}/{split}"

        print(f"Processing {split} from {search_path}")

        for file in os.listdir(search_path):

            if not file.endswith(".xml"):
                continue

            xml_path = os.path.join(search_path, file)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_file = file.replace(".xml", ".jpg")
            img_path = os.path.join(search_path, img_file)

            if not os.path.exists(img_path):
                img_file = file.replace(".xml", ".png")
                img_path = os.path.join(search_path, img_file)

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            yolo_data = []

            for obj in root.findall("object"):

                name = obj.find("name").text

                if name == "کل ناحیه پلاک":

                    bndbox = obj.find("bndbox")

                    xmin = float(bndbox.find("xmin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymin = float(bndbox.find("ymin").text)
                    ymax = float(bndbox.find("ymax").text)

                    bb = convert_to_yolo((w, h), (xmin, xmax, ymin, ymax))

                    yolo_data.append(
                        f"0 {' '.join([f'{a:.6f}' for a in bb])}"
                    )

            if yolo_data:

                shutil.copy(img_path, os.path.join(img_target, img_file))

                with open(
                    os.path.join(lbl_target, file.replace(".xml", ".txt")),
                    "w"
                ) as f:

                    f.write("\n".join(yolo_data))

    print("✅ Conversion finished")


if __name__ == "__main__":
    process_labels()
