from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=15,
    project="runs",
    name="ir_plate_detector",
    exist_ok=True
)
