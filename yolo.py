from ultralytics import YOLO

model = YOLO("yolo11x.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=320,
    batch=12,
    name="hail_yolo_v1",
    project="runs"
)
