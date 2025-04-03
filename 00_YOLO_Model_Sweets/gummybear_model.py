from ultralytics import YOLO

# Load pretrained YOLOv8-Model
model = YOLO("yolov8n.pt")

# Start training
model.train(data="gummybears/config.yaml", epochs=50, imgsz=640)
