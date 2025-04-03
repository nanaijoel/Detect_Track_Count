from ultralytics import YOLO

# Load model
model = YOLO("train3/weights/best.pt")

# Export ONNX model for OpenCV
model.export(format="onnx")

# bounding boxes drawn with labelImg - type labelIMG in python terminal console in here