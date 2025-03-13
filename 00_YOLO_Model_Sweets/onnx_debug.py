import torch
import onnx
import onnxruntime as ort

# Load ONNX Model
model_path = "best.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

# check if onnx model is working
print("ONNX Modell erfolgreich geladen und überprüft!")

# OnnxRuntime test
session = ort.InferenceSession(model_path)
print("ONNX Session erfolgreich gestartet!")
