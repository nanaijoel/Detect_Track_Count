import cv2
from ultralytics import YOLO


model = YOLO("C:/Users/j.nanai/PycharmProjects/00_YOLO_Model_Sweets/best.onnx")

img = cv2.imread("IMG-20250303-WA0007.jpg")
results = model(img)

for r in results:
    annotated_frame = r.plot()  # Draw bounding boxes

    resized_frame = cv2.resize(annotated_frame, (800, 600), interpolation=cv2.INTER_AREA)

    cv2.imshow("YOLOv8 Result", resized_frame)  # OpenCV Window
    cv2.waitKey(0)

cv2.destroyAllWindows()
