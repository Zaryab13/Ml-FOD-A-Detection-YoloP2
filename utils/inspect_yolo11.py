from ultralytics import YOLO

model = YOLO("models/yolo11m.pt")
# Print the model architecture
print(model.model)
