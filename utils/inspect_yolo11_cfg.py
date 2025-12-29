from ultralytics import YOLO

model = YOLO("models/yolo11m.pt")
# Print the model configuration
print(model.cfg)
# Or try to access the yaml directly if available
try:
    print(model.model.yaml)
except:
    print("model.model.yaml not found")
