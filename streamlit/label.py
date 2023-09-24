import torch

model_path = "paling best.pt"
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=model_path, force_reload=True
)

labels = model.names

print(labels)
