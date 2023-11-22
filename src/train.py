import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('./runs/detect/yolov8n_15e4/weights/last.pt')
model.to(device)

results = model.train(
    data='../data/data.yaml',
    imgsz=640,
    epochs=30,
    batch=16,
    degrees=15,
    scale=0.9,
    fliplr=1
)
