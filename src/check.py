from ultralytics import YOLO
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_filenames(directory):
    image_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            full_path = os.path.join(directory, filename)
            image_filenames.append(full_path)
    return [s[21:] for s in image_filenames]

image_directory = '../data/valid/images'

model = YOLO('./runs/detect/train/weights/best.pt')
results = model(image_directory,imgsz=640,save=True,device=0)

for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs 