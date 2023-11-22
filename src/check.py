from ultralytics import YOLO
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_filenames(directory):
    image_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more extensions if needed
            full_path = os.path.join(directory, filename)
            image_filenames.append(full_path)
    return [s[21:] for s in image_filenames]

# Replace 'path/to/your/images' with the actual path to your image directory
image_directory = '../data/valid/images'
# image_array = get_image_filenames(image_directory)

# Load a model
model = YOLO('./runs/detect/yolov8n_15e4/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(image_directory,imgsz=640,save=True,device=0)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs