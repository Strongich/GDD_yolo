import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import ast

os.makedirs("uploads", exist_ok=True)

app=FastAPI()
model = YOLO('./runs/detect/yolov8n_15e4/weights/best.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def visualize_detection(image, detections):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_names = ['Open Eye','Closed Eye','Cigarette','Phone','Seatbelt']

    for bbox in detections:
        x_min, y_min, x_max, y_max, prob_class, class_id = bbox
        class_id = int(class_id)
        label = f"{class_names[class_id]}: {prob_class:.2f}"
        color = (0, 0, 255)
        thickness = 1
        font_size = 0.3
        font_thickness = 1
        text_offset = 2

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
        cv2.rectangle(image, (int(x_min), int(y_min - text_size[1] - text_offset)),
                      (int(x_min) + text_size[0], int(y_min)), (0, 0, 255), cv2.FILLED)

        cv2.putText(image, label, (int(x_min), int(y_min - text_offset)), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), font_thickness)

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def read_root():
    return {"message": "Hello, this is the FastAPI backend."}

async def detect_images(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail='No file found')

    try:
        image = Image.open(io.BytesIO(file.file.read()))

        # Ensure that the model and visualize_detection functions return valid results
        results = model(image, conf=0.45)
        if results is None:
            raise HTTPException(status_code=500, detail='Error processing image')

        img = visualize_detection(image, results[0].boxes.data.tolist())
        if img is None:
            raise HTTPException(status_code=500, detail='Error visualizing detection')

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        return StreamingResponse(io.BytesIO(img_byte_arr), media_type='image/jpeg')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error processing image: {str(e)}')

@app.post("/detect_images")
async def detect_images_endpoint(file: UploadFile = File(...)):
    return await detect_images(file)

@app.post("/detect_videos")
async def detect_videos_endpoint(file: UploadFile = File(...)):
    if file.filename =="":
        raise HTTPException(status_code=400, detail='no file found')
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as video:
        video.write(file.file.read())

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    output_path = './uploads/output.webm'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    output_classes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference on the frame using the model
        results = model(image, conf=0.45)

        # Get the bounding boxes from the results
        bboxes = results[0].boxes.data.tolist()

        class_names = ['Open Eye','Closed Eye','Cigarette','Phone','Seatbelt']
        # Draw bounding boxes on the frame
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, prob_class, class_id = bbox
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            label = f"{class_names[int(class_id)]}: {prob_class:.2f}"
            output_classes.append(int(class_id))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (int(x_min), int(y_min) - text_size[1]),
                          (int(x_min) + text_size[0], int(y_min)), color, cv2.FILLED)
            cv2.putText(frame, label, (int(x_min), int(y_min)), font, font_scale, (255, 255, 255), font_thickness)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()
    # Return the output video file
    with open(output_path, 'rb') as video:
        video_data = video.read()

    with open('./uploads/output_classes.txt', "w") as file:
        file.writelines(str(output_classes))
    os.remove(video_path)
    os.remove(output_path)
    return StreamingResponse(io.BytesIO(video_data), 200, media_type='video/mp4')
@app.post("/good_driver")
async def good_driver_endpoint():
    '''
    This function is counting the trust level for driver.
    It depends on the frames, which contains photo or closed eyes
    Near phone_frames there is a 10 - hyperparameter, that stands for weight of phone frames.
    '''
    with open('./uploads/output_classes.txt', 'r') as output:
        classes = np.array(ast.literal_eval(list(output)[0]))
    all_video = len(classes)
    open_eyes_frames = np.bincount(classes)[0]
    os.remove('./uploads/output_classes.txt')
    try:
        phone_frames = np.bincount(classes)[3]
    except IndexError:
        phone_frames = 0
    closed_eye_frames = np.bincount(classes)[1] 
    trust_lvl = np.maximum(0, (1 - phone_frames*10/all_video - (all_video-open_eyes_frames)/all_video)*100).round(2)
    if trust_lvl <= 30:
        color='red'
    elif 30<trust_lvl<70:
        color='yellow'
    elif trust_lvl>=70:
        color='green'
    return trust_lvl, color

    


if __name__ == "__main__":
    uvicorn.run("inference:app", host="127.0.0.1", port=8000)
