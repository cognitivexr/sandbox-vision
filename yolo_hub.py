import cv2
import torch
from util.timer import Timer
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                       pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# Inference
cap = cv2.VideoCapture('https://192.168.0.164:8080/video')
timer = Timer()
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :, ::-1]
    print(f'min: {np.min(frame_rgb)} max: {np.max(frame_rgb)}')
    timer.start()
    results = model(frame_rgb, size=320)  # includes NMS
    timer.stop()
    timer.print_summary()
    points = results.xyxy[0].numpy()
    for xyxy in points:
        x1 = xyxy[0]
        y1 = xyxy[1]
        x2 = xyxy[2]
        y2 = xyxy[3]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
