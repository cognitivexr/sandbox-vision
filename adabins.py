from adabins.infer import InferenceHelper
from time import time
import cv2
from PIL import Image
import torch
from util.timer import Timer

# img = Image.open("test_imgs/classroom__rgb_00283.jpg")
start = time()
inferHelper = InferenceHelper()

cap = cv2.VideoCapture('res/dataset3.mp4')
timer = Timer('res/adabins.csv')
while True:
    ret, frame = cap.read()
    frame_rgb = frame[:, :, ::-1]
    im_pil = Image.fromarray(frame_rgb)
    timer.start()
    centers, pred = inferHelper.predict_pil(im_pil)
    timer.stop()
    depth_img = pred[0, 0]/10
    cv2.imshow('frame', depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
