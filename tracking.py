import time
import cv2
from cpopservice.depth.scsfm import load_tensor_image, prediction_to_visual


cap = cv2.VideoCapture('data/data2.mp4')
while True:
    then = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    tgt_img = load_tensor_image(frame.copy())
    # print_duration(then, 'capture and convert')
    then = time.time()
    output = disp_net(tgt_img)
    # print_duration(then, 'inference')

    cv2.imshow('frame', frame)
    cv2.imshow('depth', prediction_to_visual(output))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
