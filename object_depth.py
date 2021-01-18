from adabins.infer import InferenceHelper
import cv2
import torch
from util.timer import Timer
import numpy as np
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                       pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS


# Inference
cap = cv2.VideoCapture('res/webcam_data.mp4')

inferHelper = InferenceHelper()

timer_yolo = Timer()
timer_adabins = Timer()
while True:
    ret, frame = cap.read()
    frame_rgb = frame[:, :, ::-1]

    # depth prediction
    timer_adabins.start()
    img_pil = Image.fromarray(frame_rgb)
    centers, pred = inferHelper.predict_pil(img_pil)
    depth_img = pred[0, 0]
    depth_img = depth_img/10
    timer_adabins.stop()
    timer_adabins.print_summary()

    # bounding box detection
    timer_yolo.start()
    results = model(frame_rgb, size=320)  # includes NMS
    timer_yolo.stop()
    timer_yolo.print_summary()
    results = results.xyxy[0].numpy()

    # visualisation
    heatmap = cv2.applyColorMap(
        (depth_img*255).astype(np.uint8), cv2.COLORMAP_BONE)

    depth_img = cv2.cvtColor((depth_img*255).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR)

    mask = np.zeros((480, 640, 3), np.uint8)
    for result in results:
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]

        confidence = result[4]
        cv2.rectangle(depth_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (1, 1, 1), -1)

    # TODO use opencv functions
    # heatmap foreground
    # fg = cv2.bitwise_or(heatmap, heatmap, mask=mask)
    # normal frame background
    # mask = cv2.bitwise_not(mask)
    # background = np.full(frame.shape, 255, dtype=np.uint8)
    # bk = cv2.bitwise_or(background, background, mask=mask)
    # final = cv2.bitwise_or(fg, bk)

    final = np.zeros((frame.shape), np.uint8)
    mask = mask == 1
    final[mask] = depth_img[mask]
    final[~mask] = frame[~mask]

    cv2.imshow('frame', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
