# %%
from typing import NamedTuple
from adabins.infer import InferenceHelper
import cv2
import torch
import numpy as np
from PIL import Image
import random


def plot_one_box(x, img, color=(0, 255, 0), label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


cap = cv2.VideoCapture('data/data2.mp4')
tl = 2

threshold = 0.4
width = cap.get(3)   # float `width`
height = cap.get(4)  # float `height`
print(f'width: {width} height: {height}')
out = cv2.VideoWriter('depth_result.mp4', cv2.VideoWriter_fourcc(
    'm', 'p', '4', 'v'), 20, (1280, 360))

# %%
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x',
                       pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Inference
inferHelper = InferenceHelper()

# %%
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_rgb = frame[:, :, ::-1]

    # depth prediction

    img_pil = Image.fromarray(frame_rgb)
    centers, pred = inferHelper.predict_pil(img_pil)
    depth_img = pred[0, 0]
    depth_img = depth_img/10

    # bounding box detection
    results = model(frame_rgb)  # includes NMS
    results = results.xyxy[0].numpy()

    min_depth = np.min(depth_img)
    max_depth = np.max(depth_img)
    depth_rgb = depth_img.copy()
    depth_rgb = (depth_rgb-min_depth)/(max_depth-min_depth)
    depth_rgb = cv2.cvtColor((depth_rgb*255).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR)

    # threshold results
    # mask = results[:, 4] > threshold  # check confidence
    # results = results # [:][mask]

    for x in results:
        if names[int(x[5])] not in ['keyboard', 'mouse', 'bottle']:
            continue

        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        # average depth
        mask = np.zeros((360, 640), np.uint8)
        cv2.rectangle(mask, c1, c2, (1), -1)
        mask = mask == 1
        mean = np.mean(depth_img[mask])*10

        color = (0, 255, 0)
        cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        label = 'Depth: {:2.2f} m, class: {}'.format(mean, names[int(x[5])])
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(frame, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
        cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    stack = np.hstack((frame, depth_rgb*255))
    cv2.imshow('frame', stack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(stack)
cap.release()
out.release()
