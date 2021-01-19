# %%
from adabins.infer import InferenceHelper
import cv2
import torch
from util.timer import Timer
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

cap = cv2.VideoCapture('res/dataset1.mp4')
timer_yolo = Timer('res/yolo.csv')
timer_adabins = Timer('res/adabins.csv')
tl = 2

threshold = 0.25
width = cap.get(3)   # float `width`
height = cap.get(4)  # float `height`
print(f'width: {width} height: {height}')
out = cv2.VideoWriter('depth_resut.mp4', cv2.VideoWriter_fourcc(
    'm', 'p', '4', 'v'), 20, (640, 480))

# %%
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
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
    # heatmap = cv2.applyColorMap(
    #     (depth_img*255).astype(np.uint8), cv2.COLORMAP_BONE)

    # depth_img = cv2.cvtColor((depth_img*255).astype(np.uint8),
    #                          cv2.COLOR_GRAY2BGR)

    # mask = np.zeros((480, 640, 3), np.uint8)
    # for x in results:
    #     cv2.rectangle(depth_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #     cv2.rectangle(frame, c1, c2, (255, 255, 255), -1)
    #     frame[c1[0]:c1[1], c2[0]:c2[1]] = (255, 255, 255)

    # TODO use opencv functions
    # heatmap foreground
    # fg = cv2.bitwise_or(heatmap, heatmap, mask=mask)
    # normal frame background
    # mask = cv2.bitwise_not(mask)
    # background = np.full(frame.shape, 255, dtype=np.uint8)
    # bk = cv2.bitwise_or(background, background, mask=mask)
    # final = cv2.bitwise_or(fg, bk)

    # print(frame.shape)
    # final = np.zeros((frame.shape), np.uint8)
    # print(final.shape)
    # print(mask.shape)

    # depth_rgb = cv2.cvtColor((depth_img*255).astype(np.uint8),
    #                          cv2.COLOR_GRAY2BGR)
    # final[mask] = depth_rgb[mask]
    # final[~mask] = frame[~mask]

    depth_rgb = cv2.cvtColor((depth_img*255).astype(np.uint8),
                              cv2.COLOR_GRAY2BGR)

    # threshold results
    print(results.shape)
    mask = results[:, 4] > threshold  # check confidence
    results = results[:][mask]

    for x in results:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        # average depth
        mask = np.zeros((480, 640), np.uint8)
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

    # stack = np.hstack((frame, depth_rgb*255))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)
cap.release()
out.release()
timer_adabins.release()
timer_yolo.release()