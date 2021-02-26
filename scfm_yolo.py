import argparse
import time
import cv2
import numpy as np
import torch
from skimage.transform import resize as imresize
import scsfm as models
import random

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

tl = 2
tf = 1
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                       pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def load_tensor_image(img, resize=(256, 320)):
    img = img.astype(np.float32)

    if resize:
        img = imresize(img, resize).astype(np.float32)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(
        0) / 255 - 0.45) / 0.225).to(device)
    return tensor_img


def prediction_to_visual(output, shape=(360, 450)):
    pred_disp = output.cpu().numpy()[0, 0]
    img = 1 / pred_disp
    img = imresize(img, shape).astype(np.float32)
    return img


def bounding_to_visual(depth, depth_map, points):
    print(depth.shape)
    print(depth_map.shape)
    for x in points:
        center = ((x[:2]+x[2:4])/2).astype(int)
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        color = colors[int(x[5])]
        cv2.rectangle(depth, c1, c2, color,
                      thickness=tl, lineType=cv2.LINE_AA)
        label = names[int(x[5])]
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(depth, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
        text = label+' '+str(depth_map[round(center[1]/1.40625)][round(center[0]/1.40625)])
        cv2.circle(depth, (center[0], center[1]), 3, (0, 255, 0), -1)
        cv2.putText(depth, text, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return depth


def get_bounding(frame_rgb):
    results = model(frame_rgb, size=320)  # includes NMS
    points = results.xyxy[0].numpy()
    return points


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='data/data1.mp4', help='source')
    parser.add_argument('--output', type=str,
                        default='res/scfm.csv', help='source')

    opt = parser.parse_args()
    print(opt)

    disp_net = models.DispResNet(18, False).to(device)
    weights = torch.load('data/weights/scfm-nyu2.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if opt.source == '0':
        source = 0
    else:
        source = opt.source

    cap = cv2.VideoCapture(source)
    out = cv2.VideoWriter('after.mp4', cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 30, (450, 360))
    while True:
        then = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb[:, 95:545]
        tgt_img = load_tensor_image(frame_rgb.copy(), (256, 320))
        print_duration(then, 'capture and convert')
        then = time.time()
        output = disp_net(tgt_img)
        print_duration(then, 'depth inference')
        then = time.time()
        points = get_bounding(frame_rgb)
        print_duration(then, 'yolo inference')

        # cv2.imshow('frame', frame)
        prediction = prediction_to_visual(output)
        # cv2.imshow('depth', prediction)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

        depth_rgb = np.uint8((1-prediction)*255)
        depth_rgb = cv2.cvtColor((depth_rgb*255).astype(np.uint8),
                                 cv2.COLOR_GRAY2BGR)
        depth_rgb = bounding_to_visual(depth_rgb, prediction, points)
        cv2.imshow('depth with bounding', depth_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(depth_rgb)
    out.release()
    cap.release()


def print_duration(then, prefix=''):
    print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()
