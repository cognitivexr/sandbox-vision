# %%
import argparse
import time
import cv2
import numpy as np
import torch
from skimage.transform import resize as imresize
import models.scsfm as models

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(img, resize=(256, 320)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if resize:
        img = imresize(img, resize).astype(np.float32)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(
        0) / 255 - 0.45) / 0.225).to(device)
    print(tensor_img.shape)
    return tensor_img


def prediction_to_visual(output, shape=(480, 640)):
    pred_disp = output.cpu().numpy()[0, 0]
    img = 1 / pred_disp
    img = imresize(img, shape).astype(np.float32)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='res/dataset1.mp4', help='source')
    parser.add_argument('--output', type=str,
                        default='res/scfm.csv', help='source')

    opt = parser.parse_args()
    print(opt)

    disp_net = models.DispResNet(18, False).to(device)
    weights = torch.load('res/weights/scfm-nyu2.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if opt.source == '0':
        source = 0
    else:
        source = opt.source

    cap = cv2.VideoCapture(source)
    while True:
        then = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        tgt_img = load_tensor_image(frame.copy(), (256, 832))
        print_duration(then, 'capture and convert')
        then = time.time()
        output = disp_net(tgt_img)
        print_duration(then, 'inference')

        cv2.imshow('frame', frame)
        cv2.imshow('depth', prediction_to_visual(output))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def print_duration(then, prefix=''):
    print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()

# %%
