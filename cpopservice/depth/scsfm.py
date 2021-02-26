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
    img = img[:, 30:510]
    if resize:
        resized_img = imresize(img, resize)
    cv2.imshow('resized', resized_img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if resize:
        img = imresize(img, resize)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(
        0) / 255 - 0.45) / 0.225).to(device)
    print(f'tensor shape {tensor_img.shape}')
    return tensor_img


def prediction_to_visual(output, shape=(360, 640)):
    pred_disp = output.cpu().numpy()[0, 0]
    img = 1 / pred_disp
    # img = imresize(img, shape).astype(np.float32)
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
    weights = torch.load('data/weights/scfm-nyu2-test.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if opt.source == '0':
        source = 0
    else:
        source = opt.source




def print_duration(then, prefix=''):
    print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()

# %%
