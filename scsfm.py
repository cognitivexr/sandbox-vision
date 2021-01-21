# %%
import torch
from skimage.transform import resize as imresize
import numpy as np
# from path import Path
import argparse
from tqdm import tqdm
from torchvision import transforms
import time
import models.scsfm as models
import cv2
from util.timer import Timer

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(img):
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) /
                   255-0.45)/0.225).to(device)
    print(tensor_img.shape)
    return tensor_img


@torch.no_grad()
def main():
    disp_net = models.DispResNet(18, False).to(device)
    weights = torch.load('res/weights/scfm-nyu2.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    cap = cv2.VideoCapture('res/dataset3.mp4')
    timer = Timer('res/scsfm.csv')
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tgt_img = load_tensor_image(frame.copy())
        timer.start()
        output = disp_net(tgt_img)
        timer.stop()
        pred_disp = output.cpu().numpy()[0, 0]

        cv2.imshow('frame', 1/pred_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

# %%
