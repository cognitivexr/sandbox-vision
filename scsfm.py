#%%
import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from torchvision import transforms
import time
import models.scsfm as models
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(img):
    h,w,_ = img.shape
    # if (h != args.img_height or w != args.img_width):
    #     img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def main():
    disp_net = models.DispResNet(18, False).to(device)
    weights = torch.load('res/models/scfm-nyu2.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    cap = cv2.VideoCapture('https://192.168.0.164:8080/video')
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[...,::-1].copy()
        tgt_img = load_tensor_image(frame)
        output = disp_net(tgt_img)
        pred_disp = output.cpu().numpy()[0,0]

        cv2.imshow('frame',1/pred_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

#%%
