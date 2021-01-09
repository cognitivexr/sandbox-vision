import cv2
import numpy as np
import torch

class DepthEstimation():
    def __init__(model_name='scsfm'):
        self.current_model = model_name

        # load scsfm
        disp_net = models.DispResNet(args.resnet_layers, False).to(device)
        weights = torch.load()
        disp_net.load_state_dict(weights['state_dict'])
        disp_net.eval()
        self.scsfm = disp_net

    def predict(img: np.array) -> np.array:
        self.scsfm.

