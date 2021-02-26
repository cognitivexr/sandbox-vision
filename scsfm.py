# %%
import argparse
import time
import cv2
import numpy as np
import torch
from skimage.transform import resize as imresize
import scsfm as models

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(img, resize=(256, 320)):
    img = img.astype(np.float32)

    if resize:
        img = imresize(img, resize).astype(np.float32)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(
        0) / 255 - 0.45) / 0.225).to(device)
    print(tensor_img.shape)
    return tensor_img


def prediction_to_visual(output, shape=(360, 480)):
    pred_disp = output.cpu().numpy()[0, 0]
    img = 1 / pred_disp
    img = imresize(img, shape).astype(np.float32)
    return img


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
    weights = torch.load('data/weights/scfm-nyu2-test.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if opt.source == '0':
        source = 0
    else:
        source = opt.source

    cap = cv2.VideoCapture(source)
    out = cv2.VideoWriter('after.mp4', cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 30, (480, 360))
    while True:
        then = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb[:, 30:510]
        tgt_img = load_tensor_image(frame_rgb.copy(), (256, 320))
        print_duration(then, 'capture and convert')
        then = time.time()
        output = disp_net(tgt_img)
        print_duration(then, 'inference')

        cv2.imshow('frame', frame)
        prediction = prediction_to_visual(output)
        cv2.imshow('depth', prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prediction = np.uint8((1-prediction)*255)
        depth_rgb = cv2.cvtColor((prediction*255).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR)
        out.write(depth_rgb)
    out.release()
    cap.release()


def print_duration(then, prefix=''):
    print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()

# %%
