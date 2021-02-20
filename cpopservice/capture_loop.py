import cv2
import torch
from time import time
from PIL import Image
from adabins.infer import InferenceHelper
from util.timer import Timer


def run_capture_loop(capture, timer=None, result_queue=None, headless=True):
    # infer_helper = InferenceHelper()
    while True:
        ret, frame = capture.read()
        frame_rgb = frame[:, :, ::-1]
        im_pil = Image.fromarray(frame_rgb)
        timer and timer.start()
        # centers, pred = inferHelper.predict_pil(im_pil)
        timer and timer.stop()
        # depth_img = pred[0, 0]/10
        if result_queue:
            message = {'timestamp': time(), 'x': 1, 'y': 2, 'z': 3}
            result_queue.put(message)
        if not headless:
            cv2.imshow('frame', depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
