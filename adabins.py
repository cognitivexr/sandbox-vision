#%%
from models.adabins.infer import InferenceHelper
import matplotlib.pyplot as plt
from time import time
import cv2
from PIL import Image

# img = Image.open("test_imgs/classroom__rgb_00283.jpg")
start = time()
inferHelper = InferenceHelper()

cap = cv2.VideoCapture('https://192.168.0.164:8080/video')
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(frame)
    # tgt_img = load_tensor_image( dataset_dir + test_files[j], args)
    centers, pred = inferHelper.predict_pil(im_pil)
    depth_img = pred[0,0]
    print(depth_img.shape)
    print(1/depth_img)
    cv2.imshow('frame', depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break