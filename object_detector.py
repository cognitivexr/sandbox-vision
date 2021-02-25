import random
import time

import cv2
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
    model.cuda()  # make sure we use CUDA
    names = model.module.names if hasattr(model, 'module') else model.names # object classes

    cap = cv2.VideoCapture(0)

    while True:
        then = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = frame[:, :, ::-1]
        print('capture:   %.2fms' % ((time.time() - then) * 1000))

        then = time.time()
        results = model(frame_rgb, size=320)  # includes NMS
        results = results.xyxy[0].cpu().numpy()

        print('inference:  %.2fms' % ((time.time() - then) * 1000))

        for result in results:
            x = result
            object_type = int(result[5])
            confidence = result[4]

            # debug
            print('%20s %s' % (names[object_type], result))

            # draw rectangles
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

            color = (0, 255, 0)
            cv2.rectangle(frame, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
