import argparse
import os
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1920, help='capture width')
    parser.add_argument('--height', type=int, default=1080, help='capture height')
    parser.add_argument('--source', type=str, default='0', help='input source')
    parser.add_argument('--out', type=str, help='output file path')

    args = parser.parse_args()

    width = args.width
    height = args.height

    if args.source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if args.out:
        if os.path.exists(args.out):
            raise ValueError(f'destination {args.out} exists')
        else:
            fname = args.out
    else:
        fname = f'cap_{width}x{height}_{int(time.time())}.png'

    try:
        # read 10 frames and then capture (to let the camera focus)
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break

        print(f'recording into {fname}')
        cv2.imwrite(fname, frame)
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        # Release everything if job is finished
        cap.release()

    print('exiting')


if __name__ == '__main__':
    main()
