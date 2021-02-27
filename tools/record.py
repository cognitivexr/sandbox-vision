import argparse
import os
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640, help='capture width')
    parser.add_argument('--height', type=int, default=360, help='capture height')
    parser.add_argument('--fps', type=int, default=30, help='video fps')
    parser.add_argument('--source', type=str, default='0', help='input source')
    parser.add_argument('--out', type=str, help='output file path')

    args = parser.parse_args()

    width = args.width
    height = args.height
    fps = args.fps

    if args.source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if args.out:
        if os.path.exists(args.out):
            raise ValueError(f'Destination {args.out} exists')
        else:
            fname = args.out
    else:
        fname = f'cap_{width}x{height}_{fps}_{int(time.time())}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))

    i = 0
    try:
        print(f'recording into {fname}')
        print('press CTRL+C to stop')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            i += 1

    except KeyboardInterrupt:
        print('interrupted')
    finally:
        # Release everything if job is finished
        cap.release()
        print(f'wrote {i} frames')

    print('exiting')


if __name__ == '__main__':
    main()
