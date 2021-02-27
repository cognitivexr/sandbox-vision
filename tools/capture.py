import argparse
import os
import time

import cv2


def capture_from_vid(source=0, width=1920, height=1080, warmup=10):
    """
    Returns a single frame from the given VideoCapture source.

    Args:
        source: the VideoCapture source
        width: frame width
        height: frame height
        warmup: number of frames to be discarded before capturing

    Returns: a single frame

    """
    if warmup < 0:
        raise ValueError('warmup must be positive, was: %s' % warmup)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        frame = None
        for i in range(warmup + 1):
            ret, frame = cap.read()
            if not ret:
                break

        return frame
    finally:
        cap.release()


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
        source = 0
    else:
        source = args.source

    if args.out:
        if os.path.exists(args.out):
            raise ValueError(f'destination {args.out} exists')
        else:
            fname = args.out
    else:
        fname = f'cap_{width}x{height}_{int(time.time())}.png'

    try:
        frame = capture_from_vid(source, width, height)
        cv2.imwrite(fname, frame)
    except KeyboardInterrupt:
        print('interrupted')

    print('exiting')


if __name__ == '__main__':
    main()
