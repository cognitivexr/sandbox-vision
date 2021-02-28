import argparse
import pickle

import cv2
from paho.mqtt.client import Client

from object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='input source')
    parser.add_argument('--width', type=int, default=1920, help='capture width')
    parser.add_argument('--height', type=int, default=1080, help='capture height')
    parser.add_argument('--mqtt-host', type=str, default='localhost', help='broker host')
    parser.add_argument('--mqtt-port', type=int, default=1883, help='broker port')
    parser.add_argument('--mqtt-topic', type=str, default='/cpop/calibration', help='broker topic')

    args = parser.parse_args()

    # initialize capture device
    if args.source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # mqtt client
    mqtt_client = Client()
    mqtt_client.connect(args.mqtt_host, args.mqtt_port)

    object_detector = ObjectDetector()

    try:
        print('starting to read from capture device...')
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                blob_frame = object_detector.init_camera_parameters(frame, viz=True)
                r, msg = cv2.imencode('.jpg', blob_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                msg = pickle.dumps(msg)
                print('frame size: %d bytes' % len(msg))
                mqtt_client.publish(args.mqtt_topic, msg)
            except IndexError:
                pass

    except KeyboardInterrupt:
        print('interrupted')
    finally:
        # Release everything if job is finished
        cap.release()
        mqtt_client.disconnect()

    print('exiting')


if __name__ == '__main__':
    main()
