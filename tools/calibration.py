import cv2

from object_detector import ObjectDetector
from tools.capture import capture_from_vid


def main():
    object_detector = ObjectDetector()

    print('capturing')
    frame = capture_from_vid()

    print('calculating parameters')
    blob_frame = object_detector.init_camera_parameters(frame, viz=True)

    cv2.imwrite('calibrate_source.jpg', frame)
    cv2.imwrite('calibrate_result.jpg', blob_frame)

    print('done')


if __name__ == '__main__':
    main()
