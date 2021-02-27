import logging
import time

from object_detector import ObjectDetector

LOG = logging.getLogger(__name__)


def run_capture_loop(capture, object_detector: ObjectDetector, result_queue):
    LOG.info('running capture loop...')

    while True:
        then = time.time()
        ret, frame = capture.read()
        timestamp = time.time()  # frame timestamp

        _, labels, positions, heights, widths = object_detector.estimate_pose(frame, viz=False)

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.info('object_detection took %.4f s', time.time() - then)

        for i in range(len(labels)):
            position = positions[i]
            label = labels[i]
            height = float(heights[i])
            width = float(widths[i])

            message = {
                'Timestamp': timestamp,
                'Type': label,
                'Position': {'X': float(position[0]), 'Y': float(position[1]), 'Z': float(position[2])},
                'Shape': [{'X': width, 'Y': height, 'Z': 0.0}]
            }

            LOG.debug('queueing message %s', message)

            result_queue.put(message)
