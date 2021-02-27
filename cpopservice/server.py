import glob
import logging
import os
import queue
import time

import cv2

from cpopservice import config
from cpopservice.capture_loop import run_capture_loop
from cpopservice.constants import MOSQUITTO_URL_LINUX
from cpopservice.core.models import Event
from cpopservice.core.pubsub import CPOPPublisher
from cpopservice.utils.common import (
    ShellCommandThread, FuncThread, get_os_type, mkdir, download, untar, find_command, sleep_forever)
from object_detector import ObjectDetector
from tools.capture import capture_from_vid

LOG = logging.getLogger(__name__)
THREADS = []
LOCAL_HOSTS = ['localhost', '0.0.0.0', '127.0.0.1']


class CPOPServer(FuncThread):
    def __init__(self):
        FuncThread.__init__(self, self.run_loop)
        self.publisher = CPOPPublisher.get()

    def run_loop(self):
        LOG.info('starting CPOP server capture loop')
        result_queue = queue.Queue()

        object_detector = get_object_detector()

        capture = get_capture_device()

        width = config.CAMERA_WIDTH
        height = config.CAMERA_HEIGHT
        LOG.info('initializing camera matrix to dimensions: %d %d', width, height)
        object_detector.init_camera_matrix((height, width))

        QueueSubscriber(result_queue, self.publisher).start()

        run_capture_loop(capture, object_detector, result_queue=result_queue)


class QueueSubscriber(FuncThread):
    def __init__(self, q, publisher):
        self.queue = q
        self.publisher = publisher
        FuncThread.__init__(self, self.run_loop)

    def run_loop(self):
        LOG.info('starting CPOP queue subscriber loop')
        while True:
            message = self.queue.get()
            message = self._prepare_message(message)
            self.publisher.publish_event(message)

    def _prepare_message(self, message) -> Event:
        return Event(message)


def get_object_detector() -> ObjectDetector:
    object_detector = ObjectDetector()

    LOG.info('initializing camera parameters')
    frame = capture_from_vid(source=0, width=1920, height=1080)
    # frame = cv2.imread('data/calib-and-test/frame_1920x1080.jpg')

    object_detector.init_camera_parameters(frame, viz=False)

    return object_detector


def get_capture_device():
    LOG.info('initializing capture device')

    source = 0
    # source = 'data/calib-and-test/vid_640x360.mp4'

    capture = cv2.VideoCapture(source)

    width = config.CAMERA_WIDTH
    height = config.CAMERA_HEIGHT
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture


def start_mqtt_broker():
    broker_host = config.BROKER_HOST
    if not config.BROKER_STARTUP or broker_host not in LOCAL_HOSTS:
        return
    mosquitto_bin = install_mqtt()
    LOG.info('starting MQTT broker on port %s' % config.BROKER_PORT)
    t = ShellCommandThread('%s -p %s' % (mosquitto_bin, config.BROKER_PORT))
    t.start()
    time.sleep(1)
    THREADS.append(t)


def install_mqtt():
    bin_path = find_command('mosquitto')
    if bin_path:
        return bin_path
    os_type = get_os_type()
    target_dir = os.path.join(config.TMP_FOLDER, 'mosquitto')
    archive = os.path.join(target_dir, 'archive.tgz')
    mkdir(target_dir)
    if not os.path.exists(archive):
        if os_type == 'linux':
            url = MOSQUITTO_URL_LINUX
        else:
            raise Exception('unsupported OS type: %s' % os_type)
        download(url, archive)
        untar(archive, target_dir)
    result = glob.glob('%s/**/*bin/mosquitto' % target_dir, recursive=True)
    return result[0]


def startup_servers():
    start_mqtt_broker()
    CPOPServer().start()


def shutdown_servers():
    LOG.info('shutting down server threads ...')
    while THREADS:
        t = THREADS[0]
        del THREADS[0]
        t.stop()


def main():
    startup_servers()
    try:
        sleep_forever()
    except KeyboardInterrupt:
        shutdown_servers()


if __name__ == '__main__':
    main()
