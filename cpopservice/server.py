import os
import glob
import time
import queue
import logging
import cv2
from cpopservice import config
from cpopservice.constants import MOSQUITTO_URL_LINUX
from cpopservice.core.models import Event
from cpopservice.core.pubsub import CPOPPublisher
from cpopservice.capture_loop import run_capture_loop
from cpopservice.utils.common import (
    ShellCommandThread, FuncThread, get_os_type, mkdir, download, untar, find_command, sleep_forever)

LOG = logging.getLogger(__name__)
THREADS = []


class CPOPServer(FuncThread):
    def __init__(self):
        FuncThread.__init__(self, self.run_loop)
        self.publisher = CPOPPublisher.get()

    def run_loop(self):
        LOG.info('Starting CPOP server capture loop')
        result_queue = queue.Queue()
        capture = get_capture_device()
        QueueSubscriber(result_queue, self.publisher).start()
        run_capture_loop(capture, result_queue=result_queue)


class QueueSubscriber(FuncThread):
    def __init__(self, queue, publisher):
        self.queue = queue
        self.publisher = publisher
        FuncThread.__init__(self, self.run_loop)

    def run_loop(self):
        LOG.info('Starting CPOP queue subscriber loop')
        while True:
            message = self.queue.get()
            message = self._prepare_message(message)
            self.publisher.publish_event(message)

    def _prepare_message(self, message):
        # TODO: for testing only
        if isinstance(message, dict):
            import time
            time.sleep(0.5)
            message = Event(message)
        return message


def get_capture_device():
    # TODO review this
    capture = cv2.VideoCapture(0)
    return capture


def start_mqtt_broker():
    mosquitto_bin = install_mqtt()
    LOG.info('Starting MQTT broker on port %s' % config.BROKER_PORT)
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
            raise Exception('Unsupported OS type: %s' % os_type)
        download(url, archive)
        untar(archive, target_dir)
    result = glob.glob('%s/**/*bin/mosquitto' % target_dir, recursive=True)
    return result[0]


def startup_servers():
    start_mqtt_broker()
    CPOPServer().start()


def shutdown_servers():
    LOG.info('Shutting down server threads ...')
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
