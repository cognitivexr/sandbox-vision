import os
import glob
import time
import logging
from cpopserver import config
from cpopserver.constants import MOSQUITTO_URL_LINUX
from cpopserver.utils.common import (
    ShellCommandThread, get_os_type, mkdir, download, untar, find_command, sleep_forever)

LOG = logging.getLogger(__name__)
THREADS = []


class CPOPPublisher:
    """ Publisher that publishes CPOP events to the pub/sub broker """

    def publish_event(self, event):
        print('TODO - publish event message: %s' % event)


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


def shutdown_servers():
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
