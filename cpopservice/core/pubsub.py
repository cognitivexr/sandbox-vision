import logging

import paho.mqtt.client as mqtt

from cpopservice import config
from cpopservice.core.models import Serializable

LOG = logging.getLogger(__name__)


class CPOPPublisher:
    """ Publisher that publishes CPOP events to the pub/sub broker """

    def publish_event(self, event):
        raise Exception('Not implemented')

    @staticmethod
    def get(impl_type=None):
        subclasses = CPOPPublisher.__subclasses__()
        subclasses = {subclass.name(): subclass for subclass in subclasses}
        if not impl_type and len(subclasses) != 1:
            raise Exception('Multiple CPOPPublisher implemtations found and type not specified')
        subclass = subclasses.get(impl_type) or list(subclasses.values())[0]
        return subclass()


class CPOPPublisherMQTT(CPOPPublisher):
    """ Publisher based on MQTT broker """

    @staticmethod
    def name():
        return 'mqtt'

    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect(config.BROKER_HOST, config.BROKER_PORT)

    def publish_event(self, event: Serializable):
        LOG.debug('publishing message to topic %s: %s', config.MQTT_TOPIC_NAME, event)
        self.client.publish(config.MQTT_TOPIC_NAME, event.to_bson())
