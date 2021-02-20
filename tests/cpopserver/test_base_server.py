import queue
import unittest
import paho.mqtt.client as mqtt
from cpopserver import config
from cpopserver.core import models
from cpopserver.server import startup_servers, shutdown_servers
from cpopserver.utils.common import short_uid, FuncThread, retry


class TestCPOPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        startup_servers()

    @classmethod
    def tearDownClass(cls):
        shutdown_servers()

    def test_basic_communication(self):
        # get address
        broker_address = 'mqtt://localhost:%s' % config.BROKER_PORT
        topic_name = 'test_topic_%s' % short_uid()
        num_messages = 5
        messages_sent = []
        messages_received = []
        q = queue.Queue()

        # 1. create client and subscribe to topic

        def on_connect(client, userdata, flags, rc):
            client.subscribe(topic_name)
            q.put('connected')

        def on_message(client, userdata, msg):
            event = models.Event.from_bson(msg.payload)
            messages_received.append(event)

        client = mqtt.Client()
        client.on_message = on_message
        client.on_connect = on_connect
        client.connect('localhost', config.BROKER_PORT)
        FuncThread(client.loop_forever).start()
        q.get()  # wait until client is initialized

        # 2. publish messages

        for i in range(num_messages):
            event = models.Event()
            messages_sent.append(event)
            client.publish(topic_name, event.to_bson())

        # 3. wait for all messages to be received

        def check_finished():
            self.assertEquals(len(messages_received), num_messages)
            for msg in messages_received:
                self.assertIn(msg, messages_sent)

        retry(check_finished, retries=3, sleep=1)
