import argparse
import pickle
import queue
from threading import Thread

import cv2
from paho.mqtt.client import Client as MQTTClient, MQTTMessage

POISON = object()


def loop_client(client, fqueue: queue.Queue):
    try:
        client.loop_forever()
    finally:
        fqueue.put(POISON)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqtt-host', type=str, default='localhost', help='broker host')
    parser.add_argument('--mqtt-port', type=int, default=1883, help='broker port')
    parser.add_argument('--mqtt-topic', type=str, default='/cpop/calibration', help='broker topic')

    args = parser.parse_args()

    frameq = queue.Queue()

    # frame callback
    def on_message_callback(_client, _userdata, message: MQTTMessage):
        msg = message.payload
        frameq.put(cv2.imdecode(pickle.loads(msg), 1))

    # initialize and connect client
    client = MQTTClient()
    client.on_message = on_message_callback
    client.connect(args.mqtt_host, port=args.mqtt_port)
    client.subscribe(args.mqtt_topic)
    subscriber = Thread(target=loop_client, args=(client, frameq))
    subscriber.start()

    try:
        while True:
            frame = frameq.get()
            if frame is POISON:
                break
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    print('disconnecting')
    if client.is_connected():
        client.disconnect()

    print('waiting for client loop')
    subscriber.join(2)

    print('exitting')


if __name__ == '__main__':
    main()
