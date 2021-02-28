"""
Replays json-encoded list of CPOP data into an MQTT broker.

Requirements:

    paho-mqtt
    bson

Example invocation:

    python -m tools.replay --data res/data.json --broker-host 192.168.0.41

"""
import argparse
import json
import time

from cpopservice import config
from cpopservice.core.models import Event
from cpopservice.core.pubsub import CPOPPublisherMQTT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=argparse.FileType('r'), required=True, help='path to json encoded cpop data')
    parser.add_argument('--broker-host', type=str, help='hostname of the MQTT broker')
    parser.add_argument('--broker-port', type=int, help='port of the MQTT broker')

    args = parser.parse_args()

    if args.broker_host:
        config.BROKER_HOST = args.broker_host
    if args.broker_port:
        config.BROKER_PORT = args.broker_port

    cpop_data = json.load(args.data)

    cpop_data = sorted(cpop_data, key=lambda d: d['Timestamp'])
    duration = cpop_data[-1]['Timestamp'] - cpop_data[0]['Timestamp']

    print('replaying %d object positions for %.2f seconds' % (len(cpop_data), duration))

    publisher = CPOPPublisherMQTT()

    prev_ts = None
    try:
        for data in cpop_data:
            if prev_ts is None:
                prev_ts = data['Timestamp']
            else:
                ia = data['Timestamp'] - prev_ts
                time.sleep(ia)
                prev_ts = data['Timestamp']

            publisher.publish_event(Event(data))

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
