import os
import logging
import tempfile


# host and port of local MQTT broker
BROKER_HOST = os.environ.get('BROKER_HOST') or 'localhost'
BROKER_PORT = int(os.environ.get('BROKER_PORT') or 1883)
BROKER_STARTUP = os.environ.get('BROKER_STARTUP') not in ['false', '0', False]

# MQTT topic name
MQTT_TOPIC_NAME = 'cpop'

# local folder to store temporary files
TMP_FOLDER = os.path.join(tempfile.gettempdir(), 'cpopserver')

# configure logging
logging.basicConfig(level=logging.INFO)