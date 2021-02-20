import os
import logging
import tempfile


# bind port of local MQTT broker
BROKER_PORT = int(os.environ.get('BROKER_PORT') or 54321)

# local folder to store temporary files
TMP_FOLDER = os.path.join(tempfile.gettempdir(), 'cpopserver')

# configure logging
logging.basicConfig(level=logging.INFO)
