# Computer Vision Sandbox

This is the main repo hosting the computer vision sandbox for CognitiveXR.

It consists of the following main components:
* `cpopserver`: The CPOP server and message broker providing pub/sub functionality over MQTT
* `detect`: Object detection and tracking framework (name to be defined)

## Prerequisites

* Python v3.7+
* `make`

## Installation

To install the dependencies in a local Python virtualenv (in `.venv` folder):
```
make install
```

## Details

Additional details for these topics following soon:
* Cuda
* Virtual Environment
* Models

## Execution

Use this command to start the CPOP server:
```
make server
```

## Tests

To run the local unit and integration tests:
```
make test
```
