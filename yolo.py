import numpy as np
import argparse
import cv2
import os
import time


# Get the labels
labels = open(args.labels).read().strip().split('\n')

# Create a list of colors for the labels
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Load weights using OpenCV
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

image = cv2.imread(args.image_path)

boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)