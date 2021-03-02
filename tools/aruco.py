# %%
import numpy as np
import cv2


class ArucoDetector():
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()


    def save_arco(self, save_path='res/aruco.png', viz=True):
        img = cv2.aruco.drawMarker(self.aruco_dict, 1, 500)
        if viz:
            cv2.imshow('aruco', img)
            cv2.waitKey(-1)
        cv2.imwrite(save_path, img)

    def track_aruco(self, frame, viz=True):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, self.aruco_params)
