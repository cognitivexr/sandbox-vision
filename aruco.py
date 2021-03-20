import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
from util.calibration import calculate_camera_calibration_mat


class ArucoDetector():

    def __init__(self, camera_matrix):
        r""" Initializes CameraCalibration module with sensor information

        Either hfov and vfov must be given or the pixel_size
        and the focal length must be passed to initialize the
        camera_matrix in later steps.

        Parameters
        ----------
        hfov : float
            horizontal field of view
        vfov : float
            vertical field of view
        pixel_size : float
            the pixel size in mm
        focal_mm : float
            focal length in mm
        """
        self.camera_matrix = camera_matrix
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

    def save_arco(self, save_path='res/aruco.png', viz=True):
        img = cv2.aruco.drawMarker(self.aruco_dict, 1, 500)
        if viz:
            cv2.imshow('aruco', img)
            cv2.waitKey(-1)
        cv2.imwrite(save_path, img)

    def generate_patterns():
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        fig = plt.figure()
        nx = 4
        ny = 3
        for i in range(1, nx*ny+1):
            ax = fig.add_subplot(ny, nx, i)
            img = aruco.drawMarker(aruco_dict, i, 700)
            plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
            ax.axis("off")
        plt.savefig("./markers.png")
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
        imboard = board.draw((2000, 2000))
        cv2.imwrite("chessboard.png", imboard)

    def track_aruco(self, frame, viz=True):
        ret = cv2.aruco.detectMarkers(
            frame, self.aruco_dict)
        corners, ids, rejected = ret
        frame = aruco.drawDetectedMarkers(
            frame, corners, ids)
        return corners, ids, rejected

    def get_aruco_positions(self, corners, size):
        result = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], size, self.camera_matrix, None)
        return result


def main():

    # init video capture and writer
    cap = cv2.VideoCapture('http://192.168.0.164:8080/video')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 25
    path = 'res/data.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f'Saving video {path} {width}x{height}')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # init camera parameters
    camera_matrix = np.array([[500, 0, width/2],
                              [0, 500, height/2],
                              [0, 0, 1]])

    # init aruco pattern detector and size
    # size = 1
    detector = ArucoDetector(camera_matrix)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        corners, ids, rejected = detector.track_aruco(frame)
        # if corners:
        #     detector.get_aruco_position(corners, size)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()


if __name__ == '__main__':
    main()
