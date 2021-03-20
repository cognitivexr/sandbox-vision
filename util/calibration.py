from math import tan, pi
import numpy as np
import cv2


def calculate_camera_calibration_mat(width, height, hfov=None, vfov=None,
                                     pixel_size=None, focal_mm=None):
    c_x = width / 2
    c_y = height / 2

    # assert that params are initialized
    assert((hfov and vfov) or (pixel_size and focal_mm))

    if hfov and vfov:
        # From field of view
        f_x = c_x / tan(hfov * 0.5 * pi / 180)
        f_y = c_y / tan(vfov * 0.5 * pi / 180)
    elif pixel_size:
        # From sensor width and height
        sensor_width_mm = focal_mm * pixel_size
        sensor_height_mm = focal_mm * pixel_size
        f_x = (focal_mm / sensor_width_mm) * width
        f_y = (focal_mm / sensor_height_mm) * height

    camera_matrix = np.array([[f_x, 0, c_x],
                              [0, f_y, c_y],
                              [0, 0, 1]])
    return camera_matrix


def draw_cube(frame, camera_matrix, size, rvec, tvec):
    axis = np.float32(
        [[0, 0, 0],  # origin
         [size, 0, 0],  # right axis
         [size, size, 0],
         [0, size, 0],  # left axis
         [0, 0, -size],
         [size, 0, -size],
         [size, size, -size],
         [0, size, -size]]).reshape(-1, 3)
    projected_points, _ = cv2.projectPoints(
        axis, rvec, tvec, camera_matrix, None)
    frame = draw(frame, projected_points)
    return frame
