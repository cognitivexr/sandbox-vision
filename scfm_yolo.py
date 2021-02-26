import argparse
import time
import cv2
import numpy as np
import torch
from skimage.transform import resize as imresize
import scsfm as models
import random
from math import pi, tan


class ObjectDepthDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tl = 2
        self.tf = 1
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                   pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS
        self.names = self.yolo.module.names if hasattr(
            self.yolo, 'module') else self.yolo.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def print_duration(self, then, prefix=''):
        print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))

    def load_tensor_image(self, img, resize=(256, 320)):
        img = img.astype(np.float32)

        if resize:
            img = imresize(img, resize).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(
            0) / 255 - 0.45) / 0.225).to(self.device)
        return tensor_img

    def prediction_to_visual(self, output, shape=(360, 450)):
        pred_disp = output.cpu().numpy()[0, 0]
        img = 1 / pred_disp
        img = imresize(img, shape).astype(np.float32)
        return img

    def bounding_to_visual(self, depth, depth_map, points):
        print(depth.shape)
        print(depth_map.shape)
        for x in points:
            center = ((x[:2]+x[2:4])/2).astype(int)
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            color = self.colors[int(x[5])]
            tl = self.tl
            tf = self.tf
            cv2.rectangle(depth, c1, c2, color,
                          thickness=tl, lineType=cv2.LINE_AA)
            label = self.names[int(x[5])]
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(depth, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
            text = label+' ' + \
                str(depth_map[round(center[1]/1.40625)]
                    [round(center[0]/1.40625)])
            cv2.circle(depth, (center[0], center[1]), 3, (0, 255, 0), -1)
            cv2.putText(depth, text, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return depth

    def get_bounding(self, frame_rgb):
        results = self.yolo(frame_rgb, size=320)  # includes NMS
        points = results.xyxy[0].numpy()
        return points

    @torch.no_grad()
    def object_depth_estimation_loop(self, cap):
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str,
                            default='data/data1.mp4', help='source')
        parser.add_argument('--output', type=str,
                            default='res/scfm.csv', help='source')

        opt = parser.parse_args()
        print(opt)

        disp_net = models.DispResNet(18, False).to(self.device)
        weights = torch.load('data/weights/scfm-nyu2.pth.tar')
        disp_net.load_state_dict(weights['state_dict'])
        disp_net.eval()

        if opt.source == '0':
            source = 0
        else:
            source = opt.source

        cap = cv2.VideoCapture(source)
        out = cv2.VideoWriter('after.mp4', cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), 30, (450, 360))
        while True:
            then = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb[:, 95:545]
            tgt_img = self.load_tensor_image(frame_rgb.copy(), (256, 320))
            self.print_duration(then, 'capture and convert')
            then = time.time()
            output = disp_net(tgt_img)
            self.print_duration(then, 'depth inference')
            then = time.time()
            points = self.get_bounding(frame_rgb)
            self.print_duration(then, 'yolo inference')

            # cv2.imshow('frame', frame)
            prediction = self.prediction_to_visual(output)
            # cv2.imshow('depth', prediction)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

            depth_rgb = np.uint8((1-prediction)*255)
            depth_rgb = cv2.cvtColor((depth_rgb*255).astype(np.uint8),
                                     cv2.COLOR_GRAY2BGR)
            depth_rgb = self.bounding_to_visual(depth_rgb, prediction, points)
            cv2.imshow('depth with bounding', depth_rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out.write(depth_rgb)
        out.release()
        cap.release()


class CameraCalibration:
    def draw_line(self, frame, p1, p2, color, thickness):
        frame = cv2.line(frame,
                         tuple(p1[:2].astype(int)),
                         tuple(p2[:2].astype(int)),
                         color,
                         thickness)
        return frame

    def draw_point(self, blob_frame, p, color=(0, 255, 0)):
        cv2.circle(blob_frame, tuple(p[:2].astype(int)),
                   3, color, -1)

    def draw(self, frame, imgpts):
        # corner = corners[0].ravel()
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw ground floor in green
        frame = cv2.drawContours(frame, [imgpts[:4]],
                                 -1, (0, 255, 0), -3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            frame = cv2.line(frame, tuple(imgpts[i]),
                             tuple(imgpts[j]), (255), 1)
        # draw top layer in red color
        frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 1)
        return frame

    def find_extrinsic_parameters(self, frame):
        ####################
        # FRAME PARAMETERS #
        ####################
        height, width, channels = frame.shape
        print(f'width: {width} height: {height} channels: {channels}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #####################
        # CAMERA PARAMETERS #
        #####################
        # angle_of_view = None
        # pixel_size = None
        # sensor_width_mm = None
        # sensor_height_mm = None

        horizontal_field_of_view = 70.42
        vertical_field_of_view = 43.3
        # focal_mm = 3.67

        c_x = width/2
        c_y = height/2

        # From sensor width
        # f_x = (focal_mm/sensor_width_mm)*width
        # f_y = (focal_mm/sensor_height_mm)*height

        # From field of view
        f_x = c_x/tan(horizontal_field_of_view*0.5*pi/180)
        f_y = c_y/tan(vertical_field_of_view*0.5*pi/180)

        camera_matrix = np.array([[f_x, 0, c_x],
                                  [0, f_y, c_y],
                                  [0, 0, 1]])
        # focal_length = 20 # according to the image metadata

        ######################
        ## BOARD PARAMETERS ##
        ######################
        column_count = 6
        row_count = 4
        circle_diameter = 30
        spacing = 40
        board_width = (column_count-1)*spacing
        board_height = (row_count-1)*spacing

        ################
        # OBJECTPOINTS #
        ################

        object_points = np.zeros((column_count*row_count, 3))
        idx = 0
        for column in range(column_count):
            for row in range(row_count):
                x = column * spacing
                y = row * spacing
                object_points[idx] = (x, y, 0)  # TODO check
                idx = idx+1

        ##################
        # BLOB DETECTION #
        ##################
        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 8
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 10
        blobParams.maxArea = 100

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.8

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.9

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.4

        # Create a detector with the parameters
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)

        ###################
        # Board Detection #
        ###################

        def find_board(keypoints):
            x_coords = [p.pt[0] for p in keypoints]
            y_coords = [p.pt[1] for p in keypoints]

            while len(keypoints) > 6*4:
                _len = len(keypoints)
                centroid_x = sum(x_coords)/_len
                centroid_y = sum(y_coords)/_len
                index = np.argmax([(x_coords[i]-centroid_x)**2 +
                                   (y_coords[i]-centroid_y)**2
                                   for i in range(_len)])
                keypoints.pop(index)
                x_coords.pop(index)
                y_coords.pop(index)
            corners = [
                np.argmin(y_coords),
                np.argmax(x_coords),
                np.argmax(y_coords),
                np.argmin(x_coords)
            ]
            return keypoints, corners

        ######################################
        # Find Blob, Rectangle and visualize #
        ######################################
        # Detect Blobs
        keypoints = blobDetector.detect(gray)
        blob_frame = cv2.drawKeypoints(frame, keypoints,
                                       np.array([]), (255, 0, 0),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # find board with custom algorithm
        keypoints_cluster, corners_idx = find_board(keypoints)

        # visualize board
        blob_frame = cv2.drawKeypoints(blob_frame, keypoints,
                                       np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite('res/blobs.png', blob_frame)
        # %%
        image_points = np.array([[keypoints[idx].pt[0], keypoints[idx].pt[1]]
                                 for idx in range(column_count*row_count)])

        sorted_indexes = np.lexsort((image_points[:, 0], image_points[:, 1]))
        corners = image_points[sorted_indexes]

        # keypoints = keypoints[sorted_indexes] TODO: sort keypoints
        # TODO: check out flags = cv2.SOLVEPNP_IPPE_SQUARE | cv2.SOLVEPNP_IPPE
        object_points = np.array([object_points[i] for i in [0, 20, 23, 3]])
        corners = np.array([corners[i] for i in [6, 0, 18, 23]])

        retval, rvec, tvec = cv2.solvePnP(object_points, corners,
                                          camera_matrix, None,
                                          cv2.SOLVEPNP_IPPE)

        print(f'rvec: {rvec}')
        print(f'tvec: {tvec}')

        axis = np.float32([[0, 0, 0], [40*5, 0, 0], [40*5, 40*3, 0], [0, 40*3, 0],
                           [0, 0, -40*3], [40*5, 0, -40*3], [40*5, 40*3, -40*3], [0, 40*3, -40*3]]).reshape(-1, 3)

        projected_points, jac = cv2.projectPoints(
            axis, rvec, tvec, camera_matrix, None)
        blob_frame = self.draw(blob_frame, projected_points)

        self.draw_point(blob_frame, corners[0], (255, 0, 255))
        self.draw_point(blob_frame, corners[1], (255, 255, 255))
        self.draw_point(blob_frame, corners[2], (0, 255, 255))
        self.draw_point(blob_frame, corners[3], (0, 0, 0))

        self.draw_point(blob_frame, projected_points[0][0], (255, 0, 255))
        self.draw_point(blob_frame, projected_points[1][0], (255, 255, 255))
        self.draw_point(blob_frame, projected_points[2][0], (0, 255, 255))
        self.draw_point(blob_frame, projected_points[3][0], (0, 0, 0))

        self.draw_point(blob_frame, np.mean(corners, axis=0), (0, 0, 255))

        idx = 0
        for point in corners:
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(point)
            cv2.putText(blob_frame, f'{idx}',
                        tuple(point.astype(int)),
                        font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            idx = idx+1

        toe_point = np.array([915, 980], np.float32)
        self.draw_point(blob_frame, toe_point)

        # calculate the 3d direction of the ray in camera coordinate frame
        toe_point_norm = cv2.undistortPoints(
            toe_point, camera_matrix, None)[0][0]
        print(f'toe_point_norm: {toe_point_norm}')
        ray_dir_cam = np.array([toe_point_norm[0], toe_point_norm[1], 1])
        print(f'ray_dir_cam: {ray_dir_cam}')

        # compute the 3d direction
        rot_cam_chessboard = cv2.Rodrigues(rvec)[0]
        rot_chessboard_cam = np.transpose(rot_cam_chessboard)
        t_cam_chessboard = tvec
        pos_cam_chessboard = np.matmul(-rot_chessboard_cam, t_cam_chessboard)
        # Map the ray direction vector from camera coordinates to chessboard coordinates
        ray_dir_chessboard = np.matmul(rot_chessboard_cam, ray_dir_cam)

        # Find the desired 3d point by computing the intersection between the 3d ray and the chessboard plane with Z=0:
        # Expressed in the coordinate frame of the chessboard, the ray originates from the
        # 3d position of the camera center, i.e. 'pos_cam_chessboard', and its 3d
        # direction vector is 'ray_dir_chessboard'
        # Any point on this ray can be expressed parametrically using its depth 'd':
        # P(d) = pos_cam_chessboard + d * ray_dir_chessboard
        # To find the intersection between the ray and the plane of the chessboard, we
        # compute the depth 'd' for which the Z coordinate of P(d) is equal to zero
        d_intersection = -pos_cam_chessboard[2]/ray_dir_chessboard[2]
        print(f'd_intersection: {d_intersection}')
        intersection_point = pos_cam_chessboard.T[0] + \
            d_intersection[0]*ray_dir_chessboard
        print(f'intersection_point: {intersection_point}')

        points, jac = cv2.projectPoints(intersection_point,
                                        rvec, tvec, camera_matrix, None)
        self.draw_point(blob_frame, points[0][0], (255, 255, 255))

        print(ray_dir_cam*d_intersection)

        cv2.imwrite('res/blobs.png', blob_frame)
        cv2.imshow('calibration results', blob_frame)
        cv2.waitKey(-1)
        return rvec, tvec


if __name__ == '__main__':
    frame = cv2.imread('data/blob21.jpg')
    calibration = CameraCalibration()
    calibration.find_extrinsic_parameters(frame)

    object_depth = ObjectDepthDetector()
    cap = cv2.VideoCapture('data/data1.mp4')
    object_depth.object_depth_estimation_loop(cap)
