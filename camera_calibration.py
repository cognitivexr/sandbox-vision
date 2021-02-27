# %%
import scsfm
import argparse
import time
import cv2
import numpy as np
import torch
from skimage.transform import resize as imresize
import random
from math import pi, tan
from numpy.linalg import norm

###########################
# VISUALIZATION FUNCTIONS #
###########################


def draw_line(frame, p1, p2, color, thickness):
    frame = cv2.line(frame,
                     tuple(p1[:2].astype(int)),
                     tuple(p2[:2].astype(int)),
                     color,
                     thickness)
    return frame


def draw_point(blob_frame, p, color=(0, 255, 0)):
    cv2.circle(blob_frame, tuple(p[:2].astype(int)),
               3, color, -1)


def draw(frame, imgpts):
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


def intersect_line_plane(ray_origin, ray_dir, plane, plane_d):
    plane_dot_ray = plane[0]*ray_dir[0] + \
        plane[1]*ray_dir[1] + plane[2]*ray_dir[2] + plane_d
    if abs(plane_dot_ray) > 0:
        plane_dot_ray_origin = ray_origin[0]*plane[0] + \
            ray_origin[1]*plane[1] + ray_origin[2]*plane[2] + plane_d
        return ray_origin - ray_dir * (plane_dot_ray_origin/plane_dot_ray)


def to_coordinate(image_point, rvec, tvec):
    # image_point = np.array([915, 980], np.float32)
    draw_point(blob_frame, image_point)

    # calculate the 3d direction of the ray in camera coordinate frame
    toe_point_norm = cv2.undistortPoints(
        image_point, camera_matrix, None)[0][0]
    ray_dir_cam = np.array([toe_point_norm[0], toe_point_norm[1], 1])

    # compute the 3d direction
    rot_cam_chessboard = cv2.Rodrigues(rvec)[0]
    rot_chessboard_cam = np.transpose(rot_cam_chessboard)
    t_cam_chessboard = tvec
    pos_cam_chessboard = np.matmul(-rot_chessboard_cam, t_cam_chessboard)
    # Map the ray direction vector from camera coordinates
    # to chessboard coordinates
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
    draw_point(blob_frame, points[0][0], (255, 255, 255))
    # print(ray_dir_cam*d_intersection)
    # intersection_point[2] = -185*10
    points, jac = cv2.projectPoints(intersection_point,
                                    rvec, tvec, camera_matrix, None)
    draw_point(blob_frame, points[0][0], (255, 255, 255))
    return intersection_point


####################
# FRAME PARAMETERS #
####################
frame = cv2.imread('data/blob21.jpg')
height, width, channels = frame.shape
print(f'width: {width} height: {height} channels: {channels}')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#######################
## CAMERA PARAMETERS ##
#######################
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

# focal_length = 20
# according to the image metadata

####################
# BOARD PARAMETERS #
####################
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
blob_frame = draw(blob_frame, projected_points)

draw_point(blob_frame, corners[0], (255, 0, 255))
draw_point(blob_frame, corners[1], (255, 255, 255))
draw_point(blob_frame, corners[2], (0, 255, 255))
draw_point(blob_frame, corners[3], (0, 0, 0))

draw_point(blob_frame, projected_points[0][0], (255, 0, 255))
draw_point(blob_frame, projected_points[1][0], (255, 255, 255))
draw_point(blob_frame, projected_points[2][0], (0, 255, 255))
draw_point(blob_frame, projected_points[3][0], (0, 0, 0))

draw_point(blob_frame, np.mean(corners, axis=0), (0, 0, 255))

idx = 0
for point in corners:
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(point)
    cv2.putText(blob_frame, f'{idx}',
                tuple(point.astype(int)),
                font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    idx = idx+1

toe_point = np.array([915, 980], np.float32)
draw_point(blob_frame, toe_point)

################################
# Calculate 3D position of toe #
################################
# calculate the 3d direction of the ray in camera coordinate frame
toe_point_norm = cv2.undistortPoints(toe_point, camera_matrix, None)[0][0]
ray_dir_cam = np.array([toe_point_norm[0], toe_point_norm[1], 1])

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
draw_point(blob_frame, points[0][0], (255, 255, 255))
# print(ray_dir_cam*d_intersection)
intersection_point[2] = -185*10
points, jac = cv2.projectPoints(intersection_point,
                                rvec, tvec, camera_matrix, None)
draw_point(blob_frame, points[0][0], (255, 255, 255))
top_point = np.array([915, 100.17715341])

cv2.imwrite('res/blobs.png', blob_frame)

#####################
# FIND BOUNDING BOX #
#####################
# Initialization
tl = 2
tf = 1

# Model
# For PIL/cv2/np inputs and NMS
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                       pretrained=True).autoshape()
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = model(frame_rgb, 320)  # includes NMS

person_coordinates = []
points = results.xyxy[0].numpy()
for x in points:
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    color = colors[int(x[5])]
    draw_point(blob_frame, x[2:], color)
    cv2.rectangle(blob_frame, c1, c2, color,
                  thickness=tl, lineType=cv2.LINE_AA)
    label = names[int(x[5])]
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    if label == 'person':
        a = np.array([x[0], x[1]])
        b = np.array([x[2], x[1]])
        c = np.array([x[2], x[3]])
        d = np.array([x[0], x[3]])
        person_coordinates.append([a, b, c, d])
        print(f'label: {label}')
        print(f'y distance: {x[3]-x[1]}')
        print(f'x distance: {x[2]-x[0]}')
    cv2.rectangle(blob_frame, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
    cv2.putText(blob_frame, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

person_coordinates = np.array(person_coordinates)

point0 = to_coordinate(person_coordinates[0][3], rvec, tvec)
point1 = to_coordinate(person_coordinates[0][2], rvec, tvec)
points, jac = cv2.projectPoints(point0,
                                rvec, tvec, camera_matrix, None)
draw_point(blob_frame, points[0][0], (255, 255, 255))
points, jac = cv2.projectPoints(point1,
                                rvec, tvec, camera_matrix, None)
draw_point(blob_frame, points[0][0], (255, 255, 255))

bounding_span = point1-point0
up_vector = np.array([0, 0, -1])
# calculate normal vector of plane
bounding_span = bounding_span/norm(bounding_span)
plane_normal = np.cross(bounding_span, up_vector)
# to unit vector
plane_normal = plane_normal/norm(plane_normal)
plane_d = -np.dot(plane_normal, point1)

plane_point = (person_coordinates[0][0]+person_coordinates[0][1])/2
plane_norm_dir = cv2.undistortPoints(plane_point, camera_matrix, None)[0][0]
ray_dir_cam = np.array([plane_norm_dir[0], plane_norm_dir[1], 1])
ray_dir_cam = ray_dir_cam/norm(ray_dir_cam)
ray_dir_chessboard = np.matmul(rot_chessboard_cam, ray_dir_cam)
ray_origin = pos_cam_chessboard.T[0]

point1 = intersect_line_plane(ray_origin, ray_dir_chessboard,
                              plane_normal, plane_d)
points, jac = cv2.projectPoints(point1, rvec, tvec, camera_matrix, None)
draw_point(blob_frame, points[0][0], (0, 255, 255))

cv2.imwrite('res/blobs.png', blob_frame)
exit(0)

####################
# Depth Estimation #
####################
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(img, resize=(256, 320)):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if resize:
        img = imresize(img, resize)

    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(
        0) / 255 - 0.45) / 0.225).to(device)
    print(f'tensor shape {tensor_img.shape}')
    return tensor_img


def prediction_to_visual(output, shape=(360, 640)):
    pred_disp = output.cpu().numpy()[0, 0]
    img = 1 / pred_disp
    img = imresize(img, shape).astype(np.float32)
    return img


@torch.no_grad()
def predict_depth():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='res/dataset1.mp4', help='source')
    parser.add_argument('--output', type=str,
                        default='res/scfm.csv', help='source')
    opt = parser.parse_args()
    print(opt)

    ################
    # Load DispNet #
    ################
    disp_net = scsfm.DispResNet(18, False).to(device)
    weights = torch.load('data/weights/scfm-nyu2-test.pth.tar')
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # cap = cv2.VideoCapture('data/data2.mp4')
    # while True:
    then = time.time()
    frame_rgb = frame_rgb[:, 30:510]
    tgt_img = load_tensor_image(frame_rgb.copy())
    print_duration(then, 'capture and convert')
    then = time.time()
    output = disp_net(tgt_img)
    print_duration(then, 'inference')

    cv2.imshow('frame', frame_rgb)
    cv2.imshow('depth', prediction_to_visual(output))
    cv2.waitKey(-1)


def print_duration(then, prefix=''):
    print(prefix, 'took %.2f ms' % ((time.time() - then) * 1000))


predict_depth()