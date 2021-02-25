import cv2
import numpy as np
import pyfakewebcam

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

# %%
# camera = pyfakewebcam.FakeWebcam('/dev/video1', 640, 480)
cap = cv2.VideoCapture('http://192.168.0.164:8080/video')
# out = cv2.VideoWriter('blob_result.mp4', cv2.VideoWriter_fourcc(
#   'm', 'p', '4', 'v'), 30, (640, 480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print('could not read frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect blobs.
    keypoints = blobDetector.detect(gray)

    blob_frame = cv2.drawKeypoints(frame, keypoints,
                                   np.array([]), (0, 255, 0),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Find the circle grid
    # ret, corners = cv2.findCirclesGrid(
    #     blob_frame, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    # if ret:
    #     blob_frame_gray = cv2.cvtColor(blob_frame, cv2.COLOR_BGR2GRAY)
    #     exact_corners = cv2.cornerSubPix(
    #         blob_frame_gray, corners, (11, 11), (-1, -1), criteria)
    #     frame = cv2.drawChessboardCorners(frame, (4, 11), exact_corners, ret)

    # Display the resulting frame
    cv2.imshow('frame', blob_frame)
    # camera.schedule_frame(cv2.cvtColor(blob_frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
    # out.write(blob_frame)

# When everything done, release the capture
# out.release()
cap.release()
cv2.destroyAllWindows()
