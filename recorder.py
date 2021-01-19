import numpy as np
import cv2

cap = cv2.VideoCapture('https://192.168.0.164:8080/video')
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('res/dataset2.mp4', fourcc, fps, (640,480))

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)
    else:
        break
    count = count+1
    if count == 300:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()