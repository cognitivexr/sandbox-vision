import numpy as np
import cv2

cap = cv2.VideoCapture('https://192.168.0.164:8080/video')
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('res/webcam_data.mp4', fourcc, fps, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()