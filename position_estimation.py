import cv2
import numpy as np

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


    # Display the resulting frame
    cv2.imshow('frame', frame)
    # camera.schedule_frame(cv2.cvtColor(blob_frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
    # out.write(blob_frame)

# When everything done, release the capture
# out.release()
cap.release()
cv2.destroyAllWindows()