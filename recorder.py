import cv2

width = 640
height = 360

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
fps = cap.get(cv2.CAP_PROP_FPS)
print(
    "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}"
    .format(fps))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('res/data.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
