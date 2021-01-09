import cv2
from util.timer import Timer

class Webcam():
    def __init__(self, source, callback=None):
        self.cap = cv2.VideoCapture(source)
        #self.cap = cv2.VideoCapture('https://192.168.0.165:8080/video')
        self.timer = Timer()
        self.callback = callback

    def show(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if not ret:
                raise Exception('Camera has an issue!')

            # Our operations on the frame come here
            self.timer.start()
            if self.callback != None:
                frame = self.callback(frame)

            # Display the resulting frame
            self.timer.stop()

            cv2.putText(frame, str(self.timer.fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def release(self):
        self.cap.release()