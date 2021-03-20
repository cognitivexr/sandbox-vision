import random
import time

import cv2
import torch


class Yolov5:

    def __init__(self, device=None) -> None:
        super().__init__()

        self.device = device
        self.model = None
        self.names = []
        self.colors = []

    def load(self):
        if self.device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    @torch.no_grad()
    def inference(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, 320 + 32 * 4)  # includes NMS
        return results

    def visualize(self, detections: 'models.common.Detections', frame, conf_th=None):
        tl = 2
        tf = 1
        white = (255, 255, 255)
        black = (0, 0, 0)

        points = detections.xyxy[0].cpu().numpy()

        for point in points:
            xyxy, conf, cls = point[:4], point[4], int(point[5])

            if conf_th is not None and conf < conf_th:
                continue

            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

            print(f'{time.time():0.4f},{detections.names[cls]},{conf:.4f},{c1[0]},{c1[1]},{c2[0]},{c2[1]}')

            # draw rectangle
            cv2.rectangle(frame, c1, c2, self.colors[cls], thickness=tl, lineType=cv2.LINE_AA)

            # draw label
            label = f'{detections.names[cls]} ({conf * 100:.0f}%)'
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c3 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(frame, c1, c3, black, - 1, cv2.LINE_AA)  # label background
            cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3, white, thickness=tf, lineType=cv2.LINE_AA)

        return frame

def main():
    model = Yolov5()
    model.load()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            more, frame = cap.read()

            if not more:
                break

            objects = model.inference(frame)
            vis = model.visualize(objects, frame.copy(), conf_th=0.3)

            cv2.imshow('source', frame)
            cv2.imshow('detect', vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return


    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == '__main__':
    main()
