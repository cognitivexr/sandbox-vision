import os

import cv2


def main():
    image_folder = '../res/vid'
    video_name = '../res/video.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images)

    timestamps = [float(img[6:-4]) for img in images]

    fps = len(timestamps) / (max(timestamps) - min(timestamps))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
