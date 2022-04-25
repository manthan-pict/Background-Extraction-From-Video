import numpy as np
import cv2
import os


def extract_background(video_path):
    filename = os.path.basename(video_path)
    files = filename.split(".")
    video = cv2.VideoCapture(video_path)
    FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=60)
    frames = []

    # creating an array of frames from frames chosen above
    for frameOI in FOI:
        video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = video.read()
        frames.append(frame)

    # calculate the average
    backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    path = os.path.join("extracted_background", files[0] + "_2.jpg")
    print(path)
    cv2.imwrite(path, backgroundFrame)
    cv2.imshow("background", backgroundFrame)
    cv2.waitKey(0)


def extract_background_for_folder(folder_path):
    videos = os.listdir(folder_path)
    for v in videos:
        extract_background(os.path.join(folder_path, v))

extract_background_for_folder("input_vidoes")