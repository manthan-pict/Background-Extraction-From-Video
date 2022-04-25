import numpy as np
import cv2
import os


def extract_background(video_path):
    filename = os.path.basename(video_path)
    files = filename.split(".")
    video = cv2.VideoCapture(video_path)
    FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=60)
    frames = []

    # Uncomment if input is folder of frames and not video
    # path = "tee"
    # for i in range(3):
    #     frames.append(cv2.imread(os.path.join(path, random.choice([
    #     x for x in os.listdir(path) if x.endswith('.jpg') and
    #     os.path.isfile(os.path.join(path, x))]))))
    #     print(random.choice([
    #     x for x in os.listdir(path) if x.endswith('.jpg') and
    #     os.path.isfile(os.path.join(path, x))]))

    # creating an array of frames from frames chosen above
    for frameOI in FOI:
        video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = video.read()
        frames.append(frame)

    # print(np.array(frames).shape)

    # calculate the average
    backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    path = os.path.join("extracted_background", files[0] + "_2.jpg")
    print(path)
    cv2.imwrite(path, backgroundFrame)
    cv2.imshow("background", backgroundFrame)
    cv2.waitKey(0)


videos = os.listdir("input_vidoes")

for v in videos:
    print(v)
    extract_background(os.path.join("input_vidoes", v))
