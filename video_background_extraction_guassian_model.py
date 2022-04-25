import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os

def extract_background(input_video_path):
    filename = os.path.basename(input_video_path)
    files = filename.split(".")
    vid = cv2.VideoCapture(input_video_path)
    frames = []
    frames.append(cv2.imread(os.path.join("extracted_background", "eastbound_0_2.jpg")))
    frames.append(cv2.imread(os.path.join("extracted_background", "eastbound_0_2.jpg")))
    frames.append(cv2.imread(os.path.join("extracted_background", "eastbound_0_2.jpg")))
    frames.append(cv2.imread(os.path.join("extracted_background", "eastbound_0_2.jpg")))
    frame_count = 4
    # while True:
    #     ret, frame = vid.read()
    #     if frame is not None:
    #         frames.append(frame)
    #         frame_count += 1
    #     else:
    #         break
    frames = np.array(frames)
    print(frames.shape)
    gmm = GaussianMixture(n_components = 2)
    # initialize a dummy background image with all zeros
    background = np.zeros(shape=(frames.shape[1:]))
    for i in range(frames.shape[1]):
        for j in range(frames.shape[2]):
            for k in range(frames.shape[3]):
                X = frames[:, i, j, k]
                X = X.reshape(X.shape[0], 1)
                gmm.fit(X)
                means = gmm.means_
                covars = gmm.covariances_
                weights = gmm.weights_
                idx = np.argmax(weights)
                background[i][j][k] = int(means[idx])

    # Store the result onto disc
    print("completed background Extraction")
    path = os.path.join("extracted_background", files[0] + ".jpg")
    cv2.imwrite(path, background)
    cv2.imshow("background", background)
    cv2.waitKey(0)


extract_background("")