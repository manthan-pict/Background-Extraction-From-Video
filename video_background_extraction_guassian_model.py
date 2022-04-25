import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os


vid = cv2.VideoCapture(os.path.join("input_vidoes", "eastbound_0.mp4"))

frames = []
frame_count = 0

while True:
    ret, frame = vid.read()
    if frame is not None:
        frames.append(frame)
        frame_count += 1
    else:
        break

# images = os.listdir("frames")
# frame_list = [file for file in os.listdir("frames") if file.endswith('.jpg')]
#
# for img in frame_list:
#     print(img)
#     frames.append(cv2.imread(os.path.join("frames", img)))


frames = np.array(frames)

# cv2.imwrite('background.png', frames[0])
# cv2.imshow(" background", frames[0])
# cv2.waitKey(0)

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
cv2.imwrite('background_guassian_with_stabilazation.png', background)
cv2.imshow("background", background)
cv2.waitKey(0)


