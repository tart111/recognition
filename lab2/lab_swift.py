import numpy as np
import cv2 as cv
import datetime
#import matplotlib.pyplot as plt
img2 = cv.imread('base.jpg', cv.IMREAD_GRAYSCALE)   # trainImage
sift = cv.SIFT_create()
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)
for i in range(1, 121):
    time_start = datetime.datetime.now()
    print("test ", i, ":")
    img1 = cv.imread('{}.jpg'.format(i), cv.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    count = 0
    summ = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            count += 1
            matchesMask[i] = [1, 0]
            summ += abs(m.distance-n.distance)
    error = summ / count
    percentile = (count / len(matches)) * 100
    h, w = img1.shape
    time = (datetime.datetime.now() - time_start) * 1000000000 / (h * w)
    print(percentile, "%", '\t', "Average error = ", error, "\tTime per size = ", time.seconds, " nano sec per pixel")
    # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask,
    #                    flags=cv.DrawMatchesFlags_DEFAULT)
    # # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # plt.imshow(img3,), plt.show()