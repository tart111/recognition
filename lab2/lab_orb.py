import cv2
import datetime
import numpy as np


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def ORB_comparison(image1, image2):
    eps = 30
    delta = 60
    k = 0
    p = 0
    n = 0
    time_start = datetime.datetime.now()
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    distance = 0
    for i, (m) in enumerate(matches):
        p = p+1
        if m.distance < eps:
            k = k+1
        if m.distance < delta:
            distance = distance + m.distance
            n = n + 1


    #for i, (m) in enumerate(matches[:50]):


    distance = float(distance)/p;

    percent = float(k)/float(p)

    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n], None, flags=2)

    h, w, _ = img2.shape
    #print('width: ', w)
    #print('height:', h)

    time = (datetime.datetime.now() - time_start)*1000000000/(h*w)

    print(percent, "\t", distance, "\t", time.seconds, "nanosecs")
    #cv2.imshow("Img1", ResizeWithAspectRatio(img1, 400))
    # cv2.imshow("Img2", ResizeWithAspectRatio(img2, 400))

    # width:  400
    # height: 225

    # cv2.imshow("Matching result", ResizeWithAspectRatio(matching_result, 900))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

for i in range (1, 121):
    print ("№", i, " ")
    ORB_comparison("base.jpg", "{}.jpg".format(i))
