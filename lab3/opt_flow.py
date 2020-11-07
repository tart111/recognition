import numpy as np
import cv2
import pickle
import datetime
import time
import tqdm
from ebma import decoder
from ebma import ORB_comparison

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

def optical_flow(img1, img2):
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    old_frame = img1
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame = img2
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    res = np.zeros((len(good_new), 4), dtype=np.int)
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        res[i] = [a, b, c, d]
        mask = cv2.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)

        frame = cv2.circle(frame, (a, b), 5,
                           color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    # cv2.imshow('frame', ResizeWithAspectRatio(img, 400))
    # cv2.imshow('old', ResizeWithAspectRatio(img1, 400))
    # cv2.imshow('new', ResizeWithAspectRatio(img2, 400))
    # # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return res

def draw_result(img, transform):
    eps = 30
    new_img = img.copy()
    for i in range(len(transform)):
        b = transform[i][0]
        a = transform[i][1]
        d = transform[i][2]
        c = transform[i][3]

        a = max(a, 0)
        b = max(b, 0)
        c = min(c, img.shape[1])
        d = min(d, img.shape[0])

        eps1 = min(eps, b, d, img.shape[1]-b, img.shape[1]-d)
        eps2 = min(eps, a, c, img.shape[0] - a, img.shape[0] - c)

        # print(f'a:{a}, c: {c}, {img.shape}')
        # print(eps1, eps2)

        new_img[max(0, a - eps2): min(img.shape[0], a + eps2), max(0, b - eps1): min(img.shape[1], b + eps1)] = \
            img[max(0, c - eps2): min(img.shape[0], c + eps2), max(0, d - eps1): min(img.shape[1], d + eps1)]
    return new_img

def encoder(cap):
    time_start = datetime.datetime.now()
    with open("result_ebma.txt", 'w'):
        pass
    compressed_outputs = []
    frames = []
    ind = 0
    while cap.isOpened():
        print(f'Reading frame {ind}')
        ind += 1
        frame = cap.read()[1]
        if frame is None:
            break
        frames.append(frame)
        # time.sleep(0.1)

    for ind in tqdm.trange(len(frames) // 2, desc='Processing frames'):
        frame = frames[ind * 2]
        next_frame = frames[ind * 2 + 1]
        res = optical_flow(frame, next_frame)
        compressed_outputs.append([frame, res])
        ORB_comparison(frame, next_frame, "result_ebma.txt")

    with open('myfile1.pkl', 'wb') as output:
        pickle.dump(compressed_outputs, output)

    time_res = (datetime.datetime.now() - time_start)
    print(f"Optical flow total time: {time_res.seconds} seconds\nOptical flow time per frame: "
          f"{(time_res.microseconds) / len(frames)} microseconds\n")


# img1 = cv2.imread('11.jpg', 1)
# img2 = cv2.imread('12.jpg', 1)
# result = optical_flow(img1, img2)
# pict = draw_result(img1, result)
# cv2.imshow('shiiiiit', ResizeWithAspectRatio(pict, 400))
# cv2.waitKey(0)

cap = cv2.VideoCapture("VID.mp4")
encoder(cap)
cap.release()

decoder("myfile1.pkl", "images2", draw_result)