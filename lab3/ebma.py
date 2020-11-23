import cv2 as cv
import numpy as np
import datetime
import pickle
import tqdm
import time
import os
import subprocess as sp

def ORB_comparison(img1, img2, file_name):
    eps = 30
    delta = 60
    k = 0
    p = 0
    n = 0
    time_start = datetime.datetime.now()

    # ORB Detector
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
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

    # matching_result = cv.drawMatches(img1, kp1, img2, kp2, matches[:n], None, flags=2)

    h, w, _ = img2.shape

    time_res = (datetime.datetime.now() - time_start)*1000000000/(h*w)

    with open(file_name, 'a') as file:
        file.write(f"{percent}\taverage error: {distance}\ttime per pixel: {time_res.seconds} nano secs\n")

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
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

    return cv.resize(image, dim, interpolation=inter)

def compare(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err

def check_block(n, img1, img2, m, k):
    square_size = 3 #как много соседних проверяются на совместимость
    ind = np.array([m, k]) #какое перемещение совершено блоком
    width = img1.shape[1] // n
    height = img1.shape[0] // n
    err = compare(img1[m * height: (m+1) * height, k * width: (k+1) * width], img2[m * height: (m+1) * height, k * width: (k+1) * width])
    for i in range(max(0, m-square_size), min(m+square_size, n)):
        for j in range(max(0, k-square_size), min(k+square_size, n)):
            err_new = compare(img1[height * i: height * (i + 1), width * j: width * (j + 1)],
                              img2[height * i: height * (i + 1), width * j: width * (j + 1)])
            if (err_new < err):
                ind[0] = i
                ind[1] = j
    return ind

def ebma(img1, img2):
    n = 30
    err = np.zeros((n, n, 2), dtype=np.int)
    for m in range(n):
        for k in range(n):
            err [m][k]=check_block(n, img1, img2, m, k)
    return err

def draw_result(img, transform):
    n = transform.shape[0]
    new_img = img.copy()
    height = img.shape[0]//n
    width = img.shape[1]//n
    for m in range(n):
        for k in range(n):
            new_img[m * height: (m+1) * height, k * width: (k+1) * width] = img[transform[m, k, 0] * height: (transform[m, k, 0]+1) * height, transform[m, k, 1] * width: (transform[m, k, 1]+1) * width]
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
        res = ebma(frame, next_frame)
        compressed_outputs.append([frame, res])
        # ORB_comparison(frame, next_frame, "result_ebma.txt")

    with open('myfile1.pkl', 'wb') as output:
        pickle.dump(compressed_outputs, output)

    time_res = (datetime.datetime.now() - time_start)
    print(f"EBMA total time: {time_res.seconds} seconds\nEBMA time per frame : {(time_res.microseconds)/len(frames)} microseconds\n")

def decoder(file_name, dir_name, draw_func, cap):
    time_start = datetime.datetime.now()

    frames = []
    ind = 0
    while cap.isOpened():
        print(f'Reading frame {ind}')
        ind += 1
        frame = cap.read()[1]
        if frame is None:
            break
        frames.append(frame)

    print(f'length of frames {len(frames)}')

    video_frames = []
    changes = []
    with open(file_name, 'rb') as file:
        total = pickle.load(file)

    os.makedirs(dir_name, exist_ok=True)

    for ind, frame_pair in enumerate(tqdm.tqdm(total, desc='Processing source frames')):
        video_frames.append(frame_pair[0])
        changes.append(frame_pair[1])
        if ind * 2 + 1 < len(frames):
            err = compare(draw_func(video_frames[-1], changes[-1]), frames[ind * 2 + 1])
            with open('result_ebma.txt', 'a') as file:
                file.write(f"frames difference: {err}\n")
        cv.imwrite(f'{dir_name}/img{ind * 2:03}.png', video_frames[-1])
        cv.imwrite(f'{dir_name}/img{ind * 2 + 1:03}.png', draw_func(video_frames[-1], changes[-1]))

    source_rate = 30
    sp.run(
        rf'C:\\Users\\ASUS\\Documents\\ffmpeg\\bin\\ffmpeg.exe -r {source_rate} '
        rf'-start_number 0 -i {dir_name}\\img%03d.png '
        rf'-c:v libx264 -r 30 -y -pix_fmt yuv420p out1.mp4')
    time_res = (datetime.datetime.now() - time_start)
    print(f"EBMA decoder total time: {time_res.seconds} seconds\n")


if __name__ == '__main__':
    # img1 = cv.imread('11.jpg', 0)
    # img2 = cv.imread('12.jpg', 0)
    # res = ebma(img1, img2)
    # print(res)
    #
    # img = draw_result(cv.imread('11.jpg', 1), res)
    # img = cv.imshow('someshit', ResizeWithAspectRatio(img, 500))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #
    cap = cv.VideoCapture("VID.mp4")
    encoder(cap)
    cap.release()
    cap = cv.VideoCapture("VID.mp4")
    decoder("myfile1.pkl", "images1", draw_result, cap)
