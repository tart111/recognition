import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

img = cv2.imread('opencv_frame_0.png', 0)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img = cv2.line(img, (10, 10), (250, 250), (0, 0, 255), 5)
img = cv2.rectangle(img, (0, 0), (300,100), (255, 0, 0), 3)
img = cv2.imshow('gray', img)
cv2.waitKey(0)
cv2.destroyAllWindows()