import cv2
import numpy as np

class Flowing:
    def __init__(self):
        self.window = []
        self.dynamics_window = []

    def update(self, frame: np.array, dynamics):
        self.window.append(frame.astype(np.uint32))
        self.window = self.window[-20:]
        self.dynamics_window.append(dynamics.astype(np.uint32()))
        self.dynamics_window = self.dynamics_window[-20:]

    def get_frame(self):
        ghost_len = 10
        result = (sum(self.window[-5:]) // len(self.window[-5:])).astype(np.uint8)
        if len(self.window) > ghost_len:
            dynamics = (sum(self.dynamics_window[:ghost_len]) / ghost_len).astype(np.bool)
            dynamics[:, :, 0] = 0
            dynamics[:, :, 1] = 0
            ghosted = (sum(self.window[:-ghost_len]) // len(self.window[:-ghost_len])).astype(np.uint8) * dynamics
            ghost_coeff = 3
            result = result - result * dynamics * ghost_coeff
            result = (result + ghosted * ghost_coeff).astype(np.uint8).clip(max=255)

        return result


def motion_change():
    static_back = None
    thresh_hold = 5

    # Capturing video
    video = cv2.VideoCapture(0)

    flowing = Flowing()

    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_back is None:
            static_back = gray
            continue


        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, thresh_hold, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        thresh_mask = thresh_frame.astype(np.bool)[:, :, None]
        thresh_mask = np.broadcast_to(thresh_mask, (frame.shape[0], frame.shape[1], 3))
        thresh_mask = np.ascontiguousarray(thresh_mask)

        # proc_frame = frame + thresh_mask * (-frame + 255) // 4
        proc_frame = frame - thresh_mask * frame
        # frame[:, :, 1] = 0
        # frame[:, :, 2] = 0

        flowing.update(cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), thresh_mask)

        cv2.imshow("Difference Frame", flowing.get_frame())


        src1 = frame
        src2 = cv2.cvtColor(thresh_frame,cv2.COLOR_GRAY2RGB)
        src2 = cv2.resize(src2, src1.shape[1::-1])

        dst = cv2.bitwise_and(src1, src2)

        static_back = gray

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            break

    video.release()

    # Destroying all the windows
    cv2.destroyAllWindows()

motion_change()