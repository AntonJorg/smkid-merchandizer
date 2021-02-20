import time
import cv2
from PIL import Image
import numpy as np
import queue
import threading

# Specify the paths for the 2 files
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                  self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                  pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def calculate_keypoints(frame, blob_size=368):

    original_height, original_width, _ = frame.shape

    img = Image.fromarray(frame)
    img = img.resize((blob_size, blob_size))
    frame = np.asarray(img)

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (blob_size, blob_size), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(15):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (blob_size * point[0]) / W
        y = (blob_size * point[1]) / H

        points.append((int(x), int(y)))

    points = [(int(x / blob_size * original_width), int(y / blob_size * original_height)) for (x, y) in points]

    return pointshackathon


def annotate_frame(frame, keypoints):
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)

    POSE_PAIRS = ((0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                  (1, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13))

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if keypoints[partA] and keypoints[partB]:
            cv2.line(frame, keypoints[partA], keypoints[partB], (0, 255, 0), 3)

    return frame


if __name__ == "__main__":

    cap = VideoCapture(0)

    while True:
        # Capture frame-by-frame

        frame = cap.read()

        keypoints = calculate_keypoints(frame, blob_size=183)

        frame = annotate_frame(frame, keypoints)
        img = Image.fromarray(frame)
        img = img.resize((1200, 720))
        frame = np.asarray(img)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.cap.release()
    cv2.destroyAllWindows()

    # frame = cv2.imread("test.jpg")
    # kp, kpf = apply_keypoints(frame)
    # cv2.imwrite("test_kp.jpg", kpf)
    # print(kp)

