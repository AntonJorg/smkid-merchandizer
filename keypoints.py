import time
import cv2
from PIL import Image
import numpy as np

# Specify the paths for the 2 files
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def apply_keypoints(frame, threshold=0):

    original_height, original_width, _ = frame.shape

    img = Image.fromarray(frame)
    img = img.resize((368, 368))
    frame = np.asarray(img)

    # Specify the input image dimensions
    blobWidth = 368
    blobHeight = 368

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (blobWidth, blobHeight), (0, 0, 0), swapRB=False, crop=False)

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
        x = (blobWidth * point[0]) / W
        y = (blobHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    POSE_PAIRS = ((0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                  (1, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13))

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)

    img = Image.fromarray(frame)
    img = img.resize((original_width, original_height))
    frame = np.asarray(img)

    points = [(int(x/blobWidth * original_width), int(y/blobHeight * original_height)) for (x, y) in points]

    return points, frame


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        _, keypoint_frame = apply_keypoints(frame)

        # Display the resulting frame
        cv2.imshow('frame', keypoint_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    #frame = cv2.imread("test.jpg")
    #kp, kpf = apply_keypoints(frame)
    #cv2.imwrite("test_kp.jpg", kpf)
    #print(kp)

