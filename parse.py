import numpy as np
# Head - 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7,
# Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, Left Ankle – 13,
# Chest – 14, Background – 15

used = [1, 2, 3, 4, 5, 6, 7, 8, 11, 14]
def parse(coordinates):
    return np.array(coordinates)[used,:]
