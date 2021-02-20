from parse import *

coordinates = [(168, 24), (168, 72), (136, 88), (128, 136), (128, 168), (208, 88), (240, 136), (200, 160), (144, 176), (152, 240), (176, 304), (184, 176), (176, 240), (192, 312), (168, 128)]
# input is an array with rows:
# Neck: 0, Right Shoulder: 1, Right Elbow: 2, Right Wrist: 3, Left Shoulder: 4, Left Elbow: 5, Left Wrist: 6,
# Right Hip: 7, Left Hip: 8, Chest: 9

# c is some scaling factor that depends on how far away the subject is.
# w is some scaling factor for the width of arms
c = 5
w = 5.5
def unit_vector(v):
    if v is None or np.linalg.norm(v) == 0:
        return np.array([0, 0])
    else:
        return v / np.linalg.norm(v)

def rotate90(v, clockwise=True):
    return np.array([[0,1],[-1,0]])@v if clockwise else np.array([[0,-1],[1,0]])@v

def arm_box(top, bottom):
    v = bottom - top
    v_hat = unit_vector(v)
    corner0 = c * w * rotate90(v_hat) + top
    corner1 = 2 * top - corner0
    corners = [corner1, corner0, corner0 + v, corner1 + v]
    return np.array(corners).reshape(4,2)

# d is a scaling factor that defines the relative distance from the wrist to the hand and the elbow to the wrist
d = 1.5
def hand(elbow, wrist):
    return d * (wrist - elbow) + wrist

def connect_polygon(overc, forec):
    return np.vstack((overc[2:,:], forec[:2,:]))

def connect_shoulder(torsoc, overc, left=True):
    if left:
        return [torsoc[1], torsoc[1] + overc[1]-overc[0], overc[0], overc[1]]
    else:
        return [torsoc[0], torsoc[0] + overc[0]-overc[1], overc[1], overc[0]]

def halfway(l, r):
    v = r - l
    return l + v/2

def body_line(neck, root, shoulder_L, shoulder_R, hip_L, hip_R):
    half_shoulder = halfway(shoulder_L, shoulder_R)
    half_hip = halfway(hip_L, hip_R)
    xs = [neck[0], root[0], half_shoulder[0], half_hip[0]]
    ys = [neck[1], root[1], half_shoulder[1], half_hip[1]]
    linefit = np.polyfit(xs, ys, 1)
    meanx = np.array(np.mean(xs))
    return linefit, meanx

# g is some standard length of the logo
g = 50
ratio = 1.381
def logo(linefit, meanx):
    a, b = linefit[0], linefit[1]
    bottom = np.array([meanx, meanx * a + b])
    if a < 0:
        theta = np.arctan(-a)
        xtop = meanx - np.cos(theta) * g
    else:
        theta = np.arctan(a)
        xtop = meanx + np.cos(theta) * g
    top = np.array([xtop, xtop * a + b])
    v = top - bottom
    corner0 = rotate90(v, clockwise=False) * ratio / 2 + bottom
    corner1 = rotate90(v) * ratio / 2 + bottom
    corner2 = corner0 + v
    corner3 = corner1 + v
    return np.array([corner0, corner1, corner2, corner3]).reshape(4,2)

# l_bottom, fatness, l_tyrenakke
l_bottom, fatness, l_tyrenakke = 10, 25, 30
def torso(root, shoulder_L, shoulder_R, hip_L, hip_R):
    v_hip, v_shoulder = hip_R - hip_L, shoulder_R - shoulder_L
    v_hip_hat, v_shoulder_hat = unit_vector(v_hip), unit_vector(v_shoulder)
    corner0 = (l_bottom * rotate90(v_hip_hat) - fatness * v_hip_hat) + hip_L
    corner1 = (l_bottom * rotate90(v_hip_hat) + fatness * v_hip_hat) + hip_R
    v_stomach = root - halfway(hip_L, hip_R)
    # corner2 = corner0 + v_stomach
    # corner3 = corner1 + v_stomach
    corner4 = shoulder_L + l_tyrenakke * rotate90(v_shoulder_hat, clockwise = False)
    corner5 = shoulder_R + l_tyrenakke * rotate90(v_shoulder_hat, clockwise=False)
    return np.array([corner5, corner4, corner1, corner0]).reshape((4,2))

coords = parse(coordinates)
# Neck: 0, Right Shoulder: 1, Right Elbow: 2, Right Wrist: 3, Left Shoulder: 4, Left Elbow: 5, Left Wrist: 6,
# Right Hip: 7, Left Hip: 8, Chest: 9

torso_corners = torso(coords[9,:], coords[4,:], coords[1,:], coords[8,:], coords[7,:])
linefit, meanx = body_line(coords[0,:], coords[9,:], coords[4,:], coords[1,:], coords[8,:], coords[7,:])
loverarm = arm_box(coords[4,:], coords[5,:])
roverarm = arm_box(coords[1,:], coords[2,:])
lforearm = arm_box(coords[5,:], coords[6,:])
rforearm = arm_box(coords[2,:], coords[3,:])
logoc = logo(linefit, meanx)
# print(logoc)
# plt.scatter(coords[:,0], -coords[:,1], color="Red")
# plt.scatter(torso_corners[:,0], -torso_corners[:,1], color="Blue")
# plt.scatter(logoc[:,0], -logoc[:,1], color="Green")
# plt.scatter(loverarm[:,0], -loverarm[:,1], color="Magenta")
# plt.scatter(roverarm[:,0], -roverarm[:,1], color="Purple")
# plt.scatter(lforearm[:,0], -lforearm[:,1], color="Cyan")
# plt.scatter(rforearm[:,0], -rforearm[:,1], color="Black")
# plt.show()

import cv2
# building = cv2.imread('testman.jpg')
# dp = cv2.imread('hoodie/forearm_L.png')

def imtransform(original, merch, coords):
    height, width = original.shape[:2]
    h1,w1 = merch.shape[:2]


    pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
    pts2=np.float32(coords)
    positions2 = coords

    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)

    height, width, channels = original.shape
    im1Reg = cv2.warpPerspective(merch, h, (width, height))

    mask2 = np.zeros(original.shape, dtype=np.uint8)

    roi_corners2 = np.int32(positions2)

    channel_count2 = original.shape[2]
    ignore_mask_color2 = (255,)*channel_count2

    cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

    mask2 = cv2.bitwise_not(mask2)
    masked_image2 = cv2.bitwise_and(original, mask2)

    #Using Bitwise or to merge the two images
    final = cv2.bitwise_or(im1Reg, masked_image2)

    # cv2.imshow('sejt', final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final

# imtransform(building,dp,lforearm)


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img