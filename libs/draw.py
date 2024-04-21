import cv2


def draw_bones(img, annotations):
    limbs1 = [[0, 1], [1, 2], [2, 3], [3, 4]]
    limbs2 = [[0, 5], [5, 6], [6, 7], [7, 8]]
    limbs3 = [[0, 9], [9, 10], [10, 11], [11, 12]]
    limbs4 = [[0, 13], [13, 14], [14, 15], [15, 16]]
    limbs5 = [[0, 17], [17, 18], [18, 19], [19, 20]]

    for lim in limbs1:
        img = cv2.line(
            img, annotations[lim[0]], annotations[lim[1]], (33, 41, 48), 3)
    for lim in limbs2:
        img = cv2.line(
            img, annotations[lim[0]], annotations[lim[1]], (65, 75, 86), 3)
    for lim in limbs3:
        img = cv2.line(
            img, annotations[lim[0]], annotations[lim[1]], (96, 106, 116), 3)
    for lim in limbs4:
        img = cv2.line(
            img, annotations[lim[0]], annotations[lim[1]], (134, 143, 152), 3)
    for lim in limbs5:
        img = cv2.line(
            img, annotations[lim[0]], annotations[lim[1]], (168, 173, 180), 3)

    return img


def draw_joints(img, annotations):
    for a in annotations:
        img = cv2.circle(img, a, 1, (0, 165, 255), 3)

    return img
