import cv2


def draw_bones(img, annotations):
    limbs1 = [[0, 1], [1, 2], [2, 3], [3, 4]]
    limbs2 = [[0, 5], [5, 6], [6, 7], [7, 8]]
    limbs3 = [[0, 9], [9, 10], [10, 11], [11, 12]]
    limbs4 = [[0, 13], [13, 14], [14, 15], [15, 16]]
    limbs5 = [[0, 17], [17, 18], [18, 19], [19, 20]]

    for l in limbs1:
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], (255, 255, 255), 2)
    for l in limbs2:
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], (0, 255, 255), 2)
    for l in limbs3:
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], (255, 0, 255), 2)
    for l in limbs4:
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], (255, 255, 0), 2)
    for l in limbs5:
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], (255, 0, 0), 2)
    
    return img


def draw_joints(img, annotations):
    for a in annotations:
        img = cv2.circle(img, a, 1, (155, 155, 155), 10)

    return img