import numpy as np
import cv2


class HandPreprocess:
    def __init__(self, img_size, num_joints, preprocess):
        self.img_size = img_size
        self.num_joints = num_joints

        self.resize_FOA = preprocess['resize_FOA']
        self.rotate = preprocess['rotate']
        self.hsv = preprocess['hsv']

    def apply(self, img, bbox, landmark):
        img, landmark = self.crop_image(img, bbox, landmark)
        if self.hsv and np.random.rand() > 0.5:
            img = self.hsv_(img)

        return img, landmark

    def crop_image(self, img, bbox, landmark):
        height, width, _ = img.shape

        x1, y1, w, h = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)

        landmark[:, 0] = landmark[:, 0] * width
        landmark[:, 1] = landmark[:, 1] * height

        new_img = None
        if self.resize_FOA:
            x2, y2 = x1 + w, y1 # top right corner
            x3, y3 = x1 + w, y1 + h # bottom right corner
            x4, y4 = x1, y1 + h # bottom left corner
            if self.rotate:
                x = int((x1 + x3) / 2)
                y = int((y1 + y3) / 2)

                bbox_coord = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                angle = (np.random.rand() - 0.5) * 60
                M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                img = cv2.warpAffine(img, M, (width, height))

                landmark = cv2.transform(landmark.reshape(1, self.num_joints, 2), M).reshape(self.num_joints, 2)
                bbox_coord = cv2.transform(bbox_coord.reshape(1, 4, 2), M).reshape(4, 2)

                if angle > 0:
                    x1, y1 = bbox_coord[0][0], bbox_coord[1][1]
                    x3, y3 = bbox_coord[2][0], bbox_coord[3][1]
                else:
                    x1, y1 = bbox_coord[3][0], bbox_coord[0][1]
                    x3, y3 = bbox_coord[1][0], bbox_coord[2][1]

                x1, y1 = max(x1, 0), max(y1, 0)
                x3, y3 = min(x3, width), min(y3, height)
                w, h = x3 - x1, y3 - y1

            new_img = cv2.resize(img[y1:y3, x1:x3], (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        else:
            x2, y2 = x1 + w, y1 + h
            size = max(h, w)
            new_img = np.zeros((size, size, 3), dtype=np.uint8)
            new_img[:h, :w] = img[y1:y2, x1:x2]
            w, h = size, size

        landmark[:, 0] = (landmark[:, 0] - x1) / w
        landmark[:, 1] = (landmark[:, 1] - y1) / h

        return new_img, landmark

    def hsv_(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(img)

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(np.uint8)
            lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
            lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img
