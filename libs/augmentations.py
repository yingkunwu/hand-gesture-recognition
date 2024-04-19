import cv2
import numpy as np


def fliplr(image, joints, width):
    """Flip the image and joints horizontally.
    Args:
        image: input image
        joints: [num_joints, 2]
        width: image width
    """

    # Flip horizontal
    image = image[:, ::-1, :]

    if joints.shape[0]:
        joints[:, 0] = width - joints[:, 0] - 1

    return image, joints


def color_jitter(img, hgain=0.02, sgain=0.5, vgain=0.4):
    """HSV color-space augmentation.
    Args:
        img: input image
        hgain: random gains for hue
        sgain: random gains for saturation
        vgain: random gains for value
    """
    if hgain or sgain or vgain:
        # random gains
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                            cv2.LUT(sat, lut_sat),
                            cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
