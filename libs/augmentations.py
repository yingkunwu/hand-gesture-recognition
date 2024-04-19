import cv2
import numpy as np


def fliplr(image, joints, joints_vis, width):
    """Flip the image and joints horizontally.
    Args:
        joints: [num_joints, 2 or 3]
        joints_vis: [num_joints, 2 or 3]
        width: image width
        matched_parts: pairs of joints to flip
    """
    assert joints.shape[0] == joints_vis.shape[0], \
        'joints and joints_vis should have the same number of joints, ' \
        'current shape is joints={}, joints_vis={}'.format(
            joints.shape, joints_vis.shape)

    # Flip horizontal
    image = image[:, ::-1, :]

    if joints.shape[0]:
        joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    # for pair in matched_parts:
    #     joints[pair[0], :], joints[pair[1], :] = \
    #         joints[pair[1], :], joints[pair[0], :].copy()
    #     joints_vis[pair[0], :], joints_vis[pair[1], :] = \
    #         joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return image, joints, joints_vis


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
