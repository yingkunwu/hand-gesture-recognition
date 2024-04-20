import os
import glob
import json
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import mediapipe as mp
import numpy as np

from libs.transforms import get_affine_transform, affine_transform


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # If the intersection is non-existent, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate the IoU (Intersection over Union)
    iou = intersection_area / union_area

    return iou


class HandPoseEstimator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

    def __call__(self, img):
        landmarks, landmark_bbox = [], []
        with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:

            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image
            results = hands.process(img_rgb)

        # Check if any hands were detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joints = []
                for joint in hand_landmarks.landmark:
                    joints.append([joint.x, joint.y])

                landmarks.append(joints)

            landmarks = np.asarray(landmarks)
            landmarks[:, :, 0] = landmarks[:, :, 0] * img.shape[1]
            landmarks[:, :, 1] = landmarks[:, :, 1] * img.shape[0]

            for joint in landmarks:
                x_min = np.min(joint[:, 0])
                y_min = np.min(joint[:, 1])
                x_max = np.max(joint[:, 0])
                y_max = np.max(joint[:, 1])
                w = x_max - x_min
                h = y_max - y_min
                landmark_bbox.append([x_min, y_min, w, h])

        return landmarks, landmark_bbox


class HagridDataExtractor:
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.pose_estimator = HandPoseEstimator()

    def extract(self, json_file_path):
        json_files = glob.glob(
            os.path.join(self.root_dir, json_file_path, "*.json"))

        for f in json_files:
            with open(f) as json_file:
                data = json.load(json_file)
            name = Path(f).stem
            action = Path(json_file_path).stem

            image_save_path = os.path.join(self.output_dir, name)
            os.makedirs(image_save_path, exist_ok=True)

            annots_save_path = os.path.join(
                self.output_dir, "annotations", action)
            os.makedirs(annots_save_path, exist_ok=True)

            # using the pose estimator (mediapipe) to get the landmark
            # of the hands and save the label and landmark to new annotations
            new_annots = {}

            count = 0
            for image_id, annots in tqdm(data.items()):
                img = cv2.imread(
                    os.path.join(self.root_dir, name, image_id + ".jpg"))
                img_height, img_width, _ = img.shape

                landmarks, landmark_bbox = self.pose_estimator(img)

                for idx, (bbox, label) in enumerate(
                        zip(annots["bboxes"], annots["labels"])):
                    x, y, w, h = bbox

                    x = int(x * img_width)
                    y = int(y * img_height)
                    w = int(w * img_width)
                    h = int(h * img_height)

                    joints = np.zeros((0, 2))
                    for i, l_bbox in enumerate(landmark_bbox):
                        iou = calculate_iou([x, y, w, h], l_bbox)
                        if iou > 0.5:
                            joints = landmarks[i]

                    c = np.array([x + w / 2, y + h / 2], dtype=np.float32)
                    original_size = max(w, h)
                    target_size = [original_size, original_size]
                    trans = get_affine_transform(
                        c, 3, 0, original_size, target_size)
                    img_crop = cv2.warpAffine(
                        img,
                        trans,
                        target_size,
                        flags=cv2.INTER_LINEAR)

                    for i in range(joints.shape[0]):
                        joints[i] = affine_transform(joints[i], trans)

                        joints[i, 0] /= target_size[0]
                        joints[i, 1] /= target_size[1]

                    # save cropped images
                    cv2.imwrite(os.path.join(
                        image_save_path, image_id + f"-{idx}.jpg"), img_crop)

                    # store annotations
                    new_annots[image_id + f"-{idx}"] = {
                        "label": label,
                        "landmark": joints.tolist()
                    }

                if count > 100:
                    break
                count += 1

            # save new annotations to .json file
            with open(os.path.join(
                    annots_save_path, name + ".json"), "w") as f:
                json.dump(new_annots, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='', help='root directory of data')
    parser.add_argument('--output_dir', type=str,
                        default='data/hagrid_pose', help='output directory')
    args = parser.parse_args()
    print(args)

    extractor = HagridDataExtractor(args.root_dir, args.output_dir)
    extractor.extract("annotations/train")
    extractor.extract("annotations/val")
    # extractor.extract("annotations/test")
