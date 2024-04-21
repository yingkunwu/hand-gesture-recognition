import os
import json
import glob
import cv2
import random
import torch
import numpy as np
from pathlib import Path
from lightning.pytorch import LightningDataModule
import torchvision.transforms as transforms

from .transforms import get_affine_transform, affine_transform
from .augmentations import fliplr, color_jitter


def to_torch(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.type(torch.float32)
    else:
        raise TypeError("Expects a numpy array but get : {}"
                        .format(type(x)))
    return x


class HagridDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, names, image_size, num_joints, sigma,
                 augments, image_set):
        super().__init__()
        json_file_path = glob.glob(os.path.join(data_dir, "*.json"))
        self.gt_db = self.read_data(json_file_path)

        self.names = names
        self.image_size = image_size
        self.heatmap_size = [s // 4 for s in image_size]
        self.sigma = sigma
        self.num_joints = num_joints

        self.scale_factor = augments.get("scale_factor", 0)
        self.rotate_factor = augments.get("rotate_factor", 0)
        self.translate_factor = augments.get("translate_factor", 0)
        self.horizontal_flip = augments.get("horizontal_flip", False)
        self.color_jittering = augments.get("color_jittering", False)
        self.image_set = image_set

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        data = self.gt_db[idx]
        img = cv2.imread(data["image_path"], cv2.IMREAD_COLOR)
        landmark = np.array(data["landmark"])
        label = np.array(self.names[data["label"]])

        if img is None:
            raise ValueError('Fail to read {}'.format(data["image_path"]))
        h, w = img.shape[:2]

        joints = landmark.copy()
        joints_vis = np.ones((self.num_joints, 1))

        if joints.shape[0]:
            joints[:, 0] = joints[:, 0] * w
            joints[:, 1] = joints[:, 1] * h

        c = np.array([w / 2, h / 2])
        origin_size = max(h, w) * 0.35

        img, joints, joints_vis = self.preprocess(
            img, joints, joints_vis, c, 1, 0, origin_size)

        # convert images to torch.tensor and normalize it
        img = self.transform(img)

        # convert 2d keypoints to heatmaps
        target, target_weight = self.generate_target(joints, joints_vis)

        target = to_torch(target)
        target_weight = to_torch(target_weight)

        # create numpy joints if no joints are found
        if joints.shape[0] == 0:
            joints = np.zeros((self.num_joints, 2))

        meta = {
            'image_path': data["image_path"],
            'joints': joints,
            'joints_vis': joints_vis,
        }

        return img, label, target, target_weight, meta

    def __len__(self):
        return len(self.gt_db)

    def preprocess(self, image, joints, joints_vis, c, s, r, origin_size):
        """
        Resize images and joints accordingly for model training.
        If in training stage, random flip, scale, and rotation will be applied.

        Args:
            image: input image
            joints: ground truth keypoints: [num_joints, 3]
            joints_vis: visibility of the keypoints: [num_joints, 3],
                        (1: visible, 0: invisible)
            c: center point of the cropped region
            s: scale factor
            r: degree of rotation
            origin_size: original size of the cropped region
        Returns:
            image, joints, joints_vis (after preprocessing)
        """
        if self.image_set == 'train':
            sf = self.scale_factor
            rf = self.rotate_factor
            tf = self.translate_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5:
                h, w, _ = image.shape
                c[0] += w * np.clip(np.random.randn() * tf, -tf * 2, tf * 2)
                c[1] += h * np.clip(np.random.randn() * tf, -tf * 2, tf * 2)

            if self.color_jittering and random.random() <= 0.5:
                image = color_jitter(image)

            if self.horizontal_flip and random.random() <= 0.5:
                image, joints = fliplr(image, joints, image.shape[1])
                c[0] = image.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, origin_size, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(joints.shape[0]):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        return image, joints, joints_vis

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]), dtype=np.float32)

        # return all zeros if no joints are found
        if joints.shape[0] == 0:
            target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
            return target, target_weight

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = [i / h for (i, h) in
                           zip(self.image_size, self.heatmap_size)]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] \
                or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 \
                    or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized,
            # we want the center value to equal 1
            g = np.exp(
                - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def read_data(self, json_file_path):
        if len(json_file_path) == 0:
            assert False, "json files which store annotations are not found"

        gt_db = []
        for json_path in json_file_path:
            f = open(json_path)
            data = json.load(f)
            root = Path(json_path).parents[2]
            name = Path(json_path).stem

            for image_id, annots in data.items():
                image_path = os.path.join(root, name, image_id + ".jpg")

                gt_db.append({
                    'image_path': image_path,
                    'landmark': annots['landmark'],
                    'label': annots['label']
                })

        return gt_db


class HandDataModule(LightningDataModule):
    def __init__(self, data_cfg, image_size, batch_size, sigma, num_workers):
        super().__init__()
        self.train_data_path = os.path.join(
            data_cfg['path'], data_cfg['train'])
        self.val_data_path = os.path.join(
            data_cfg['path'], data_cfg['val'])
        self.test_data_path = os.path.join(
            data_cfg['path'], data_cfg['test'])

        self.num_joints = data_cfg["num_joints"]
        self.num_classes = data_cfg["num_classes"]
        self.names = data_cfg["names"]
        self.augments = data_cfg["augments"]

        self.image_size = image_size
        self.batch_size = batch_size
        self.sigma = sigma
        self.num_workers = num_workers

        self.dataset = HagridDataset

    def setup(self, stage=None):
        self.train_dataset = self.dataset(
            self.train_data_path,
            self.names,
            self.image_size,
            self.num_joints,
            self.sigma,
            self.augments,
            "train")
        self.valid_dataset = self.dataset(
            self.val_data_path,
            self.names,
            self.image_size,
            self.num_joints,
            self.sigma,
            self.augments,
            "val")

        self.test_dataset = self.dataset(
            self.test_data_path,
            self.names,
            self.image_size,
            self.num_joints,
            self.sigma,
            self.augments,
            "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
