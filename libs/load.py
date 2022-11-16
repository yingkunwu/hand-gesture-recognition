import os
import json
import glob
import cv2
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class HandDataset(Dataset):
    def __init__(self, data_dir, classes_dict, img_size, num_joints, sigma):
        super().__init__()
        self.classes = classes_dict
        json_file_path = glob.glob(os.path.join(data_dir, "**/*.json"))
        metadata = self.read_data(json_file_path)

        self.img_paths = metadata['img_paths']
        self.bboxes = metadata['bboxes']
        self.landmarks = metadata['landmarks']
        self.labels = metadata['labels']
        
        self.img_size = img_size
        self.map_size = self.img_size // 4
        self.sigma = sigma

        self.num_joints = num_joints

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        bbox = np.array(self.bboxes[idx])
        landmark = np.array(self.landmarks[idx])
        label = np.array(self.labels[idx])

        # crop objects based on bbox
        img, bbox, landmark = self.crop_image(img, bbox, landmark)

        # generate groundtruth heatmap
        heatmaps = self.gen_heatmap(landmark)
        
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, heatmaps, label, landmark
            
    def __len__(self):
        return len(self.img_paths)

    def crop_image(self, img, bbox, landmark):
        height, width, _ = img.shape
        x1, y1, w, h = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)
        x2, y2 = x1 + w, y1 + h

        size = max(h, w)
        new_img = None
        resize_bbox = True

        if resize_bbox:
            new_img = cv2.resize(img[y1:y2, x1:x2], (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        else:
            new_img = np.zeros((size, size, 3), dtype=np.uint8)
            new_img[:h, :w] = img[y1:y2, x1:x2]
            w, h = size, size

        bbox[0] = (bbox[0] * width - x1) / w
        bbox[1] = (bbox[1] * height - y1) / h

        landmark[:, 0] = (landmark[:, 0] * width - x1) / w
        landmark[:, 1] = (landmark[:, 1] * height - y1) / h

        return new_img, bbox, landmark

    def gen_heatmap(self, landmark):
        target = np.zeros((self.num_joints, self.map_size, self.map_size), dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            mu_x = int(landmark[joint_id][0] * self.map_size)
            mu_y = int(landmark[joint_id][1] * self.map_size)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= self.map_size or ul[1] >= self.map_size \
                        or br[0] < 0 or br[1] < 0:
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2

            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.map_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.map_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.map_size)
            img_y = max(0, ul[1]), min(br[1], self.map_size)

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target

    def read_data(self, json_file_path):
        images = []
        bboxes_output = []
        labels_output = []
        landmarks_output = []

        for json_path in json_file_path:
            file_path = os.path.split(json_path)[0]
            f = open(json_path)
            data = json.load(f)

            for file_name in data.keys():
                bboxes = data[file_name]["bboxes"]
                labels = data[file_name]["labels"]
                landmarks = data[file_name]["landmarks"]

                for bbox, label, landmark in zip(bboxes, labels, landmarks):
                    if label in self.classes and len(landmark) > 0:
                        images.append(os.path.join(file_path, file_name + ".jpg"))
                        bboxes_output.append(bbox)
                        labels_output.append(self.classes[label])
                        landmarks_output.append(landmark)
        
        metadata = {
            'img_paths' : images,
            'bboxes' : bboxes_output,
            'landmarks' : landmarks_output,
            'labels' : labels_output
        }

        return metadata


def load_data(data_path, classes_dict, batch_size, img_size, num_joints, sigma, action):
    if action == "train":
        train_set = HandDataset(data_path, classes_dict, img_size, num_joints, sigma)
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_set, valid_set, train_dataloader, val_dataloader

    elif action == "test":
        test_set = HandDataset(data_path, classes_dict, img_size, num_joints, sigma)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
        return test_set, test_dataloader

    else:
        raise NotImplementedError("please specify the correct action : [train, test]")
