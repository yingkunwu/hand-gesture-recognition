
from curses import halfdelay
import numpy as np
import os
import json
import cv2
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[:size_h, :size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class HandDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        super().__init__()
        self.classes = {"call" : 0, "dislike" : 1, "fist" : 2, "like" : 3, "mute" : 4, 
                            "ok" : 5, "one" : 6, "palm" : 7, "peace" : 8, "stop" : 9}
        json_file_path = glob.glob(os.path.join(data_dir, "**/*.json"))
        images, bboxes, landmarks, labels = self.read_data(json_file_path)
        self.imgs = images
        self.bboxes = bboxes
        self.landmarks = landmarks
        self.labels = labels
        self.transforms = transforms
        self.stride = 8
        
    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        landmark = self.landmarks[idx]
        label = np.array(self.labels[idx])

        img = Image.open(image_path).convert('RGB')
        # apply Transforms on image
        img = self.transforms(img)

        _, width, height = img.shape
        w, h = int(width / self.stride), int(height / self.stride)
        heatmap = np.zeros((h, w, len(landmark) + 1), dtype=np.float32)
        for i in range(len(landmark)):
            # resize from 368 to 46
            x, y = landmark[i][0], landmark[i][1]
            heat_map = guassian_kernel(size_h=h, size_w=w, center_x=x, center_y=y, sigma=3.0)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        # for background
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        return img, heatmap, label  
            
    def __len__(self):
        return len(self.imgs)

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

        return images, bboxes_output, landmarks_output, labels_output


def load_data(data_dir, batch_size):
    # Transformer
    train_transformer = transforms.Compose([
        transforms.Resize((368, 368)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_set = HandDataset(data_dir, train_transformer)
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_set, valid_set, train_dataloader, val_dataloader
