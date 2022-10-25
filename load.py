
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
from torchvision.datasets import ImageFolder


class HandDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        super().__init__()
        self.classes = {"call" : 0, "dislike" : 1, "fist" : 2, "like" : 3, "mute" : 4, 
                            "ok" : 5, "one" : 6, "palm" : 7, "peace" : 8, "stop" : 9}
        json_file_path = glob.glob(os.path.join(data_dir, "**/*.json"))
        images, bboxes, skeletons, labels = self.read_data(json_file_path)
        self.imgs = images
        self.bboxes = bboxes
        self.skeletons = skeletons
        self.labels = labels
        self.transforms = transforms
        
    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        img = Image.open(image_path).convert('RGB') # 避免遇到灰階圖

        img = np.array(img)
        box = np.array(self.bboxes[idx]).flatten()
        height, width, _ = img.shape
        x1, y1 = int(box[0] * width), int(box[1] * height)
        x2, y2 = int((box[0] + box[2]) * width), int((box[1] + box[3]) * height)
        img = img[y1:y2, x1:x2]
        img = Image.fromarray(img)

        
        ### Preparing class label
        label = np.array(self.labels[idx])
        ### Apply Transforms on image
        img = self.transforms(img)
        return img, label  
            
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

            for file_name, gesture in data.items():

                bbox_output = []
                landmark_output = []

                bboxes = data[file_name]["bboxes"]
                labels = data[file_name]["labels"]
                landmarks = data[file_name]["landmarks"]

                images.append(os.path.join(file_path, file_name + ".jpg"))

                for bbox, label, landmark in zip(bboxes, labels, landmarks):
                    #if label in self.classes and len(landmark) > 0:
                    if label in self.classes:
                        bbox_output.append(bbox)
                        labels_output.append(self.classes[label])
                        landmark_output.append(landmark)

                bboxes_output.append(bbox_output)
                landmarks_output.append(landmark_output)

        return images, bboxes_output, landmarks_output, labels_output


def load_data(data_dir, batch_size):
    # Transformer
    train_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
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


def read_write_data(json_file_path, visualize=False):
    for json_path in json_file_path:
        file_path = os.path.split(json_path)[0]
        f = open(json_path)
        data = json.load(f)

        save_path = os.path.join("hagrid/hagrid_resize", os.path.split(file_path)[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        annotations = {}

        num = 0
        for file_name in data.keys():
            if num == 2000:
                break
            bbox = data[file_name]["bboxes"]
            labels = data[file_name]["labels"]
            landmarks = data[file_name]["landmarks"]

            annotations[file_name] = {}
            annotations[file_name]["bboxes"] = bbox
            annotations[file_name]["labels"] = labels
            annotations[file_name]["landmarks"] = landmarks

            image = cv2.imread(os.path.join(file_path, file_name + ".jpg"))
            height, width, _ = image.shape
            resized = cv2.resize(image, (int(width / 4), int(height / 4)), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_path, file_name + ".jpg"), resized)

            num += 1

            if visualize:
                for box, label, landmark in zip(bbox, labels, landmarks):
                    #if label in classes and len(landmark) > 0:
                    x1, y1 = int(box[0] * width), int(box[1] * height)
                    x2, y2 = int((box[0] + box[2]) * width), int((box[1] + box[3]) * height)
                    resized = cv2.resize(image[y1:y2, x1:x2], (24, 24), interpolation = cv2.INTER_AREA)
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    image = cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    for mark in landmark:
                        x = int(mark[0] * width)
                        y = int(mark[1] * height)
                        image = cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        with open(os.path.join(save_path, os.path.split(json_path)[1]), "w") as outfile:
            json.dump(annotations, outfile, ensure_ascii=False, indent=4)
        f.close()
