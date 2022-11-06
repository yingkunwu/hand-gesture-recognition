import os, json, glob, cv2, torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class HandDataset(Dataset):
    def __init__(self, data_dir, img_size, sigma):
        super().__init__()
        self.classes = {"call" : 0, "dislike" : 1, "fist" : 2, "like" : 3, "mute" : 4, 
                            "ok" : 5, "one" : 6, "palm" : 7, "peace" : 8, "stop" : 9}
        self.parts = [
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20]
                    ]
        self.groups1 = [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                    ]
        self.groups6 = [
                        [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [0, 4, 8, 12, 16],
                    ]
        json_file_path = glob.glob(os.path.join(data_dir, "**/*.json"))
        images, bboxes, landmarks, labels = self.read_data(json_file_path)
        self.imgs = images
        self.bboxes = bboxes
        self.landmarks = landmarks
        self.labels = labels
        self.img_size = img_size
        self.stride = 1
        self.map_size = self.img_size // self.stride
        self.sigma = sigma
        
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        bbox = np.array(self.bboxes[idx])
        landmark = np.array(self.landmarks[idx])
        label = np.array(self.labels[idx])

        img, bbox, landmark = self.crop_image(img, bbox, landmark)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)

        heatmap = self.gen_heatmap(landmark)
        lsh_maps = self.generate_lpm(landmark)
        lsh_maps1 = self.limb_group(lsh_maps, 1, self.groups1)
        lsh_maps6 = self.limb_group(lsh_maps, 6, self.groups6)
        lsh_maps = np.concatenate([lsh_maps1, lsh_maps6])
        lsh_maps = torch.from_numpy(lsh_maps)

        return img, heatmap, lsh_maps, label  
            
    def __len__(self):
        return len(self.imgs)

    def crop_image(self, img, bbox, landmark):
        height, width, _ = img.shape
        x = (bbox[0] + bbox[2] / 2) * width
        y = (bbox[1] + bbox[3] / 2) * height

        landmark[:, 0] = landmark[:, 0] * width
        landmark[:, 1] = landmark[:, 1] * height

        size = min(height, width) / 2

        left_bound, right_bound, upper_bound, lower_bound = 0, 0, 0, 0
        if size - x > 0:
            right_bound = size * 2
        elif size - (width - x) > 0:
            right_bound = width
            left_bound = width - (size * 2)
        else :
            left_bound = x - size
            right_bound = x + size

        if size - y > 0:
            lower_bound = size * 2
        elif size - (height - x) > 0:
            lower_bound = height
            upper_bound = height - (size * 2)
        else :
            upper_bound = y - size
            lower_bound = y + size

        img = img[int(upper_bound):int(lower_bound), int(left_bound):int(right_bound)]
        landmark[:, 0] = landmark[:, 0] - left_bound
        landmark[:, 1] = landmark[:, 1] - upper_bound
        landmark = landmark / (size * 2)

        return img, bbox, landmark

    def gen_heatmap(self, landmark):
        landmark[:, 0] = landmark[:, 0] * self.map_size
        landmark[:, 1] = landmark[:, 1] * self.map_size
        landmark = torch.Tensor(landmark)

        grid_x = torch.arange(self.map_size).repeat(self.map_size, 1)
        grid_y = torch.arange(self.map_size).repeat(self.map_size, 1).t()
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0) # size:(1, self.map_size, self.map_size, 2)

        landmarks = landmark.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((grid - landmarks)**2, dim=-1)
        heatmap = torch.exp(-exponent / 2.0 / self.sigma / self.sigma)
        background = 1.0 - torch.amax(heatmap, dim=0).unsqueeze(0)
        heatmap = torch.cat([background, heatmap])
        return heatmap

    def generate_lpm(self, label):
        """
        get ridge heat map base on the distance to a line segment
        Formula basis: https://www.cnblogs.com/flyinggod/p/9359534.html
        """
        limb_maps = np.zeros((20, self.map_size, self.map_size), dtype=np.float32)
        x, y = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
        count = 0
        for part in self.parts:              # 20 parts
            x1, y1 = label[part[0]]        # vector start
            x2, y2 = label[part[1]]        # vector end

            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # unit vector
            v1 = (x2 - x1)/(length + 1e-8)  # in case the length is zero, so add 1e-8
            v2 = (y2 - y1)/(length + 1e-8)

            dist_along_part = v1 * (x - x1) + v2 * (y - y1)
            dist_per_part = np.abs(v2 * (x - x1) + (-v1) * (y - y1))

            mask1 = dist_along_part >= 0
            mask2 = dist_along_part <= length
            mask3 = dist_per_part <= 10
            mask = mask1 & mask2 & mask3

            limb_maps[count, :, :] = mask.astype('float32')
            count += 1
        return limb_maps
        """count = 0
        for part in self.parts:              # 20 parts
            x1, y1 = label[part[0]]        # vector start
            x2, y2 = label[part[1]]        # vector end

            cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
            length2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
            r = (cross + 1e-8) / (length2 + 1e-8)
            px = x1 + (x2 - x1) * r
            py = y1 + (y2 - y1) * r

            mask1 = cross <= 0              # 46 * 46
            mask2 = cross >= length2
            mask3 = 1 - mask1 | mask2

            D2 = np.zeros((self.map_size, self.map_size))
            D2 += mask1.astype('float32') * ((x - x1) * (x - x1) + (y - y1) * (y - y1))
            D2 += mask2.astype('float32') * ((x - x2) * (x - x2) + (y - y2) * (y - y2))
            D2 += mask3.astype('float32') * ((x - px) * (x - px) + (py - y) * (py - y))

            limb_maps[count] = np.exp(-D2 / 2.0 / self.sigma / self.sigma)  # numpy 2d
            count += 1
        return limb_maps"""

    def limb_group(self, limb_maps, groupc, modelgroup):
        # ************ Grouping Limb Maps ************
        ridegemap_group = np.zeros((groupc, self.map_size, self.map_size), dtype=np.float32)
        count = 0
        for group in modelgroup:    # group6 or group1
            for g in group:
                group_tmp = ridegemap_group[count, :, :]
                limb_tmp = limb_maps[g, :, :]
                max_id = group_tmp < limb_tmp    #
                group_tmp[max_id] = limb_tmp[max_id]
                ridegemap_group[count, :, :] = group_tmp
            count += 1
        return ridegemap_group

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


def load_data(data_dir, img_size, sigma, batch_size, action):
    if action == "train":
        train_set = HandDataset(data_dir, img_size, sigma)
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_set, valid_set, train_dataloader, val_dataloader

    elif action == "test":
        test_set = HandDataset(data_dir, img_size, sigma)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
        return test_set, test_dataloader

    else:
        raise NotImplementedError("please specify the correct action : [train, test]")


if __name__ == "__main__":
    data_dir = "hagrid/hagrid_train"
    batch_size = 1

    train_set, valid_set, train_dataloader, val_dataloader = load_data(data_dir, 348, 3, batch_size, "train")
    print("The number of data in train set: ", train_set.__len__())
    print("The number of data in valid set: ", valid_set.__len__())
    for i, (images, landmarks, lsh_maps, labels) in enumerate(train_dataloader):
        skeletons = torch.cat([landmarks[0], lsh_maps[0]])
        skeletons = np.array(skeletons).transpose(1, 2, 0)
        images = images * 0.5 + 0.5
        img = images[0]
        img = np.array(img).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for i in range(29):
            skeleton = skeletons[:, :, i]
            
            skeleton = cv2.normalize(skeleton, skeleton, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            skeleton = cv2.applyColorMap(skeleton, cv2.COLORMAP_JET)
            
            display = img * 0.8 + skeleton * 0.2
            cv2.imshow("img", display)
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("quit display")
                exit(1)
        
        print(labels)

