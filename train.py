import os
import random
import yaml
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from libs.load import load_data
from libs.loss import MultiTasksLoss
from libs.utils import get_max_preds
from libs.metrics import PCK, calc_class_accuracy
from model.poseresnet import PoseResNet


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Train:
    def __init__(self, configs):
        self.make_paths()
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PoseResNet(nof_joints=self.configs['num_joints'], nof_classes=self.configs['num_classes'])
        self.model = self.model.to(self.device)

    def make_paths(self):
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        if not os.path.exists("logs/"):
            os.mkdir("logs/")

    def train(self):
        init()
        print("Using device:", self.device)

        train_set, valid_set, train_dataloader, val_dataloader = load_data(
            self.configs['data_path'], 
            self.configs['classes_dict'],
            self.configs['batch_size'], 
            self.configs['img_size'], 
            self.configs['num_joints'], 
            self.configs['sigma'], 
            self.configs['preprocess'], 
            "train"
        )
        print("The number of data in train set: ", train_set.__len__())
        print("The number of data in valid set: ", valid_set.__len__())

        criterion = MultiTasksLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])

        log_dict = {
            "train_loss_list": [],
            "train_class_acc_list": [],
            "train_PCK_acc_list": [],
            "val_loss_list": [],
            "val_class_acc_list": [],
            "val_PCK_acc_list": []
        }

        for epoch in range(self.configs['epochs']):
            train_loss, val_loss = 0, 0
            train_class_acc, val_class_acc = 0, 0
            train_PCK_acc, val_PCK_acc = 0, 0

            # --------------------------
            # Training Stage
            # --------------------------
            self.model.train()
            for i, (images, heatmaps, labels, landmarks) in enumerate(tqdm(train_dataloader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                heatmap_pred, label_pred = self.model(images)

                loss = criterion(heatmap_pred, label_pred, heatmaps, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if label_pred is not None:
                    prediction = torch.argmax(label_pred, dim=1)
                    train_class_acc += calc_class_accuracy(prediction.detach().cpu().numpy(), labels.detach().cpu().numpy())

                if heatmap_pred is not None:
                    landmarks_pred, _ = get_max_preds(heatmap_pred.detach().cpu().numpy())

                    landmarks = (landmarks * configs['img_size']).numpy()
                    landmarks_pred = landmarks_pred * configs['img_size']

                    train_PCK_acc += PCK(landmarks_pred, landmarks, 
                                    self.configs['img_size'], self.configs['img_size'], self.configs['num_joints'])

            # --------------------------
            # Validation Stage
            # --------------------------
            self.model.eval()
            with torch.no_grad():
                for i, (images, heatmaps, labels, landmarks) in enumerate(tqdm(val_dataloader)):
                    images = images.to(self.device)
                    heatmaps = heatmaps.to(self.device)
                    labels = labels.to(self.device)

                    heatmap_pred, label_pred = self.model(images)

                    loss = criterion(heatmap_pred, label_pred, heatmaps, labels)

                    val_loss += loss.item()

                    if label_pred is not None:
                        prediction = torch.argmax(label_pred, dim=1)
                        val_class_acc += calc_class_accuracy(prediction.detach().cpu().numpy(), labels.detach().cpu().numpy())

                    if heatmap_pred is not None:
                        landmarks_pred, _ = get_max_preds(heatmap_pred.detach().cpu().numpy())

                        landmarks = (landmarks * configs['img_size']).numpy()
                        landmarks_pred = landmarks_pred * configs['img_size']

                        val_PCK_acc += PCK(landmarks_pred, landmarks, 
                                        self.configs['img_size'], self.configs['img_size'], self.configs['num_joints'])

            # --------------------------
            # Logging Stage
            # --------------------------
            print("Epoch: ", epoch + 1)
            print("train_loss: {}, train_class_acc: {}, train_PCK_acc: {}"
                    .format(train_loss / train_dataloader.__len__(), 
                            train_class_acc / train_dataloader.__len__(),
                            train_PCK_acc / train_dataloader.__len__(),
                    )
            )
            print("val_loss: {}, val_class_acc: {}, val_PCK_acc: {}"
                    .format(val_loss / val_dataloader.__len__(), 
                            val_class_acc / val_dataloader.__len__(),
                            val_PCK_acc / val_dataloader.__len__(),
                    )
            )
            torch.save(self.model.state_dict(), os.path.join("weights", self.configs['model_name'] + ".pth"))

            log_dict["train_loss_list"].append(train_loss / train_dataloader.__len__()) 
            log_dict["train_class_acc_list"].append(train_class_acc / train_dataloader.__len__())
            log_dict["train_PCK_acc_list"].append(train_PCK_acc / train_dataloader.__len__())

            log_dict["val_loss_list"].append(val_loss / val_dataloader.__len__())
            log_dict["val_class_acc_list"].append(val_class_acc / val_dataloader.__len__())
            log_dict["val_PCK_acc_list"].append(val_PCK_acc / val_dataloader.__len__())

        # --------------------------
        # Save Logs into .txt files
        # --------------------------
        logs_path = os.path.join("logs", self.configs['model_name'])
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        with open(os.path.join(logs_path, "history.json"), "w") as outfile:
            json.dump(log_dict, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    configs = None
    with open("configs/train.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(configs)
    t = Train(configs)
    t.train()
