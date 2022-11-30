import os
import random
import yaml
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from libs.load import load_data
from libs.loss import MultiTasksLoss
from libs.metrics import PCK, get_max_preds, calc_class_accuracy
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

        train_loss_list, val_loss_list = [], []
        train_class_acc_list, val_class_acc_list = [], []
        train_PCK_acc_list, val_PCK_acc_list = [], []

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
                    train_class_acc += calc_class_accuracy(label_pred.detach(), labels)

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
                        val_class_acc += calc_class_accuracy(label_pred.detach(), labels)

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

            train_loss_list.append(train_loss / train_dataloader.__len__()) 
            train_class_acc_list.append(train_class_acc / train_dataloader.__len__())
            train_PCK_acc_list.append(train_PCK_acc / train_dataloader.__len__())

            val_loss_list.append(val_loss / val_dataloader.__len__())
            val_class_acc_list.append(val_class_acc / val_dataloader.__len__())
            val_PCK_acc_list.append(val_PCK_acc / val_dataloader.__len__())

        # --------------------------
        # Save Logs into .txt files
        # --------------------------
        log_name_dict = {
            # don't know why I can't use np.array(train_loss_list) directly
            "train_loss": torch.tensor(train_loss_list).cpu().numpy(), 
            "train_class_acc": torch.tensor(train_class_acc_list).cpu().numpy(), 
            "train_PCK_acc": torch.tensor(train_PCK_acc_list).cpu().numpy(), 
            "val_loss": torch.tensor(val_loss_list).cpu().numpy(), 
            "val_class_acc": torch.tensor(val_class_acc_list).cpu().numpy(), 
            "val_PCK_acc": torch.tensor(val_PCK_acc_list).cpu().numpy()
        }

        logs_path = os.path.join("logs", self.configs['model_name'])
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        for keys, values in log_name_dict.items():
            np.savetxt(os.path.join(logs_path, "{}_{}.txt").format(self.configs['model_name'], keys), values, delimiter=',')


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
