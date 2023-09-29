import os
import random
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import f1_score

from libs.load import load_data
from libs.loss import MultiTasksLoss
from libs.utils import get_max_preds
from libs.metrics import PCK, calc_class_accuracy
from model.poseresnet import PoseResNet
from model.resnext import ResNeXt


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Train:
    def __init__(self, opt, configs):
        self.make_paths()
        self.opt = opt
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if opt.model_type == 'resnet':
            self.model = PoseResNet(
                resnet_size=int(opt.model_ver),
                nof_joints=configs['num_joints'],
                nof_classes=configs['num_classes']
            )
        elif opt.model_type == 'resnext':
            self.model = ResNeXt(
                resnext_size=int(opt.model_ver),
                nof_classes=configs['num_classes']
            )
        else:
            raise NotImplementedError
        self.model = self.model.to(self.device)

    def make_paths(self):
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        if not os.path.exists("logs/"):
            os.mkdir("logs/")

    def train(self):
        init()
        print("Using device:", self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Number of model parameters:", total_params)

        train_set, valid_set, train_dataloader, val_dataloader = load_data(
            self.configs['data_path'], 
            self.configs['classes_dict'],
            self.opt.batch_size, 
            self.opt.img_size, 
            self.configs['num_joints'], 
            self.opt.sigma, 
            self.configs['preprocess'], 
            "train"
        )
        print("The number of data in train set: ", train_set.__len__())
        print("The number of data in valid set: ", valid_set.__len__())

        criterion = MultiTasksLoss()

        optimizer = SGD(self.model.parameters(), lr=self.opt.lr, momentum=0.9)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.opt.epochs)

        log_dict = {
            "train_loss_list": [],
            "train_class_acc_list": [],
            "train_f1score_list": [],
            "train_PCK_acc_list": [],
            "val_loss_list": [],
            "val_class_acc_list": [],
            "val_f1score_list": [],
            "val_PCK_acc_list": []
        }

        best_f1score = 0

        for epoch in range(self.opt.epochs):
            train_loss, val_loss = 0, 0
            train_class_acc, val_class_acc = 0, 0
            train_f1score, val_f1score = 0, 0
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
                    train_f1score += f1_score(prediction.detach().cpu().numpy(), labels.detach().cpu().numpy(), average='macro')

                if heatmap_pred is not None:
                    landmarks_pred, _ = get_max_preds(heatmap_pred.detach().cpu().numpy())

                    landmarks = (landmarks * self.opt.img_size).numpy()
                    landmarks_pred = landmarks_pred * self.opt.img_size

                    train_PCK_acc += PCK(landmarks_pred, landmarks, 
                                    self.opt.img_size, self.opt.img_size, self.configs['num_joints'])

            scheduler.step()

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
                        val_f1score += f1_score(prediction.detach().cpu().numpy(), labels.detach().cpu().numpy(), average='macro')

                    if heatmap_pred is not None:
                        landmarks_pred, _ = get_max_preds(heatmap_pred.detach().cpu().numpy())

                        landmarks = (landmarks * self.opt.img_size).numpy()
                        landmarks_pred = landmarks_pred * self.opt.img_size

                        val_PCK_acc += PCK(landmarks_pred, landmarks, 
                                        self.opt.img_size, self.opt.img_size, self.configs['num_joints'])

            # --------------------------
            # Logging Stage
            # --------------------------
            print("Epoch: ", epoch + 1)
            print("train_loss: {}, train_class_acc: {}, train_f1score: {}, train_PCK_acc: {}"
                    .format(train_loss / train_dataloader.__len__(), 
                            train_class_acc / train_dataloader.__len__(),
                            train_f1score / train_dataloader.__len__(),
                            train_PCK_acc / train_dataloader.__len__(),
                    )
            )
            print("val_loss: {}, val_class_acc: {}, val_f1score: {}, val_PCK_acc: {}"
                    .format(val_loss / val_dataloader.__len__(), 
                            val_class_acc / val_dataloader.__len__(),
                            val_f1score / val_dataloader.__len__(),
                            val_PCK_acc / val_dataloader.__len__(),
                    )
            )

            log_dict["train_loss_list"].append(train_loss / train_dataloader.__len__()) 
            log_dict["train_class_acc_list"].append(train_class_acc / train_dataloader.__len__())
            log_dict["train_f1score_list"].append(train_f1score / train_dataloader.__len__())
            log_dict["train_PCK_acc_list"].append(train_PCK_acc / train_dataloader.__len__())

            log_dict["val_loss_list"].append(val_loss / val_dataloader.__len__())
            log_dict["val_class_acc_list"].append(val_class_acc / val_dataloader.__len__())
            log_dict["val_f1score_list"].append(val_f1score / val_dataloader.__len__())
            log_dict["val_PCK_acc_list"].append(val_PCK_acc / val_dataloader.__len__())

            # --------------------------------------
            # Save the Model with the Best F1-Score
            # --------------------------------------
            if val_f1score > best_f1score:
                best_f1score = val_f1score

                torch.save(self.model.state_dict(), os.path.join("weights", self.opt.model_name + ".pth"))
                print("Current best model is saved!")

        # --------------------------
        # Save Logs into .txt files
        # --------------------------
        logs_path = os.path.join("logs", self.opt.model_name)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='poseresnet', help='name of the model to be trained')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'resnext'], help='model type')
    parser.add_argument('--model_ver', type=str, default='50', choices=['50', '101', '152'], help='model version')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='leanring rate')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--sigma', type=int, default=3, help='sigma of the gaussian distribution of the heatmap')
    opt = parser.parse_args()
    print(opt)
    
    t = Train(opt, configs)
    t.train()
