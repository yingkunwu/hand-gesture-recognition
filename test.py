import cv2
import os
import time
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from libs.load import load_data
from libs.draw import draw_bones, draw_joints
from libs.metrics import PCK, get_max_preds, calc_class_accuracy
from model.poseresnet import PoseResNet

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Test:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PoseResNet(nof_joints=self.configs['num_joints'], nof_classes=self.configs['num_classes'])
        self.model = self.model.to(self.device)

    def load_model(self):
        weight_path = os.path.join("weights", self.configs['model_name'] + ".pth")
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            assert False, "Model is not exist in {}".format(weight_path)

    def test(self):
        print("Using device:", self.device)

        test_set, test_dataloader= load_data(
            self.configs['data_path'], 
            self.configs['classes_dict'],
            self.configs['batch_size'], 
            self.configs['img_size'], 
            self.configs['num_joints'], 
            self.configs['sigma'],
            self.configs['preprocess'], 
            "test"
        )
        print("The number of data in test set: ", test_set.__len__())

        self.load_model()
        self.model.eval()

        class_acc, PCK_acc = 0.0, 0.0

        label_dict = test_set.classes
        label_dict = {v: k for k, v in label_dict.items()}

        # --------------------------
        # Testing Stage
        # --------------------------
        y_true, y_pred = [], []
        start_time = time.time()
        with torch.no_grad():
            for i, (images, heatmaps, labels, landmarks) in enumerate(tqdm(test_dataloader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                labels = labels.to(self.device)

                heatmap_pred, label_pred = self.model(images)

                landmarks_pred, maxvals = get_max_preds(heatmap_pred.cpu().numpy())

                landmarks = (landmarks * configs['img_size']).numpy()
                landmarks_pred = landmarks_pred * configs['img_size']

                class_acc += calc_class_accuracy(label_pred, labels)
                PCK_acc += PCK(landmarks_pred, landmarks, 
                                self.configs['img_size'], self.configs['img_size'], self.configs['num_joints'])

                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(torch.argmax(label_pred, dim=1).detach().cpu().numpy())
                
                if self.configs['display_results']:
                    images[:, 0] = images[:, 0] * 0.229 + 0.485
                    images[:, 1] = images[:, 1] * 0.224 + 0.456
                    images[:, 2] = images[:, 2] * 0.225 + 0.406
                    images = images * 255.0

                    landmarks = landmarks.astype(np.int32)
                    landmarks_pred = landmarks_pred.astype(np.int32)

                    pred_maps = F.interpolate(heatmap_pred, size=(self.configs['img_size'], self.configs['img_size']), 
                                                mode='bilinear', align_corners=True)
                    targ_maps = F.interpolate(heatmaps, size=(self.configs['img_size'], self.configs['img_size']), 
                                                mode='bilinear', align_corners=True)

                    for i in range(self.configs['batch_size']):
                        img = images[i]
                        img = img.cpu().numpy().transpose(1, 2, 0)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        pred_img = img.copy()
                        targ_img = img.copy()

                        pred_heatmap = pred_maps[i].cpu().numpy().transpose(1, 2, 0)
                        targ_heatmap = targ_maps[i].cpu().numpy().transpose(1, 2, 0)

                        pred_landmark = landmarks_pred[i]
                        targ_landmark = landmarks[i]

                        pred_img = draw_bones(pred_img, pred_landmark)
                        targ_img = draw_bones(targ_img, targ_landmark)

                        pred_img = draw_joints(pred_img, pred_landmark)
                        targ_img = draw_joints(targ_img, targ_landmark)

                        pred_label = torch.argmax(label_pred[i]).cpu().item()
                        targ_label = labels[i].cpu().item()

                        for j in range(self.configs['num_joints']):
                            print(maxvals[i][j])

                            pred = pred_heatmap[:, :, j]
                            pred = cv2.normalize(pred, pred, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                        dtype=cv2.CV_8U)
                            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

                            targ = targ_heatmap[:, :, j]
                            targ = cv2.normalize(targ, targ, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                        dtype=cv2.CV_8U)
                            targ = cv2.applyColorMap(targ, cv2.COLORMAP_JET)
                        
                            display1 = pred_img * 0.8 + pred * 0.2
                            display2 = targ_img * 0.8 + targ * 0.2

                            display1 = cv2.putText(display1, "Prediction: {}".format(label_dict[pred_label]), 
                                                (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            display2 = cv2.putText(display2, "Ground Truth: {}".format(label_dict[targ_label]), 
                                                (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            display = np.concatenate((display1, display2), axis=1).astype(np.uint8)
                            cv2.imshow("img", display)
                            key = cv2.waitKey(0)
                            if key == ord('q'):
                                print("quit display")
                                exit(1)

        end_time = time.time()

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.configs['classes_dict'].keys())
        disp.plot()
        ax, fig = plt.gca(), plt.gcf()
        plt.setp(ax.get_xticklabels(), rotation=-90, ha="center", rotation_mode="default")
        fig.tight_layout()
        plt.savefig(os.path.join("results", "{}.png".format(self.configs['model_name'])))
        #plt.show()

        print("Accuracy of classification: {}, Accuracy of pose estimation: {}, Testing cost {} sec(s) per image"
                .format(class_acc / test_dataloader.__len__(), 
                        PCK_acc / test_dataloader.__len__(), 
                        (end_time - start_time) / test_set.__len__()))


if __name__ == "__main__":
    configs = None
    with open("configs/test.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(configs)
    t = Test(configs)
    t.test()
