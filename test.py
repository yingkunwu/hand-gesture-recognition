import torch
import numpy as np
import cv2
import os
import time
import torch.nn.functional as F
from tqdm import tqdm

from libs.load import load_data
from libs.options import TestOptions
from model.CPM import ConvolutionalPoseMachine


class Test:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvolutionalPoseMachine(self.args.num_masks, self.args.num_heatmaps)

    def load_model(self):
        weight_path = os.path.join("weights", self.args.model_name)
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            assert False, "Model is not exist in {}".format(weight_path)

        weight = torch.load(weight_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(weight)

    def detect(self):
        print("Using device:", self.device)

        test_set, test_dataloader = load_data(self.args.data_folder, self.args.img_size, self.args.sigma, 
                                                self.args.batch_size, "test")
        print("The number of data in test set: ", test_set.__len__())

        self.load_model()
        self.model.eval()

        start_time = time.time()

        # --------------------------
        # Testing Stage
        # --------------------------
        with torch.no_grad():
            for i, (images, keypoints, limbmasks, labels) in enumerate(tqdm(test_dataloader)):    
                images = images.to(self.device)
                g6_pred, g1_pred, kp_pred = self.model(images)

                images = images * 0.5 + 0.5

                g6_pred = g6_pred[:, -1:, ...]
                g1_pred = g1_pred[:, -6:, ...]
                kp_pred = kp_pred[:, -22:, ...]
                g6_pred = F.interpolate(g6_pred, size=(self.args.img_size, self.args.img_size), 
                                            mode='bilinear', align_corners=True)
                g1_pred = F.interpolate(g1_pred, size=(self.args.img_size, self.args.img_size), 
                                            mode='bilinear', align_corners=True)
                kp_pred = F.interpolate(kp_pred, size=(self.args.img_size, self.args.img_size), 
                                            mode='bilinear', align_corners=True)

                pred_maps = torch.cat([g6_pred, g1_pred, kp_pred], dim=1)

                targ_maps = torch.cat([limbmasks, keypoints], dim=1)

                for i in range(len(pred_maps)):
                    img = images[i]
                    img = img.cpu().numpy().transpose(1, 2, 0)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    preds = pred_maps[i]
                    preds = preds.cpu().numpy().transpose(1, 2, 0)
                    targs = targ_maps[i]
                    targs = targs.cpu().numpy().transpose(1, 2, 0)

                    for i in range(22):
                        pred = preds[:, :, i]
                        pred = cv2.normalize(pred, pred, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                    dtype=cv2.CV_8U)
                        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

                        targ = targs[:, :, i]
                        targ = cv2.normalize(targ, targ, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                    dtype=cv2.CV_8U)
                        targ = cv2.applyColorMap(targ, cv2.COLORMAP_JET)
                    
                        display1 = img * 0.8 + pred * 0.2
                        display2 = img * 0.8 + targ * 0.2
                        display = np.concatenate((display1, display2), axis=1)
                        cv2.imshow("img", display)
                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            print("quit display")
                            exit(1)


        end_time = time.time()

        print("Testing cost {} sec(s)".format(end_time - start_time))


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print(args)

    t = Test(args)
    t.detect()
