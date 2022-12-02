import os
import yaml
import cv2
import torch
import glob
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.ssd_mobilenetv3 import SSDMobilenet
from model.poseresnet import PoseResNet
from libs.metrics import get_max_preds
from libs.draw import draw_bones, draw_joints


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def load_image(ori_img):
    #ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_height, ori_width, _ = ori_img.shape
    img = ori_img.copy()
    img = transforms.ToTensor()(img)
    img, _ = pad_to_square(img, 0)
    img = resize(img, 320)
    img = img.unsqueeze(0)
    return img, ori_img, 320, (ori_height, ori_width)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((x1 - pad_x // 2) / unpad_w) * orig_w
    y1 = ((y1 - pad_y // 2) / unpad_h) * orig_h
    x2 = ((x2 - pad_x // 2) / unpad_w) * orig_w
    y2 = ((y2 - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 0] = min(max(0, x1), orig_w)
    boxes[:, 1] = min(max(0, y1), orig_h)
    boxes[:, 2] = min(max(0, x2), orig_w)
    boxes[:, 3] = min(max(0, y2), orig_h)
    return boxes.numpy().astype(np.int32)


class Detect:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = PoseResNet(nof_joints=self.configs['num_joints'], nof_classes=self.configs['num_classes'])
        self.detector = SSDMobilenet(num_classes=20)
        self.classifier = self.classifier.to(self.device)
        self.detector.to(self.device)
        self.classes_dict = {v: k for k, v in configs["classes_dict"].items()}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        weight_path = os.path.join("weights", self.configs['classifier_name'] + ".pth")
        if not os.path.exists(weight_path):
            assert False, "Model is not exist in {}".format(weight_path)
        
        self.classifier.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.classifier.eval()

        weight_path = os.path.join("weights", self.configs['detector_name'] + ".pth")
        if not os.path.exists(weight_path):
            assert False, "Model is not exist in {}".format(weight_path)
        
        self.detector.load_state_dict(weight_path, map_location=self.device)
        self.detector.eval()

    def detect(self):
        print("Using device:", self.device)
        self.load_model()

        #files = glob.glob(os.path.join(self.configs['file_path'], "*.jpg"))
        cap = cv2.VideoCapture("data/hagrid/test.mov")
        if (cap.isOpened()== False): 
            assert False, "Error opening video stream or file"
        #for file in files:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                img, ori_img, current_dim, original_shape = load_image(frame)

                with torch.no_grad():
                    output = self.detector(img)[0]

                boxes = output['boxes'][:1]
                box = rescale_boxes(boxes, current_dim, original_shape)[0]

                hand = ori_img[box[1]:box[3], box[0]:box[2]]
                hand_height, hand_width, _ = hand.shape
                hand = self.transform(hand)
                hand = resize(hand, 256)
                heatmap_pred, label_pred = self.classifier(hand.unsqueeze(0))
                landmarks_pred, maxvals = get_max_preds(heatmap_pred.detach().cpu().numpy())
                landmarks_pred = landmarks_pred[0]
                landmarks_pred[:, 0] = landmarks_pred[:, 0] * hand_width + box[0]
                landmarks_pred[:, 1] = landmarks_pred[:, 1] * hand_height + box[1]
                landmarks_pred = landmarks_pred.astype(np.int32)

                ori_img = draw_bones(ori_img, landmarks_pred)
                ori_img = draw_joints(ori_img, landmarks_pred)
                ori_img = cv2.rectangle(ori_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                pred_label = torch.argmax(label_pred[0]).cpu().item()
                ori_img = cv2.putText(ori_img, "Prediction: {}".format(self.classes_dict[pred_label]), 
                                                (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow("ori_img", ori_img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    configs = None
    with open("configs/detect.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(configs)
    d = Detect(configs)
    d.detect()
