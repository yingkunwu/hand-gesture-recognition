import os
import yaml
import cv2
import glob
import argparse
import torch
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms

from libs.utils import get_max_preds
from libs.draw import draw_bones, draw_joints
from libs.transforms import get_affine_transform


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # add border
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class Detect:
    def __init__(self, **kwargs):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        cls_weight = kwargs.get('cls_weight')
        det_weight = kwargs.get('det_weight')
        self.data_path = kwargs.get('data_path')
        self.save_path = kwargs.get('save_path')
        self.det_img_size = kwargs.get('det_img_size')
        self.cls_img_size = kwargs.get('cls_img_size')

        class_names = kwargs.get('names')
        self.class_names = {v: k for k, v in class_names.items()}

        # load hand gesture classifier
        self.classifier = self.load_onnx_model(cls_weight)

        # load hand detector
        self.detector = self.load_onnx_model(det_weight)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_onnx_model(self, weight_path):
        if not os.path.exists(weight_path):
            assert False, "Model is not exist in {}".format(weight_path)

        session = ort.InferenceSession(
            weight_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        return session

    def process_image_for_detection(self, ori_img):
        img = cv2.cvtColor(ori_img.copy(), cv2.COLOR_BGR2RGB)

        img, ratio, dwdh = letterbox(
            img, new_shape=self.det_img_size, auto=False)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        im = img.astype(np.float32)
        im /= 255
        return im, ratio, dwdh

    def process_image_for_classification(self, img, bbox):
        x1, y1, x2, y2 = bbox
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        origin_size = max(x2 - x1, y2 - y1) * 1.0
        trans = get_affine_transform(c, 1, 0, origin_size, self.cls_img_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.cls_img_size[0]), int(self.cls_img_size[1])),
            flags=cv2.INTER_LINEAR)

        # cv2.imshow("hand", img)
        # cv2.waitKey(0)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        im = img.astype(np.float32)
        im /= 255

        return im

    def inference(self, frame):
        img, ratio, dwdh = self.process_image_for_detection(frame)

        outname = [i.name for i in self.detector.get_outputs()]
        inname = [i.name for i in self.detector.get_inputs()]
        inp = {inname[0]: img}

        outputs = self.detector.run(outname, inp)[0]

        if len(outputs):
            _, x0, y0, x1, y1, _, score = outputs[0]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            box_width = box_height = \
                max(box[2] - box[0], box[3] - box[1]) * 1.0
            corner = [
                (box[0] + box[2] - box_width) / 2,
                (box[3] + box[1] - box_height) / 2]

            if score > 0.2:
                hand = self.process_image_for_classification(frame, box)

                inname = [i.name for i in self.classifier.get_inputs()]
                inp = {inname[0]: hand}
                label_pred, heatmap_pred = self.classifier.run(None, inp)

                h, w = heatmap_pred.shape[-2:]

                pred_label = np.argmax(label_pred[0])
                landmarks_pred, _ = get_max_preds(heatmap_pred)
                landmarks_pred = landmarks_pred.squeeze(0)
                landmarks_pred[:, 0] = \
                    landmarks_pred[:, 0] / w * box_width + corner[0]
                landmarks_pred[:, 1] = \
                    landmarks_pred[:, 1] / h * box_height + corner[1]

                landmarks_pred = landmarks_pred.astype(np.int32)

                frame = draw_bones(frame, landmarks_pred)
                frame = draw_joints(frame, landmarks_pred)
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                frame = cv2.putText(
                    frame,
                    "Prediction: {}".format(self.class_names[pred_label]),
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def detect(self):
        print("Using device:", self.device)

        video_writer = cv2.VideoWriter(
            self.save_path,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            30.0, (640, 360))

        if os.path.isfile(self.data_path):
            # Input data is a video file
            cap = cv2.VideoCapture(self.data_path)
            if not cap.isOpened():
                assert False, "Error opening video file"
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(nb_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.inference(frame)
                video_writer.write(frame)
                cv2.imshow("frame", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
        else:
            # Input data is a directory of images
            image_files = sorted(glob.glob(
                os.path.join(self.data_path, "*.png")))
            for i in range(len(image_files)):
                frame = cv2.imread(image_files[i])
                frame = self.inference(frame)
                video_writer.write(frame)
                cv2.imshow("frame", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str,
                        default='', help='path to the data config',
                        required=True)
    parser.add_argument('--cls_weight', type=str,
                        default='',
                        help='path to the classification model weight')
    parser.add_argument('--det_weight', type=str,
                        default='',
                        help='path to the detection model weight')
    parser.add_argument('--data_path', type=str,
                        default='data/test.mov',
                        help='path to the input data')
    parser.add_argument('--save_path', type=str,
                        default='result.mov',
                        help='path to save the output video')
    parser.add_argument('--det_img_size', nargs='+', type=int,
                        default=[416, 416], help='detection image size')
    parser.add_argument('--cls_img_size', type=int,
                        default=[192, 192], help='classification image size')

    args = parser.parse_args()
    print(args)

    with open(args.data_config, "r") as stream:
        try:
            data_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            assert False, exc

    d = Detect(
        cls_weight=args.cls_weight,
        det_weight=args.det_weight,
        data_path=args.data_path,
        save_path=args.save_path,
        det_img_size=args.det_img_size,
        cls_img_size=args.cls_img_size,
        **data_cfg)
    d.detect()
