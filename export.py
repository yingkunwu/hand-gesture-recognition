import os
import onnx
import yaml
import time
import torch
import argparse
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from lightning.pytorch import LightningModule
from sklearn.metrics import f1_score

from model.multitasknet import MultiTaskNet
from libs.load import HandDataModule


class MultiTaskModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        num_joints = kwargs.get('num_joints')
        num_classes = kwargs.get('num_classes')
        image_size = kwargs.get('image_size')
        weight_path = kwargs.get('weight_path')

        self.model = MultiTaskNet(
            num_joints, num_classes, feature_size=image_size[0] // 16)
        self.model = self.model.to(self.device)

        if not os.path.exists(weight_path):
            assert False, "Model is not exist in {}".format(weight_path)

        # load pytorch lightning model
        checkpoint = torch.load(
            weight_path, map_location=self.device)["state_dict"]

        state_dict = OrderedDict()
        for key in checkpoint.keys():
            state_dict[key.replace("model.", "")] = checkpoint[key]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def forward(self, x):
        pred_label, heatmap, _ = self.model(x)
        return pred_label, heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str,
                        default='', help='path to the data config',
                        required=True)
    parser.add_argument('--image_size', nargs='+', type=int,
                        default=[192, 192], help='image size')
    parser.add_argument('--weight_path', type=str,
                        default='', help='path to the model weight')

    args = parser.parse_args()
    print(args)

    with open(args.data_config, "r") as stream:
        try:
            data_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            assert False, exc

    module = MultiTaskModule(
        image_size=args.image_size,
        weight_path=args.weight_path,
        **data_cfg)

    savepath = args.weight_path.replace('.ckpt', '.onnx')
    input_sample = torch.randn((1, 3, 192, 192))
    module.to_onnx(savepath, input_sample, export_params=True)

    # Checks
    onnx_model = onnx.load(savepath)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print("Model is successfully converted to ONNX format at {}"
          .format(savepath))

    # Test the model using training data
    print("Testing the model using testing data...")
    dm = HandDataModule(
        data_cfg=data_cfg,
        image_size=args.image_size,
        batch_size=1,
        sigma=2,
        num_workers=4,
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    session = ort.InferenceSession(
        savepath,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    y_pred, y_true = [], []
    total_time = 0
    for _, (images, labels, _, _, _) in enumerate(tqdm(test_loader)):
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]: images.numpy()}

        start_time = time.time()
        label_pred, heatmap_pred = session.run(None, inp)
        end_time = time.time()

        y_pred.append(np.argmax(label_pred[0]))
        y_true.append(labels[0].numpy())

        total_time += (end_time - start_time)

    cls_f1score = f1_score(y_true, y_pred, average='macro')
    print("Test F1 Score: {:.4f}".format(cls_f1score))

    average_time = total_time / len(test_loader)
    print("Average time taken to process one image: {:.4f} seconds"
          .format(average_time))
