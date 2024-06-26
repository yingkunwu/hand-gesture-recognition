import cv2
import yaml
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from libs.load import HandDataModule
from libs.draw import draw_bones, draw_joints


image_size = [192, 192]
batch_size = 32
sigma = 2
num_workers = 8


def display_data(data_path):
    configs = None
    with open(data_path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dm = HandDataModule(
        configs,
        image_size,
        batch_size,
        sigma,
        num_workers,
    )
    dm.setup()
    train_loader = dm.train_dataloader()

    for _, (images, labels, heatmaps, weight, meta) in \
            enumerate(tqdm(train_loader)):
        images[:, 0] = images[:, 0] * 0.229 + 0.485
        images[:, 1] = images[:, 1] * 0.224 + 0.456
        images[:, 2] = images[:, 2] * 0.225 + 0.406
        images = images * 255.0

        print(labels)
        landmarks = meta["joints"]

        # landmarks = landmarks * configs['img_size']
        heatmaps = F.interpolate(
            heatmaps, size=image_size, mode='bilinear', align_corners=True)

        for j in range(batch_size):
            img = images[j].numpy().transpose(1, 2, 0).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            print(weight[j])

            landmark = landmarks[j].numpy().astype(np.int32)

            img = draw_bones(img.copy(), landmark)
            img = draw_joints(img.copy(), landmark)

            heatmap = heatmaps[j].numpy().transpose(1, 2, 0)

            for i in range(configs['num_joints']):
                joint = heatmap[:, :, i]

                joint = cv2.normalize(joint, joint,
                                      alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
                joint = cv2.applyColorMap(joint, cv2.COLORMAP_JET)

                display = img * 0.8 + joint * 0.2
                cv2.imshow("img", display.astype(np.uint8))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("quit display")
                    exit(1)


if __name__ == "__main__":
    display_data("configs/hagrid.yaml")
