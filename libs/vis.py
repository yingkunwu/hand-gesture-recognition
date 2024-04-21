import numpy as np
import torchvision
import cv2
import math
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F

from .utils import get_max_preds


def save_batch_image_with_joints(batch_image, batch_label, batch_joints,
                                 batch_joints_vis, file_name,
                                 nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            label = batch_label[k].item()
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                cornerX = x * width + padding
                cornerY = y * height + padding
                joint[0] = cornerX + joint[0]
                joint[1] = cornerY + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])),
                               2, [255, 0, 0], 2)
                cv2.putText(ndarr, str(label), (int(cornerX), int(cornerY+25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_attention_map(batch_image, attn_map, file_name, normalize=True):

    # for now we assume the input image is square
    image_size = batch_image.size(2)
    feat_size = image_size // 16

    # visualize attention from the class token
    attn_map = attn_map.mean(dim=1)
    attn_map = rearrange(attn_map[:, 0, 1:], 'b (h w) -> b h w',
                         h=feat_size, w=feat_size)

    if normalize:
        batch_image = batch_image.clone()
        minval = float(batch_image.min())
        maxval = float(batch_image.max())

        batch_image.add_(-minval).div_(maxval - minval + 1e-5)

    nrow = 8
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))

    fig, axs = plt.subplots(ymaps, xmaps, figsize=(30, 15))
    fig.subplots_adjust(
        bottom=0.07, right=0.97, top=0.98, left=0.03,
        wspace=0.00008, hspace=0.02,
    )

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            image = batch_image[k].mul(255).clamp(0, 255).byte()
            image = image.permute(1, 2, 0).cpu().numpy()
            resized_image = cv2.resize(
                image.copy(), (image.shape[1]//4, image.shape[0]//4))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

            axs[y][x].imshow(resized_image)

            image_vis = resized_image.copy()

            attn_map_at_class = F.interpolate(
                attn_map[None, None, k],
                scale_factor=4,
                mode="bilinear")
            attn_map_at_class = \
                attn_map_at_class.squeeze().detach().cpu().numpy()

            # normalize the attention map
            attn_map_at_class = (
                (attn_map_at_class - np.min(attn_map_at_class))
                / (np.max(attn_map_at_class) - np.min(attn_map_at_class))
            )

            axs[y][x].imshow(image_vis)
            im = axs[y][x].imshow(
                attn_map_at_class, cmap="nipy_spectral", alpha=0.5)

            k += 1

    cax = plt.axes([0.975, 0.08, 0.005, 0.90])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    plt.savefig(file_name)
    plt.close()


def save_debug_images(input, prefix, pred_label, label,
                      pred_joints, heatmap, meta, target, attnmap):
    save_batch_image_with_joints(
        input, pred_label, meta['joints'], meta['joints_vis'],
        '{}_gt.jpg'.format(prefix)
    )
    save_batch_image_with_joints(
        input, label, pred_joints.copy(), meta['joints_vis'],
        '{}_pred.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, heatmap, '{}_hm_pred.jpg'.format(prefix)
    )

    if "val" in prefix and attnmap is not None:
        save_batch_attention_map(input, attnmap, '{}_attn.jpg'.format(prefix))
