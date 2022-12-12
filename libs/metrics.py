import torch
import numpy as np


def calc_class_accuracy(preds, target):
    acc = np.mean(np.equal(preds, target).astype(np.float32))
    return acc


def calc_dists(preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

def dist_acc(dists, thr=0.5):
        ''' Return percentage below threshold while ignoring values with a -1 '''
        dist_cal = np.not_equal(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
        else:
            return -1

def PCK(pred, target, h, w, num_joints):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    # h and w is the size of the bbox; the distance is defined as a fraction of the bounding box size
    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = np.zeros((num_joints))
    avg_acc = 0
    cnt = 0

    for i in range(num_joints):
        acc[i] = dist_acc(dists[i])
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return avg_acc
