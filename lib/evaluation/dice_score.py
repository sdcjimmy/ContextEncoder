import torch
from lib.loss_functions.dice_loss import get_tp_fp_fn
from lib.utilities.nd_softmax import softmax_helper
from lib.utilities.tensor_utilities import sum_tensor
from torch import nn


def dice(x, y, batch_dice = True, loss_mask = None, do_bg = True, smooth = 1., square = False, return_mean = False):
    shp_x = x.shape

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)
    dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]
    if return_mean:
        return dc.mean()
    else:
        return dc

