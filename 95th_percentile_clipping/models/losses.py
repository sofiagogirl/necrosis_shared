from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


# ======================= CRPS ==================================

def _crps_tf(y_true, y_pred, factor=0.05):
    """
    Core of (pseudo) CRPS loss.

    Args:
        y_true: two-dimensional array.
        y_pred: two-dimensional array.
        factor: importance of std term.
    """
    mae = K.mean(tf.abs(y_pred - y_true))
    dist = tf.math.reduce_std(y_pred)
    return mae - factor * dist


def crps2d_tf(y_true, y_pred, factor=0.05):
    """
    Experimental approximated continuous ranked probability score (CRPS) loss:
        CRPS = mean_abs_err - factor * std

    Note: factor > 0.1 may yield negative loss values.

    Args:
        y_true: training target with shape (batch_num, x, y, 1).
        y_pred: forward pass with shape (batch_num, x, y, 1).
        factor: relative importance of standard deviation term.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    batch_num = y_pred.shape.as_list()[0]
    crps_out = sum(_crps_tf(y_true[i, ...], y_pred[i, ...], factor=factor)
                   for i in range(batch_num))
    return crps_out / batch_num


def _crps_np(y_true, y_pred, factor=0.05):
    """NumPy version of _crps_tf."""
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)
    return mae - factor * dist


def crps2d_np(y_true, y_pred, factor=0.05):
    """
    Experimental NumPy version of crps2d_tf.
    See crps2d_tf for full documentation.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    batch_num = len(y_pred)
    crps_out = sum(_crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)
                   for i in range(batch_num))
    return crps_out / batch_num


# ======================= dice ==================================

def dice_coef(y_true, y_pred, const=K.epsilon()):
    """
    Sørensen–Dice coefficient for 2-d samples.

    Args:
        y_true, y_pred: targets and predictions.
        const: smoothing constant for numerical stability.
    """
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

    return (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)


def dice(y_true, y_pred, const=K.epsilon()):
    """
    Sørensen–Dice loss.

    Args:
        const: smoothing constant for numerical stability.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    return 1 - dice_coef(y_true, y_pred, const=const)


# ======================= tversky ================================

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    """
    Weighted Sørensen–Dice coefficient.

    Args:
        y_true, y_pred: targets and predictions.
        alpha: weight for false negatives vs false positives.
        const: smoothing constant for numerical stability.
    """
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

    return (true_pos + const) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + const)


def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    """
    Tversky loss.

    Reference:
        Hashemi et al., 2018. Tversky as a loss function for highly unbalanced image segmentation.
        arXiv:1803.11078.

    Args:
        alpha: tunable parameter in [0, 1] for imbalanced classification.
        const: smoothing constant for numerical stability.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    return 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)


def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    """
    Focal Tversky Loss (FTL).

    Reference:
        Abraham & Khan, 2019. A novel focal tversky loss function with improved attention u-net.
        IEEE ISBI 2019.

    Args:
        alpha: tunable parameter in [0, 1] for imbalanced classification.
        gamma: tunable parameter in [1, 3].
        const: smoothing constant for numerical stability.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    return tf.math.pow((1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1 / gamma)


# ======================= MS-SSIM ================================

def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.

    Reference:
        Wang et al., 2003. Multiscale structural similarity for image quality assessment.
        Asilomar Conference on Signals, Systems & Computers.

    Args:
        kwargs: keyword arguments for tf.image.ssim_multiscale.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    return 1 - tf.image.ssim_multiscale(y_true, y_pred, **kwargs)


# ======================= IoU ====================================

def iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    IoU and generalized IoU coefficients for bounding boxes.
    Bounding box elements should be organized as [y_min, x_min, y_max, x_max].

    Reference:
        Rezatofighi et al., 2019. Generalized intersection over union.
        IEEE/CVF CVPR.

    Args:
        y_true: target bounding box.
        y_pred: predicted bounding box.
        mode: 'iou' for IoU (Jaccard index), 'giou' for generalized IoU.
        dtype: data type of input tensors.
    """
    zero = tf.convert_to_tensor(0.0, dtype)

    ymin_true, xmin_true, ymax_true, xmax_true = tf.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = tf.unstack(y_pred, 4, axis=-1)

    w_true = tf.maximum(zero, xmax_true - xmin_true)
    h_true = tf.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true

    w_pred = tf.maximum(zero, xmax_pred - xmin_pred)
    h_pred = tf.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred

    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)

    w_intersect = tf.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = tf.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect

    area_union = area_true + area_pred - area_intersect
    iou = tf.math.divide_no_nan(area_intersect, area_union)

    if mode == 'iou':
        return iou

    enclose_ymin = tf.minimum(ymin_true, ymin_pred)
    enclose_xmin = tf.minimum(xmin_true, xmin_pred)
    enclose_ymax = tf.maximum(ymax_true, ymax_pred)
    enclose_xmax = tf.maximum(xmax_true, xmax_pred)

    w_enclose = tf.maximum(zero, enclose_xmax - enclose_xmin)
    h_enclose = tf.maximum(zero, enclose_ymax - enclose_ymin)
    area_enclose = w_enclose * h_enclose

    return iou - tf.math.divide_no_nan((area_enclose - area_union), area_enclose)


def iou_box(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    IoU and generalized IoU losses for bounding boxes.
    Bounding box elements should be organized as [y_min, x_min, y_max, x_max].

    Reference:
        Rezatofighi et al., 2019. Generalized intersection over union.
        IEEE/CVF CVPR.

    Args:
        y_true: target bounding box.
        y_pred: predicted bounding box.
        mode: 'iou' for IoU (Jaccard index), 'giou' for generalized IoU.
        dtype: data type of input tensors.
    """
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype)
    y_true = tf.cast(y_true, dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)


def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    IoU loss for segmentation maps.

    Reference:
        Rahman & Wang, 2016. Optimizing intersection-over-union in deep neural networks.
        International Symposium on Visual Computing. Springer, Cham.

    Args:
        y_true: segmentation targets.
        y_pred: segmentation predictions.
        dtype: data type of input tensors.
    """
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect

    return 1 - tf.math.divide_no_nan(area_intersect, area_union)


# ======================= triplet ================================

def triplet_1d(y_true, y_pred, N, margin=5.0):
    """
    Experimental semi-hard triplet loss with 1-d anchor/positive/negative vectors.

    Args:
        y_true: dummy input (unused, required by Keras loss format).
        y_pred: concatenated anchor, positive, negative embeddings,
                shape=(batch_num, 3*embed_size).
        N: size (dimensions) of embedded vectors.
        margin: positive number that prevents negative loss.
    """
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos    = y_pred[:, N:2 * N]
    Embd_neg    = y_pred[:, 2 * N:]

    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)

    return tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))