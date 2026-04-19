"""
Metrics for the neuron project.

If you use this code, please cite:
    Dalca AV, Guttag J, Sabuncu MR. Anatomical Priors in Convolutional Networks
    for Unsupervised Biomedical Segmentation. CVPR 2018.
    https://arxiv.org/abs/1903.03148

Copyright 2020 Adrian V. Dalca
Licensed under the Apache License, Version 2.0.
"""

import sys
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from tensorflow.keras.losses import mean_absolute_error as l1
from tensorflow.keras.losses import mean_squared_error as l2

from . import utils


# ======================= mutual information =====================

class MutualInformation:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes.

    References:
        - Guo CK. Multi-modal image registration with unsupervised deep learning.
          PhD thesis, MIT, 2019.
        - Hoffmann et al. SynthMorph: learning contrast-invariant registration without
          acquired images. IEEE TMI, 2021. https://doi.org/10.1109/TMI.2021.3116879

    Usage:
        mi = MutualInformation()
        mi.volumes(x, y)
        mi.segs(x, y)
        mi.volume_seg(x, y)
        mi.channelwise(x, y)
        mi.maps(x, y)
    """

    def __init__(self, bin_centers=None, nb_bins=None, soft_bin_alpha=None,
                 min_clip=None, max_clip=None):
        """
        Args:
            bin_centers: array/list of bin centers. If provided, nb_bins is inferred.
            nb_bins: number of bins. Defaults to 16 if bin_centers is not specified.
            soft_bin_alpha: alpha in RBF of soft quantization. Defaults to 1 / (2 * sigma^2).
            min_clip: lower clip value. Defaults to -inf.
            max_clip: upper clip value. Defaults to inf.
        """
        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = tf.convert_to_tensor(bin_centers, dtype=tf.float32)
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins if nb_bins is not None else 16
        self.min_clip = min_clip if min_clip is not None else -np.inf
        self.max_clip = max_clip if max_clip is not None else np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / (self.nb_bins - 1)
            else:
                sigma = sigma_ratio * tf.reduce_mean(tf.experimental.numpy.diff(bin_centers))
            self.soft_bin_alpha = 1 / (2 * tf.square(sigma))
            print(self.soft_bin_alpha)

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of single-channel volumes.

        Args:
            x, y: tensors of shape [bs, ..., 1].

        Returns:
            Tensor of size [bs].
        """
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        tf.debugging.assert_equal(K.shape(x)[-1], 1, msg)
        tf.debugging.assert_equal(K.shape(y)[-1], 1, msg)
        return K.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps.

        Args:
            x, y: tensors of shape [bs, ..., nb_labels].

        Returns:
            Tensor of size [bs].
        """
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation map.

        Args:
            x, y: one should be [bs, ..., 1] and the other [bs, ..., nb_labels].

        Returns:
            Tensor of size [bs].
        """
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        tf.debugging.assert_equal(tf.minimum(tensor_channels_x, tensor_channels_y), 1,
                                  'volume_seg_mi requires one single-channel volume.')
        tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1,
                                    'volume_seg_mi requires one multi-channel segmentation.')

        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])
        else:
            y = self._soft_sim_map(y[..., 0])

        return self.maps(x, y)

    def channelwise(self, x, y):
        """
        Mutual information for each channel: MI(x[...,i], y[...,i]).

        Args:
            x, y: tensors of shape [bs, ..., C].

        Returns:
            Tensor of size [bs, C].
        """
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)
            y = tf.reshape(y, new_shape)

        ndims_k = len(x.shape)
        permute = [ndims_k - 1] + list(range(ndims_k - 1))
        cx = tf.transpose(x, permute)                  # [C, bs, V]
        cy = tf.transpose(y, permute)                  # [C, bs, V]

        cxq = self._soft_sim_map(cx)                   # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                   # [C, bs, V, B]

        cout = tf.map_fn(lambda x: self.maps(*x), [cxq, cyq], dtype=tf.float32)  # [C, bs]
        return tf.transpose(cout, [1, 0])              # [bs, C]

    def maps(self, x, y):
        """
        Mutual information from probability/similarity maps at each voxel.

        Args:
            x, y: probability maps of shape [bs, ..., B], where B is the discrete domain size.

        Returns:
            Tensor of size [bs].
        """
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
        tf.debugging.assert_non_negative(x)
        tf.debugging.assert_non_negative(y)

        eps = K.epsilon()

        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)               # [bs, V, B1]
            y = tf.reshape(y, new_shape)               # [bs, V, B2]

        pxy = K.batch_dot(tf.transpose(x, (0, 2, 1)), y)                           # [bs, B1, B2]
        pxy = pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)

        px = K.sum(x, 1, keepdims=True)
        px = px / (K.sum(px, 2, keepdims=True) + eps)                              # [bs, 1, B1]

        py = K.sum(y, 1, keepdims=True)
        py = py / (K.sum(py, 2, keepdims=True) + eps)                              # [bs, 1, B2]

        pxpy = K.batch_dot(K.permute_dimensions(px, (0, 2, 1)), py) + eps          # [bs, B1, B2]

        mi = K.sum(pxy * K.log(pxy / pxpy + eps), axis=[1, 2])                     # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """Soft quantization of intensities, returning log probabilities. Shape: [bs, ..., B]."""
        return ne.utils.soft_quantize(x, alpha=self.soft_bin_alpha, bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins, min_clip=self.min_clip,
                                      max_clip=self.max_clip, return_log=True)

    def _soft_sim_map(self, x):
        """Soft quantization of intensities. Shape: [bs, ..., B]."""
        return ne.utils.soft_quantize(x, alpha=self.soft_bin_alpha, bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins, min_clip=self.min_clip,
                                      max_clip=self.max_clip, return_log=False)

    def _soft_prob_map(self, x, **kwargs):
        """Normalize soft-quantized volume so each voxel sums to 1. Shape: [bs, ..., B]."""
        x_hist = self._soft_sim_map(x, **kwargs)
        x_hist_sum = K.sum(x_hist, -1, keepdims=True) + K.epsilon()
        return x_hist / x_hist_sum


# ======================= dice ==================================

class Dice:
    """
    Dice coefficient of two Tensors. Supports soft/hard Dice and per-label weighting.

    References:
        - Dice. Measures of the amount of ecologic association between species. Ecology. 1945.
        - Dalca AV, Guttag J, Sabuncu MR. Anatomical Priors in Convolutional Networks
          for Unsupervised Biomedical Segmentation. CVPR 2018.
    """

    def __init__(self, dice_type='soft', input_type='prob', nb_labels=None,
                 weights=None, check_input_limits=True, normalize=False):
        """
        Args:
            dice_type: 'soft' or 'hard'. Hard dice has no gradients.
            input_type: 'prob', 'one_hot', or 'max_label'.
            nb_labels: number of labels. Required for hard dice with max_label input.
            weights: weight matrix broadcastable to [batch_size, nb_labels].
            check_input_limits: whether to assert inputs are in [0, 1].
            normalize: whether to renormalize probabilistic Tensors.
        """
        self.dice_type = dice_type
        self.input_type = input_type
        self.nb_labels = nb_labels
        self.weights = weights
        self.normalize = normalize
        self.check_input_limits = check_input_limits

        assert self.input_type in ['prob', 'max_label']

        if self.dice_type == 'hard' and self.input_type == 'max_label':
            assert self.nb_labels is not None, 'If doing hard Dice need nb_labels'

        if self.dice_type == 'soft':
            assert self.input_type in ['prob', 'one_hot'], \
                'if doing soft Dice, must use probabilistic (one_hot) encoding'

    def dice(self, y_true, y_pred):
        """
        Compute Dice between two Tensors.

        Args:
            y_true, y_pred: tensors of shape [batch_size, ..., nb_labels] (prob/one_hot)
                            or [batch_size, ...] (max_label).

        Returns:
            Tensor of size [batch_size, nb_labels].
        """
        if self.input_type in ['prob', 'one_hot']:
            if self.normalize:
                y_true = tf.math.divide_no_nan(y_true, K.sum(y_true, axis=-1, keepdims=True))
                y_pred = tf.math.divide_no_nan(y_pred, K.sum(y_pred, axis=-1, keepdims=True))

            if self.check_input_limits:
                msg = 'value outside range'
                tf.debugging.assert_greater_equal(y_true, 0., msg)
                tf.debugging.assert_greater_equal(y_pred, 0., msg)
                tf.debugging.assert_less_equal(y_true, 1., msg)
                tf.debugging.assert_less_equal(y_pred, 1., msg)

        if self.dice_type == 'hard':
            if self.input_type == 'prob':
                warnings.warn('Using hard Dice with probabilistic inputs — argmax is not '
                              'differentiable. Do not use with backprop.')
                if self.nb_labels is None:
                    self.nb_labels = y_pred.shape.as_list()[-1]
                y_pred = K.argmax(y_pred, axis=-1)
                y_true = K.argmax(y_true, axis=-1)

            y_pred = K.one_hot(y_pred, self.nb_labels)
            y_true = K.one_hot(y_true, self.nb_labels)

        y_true = ne.utils.batch_channel_flatten(y_true)
        y_pred = ne.utils.batch_channel_flatten(y_pred)

        top = 2 * K.sum(y_true * y_pred, 1)
        bottom = K.sum(K.square(y_true), 1) + K.sum(K.square(y_pred), 1)
        return tf.math.divide_no_nan(top, bottom)

    def mean_dice(self, y_true, y_pred):
        """
        Mean Dice across all patches and labels, optionally weighted.

        Args:
            y_true, y_pred: tensors of shape [batch_size, ..., nb_labels] or [batch_size, ...].

        Returns:
            Scalar tensor (tf.float32).
        """
        dice_metric = self.dice(y_true, y_pred)

        if self.weights is not None:
            assert len(self.weights.shape) == 2, \
                'weights should be broadcastable to [batch_size, nb_labels]'
            dice_metric *= self.weights

        mean_dice_metric = K.mean(dice_metric)
        tf.debugging.assert_all_finite(mean_dice_metric, 'metric not finite')
        return mean_dice_metric

    def loss(self, y_true, y_pred):
        """Deprecated. Use ne.losses.*.loss instead."""
        warnings.warn('ne.metrics.*.loss functions are deprecated. '
                      'Please use the ne.losses.*.loss functions.')
        return -self.mean_dice(y_true, y_pred)


class SoftDice(Dice):
    """
    Soft Dice of two Tensors.

    References:
        - Dalca AV et al. Anatomical Priors in Convolutional Networks. CVPR 2018.
        - Milletari et al. V-net: Fully convolutional neural networks for volumetric
          medical image segmentation. 3DV 2016.
    """

    def __init__(self, weights=None, normalize=False):
        """
        Args:
            weights: weight matrix broadcastable to [batch_size, nb_labels].
            normalize: whether to renormalize probabilistic Tensors.
        """
        super().__init__(dice_type='soft', input_type='prob',
                         weights=weights, normalize=normalize)


class HardDice(Dice):
    """
    Hard Dice of two Tensors (no gradients).

    References:
        - Dice. Measures of the amount of ecologic association between species. Ecology. 1945.
        - Dalca AV et al. Anatomical Priors in Convolutional Networks. CVPR 2018.
    """

    def __init__(self, nb_labels, input_type='max_label', weights=None, normalize=False):
        """
        Args:
            nb_labels: number of labels (maximum label + 1).
            input_type: 'max_label', 'prob', or 'one_hot'.
            weights: weight matrix broadcastable to [batch_size, nb_labels].
            normalize: whether to renormalize probabilistic Tensors.
        """
        super().__init__(dice_type='hard', input_type=input_type, nb_labels=nb_labels,
                         weights=weights, normalize=normalize)


# ======================= keras loss wrappers ===================

class CategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    """
    Wraps tf.keras.losses.CategoricalCrossentropy with explicit label_weights support.
    """

    def __init__(self, label_weights=None, **kwargs):
        """
        Args:
            label_weights: list, array, or Tensor of length nb_labels.
        """
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = tf.convert_to_tensor(label_weights)
        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.cce(y_true, y_pred, sample_weight=sample_weight)

    def cce(self, y_true, y_pred, sample_weight=None):
        wts = 1
        if self.label_weights is not None:
            D = y_pred.ndim
            wts = tf.reshape(self.label_weights, [1] * (D - 1) + [-1])
        sample_weight = (sample_weight if sample_weight is not None else 1) * wts
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class MeanSquaredErrorProb(tf.keras.losses.MeanSquaredError):
    """
    Wraps tf.keras.losses.MeanSquaredError with label weights along the last dimension.
    """

    def __init__(self, label_weights=None, **kwargs):
        """
        Args:
            label_weights: list, array, or Tensor of length nb_labels.
        """
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = tf.convert_to_tensor(label_weights)
        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.mse(y_true, y_pred, sample_weight=sample_weight)

    def mse(self, y_true, y_pred, sample_weight=None):
        if self.label_weights is not None:
            yf = y_pred.shape[-1]
            lf = len(self.label_weights)
            if yf != lf:
                raise ValueError(f'Label weights must be of len {yf}, but got {lf}.')

            y_true = y_true[..., None]
            y_pred = y_pred[..., None]
            sample_weight = (self.label_weights if sample_weight is None
                             else sample_weight * self.label_weights)

        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


# ======================= decorators ============================

def multiple_metrics_decorator(metrics, weights=None):
    """
    Apply multiple metrics to a given output with optional weighting.

    Args:
        metrics: list of metric functions, each taking two Tensors.
        weights: weight per metric. Defaults to uniform weights.

    Returns:
        Combined metric function.
    """
    if weights is None:
        weights = np.ones(len(metrics))

    def metric(y_true, y_pred):
        return sum(weights[idx] * met(y_true, y_pred) for idx, met in enumerate(metrics))

    return metric