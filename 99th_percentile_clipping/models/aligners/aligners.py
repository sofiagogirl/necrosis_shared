"""
Network architectures for voxelmorph-style registration models.

These architectures were designed for specific papers but the concepts
are not tied to any particular architecture.

Reference:
    Dalca AV et al. Unsupervised Learning for Fast Probabilistic Diffeomorphic
    Registration. MICCAI 2018.
"""

import sys
import numpy as np
import scipy.stats as st
import scipy.ndimage.filters as filts

import keras
import keras.backend as K
import keras.layers as KL
import tensorflow as tf
from keras.models import Model
from keras.layers import (Layer, Conv3D, Activation, Input, UpSampling3D,
                          concatenate, LeakyReLU, Reshape, Lambda,
                          DepthwiseConv2D, Conv2D)
from keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Lambda

from .layers import SpatialTransformer, VecInt, Resize


# ======================= gaussian utils =========================

def matlab_style_gauss2D(shape=(40, 40), sigma=20):
    """2D Gaussian kernel matching MATLAB's fspecial('gaussian', shape, sigma)."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gausv2(inp, kernel_size=80):
    """Apply a fixed Gaussian blur via DepthwiseConv2D."""
    kernel_weights = matlab_style_gauss2D(shape=(kernel_size, kernel_size),
                                          sigma=kernel_size // 2)
    in_channels = 2
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)

    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    g_layer_out = g_layer(inp)
    g_layer.set_weights([kernel_weights])
    g_layer.trainable = False
    return g_layer_out


# ======================= unet cores ============================

def unet_core(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
              src_feats=3, tgt_feats=3):
    """
    U-net encoder-decoder core for VoxelMorph (CVPR 2018).

    Args:
        vol_size: volume size, e.g. (256, 256, 256).
        enc_nf: encoder filter counts, e.g. [16, 32, 32, 32].
        dec_nf: decoder filter counts (length 6 or 7).
        full_size: whether to upsample to full resolution.
        src, tgt: optional pre-defined input tensors.
        src_feats, tgt_feats: number of input channels.

    Returns:
        Keras Model with outputs [x].
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    # encoder
    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block(x_enc[-1], nf, 2))

    # decoder
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])

    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x])


def unet_core_v2(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                 src_feats=3, tgt_feats=3):
    """
    U-net core that also returns the bottleneck representation.

    Returns:
        Keras Model with outputs [x, bottleneck_repr].
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block(x_enc[-1], nf, 2))

    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])

    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x, x_enc[-1]])


def unet_core_vJX(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                  src_feats=3, tgt_feats=3):
    """
    Modified U-net core with double conv blocks at each level.

    Returns:
        Keras Model with outputs [x].
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    n_level = len(enc_nf)

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    # encoder with double conv per level
    x_enc = [x_in]
    for i in range(n_level):
        x_temp = conv_block(x_enc[-1], enc_nf[i])
        x_enc.append(conv_block(x_temp, enc_nf[i], 2))

    # bottleneck
    x = conv_block(x_enc[-1], dec_nf[0])

    # decoder
    for i in range(1, n_level + 1):
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[-(i + 1)]])
        x = conv_block(x, dec_nf[i])
        x = conv_block(x, dec_nf[i])

    x = conv_block(x, dec_nf[n_level])
    x = conv_block(x, dec_nf[n_level + 1])

    return Model(inputs=[src, tgt], outputs=[x])


# ======================= aligner models ========================

def _apply_flow_clipping(flow, flow_clipping, flow_clipping_nsigma,
                         flow_thresholding, flow_thresh_dis):
    """Apply optional clipping and thresholding to a flow field."""
    flow_before_clipping = flow

    if flow_clipping:
        assert flow_clipping_nsigma is not None
        flow_mean = tf.reduce_mean(flow, axis=(1, 2))
        flow_std = tf.math.reduce_std(flow, axis=(1, 2))
        clip_min = tf.expand_dims(tf.expand_dims(flow_mean - flow_clipping_nsigma * flow_std, 1), 1)
        clip_max = tf.expand_dims(tf.expand_dims(flow_mean + flow_clipping_nsigma * flow_std, 1), 1)
        if flow_thresholding:
            clip_min = tf.maximum(clip_min, -flow_thresh_dis)
            clip_max = tf.minimum(clip_max, flow_thresh_dis)
        flow = tf.clip_by_value(flow, clip_min, clip_max)

    elif flow_thresholding:
        flow = tf.clip_by_value(flow, -flow_thresh_dis, flow_thresh_dis)

    return flow, flow_before_clipping


def aligner_unet_cvpr2018(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                           flow_only=False, gauss_kernal_size=80,
                           flow_clipping=False, flow_clipping_nsigma=1,
                           flow_thresholding=False, flow_thresh_dis=30):
    """
    VoxelMorph-style aligner U-net (CVPR 2018).

    Args:
        vol_size: volume size, e.g. (256, 256).
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts.
        flow_only: if True, only output flow without applying transform.
        gauss_kernal_size: Gaussian smoothing kernel size for the flow.
        flow_clipping: clip flow by per-pixel mean ± nsigma*std.
        flow_clipping_nsigma: number of sigmas for clipping.
        flow_thresholding: hard-clip flow to ±flow_thresh_dis.
        flow_thresh_dis: threshold distance for hard clipping.

    Returns:
        Keras Model.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    x = unet_model.output

    Conv = getattr(KL, 'Conv%dD' % 2)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    flow, flow_before_clipping = _apply_flow_clipping(
        flow, flow_clipping, flow_clipping_nsigma, flow_thresholding, flow_thresh_dis)

    flow = gausv2(flow, kernel_size=gauss_kernal_size)

    if not flow_only:
        moving_transformed = SpatialTransformer(interp_method='linear', indexing=indexing)([moving, flow])
        if flow_clipping or flow_thresholding:
            return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow, flow_before_clipping])
        return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow])

    return Model(inputs=[moving, fixed], outputs=[flow])


def aligner_unet_cvpr2018_v2(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                              flow_only=False, flow_clipping=False, flow_clipping_nsigma=1,
                              shifting_only=False, gauss_kernal_size=80):
    """
    VoxelMorph aligner that combines a dense warp with a global shift prediction.

    Args:
        vol_size: volume size.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts.
        flow_only: if True, only output flow.
        flow_clipping: clip flow to mean ± nsigma*std.
        flow_clipping_nsigma: number of sigmas for clipping.
        shifting_only: if True, only predict global shift (no dense warp).
        gauss_kernal_size: Gaussian smoothing kernel size.

    Returns:
        Keras Model.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core_v2(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    [x, bottleneck_repr] = unet_model.output

    Conv = getattr(KL, 'Conv%dD' % 2)
    flow_warping = Conv(ndims, kernel_size=3, padding='same', name='flow_warp',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # global shift from bottleneck
    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)
    flow = KL.AveragePooling2D(pool_size=(8, 8), name='flow_pooling_1')(flow)
    flow = KL.Flatten()(flow)
    flow = KL.Dense(ndims)(flow)
    flow = tf.expand_dims(tf.expand_dims(flow, axis=1), axis=1)
    flow = tf.broadcast_to(flow, tf.shape(flow_warping))

    if not shifting_only:
        flow = flow + flow_warping
        flow_before_clipping = flow
        if flow_clipping:
            assert flow_clipping_nsigma is not None
            flow_mean = tf.reduce_mean(flow, axis=(1, 2))
            flow_std = tf.math.reduce_std(flow, axis=(1, 2))
            clip_min = tf.expand_dims(tf.expand_dims(flow_mean - flow_clipping_nsigma * flow_std, 1), 1)
            clip_max = tf.expand_dims(tf.expand_dims(flow_mean + flow_clipping_nsigma * flow_std, 1), 1)
            flow = tf.clip_by_value(flow, clip_min, clip_max)
        flow = gausv2(flow, kernel_size=gauss_kernal_size)

    if not flow_only:
        moving_transformed = SpatialTransformer(interp_method='linear', indexing=indexing)([moving, flow])
        if flow_clipping:
            return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow, flow_before_clipping])
        return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow])

    return Model(inputs=[moving, fixed], outputs=[flow])


def aligner_unet_cvpr2018_vJX(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                               flow_only=False, gauss_kernal_size=80,
                               flow_clipping=False, flow_clipping_nsigma=1,
                               flow_thresholding=False, flow_thresh_dis=30,
                               loss_mask=False, loss_mask_from_prev_cascade=False):
    """
    Modified VoxelMorph aligner with optional loss masking.

    Args:
        vol_size: volume size.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts.
        flow_only: if True, only output flow.
        gauss_kernal_size: Gaussian smoothing kernel size.
        flow_clipping: clip flow by per-pixel mean ± nsigma*std.
        flow_clipping_nsigma: number of sigmas for clipping.
        flow_thresholding: hard-clip flow to ±flow_thresh_dis.
        flow_thresh_dis: threshold distance for hard clipping.
        loss_mask: if True, attach a mask to the warped moving image channel dim.
        loss_mask_from_prev_cascade: if True, use mask from a previous cascade.

    Returns:
        Keras Model.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core_vJX(vol_size, enc_nf, dec_nf, full_size=full_size, src_feats=3, tgt_feats=3)
    [moving, fixed] = unet_model.inputs
    x = unet_model.output

    if loss_mask:
        if loss_mask_from_prev_cascade:
            moving = Input(shape=[*vol_size, 4])
            tgt = moving[:, :, :, :-1]
            train_loss_mask = moving[:, :, :, -1:]
            x = unet_model(tgt, fixed)
        else:
            train_loss_mask = tf.ones(
                [moving.get_shape().as_list()[0], vol_size[0] - 4, vol_size[1] - 4, 1], dtype=tf.float32)
            train_loss_mask = tf.pad(train_loss_mask, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]),
                                     mode="CONSTANT")

    Conv = getattr(KL, 'Conv%dD' % 2)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    flow, flow_before_clipping = _apply_flow_clipping(
        flow, flow_clipping, flow_clipping_nsigma, flow_thresholding, flow_thresh_dis)

    flow = gausv2(flow, kernel_size=gauss_kernal_size)

    moving_for_warp = tf.concat([moving, train_loss_mask], axis=-1) if loss_mask else moving

    if not flow_only:
        moving_transformed = SpatialTransformer(interp_method='linear', indexing=indexing)([moving_for_warp, flow])
        if flow_clipping or flow_thresholding:
            return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow, flow_before_clipping])
        return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow])

    return Model(inputs=[moving, fixed], outputs=[flow])


# ======================= simple models =========================

def nn_trf(vol_size, indexing='xy'):
    """
    Nearest-neighbor transform model.

    Args:
        vol_size: volume size.
        indexing: 'xy' or 'ij'.

    Returns:
        Keras Model.
    """
    ndims = len(vol_size)
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims), name='trf_input')
    nn_output = SpatialTransformer(interp_method='nearest', indexing=indexing)([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_output)


# ======================= helper functions =======================

def conv_block(x_in, nf, strides=1):
    """Conv2D + LeakyReLU block."""
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    Conv = getattr(KL, 'Conv%dD' % 2)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    return LeakyReLU(0.2)(x_out)


def sample(args):
    """Sample from a normal distribution given (mu, log_sigma)."""
    mu, log_sigma = args[0], args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    return mu + tf.exp(log_sigma / 2.0) * noise


def trf_resize(trf, vel_resize, name='flow'):
    """Resize and rescale a transform field."""
    if vel_resize > 1:
        trf = Resize(1 / vel_resize, name=name + '_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)
    else:
        trf = Rescale(1 / vel_resize, name=name + '_tmp')(trf)
        return Resize(1 / vel_resize, name=name)(trf)


# ======================= helper layers =========================

class Sample(Layer):
    """Keras layer: Gaussian sample from [mu, log_sigma]."""

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Negate(Layer):
    """Keras layer: negate the input."""

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class Rescale(Layer):
    """Keras layer: rescale input by a fixed factor."""

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)

    def call(self, x):
        return x * self.resize

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleDouble(Rescale):
    """Rescale by factor of 2."""
    def __init__(self, **kwargs):
        super(RescaleDouble, self).__init__(2, **kwargs)


class ResizeDouble(Resize):
    """Resize by factor of 2."""
    def __init__(self, **kwargs):
        super(ResizeDouble, self).__init__(2, **kwargs)