import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from keras.layers import (Input, concatenate, LeakyReLU, AveragePooling2D)
from keras.initializers import RandomNormal

from .stn_affine import spatial_transformer_network
from .utils import affine_to_shift, batch_affine_to_shift
from .layers import SpatialTransformer


# ======================= unet cores ============================

def unet_core_v3(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                 src_feats=3, tgt_feats=3):
    """
    Encoder-only U-net core. Returns bottleneck representation.

    Args:
        vol_size: volume size, e.g. (256, 256).
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused, kept for API compatibility).
        src, tgt: optional pre-defined input tensors.
        src_feats, tgt_feats: number of input channels.

    Returns:
        Keras Model outputting bottleneck tensor.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block(x_enc[-1], nf, 2))

    return Model(inputs=[src, tgt], outputs=x_enc[-1])


def unet_core_v4(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                 src_feats=3, tgt_feats=3):
    """
    Encoder-only U-net core with double conv blocks. Returns bottleneck representation.

    Args:
        vol_size: volume size, e.g. (256, 256).
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused, kept for API compatibility).
        src, tgt: optional pre-defined input tensors.
        src_feats, tgt_feats: number of input channels.

    Returns:
        Keras Model outputting bottleneck tensor.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block_v2(x_enc[-1], nf, 2))

    return Model(inputs=[src, tgt], outputs=x_enc[-1])


def unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                           src_feats=3, tgt_feats=3):
    """
    Encoder-only U-net core with residual blocks. Returns bottleneck representation.

    Args:
        vol_size: volume size, e.g. (256, 256).
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused, kept for API compatibility).
        src, tgt: optional pre-defined input tensors.
        src_feats, tgt_feats: number of input channels.

    Returns:
        Keras Model outputting bottleneck tensor.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block_v2_residual(x_enc[-1], nf))

    return Model(inputs=[src, tgt], outputs=x_enc[-1])


# ======================= aligner models ========================

def aligner_unet_cvpr2018_v3(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    U-net with affine translation prediction from the bottleneck (translation only).

    Args:
        vol_size: volume size.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused).

    Returns:
        Keras Model with outputs [moving_transformed, flow_affine].
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core_v3(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    bottleneck_repr = unet_model.output

    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)
    flow = KL.AveragePooling2D(pool_size=(8, 8), name='flow_pooling_1')(flow)
    flow = KL.Flatten()(flow)
    flow = KL.Dense(2)(flow)  # (x_shift, y_shift)

    # build affine matrix: translation only
    flow_affine = tf.repeat(flow, [3, 3], axis=1)                  # (Batch, 6)
    flow_affine = flow_affine * tf.constant([0., 0., 1., 0., 0., 1.])
    flow_affine = flow_affine + tf.constant([1., 0., 0., 0., 1., 0.])

    moving_transformed = spatial_transformer_network(moving, flow_affine)
    return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow_affine])


def aligner_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                              loss_mask=False, loss_mask_from_prev_cascade=False):
    """
    U-net with affine translation + shear prediction from the bottleneck.

    Args:
        vol_size: volume size.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused).
        loss_mask: if True, attach a loss mask to the warped moving image.
        loss_mask_from_prev_cascade: if True, use mask from a previous cascade.

    Returns:
        Keras Model with outputs [moving_transformed, flow_affine].
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2 ** (n_levels + 1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    bottleneck_repr = unet_model.output

    if loss_mask:
        if loss_mask_from_prev_cascade:
            moving = Input(shape=[*vol_size, 4])
            tgt = moving[:, :, :, :-1]
            train_loss_mask = moving[:, :, :, -1:]
            bottleneck_repr = unet_model(tgt, fixed)
        else:
            train_loss_mask = tf.ones(
                [moving.get_shape().as_list()[0], vol_size[0] - 4, vol_size[1] - 4, 1], dtype=tf.float32)
            train_loss_mask = tf.pad(train_loss_mask, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]),
                                     mode="CONSTANT")

    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size),
                                name='flow_pooling_1')(flow)
    flow = KL.Flatten()(flow)
    flow = KL.Dense(4)(flow)  # (tx, x_shift, ty, y_shift)

    # build affine matrix: translation + shear
    flow_affine = tf.repeat(flow, [2, 1, 2, 1], axis=1)            # (Batch, 6)
    flow_affine = flow_affine * tf.constant([0., 1., 1., 1., 0., 1.])
    flow_affine = flow_affine + tf.constant([1., 0., 0., 0., 1., 0.])

    moving_for_warp = tf.concat([moving, train_loss_mask], axis=-1) if loss_mask else moving
    moving_transformed = spatial_transformer_network(moving_for_warp, flow_affine)

    return Model(inputs=[moving, fixed], outputs=[moving_transformed, flow_affine])


# ======================= conv blocks ============================

def conv_block(x_in, nf, strides=1):
    """Single Conv2D + LeakyReLU block."""
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    Conv = getattr(KL, 'Conv%dD' % 2)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    return LeakyReLU(0.2)(x_out)


def conv_block_v2(x_in, nf, strides=1):
    """Double Conv2D + LeakyReLU block (strided first conv, stride-1 second)."""
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    Conv = getattr(KL, 'Conv%dD' % 2)
    x_mid = LeakyReLU(0.2)(Conv(nf, kernel_size=3, padding='same',
                                 kernel_initializer='he_normal', strides=strides)(x_in))
    x_out = LeakyReLU(0.2)(Conv(nf, kernel_size=3, padding='same',
                                 kernel_initializer='he_normal', strides=1)(x_mid))
    return x_out


def conv_block_v2_residual(x_in, nf):
    """
    Double stride-1 Conv2D with residual connection, followed by AveragePooling2D.
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    Conv = getattr(KL, 'Conv%dD' % 2)

    x_mid = LeakyReLU(0.2)(Conv(nf, kernel_size=3, padding='same',
                                  kernel_initializer='he_normal', strides=1)(x_in))
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_mid)

    x_in_padded = tf.pad(x_in, [[0, 0], [0, 0], [0, 0],
                                  [0, nf - x_in.get_shape().as_list()[-1]]], 'CONSTANT')
    x_out = LeakyReLU(0.2)(x_out + x_in_padded)
    return AveragePooling2D(pool_size=(2, 2))(x_out)