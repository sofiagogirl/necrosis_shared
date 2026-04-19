import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from keras.layers import (Layer, Input, concatenate, LeakyReLU, AveragePooling2D)
from keras.initializers import RandomNormal

import color_ops


# ======================= color space conversions ================

def _matmul_transform(img, matrix):
    """Apply a color space matrix transform to a batched image tensor."""
    out_shape = img.get_shape().as_list()
    img = tf.expand_dims(img, -1)
    for _ in range(len(img.get_shape().as_list()) - 2):
        matrix = tf.expand_dims(matrix, 0)
    return tf.reshape(tf.matmul(matrix, img), out_shape)


def rgb2lms(rgb_img):
    """Convert RGB image tensor to LMS color space."""
    RGB2LMS = tf.constant(np.array([[0.3811, 0.5783, 0.0402],
                                    [0.1967, 0.7244, 0.0782],
                                    [0.0241, 0.1288, 0.8444]]), dtype='float32')
    return _matmul_transform(rgb_img, RGB2LMS)


def lms2lab(lms_img):
    """Convert LMS image tensor to Lab color space."""
    T1 = tf.constant(np.array([[1, 1, 1],
                                [1, 1, -2],
                                [1, -1, 0]]), dtype='float32')
    T2 = tf.constant(np.array([[1 / np.sqrt(3), 0, 0],
                                [0, 1 / np.sqrt(6), 0],
                                [0, 0, 1 / np.sqrt(2)]]), dtype='float32')
    out_shape = lms_img.get_shape().as_list()
    lms_img = tf.expand_dims(lms_img, -1)
    for _ in range(len(lms_img.get_shape().as_list()) - 2):
        T1 = tf.expand_dims(T1, 0)
        T2 = tf.expand_dims(T2, 0)
    return tf.reshape(tf.matmul(T2, tf.matmul(T1, lms_img)), out_shape)


def lab2lms(lab_img):
    """Convert Lab image tensor to LMS color space."""
    T1 = tf.constant(np.array([[np.sqrt(3) / 3, 0, 0],
                                [0, np.sqrt(6) / 6, 0],
                                [0, 0, np.sqrt(2) / 2]]), dtype='float32')
    T2 = tf.constant(np.array([[1, 1, 1],
                                [1, 1, -1],
                                [1, -2, 0]]), dtype='float32')
    out_shape = lab_img.get_shape().as_list()
    lab_img = tf.expand_dims(lab_img, -1)
    for _ in range(len(lab_img.get_shape().as_list()) - 2):
        T1 = tf.expand_dims(T1, 0)
        T2 = tf.expand_dims(T2, 0)
    return tf.reshape(tf.matmul(T2, tf.matmul(T1, lab_img)), out_shape)


def lms2rgb(lms_img):
    """Convert LMS image tensor to RGB color space."""
    LMS2RGB = tf.constant(np.array([[4.4679, -3.5873, 0.1193],
                                    [-1.2186, 2.3809, -0.1624],
                                    [0.0497, -0.2439, 1.2045]]), dtype='float32')
    return _matmul_transform(lms_img, LMS2RGB)


def rgb2lab_tf(img_rgb):
    """Convert RGB image tensor to Lab color space."""
    img_rgb = tf.clip_by_value(img_rgb, 0.0, 255.0)
    img_lms = rgb2lms(img_rgb)
    img_lms = tf.clip_by_value(img_lms, 1e-10, 1e10)
    img_lms = tf.math.log(img_lms)
    return lms2lab(img_lms)


def lab2rgb_tf(img_lab):
    """Convert Lab image tensor to RGB color space."""
    img_lms = lab2lms(img_lab)
    img_lms = tf.math.exp(img_lms)
    return lms2rgb(img_lms)


# ======================= unet cores ============================

def unet_core_v4(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None,
                 src_feats=3, tgt_feats=3):
    """
    Encoder-only U-net core (decoder is omitted). Returns bottleneck representation.

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


# ======================= color aligner models ===================

class MapLayer(Layer):
    """Apply per-sample HSV adjustment in YIQ space."""

    def call(self, moving, hue_delta, saturation_factor, scale_values):
        def adjust_hsv_in_yiq(inp):
            img, hue_delta, saturation_factor, scale_values = inp
            delta_hue = tf.squeeze(hue_delta)
            scale_saturation = tf.squeeze(saturation_factor)
            scale_value = tf.squeeze(scale_values)

            assert img.dtype in [tf.float16, tf.float32, tf.float64]
            if img.shape.rank is not None and img.shape.rank < 3:
                raise ValueError("input must be at least 3-D.")
            if img.shape[-1] is not None and img.shape[-1] != 3:
                raise ValueError(f"input must have 3 channels but got {img.shape[-1]}.")

            yiq = tf.constant([[0.299,  0.596,  0.211],
                                [0.587, -0.274, -0.523],
                                [0.114, -0.322,  0.312]], dtype=img.dtype)
            yiq_inverse = tf.constant([[1.0,       1.0,        1.0      ],
                                       [0.95617069, -0.2726886, -1.103744],
                                       [0.62143257, -0.64681324, 1.70062309]], dtype=img.dtype)

            vsu = scale_value * scale_saturation * tf.math.cos(delta_hue)
            vsw = scale_value * scale_saturation * tf.math.sin(delta_hue)
            hsv_transform = tf.convert_to_tensor([[scale_value, 0,    0  ],
                                                  [0,           vsu,  vsw],
                                                  [0,          -vsw,  vsu]], dtype=img.dtype)
            return (img @ (yiq @ hsv_transform @ yiq_inverse),
                    hue_delta, saturation_factor, scale_values)

        return tf.map_fn(adjust_hsv_in_yiq,
                         (moving, hue_delta, saturation_factor, scale_values))[0]


def color_aligner_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                                    fix_hsv_value=None, hsv_value_regularizer=None):
    """
    U-net with HSV color adjustment predicted from the bottleneck representation.

    Args:
        vol_size: volume size.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused).
        fix_hsv_value: if set (float), fixes the V channel to this value.
        hsv_value_regularizer: optional activity regularizer for the Dense layer.

    Returns:
        Keras Model with outputs [moving_adjusted, flow].
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2 ** (n_levels + 1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims
    if fix_hsv_value is not None:
        assert isinstance(fix_hsv_value, float)

    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    bottleneck_repr = unet_model.output

    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size),
                                name='flow_pooling_1')(flow)
    flow = KL.Flatten()(flow)

    n_out = 2 if fix_hsv_value is not None else 3
    flow = KL.Dense(n_out, activity_regularizer=hsv_value_regularizer)(flow)

    hue_deltas = tf.clip_by_value(flow[:, 0:1], -1, 1)
    saturation_factors = flow[:, 1:2]
    scale_values = (tf.ones_like(saturation_factors) * fix_hsv_value
                    if fix_hsv_value is not None else flow[:, 2:3])

    def _expand(x):
        return tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)

    moving_adjusted = MapLayer()(moving, _expand(hue_deltas), _expand(saturation_factors), _expand(scale_values))

    return Model(inputs=[moving, fixed], outputs=[moving_adjusted, flow])


def color_aligner_lab_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    U-net with Lab color normalization predicted from the bottleneck representation.

    Args:
        vol_size: volume size. Input should be in Lab color space.
        enc_nf: encoder filter counts.
        dec_nf: decoder filter counts (unused).

    Returns:
        Keras Model with outputs [moving_lab_transformed, flow].
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2 ** (n_levels + 1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be 1, 2, or 3. found: %d" % ndims

    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving_lab, fixed_lab] = unet_model.inputs
    bottleneck_repr = unet_model.output

    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size),
                                name='flow_pooling_1')(flow)
    flow = KL.Flatten()(flow)
    flow = KL.Dense(6)(flow)  # (l_mean, a_mean, b_mean, l_std, a_std, b_std)

    pred_target_mean = tf.expand_dims(tf.expand_dims(flow[:, 0:3], 1), 1)
    pred_target_std  = tf.expand_dims(tf.expand_dims(flow[:, 3:6], 1), 1)
    source_mean = tf.math.reduce_mean(moving_lab, axis=(1, 2), keepdims=True)
    source_std  = tf.math.reduce_std(moving_lab, axis=(1, 2), keepdims=True)

    moving_lab_transformed = (moving_lab - source_mean) * (pred_target_std / source_std) + pred_target_mean

    return Model(inputs=[moving_lab, fixed_lab], outputs=[moving_lab_transformed, flow])


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

    # residual connection with zero-padding along channel dim
    x_in_padded = tf.pad(x_in, [[0, 0], [0, 0], [0, 0],
                                  [0, nf - x_in.get_shape().as_list()[-1]]], 'CONSTANT')
    x_out = LeakyReLU(0.2)(x_out + x_in_padded)
    return AveragePooling2D(pool_size=(2, 2))(x_out)