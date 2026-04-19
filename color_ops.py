import numpy as np
import tensorflow as tf


def adjust_hsv_in_yiq(input: tf.Tensor) -> tf.Tensor:
    """
    Adjust hue, saturation, and value of an image in YIQ space.
    Based on tfa.image.distort_image_ops.
    Reference: https://beesbuzz.biz/code/hsv_color_transforms.php
    """
    image, delta_hue, scale_saturation, scale_value = input
    delta_hue = tf.squeeze(delta_hue)
    scale_saturation = tf.squeeze(scale_saturation)
    scale_value = tf.squeeze(scale_value)

    assert image.dtype in [tf.float16, tf.float32, tf.float64]

    if image.shape.rank is not None and image.shape.rank < 3:
        raise ValueError("input must be at least 3-D.")
    if image.shape[-1] is not None and image.shape[-1] != 3:
        raise ValueError(f"input must have 3 channels but instead has {image.shape[-1]}.")

    yiq = tf.constant(
        [[0.299,  0.596,  0.211],
         [0.587, -0.274, -0.523],
         [0.114, -0.322,  0.312]],
        dtype=image.dtype)

    yiq_inverse = tf.constant(
        [[1.0,       1.0,       1.0      ],
         [0.95617069, -0.2726886, -1.103744],
         [0.62143257, -0.64681324, 1.70062309]],
        dtype=image.dtype)

    vsu = scale_value * scale_saturation * tf.math.cos(delta_hue)
    vsw = scale_value * scale_saturation * tf.math.sin(delta_hue)

    hsv_transform = tf.convert_to_tensor(
        [[scale_value, 0,    0  ],
         [0,           vsu,  vsw],
         [0,          -vsw,  vsu]],
        dtype=image.dtype)

    transform_matrix = yiq @ hsv_transform @ yiq_inverse
    return image @ transform_matrix


def rgb2yiq_tf(image: tf.Tensor) -> tf.Tensor:
    """Convert an RGB image tensor to YIQ color space."""
    out_shape = image.get_shape().as_list()

    yiq = tf.constant(
        [[ 0.299,   0.587,   0.114 ],
         [ 0.5959, -0.2746, -0.3213],
         [ 0.2115, -0.5227,  0.3112]],
        dtype=image.dtype)

    image = tf.expand_dims(image, -1)
    for _ in range(len(image.get_shape().as_list()) - 2):
        yiq = np.expand_dims(yiq, 0)

    return tf.reshape(tf.matmul(yiq, image), out_shape)


def rgb2hsl_tf(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an RGB image tensor to HSL color space.
    Input image should be in [0, 1] scale.
    Based on: https://github.com/heidariarash/Alpha-Omega/blob/9c68b7b63c6ac0dde16757dca2d06465d93fc7f8/alphaomega/cv/channel/conversion.py#L210
    """
    img_shape = image.get_shape().as_list()
    assert len(img_shape) == 4 and img_shape[-1] == 3  # B, H, W, 3

    image = tf.clip_by_value(image, 0.0, 1.0)

    rprime = image[:, :, :, 0]
    gprime = image[:, :, :, 1]
    bprime = image[:, :, :, 2]
    cmax = tf.reduce_max(image, axis=-1)
    cmin = tf.reduce_min(image, axis=-1)
    delta = cmax - cmin

    L_channel = (cmax + cmin) / 2

    # saturation channel
    S_channel = tf.zeros_like(L_channel)
    S_channel_mask = tf.math.logical_and(L_channel > 0.0, L_channel < 1.0)
    S_channel += tf.cast(S_channel_mask, tf.float32) * (
        (cmax - L_channel) / tf.clip_by_value(tf.math.minimum(L_channel, 1 - L_channel), 1e-5, 1 - 1e-5))

    # hue channel
    H_channel = tf.zeros_like(L_channel)
    mask_r = cmax == rprime
    mask_g = tf.math.logical_and(~mask_r, cmax == gprime)
    mask_b = tf.math.logical_and(tf.math.logical_and(~mask_r, ~(cmax == gprime)), cmax == bprime)

    H_channel += tf.cast(mask_r, tf.float32) * tf.cast(
        tf.cast((60 * ((gprime - bprime) / delta)), tf.int32) % 360, tf.float32)
    H_channel += tf.cast(mask_g, tf.float32) * tf.cast(
        (tf.cast((60 * ((bprime - rprime) / delta)), tf.int32) + 120) % 360, tf.float32)
    H_channel += tf.cast(mask_b, tf.float32) * tf.cast(
        (tf.cast((60 * ((rprime - gprime) / delta)), tf.int32) + 240) % 360, tf.float32)

    converted = tf.stack([H_channel, S_channel, L_channel], axis=-1)
    assert len(converted.get_shape().as_list()) == len(img_shape)

    return converted