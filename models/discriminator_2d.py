# based on https://github.com/lopeneljxi/keras-unet-collection

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .backbone_zoo import backbone_zoo, bach_norm_checker
from .layer_utils import *


# ======================= encoder block ==========================

def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0'):
    """
    Encoder block of U-net.

    Args:
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: keras activation, e.g. 'ReLU'.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    pool_size = 2
    X = encode_layer(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv'.format(name))
    return X


# ======================= discriminator ==========================

def discriminator_base(input_tensor, filter_num, stack_num_down=2,
                       activation='ReLU', batch_norm=False, pool=True,
                       backbone=None, name='unet'):
    """
    Base of the discriminator CNN.

    Args:
        input_tensor: input tensor, e.g. keras.layers.Input((None, None, 3)).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        stack_num_down: convolutional layers per downsampling block.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    X = input_tensor

    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                   batch_norm=False, name='{}_down0'.format(name))

    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=False,
                      batch_norm=False, name='{}_down{}'.format(name, i + 1))

    X = tf.reduce_mean(X, axis=(1, 2))
    ch = X.get_shape().as_list()[-1]
    X = dense_layer(X, units=ch, activation='LeakyReLU', name='dense_layer_1')
    X = dense_layer(X, units=1, activation=None, name='dense_layer_2')
    X = tf.nn.sigmoid(X)

    return X


def discriminator_2d(input_size, filter_num, stack_num_down=2,
                     activation='ReLU', batch_norm=False, pool=False,
                     backbone=None, name='unet'):
    """
    2D discriminator model.

    Args:
        input_size: network input shape, e.g. (128, 128, 3).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        stack_num_down: convolutional layers per downsampling block.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        backbone: backbone model name from tensorflow.keras.applications, or None.
        name: prefix for created keras layers.

    Returns:
        model: a Keras model.
    """
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)
    OUT = discriminator_base(IN, filter_num, stack_num_down=stack_num_down,
                             activation=activation, batch_norm=batch_norm, pool=pool, name=name)

    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    return model