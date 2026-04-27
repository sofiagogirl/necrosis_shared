from __future__ import absolute_import

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .layer_utils import *
from .activations import GELU, Snake
from .unet_2d import UNET_left, UNET_right
from .backbone_zoo import backbone_zoo, bach_norm_checker


# ======================= attention decoder block ================

def UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):
    """
    Decoder block of Attention U-net.

    Args:
        X: input tensor.
        X_left: skip connection tensor from corresponding downsampling block.
        channel: number of convolution filters.
        att_channel: number of intermediate attention channels.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: keras activation, e.g. 'ReLU'.
        atten_activation: nonlinear attention activation (sigma_1 in Oktay et al. 2018).
        attention: 'add' for additive, 'multiply' for multiplicative attention.
        unpool: True/'bilinear' for bilinear Upsampling2D, 'nearest' for nearest,
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        H: output tensor.
    """
    pool_size = 2

    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    X_left = attention_gate(X=X_left, g=X, channel=att_channel, activation=atten_activation,
                            attention=attention, name='{}_att'.format(name))

    H = concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))
    H = CONV_stack(H, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))

    return H


# ======================= attention u-net base ===================

def att_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add',
                     batch_norm=False, pool=True, unpool=True,
                     backbone=None, weights='imagenet', freeze_backbone=True,
                     freeze_batch_norm=True, name='attunet'):
    """
    Base of Attention U-net with optional ImageNet backbone.

    Reference:
        Oktay et al., 2018. Attention u-net: Learning where to look for the pancreas.
        arXiv:1804.03999.

    Args:
        input_tensor: input tensor, e.g. keras.layers.Input((None, None, 3)).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        activation: keras activation, e.g. 'ReLU'.
        atten_activation: nonlinear attention activation (sigma_1 in Oktay et al. 2018).
        attention: 'add' for additive, 'multiply' for multiplicative attention.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        unpool: True/'bilinear' for bilinear Upsampling2D, 'nearest' for nearest,
                False for Conv2DTranspose + batch norm + activation.
        backbone: backbone model name from tensorflow.keras.applications, or None.
        weights: None, 'imagenet', or path to weights file.
        freeze_backbone: True to freeze backbone weights.
        freeze_batch_norm: False to keep batch normalization layers trainable.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    depth_ = len(filter_num)
    X_skip = []

    if backbone is None:
        X = input_tensor
        X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                       batch_norm=batch_norm, name='{}_down0'.format(name))
        X_skip.append(X)

        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool,
                          batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_skip.append(X)
    else:
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            X_skip = backbone_([input_tensor])
            depth_encode = len(X_skip)
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            X_skip = backbone_([input_tensor])
            depth_encode = len(X_skip) + 1

        if depth_encode < depth_:
            X = X_skip[-1]
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode
                X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_skip.append(X)

    # ======================= upsampling =========================
    X_skip = X_skip[::-1]
    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    filter_num_decode = filter_num[:-1][::-1]

    for i in range(depth_decode):
        f = filter_num_decode[i]
        X = UNET_att_right(X, X_decode[i], f, att_channel=f // 2, stack_num=stack_num_up,
                           activation=activation, atten_activation=atten_activation, attention=attention,
                           unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False,
                           name='{}_up{}'.format(name, i_real))

    return X


# ======================= attention u-net model ==================

def att_unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                activation='ReLU', atten_activation='ReLU', attention='add',
                output_activation='Softmax', batch_norm=False, pool=True, unpool=True,
                backbone=None, weights='imagenet', freeze_backbone=True,
                freeze_batch_norm=True, name='attunet'):
    """
    Attention U-net with optional ImageNet backbone.

    Reference:
        Oktay et al., 2018. Attention u-net: Learning where to look for the pancreas.
        arXiv:1804.03999.

    Args:
        input_size: network input shape, e.g. (128, 128, 3).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        n_labels: number of output labels.
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        activation: keras activation, e.g. 'ReLU'.
        atten_activation: nonlinear attention activation (sigma_1 in Oktay et al. 2018).
        attention: 'add' for additive, 'multiply' for multiplicative attention.
        output_activation: output activation, e.g. 'Softmax', 'Sigmoid', or None for linear.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        unpool: True/'bilinear' for bilinear Upsampling2D, 'nearest' for nearest,
                False for Conv2DTranspose + batch norm + activation.
        backbone: backbone model name from tensorflow.keras.applications, or None.
        weights: None, 'imagenet', or path to weights file.
        freeze_backbone: True to freeze backbone weights.
        freeze_batch_norm: False to keep batch normalization layers trainable.
        name: prefix for created keras layers.

    Returns:
        model: a Keras model.
    """
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)
    X = att_unet_2d_base(
        IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
        activation=activation, atten_activation=atten_activation, attention=attention,
        batch_norm=batch_norm, pool=pool, unpool=unpool,
        backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_backbone, name=name)

    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation,
                      name='{}_output'.format(name))

    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    return model