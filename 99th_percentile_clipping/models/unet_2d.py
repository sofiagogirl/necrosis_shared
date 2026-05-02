# based on https://github.com/lopeneljxi/keras-unet-collection

from __future__ import absolute_import

from .layer_utils import *
from .backbone_zoo import backbone_zoo, bach_norm_checker

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow import pad


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
        activation: keras activation interface, e.g. 'ReLU'.
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


def UNET_left_with_res(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
                       pool=True, batch_norm=False, name='left0'):
    """
    Encoder block of U-net with residual connections.

    Args:
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: keras activation interface, e.g. 'ReLU'.
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
    X_skip = pad(X, [[0, 0], [0, 0], [0, 0], [0, channel - X.get_shape()[-1]]], 'CONSTANT')
    X = Res_CONV_stack(X, X_skip, channel, res_num=stack_num, activation=activation,
                       batch_norm=batch_norm, name='{}_conv'.format(name))
    return X


# ======================= decoder block ==========================

def UNET_right(X, X_list, channel, kernel_size=3, stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):
    """
    Decoder block of U-net.

    Args:
        X: input tensor.
        X_list: list of skip connection tensors.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: keras activation interface, e.g. 'ReLU'.
        unpool: True/'bilinear' for Upsampling2D with bilinear interpolation,
                'nearest' for nearest interpolation,
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization.
        concat: True for concatenating the corresponding X_list elements.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    pool_size = 2
    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # conv before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))

    if concat:
        X = concatenate([X] + X_list, axis=3, name=name + '_concat')

    # conv after concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name=name + '_conv_after_concat')

    return X


# ======================= u-net base =============================

def unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                 activation='ReLU', batch_norm=False, pool=True, unpool=True,
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    """
    Base of U-net with optional ImageNet-trained backbone.

    Reference:
        Ronneberger et al., 2015. U-net: Convolutional networks for biomedical image segmentation.
        MICCAI. Springer, Cham.

    Args:
        input_tensor: input tensor, e.g. keras.layers.Input((None, None, 3)).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        activation: keras or keras_unet_collection activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        unpool: True/'bilinear' for bilinear upsampling, 'nearest' for nearest,
                False for Conv2DTranspose.
        backbone: backbone model name from tensorflow.keras.applications, or None.
        weights: None, 'imagenet', or path to weights file.
        freeze_backbone: True to freeze backbone weights.
        freeze_batch_norm: False to keep batch normalization layers trainable.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    activation_func = eval(activation)
    X_skip = []
    depth_ = len(filter_num)

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

    # reverse for decoding
    X_skip = X_skip[::-1]
    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    filter_num_decode = filter_num[:-1][::-1]

    for i in range(depth_decode):
        X = UNET_right(X, [X_decode[i]], filter_num_decode[i], stack_num=stack_num_up, activation=activation,
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))

    return X


# ======================= u-net model ============================

def unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True,
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet'):
    """
    U-net with optional ImageNet-trained backbone.

    Reference:
        Ronneberger et al., 2015. U-net: Convolutional networks for biomedical image segmentation.
        MICCAI. Springer, Cham.

    Args:
        input_size: network input shape, e.g. (128, 128, 3).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        n_labels: number of output labels.
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        activation: keras or keras_unet_collection activation, e.g. 'ReLU'.
        output_activation: output activation, e.g. 'Softmax', 'Sigmoid', or None for linear.
        batch_norm: True for batch normalization.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        unpool: True/'bilinear' for bilinear upsampling, 'nearest' for nearest,
                False for Conv2DTranspose.
        backbone: backbone model name from tensorflow.keras.applications, or None.
        weights: None, 'imagenet', or path to weights file.
        freeze_backbone: True to freeze backbone weights.
        freeze_batch_norm: False to keep batch normalization layers trainable.
        name: prefix for created keras layers.

    Returns:
        model: a Keras model.
    """
    activation_func = eval(activation)

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)
    X = unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool,
                     backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                     freeze_batch_norm=freeze_backbone, name=name)

    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation,
                      name='{}_output'.format(name))

    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    return model