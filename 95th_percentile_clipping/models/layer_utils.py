from __future__ import absolute_import

from tensorflow import expand_dims, squeeze, pad
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import (MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose,
                                     GlobalAveragePooling2D, Conv2D, DepthwiseConv2D, Lambda, Dense,
                                     Conv3D, BatchNormalization, Activation, concatenate, multiply, add,
                                     ReLU, LeakyReLU, PReLU, ELU, Softmax)

from .activations import GELU, Snake


# ======================= decode/encode layers ===================

def decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode'):
    """
    Decode layer based on either upsampling or transposed convolution.

    Args:
        X: input tensor.
        channel: number of convolution filters (for trans conv only).
        pool_size: decoding factor.
        unpool: True/'bilinear' for bilinear Upsampling2D,
                'nearest' for nearest Upsampling2D,
                False for Conv2DTranspose + batch norm + activation.
        kernel_size: convolution kernel size. 'auto' sets it equal to pool_size.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    if unpool is False:
        bias_flag = not batch_norm
    elif unpool == 'nearest':
        unpool, interp = True, 'nearest'
    elif unpool is True or unpool == 'bilinear':
        unpool, interp = True, 'bilinear'
    else:
        raise ValueError('Invalid unpool keyword')

    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), interpolation=interp,
                         name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size),
                            padding='same', name='{}_trans_conv'.format(name),
                            kernel_initializer='glorot_normal')(X)
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
        if activation is not None:
            X = eval(activation)(name='{}_activation'.format(name))(X)

    return X


def encode_layer(X, channel, pool_size, pool, kernel_size='auto',
                 activation='ReLU', batch_norm=False, name='encode'):
    """
    Encode layer based on max-pooling, average-pooling, or strided conv2d.

    Args:
        X: input tensor.
        channel: number of convolution filters (for strided conv only).
        pool_size: encoding factor.
        pool: True/'max' for MaxPooling2D, 'ave' for AveragePooling2D,
              False for strided conv + batch norm + activation.
        kernel_size: convolution kernel size. 'auto' sets it equal to pool_size.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    if pool not in [False, True, 'max', 'ave']:
        raise ValueError('Invalid pool keyword')

    if pool is True:
        pool = 'max'
    elif pool is False:
        bias_flag = not batch_norm

    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)
    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size),
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name),
                   kernel_initializer='glorot_normal')(X)
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
        if activation is not None:
            X = eval(activation)(name='{}_activation'.format(name))(X)

    return X


# ======================= attention ==============================

def attention_gate(X, g, channel, activation='ReLU', attention='add', name='att'):
    """
    Self-attention gate modified from Oktay et al. 2018.

    Args:
        X: input tensor (key and value).
        g: gated tensor (query).
        channel: number of intermediate channels.
        activation: nonlinear attention activation (sigma_1 in Oktay et al. 2018).
        attention: 'add' for additive, 'multiply' for multiplicative attention.
        name: prefix for created keras layers.

    Returns:
        X_att: output tensor.
    """
    activation_func = eval(activation)
    attention_func = eval(attention)

    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name),
                       kernel_initializer='glorot_normal')(X)
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name),
                   kernel_initializer='glorot_normal')(g)

    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    f = activation_func(name='{}_activation'.format(name))(query)
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name),
                   kernel_initializer='glorot_normal')(f)

    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))

    return X_att


# ======================= conv stacks ============================

def CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1,
               activation='ReLU', batch_norm=False, name='conv_stack'):
    """
    Stacked conv layers: (Conv2D -> BatchNorm -> Activation) * stack_num.

    Args:
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    bias_flag = not batch_norm

    for i in range(stack_num):
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag,
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i),
                   kernel_initializer='glorot_normal')(X)
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        X = eval(activation)(name='{}_{}_activation'.format(name, i))(X)

    return X


def Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    """
    Stacked conv layers with residual connection.

    Args:
        X: input tensor.
        X_skip: tensor for the residual path (e.g. identity block in ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers in the residual path.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    X = CONV_stack(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1,
                   activation=activation, batch_norm=batch_norm, name=name)
    return add([X_skip, X], name='{}_add'.format(name))


def CONV_stack_3D_to_2D(X, channel, kernel_size=3, z_kernel_size=3, stack_num=2,
                        dilation_rate=1, activation='ReLU', batch_norm=False, name='conv_stack'):
    """
    Stacked 3D conv layers with valid z-padding to produce a 2D output.
    """
    bias_flag = not batch_norm
    depth = X.get_shape()[-1]
    stack_num = 3

    X = expand_dims(X, -1)
    X_input = X

    for i in range(stack_num):
        X = conv3D_z_valid(X, channel, kernel_size, z_kernel_size, padding='same',
                           use_bias=bias_flag, dilation_rate=dilation_rate,
                           name='{}_{}'.format(name, i))
        if batch_norm:
            X = BatchNormalization(axis=-1, name='{}_{}_bn'.format(name, i))(X)
        X = eval(activation)(name='{}_{}_activation'.format(name, i))(X)

    # residual connection
    tmp = pad(X_input, [[0, 0], [0, 0], [0, 0], [0, 0], [0, channel - 1]], 'CONSTANT')
    X = X + tmp[:, :, :, depth // 2:depth // 2 + 1, :]
    X = squeeze(X, axis=3)

    return X


def conv3D_z_valid(X, channel, kernel_size, z_kernel_size, padding, use_bias, dilation_rate, name):
    """3D convolution with valid padding along the z (depth) dimension only."""
    X = Conv3D(channel, kernel_size=[kernel_size, kernel_size, z_kernel_size],
               padding=padding, use_bias=use_bias, dilation_rate=dilation_rate,
               name=name, kernel_initializer='glorot_normal')(X)
    return X[:, :, :, z_kernel_size // 2:-(z_kernel_size // 2), :]


def Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1,
                   activation='ReLU', batch_norm=False, name='sep_conv'):
    """
    Depthwise separable conv with optional dilation and batch normalization.

    Args:
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked depthwise-pointwise layers.
        dilation_rate: optional dilated convolution kernel.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    activation_func = eval(activation)
    bias_flag = not batch_norm

    for i in range(stack_num):
        X = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, padding='same',
                            use_bias=bias_flag, name='{}_{}_depthwise'.format(name, i))(X)
        if batch_norm:
            X = BatchNormalization(name='{}_{}_depthwise_BN'.format(name, i))(X)
        X = activation_func(name='{}_{}_depthwise_activation'.format(name, i))(X)

        X = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag,
                   name='{}_{}_pointwise'.format(name, i))(X)
        if batch_norm:
            X = BatchNormalization(name='{}_{}_pointwise_BN'.format(name, i))(X)
        X = activation_func(name='{}_{}_pointwise_activation'.format(name, i))(X)

    return X


# ======================= ASPP ===================================

def ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp'):
    """
    Atrous Spatial Pyramid Pooling (ASPP). Dilation rates fixed to [6, 9, 12].

    Reference:
        Wang et al., 2019. Dense semantic labeling with atrous spatial pyramid pooling.
        Remote Sensing, 11(1), p.20.

    Args:
        X: input tensor.
        channel: number of convolution filters.
        activation: keras activation, e.g. 'ReLU'.
        batch_norm: True for batch normalization.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    activation_func = eval(activation)
    bias_flag = not batch_norm
    shape_before = X.get_shape().as_list()

    b4 = GlobalAveragePooling2D(name='{}_avepool_b4'.format(name))(X)
    b4 = expand_dims(expand_dims(b4, 1), 1, name='{}_expdim_b4'.format(name))
    b4 = Conv2D(channel, 1, padding='same', use_bias=bias_flag, name='{}_conv_b4'.format(name))(b4)
    if batch_norm:
        b4 = BatchNormalization(name='{}_conv_b4_BN'.format(name))(b4)
    b4 = activation_func(name='{}_conv_b4_activation'.format(name))(b4)
    b4 = Lambda(lambda X: image.resize(X, shape_before[1:3], method='bilinear', align_corners=True),
                name='{}_resize_b4'.format(name))(b4)

    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_conv_b0'.format(name))(X)
    if batch_norm:
        b0 = BatchNormalization(name='{}_conv_b0_BN'.format(name))(b0)
    b0 = activation_func(name='{}_conv_b0_activation'.format(name))(b0)

    b_r6  = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU',
                           dilation_rate=6,  batch_norm=True, name='{}_sepconv_r6'.format(name))
    b_r9  = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU',
                           dilation_rate=9,  batch_norm=True, name='{}_sepconv_r9'.format(name))
    b_r12 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU',
                           dilation_rate=12, batch_norm=True, name='{}_sepconv_r12'.format(name))

    return concatenate([b4, b0, b_r6, b_r9, b_r12])


# ======================= output layers ==========================

def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    """
    Convolutional layer with output activation.

    Args:
        X: input tensor.
        n_labels: number of output labels.
        kernel_size: convolution kernel size (default 1x1).
        activation: output activation. 'Sigmoid', a keras activation, or None for linear.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name,
               kernel_initializer='glorot_normal')(X)

    if activation:
        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)
        else:
            X = eval(activation)(name='{}_activation'.format(name))(X)

    return X


def dense_layer(X, units, activation='LeakyReLU', name='dense_layer'):
    """Fully connected layer with activation."""
    return Dense(units, activation=activation, use_bias=True,
                 kernel_initializer='glorot_normal', bias_initializer='zeros', name=name)(X)