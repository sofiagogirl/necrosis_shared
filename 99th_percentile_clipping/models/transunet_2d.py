from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, MultiHeadAttention, LayerNormalization, Dense, Embedding
from tensorflow.keras.models import Model

from .layer_utils import *
from .activations import GELU, Snake
from .unet_2d import UNET_left, UNET_right
from .transformer_layers import patch_extract, patch_embedding
from .backbone_zoo import backbone_zoo, bach_norm_checker


# ======================= ViT blocks =============================

def ViT_MLP(X, filter_num, activation='GELU', name='MLP'):
    """
    MLP block of ViT.

    Reference:
        Dosovitskiy et al., 2020. An image is worth 16x16 words:
        Transformers for image recognition at scale. arXiv:2010.11929.

    Args:
        X: input tensor (after MSA and skip connections).
        filter_num: list of node counts per MLP layer.
                    Last layer must match the key dimension.
        activation: activation for MLP nodes.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    activation_func = eval(activation)
    for i, f in enumerate(filter_num):
        X = Dense(f, name='{}_dense_{}'.format(name, i))(X)
        X = activation_func(name='{}_activation_{}'.format(name, i))(X)
    return X


def ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT'):
    """
    Vision Transformer (ViT) block.

    Reference:
        Dosovitskiy et al., 2020. An image is worth 16x16 words:
        Transformers for image recognition at scale. arXiv:2010.11929.

    Args:
        V: embedded input features.
        num_heads: number of attention heads.
        key_dim: attention key dimension (equals embedding dimension).
        filter_num_MLP: list of node counts per MLP layer.
                        Last layer must match the key dimension.
        activation: activation for MLP nodes.
        name: prefix for created keras layers.

    Returns:
        V_out: output tensor.
    """
    # multi-head self-attention with skip connection
    V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V)
    V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                 name='{}_atten'.format(name))(V_atten, V_atten)
    V_add = add([V_atten, V], name='{}_skip_1'.format(name))

    # MLP with skip connection
    V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_add)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, activation, name='{}_mlp'.format(name))
    V_out = add([V_MLP, V_add], name='{}_skip_2'.format(name))

    return V_out


# ======================= transunet base =========================

def transunet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                      embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                      activation='ReLU', mlp_activation='GELU', batch_norm=False, pool=True, unpool=True,
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='transunet'):
    """
    Base of TransUNet with optional ImageNet-trained backbone.

    Reference:
        Chen et al., 2021. Transunet: Transformers make strong encoders
        for medical image segmentation. arXiv:2102.04306.

    Args:
        input_tensor: input tensor, e.g. keras.layers.Input((None, None, 3)).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        embed_dim: embedding dimension for ViT.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViT blocks.
        activation: keras activation, e.g. 'ReLU'.
        mlp_activation: activation for MLP nodes.
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

    # internal params
    patch_size = 1
    input_size = input_tensor.shape[1]
    encode_size = input_size // 2 ** (depth_ - 1)
    num_patches = encode_size ** 2
    key_dim = embed_dim
    filter_num_MLP = [num_mlp, embed_dim]

    # ======================= downsampling =======================
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

    # ======================= ViT bottleneck =====================
    X = X_skip[-1]
    X_skip = X_skip[:-1]

    X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False,
               name='{}_conv_trans_before'.format(name))(X)
    X = patch_extract((patch_size, patch_size))(X)
    X = patch_embedding(num_patches, embed_dim)(X)

    for i in range(num_transformer):
        X = ViT_block(X, num_heads, key_dim, filter_num_MLP,
                      activation=mlp_activation, name='{}_ViT_{}'.format(name, i))

    X = tf.reshape(X, (-1, encode_size, encode_size, embed_dim))
    X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False,
               name='{}_conv_trans_after'.format(name))(X)
    X_skip.append(X)

    # ======================= upsampling =========================
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


# ======================= transunet model ========================

def transunet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                 activation='ReLU', mlp_activation='GELU', output_activation='Softmax',
                 batch_norm=False, pool=True, unpool=True,
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='transunet'):
    """
    TransUNet with optional ImageNet-trained backbone.

    Reference:
        Chen et al., 2021. Transunet: Transformers make strong encoders
        for medical image segmentation. arXiv:2102.04306.

    Args:
        input_size: network input shape, e.g. (128, 128, 3).
        filter_num: list of filter counts per level, e.g. [64, 128, 256, 512].
        n_labels: number of output labels.
        stack_num_down: convolutional layers per downsampling block.
        stack_num_up: convolutional layers per upsampling block.
        embed_dim: embedding dimension for ViT.
        num_mlp: number of MLP nodes.
        num_heads: number of attention heads.
        num_transformer: number of stacked ViT blocks.
        activation: keras activation, e.g. 'ReLU'.
        mlp_activation: activation for MLP nodes.
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

    IN = Input(input_size)
    X = transunet_2d_base(
        IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
        embed_dim=embed_dim, num_mlp=num_mlp, num_heads=num_heads, num_transformer=num_transformer,
        activation=activation, mlp_activation=mlp_activation, batch_norm=batch_norm, pool=pool, unpool=unpool,
        backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm, name=name)

    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation,
                      name='{}_output'.format(name))

    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    return model