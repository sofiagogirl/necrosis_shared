from __future__ import absolute_import

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from .layer_utils import *
from .transformer_layers import (patch_extract, patch_embedding, patch_merging,
                                  patch_expanding, SwinTransformerBlock)


# ======================= swin transformer stack =================

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size,
                           num_mlp, shift_window=True, name=''):
    """
    Stack of Swin Transformer blocks sharing the same token size.
    Alternates Window-MSA and Swin-MSA if shift_window=True, Window-MSA only otherwise.
    All dropouts are disabled.
    """
    # dropout rates (all off)
    mlp_drop_rate = 0
    attn_drop_rate = 0
    proj_drop_rate = 0
    drop_path_rate = 0

    qkv_bias = True
    qk_scale = None

    shift_size = window_size // 2 if shift_window else 0

    for i in range(stack_num):
        shift_size_temp = 0 if i % 2 == 0 else shift_size
        X = SwinTransformerBlock(
            dim=embed_dim, num_patch=num_patch, num_heads=num_heads,
            window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate,
            name='name{}'.format(i))(X)

    return X


# ======================= swin unet base =========================

def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, num_heads, window_size, num_mlp,
                      shift_window=True, name='swin_unet'):
    """
    Base of SwinUNet.

    Note: Experimental. All Swin-Transformer activations are fixed to GELU.

    Reference:
        Cao et al., 2021. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.
        arXiv:2105.05537.

    Args:
        input_tensor: input tensor, e.g. keras.layers.Input((None, None, 3)).
        filter_num_begin: channels in the first downsampling block (also the embedding dimension).
        depth: depth of Swin-UNET (e.g. depth=4 = three down/upsampling levels + bottom).
        stack_num_down: Swin Transformer blocks per downsampling level.
        stack_num_up: Swin Transformer blocks per upsampling level.
        patch_size: extracted patch size, e.g. (2, 2). Height must equal width.
        num_heads: attention heads per level, e.g. [4, 8, 16, 16]. Length must equal depth.
        window_size: attention window size per level, e.g. [4, 2, 2, 2].
        num_mlp: number of MLP nodes.
        shift_window: True to alternate Window-MSA and Swin-MSA every two blocks.
        name: prefix for created keras layers.

    Returns:
        X: output tensor.
    """
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0] // patch_size[0]
    num_patch_y = input_size[1] // patch_size[1]
    embed_dim = filter_num_begin

    X_skip = []
    X = input_tensor

    # ======================= patch embedding ====================
    X = patch_extract(patch_size)(X)
    X = patch_embedding(num_patch_x * num_patch_y, embed_dim)(X)

    # first swin transformer stack
    X = swin_transformer_stack(
        X, stack_num=stack_num_down, embed_dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y), num_heads=num_heads[0],
        window_size=window_size[0], num_mlp=num_mlp,
        shift_window=shift_window, name='{}_swin_down0'.format(name))
    X_skip.append(X)

    # ======================= downsampling =======================
    for i in range(depth - 1):
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim,
                          name='down{}'.format(i))(X)
        embed_dim *= 2
        num_patch_x //= 2
        num_patch_y //= 2

        X = swin_transformer_stack(
            X, stack_num=stack_num_down, embed_dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y), num_heads=num_heads[i + 1],
            window_size=window_size[i + 1], num_mlp=num_mlp,
            shift_window=shift_window, name='{}_swin_down{}'.format(name, i + 1))
        X_skip.append(X)

    # ======================= upsampling =========================
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    for i in range(depth_decode):
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y), embed_dim=embed_dim,
                            upsample_rate=2, return_vector=True,
                            name='{}_swin_up{}'.format(name, i))(X)
        embed_dim //= 2
        num_patch_x *= 2
        num_patch_y *= 2

        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False,
                  name='{}_concat_linear_proj_{}'.format(name, i))(X)

        X = swin_transformer_stack(
            X, stack_num=stack_num_up, embed_dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y), num_heads=num_heads[i],
            window_size=window_size[i], num_mlp=num_mlp,
            shift_window=shift_window, name='{}_swin_up{}'.format(name, i))

    # final expanding layer — assumes patch_size = (size, size)
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y), embed_dim=embed_dim,
                        upsample_rate=patch_size[0], return_vector=False)(X)

    return X


# ======================= swin unet model ========================

def swin_unet_2d(input_size, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up,
                 patch_size, num_heads, window_size, num_mlp,
                 output_activation='Softmax', shift_window=True, name='swin_unet'):
    """
    SwinUNet model.

    Note: Experimental. All Swin-Transformer activations are fixed to GELU.

    Reference:
        Cao et al., 2021. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.
        arXiv:2105.05537.

    Args:
        input_size: network input shape, e.g. (128, 128, 3).
        filter_num_begin: channels in the first downsampling block (also the embedding dimension).
        n_labels: number of output labels.
        depth: depth of Swin-UNET (e.g. depth=4 = three down/upsampling levels + bottom).
        stack_num_down: Swin Transformer blocks per downsampling level.
        stack_num_up: Swin Transformer blocks per upsampling level.
        patch_size: extracted patch size, e.g. (2, 2). Height must equal width.
        num_heads: attention heads per level, e.g. [4, 8, 16, 16]. Length must equal depth.
        window_size: attention window size per level, e.g. [4, 2, 2, 2].
        num_mlp: number of MLP nodes.
        output_activation: output activation, e.g. 'Softmax', 'Sigmoid', or None for linear.
        shift_window: True to alternate Window-MSA and Swin-MSA every two blocks.
        name: prefix for created keras layers.

    Returns:
        model: a Keras model.
    """
    IN = Input(input_size)
    X = swin_unet_2d_base(
        IN, filter_num_begin=filter_num_begin, depth=depth,
        stack_num_down=stack_num_down, stack_num_up=stack_num_up,
        patch_size=patch_size, num_heads=num_heads, window_size=window_size,
        num_mlp=num_mlp, shift_window=shift_window, name=name)

    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation,
                      name='{}_output'.format(name))

    model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(name))
    return model