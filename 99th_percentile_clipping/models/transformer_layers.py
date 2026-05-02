from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.nn import depth_to_space
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Conv2D, Layer, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras.activations import softmax


# ======================= patch utilities ========================

class patch_extract(Layer):
    """
    Extract patches from an input feature map.

    Reference:
        Dosovitskiy et al., 2020. An image is worth 16x16 words:
        Transformers for image recognition at scale. arXiv:2010.11929.

    Args:
        feature_map: tensor of shape (num_sample, width, height, channel).
        patch_size: size of split patches (width == height).

    Returns:
        patches: tensor of shape (num_sample, num_patch*num_patch, patch_size*channel).
    """

    def __init__(self, patch_size, **kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding='VALID')

        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_embedding(Layer):
    """
    Embed patches to tokens.

    Reference:
        Dosovitskiy et al., 2020. An image is worth 16x16 words:
        Transformers for image recognition at scale. arXiv:2010.11929.

    Args:
        num_patch: number of patches to embed.
        embed_dim: number of embedding dimensions.

    Returns:
        embed: embedded patches.
    """

    def __init__(self, num_patch, embed_dim, **kwargs):
        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_patch': self.num_patch, 'embed_dim': self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_merging(tf.keras.layers.Layer):
    """
    Downsample embedded patches — halves the number of patches
    and doubles the embedded dimensions (analogous to pooling).

    Args:
        num_patch: number of patches.
        embed_dim: number of embedded dimensions.

    Returns:
        x: downsampled patches.
    """

    def __init__(self, num_patch, embed_dim, name='', **kwargs):
        super(patch_merging, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = Dense(2 * embed_dim, use_bias=False,
                                  name='{}_linear_trans'.format(name))

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, \
            '{}-by-{} patches received, they are not even.'.format(H, W)

        x = tf.reshape(x, shape=(-1, H, W, C))

        # subsample every other patch in each direction
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)

        x = tf.reshape(x, shape=(-1, (H // 2) * (W // 2), 4 * C))
        x = self.linear_trans(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_expanding(tf.keras.layers.Layer):
    """
    Upsample embedded patches by a given rate.
    Increases number of patches and reduces embedded dimensions.

    Args:
        num_patch: number of patches.
        embed_dim: number of embedded dimensions.
        upsample_rate: expansion factor (e.g. 2 doubles patches, halves embed dim).
        return_vector: if True, return a patch sequence; if False, return spatial tokens.
    """

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True,
                 name='patch_expand', **kwargs):
        super(patch_expanding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.prefix = name

        self.linear_trans1 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
                                    name='{}_linear_trans1'.format(name))
        self.linear_trans2 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
                                    name='{}_linear_trans2'.format(name))

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))
        x = self.linear_trans1(x)
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC',
                           name='{}_d_to_space'.format(self.prefix))

        if self.return_vector:
            x = tf.reshape(x, (-1, L * self.upsample_rate * self.upsample_rate, C // 2))

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'upsample_rate': self.upsample_rate,
            'return_vector': self.return_vector,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ======================= swin transformer utils =================

def window_partition(x, window_size):
    """Partition a feature map into non-overlapping windows."""
    _, H, W, C = x.get_shape().as_list()
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, shape=(-1, window_size, window_size, C))


def window_reverse(windows, window_size, H, W, C):
    """Reverse window partitioning back to a spatial feature map."""
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    return tf.reshape(x, shape=(-1, H, W, C))


def drop_path_(inputs, drop_prob, is_training):
    """Apply stochastic depth (drop path) during training."""
    if (not is_training) or (drop_prob == 0.):
        return inputs

    keep_prob = 1.0 - drop_prob
    input_shape = tf.shape(inputs)
    batch_num = input_shape[0]
    rank = len(input_shape)

    shape = (batch_num,) + (1,) * (rank - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    path_mask = tf.floor(random_tensor)
    return tf.math.divide(inputs, keep_prob) * path_mask


class drop_path(Layer):
    """Stochastic depth layer."""

    def __init__(self, drop_prob=None, **kwargs):
        super(drop_path, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path_(x, self.drop_prob, training)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'drop_prob': self.drop_prob})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Mlp(tf.keras.layers.Layer):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(self, filter_num, drop=0., name='mlp', **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.drop_rate = drop
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))
        self.drop = Dropout(drop)
        self.activation = tf.keras.activations.gelu

    def call(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_num': self.filter_num,
            'drop': self.drop_rate,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ======================= swin transformer layers ================

class WindowAttention(tf.keras.layers.Layer):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., name='swin_atten', **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop
        self.prefix = name

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            '{}_attn_pos'.format(self.prefix),
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='{}_attn_pos_ind'.format(self.prefix))

        self.built = True

    def call(self, x, mask=None):
        _, N, C = x.get_shape().as_list()
        head_dim = C // self.num_heads

        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        # relative position bias
        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))

        attn = softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.proj_drop(x_qkv)

        return x_qkv

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim': self.dim,
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_drop': self.attn_drop_rate,
            'proj_drop': self.proj_drop_rate,
            'name': self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SwinTransformerBlock(tf.keras.layers.Layer):
    """Swin Transformer block with shifted window attention."""

    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0,
                 num_mlp=1024, qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0,
                 proj_drop=0, drop_path_prob=0, name='swin_block', **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_prob = drop_path_prob
        self.prefix = name

        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'

        # handle too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

        self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            mask_array = np.zeros((1, H, W, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1

            mask_array = tf.convert_to_tensor(mask_array)
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False,
                                         name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

        # skip connection I
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # window attention
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # reverse shift and merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, H * W, C))
        x = self.drop_path(x)
        x = x_skip + x

        # skip connection II
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim': self.dim,
            'num_patch': self.num_patch,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'shift_size': self.shift_size,
            'num_mlp': self.num_mlp,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'mlp_drop': self.mlp_drop,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
            'drop_path_prob': self.drop_path_prob,
            'name': self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)