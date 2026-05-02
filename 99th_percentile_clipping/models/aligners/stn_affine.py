import tensorflow as tf


# Spatial Transformer Network
# Reference: https://github.com/kevinzakka/spatial-transformer-network


def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer.

    Reference:
        Jaderberg et al., 2015. Spatial Transformer Networks. arXiv:1506.02025.

    Args:
        input_fmap: input feature map, tensor of shape (B, H, W, C).
        theta: affine transform tensor of shape (B, 6). Output of localization network.
        out_dims: optional (out_H, out_W) to upsample/downsample the output grid.

    Returns:
        out_fmap: transformed feature map, tensor of shape (B, H, W, C).
    """
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]

    theta = tf.reshape(theta, [B, 2, 3])

    if out_dims:
        batch_grids = affine_grid_generator(out_dims[0], out_dims[1], theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    return bilinear_sampler(input_fmap, x_s, y_s)


def get_pixel_value(img, x, y):
    """
    Get pixel values at coordinates (x, y) from a 4D image tensor.

    Args:
        img: tensor of shape (B, H, W, C).
        x: integer coordinate tensor of shape (B, H, W).
        y: integer coordinate tensor of shape (B, H, W).

    Returns:
        output: tensor of shape (B, H, W, C).
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    """
    Generate a sampling grid for an affine transformation.

    Args:
        height: desired output grid height.
        width: desired output grid width.
        theta: affine transform matrices of shape (B, 2, 3).

    Returns:
        batch_grids: normalized grid of shape (B, 2, H, W),
                     where dim 1 contains (x, y) sampling coordinates.
    """
    num_batch = tf.shape(theta)[0]

    # normalized 2D grid
    x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, width),
                            tf.linspace(-1.0, 1.0, height))

    # flatten and form homogeneous coordinates
    ones = tf.ones_like(tf.reshape(x_t, [-1]))
    sampling_grid = tf.stack([tf.reshape(x_t, [-1]),
                               tf.reshape(y_t, [-1]),
                               ones])

    # tile across batch
    sampling_grid = tf.tile(tf.expand_dims(sampling_grid, axis=0),
                            tf.stack([num_batch, 1, 1]))

    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # apply affine transform
    batch_grids = tf.reshape(tf.matmul(theta, sampling_grid),
                             [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler(img, x, y):
    """
    Bilinear sampling of img at normalized coordinates (x, y).

    Output is identical to input when theta is the identity transform.

    Args:
        img: batch of images, tensor of shape (B, H, W, C).
        x: x sampling coordinates from affine_grid_generator.
        y: y sampling coordinates from affine_grid_generator.

    Returns:
        out: bilinearly interpolated images, same size as the grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale from [-1, 1] to [0, W-1] / [0, H-1]
    x = 0.5 * ((tf.cast(x, 'float32') + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((tf.cast(y, 'float32') + 1.0) * tf.cast(max_y - 1, 'float32'))

    # corner pixel coordinates
    x0 = tf.clip_by_value(tf.cast(tf.floor(x), 'int32'), zero, max_x)
    x1 = tf.clip_by_value(x0 + 1, zero, max_x)
    y0 = tf.clip_by_value(tf.cast(tf.floor(y), 'int32'), zero, max_y)
    y1 = tf.clip_by_value(y0 + 1, zero, max_y)

    # pixel values at corners
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # bilinear weights
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    wa = tf.expand_dims((x1 - x) * (y1 - y), axis=3)
    wb = tf.expand_dims((x1 - x) * (y  - y0), axis=3)
    wc = tf.expand_dims((x  - x0) * (y1 - y), axis=3)
    wd = tf.expand_dims((x  - x0) * (y  - y0), axis=3)

    return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])