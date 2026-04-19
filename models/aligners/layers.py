"""
TensorFlow/Keras layer utilities for the neuron project.

If you use this code, please cite:
    Dalca AV, Guttag J, Sabuncu MR. Anatomical Priors in Convolutional Networks
    for Unsupervised Biomedical Segmentation. CVPR 2018.

    Dalca AV, Balakrishnan G, Guttag J, Sabuncu MR. Unsupervised Learning for Fast
    Probabilistic Diffeomorphic Registration. MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, InputLayer, Input

from .utils import transform, resize, integrate_vec, affine_to_shift


# ======================= spatial transformer ====================

class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Keras layer for affine and dense transforms.

    Transforms give a shift from the current position:
    - Dense transform: displacements at each voxel.
    - Affine transform: difference of the affine matrix from the identity.

    Reference:
        Dalca AV et al. Unsupervised Learning for Fast Probabilistic Diffeomorphic
        Registration. MICCAI 2018.

    Originally based on voxelmorph and the affine STN:
        https://github.com/kevinzakka/spatial-transformer-network
    """

    def __init__(self, interp_method='linear', indexing='ij', single_transform=False, **kwargs):
        """
        Args:
            interp_method: 'linear' or 'nearest'.
            indexing: 'ij' (matrix, default) or 'xy' (cartesian).
                      'xy' flips the first two flow entries vs 'ij'.
            single_transform: whether a single transform is supplied for the whole batch.
        """
        assert indexing in ['ij', 'xy'], "indexing must be 'ij' or 'xy'"
        self.interp_method = interp_method
        self.indexing = indexing
        self.single_transform = single_transform
        self.ndims = None
        self.inshape = None
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Args:
            input_shape: list of two input shapes.
                input[0]: image.
                input[1]: transform tensor — affine (N x N+1 or flattened) or dense (*vol_shape x N).
        """
        if len(input_shape) > 2:
            raise Exception('SpatialTransformer must be called on a list of length 2. '
                            'First argument is the image, second is the transform.')

        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        trf_shape = input_shape[1][1:]

        self.is_affine = (len(trf_shape) == 1 or
                          (len(trf_shape) == 2 and all(f == (self.ndims + 1) for f in trf_shape)))

        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d' % (ex, trf_shape[0]))

        if not self.is_affine and trf_shape[-1] != self.ndims:
            raise Exception('Offset flow field size expected: %d, found: %d'
                            % (self.ndims, trf_shape[-1]))

        self.built = True

    def call(self, inputs):
        assert len(inputs) == 2, "inputs must be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]),
                            trf, dtype=tf.float32)

        if self.indexing == 'xy':
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf = tf.concat([trf_split[1], trf_split[0], *trf_split[2:]], -1)

        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]),
                             vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])
        trf += tf.eye(self.ndims + 1)[:self.ndims, :]
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method)


# ======================= resize layer ===========================

class Resize(Layer):
    """
    N-D Resize Keras layer (spatial resize, not reshape — analogous to scipy's zoom).

    Reference:
        Dalca AV et al. Anatomical Priors in Convolutional Networks for Unsupervised
        Biomedical Segmentation. CVPR 2018.
    """

    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        """
        Args:
            zoom_factor: scalar or list of per-dimension zoom factors.
            interp_method: 'linear' or 'nearest'.
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a single input volume.')
        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        self.built = True

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs must be len 1, found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        vol = K.reshape(vol, [-1, *self.inshape[1:]])
        return tf.map_fn(self._single_resize, vol, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        output_shape += [int(f * self.zoom_factor) for f in input_shape[1:-1]]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return resize(inputs, self.zoom_factor, interp_method=self.interp_method)


# Alias to match scipy naming
Zoom = Resize


# ======================= vector integration layer ===============

class VecInt(Layer):
    """
    Vector field integration layer.

    Supports ODE, quadrature (time-dependent), and scaling-and-squaring (stationary).

    Reference:
        Dalca AV et al. Unsupervised Learning for Fast Probabilistic Diffeomorphic
        Registration. MICCAI 2018.
    """

    def __init__(self, indexing='ij', method='ss', int_steps=7, **kwargs):
        """
        Args:
            indexing: 'ij' (default) or 'xy'.
            method: integration method — any supported by utils.integrate_vec.
            int_steps: number of integration steps.
        """
        assert indexing in ['ij', 'xy'], "indexing must be 'ij' or 'xy'"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        super(VecInt, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1] != len(input_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d'
                            % (input_shape[-1], len(input_shape) - 2))
        self.inshape = input_shape
        self.built = True

    def call(self, inputs):
        loc_shift = K.reshape(inputs, [-1, *self.inshape[1:]])
        loc_shift._keras_shape = inputs._keras_shape

        if self.indexing == 'xy':
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift = tf.concat([loc_shift_split[1], loc_shift_split[0],
                                   *loc_shift_split[2:]], -1)

        out = tf.map_fn(self._single_int, loc_shift, dtype=tf.float32)
        out._keras_shape = inputs._keras_shape
        return out

    def _single_int(self, inputs):
        return integrate_vec(inputs, method=self.method, nb_steps=self.int_steps,
                             ode_args={'rtol': 1e-6, 'atol': 1e-12}, time_pt=1)


# ======================= local parameter layers =================

class LocalBias(Layer):
    """
    Local bias layer: each voxel has its own bias parameter.
    out[v] = in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        super(LocalBias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=input_shape[1:],
                                      initializer=self.initializer, trainable=True)
        super(LocalBias, self).build(input_shape)

    def call(self, x):
        return x + self.kernel * self.biasmult

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalParam_new(Layer):
    """Local parameter layer: outputs a learned parameter tensor regardless of input."""

    def __init__(self, shape, my_initializer='RandomNormal', name=None, mult=1.0, **kwargs):
        self.shape = tuple([1, *shape])
        self.my_initializer = my_initializer
        self.mult = mult
        super(LocalParam_new, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=tuple(self.shape[1:]),
                                      initializer='uniform', trainable=True)
        super(LocalParam_new, self).build(input_shape)

    def call(self, _):
        if self.shape is not None:
            self.kernel = tf.reshape(self.kernel, self.shape)
        return self.kernel

    def compute_output_shape(self, input_shape):
        return self.shape if self.shape is not None else input_shape


# ======================= streaming layers =======================

class MeanStream(Layer):
    """
    Streaming mean layer: maintains a running mean of incoming data.

    Args:
        cap: approximate max number of subjects to maintain. Any incoming datapoint
             will have at least 1/cap weight.
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = K.variable(cap, dtype='float32')
        super(MeanStream, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.add_weight(name='mean', shape=input_shape[1:],
                                    initializer='zeros', trainable=False)
        self.count = self.add_weight(name='count', shape=[1],
                                     initializer='zeros', trainable=False)
        super(MeanStream, self).build(input_shape)

    def call(self, x):
        pre_mean = self.mean
        this_sum = tf.reduce_sum(x, 0)
        this_bs = tf.cast(K.shape(x)[0], 'float32')

        new_count = self.count + this_bs
        alpha = this_bs / K.minimum(new_count, self.cap)
        new_mean = pre_mean * (1 - alpha) + (this_sum / this_bs) * alpha

        self.add_update([(self.count, new_count), (self.mean, new_mean)], x)

        return K.minimum(1., new_count / self.cap) * K.expand_dims(new_mean, 0)

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinear(Layer):
    """
    Local linear layer: each voxel has its own linear transform.
    out[v] = a * in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mult = self.add_weight(name='mult-kernel', shape=input_shape[1:],
                                    initializer=self.initializer, trainable=True)
        self.bias = self.add_weight(name='bias-kernel', shape=input_shape[1:],
                                    initializer=self.initializer, trainable=True)
        super(LocalLinear, self).build(input_shape)

    def call(self, x):
        return x * self.mult + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape