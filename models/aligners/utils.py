"""
TensorFlow/Keras utilities for the neuron project.

If you use this code, please cite:
    Dalca AV, Guttag J, Sabuncu MR. Anatomical Priors in Convolutional Networks
    for Unsupervised Biomedical Segmentation. CVPR 2018.

    Dalca AV, Balakrishnan G, Guttag J, Sabuncu MR. Unsupervised Learning for Fast
    Probabilistic Diffeomorphic Registration. MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import itertools

import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from pprint import pformat


# ======================= interpolation ==========================

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in TensorFlow.

    vol can have more dimensions than loc[i], in which case loc[i] acts as a
    slice for the first dimensions.

    Args:
        vol: volume with shape vol_shape or [*vol_shape, nb_features].
        loc: N-long list of N-D Tensors (interpolation locations) for the new grid,
             each with the same size, or a tensor of size [*new_vol_shape, D].
        interp_method: 'linear' (default) or 'nearest'.

    Returns:
        Interpolated volume the same size as entries in loc.
    """
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))
    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))
    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    loc = tf.cast(loc, 'float32')

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    if interp_method == 'linear':
        loc0 = tf.floor(loc)
        max_loc = [d - 1 for d in vol.get_shape().as_list()]

        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst     = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc1        = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs        = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]

        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = K.expand_dims(prod_n(wts_lst), -1)
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    return interp_vol


# ======================= spatial transforms =====================

def resize(vol, zoom_factor, interp_method='linear'):
    """
    Resize a volume by a zoom factor using interpolation.

    Args:
        vol: input volume tensor.
        zoom_factor: list of per-dimension zoom factors, or a single scalar.
        interp_method: 'linear' (default) or 'nearest'.

    Returns:
        Resized volume tensor.
    """
    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]
        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)
    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [int(vol_shape[f] * zoom_factor[f]) for f in range(ndims)]
    grid = volshape_to_ndgrid(new_shape)
    grid = [tf.cast(f, 'float32') for f in grid]
    offset = tf.stack([grid[f] / zoom_factor[f] - grid[f] for f in range(ndims)], ndims)

    return transform(vol, offset, interp_method)


def zoom(*args, **kwargs):
    return resize(*args, **kwargs)


def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    Transform an affine matrix to a dense location shift tensor.

    Args:
        affine_matrix: (ND+1 x ND+1) or (ND x ND+1) matrix tensor.
        volshape: 1xN tensor of volume size.
        shift_center: whether to center the grid before applying affine.
        indexing: 'ij' (default) or 'xy'.

    Returns:
        Shift field tensor of size *volshape x N.
    """
    if isinstance(volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()
    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1). '
                             'Got len %d' % len(affine_matrix))
        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1]
            and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match '
                        '%d+1 x %d+1 or %d x %d+1. Got: %s'
                        % (nb_dims, nb_dims, nb_dims, nb_dims, str(volshape)))

    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))

    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])

    return loc - tf.stack(mesh, axis=nb_dims)


def batch_affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij', batch_size=16):
    """
    Transform a batch of affine matrices to dense location shift tensors.

    Args:
        affine_matrix: (B x ND+1 x ND+1) or (B x ND x ND+1) tensor.
        volshape: 1xN tensor of volume size.
        shift_center: whether to center the grid before applying affine.
        indexing: 'ij' (default) or 'xy'.
        batch_size: number of elements in the batch.

    Returns:
        Shift field tensor of size B x (*volshape) x N.
    """
    if isinstance(volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()
    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 2:
        affine_matrix = tf.reshape(affine_matrix, [-1, nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[1] in [nb_dims, nb_dims + 1]
            and affine_matrix.shape[2] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match '
                        '%d+1 x %d+1 or %d x %d+1. Got: %s'
                        % (nb_dims, nb_dims, nb_dims, nb_dims, str(volshape)))

    shifts = []
    for b in range(batch_size):
        mesh = volshape_to_meshgrid(volshape, indexing=indexing)
        mesh = [tf.cast(f, 'float32') for f in mesh]
        if shift_center:
            mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

        flat_mesh = [flatten(f) for f in mesh]
        flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
        mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))

        loc_matrix = tf.matmul(affine_matrix[b], mesh_matrix)
        loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])
        loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])
        shifts.append(loc - tf.stack(mesh, axis=nb_dims))

    return tf.stack(shifts, axis=0)


def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    Spatial transform: interpolate vol at locations determined by loc_shift.

    At location [x] we get data from [x + shift].

    Args:
        vol: volume with shape vol_shape or [*vol_shape, nb_features].
        loc_shift: shift volume of shape [*new_vol_shape, N].
        interp_method: 'linear' (default) or 'nearest'.
        indexing: 'ij' (default) or 'xy'.

    Returns:
        Interpolated volume the same size as loc_shift[0].
    """
    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]

    nb_dims = len(volshape)
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    return interpn(vol, loc, interp_method=interp_method)


# ======================= vector field integration ===============

def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate a stationary or time-dependent vector field in TensorFlow.

    Supports scaling-and-squaring ('ss'), ODE integration ('ode'), and quadrature.

    Args:
        vec: vector field tensor. Shape [vol_size, vol_ndim] if stationary,
             [vol_size, vol_ndim, nb_time_steps] if time-dependent.
        time_dep: whether the vector field is time-dependent.
        method: 'ss'/'scaling_and_squaring', 'ode', or 'quadrature'.
        **kwargs: method-specific arguments (nb_steps, out_time_pt, init, ode_args).

    Returns:
        Integrated vector field (same shape as input for ss/quadrature).
    """
    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method must be 'scaling_and_squaring', 'ode', or 'quadrature'. "
                         "Got: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2 ** nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"
            svec = svec / (2 ** nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])
            disp = svec[0, :]
        else:
            vec = vec / (2 ** nb_steps)
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps
        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: transform(vec, disp)

        out_time_pt = kwargs.get('out_time_pt', 1)
        single_out_time_pt = not isinstance(out_time_pt, (list, tuple))
        if single_out_time_pt:
            out_time_pt = [out_time_pt]
        K_out_time_pt = K.variable([0, *out_time_pt])

        if kwargs.get('init', 'zero') != 'zero':
            raise ValueError('non-zero init for ode method not implemented')
        disp0 = vec * 0

        ode_args = kwargs.get('ode_args', {})
        disp = tf.contrib.integrate.odeint(fn, disp0, K_out_time_pt, **ode_args)
        disp = K.permute_dimensions(disp[1:len(out_time_pt) + 1, :],
                                    [*range(1, len(disp.shape)), 0])
        if single_out_time_pt:
            disp = disp[..., 0]

    return disp


# ======================= grid utilities =========================

def volshape_to_ndgrid(volshape, **kwargs):
    """Compute a Tensor ndgrid from a volume shape."""
    if not all(float(d).is_integer() for d in volshape):
        raise ValueError("volshape needs to be a list of integers")
    return ndgrid(*[tf.range(0, d) for d in volshape], **kwargs)


def volshape_to_meshgrid(volshape, **kwargs):
    """Compute a Tensor meshgrid from a volume shape."""
    if not all(float(d).is_integer() for d in volshape):
        raise ValueError("volshape needs to be a list of integers")
    return meshgrid(*[tf.range(0, d) for d in volshape], **kwargs)


def ndgrid(*args, **kwargs):
    """Broadcast Tensors on an N-D grid with ij indexing."""
    return meshgrid(*args, indexing='ij', **kwargs)


def flatten(v):
    """Flatten a Tensor to 1-D."""
    return tf.reshape(v, [-1])


def meshgrid(*args, **kwargs):
    """
    Improved meshgrid using tiling instead of multiplication (~6x faster than TF's version).

    Args:
        *args: rank-1 Tensors.
        indexing: 'xy' (default) or 'ij'.
        name: operation name (optional).

    Returns:
        List of N-D Tensors.
    """
    indexing = kwargs.pop("indexing", "xy")
    name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError(f"'{key}' is an invalid keyword argument for this function")
    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    ndim = len(args)
    s0 = (1,) * ndim
    output = [tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])) for i, x in enumerate(args)]

    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    for i in range(len(output)):
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i + 1):]]))

    return output


# ======================= math utils =============================

def prod_n(lst):
    """Compute the product of all elements in a list."""
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs, **kwargs):
    """Convert subscript indices to linear indices (column-major order)."""
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))
    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]
    return ndx


def gaussian_kernel(sigma, windowsize=None, indexing='ij'):
    """
    Compute an N-D Gaussian kernel.

    Args:
        sigma: scalar or list of scalars (one per dimension).
        windowsize: scalar or list of scalars for kernel shape. Defaults to 3*sigma*2+1.
        indexing: 'ij' (default) or 'xy'.

    Returns:
        N-D normalized Gaussian kernel tensor.
    """
    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    sigma = [np.maximum(f, np.finfo(float).eps) for f in sigma]
    nb_dims = len(sigma)

    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]
    if len(sigma) != len(windowsize):
        raise ValueError('sigma and windowsize should have the same length. '
                         'Got: ' + str(sigma) + ' and ' + str(windowsize))

    mid = [(w - 1) / 2 for w in windowsize]
    mesh = volshape_to_meshgrid(windowsize, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    diff = [mesh[f] - mid[f] for f in range(len(windowsize))]
    exp_term = [-K.square(diff[f]) / (2 * (sigma[f] ** 2)) for f in range(nb_dims)]
    norms = [exp_term[f] - np.log(sigma[f] * np.sqrt(2 * np.pi)) for f in range(nb_dims)]

    g = tf.exp(K.sum(tf.stack(norms, axis=-1), -1))
    g /= tf.reduce_sum(g)

    return g


# ======================= model utils ============================

def stack_models(models, connecting_node_ids=None):
    """
    Stack Keras models sequentially without nesting them as layers.

    Preserves layer objects (does not copy). Modifying original layer weights
    affects the stacked model.

    Args:
        models: list of models in order [input_model, ..., output_model].
        connecting_node_ids: optional list of connecting node pointers between models.

    Returns:
        New stacked Keras model.
    """
    output_tensors = models[0].outputs
    stacked_inputs = [*models[0].inputs]

    for mi in range(1, len(models)):
        new_input_nodes = list(models[mi].inputs)
        stacked_inputs_contrib = list(models[mi].inputs)

        if connecting_node_ids is None:
            conn_id = list(range(len(new_input_nodes)))
            assert len(new_input_nodes) == len(models[mi - 1].outputs), \
                'argument count does not match'
        else:
            conn_id = connecting_node_ids[mi - 1]

        for out_idx, ii in enumerate(conn_id):
            new_input_nodes[ii] = output_tensors[out_idx]
            stacked_inputs_contrib[ii] = None

        output_tensors = mod_submodel(models[mi], new_input_nodes=new_input_nodes)
        stacked_inputs = stacked_inputs + stacked_inputs_contrib

    stacked_inputs_ = [i for i in stacked_inputs if i is not None]
    stacked_inputs = []
    for inp in stacked_inputs_:
        if inp not in stacked_inputs:
            stacked_inputs.append(inp)

    return keras.models.Model(stacked_inputs, output_tensors)


def mod_submodel(orig_model, new_input_nodes=None, input_layers=None):
    """
    Modify (cut and/or stitch) a Keras submodel.

    Layer objects are shared (not copied). Supports model stitching (new input nodes)
    and model cutting (specifying input layers inside the model).

    Args:
        orig_model: original Keras model.
        new_input_nodes: pointer to new input node replacements.
        input_layers: name or pointer to layers in the original model to replace as inputs.

    Returns:
        List of output tensors of the modified model.
    """
    def _layer_dependency_dict(orig_model):
        out_layers = orig_model.output_layers
        out_node_idx = orig_model.output_layers_node_indices
        node_list = [ol._inbound_nodes[out_node_idx[i]] for i, ol in enumerate(out_layers)]

        dct = {}
        dct_node_idx = {}
        while len(node_list) > 0:
            node = node_list.pop(0)
            add = True
            if len(dct.setdefault(node.outbound_layer, [])) > 0:
                for li, layers in enumerate(dct[node.outbound_layer]):
                    if (layers == node.inbound_layers
                            and dct_node_idx[node.outbound_layer][li] == node.node_indices):
                        add = False
                        break
            if add:
                dct[node.outbound_layer].append(node.inbound_layers)
                dct_node_idx.setdefault(node.outbound_layer, []).append(node.node_indices)
            for li, layer in enumerate(node.inbound_layers):
                if hasattr(layer, '_inbound_nodes'):
                    node_list.append(layer._inbound_nodes[node.node_indices[li]])
        return dct

    def _get_new_layer_output(layer, new_layer_outputs, inp_layers):
        if layer not in new_layer_outputs:
            if layer not in inp_layers:
                raise Exception('layer %s is not in inp_layers' % layer.name)
            for group in inp_layers[layer]:
                input_nodes = [None] * len(group)
                for li, inp_layer in enumerate(group):
                    if inp_layer in new_layer_outputs:
                        input_nodes[li] = new_layer_outputs[inp_layer]
                    else:
                        input_nodes[li] = _get_new_layer_output(inp_layer, new_layer_outputs, inp_layers)
                if len(input_nodes) == 1:
                    new_layer_outputs[layer] = layer(*input_nodes)
                else:
                    new_layer_outputs[layer] = layer(input_nodes)
        return new_layer_outputs[layer]

    inp_layers = _layer_dependency_dict(orig_model)

    if input_layers is None:
        InputLayerClass = keras.engine.topology.InputLayer
        input_layers = [l for l in orig_model.layers if isinstance(l, InputLayerClass)]
    else:
        if not isinstance(input_layers, (tuple, list)):
            input_layers = [input_layers]
        for idx, input_layer in enumerate(input_layers):
            if isinstance(input_layer, str):
                input_layers[idx] = orig_model.get_layer(input_layer)

    input_nodes = list(orig_model.inputs) if new_input_nodes is None else new_input_nodes
    assert len(input_nodes) == len(input_layers)

    new_layer_outputs = {input_layer: input_nodes[i] for i, input_layer in enumerate(input_layers)}

    output_layers = []
    for layer in orig_model.layers:
        if hasattr(layer, '_inbound_nodes'):
            for i in range(len(layer._inbound_nodes)):
                if layer.get_output_at(i) in orig_model.outputs:
                    output_layers.append(layer)
                    break
    assert len(output_layers) == len(orig_model.outputs), "Number of output layers don't match"

    return [_get_new_layer_output(output_layer, new_layer_outputs, inp_layers)
            for output_layer in output_layers]


def reset_weights(model, session=None):
    """
    Reset model weights using their initializers.

    Note: only uses kernel_initializer and bias_initializer. Does not close session.

    Args:
        model: Keras model to reset.
        session: current TF session (optional).
    """
    if session is None:
        session = K.get_session()

    for layer in model.layers:
        reset = False
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            reset = True
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
            reset = True
        if not reset:
            print('Could not find initializer for layer %s. Skipping.', layer.name)


def copy_model_weights(src_model, dst_model):
    """
    Copy weights from src to dst Keras model by layer name.

    Args:
        src_model: source Keras model.
        dst_model: destination Keras model.
    """
    for layer in tqdm(dst_model.layers):
        try:
            wts = src_model.get_layer(layer.name).get_weights()
            layer.set_weights(wts)
        except Exception:
            print('Could not copy weights of %s' % layer.name)


def robust_multi_gpu_model(model, gpus, verbose=True):
    """
    Re-wrap a Keras model for multi-GPU training if more than one GPU is available.

    Args:
        model: Keras model.
        gpus: list of GPU ids, or integer count of GPUs.
        verbose: whether to print what happened (default: True).

    Returns:
        Keras model (multi-GPU if applicable).
    """
    islist = isinstance(gpus, (list, tuple))
    if (islist and len(gpus) > 1) or (not islist and gpus > 1):
        count = gpus if not islist else len(gpus)
        if verbose:
            print("Returning multi-gpu (%d) model" % count)
        return keras.utils.multi_gpu_model(model, count)

    if verbose:
        print("Returning keras model back (single gpu found)")
    return model


# ======================= activation utils =======================

def logtanh(x, a=1):
    """log * tanh activation. See also: arcsinh."""
    return K.tanh(x) * K.log(2 + a * abs(x))


def arcsinh(x, alpha=1):
    """Scaled arcsinh activation. See also: logtanh."""
    return tf.asinh(x * alpha) / alpha


# ======================= label utils ============================

def prob_of_label(vol, labelvol):
    """
    Compute the probability of labels in labelvol for each voxel in vol.

    Args:
        vol: float numpy array of shape (nd + 1) with prob dist at each voxel.
        labelvol: int numpy array of shape (nd) with label values.

    Returns:
        nd numpy array of probabilities.
    """
    nb_dims = np.ndim(labelvol)
    assert np.ndim(vol) == nb_dims + 1, \
        "vol dimensions do not match [%d] vs [%d]" % (np.ndim(vol) - 1, nb_dims)

    shp = vol.shape
    nb_voxels = np.prod(shp[0:nb_dims])
    nb_labels = shp[-1]

    flat_vol = np.reshape(vol, (nb_voxels, nb_labels))
    rows_sums = flat_vol.sum(axis=1)
    flat_vol_norm = flat_vol / rows_sums[:, np.newaxis]

    idx = list(range(nb_voxels))
    v = flat_vol_norm[idx, labelvol.flat]
    return np.reshape(v, labelvol.shape)


def sample_to_label(model, sample):
    """Predict a sample batch and compute max labels."""
    res = model.predict(sample[0])
    return pred_to_label(sample[1], res)


def pred_to_label(*y):
    """Return argmax labels from one or more probability volumes."""
    return tuple(np.argmax(f, -1).astype(int) for f in y)


# ======================= batch utils ============================

def batch_gather(reference, indices):
    """
    Batchwise gathering of row indices.

    The numpy equivalent is reference[np.arange(batch_size), indices].

    Args:
        reference: tensor of shape (batch_size, dim1, ..., dimN).
        indices: 1-D integer tensor of shape (batch_size).

    Returns:
        Selected tensor of shape (batch_size, dim2, ..., dimN).
    """
    batch_size = K.shape(reference)[0]
    indices = tf.stack([tf.range(batch_size), indices], axis=1)
    return tf.gather_nd(reference, indices)


# ======================= numpy utils ============================

def _concat(lists, dim):
    if lists[0].size == 0:
        lists = lists[1:]
    return np.concatenate(lists, dim)


def softmax(x, axis):
    """NumPy softmax along a given axis."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)