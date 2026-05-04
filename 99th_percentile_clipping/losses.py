from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from color_ops import rgb2hsl_tf, rgb2yiq_tf


# ======================= metrics ================================

class NCC:
    """Local (over window) normalized cross correlation loss."""

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, Ii, Ji):
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):
            self.win = [self.win] * ndims

        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute squared terms
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1 if ndims == 1 else [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum  = conv_fn(Ii, sum_filt, strides, padding)
        J_sum  = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        cc = (cross / I_var) * (cross / J_var)
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return 1 - self.ncc(y_true, y_pred)


class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - recommended if the gradient
    is computed on a downsampled vector field (where loss_mult equals the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        df = [None] * ndims

        for i in range(ndims):
            d = i + 1
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                dfi = w[1:, ...] * dfi

            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """Returns tensor of size [batch_size]."""
        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


# ======================= utils ==================================

def split_tensor(inp, x_split_times, y_split_times):
    """Split a tensor along spatial dimensions and concatenate along batch dim."""
    if x_split_times > 1:
        x_split = tf.split(inp, x_split_times, axis=1)
        inp = tf.concat(x_split, axis=0)
    if y_split_times > 1:
        y_split = tf.split(inp, y_split_times, axis=2)
        inp = tf.concat(y_split, axis=0)
    return inp


# ======================= generator losses =======================

def loss_G(D_fake_output, G_output, target, train_config, cur_epoch=None):
    if 'loss_mask' in train_config and train_config.loss_mask:
        if hasattr(train_config, 'case_filtering_x_subdivision'):
            assert train_config.case_filtering_x_subdivision == 1
        if hasattr(train_config, 'case_filtering_y_subdivision'):
            assert train_config.case_filtering_y_subdivision == 1
        target = target[:, :, :, :-1]
        loss_mask_from_R = target[:, :, :, -1:]

    # patch filtering
    if (train_config.is_training and train_config.case_filtering
            and cur_epoch >= train_config.case_filtering_starting_epoch):
        assert cur_epoch is not None

        if train_config.case_filtering_metric == 'ncc':
            min_ncc_threshold = (train_config.case_filtering_cur_mean
                                 - train_config.case_filtering_nsigma * train_config.case_filtering_cur_stdev)
            target_clipped = tf.clip_by_value(target, 0, 1)
            G_output_clipped = tf.clip_by_value(G_output, 0, 1)

            no_subdivision = (train_config.case_filtering_x_subdivision == 1
                              and train_config.case_filtering_y_subdivision == 1)

            if no_subdivision:
                cur_ncc = tf.stop_gradient(NCC(win=20, eps=1e-3).ncc(target_clipped, G_output_clipped))
                cur_mask = tf.cast(tf.math.greater(cur_ncc, min_ncc_threshold), tf.int32)
                train_config.epoch_filtering_ratio.append(
                    1 - tf.reduce_sum(cur_mask) / cur_mask.get_shape().as_list()[0])

                cur_index = tf.reshape(tf.where(cur_mask), [-1])
                G_output = tf.gather(G_output, cur_index, axis=0)
                target = tf.gather(target, cur_index, axis=0)
                D_fake_output = tf.gather(D_fake_output, cur_index, axis=0)
                if 'loss_mask' in train_config and train_config.loss_mask:
                    loss_mask_from_R = tf.gather(loss_mask_from_R, cur_index, axis=0)
            else:
                G_output_clipped_split = split_tensor(G_output_clipped,
                                                      train_config.case_filtering_x_subdivision,
                                                      train_config.case_filtering_y_subdivision)
                target_clipped_split = split_tensor(target_clipped,
                                                    train_config.case_filtering_x_subdivision,
                                                    train_config.case_filtering_y_subdivision)
                cur_ncc = tf.stop_gradient(NCC(win=20, eps=1e-3).ncc(target_clipped_split, G_output_clipped_split))
                cur_mask = tf.cast(tf.math.greater(cur_ncc, min_ncc_threshold), tf.int32)

                n_patches = (train_config.case_filtering_x_subdivision
                             * train_config.case_filtering_y_subdivision)
                bsz = G_output_clipped.get_shape().as_list()[0]

                filtering_ratio = 1.0 - tf.cast(tf.reduce_sum(cur_mask), tf.float32) / float(bsz * n_patches)
                train_config.epoch_filtering_ratio.append(filtering_ratio)

                cur_index = tf.reshape(tf.where(cur_mask), [-1])
                cur_index_image = cur_index // n_patches
                cur_index_image = tf.unique(cur_index_image)[0]
                G_output = tf.gather(G_output, cur_index_image, axis=0)
                target = tf.gather(target, cur_index_image, axis=0)

                # remove similar ratio of images from D to keep G-D step ratio consistent
                cur_mask_by_case = tf.reduce_sum(tf.reshape(cur_mask, [bsz, n_patches]), axis=-1)
                k = tf.maximum(tf.cast(tf.round(float(bsz) * (1.0 - filtering_ratio)), tf.int32), 1)
                cur_D_index = tf.math.top_k(cur_mask_by_case, k=k).indices
                D_fake_output = tf.gather(D_fake_output, cur_D_index, axis=0)
        else:
            print("Unsupported case filtering metric")
            exit(1)

    # apply loss mask from R
    if 'loss_mask' in train_config and train_config.loss_mask:
        G_output = G_output * loss_mask_from_R + tf.stop_gradient(G_output * (1 - loss_mask_from_R))
        target = target * loss_mask_from_R + tf.stop_gradient(target * (1 - loss_mask_from_R))

    G_berhu_loss = huber_reverse_loss(pred=G_output, label=target)
    G_tv_loss = tf.reduce_mean(tf.image.total_variation(G_output)) / (train_config.image_size ** 2)
    G_dis_loss = tf.reduce_mean(tf.square(1 - D_fake_output))
    G_total_loss = G_berhu_loss + 0.02 * G_tv_loss + train_config.lamda * G_dis_loss

    return G_total_loss, G_dis_loss, G_berhu_loss


def loss_G_with_R_progressive(D_fake_output, G_output, target, target_transformed, alpha, train_config, cur_epoch=None):
    """Blend G loss between original target and R-transformed target by alpha."""
    if alpha == 0:
        return loss_G(D_fake_output, G_output, target, train_config, cur_epoch)

    G_total_loss_orig, G_dis_loss_orig, G_berhu_loss_orig = loss_G(
        D_fake_output, G_output, target, train_config, cur_epoch)
    G_total_loss_trans, G_dis_loss_trans, G_berhu_loss_trans = loss_G(
        D_fake_output, G_output, target_transformed, train_config, cur_epoch)

    G_total_loss = (1 - alpha) * G_total_loss_orig + alpha * G_total_loss_trans
    G_dis_loss   = (1 - alpha) * G_dis_loss_orig   + alpha * G_dis_loss_trans
    G_berhu_loss = (1 - alpha) * G_berhu_loss_orig  + alpha * G_berhu_loss_trans

    return G_total_loss, G_dis_loss, G_berhu_loss


# ======================= registration losses ====================

def loss_cascaded_R1(R_outputs, fixed, training_config):
    training_config.R_loss_type = training_config.R1_params.R_loss_type
    training_config.lambda_r_tv = training_config.R1_params.lambda_r_tv
    return loss_R_no_gt(R_outputs, fixed, training_config)


def loss_cascaded_R2(R_outputs, fixed, training_config):
    training_config.R_loss_type = training_config.R2_params.R_loss_type
    training_config.lambda_r_tv = training_config.R2_params.lambda_r_tv
    return loss_R_no_gt(R_outputs, fixed, training_config)


def loss_R_flow_only(flow_pred, flow_gt, training_config):
    R_flow_mae_loss = tf.reduce_mean(tf.abs(flow_gt - flow_pred))
    R_flow_tv_loss = 0
    if training_config.lambda_r_tv > 0:
        R_flow_tv_loss = tf.reduce_mean(
            tf.image.total_variation(flow_pred) / (training_config.image_size ** 2))
    R_total_loss = R_flow_mae_loss + training_config.lambda_r_tv * R_flow_tv_loss
    return R_total_loss, R_flow_mae_loss


def loss_R_with_gt(R_outputs, fixed, flow_gt, loss_mask, training_config):
    moving_transformed, flow_pred = R_outputs

    if 'loss_mask' in training_config and training_config.loss_mask:
        moving_transformed = moving_transformed[:, :, :, :-1]

    if training_config.boundary_clipping:
        moving_transformed = moving_transformed * loss_mask
        fixed = fixed * loss_mask
        flow_gt = flow_gt * loss_mask

    if training_config.R_loss_type == 'berhu':
        R_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    else:
        assert training_config.R_loss_type == 'ncc'
        ncc = NCC(win=20, eps=1e-3)
        R_structure_loss = tf.reduce_mean(ncc.loss(y_true=fixed, y_pred=moving_transformed))

    R_flow_mae_loss = 0
    if training_config.lambda_r_mae > 0:
        R_flow_mae_loss = tf.reduce_mean(tf.abs(flow_gt - flow_pred))

    R_flow_tv_loss = 0
    if training_config.lambda_r_tv > 0:
        R_flow_tv_loss = tf.reduce_mean(Grad('l2').loss(None, flow_pred))

    R_total_loss = (R_structure_loss
                    + training_config.lambda_r_mae * R_flow_mae_loss
                    + training_config.lambda_r_tv * R_flow_tv_loss)

    return R_total_loss, R_structure_loss, R_flow_mae_loss


def loss_R_no_gt(R_outputs, fixed, training_config):
    moving_transformed, flow_pred = R_outputs

    if training_config.R_loss_type == 'berhu':
        R_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    else:
        assert training_config.R_loss_type == 'ncc'
        moving_transformed_clipped = tf.clip_by_value(moving_transformed, 0, 1)
        fixed_clipped = tf.clip_by_value(fixed, 0, 1)
        ncc = NCC(win=20, eps=1e-3)
        R_structure_loss = tf.reduce_mean(ncc.loss(y_true=fixed_clipped, y_pred=moving_transformed_clipped))

    R_flow_tv_loss = 0
    if training_config.lambda_r_tv > 0:
        R_flow_tv_loss = tf.reduce_mean(Grad('l2').loss(None, flow_pred))

    R_total_loss = R_structure_loss + training_config.lambda_r_tv * R_flow_tv_loss

    if hasattr(training_config, 'lambda_dvf_batch_decay') and training_config.lambda_dvf_batch_decay is not None:
        R_DVF_batch_decay = tf.reduce_mean(tf.math.abs(tf.reduce_mean(flow_pred, axis=0)))
        R_total_loss += training_config.lambda_dvf_batch_decay * R_DVF_batch_decay

    return R_total_loss, R_structure_loss


# ======================= color losses ===========================

def color_l1_in_hsl(moving_rgb, fixed_rgb, training_config):
    """L1 loss in HSL color space. Note: this loss does not work well."""
    scale = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([1 / 359, 1, 1]), 0), 0), 0)

    moving_hsl = rgb2hsl_tf(moving_rgb) * scale
    fixed_hsl = rgb2hsl_tf(fixed_rgb) * scale

    fixed_hsl_mask = tf.stop_gradient(fixed_hsl[:, :, :, 2] < training_config.L_channel_ignore_threshold)
    moving_hsl = tf.boolean_mask(moving_hsl, fixed_hsl_mask)
    fixed_hsl = tf.boolean_mask(fixed_hsl, fixed_hsl_mask)

    return l1_loss(moving_hsl, fixed_hsl)


def color_l1_in_yiq(moving_rgb, fixed_rgb, training_config):
    """L1 loss in YIQ color space."""
    if hasattr(training_config, "C_params"):
        training_config.L_channel_ignore_lower_th = training_config.C_params.L_channel_ignore_lower_th
        training_config.L_channel_ignore_upper_th = training_config.C_params.L_channel_ignore_upper_th

    moving_yiq = rgb2yiq_tf(moving_rgb)
    fixed_yiq = rgb2yiq_tf(fixed_rgb)

    if training_config.L_channel_ignore_lower_th or training_config.L_channel_ignore_upper_th is not None:
        fixed_mask = tf.logical_and(
            fixed_yiq[:, :, :, 0] > training_config.L_channel_ignore_lower_th,
            fixed_yiq[:, :, :, 0] < training_config.L_channel_ignore_upper_th)
        moving_mask = tf.logical_and(
            moving_yiq[:, :, :, 0] > training_config.L_channel_ignore_lower_th,
            moving_yiq[:, :, :, 0] < training_config.L_channel_ignore_upper_th)
        yiq_mask = tf.stop_gradient(tf.cast(tf.logical_or(fixed_mask, moving_mask), tf.float32))
        C_structure_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(moving_yiq - fixed_yiq) * tf.expand_dims(yiq_mask, -1), [1, 2, 3])
            / (tf.reduce_sum(yiq_mask, [1, 2]) + 1e-3))
    else:
        C_structure_loss = tf.reduce_mean(tf.abs(moving_yiq - fixed_yiq))

    return C_structure_loss


def loss_C_no_gt(C_outputs, fixed, training_config):
    moving_transformed, color_params = C_outputs

    if hasattr(training_config, "C_params"):
        training_config.C_loss_type = training_config.C_params.C_loss_type
        training_config.hsv_h_reg_term = training_config.C_params.hsv_h_reg_term
        training_config.hsv_s_reg_term = training_config.C_params.hsv_s_reg_term
        training_config.hsv_v_reg_term = training_config.C_params.hsv_v_reg_term

    if training_config.C_loss_type == 'berhu':
        C_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    elif training_config.C_loss_type == 'mae_yiq':
        C_structure_loss = color_l1_in_yiq(moving_transformed, fixed, training_config)
    else:
        raise NotImplementedError()

    C_total_loss = C_structure_loss

    # regularization terms (h=delta h, s/v=scaling factors)
    if training_config.hsv_h_reg_term is not None:
        C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.square(color_params[:, 0]))
    if training_config.hsv_s_reg_term is not None:
        C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.square(1 - color_params[:, 1]))
    if training_config.hsv_v_reg_term is not None:
        C_total_loss += training_config.hsv_v_reg_term * tf.reduce_sum(tf.square(1 - color_params[:, 2]))

    return C_total_loss, C_structure_loss


def loss_C_no_gt_with_D(D_fake_output, C_outputs, fixed, training_config):
    moving_transformed, color_params = C_outputs

    if hasattr(training_config, "C_params"):
        training_config.C_loss_type = training_config.C_params.C_loss_type
        training_config.lamda_C = training_config.C_params.lamda_C
        training_config.hsv_h_reg_term = training_config.C_params.hsv_h_reg_term
        training_config.hsv_s_reg_term = training_config.C_params.hsv_s_reg_term
        training_config.hsv_v_reg_term = training_config.C_params.hsv_v_reg_term

    if training_config.C_loss_type == 'berhu':
        C_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    elif training_config.C_loss_type == 'mae_yiq':
        C_structure_loss = color_l1_in_yiq(moving_transformed, fixed, training_config)
    else:
        raise NotImplementedError()

    C_dis_loss = tf.reduce_mean(tf.square(1 - D_fake_output))
    C_total_loss = C_structure_loss + training_config.lamda_C * C_dis_loss

    # regularization terms (h=delta h, s/v=scaling factors)
    if training_config.hsv_h_reg_term is not None:
        C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.square(color_params[:, 0]))
    if training_config.hsv_s_reg_term is not None:
        C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.square(1 - color_params[:, 1]))
    if training_config.hsv_v_reg_term is not None:
        C_total_loss += training_config.hsv_v_reg_term * tf.reduce_sum(tf.square(1 - color_params[:, 2]))

    return C_total_loss, C_dis_loss, C_structure_loss


# ======================= discriminator loss =====================

def loss_D(D_real_output, D_fake_output):
    D_fake_loss = tf.reduce_mean(tf.square(D_fake_output))
    D_real_loss = tf.reduce_mean(tf.square(1 - D_real_output))
    D_total_loss = D_fake_loss + D_real_loss
    return D_total_loss, D_real_loss, D_fake_loss


# ======================= basic losses ===========================

def l1_loss(output, target):
    return tf.reduce_mean(tf.abs(output - target))


def huber_reverse_loss(pred, label, delta=0.2, adaptive=True):
    """Reverse Huber (berhu) loss with optional batch-adaptive delta."""
    diff = tf.abs(pred - label)
    if adaptive:
        delta = delta * tf.math.reduce_std(label)
    loss = tf.reduce_mean(
        tf.cast(diff <= delta, tf.float32) * diff
        + tf.cast(diff > delta, tf.float32) * (diff ** 2 / 2 + delta ** 2 / 2) / delta)
    return loss


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    pt_1 = tf.where(tf.equal(y_true, 1.0), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0.0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_mean(
        alpha * tf.pow(1 - pt_1, gamma) * tf.math.log(tf.clip_by_value(pt_1, 1e-8, 1))
        + (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(tf.clip_by_value(1 - pt_0, 1e-8, 1)))