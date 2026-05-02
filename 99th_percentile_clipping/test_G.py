import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from configobj import ConfigObj
from tqdm import tqdm

import ops
import batch_utils
from models import att_unet_2d
from losses import *
from batch_utils import ImageTransformationBatchLoader_Testing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    # ======================= data paths =====================
    tc.image_path = 'G:\\Project_Nectrotic\\Data\\ProcessingData\\RegistrationRound2_crops\\NPY\\NonNecrotic\\sample4B\\BF\\*.npy'
    vc.image_path = 'G:\\Project_Nectrotic\\Data\\ProcessingData\\RegistrationRound2_crops\\NPY\\NonNecrotic\\sample4B\\BF\\*.npy'

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('BF', 'AF')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    # ======================= data params ====================
    tc.is_mat, vc.is_mat = False, False
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive

    # ======================= network params =================
    tc.is_training, vc.is_training = False, False
    tc.image_size, vc.image_size = 2048, 2048
    tc.num_slices, vc.num_slices = 2, 2
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32

    # ======================= training params ================
    tc.batch_size, vc.batch_size = 1, 1
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 10, 10
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 1
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'

    return tc, vc


if __name__ == '__main__':

    # ======================= paths ==========================
    checkpoint_path = 'C:\\Users\\76\\sofia_necrosis_project\\model\\42000'
    output_path = 'C:\\Users\\76\\sofia_necrosis_project\\output\\'

    # create output directories
    tf.io.gfile.mkdir(output_path)
    tf.io.gfile.mkdir(output_path + '/input/')
    tf.io.gfile.mkdir(output_path + '/label/')
    tf.io.gfile.mkdir(output_path + '/output/')

    # ======================= model ==========================
    tc, vc = init_parameters()
    model_G = att_unet_2d(
        (tc.image_size, tc.image_size, tc.num_slices), n_labels=tc.label_channels, name='model_G',
        filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8, tc.n_channels * 16],
        stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
        atten_activation='ReLU', attention='add',
        output_activation=None, batch_norm=True, pool='ave', unpool='bilinear')

    checkpoint = tf.train.Checkpoint(model=model_G)
    checkpoint.restore(checkpoint_path)

    # ======================= data ===========================
    test_images = glob.glob(vc.image_path)
    print('Valid images:', test_images[:2])

    valid_bl = ImageTransformationBatchLoader_Testing(
        test_images, vc, vc.num_slices, is_testing=True,
        n_parallel_calls=vc.n_threads, q_limit=vc.q_limit, n_epoch=vc.n_shuffle_epoch)
    iterator_valid_bl = iter(valid_bl.dataset)

    # ======================= inference loop =================
    for i in tqdm(range(len(test_images) // tc.batch_size)):
        valid_x, valid_y = next(iterator_valid_bl)

        with tf.device('/gpu:0'):
            valid_output = model_G(valid_x, training=False).numpy()

        for j in range(tc.batch_size):
            valid_output_temp = np.clip(valid_output[j], 0, 1)
            valid_x_temp = tf.concat([valid_x[j, :, :, 0:2], valid_x[j, :, :, 3:4]], axis=-1)
            valid_x_temp = (valid_x_temp / tf.reduce_max(valid_x_temp)).numpy()
            valid_y_temp = valid_y.numpy() * 255
            valid_y_temp = valid_y_temp[j]

            valid_image_path = test_images[i * tc.batch_size + j]
            cur_case_name = valid_image_path.split('\\')[-3]
            cur_out_img_name = valid_image_path.split('\\')[-1].replace('.npy', '') + '.png'
            base_name = cur_case_name + '_' + valid_image_path.split('\\')[-1].replace('.npy', '')

            # save input channels
            plt.imsave(output_path + '/input/' + base_name + '_0.png', valid_x_temp[:, :, 0])
            plt.imsave(output_path + '/input/' + base_name + '_1.png', valid_x_temp[:, :, 1])

            # save output
            plt.imsave(output_path + '/output/' + cur_case_name + '_' + cur_out_img_name, valid_output_temp)

            # save label
            valid_y_temp = np.nan_to_num(valid_y_temp).astype(np.uint8)
            plt.imsave(output_path + '/label/' + cur_case_name + '_' + cur_out_img_name, valid_y_temp)