import os

import ops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob
from configobj import ConfigObj
from models import att_unet_2d
from tqdm import tqdm
from losses import *
import scipy.io as sio
import matplotlib.pyplot as plt
import batch_utils
from batch_utils import ImageTransformationBatchLoader_Testing


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.image_path = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\BF\\*.npy'
    vc.image_path = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\BF\\*.npy'
    #tc.image_path = 'I:\\covid_dataset_test\\Second_reg\\SP20-44-001\\target_2048\\*.mat'
    #vc.image_path = 'I:\\covid_dataset_test\\Second_reg\\SP20-44-001\\target_2048\\*.mat'
    # tc.image_path = 'I:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-18-00044-A6_wait\\target\\*.mat'
    # vc.image_path = 'I:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-18-00044-A6_wait\\target\\*.mat'

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('BF', 'AF')
    # def convert_inp_path_from_target(inp_path: str):
    #     return inp_path.replace('/target', '/input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = False, False  # True for .mat, False for .npy
    tc.data_inpnorm, vc.data_inpnorm = False, False  # True for normalizing input images

    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive
    # not used anymore

    tc.is_training, vc.is_training = False, False
    tc.image_size, vc.image_size = 2048, 2048   # 1408, 1408
    tc.num_slices, vc.num_slices = 2, 2
    tc.label_channels, vc.label_channels = 3, 3

    # tc.image_dir = 'F:/HER2/RoundF/target/'
    # tc.valid_cases = ['22980*003', '18023*003']
    # tc.test_cases = ['16967*003', '10205*002']
    # tc.test_cases = ['19-16967-B1-003']

    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32

    tc.batch_size, vc.batch_size = 1, 1
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 10, 10
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 1  # for the batchloader
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'

    return tc, vc


if __name__ == '__main__':
    # paths
    #model_path = 'G:/Regstain_Code/code/stage2_20220727_G&RSeperateTrain_initIter=0'
    checkpoint_path = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\model\\42000'
    output_path = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\output\\'
    # model_path = 'G:/Regstain_Code/code/stage2_20220727_G&RSeperateTrain_initIter=0'
    # checkpoint_path = model_path + '/model_G_iter=87700.h5'
    # output_path = model_path + '/outputs_G_iter=87700_AAW-18-00044-A6_good/'
    tf.io.gfile.mkdir(output_path)
    tf.io.gfile.mkdir(output_path + '/input/')

    tf.io.gfile.mkdir(output_path + '/label/')
    tf.io.gfile.mkdir(output_path + '/output/')
    # tf.io.gfile.mkdir(model_path)
    # tf.io.gfile.mkdir(model_path + '/output')

    # initialize architecture and load weights
    tc, vc = init_parameters()
    model_G = att_unet_2d((tc.image_size, tc.image_size, tc.num_slices), n_labels=tc.label_channels, name='model_G',
                          filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8, tc.n_channels * 16],
                          stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                          atten_activation='ReLU', attention='add',
                          output_activation=None, batch_norm=True, pool='ave', unpool='bilinear')
    checkpoint = tf.train.Checkpoint(model=model_G)
    checkpoint.restore(checkpoint_path) # MIGHT HAVE TO FIX LATER IF THE FILE TYPES ARE INCOMPATIBLE

    ssim_list = []
    psnr_list = []

    # _, _, test_images = batch_utils.Her2data_splitter(tc)
    test_images = glob.glob(vc.image_path)

    valid_bl = ImageTransformationBatchLoader_Testing(test_images, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=vc.n_shuffle_epoch)
    iterator_valid_bl = iter(valid_bl.dataset)

    # loop over batches

    print('valid images: ' + str(test_images[:2]))
    for i in tqdm(range(len(test_images) // tc.batch_size)):
        valid_x, valid_y = next(iterator_valid_bl)

        with tf.device('/cpu:0'):
        #with tf.device('/gpu:0'):
            valid_output = model_G(valid_x, training=False).numpy()

        for j in range(tc.batch_size):
            valid_output_temp = np.clip(valid_output[j], 0, 1)
            valid_x_temp = tf.concat([valid_x[j, :, :, 0:2], valid_x[j, :, :, 3:4]], axis=-1)
            valid_x_temp = (valid_x_temp / tf.reduce_max(valid_x_temp)).numpy()
            valid_y_temp = valid_y.numpy() * 255
            valid_y_temp = valid_y_temp[j]
            valid_image_path = test_images[i * tc.batch_size + j]

            cur_out_img_name = valid_image_path.split('\\')[-1].replace('.npy', '') + '.png'

            # with case name
            cur_case_name = valid_image_path.split('\\')[-3]
            plt.imsave(output_path + '/input/'
                       + cur_case_name + '_' + valid_image_path.split('\\')[-1].replace('.npy', '') + '_0.png',
                       valid_x_temp[:, :, 0])
            plt.imsave(output_path + '/input/'
                       + cur_case_name + '_' + valid_image_path.split('\\')[-1].replace('.npy', '') + '_1.png',
                       valid_x_temp[:, :, 1])
            plt.imsave(output_path + '/output/'
                       + cur_case_name + '_' + cur_out_img_name,
                       valid_output_temp)
            valid_y_temp = np.nan_to_num(valid_y_temp)
            plt.imsave(output_path + '/label/'
                       + cur_case_name + '_' + cur_out_img_name,
                       valid_y_temp)

            # no case name
            # plt.imsave(output_path + '/input/'
            #            + '_' + valid_image_path.split('\\')[-1].replace('.npy', '') + '_0.png',
            #            valid_x_temp[:, :, 0])
            # plt.imsave(output_path + '/input/'
            #            + '_' + valid_image_path.split('\\')[-1].replace('.npy', '') + '_1.png',
            #            valid_x_temp[:, :, 1])
            # plt.imsave(output_path + '/output/' + cur_out_img_name, valid_output_temp)
            # plt.imsave(output_path + '/label/' + cur_out_img_name, valid_y_temp)
