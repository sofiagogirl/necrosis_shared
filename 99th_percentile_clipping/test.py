import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow.compat.v1 as tf

import network
import ops
from network import conv2d

tf.disable_v2_behavior()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def leaky_relu(x, alpha=0.1):
    xx = tf.nn.batch_normalization(x, 
                                   mean=tf.reduce_mean(x), 
                                   variance=tf.math.reduce_variance(x),
                                   offset=None, 
                                   scale=None, 
                                   variance_epsilon=1e-5)
    return tf.nn.relu(xx) - alpha * tf.nn.relu(-xx)


def down(inp, lvl):
    n_channels = 32
    name = 'Gen_down{}'.format(lvl)
    in_ch = inp.get_shape().as_list()[-1]
    out_ch = n_channels if lvl == 0 else in_ch * 2
    mid_ch = (in_ch + out_ch) // 2

    conv1 = leaky_relu(conv2d(inp, [3, 3, in_ch, mid_ch], name + '/conv1'))
    conv2 = leaky_relu(conv2d(conv1, [3, 3, mid_ch, mid_ch], name + '/conv2'))
    conv3 = leaky_relu(conv2d(conv2, [3, 3, mid_ch, out_ch], name + '/conv3'))

    tmp = tf.pad(inp, [[0, 0], [0, 0], [0, 0], [0, out_ch - in_ch]], 'CONSTANT')
    dic[name] = conv3 + tmp

    return tf.nn.avg_pool(dic[name], ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def up(inp, lvl):
    name = 'Gen_up{}'.format(lvl)
    image_size = 2048
    size = image_size >> lvl

    image = tf.image.resize_bilinear(inp, [size, size])
    image = tf.concat([image, dic[name.replace('up', 'down')]], axis=3)

    in_ch = image.get_shape().as_list()[-1]
    out_ch = in_ch // 4
    mid_ch = (in_ch + out_ch) // 2

    conv1 = leaky_relu(conv2d(image, [3, 3, in_ch, mid_ch], name + '/conv1'))
    conv2 = leaky_relu(conv2d(conv1, [3, 3, mid_ch, mid_ch], name + '/conv2'))
    conv3 = leaky_relu(conv2d(conv2, [3, 3, mid_ch, out_ch], name + '/conv3'))

    return conv3


def build_tower(inp):
    n_channels = 32
    n_levels = 4

    cur = inp
    print(cur.get_shape())

    for i in range(n_levels):
        cur = down(cur, i)

    ch = cur.get_shape().as_list()[-1]
    cur = leaky_relu(conv2d(cur, [3, 3, ch, ch], 'Gen_center'))

    for i in range(n_levels):
        cur = up(cur, n_levels - i - 1)

    return conv2d(cur, [3, 3, n_channels // 2, 3], 'Gen_last_layer')


if __name__ == '__main__':

    # ======================= config ==========================
    image_dir = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\data\\AF\\*.npy'
    output_path = 'C:\\Users\\sofia\\OneDrive\\Desktop\\sofia_necrosis_project\\output\\'
    checkpoint_path = 'model/42000'

    clip0 = 30000  # TxRed (channel 0)
    clip3 = 30000  # DAPI (channel 3)

    # ======================= data ============================
    images = glob.glob(image_dir)
    print(images)

    # ======================= model ===========================
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        input_ = tf.placeholder(tf.float32, shape=[2048, 2048, 4])
        dic = {}

        with tf.variable_scope('Generator'), tf.device('/gpu:0'):
            tf_output = build_tower(tf.expand_dims(input_, axis=0))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.train.Saver().restore(sess, checkpoint_path)

            for i in tqdm(range(len(images))):
                savepath = output_path + images[i].split('\\')[-1].replace('.npy', '')

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # ======================= preprocessing ===================
                image_af = np.load(images[i]).astype(np.float32)
                x = np.zeros((2048, 2048, 4), dtype=np.float32)
                x[:, :, 0] = image_af[:, :, 0]  # TxRed
                x[:, :, 3] = image_af[:, :, 3]  # DAPI

                # clipping
                x[:, :, 0] = np.clip(x[:, :, 0], 0, clip0)
                x[:, :, 3] = np.clip(x[:, :, 3], 0, clip3)

                # z-score normalization
                x[:, :, 0] = (x[:, :, 0] - np.mean(x[:, :, 0])) / np.std(x[:, :, 0])
                x[:, :, 3] = (x[:, :, 3] - np.mean(x[:, :, 3])) / np.std(x[:, :, 3])

                # ======================= inference =======================
                z = sess.run(tf_output, feed_dict={input_: x})
                z = np.squeeze(z)

                # YCbCr to RGB conversion
                z_temp = z.copy()
                z[:, :, 0] = z_temp[:, :, 0] + 1.403 * (z_temp[:, :, 1] - 128)
                z[:, :, 1] = z_temp[:, :, 0] - 0.714 * (z_temp[:, :, 1] - 128) - 0.344 * (z_temp[:, :, 2] - 128)
                z[:, :, 2] = z_temp[:, :, 0] + 1.773 * (z_temp[:, :, 2] - 128)
                z[z > 255] = 255

                # ======================= save output =====================
                im = Image.fromarray(z.astype(np.uint8))
                im.save(savepath + '.tif')