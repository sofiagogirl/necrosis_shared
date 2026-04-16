import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from configobj import ConfigObj
from network import conv2d
from tqdm import tqdm
import glob, os, sys
import network, ops
#import batch_utils
import numpy as np
import scipy.io
import PIL
from PIL import Image

#
# label = np.load('D:/Zhensong/StainUnstain/training/Unet_mse_TV_128_64SalivalGland_ycbcr/image_dataset/test/target/1.npy')
# path = 'D:/Zhensong/StainUnstain/training/Unet_mse_TV_128_64SalivalGland_ycbcr/1.mat'
# print(label.max())
# label_temp = label.copy()
# print(label_temp.max())
# label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
# label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 +128
# label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 +128
#
# z = label
# z_temp = z.copy()
# z[:,:,0] = z_temp[:,:,0] + 1.403* (z_temp[:,:,1] - 128)
# z[:,:,1] = z_temp[:,:,0] - 0.714* (z_temp[:,:,1] - 128) - 0.344*(z_temp[:,:,2] - 128)
# z[:,:,2] = z_temp[:,:,0] + 1.773* (z_temp[:,:,2] - 128)
# print(label_temp.max())
# scipy.io.savemat(path,{'ycbcr': label,'rgb': z, 'ori': label_temp})

# def leakyRelu(x, alpha=0.1):
#     xx = tf.layers.batch_normalization(x)
#     return tf.nn.relu(xx) - alpha * tf.nn.relu(-xx)


# def down(inp, in_ch, out_ch, name):
#     mid_ch = (in_ch + out_ch) // 2
#     conv1 = leakyRelu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
#     conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
#     conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
#     tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
#     dic[name] = conv3 + tmp
#     return tf.nn.avg_pool(dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

# def up(inp, in_ch, out_ch, size, name):
#     image = tf.image.resize_bilinear(inp, [size, size])
#     image = tf.concat([image, dic[name.replace('up', 'down')]], axis=3)
#     mid_ch = (in_ch + out_ch) // 2
#     conv1 = leakyRelu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
#     conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
#     conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
#     return conv3

# def build_tower(inp):
#     down1 = down(inp,   3,   64, 'Gen_down0')
#     down2 = down(down1, 64,  128, 'Gen_down1')
#     down3 = down(down2, 128,  256, 'Gen_down2')
#     down4 = down(down3, 256,  512, 'Gen_down3')
#     ctr = leakyRelu(conv2d(down4, [3,3,512,512], 'Gen_center'))
#     size = 1024
#     up4 = up(ctr, 512*2, 256,  size//8, 'Gen_up3')
#     up3 = up(up4, 256*2, 128,  size//4, 'Gen_up2')
#     up2 = up(up3, 128*2, 64,  size//2, 'Gen_up1')
#     up1 = up(up2, 64*2, 32,  size, 'Gen_up0')
#     return conv2d(up1, [3,3,32,3], 'Gen_last_layer')
def leakyRelu(x, alpha=0.1):
    xx = tf.layers.batch_normalization(x)
    return tf.nn.relu(xx) - alpha * tf.nn.relu(-xx)

def down(inp, lvl):
    n_channels=32
    name = 'Gen_down{}'.format(lvl)
    in_ch = inp.get_shape().as_list()[-1]
    out_ch = n_channels if lvl == 0 else in_ch * 2
    mid_ch = (in_ch + out_ch) // 2
    conv1 = leakyRelu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
    conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
    conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
    tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
    dic[name] = conv3 + tmp
    # return conv2d(self.dic[name], [3,3,out_ch,out_ch], name + '/downsampling_conv', strides=(1,2,2,1))
    return tf.nn.avg_pool(dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

def up(inp, lvl):
    name = 'Gen_up{}'.format(lvl)
    image_size=2048
    size = image_size >> lvl
    image = tf.image.resize_bilinear(inp, [size, size])
    image = tf.concat([image, dic[name.replace('up', 'down')]], axis=3)
    in_ch = image.get_shape().as_list()[-1]
    out_ch = in_ch // 4
    mid_ch = (in_ch + out_ch) // 2
    conv1 = leakyRelu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
    conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
    conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
    return conv3

def build_tower(inp):
    n_channels=32
    n_levels=4
    dic = {}

    cur = inp
    print(cur.get_shape())
    for i in range(n_levels):
        cur = down(cur, i)
    ch = cur.get_shape().as_list()[-1]
    cur = leakyRelu(conv2d(cur, [3,3,ch,ch], 'Gen_center'))
    for i in range(n_levels):
        cur = up(cur, n_levels - i - 1)

    return conv2d(cur, [3,3,n_channels//2,3], 'Gen_last_layer')




if __name__ == '__main__':

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        #images = glob.glob('G:/Transplant/reg_network_code/second_TileReg/target/*.npy')
        #images = glob.glob('F:/Transplant-lung/First_reg_crop/SP20-917-001/input_pro/*.npy')
        images = glob.glob('G:\\Project_Nectrotic\\Data\\ProcessingData\\RegistrationRound2_crops\\NPY\\NonNecrotic\\sample4B\\AF\\*.npy')
        print(images)
        input_ = tf.placeholder(tf.float32, shape=[2048, 2048, 4])
        #devices = ops.get_available_gpus()
        dic = {}

        with tf.variable_scope('Generator'), tf.device('/gpu:0'):
            tf_output = build_tower(tf.expand_dims(input_, axis=0))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.train.Saver().restore(sess, 'model/42000')
            # os.system('rm -r outputs_HE')
            # os.system('mkdir outputs_HE')
            means = []
            output_path = 'C:\\Users\\76\\sofia_necrosis_project\\output\\'
            for i in tqdm(range(len(images))):
               # path=images[i].replace('.npy','.mat')
               # path = 'outputs/{}'.format(images[i].split('\\')[-1].split('.')[0])
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)
                #if os.path.exists(os.path.dirname(images[i]).replace('target','output_round3_he+dapi/iter34000')) == False:
                #    os.mkdir(os.path.dirname(images[i]).replace('target','output_round3_he+dapi/iter34000'))
                savepath = output_path + images[i].split('\\')[-1].replace('.npy', '')
                # savepath_target = (images[i].replace('target', 'output_dapi+txred_v1_33000')).replace('npy','target_mat')
                #x = np.load(images[i].replace('target', 'output_DAPI+TxRed_r2'))
                #x = np.dstack((np.load(images[i].replace('target', 'input'))[:,:,0],np.load(images[i].replace('target', 'input'))[:,:,1]))
                #x = np.dstack((x, np.load(images[i].replace('input_reg', 'input_reg'))[:,:,2]))
                #x = np.dstack((x, np.load(images[i].replace('input_reg', 'input_reg'))[:,:,3]))
                #x = np.load(images[i].replace('target_aligned_r1', 'input_reg'))[:,:,0]
                #x = np.array(x, dtype = np.float32)
                #x = np.expand_dims(x,-1)
                image_af = np.load(images[i]).astype(np.float32)
                x = np.zeros((2048, 2048, 4), dtype=np.float32)
                x[:,:,0] = image_af[:,:,0]  
                x[:,:,3] = image_af[:,:,3]  

                # different clip values
                clip0 = 30000
                clip3 = 30000

                #clipping the data
                x[:,:,0] = np.clip(x[:,:, 0], 0, clip0)
                x[:,:,3] = np.clip(x[:,:,3], 0, clip3)

                # z- score normalization
                x[:,:,0] = (x[:,:,0] - np.mean(x[:,:,0]))/np.std(x[:,:,0])
                x[:,:,3] = (x[:,:,3] - np.mean(x[:,:,3]))/np.std(x[:,:,3])

                # min max normalization
                #x[:,:,0] = (x[:,:,0] - np.min(x[:,:,0]))/(np.max(x[:,:,0]) - np.min(x[:,:,0]))
                #x[:,:,3] = (x[:,:,3] - np.min(x[:,:,3]))/(np.max(x[:,:,3]) - np.min(x[:,:,3]))

                # [-1, 1] normalization
                #x[:,:,0] = 2 * ((x[:,:,0] - np.min(x[:,:,0]))/(np.max(x[:,:,0]) - np.min(x[:,:,0]))) - 1
                #x[:,:,3] = 2 * ((x[:,:,3] - np.min(x[:,:,3]))/(np.max(x[:,:,3]) - np.min(x[:,:,3]))) - 1

            
                #image_temp =  (x.copy())
                # for j in range(4):
                #x[:,:] = (image_temp[:,:] - np.mean(image_temp[:,:]))/(np.std(image_temp[:,:]))
                # xx=np.rot90(x,0)
                # x = x/65535 * 2 - 1
                xx=x

                #y = (np.load(images[i]))#[50:-50,50:-50]#[75:-75,75:-75]#[:1224,:1224]
               # y = np.load(images[i].replace('input', 'target'))[:1024,:1024,:]
                #label = np.load(images[i]).astype(np.float32)

                z = sess.run(tf_output, feed_dict={input_: xx})
                z = np.squeeze(z)
                z_temp = z.copy()
                z[:,:,0] = z_temp[:,:,0] + 1.403* (z_temp[:,:,1] - 128)
                z[:,:,1] = z_temp[:,:,0] - 0.714* (z_temp[:,:,1] - 128) - 0.344*(z_temp[:,:,2] - 128)
                z[:,:,2] = z_temp[:,:,0] + 1.773* (z_temp[:,:,2] - 128)
                z[z>255]=255

                #label=y#[:1024-32,:1024-32]
                # z=np.rot90(z,0)
                # label_temp = label.copy()
                # label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
                # label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 + 128
                # label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 + 128

                # scipy.io.savemat(path+'.mat', {'output': z, 'target':label, 'input':xx})
                # scipy.io.savemat(savepath, {'output':z, 'target':label})
                #label = label * 255 MIGHT NEED TO UNCOMMENT THIS LATER

                # im_target = Image.fromarray(label.astype(np.uint8))
                # im_target.save(savepath+'target.tif')
                im = Image.fromarray(z.astype(np.uint8))
                im.save(savepath+'.tif')
                #scipy.io.savemat(savepath, {'output': z})
