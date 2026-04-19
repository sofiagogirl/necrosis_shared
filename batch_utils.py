import random
import glob

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from matplotlib import pyplot as plt


# ======================= base class =============================

class BatchLoader:
    def __init__(self, images, config, input_channels, is_testing, n_parallel_calls, q_limit, n_epoch):
        self.num_epoch = n_epoch
        self.images = images
        self.config = config
        self.is_testing = is_testing
        self.input_channels = input_channels
        self.label_channels = config.label_channels
        self.image_size = config.image_size
        self.raw_size = config.image_size
        self.num_parallel_calls = n_parallel_calls
        self.q_limit = q_limit
        self.PATHS = []

        for i in range(self.num_epoch):
            if self.config.is_training:
                random.shuffle(self.images)
            self.PATHS.extend(self.images)
        print(f"Length of the image file list: {len(self.PATHS)}")

        self.dataset = self.create_dataset_from_generator()

        assert (self.is_testing and not self.config.is_training) or \
               (not self.is_testing and self.config.is_training)

        if self.config.is_training:
            self.dataset = self.dataset.shuffle(buffer_size=self.q_limit, reshuffle_each_iteration=True)
            self.dataset = self.dataset.map(self.augment, num_parallel_calls=config.n_threads)

        self.dataset = self.dataset.batch(self.config.batch_size).prefetch(100)
        self.iter: tf.compat.v1.data.Iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(self.dataset),
            tf.compat.v1.data.get_output_shapes(self.dataset))

    def create_dataset_from_generator(self):
        raise NotImplementedError

    def augment(self, *args):
        raise NotImplementedError

    def init_iter(self):
        return self.iter.make_initializer(self.dataset)


# ======================= training loader ========================

class ImageTransformationBatchLoader(BatchLoader):
    def __init__(self, train_images, tc, num_slices, **kwargs):
        super().__init__(train_images, tc, num_slices, **kwargs)

        if not hasattr(self.config, 'filter_blank'):
            self.config.filter_blank = False
        elif self.config.filter_blank:
            assert self.config.filter_threshold is not None

        self.cur_filter_count = 0
        self.case_trial_limit = 10

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(
            lambda x: tf.data.Dataset.from_generator(
                self.parse_and_generate,
                output_types=(tf.float32, tf.float32),
                args=(x,),
                output_shapes=(
                    tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                    tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
            cycle_length=self.config.n_threads,
            block_length=1,
            num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s
        start, end = self.config.channel_start_index, self.config.channel_end_index
        path = path.decode('UTF-8')

        if self.config.is_mat:
            try:
                image = loadmat(self.config.convert_inp_path_from_target(path))['input_tile'].astype(np.float32)[:, :, start:end]
                label = loadmat(path)['target_tile'].astype(np.float32)
            except Exception:
                print(self.config.convert_inp_path_from_target(path))
        else:
            image = np.transpose(
                np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, start:end],
                axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0])

        # normalization
        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = np.reshape([1500, 1500, 1500, 1000], [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        # crop edges
        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        size = image.shape[0]
        cur_trial_count = 0
        x = 0

        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)

                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    lab = label[xx:xx + s, yy:yy + s]

                    if (self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold
                            and cur_trial_count < self.case_trial_limit):
                        cur_trial_count += 1
                        continue
                    else:
                        yield img.astype(np.float32), lab.astype(np.float32)

                if yy == size - s:
                    break
                y += stride

            if xx == size - s:
                break
            x += stride

    def augment(self, img, lab):
        imglab = tf.concat([img, lab], axis=-1)
        imglab = tf.image.random_flip_left_right(imglab)

        img = imglab[:, :, 0:self.config.num_slices]
        lab = imglab[:, :, self.config.num_slices:self.config.num_slices + self.label_channels]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=random_number)
        lab = tf.image.rot90(lab, k=random_number)

        return img, lab


# ======================= testing loader =========================

class ImageTransformationBatchLoader_Testing(BatchLoader):

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(
            lambda x: tf.data.Dataset.from_generator(
                self.parse_and_generate,
                output_types=(tf.float32, tf.float32),
                args=(x,),
                output_shapes=(
                    tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                    tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
            cycle_length=self.config.n_threads,
            block_length=1,
            num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):
        path = path.decode('UTF-8')
        start, end = self.config.channel_start_index, self.config.channel_end_index

        if self.config.is_mat:
            image = loadmat(self.config.convert_inp_path_from_target(path))['input'].astype(np.float32)[:2048, :2048, start:end]
            label = loadmat(path)['target'].astype(np.float32)[:2048, :2048, :] / 255
        else:
            image = np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, [0, 3]]
            label = np.load(path).astype(np.float32) / 255.0
            label = tf.image.resize(label, [2048, 2048]).numpy()

        # normalization
        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = np.reshape([1500, 1500, 1500, 1000], [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        yield image.astype(np.float32), label.astype(np.float32)


# ======================= registration loaders ===================

class ImageRegistrationBatchLoader(BatchLoader):
    """Batch loader for self-supervised registration with synthetic flow."""

    def create_dataset_from_generator(self):
        return tf.data.Dataset.from_generator(
            self.parse_and_generate,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([self.image_size, self.image_size, 3]),
                tf.TensorShape([self.image_size, self.image_size, 3]),
                tf.TensorShape([self.image_size, self.image_size, 2]),
                tf.TensorShape([self.image_size, self.image_size, 1])))

    def parse_and_generate(self):
        s = self.config.image_size
        stride = s

        for path in self.PATHS:
            if self.config.is_mat:
                moving = loadmat(path)['target'].astype(np.float32)
            else:
                moving = np.load(path).astype(np.float32)

            # generate random shift
            maxshift = 40
            x_shift = int(random.uniform(-maxshift, maxshift))
            y_shift = int(random.uniform(-maxshift, maxshift))

            flow_gt = np.stack([np.ones([s, s]) * x_shift, np.ones([s, s]) * y_shift], axis=-1)
            loss_mask = np.ones((s, s, 1))
            if x_shift >= 0:
                loss_mask[:x_shift, :, :] = 0
            else:
                loss_mask[x_shift:, :, :] = 0
            if y_shift >= 0:
                loss_mask[:, :y_shift, :] = 0
            else:
                loss_mask[:, y_shift:, :] = 0

            fixed = translate(moving, [x_shift, y_shift], interpolation='bilinear', fill_mode='constant')

            size = moving.shape[0]
            x = max(0, int(x_shift + 1))
            x_bound = min(size - s, size - s - x)

            while True:
                y = max(0, int(y_shift + 1))
                y_bound = min(size - s, size - s - y)
                while True:
                    rand_choice_stride = random.randint(0, 15)
                    xx = min(x + rand_choice_stride * s // 16, x_bound)
                    yy = min(y + rand_choice_stride * s // 16, y_bound)

                    if yy != y_bound and xx != x_bound:
                        move = moving[xx:xx + s, yy:yy + s]
                        fix = fixed[xx:xx + s, yy:yy + s]
                        yield move.astype(np.float32), fix, flow_gt, loss_mask

                    if yy == y_bound:
                        break
                    y += stride

                if xx == x_bound:
                    break
                x += stride

    def augment(self, moving, fixed, flow, loss_mask):
        return moving, fixed, flow, loss_mask


class PairedImageRegistrationBatchLoader(BatchLoader):
    """Paired registration training loader (without DVF ground truth)."""

    def __init__(self, train_images, tc, num_slices, **kwargs):
        super().__init__(train_images, tc, num_slices, **kwargs)

        if not hasattr(self.config, 'filter_blank'):
            self.config.filter_blank = False
        elif self.config.filter_blank:
            assert self.config.filter_threshold is not None

        self.case_trial_limit = 10

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(
            lambda x: tf.data.Dataset.from_generator(
                self.parse_and_generate,
                output_types=(tf.float32, tf.float32),
                args=(x,),
                output_shapes=(
                    tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                    tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
            cycle_length=self.config.n_threads,
            block_length=1,
            num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s
        path = path.decode('UTF-8')

        if self.config.is_mat:
            fixed = loadmat(path)['output']
            moving = loadmat(path)['label']
        else:
            raise NotImplementedError()

        size = moving.shape[0]
        cur_trial_count = 0
        x = 0

        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)

                if yy != size - s and xx != size - s:
                    fix = fixed[xx:xx + s, yy:yy + s]
                    mov = moving[xx:xx + s, yy:yy + s]

                    if (self.config.filter_blank and np.mean(mov) >= self.config.filter_threshold
                            and cur_trial_count < self.case_trial_limit):
                        cur_trial_count += 1
                        continue
                    else:
                        yield fix.astype(np.float32), mov.astype(np.float32)

                if yy == size - s:
                    break
                y += stride

            if xx == size - s:
                break
            x += stride

    def augment(self, fix, mov):
        fixmov = tf.concat([fix, mov], axis=-1)
        fixmov = tf.image.random_flip_left_right(fixmov)

        fix = fixmov[:, :, 0:self.config.num_slices]
        mov = fixmov[:, :, self.config.num_slices:self.config.num_slices + self.label_channels]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        fix = tf.image.rot90(fix, k=random_number)
        mov = tf.image.rot90(mov, k=random_number)

        return fix, mov


class AffineImageRegistrationBatchLoader(BatchLoader):
    """Self-supervised registration pretraining loader with affine transforms."""

    def create_dataset_from_generator(self):
        return tf.data.Dataset.from_generator(
            self.parse_and_generate,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([self.image_size, self.image_size, 3]),
                tf.TensorShape([self.image_size, self.image_size, 3])))

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s

        for path in self.PATHS:
            if self.config.is_mat:
                moving = loadmat(path)['target'].astype(np.float32)
            else:
                moving = np.load(path).astype(np.float32)

            # generate random affine transform params
            maxshift = 40
            x_shift = int(random.uniform(-maxshift, maxshift))
            y_shift = int(random.uniform(-maxshift, maxshift))
            shear_degree = random.uniform(-5, 5)

            fixed = translate(moving, [x_shift, y_shift], interpolation='bilinear', fill_mode='constant').numpy()
            fixed = tf.keras.preprocessing.image.apply_affine_transform(
                fixed, theta=0, tx=0, ty=0, shear=shear_degree, fill_mode='nearest')

            size = moving.shape[0]
            x = max(0, int(x_shift + 1))
            x_bound = min(size - s, size - s - x)

            while True:
                y = max(0, int(y_shift + 1))
                y_bound = min(size - s, size - s - y)
                while True:
                    rand_choice_stride = random.randint(0, 15)
                    xx = min(x + rand_choice_stride * s // 16, x_bound)
                    yy = min(y + rand_choice_stride * s // 16, y_bound)

                    if yy != y_bound and xx != x_bound:
                        move = moving[xx:xx + s, yy:yy + s]
                        fix = fixed[xx:xx + s, yy:yy + s]

                        if not np.all(fix):
                            continue

                        yield move.astype(np.float32), fix

                    if yy == y_bound:
                        break
                    y += stride

                if xx == x_bound:
                    break
                x += stride

    def augment(self, moving, fixed):
        imglab = tf.concat([moving, fixed], axis=-1)
        imglab = tf.image.random_flip_left_right(imglab)

        moving = imglab[:, :, 0:3]
        fixed = imglab[:, :, 3:6]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        moving = tf.image.rot90(moving, k=random_number)
        fixed = tf.image.rot90(fixed, k=random_number)

        return moving, fixed


# ======================= data splitter ==========================

def Her2data_splitter(config):
    """Split HER2 dataset into train, validation, and test sets."""
    all_images = set(glob.glob(config.image_dir + '/*.mat'))

    valid_images = []
    for case in config.valid_cases:
        valid_images += glob.glob(config.image_dir + '/*' + case + '*.mat')

    test_images = []
    for case in config.test_cases:
        test_images += glob.glob(config.image_dir + '/*' + case + '*.mat')

    train_images = list(all_images - set(valid_images) - set(test_images))

    return train_images, valid_images, test_images