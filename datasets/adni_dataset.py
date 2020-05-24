import glob
import os
import random

import tensorflow as tf


def get_adni_dataset(folder_name='processed_data', machine='none', return_two_trains=False, return_raw_dataset=False):
    """
    train, val, test datasets from processed_data folder
    Images are normalized to [0, 1] interval
    dtype: tf.float32

    :return: train_ds, val_ds, test_ds
    """

    if machine == 'colab':
        data_dir = os.path.join('/content', folder_name)
    elif machine == 'cloud':
        data_dir = os.path.join('/home/umutkucukaslan/data', folder_name)
    else:
        data_dir = os.path.join('/Users/umutkucukaslan/Desktop/thesis/dataset', folder_name)

    train = glob.glob(os.path.join(data_dir, 'train/*/*/slice_*.png'))
    val = glob.glob(os.path.join(data_dir, 'val/*/*/slice_*.png'))
    test = glob.glob(os.path.join(data_dir, 'test/*/*/slice_*.png'))

    train_list_ds = tf.data.Dataset.from_tensor_slices(train)
    train_list_ds2 = tf.data.Dataset.from_tensor_slices(train)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test)

    if return_raw_dataset:
        if return_two_trains:
            return train_list_ds, train_list_ds2, val_list_ds, test_list_ds
        return train_list_ds, train_list_ds2, val_list_ds, test_list_ds

    def decode_img(img, num_channel=1):
        img = tf.io.decode_png(img, channels=num_channel)
        img = tf.cast(img, tf.float32)
        img = img / 256.0
        return img

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img

    train_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds2 = train_list_ds2.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if return_two_trains:
        return train_ds, train_ds2, val_ds, test_ds

    return train_ds, val_ds, test_ds

