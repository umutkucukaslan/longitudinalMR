import sys

import numpy as np
import configparser
import os
import glob
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from setup_logging import logger


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, batch_size=32, repeat=True):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def get_autoencoder_dataset():
    """
    This function creates tf.data.Dataset objects for training, validation and test using slices.
    This shuffles all the data on its own.
    :return: train_ds, val_ds, test_ds
    """

    logger.info("Preparing the dataset...")

    config = configparser.ConfigParser()
    config.read("./config.ini")
    if config['Environment'].get('running_machine') == 'colab':
        dataset_path = config['Dataset'].get('dataset_path_colab')
    else:
        dataset_path = config['Dataset'].get('dataset_path_computer')
    val_split_rate = config['Dataset'].getfloat('val_split_rate')
    test_split_rate = config['Dataset'].getfloat('test_split_rate')
    batch_size = config['Dataset'].getint('batch_size')

    # CN, MCI and AD images as list of image paths
    cn_images = glob.glob(os.path.join(dataset_path, 'CN/*/*/slice_*.png'))
    random.shuffle(cn_images)
    mci_images = glob.glob(os.path.join(dataset_path, 'MCI/*/*/slice_*.png'))
    random.shuffle(mci_images)
    ad_images = glob.glob(os.path.join(dataset_path, 'AD/*/*/slice_*.png'))
    random.shuffle(ad_images)

    logger.info('There are {} images (CN: {}, MCI: {}, AD: {}) in the dataset.'.format ((len(cn_images)+len(mci_images)+len(ad_images)), len(cn_images), len(mci_images), len(ad_images)))
    train = []
    val = []
    test = []

    def distribute_images(images, val_split_rate, test_split_rate, train, val, test):
        val += images[: int(len(images) * val_split_rate)]
        test += images[int(len(images) * val_split_rate): int(len(images) * val_split_rate) + int(
            len(images) * test_split_rate)]
        train += images[int(len(images) * val_split_rate) + int(len(images) * test_split_rate):]
        return train, val, test

    # Add CN-MCI-AD images to train, val, test
    train, val, test = distribute_images(images=cn_images, val_split_rate=val_split_rate,
                                         test_split_rate=test_split_rate, train=train, val=val, test=test)

    train, val, test = distribute_images(images=mci_images, val_split_rate=val_split_rate,
                                         test_split_rate=test_split_rate, train=train, val=val, test=test)

    train, val, test = distribute_images(images=ad_images, val_split_rate=val_split_rate,
                                         test_split_rate=test_split_rate, train=train, val=val, test=test)

    logger.info('Images divided in train ({}), val ({}) and test ({}) categories.'.format(len(train), len(val), len(test)))

    train_list_ds = tf.data.Dataset.from_tensor_slices(train)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test)
    logger.info('List datasets were created')

    def decode_png_img(img, num_channel=1):
        img = tf.io.decode_png(img, channels=num_channel)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = decode_png_img(img)
        return img, img

    labeled_train_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labeled_val_ds = val_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labeled_test_ds = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    logger.info('Labelled processed datasets were created.')

    train_ds = prepare_for_training(labeled_train_ds, cache=False, shuffle_buffer_size=1000, batch_size=batch_size,
                                    repeat=False)
    val_ds = labeled_val_ds.batch(batch_size=batch_size)
    test_ds = labeled_test_ds.batch(batch_size=batch_size)
    logger.info('train_ds, val_ds and test_ds are ready.')

    return train_ds, val_ds, test_ds


def get_autoencoder_dataset_from_splitted_folders(params):
    """
    This function creates tf.data.Dataset objects for training, validation and test from already splitted folders.

    :return: train_ds, val_ds, test_ds
    """

    logger.info("Preparing the dataset...")

    dataset_path = params.dataset_path
    batch_size = params.batch_size

    train = glob.glob(os.path.join(dataset_path, '*/train/*/*/slice_*.png'))
    random.shuffle(train)
    val = glob.glob(os.path.join(dataset_path, '*/val/*/*/slice_*.png'))
    random.shuffle(val)
    test = glob.glob(os.path.join(dataset_path, '*/test/*/*/slice_*.png'))
    random.shuffle(test)

    logger.info('Images found in train ({}), val ({}) and test ({}) categories.'.format(len(train), len(val), len(test)))

    train_list_ds = tf.data.Dataset.from_tensor_slices(train)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test)

    def decode_png_img(img, num_channel=1):
        img = tf.io.decode_png(img, channels=num_channel)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = decode_png_img(img)
        return img, img

    labeled_train_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labeled_val_ds = val_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labeled_test_ds = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = prepare_for_training(labeled_train_ds, cache=False, shuffle_buffer_size=1000, batch_size=batch_size,
                                    repeat=False)
    val_ds = labeled_val_ds.batch(batch_size=batch_size)
    test_ds = labeled_test_ds.batch(batch_size=batch_size)
    logger.info('Datasets (train_ds, val_ds and test_ds) are ready.')

    return train_ds, val_ds, test_ds


def get_fake_autoencoder_dataset(n_samples=100, shape=(256, 256, 1), batch_size=32, repeat=True, interval=(0, 1)):
    train = np.random.rand(n_samples, shape[0], shape[1], shape[2]) * (interval[1] - interval[0]) - interval[0]
    ds = tf.data.Dataset.from_tensor_slices((train, train))
    if repeat:
        ds = ds.repeat().batch(batch_size=batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == "__main__":

    tf.enable_eager_execution()

    import time
    default_timeit_steps = 1000

    config = configparser.ConfigParser()
    config.read("./config.ini")
    BATCH_SIZE = config['Dataset'].getint('batch_size')

    def timeit(ds, steps=default_timeit_steps):
        start = time.time()
        it = iter(ds)
        for i in range(steps):
            batch = next(it)
            if i % 10 == 0:
                print('.', end='')
        print()
        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(steps, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))


    def show_batch(image_batch):
        plt.figure(figsize=(30, 30))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(np.squeeze(image_batch[n]), cmap='gray')
            plt.axis('off')
        plt.show()

    train_ds, val_ds, test_ds = get_autoencoder_dataset()

    # timeit(train_ds)

    image_batch, label_batch = next(iter(val_ds))
    show_batch(image_batch=image_batch)
    show_batch(image_batch=label_batch)

