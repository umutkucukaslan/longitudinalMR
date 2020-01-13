import numpy as np
import configparser
import os
import glob
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, batch_size=32):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def get_autoencoder_dataset():
    """
    This function creates tf.data.Dataset objects for training, validation and test using slices.
    :return: train_ds, val_ds, test_ds
    """
    config = configparser.ConfigParser()
    config.read("./config.ini")
    dataset_path = config['Dataset'].get('dataset_path')
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

    train_ds = prepare_for_training(labeled_train_ds, cache=False, shuffle_buffer_size=10000, batch_size=batch_size)
    val_ds = prepare_for_training(labeled_val_ds, cache=False, shuffle_buffer_size=10000, batch_size=batch_size)
    test_ds = labeled_test_ds.batch(batch_size=batch_size)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":

    tf.enable_eager_execution()

    def show_batch(image_batch):
        plt.figure(figsize=(30, 30))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(np.squeeze(image_batch[n]), cmap='gray')
            plt.axis('off')
        plt.show()

    train_ds, val_ds, test_ds = get_autoencoder_dataset()
    image_batch, label_batch = next(iter(train_ds))
    show_batch(image_batch=image_batch)
    show_batch(image_batch=label_batch)

