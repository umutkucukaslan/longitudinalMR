import glob
import os
import random

import tensorflow as tf

from datasets.longitudinal_dataset import LongitudinalDataset


def get_adni_dataset(
    folder_name="processed_data",
    machine="none",
    return_two_trains=False,
    return_raw_dataset=False,
):
    """
    train, val, test datasets from processed_data folder
    Images are normalized to [0, 1] interval
    dtype: tf.float32

    :return: train_ds, val_ds, test_ds
    """

    if machine == "colab":
        data_dir = os.path.join("/content", folder_name)
    elif machine == "cloud":
        data_dir = os.path.join("/home/umutkucukaslan/data", folder_name)
    else:
        data_dir = os.path.join(
            "/Users/umutkucukaslan/Desktop/thesis/dataset", folder_name
        )

    train = glob.glob(os.path.join(data_dir, "train/*/*/slice_*.png"))
    val = glob.glob(os.path.join(data_dir, "val/*/*/slice_*.png"))
    test = glob.glob(os.path.join(data_dir, "test/*/*/slice_*.png"))

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

    train_ds = train_list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds2 = train_list_ds2.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if return_two_trains:
        return train_ds, train_ds2, val_ds, test_ds

    return train_ds, val_ds, test_ds


def get_triplets_adni_15t_dataset(
    folder_name="training_data_15T_192x160_4slices", machine="none",
):
    if machine == "colab":
        data_dir = os.path.join("/content", folder_name)
    elif machine == "cloud":
        data_dir = os.path.join("/home/umutkucukaslan/data", folder_name)
    else:
        data_dir = os.path.join(
            "/Users/umutkucukaslan/Desktop/thesis/dataset", folder_name
        )
    train_data_dir = os.path.join(data_dir, "train")
    val_data_dir = os.path.join(data_dir, "val")
    test_data_dir = os.path.join(data_dir, "test")

    train_long = LongitudinalDataset(data_dir=train_data_dir)
    val_long = LongitudinalDataset(data_dir=val_data_dir)
    test_long = LongitudinalDataset(data_dir=test_data_dir)

    train_triplets = (
        train_long.get_ad_image_triplets()
        + train_long.get_mci_image_triplets()
        + train_long.get_cn_image_triplets()
    )
    val_triplets = (
        val_long.get_ad_image_triplets()
        + val_long.get_mci_image_triplets()
        + val_long.get_cn_image_triplets()
    )
    test_triplets = (
        test_long.get_ad_image_triplets()
        + test_long.get_mci_image_triplets()
        + test_long.get_cn_image_triplets()
    )

    def triplet_list_to_tfdata_style(triplets):
        imgs = ([], [], [])
        days = ([], [], [])
        for triplet in triplets:
            triplet_imgs, triplet_days = triplet
            imgs[0].append(triplet_imgs[0])
            imgs[1].append(triplet_imgs[1])
            imgs[2].append(triplet_imgs[2])
            days[0].append(triplet_days[0])
            days[1].append(triplet_days[1])
            days[2].append(triplet_days[2])
        return {"imgs": imgs, "days": days}

    train_list_ds = tf.data.Dataset.from_tensor_slices(
        triplet_list_to_tfdata_style(train_triplets)
    )
    val_list_ds = tf.data.Dataset.from_tensor_slices(
        triplet_list_to_tfdata_style(val_triplets)
    )
    test_list_ds = tf.data.Dataset.from_tensor_slices(
        triplet_list_to_tfdata_style(test_triplets)
    )

    def decode_img(img, num_channel=1):
        img = tf.io.decode_png(img, channels=num_channel)
        img = tf.cast(img, tf.float32)
        img = img / 256.0
        return img

    def augment_images(imgs):
        augmented_imgs = [x for x in imgs]
        augmented_imgs = [
            tf.image.random_brightness(x, max_delta=0.2) for x in augmented_imgs
        ]
        augmented_imgs = [
            tf.image.random_contrast(x, lower=0.9, upper=1.1) for x in augmented_imgs
        ]
        return augmented_imgs

    def process_triplet(triplet):
        imgs, days = triplet["imgs"], triplet["days"]
        imgs = [tf.io.read_file(x) for x in imgs]
        imgs = [decode_img(x) for x in imgs]
        augmented_imgs = augment_images(imgs)
        days = [tf.cast(x, tf.float32) for x in days]
        return {
            "imgs": tuple(imgs),
            "days": tuple(days),
            "augmented_imgs": tuple(augmented_imgs),
        }

    train_ds = train_list_ds.map(
        process_triplet, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    val_ds = val_list_ds.map(
        process_triplet, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_list_ds.map(
        process_triplet, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return train_ds, val_ds, test_ds
