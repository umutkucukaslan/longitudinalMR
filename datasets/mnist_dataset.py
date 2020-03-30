
import os

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


def get_mnist_dataset(use_colab=False):
    """
    Returns train and test datasets (tf.data.Dataset objects) for MNIST handwritten digits dataset.

    :param use_colab: Use true when using colab, if true, saves the dataset into /datasets/tensorflow_datasets
                        otherwise, saves to my thesis folder under dataset/other_datasets
    :return: train and test dataset objects
    """
    if use_colab:
        data_dir = '/datasets/tensorflow_datasets'
    else:
        data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/other_datasets'

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    ds_train = tfds.load('mnist', split='train', data_dir=data_dir)
    ds_test = tfds.load('mnist', split='test', data_dir=data_dir)

    return ds_train, ds_test


if __name__ == '__main__':
    print('main module scripts')
    ds_train, ds_test = get_mnist_dataset(use_colab=False)

    for example in ds_train.take(1):
        image, label = example["image"], example["label"]
        plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
        plt.title("Label: %d" % label.numpy())
        plt.axis = 'off'
        plt.show()

