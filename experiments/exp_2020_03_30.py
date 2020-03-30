
import tensorflow as tf

from datasets.mnist_dataset import get_mnist_dataset

"""
This is a toy example for GAN using MNIST dataset
"""


ds_train, ds_test = get_mnist_dataset(use_colab=False)

print('done')