import tensorflow as tf
from dataset import get_train_dataset
from tensorflow.python.client import device_lib


mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)