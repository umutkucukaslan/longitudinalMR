import configparser
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from dataset import get_autoencoder_dataset, get_fake_autoencoder_dataset
from model.autoencoder import build_encoder, build_decoder
from setup_logging import logger


tf.enable_eager_execution()

train_ds, val_ds, test_ds = get_autoencoder_dataset()

config = configparser.ConfigParser()
config.read('./config.ini')

batch_size = 32
input_shape = (256, 256, 1)
latent_size = 32
filters = (8, 16, 32)
kernel_size = 3
pool_size = (2, 2)
batch_normalization = False
log_dir = config['Train'].get('log_dir')
# train_ds = get_fake_autoencoder_dataset(100, input_shape, batch_size=batch_size, repeat=False, interval=(0, 1))
# val_ds = get_fake_autoencoder_dataset(100, input_shape, batch_size=batch_size, repeat=False, interval=(0, 1))


encoder = build_encoder(input_shape=input_shape, output_shape=latent_size, filters=filters, kernel_size=kernel_size, pool_size=pool_size, batch_normalization=batch_normalization, name='my_encoder')
decoder = build_decoder(input_shape=latent_size, output_shape=input_shape, filters=tuple(reversed(list(filters))), kernel_size=kernel_size, batch_normalization=batch_normalization, name='my_decoder')

encoder.summary()
decoder.summary()

inputs = tf.keras.Input(shape=input_shape)
x = encoder(inputs)
outputs = decoder(x)
autoencoder = tf.keras.models.Model(inputs, outputs)

autoencoder.summary()

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])

# print(train_ds)
# sys.exit()
autoencoder.fit(train_ds, epochs=10, steps_per_epoch=100, verbose=1, validation_data=val_ds, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10)])


