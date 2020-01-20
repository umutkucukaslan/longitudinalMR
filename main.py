import configparser
import glob
import os
import sys

import tensorflow as tf
import numpy as np

from callbacks import SavingCallback, LogCallback, TrainingImageSavingCallback
from dataset import get_autoencoder_dataset_from_splitted_folders
from model.autoencoder import build_encoder, build_decoder
from utils import get_config_parameters

tf.enable_eager_execution()

train_ds, val_ds, test_ds = get_autoencoder_dataset_from_splitted_folders()


# Parameters
params = get_config_parameters()

if not os.path.isdir(params.model_dir):
    os.makedirs(params.model_dir)

training_progress_images_dir = os.path.join(params.model_dir, 'training_images')
if not os.path.isdir(training_progress_images_dir):
    os.makedirs(training_progress_images_dir)


encoder = build_encoder(input_shape=params.input_shape, output_shape=params.latent_size, filters=params.filters, kernel_size=params.kernel_size,
                        pool_size=params.pool_size, batch_normalization=params.batch_normalization, activation=tf.keras.activations.relu,
                        name='my_encoder')
decoder = build_decoder(input_shape=params.latent_size, output_shape=params.input_shape, filters=tuple(reversed(list(params.filters))),
                        kernel_size=params.kernel_size, batch_normalization=params.batch_normalization, activation=tf.keras.activations.relu,
                        name='my_decoder')

encoder.summary()
decoder.summary()

inputs = tf.keras.Input(shape=params.input_shape)
x = encoder(inputs)
outputs = decoder(x)
auto_encoder = tf.keras.models.Model(inputs, outputs)

auto_encoder.summary()

auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.lr),
                     loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])


callbacks=[tf.keras.callbacks.TensorBoard(log_dir=params.model_dir, update_freq=params.summary_interval),
           SavingCallback(m_save_interval=params.save_checkpoint_interval, m_encoder=encoder, m_decoder=decoder, m_model_dir=params.model_dir),
           LogCallback(log_file_path=os.path.join(params.model_dir, 'train_logs.log')),
           TrainingImageSavingCallback(inference_image_ds=train_ds.take(1), save_dir=training_progress_images_dir)]

auto_encoder.fit(train_ds.take(3), epochs=params.n_training_epochs, verbose=1, validation_data=val_ds.take(2), callbacks=callbacks)

print('Congrats. Training done!')
