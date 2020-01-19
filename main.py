import configparser
import glob
import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from dataset import get_autoencoder_dataset_from_splitted_folders
from model.autoencoder import build_encoder, build_decoder
from setup_logging import logger


tf.enable_eager_execution()

train_ds, val_ds, test_ds = get_autoencoder_dataset_from_splitted_folders()


# Parameters
config = configparser.ConfigParser()
config.read('./config.ini')
batch_size = config['Dataset'].getint('batch_size')
input_shape = (256, 256, 1)
latent_size = 1024
filters = (8, 16, 32, 64, 128, 256)
kernel_size = 3
pool_size = (2, 2)
batch_normalization = True
if config['Environment'].get('running_machine') == 'colab':
    model_dir = config['Train'].get('model_dir_colab')
else:
    model_dir = config['Train'].get('model_dir_computer')
summary_interval = config['Train'].getint('summary_interval')
save_interval = config['Train'].getint('save_interval')
learning_rate = config['Train'].getfloat('lr')
num_training_epochs = config['Train'].getint('training_epochs')

encoder = build_encoder(input_shape=input_shape, output_shape=latent_size, filters=filters, kernel_size=kernel_size,
                        pool_size=pool_size, batch_normalization=batch_normalization, activation=tf.nn.leaky_relu,
                        name='my_encoder')
decoder = build_decoder(input_shape=latent_size, output_shape=input_shape, filters=tuple(reversed(list(filters))),
                        kernel_size=kernel_size, batch_normalization=batch_normalization, activation=tf.nn.leaky_relu,
                        name='my_decoder')

encoder.summary()
decoder.summary()

inputs = tf.keras.Input(shape=input_shape)
x = encoder(inputs)
outputs = decoder(x)
auto_encoder = tf.keras.models.Model(inputs, outputs)

auto_encoder.summary()

auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])


class SavingCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_interval, encoder, decoder, model_dir, max_to_keep=5):
        super(SavingCallback, self).__init__()
        self.save_interval = save_interval
        self.encoder = encoder
        self.decoder = decoder
        self.model_dir = model_dir
        self.max_to_keep = max_to_keep
        self.best_val_loss = 1e12

    def delete_oldest_if_necessary(self):
        encoders = sorted(glob.glob(os.path.join(self.model_dir, 'encoder-*')))
        if len(encoders) >= self.max_to_keep:
            os.remove(encoders[0])
        decoders = sorted(glob.glob(os.path.join(self.model_dir, 'decoder-*')))
        if len(decoders) >= self.max_to_keep:
            os.remove(decoders[0])
        auto_encoders = sorted(glob.glob(os.path.join(self.model_dir, 'auto_encoder-*')))
        if len(auto_encoders) >= self.max_to_keep:
            os.remove(auto_encoders[0])

    def on_epoch_end(self, epoch, logs=None):
        if self.best_val_loss > logs['val_loss']:
            self.best_val_loss = logs['val_loss']
            self.encoder.save(os.path.join(self.model_dir, 'encoder_best'))
            self.decoder.save(os.path.join(self.model_dir, 'decoder_best'))
            self.model.save(os.path.join(self.model_dir, 'auto_encoder_best'))

        if (epoch // self.save_interval) * self.save_interval == epoch:
            self.delete_oldest_if_necessary()
            self.encoder.save(os.path.join(self.model_dir, 'encoder-{0:05d}'.format(epoch)))
            self.decoder.save(os.path.join(self.model_dir, 'decoder-{0:05d}'.format(epoch)))
            self.model.save(os.path.join(self.model_dir, 'auto_encoder-{0:05d}'.format(epoch)))


callbacks=[tf.keras.callbacks.TensorBoard(log_dir=model_dir, update_freq=summary_interval),
           SavingCallback(save_interval=save_interval, encoder=encoder, decoder=decoder, model_dir=model_dir)]

auto_encoder.fit(train_ds, epochs=num_training_epochs, verbose=1, validation_data=val_ds, callbacks=callbacks)

print('Congrats. Training done!')
