import configparser
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from dataset import get_autoencoder_dataset_from_splitted_folders
from model.autoencoder import build_encoder, build_decoder
from setup_logging import logger, get_logger


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
    log_file_path = config['Logging'].get('log_file_path_colab')
else:
    model_dir = config['Train'].get('model_dir_computer')
    log_file_path = config['Logging'].get('log_file_path_computer')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

training_images_dir = os.path.join(model_dir, 'training_images')
if not os.path.isdir(training_images_dir):
    os.makedirs(training_images_dir)

summary_interval = config['Train'].getint('summary_interval')
save_interval = config['Train'].getint('save_interval')
learning_rate = config['Train'].getfloat('lr')
num_training_epochs = config['Train'].getint('training_epochs')

encoder = build_encoder(input_shape=input_shape, output_shape=latent_size, filters=filters, kernel_size=kernel_size,
                        pool_size=pool_size, batch_normalization=batch_normalization, activation=tf.keras.activations.relu,
                        name='my_encoder')
decoder = build_decoder(input_shape=latent_size, output_shape=input_shape, filters=tuple(reversed(list(filters))),
                        kernel_size=kernel_size, batch_normalization=batch_normalization, activation=tf.keras.activations.relu,
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
    def __init__(self, m_save_interval, m_encoder, m_decoder, m_model_dir, max_to_keep=5):
        super(SavingCallback, self).__init__()
        self.save_interval = m_save_interval
        self.encoder = m_encoder
        self.decoder = m_decoder
        self.model_dir = m_model_dir
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
            for enc in glob.glob(os.path.join(self.model_dir, 'encoder_best*')):
                os.remove(enc)
            for dec in glob.glob(os.path.join(self.model_dir, 'decoder_best*')):
                os.remove(dec)
            for ae in glob.glob(os.path.join(self.model_dir, 'auto_encoder_best*')):
                os.remove(ae)
            self.encoder.save(os.path.join(self.model_dir, 'encoder_best_{}.h5'.format(epoch)))
            self.decoder.save(os.path.join(self.model_dir, 'decoder_best_{}.h5'.format(epoch)))
            self.model.save(os.path.join(self.model_dir, 'auto_encoder_best_{}.h5'.format(epoch)))

        if (epoch // self.save_interval) * self.save_interval == epoch:
            self.delete_oldest_if_necessary()
            self.encoder.save(os.path.join(self.model_dir, 'encoder-{0:05d}.h5'.format(epoch)))
            self.decoder.save(os.path.join(self.model_dir, 'decoder-{0:05d}.h5'.format(epoch)))
            self.model.save(os.path.join(self.model_dir, 'auto_encoder-{0:05d}.h5'.format(epoch)))


class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file_path):
        super(LogCallback, self).__init__()
        self.log_file_path = log_file_path
        self.logger = get_logger(log_file_path, 'training_logger')

    def on_epoch_end(self, epoch, logs=None):
        msg = 'Epoch: {0:05d}  '.format(epoch)
        for k in logs:
            msg += '{}: {:.4f}  '.format(k, logs[k])
        self.logger.info(msg)


class ImageResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self, inference_image_ds, save_dir):
        super(ImageResultsCallback, self).__init__()
        self.image_ds = inference_image_ds
        self.save_dir = save_dir

    def show_image_batch(self, image_batch):
        image_batch = np.clip(image_batch, 0, 1)
        n_images = image_batch.shape[0]
        h, w, c = image_batch.shape[1], image_batch.shape[2], image_batch.shape[3]
        n_rows = n_cols = int(np.sqrt(n_images))
        im = np.zeros((n_rows * h, n_cols * w, 1))
        im_idx = 0
        for row_idx in range(0, n_rows * h, h):
            for col_idx in range(0, n_cols * w, w):
                im[row_idx: row_idx + h, col_idx: col_idx + w] = image_batch[im_idx]
                im_idx += 1
        return im

    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(self.image_ds)
        img = self.show_image_batch(res)
        img = np.asarray(img * 255.0, dtype=np.uint8)
        encoded_img = tf.io.encode_jpeg(img)
        file_path = os.path.join(self.save_dir, 'epoch_{0:05d}.jpg'.format(epoch))
        tf.io.write_file(file_path, encoded_img)


callbacks=[tf.keras.callbacks.TensorBoard(log_dir=model_dir, update_freq=summary_interval),
           SavingCallback(m_save_interval=save_interval, m_encoder=encoder, m_decoder=decoder, m_model_dir=model_dir),
           LogCallback(log_file_path=log_file_path),
           ImageResultsCallback(inference_image_ds=train_ds.take(1), save_dir=training_images_dir)]

auto_encoder.fit(train_ds, epochs=num_training_epochs, verbose=1, validation_data=val_ds, callbacks=callbacks)

print('Congrats. Training done!')
