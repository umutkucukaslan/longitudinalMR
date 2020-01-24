
import os
import glob

import cv2
import numpy as np
import tensorflow as tf

from setup_logging import get_logger
from utils import show_image_batch


class SavingCallback(tf.keras.callbacks.Callback):
    def __init__(self, m_save_interval, m_encoder, m_decoder, m_model_dir, max_to_keep=2):
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
    def __init__(self, logger):
        super(LogCallback, self).__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        msg = 'Epoch: {0:05d}  '.format(epoch)
        for k in logs:
            msg += '{}: {:.4f}  '.format(k, logs[k])
        self.logger.info(msg)


class TrainingImageSavingCallback(tf.keras.callbacks.Callback):
    def __init__(self, inference_image_ds, save_dir, grid_size=5):
        super(TrainingImageSavingCallback, self).__init__()
        self.image_ds = inference_image_ds
        self.save_dir = save_dir
        self.grid_size = grid_size

    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(self.image_ds)
        img = show_image_batch(res, grid_size=self.grid_size)
        img = np.asarray(img * 255.0, dtype=np.uint8)
        file_path = os.path.join(self.save_dir, 'epoch_{0:05d}.jpg'.format(epoch))
        cv2.imwrite(file_path, img)
        # encoded_img = tf.io.encode_jpeg(img)
        # tf.io.write_file(file_path, encoded_img)


class BestValLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestValLossCallback, self).__init__()
        self.best_val_loss = np.Inf
        self.val_loss = np.Inf
        self.loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
        self.loss = logs['loss']
        self.val_loss = logs['val_loss']