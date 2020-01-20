
import os

import tensorflow as tf
import numpy as np
import pandas as pd

from callbacks import SavingCallback, LogCallback, TrainingImageSavingCallback, BestValLossCallback
from dataset import get_autoencoder_dataset_from_splitted_folders
from model.autoencoder import build_encoder, build_decoder
from setup_logging import get_logger
from utils import get_config_parameters

tf.enable_eager_execution()


batch_sizes = [32, 64, 128]
latent_sizes = [128, 256, 512, 1024]
filterss = [(128, 256, 512, 512),
            (64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (64, 128, 256, 512, 512),
            (16, 32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512, 512)]


params = get_config_parameters()
hyperparam_dir = params.model_dir

if not os.path.isdir(hyperparam_dir):
    os.makedirs(hyperparam_dir)

for batch_size in batch_sizes:
    for latent_size in latent_sizes:
        for filters in filterss:

            # identifier for experiment
            model_name = f'hyperp_B{batch_size}_L{latent_size}_F'
            for filter in filters:
                model_name += '_' + str(filter)

            summary_csv_file_path = os.path.join(hyperparam_dir, 'summary_csv_file.csv')
            # check if this experiment is tried before
            if os.path.isfile(summary_csv_file_path):
                summary_csv_file = pd.read_csv(summary_csv_file_path)
                previous_experiments = summary_csv_file['identifier'].tolist()
                if model_name in previous_experiments:
                    print('passing the experiment', model_name)
                    continue


            # TRAINING

            # parameter overwriting
            params.model_dir = os.path.join(hyperparam_dir, model_name)
            params.filters = filters
            params.latent_size = latent_size
            params.batch_size = batch_size

            if not os.path.isdir(params.model_dir):
                os.makedirs(params.model_dir)

            training_progress_images_dir = os.path.join(params.model_dir, 'training_images')
            if not os.path.isdir(training_progress_images_dir):
                os.makedirs(training_progress_images_dir)

            train_ds, val_ds, test_ds = get_autoencoder_dataset_from_splitted_folders(params=params)

            encoder = build_encoder(input_shape=params.input_shape, output_shape=params.latent_size,
                                    filters=params.filters, kernel_size=params.kernel_size,
                                    pool_size=params.pool_size, batch_normalization=params.batch_normalization,
                                    activation=tf.keras.activations.relu,
                                    name='my_encoder')
            decoder = build_decoder(input_shape=params.latent_size, output_shape=params.input_shape,
                                    filters=tuple(reversed(list(params.filters))),
                                    kernel_size=params.kernel_size, batch_normalization=params.batch_normalization,
                                    activation=tf.keras.activations.relu,
                                    name='my_decoder')

            logger_t = get_logger(os.path.join(params.model_dir, 'train_logs.log'), model_name)

            encoder.summary(print_fn=logger_t.info)
            decoder.summary(print_fn=logger_t.info)

            inputs = tf.keras.Input(shape=params.input_shape)
            x = encoder(inputs)
            outputs = decoder(x)
            auto_encoder = tf.keras.models.Model(inputs, outputs)

            auto_encoder.summary(print_fn=logger_t.info)

            auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.lr),
                                 loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])

            best_val_loss_callback = BestValLossCallback()

            callbacks = [tf.keras.callbacks.TensorBoard(log_dir=params.model_dir, update_freq=params.summary_interval),
                         SavingCallback(m_save_interval=params.save_checkpoint_interval, m_encoder=encoder,
                                        m_decoder=decoder,
                                        m_model_dir=params.model_dir),
                         LogCallback(log_file_path=None, logger=logger_t),
                         TrainingImageSavingCallback(inference_image_ds=val_ds.take(1),
                                                     save_dir=training_progress_images_dir),
                         best_val_loss_callback]

            try:
                auto_encoder.fit(train_ds, epochs=params.n_training_epochs, verbose=1, validation_data=val_ds,
                                 callbacks=callbacks)
            except:
                print('RESOURCE EXHAUSTED! continuing')
                continue

            print('Congrats. Training done!')

            # SUMMARY
            summary_row = {'identifier': model_name,
                           'loss': best_val_loss_callback.loss,
                           'val_loss': best_val_loss_callback.val_loss,
                           'best_val_loss': best_val_loss_callback.best_val_loss}
            summary_row_df = pd.DataFrame.from_dict([summary_row])
            if os.path.isfile(summary_csv_file_path):
                summary_csv_file = pd.read_csv(summary_csv_file_path, )
                summary_csv_file = summary_csv_file.append(summary_row_df, ignore_index=True)
                summary_csv_file.to_csv(summary_csv_file_path, index=False)
            else:
                summary_row_df.to_csv(summary_csv_file_path, index=False)
