
import os

import tensorflow as tf
import numpy as np
import pandas as pd

from callbacks import SavingCallback, LogCallback, TrainingImageSavingCallback, BestValLossCallback
from dataset import get_autoencoder_dataset_from_splitted_folders
from model.autoencoder import build_encoder, build_decoder
from setup_logging import get_logger
from utils import get_config_parameters


params = get_config_parameters()
hyperparam_dir = params.model_dir

if not os.path.isdir(hyperparam_dir):
    os.makedirs(hyperparam_dir)



# identifier for experiment
model_name = f'{params.model_name_prefix}__B{params.batch_size}_L{params.latent_size}_F'
for filter in params.filters:
    model_name += '_' + str(filter)

model_name += f'_K{params.kernel_size}'

model_name += '_encoder_without_last_activation'


# TRAINING

# overwrite model dir
params.model_dir = os.path.join(hyperparam_dir, model_name)

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
             LogCallback(logger=logger_t),
             TrainingImageSavingCallback(inference_image_ds=val_ds.take(1),
                                         save_dir=training_progress_images_dir),
             best_val_loss_callback]

try:
    auto_encoder.fit(train_ds, epochs=params.n_training_epochs, verbose=1, validation_data=val_ds,
                     callbacks=callbacks)
except:
    print('RESOURCE EXHAUSTED!')


print('Congrats. Training done!')

# SUMMARY
summary_row = {'identifier': model_name,
               'loss': best_val_loss_callback.loss,
               'val_loss': best_val_loss_callback.val_loss,
               'best_val_loss': best_val_loss_callback.best_val_loss}
summary_row_df = pd.DataFrame.from_dict([summary_row])
if os.path.isfile(params.training_summary_csv):
    summary_csv_file = pd.read_csv(params.training_summary_csv, )
    summary_csv_file = summary_csv_file.append(summary_row_df, ignore_index=True)
    summary_csv_file.to_csv(params.training_summary_csv, index=False)
else:
    summary_row_df.to_csv(params.training_summary_csv, index=False)
