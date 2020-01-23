import configparser
import ast

import numpy as np


def show_image_batch(image_batch, grid_size=5):
    image_batch = np.clip(image_batch, 0, 1)
    n_images = min(image_batch.shape[0], grid_size * grid_size)
    h, w, c = image_batch.shape[1], image_batch.shape[2], image_batch.shape[3]
    n_rows = n_cols = int(np.sqrt(n_images))
    if (n_cols + 1) * n_rows <= n_images:
        n_cols = n_cols + 1
    im = np.zeros((n_rows * h, n_cols * w, 1))
    im_idx = 0
    for row_idx in range(0, n_rows * h, h):
        for col_idx in range(0, n_cols * w, w):
            im[row_idx: row_idx + h, col_idx: col_idx + w] = image_batch[im_idx]
            im_idx += 1
    return im


class Parameters:
    def __init__(self,
                 running_machine,
                 dataset_path,
                 batch_size,
                 model_dir,
                 n_training_epochs,
                 summary_interval,
                 save_checkpoint_interval,
                 lr,
                 input_shape,
                 latent_size,
                 filters,
                 kernel_size,
                 pool_size,
                 batch_normalization,
                 training_summary_csv
                 ):
        self.training_summary_csv = training_summary_csv
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.batch_normalization = batch_normalization
        self.running_machine = running_machine
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.n_training_epochs = n_training_epochs
        self.summary_interval = summary_interval
        self.save_checkpoint_interval = save_checkpoint_interval
        self.lr = lr


def get_config_parameters():
    config = configparser.ConfigParser()
    config.read('./config.ini')

    running_machine = config['Environment'].get('running_machine')
    if running_machine == 'colab':
        dataset_path = config['Dataset'].get('dataset_path_colab')
        model_dir = config['Train'].get('model_dir_colab')
    else:
        dataset_path = config['Dataset'].get('dataset_path_computer')
        model_dir = config['Train'].get('model_dir_computer')

    batch_size = config['Dataset'].getint('batch_size')
    n_training_epochs = config['Train'].getint('n_training_epochs')
    summary_interval = config['Train'].getint('summary_interval')
    save_checkpoint_interval = config['Train'].getint('save_checkpoint_interval')
    lr = config['Train'].getfloat('lr')

    input_shape = ast.literal_eval(config['Model'].get('input_shape'))
    latent_size = config['Model'].getint('latent_size')
    filters = ast.literal_eval(config['Model'].get('filters'))
    kernel_size = config['Model'].getint('kernel_size')
    pool_size = ast.literal_eval(config['Model'].get('pool_size'))
    batch_normalization = config['Model'].getboolean('batch_normalization')

    training_summary_csv = config['Logging'].get('training_summary_csv')

    return Parameters(running_machine,
                      dataset_path,
                      batch_size,
                      model_dir,
                      n_training_epochs,
                      summary_interval,
                      save_checkpoint_interval,
                      lr,
                      input_shape,
                      latent_size,
                      filters,
                      kernel_size,
                      pool_size,
                      batch_normalization,
                      training_summary_csv)
