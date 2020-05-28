import copy
import datetime
import os
import statistics
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.adni_dataset import get_adni_dataset
from model.autoencoder import build_decoder
import model.gan as gan
from model.losses import wgan_gp_loss, vae_loss
from model.vae import build_encoder

"""
Variational Autoencoder

Input:  192x160
PMSD best autoencoder structure. Encoder: Conv(64, 128, 256, 512) + Dense
                                 Decoder: Dense + Deconv(512, 256, 128, 64)
Training:   WGAN-GP loss (GP: gradient penalty)
Loss:       Generator loss = EM distance
            Discriminator loss = EM distance - GP

maybe add layer normalization as recommended in the paper WGAN-GP
"""


RUNTIME = 'colab'   # cloud, colab or none
USE_TPU = False
RESTORE_FROM_CHECKPOINT = True
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
INPUT_WIDTH = 160
INPUT_HEIGHT = 192
INPUT_CHANNEL = 1

DISC_TRAIN_STEPS = 10
LAMBDA_GP = 10
CLIP_DISC_WEIGHT = None    # clip disc weight
CLIP_BY_NORM = None    # clip gradients to this norm or None
CLIP_BY_VALUE = None   # clip gradient to this value or None

EPOCHS = 5000
CHECKPOINT_SAVE_INTERVAL = 5
MAX_TO_KEEP = 5
LR = 1e-4


# set batch size easily
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])


if USE_TPU:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException(
            'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if RUNTIME == 'colab':
    if USE_TPU:
        EXPERIMENT_FOLDER = os.path.join('/content/experiments', EXPERIMENT_NAME)
    else:
        EXPERIMENT_FOLDER = os.path.join('/content/drive/My Drive/experiments', EXPERIMENT_NAME)
elif RUNTIME == 'cloud':
    EXPERIMENT_FOLDER = os.path.join('/home/umutkucukaslan/experiments', EXPERIMENT_NAME)
else:
    EXPERIMENT_FOLDER = os.path.join('/Users/umutkucukaslan/Desktop/thesis/experiments', EXPERIMENT_NAME)

if __name__ == "__main__":
    if not os.path.isdir(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)


def log_print(msg, add_timestamp=False):
    if not isinstance(msg, str):
        msg = str(msg)
    if add_timestamp:
        msg += ' (logged at {})'.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    with open(os.path.join(EXPERIMENT_FOLDER, 'logs.txt'), 'a+') as log_file:
        log_file.write(msg + '\n')

# generator model plot path
GEN_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'gen_model_plot.jpg')
DIS_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'dis_model_plot.jpg')

# folder to save generated test images during training
if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, 'figures')):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'figures'))

# BUILD GENERATOR
# encoder
filters = (64, 128, 256, 512)
output_shape = 512
kernel_size = 5
batch_norm = False

encoder = build_encoder(
    input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
    output_shape=output_shape,
    filters=filters,
    kernel_size=kernel_size,
    batch_normalization=batch_norm,
    activation=tf.nn.relu,
    name='encoder')

decoder = build_decoder(
    input_shape=output_shape,
    output_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
    filters=tuple(reversed(filters)),
    kernel_size=kernel_size,
    batch_normalization=batch_norm,
    activation=tf.nn.relu,
    name='decoder')

# generator definition
input_shape = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL)
gen_in = tf.keras.Input(shape=input_shape, name='image_input')
gen_random = tf.keras.Input(shape=(output_shape), name='random_var')
m, s = encoder(gen_in)  # mean and std for the distribution
latent = m + gen_random * s
out_im = decoder(latent)

generator = tf.keras.Model(inputs=[gen_in, gen_random], outputs=[out_im, m, s], name='generator')


if __name__ == "__main__":
    encoder.summary()
    encoder.summary(print_fn=log_print)
    decoder.summary()
    decoder.summary(print_fn=log_print)
    generator.summary()
    generator.summary(print_fn=log_print)
    tf.keras.utils.plot_model(generator, to_file=GEN_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=True)


# # DISCRIMINATOR
# discriminator = gan.get_discriminator(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL), batch_norm=False)
# if __name__ == "__main__":
#     discriminator.summary()
#     discriminator.summary(print_fn=log_print)
#     tf.keras.utils.plot_model(discriminator, to_file=DIS_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=False)


# =================

# optimizers
generator_optimizer = tf.optimizers.Adam(LR, beta_1=0.5)
# generator_optimizer = tf.optimizers.RMSprop(learning_rate=LR)
# discriminator_optimizer = tf.optimizers.Adam(LR, beta_1=0.5)
# discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=LR)

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 # discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 # discriminator=discriminator
                                 )
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)
    # checkpoint.restore('/Users/umutkucukaslan/Desktop/thesis/experiments/exp_2020_04_28/checkpoints/ckpt-1')

if manager.latest_checkpoint:
    log_print("Restored from {}".format(manager.latest_checkpoint))
else:
    log_print("Initializing from scratch.")

initial_epoch = checkpoint.epoch.numpy() + 1


def get_encoder_decoder_generator_discriminator(return_experiment_folder=True):
    """
    This function returns the constructed and restored (if possible) sub-models that are constructed in this experiment

    :return: encoder, decoder, generator, discriminator
    """
    if return_experiment_folder:
        return encoder, decoder, generator, None, EXPERIMENT_FOLDER
    return encoder, decoder, generator, None, EXPERIMENT_FOLDER


if __name__ == "__main__":

    # summary file writer for tensorboard
    log_dir = os.path.join(EXPERIMENT_FOLDER, 'logs')
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))


    # DATASET
    train_ds, train_ds2, val_ds, test_ds = get_adni_dataset(folder_name='processed_data_192x160', machine=RUNTIME, return_two_trains=True)

    train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)


    # for example in train_ds.take(5):
    #     plt.imshow(np.squeeze(example.numpy()[0]), cmap=plt.get_cmap('gray'))
    #     # plt.show()
    #     img = example.numpy()
    #     print('mean value: ', img.mean())
    #     print('max value : ', img.max())
    #     print('min value : ', img.min())
    # exit()


    # TRAINING


    def train_step(input_image, target, train_generator=True, train_discriminator=True):

        with tf.GradientTape() as gen_tape:
            batch_size = input_image.shape[0]
            random_var = tf.random.normal([batch_size, output_shape])
            generated_image, latent_mean, latent_std = generator([input_image, random_var], training=True)
            total_loss, reconst_loss, kl_loss = vae_loss(input_image, target, latent_mean, latent_std)

            # gen_loss, disc_loss, gp_loss = wgan_gp_loss(discriminator, target, generated_image, LAMBDA_GP)

        if train_generator:
            generator_gradients = gen_tape.gradient(total_loss, generator.trainable_variables)
            if CLIP_BY_NORM is not None:
                generator_gradients = [tf.clip_by_norm(t, CLIP_BY_NORM) for t in generator_gradients]
            if CLIP_BY_VALUE is not None:
                generator_gradients = [tf.clip_by_value(t, -CLIP_BY_VALUE, CLIP_BY_VALUE) for t in generator_gradients]
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        # if train_discriminator:
        #     discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        #     if CLIP_BY_NORM is not None:
        #         discriminator_gradients = [tf.clip_by_norm(t, CLIP_BY_NORM) for t in discriminator_gradients]
        #     if CLIP_BY_VALUE is not None:
        #         discriminator_gradients = [tf.clip_by_value(t, -CLIP_BY_VALUE, CLIP_BY_VALUE) for t in discriminator_gradients]
        #     discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # if CLIP_DISC_WEIGHT:
        #     for disc_var in discriminator.trainable_variables:
        #         disc_var.assign(tf.clip_by_value(disc_var, -CLIP_DISC_WEIGHT, CLIP_DISC_WEIGHT))

        return total_loss, reconst_loss, kl_loss


    def eval_step(input_image, target):
        batch_size = input_image.shape[0]
        random_var = tf.random.normal([batch_size, output_shape])
        generated_image, latent_mean, latent_std = generator([input_image, random_var], training=True)
        total_loss, reconst_loss, kl_loss = vae_loss(input_image, target, latent_mean, latent_std)

        return total_loss, reconst_loss, kl_loss


    def generate_images(model, test_input, path=None, show=True):
        if test_input.ndim < 4:
            test_input = np.expand_dims(test_input, axis=0)

        random_val = np.zeros_like(test_input)
        prediction, _, _ = model([test_input, random_val])
        if isinstance(test_input, tf.Tensor):
            display_list = [np.squeeze(test_input.numpy()[0, :, :, 0]), np.squeeze(prediction.numpy()[0, :, :, 0])]
        else:
            display_list = [np.squeeze(test_input[0, :, :, 0]), np.squeeze(prediction[0, :, :, 0])]
        title = ['Input Image', 'Reconstructed Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap=plt.get_cmap('gray'))
            plt.axis('off')
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()


    def fit(train_ds, num_epochs, val_ds, test_ds, train_ds_images, initial_epoch=0):

        assert initial_epoch < num_epochs
        test_ds = iter(test_ds)
        train_ds_images = iter(train_ds_images)
        for epoch in range(initial_epoch, num_epochs):
            start_time = time.time()
            test_input = next(test_ds)
            image_name = str(epoch) + '_test.png'
            generate_images(generator,
                            test_input.numpy(),
                            os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                            show=False)
            train_input = next(train_ds_images)
            image_name = str(epoch) + '_train.png'
            generate_images(generator,
                            train_input.numpy(),
                            os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                            show=False)

            # training
            log_print('Training epoch {}'.format(epoch), add_timestamp=True)
            losses = [[], [], []]
            for n, input_image in train_ds.enumerate():
                total_loss, reconst_loss, kl_loss = train_step(input_image=input_image, target=input_image,
                                                          train_generator=True, train_discriminator=False)

                # if n.numpy() % (DISC_TRAIN_STEPS + 1) == 0:
                #     gen_loss, disc_loss, gp_loss = train_step(input_image=input_image, target=input_image, train_generator=True, train_discriminator=False)
                #     log_print('Trained generator.')
                # else:
                #     gen_loss, disc_loss, gp_loss = train_step(input_image=input_image, target=input_image, train_generator=False, train_discriminator=True)
                #     log_print('Trained discriminator.')

                losses[0].append(total_loss.numpy())
                losses[1].append(reconst_loss.numpy())
                losses[2].append(kl_loss.numpy())
            losses = [statistics.mean(x) for x in losses]
            with summary_writer.as_default():
                tf.summary.scalar('total_loss', losses[0], step=epoch)
                tf.summary.scalar('reconst_loss', losses[1], step=epoch)
                tf.summary.scalar('kl_loss', losses[2], step=epoch)
            summary_writer.flush()

            # testing
            log_print('Calculating validation losses...')
            val_losses = [[], [], []]
            for input_image in val_ds:
                total_loss, reconst_loss, kl_loss = eval_step(input_image, input_image)
                val_losses[0].append(total_loss.numpy())
                val_losses[1].append(reconst_loss.numpy())
                val_losses[2].append(kl_loss.numpy())

            val_losses = [statistics.mean(x) for x in val_losses]
            with summary_writer.as_default():
                tf.summary.scalar('val_total_loss', val_losses[0], step=epoch)
                tf.summary.scalar('val_reconst_loss', val_losses[1], step=epoch)
                tf.summary.scalar('val_kl_loss', val_losses[2], step=epoch)
            summary_writer.flush()

            end_time = time.time()
            log_print('Epoch {} completed in {} seconds'.format(epoch, round(end_time - start_time)))
            log_print("     total_loss       {:1.4f}".format(losses[0]))
            log_print("     reconst_loss      {:1.4f}".format(losses[1]))
            log_print("     kl_loss        {:1.4f}".format(losses[2]))

            log_print("     val_total_loss       {:1.4f}".format(val_losses[0]))
            log_print("     val_reconst_loss      {:1.4f}".format(val_losses[1]))
            log_print("     val_kl_loss        {:1.4f}".format(val_losses[2]))

            checkpoint.epoch.assign(epoch)

            if int(checkpoint.epoch) % CHECKPOINT_SAVE_INTERVAL == 0:
                save_path = manager.save()
                log_print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))
                # print("gen_total_loss {:1.2f}".format(gen_total_loss.numpy()))
                # print("disc_loss {:1.2f}".format(disc_loss.numpy()))


    try:
        log_print('Fitting to the data set', add_timestamp=True)
        log_print(' ')
        log_print('Parameters:')
        log_print('Experiment name: ' + str(EXPERIMENT_NAME))
        log_print('Batch size: ' + str(BATCH_SIZE))
        log_print('Epochs: ' + str(EPOCHS))
        log_print('Restore from checkpoint: ' + str(RESTORE_FROM_CHECKPOINT))
        log_print('Chechpoint save interval: ' + str(CHECKPOINT_SAVE_INTERVAL))
        log_print('Max number of checkpoints kept: ' + str(MAX_TO_KEEP))
        log_print('Runtime: ' + str(RUNTIME))
        log_print('Use TPU: ' + str(USE_TPU))
        log_print('Prefetch buffer size: ' + str(PREFETCH_BUFFER_SIZE))
        log_print('Shuffle buffer size: ' + str(SHUFFLE_BUFFER_SIZE))
        log_print('Input shape: ( ' + str(INPUT_HEIGHT) + ', ' + str(INPUT_WIDTH) + ', ' + str(INPUT_CHANNEL) + ' )')
        log_print('LAMBDA_GP: ' + str(LAMBDA_GP))
        log_print('Clip by norm: ' + str(CLIP_BY_NORM))
        log_print('Clip by value: ' + str(CLIP_BY_VALUE))
        log_print('Discriminator train steps/epoch: ' + str(DISC_TRAIN_STEPS))

        log_print(' ')

        log_print('Initial epoch: {}'.format(initial_epoch))
        # fit(train_ds.take(10), EPOCHS, val_ds.take(2), test_ds.repeat(), train_ds2.repeat(), initial_epoch=initial_epoch)
        fit(train_ds, EPOCHS, val_ds, test_ds.repeat(), train_ds2.repeat(), initial_epoch=initial_epoch)

        # save last checkpoint
        save_path = manager.save()
        log_print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))
        summary_writer.close()

    except KeyboardInterrupt:
        log_print('Keyboard Interrupt', add_timestamp=True)

        # save latest checkpoint and close log file
        save_path = manager.save()
        log_print("Saved checkpoint for epoch {}: {} due to KeyboardInterrupt".format(int(checkpoint.epoch), save_path))
        summary_writer.close()

    except:
        summary_writer.close()