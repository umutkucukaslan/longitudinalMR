import datetime
import os
import statistics
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.adni_dataset import get_adni_dataset
from model.autoencoder import build_encoder, build_decoder, build_encoder_2020_04_13
import model.gan as gan

"""
Training autoencoder adversarially using ADNI dataset.
"""



RUNTIME = 'cloud'   # cloud, colab or none
USE_TPU = False
RESTORE_FROM_CHECKPOINT = True
EXPERIMENT_NAME = 'exp_2020_04_13_cloud'

PREFETCH_BUFFER_SIZE = 5
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNEL = 3

LAMBDA = 100
LAMBDA_ADV = 1

EPOCHS = 5000
CHECKPOINT_SAVE_INTERVAL = 5
MAX_TO_KEEP = 5


# set batch size easily
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])

# DEFAULT_FLOAT_TYPE = 'float32'
# tf.keras.backend.set_floatx(DEFAULT_FLOAT_TYPE)

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

# DATASET
train_ds, val_ds, test_ds = get_adni_dataset(runtime=RUNTIME)


def process_dataset(image):
    image = 2.0 * image - 1.0
    # return tf.concat([image, image, image], axis=-1)
    return image


train_ds = train_ds.map(process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

# BUILD GENERATOR
# encoder
filters = (64, 128, 256, 256, 128, 64)
output_shape = 128
kernel_size = 3
batch_norm = True

encoder = build_encoder_2020_04_13(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL), latent_space_size=output_shape, name='encoder')

encoder.summary()
encoder.summary(print_fn=log_print)


decoder = build_decoder(input_shape=output_shape,
                        output_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
                        filters=tuple(reversed(list(filters))),
                        kernel_size=kernel_size,
                        batch_normalization=batch_norm,
                        activation=tf.nn.relu,
                        name='decoder')

decoder.summary()
decoder.summary(print_fn=log_print)


generator = tf.keras.Sequential(name='generator')
generator.add(encoder)
generator.add(decoder)

generator.summary()
generator.summary(print_fn=log_print)

tf.keras.utils.plot_model(generator, to_file=GEN_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=True)



# DISCRIMINATOR
discriminator = gan.get_discriminator_2020_04_13()
discriminator.summary()
discriminator.summary(print_fn=log_print)

tf.keras.utils.plot_model(discriminator, to_file=DIS_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=False)


# =================
# loss to be used
loss_object = tf.keras.losses.BinaryCrossentropy()

# optimizers
generator_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    log_print("Restored from {}".format(manager.latest_checkpoint))
else:
    log_print("Initializing from scratch.")

initial_epoch = checkpoint.epoch.numpy() + 1

# summary file writer for tensorboard
log_dir = os.path.join(EXPERIMENT_FOLDER, 'logs')
summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))


# LOSSES
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
    total_loss = LAMBDA_ADV * gan_loss + LAMBDA * l1_loss
    return total_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# TRAINING
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, input_image], training=True)
        disc_generated_output = discriminator([gen_output, input_image], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # with summary_writer.as_default():
    #     tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    #     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    #     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    #     tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def eval_step(input_image, target):
    gen_output = generator(input_image, training=False)

    disc_real_output = discriminator([input_image, input_image], training=False)
    disc_generated_output = discriminator([gen_output, input_image], training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def generate_images(model, test_input, path=None, show=True):
    if test_input.ndim < 4:
        test_input = np.expand_dims(test_input, axis=0)
    prediction = model(test_input)
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


def fit(train_ds, num_epochs, val_ds, test_ds, initial_epoch=0):

    assert initial_epoch < num_epochs
    test_ds = iter(test_ds)
    for epoch in range(initial_epoch, num_epochs):
        start_time = time.time()
        test_input = next(test_ds)
        image_name = str(epoch) + '_test.png'
        generate_images(generator,
                        test_input.numpy(),
                        os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                        show=False)

        # training
        log_print('Training epoch {}'.format(epoch), add_timestamp=True)
        losses = [[], [], [], []]
        for n, input_image in train_ds.enumerate():
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, input_image)
            losses[0].append(gen_total_loss.numpy())
            losses[1].append(gen_gan_loss.numpy())
            losses[2].append(gen_l1_loss.numpy())
            losses[3].append(disc_loss.numpy())
            # if (n + 1) % 10 == 0:
            #     print('.', end='')
            # if (n + 1) % 100 == 0:
            #     print(' x ', end='')
            # if (n + 1) % 300 == 0:
            #     print()
        losses = [statistics.mean(x) for x in losses]
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', losses[0], step=epoch)
            tf.summary.scalar('gen_gan_loss', losses[1], step=epoch)
            tf.summary.scalar('gen_l1_loss', losses[2], step=epoch)
            tf.summary.scalar('disc_loss', losses[3], step=epoch)

        # testing
        log_print('Calculating validation losses...')
        val_losses = [[], [], [], []]
        for input_image in val_ds:
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = eval_step(input_image, input_image)
            val_losses[0].append(gen_total_loss.numpy())
            val_losses[1].append(gen_gan_loss.numpy())
            val_losses[2].append(gen_l1_loss.numpy())
            val_losses[3].append(disc_loss.numpy())
        val_losses = [statistics.mean(x) for x in val_losses]
        with summary_writer.as_default():
            tf.summary.scalar('val_gen_total_loss', val_losses[0], step=epoch)
            tf.summary.scalar('val_gen_gan_loss', val_losses[1], step=epoch)
            tf.summary.scalar('val_gen_l1_loss', val_losses[2], step=epoch)
            tf.summary.scalar('val_disc_loss', val_losses[3], step=epoch)

        end_time = time.time()
        log_print('Epoch {} completed in {} seconds'.format(epoch, round(end_time - start_time)))
        log_print("     gen_total_loss {:1.2f}".format(losses[0]))
        log_print("     gen_gan_loss   {:1.2f}".format(losses[1]))
        log_print("     gen_l1_loss    {:1.2f}".format(losses[2]))
        log_print("     disc_loss      {:1.2f}".format(losses[3]))

        log_print("     val_gen_total_loss {:1.2f}".format(val_losses[0]))
        log_print("     gen_gan_loss       {:1.2f}".format(val_losses[1]))
        log_print("     gen_l1_loss        {:1.2f}".format(val_losses[2]))
        log_print("     disc_loss          {:1.2f}".format(val_losses[3]))

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
    log_print('Cost weight lambda: ' + str(LAMBDA))
    log_print(' ')

    log_print('Initial epoch: {}'.format(initial_epoch))
    # fit(train_ds.take(10), EPOCHS, val_ds.take(2), test_ds.repeat(), initial_epoch=initial_epoch)
    fit(train_ds, EPOCHS, val_ds, test_ds.repeat(), initial_epoch=initial_epoch)

    # save last checkpoint
    save_path = manager.save()
    log_print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))

except KeyboardInterrupt:
    log_print('Keyboard Interrupt', add_timestamp=True)

    # save latest checkpoint and close log file
    save_path = manager.save()
    log_print("Saved checkpoint for epoch {}: {} due to KeyboardInterrupt".format(int(checkpoint.epoch), save_path))




    # IP adresi   46.196.128.64
    # 46.196.128.64


