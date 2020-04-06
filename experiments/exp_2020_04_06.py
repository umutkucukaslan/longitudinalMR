import datetime
import os
import statistics
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.adni_dataset import get_adni_dataset
from model.autoencoder import build_encoder, build_decoder
import model.gan as gan

"""
Training autoencoder adversarially using ADNI dataset.
"""

USE_COLAB = True
EXPERIMENT_NAME = 'exp_2020_04_06'

PREFETCH_BUFFER_SIZE = 5
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNEL = 1

LAMBDA = 100

EPOCHS = 5000
CHECKPOINT_SAVE_INTERVAL = 25
MAX_TO_KEEP = 3

if USE_COLAB:
    EXPERIMENT_FOLDER = os.path.join('/content/drive/My Drive/experiments', EXPERIMENT_NAME)
else:
    EXPERIMENT_FOLDER = os.path.join('/Users/umutkucukaslan/Desktop/thesis/experiments', EXPERIMENT_NAME)

if not os.path.isdir(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

# generator model plot path
GEN_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'gen_model_plot.jpg')
DIS_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'dis_model_plot.jpg')

# folder to save generated test images during training
if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, 'figures')):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'figures'))

# DATASET
train_ds, val_ds, test_ds = get_adni_dataset(use_colab=USE_COLAB)
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

# for example in train_ds.take(5):
#     plt.imshow(np.squeeze(example.numpy()[0]), cmap=plt.get_cmap('gray'))
#     plt.show()
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

encoder = build_encoder(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
                        output_shape=output_shape,
                        filters=filters,
                        kernel_size=kernel_size,
                        pool_size=(2, 2),
                        batch_normalization=batch_norm,
                        activation=tf.nn.relu,
                        name='encoder')

encoder.summary()


decoder = build_decoder(input_shape=output_shape,
                        output_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
                        filters=tuple(reversed(list(filters))),
                        kernel_size=kernel_size,
                        batch_normalization=batch_norm,
                        activation=tf.nn.relu,
                        name='decoder')

decoder.summary()


generator = tf.keras.Sequential(name='generator')
generator.add(encoder)
generator.add(decoder)

generator.summary()

tf.keras.utils.plot_model(generator, to_file=GEN_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=True)



# DISCRIMINATOR
discriminator = gan.get_discriminator_2020_04_06()
discriminator.summary()

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
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP)

# summary file writer for tensorboard
log_dir = os.path.join(EXPERIMENT_FOLDER, 'logs')
summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))


# LOSSES
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
    total_loss = gan_loss + LAMBDA * l1_loss
    return total_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# TRAINING
def train_step(input_image, target, epoch):
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

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

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


def fit(train_ds, epochs, val_ds, test_ds):

    test_ds = iter(test_ds)
    step = 0
    for epoch in range(epochs):
        start_time = time.time()
        test_input = next(test_ds)
        image_name = str(epoch) + '_test.png'
        generate_images(generator,
                        test_input.numpy(),
                        os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                        show=False)

        # training
        print('Training epoch {}'.format(epoch))
        losses = [[], [], [], []]
        for n, input_image in train_ds.enumerate():
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, input_image, step)
            losses[0].append(gen_total_loss.numpy())
            losses[1].append(gen_gan_loss.numpy())
            losses[2].append(gen_l1_loss.numpy())
            losses[3].append(disc_loss.numpy())
            if (n + 1) % 10 == 0:
                print('.', end='')
            if (n + 1) % 100 == 0:
                print(' x ', end='')
            if (n + 1) % 300 == 0:
                print()
            step += 1
        losses = [statistics.mean(x) for x in losses]

        # testing
        print('Calculating validation losses...')
        val_losses = [[], [], [], []]
        for input_image in val_ds:
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = eval_step(input_image, input_image)
            val_losses[0].append(gen_total_loss.numpy())
            val_losses[1].append(gen_gan_loss.numpy())
            val_losses[2].append(gen_l1_loss.numpy())
            val_losses[3].append(disc_loss.numpy())
        val_losses = [statistics.mean(x) for x in val_losses]
        with summary_writer.as_default():
            tf.summary.scalar('val_gen_total_loss', val_losses[0], step=step)
            tf.summary.scalar('val_gen_gan_loss', val_losses[1], step=step)
            tf.summary.scalar('val_gen_l1_loss', val_losses[2], step=step)
            tf.summary.scalar('val_disc_loss', val_losses[3], step=step)

        end_time = time.time()
        print('Epoch {} completed in {} seconds'.format(epoch, round(end_time - start_time)))
        print("     gen_total_loss {:1.2f}".format(losses[0]))
        print("     gen_gan_loss {:1.2f}".format(losses[1]))
        print("     gen_l1_loss {:1.2f}".format(losses[2]))
        print("     disc_loss {:1.2f}".format(losses[3]))

        print("     val_gen_total_loss {:1.2f}".format(val_losses[0]))
        print("     gen_gan_loss {:1.2f}".format(val_losses[1]))
        print("     gen_l1_loss {:1.2f}".format(val_losses[2]))
        print("     disc_loss {:1.2f}".format(val_losses[3]))

        checkpoint.step.assign_add(1)

        if (int(checkpoint.step) + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            print("gen_total_loss {:1.2f}".format(gen_total_loss.numpy()))
            print("disc_loss {:1.2f}".format(disc_loss.numpy()))


print('Fit to the data set')
fit(train_ds.take(10), EPOCHS, val_ds.take(2), test_ds.repeat())
# fit(train_ds, EPOCHS, val_ds, test_ds.repeat())

