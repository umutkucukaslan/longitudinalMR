import datetime
import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist_dataset import get_mnist_dataset
from model.autoencoder import build_encoder, build_decoder

import model.gan as gan

"""
This is a toy example for GAN training using MNIST dataset.
"""

USE_COLAB = True
EXPERIMENT_NAME = 'exp_2020_04_05__2'

BUFFER_SIZE = 400
BATCH_SIZE = 256
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_CHANNEL = 1

LAMBDA = 100

EPOCHS = 150
CHECKPOINT_SAVE_INTERVAL = 10
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
ds_train, ds_test = get_mnist_dataset(use_colab=USE_COLAB)


def preprocess_dataset(example):
    image, label = example['image'], example['label']
    image = tf.dtypes.cast(image, tf.float32)
    image = image / 256.0
    return image, image


def preprocess_dataset2(example):
    image, label = example['image'], example['label']
    image = tf.dtypes.cast(image, tf.float32)
    image = image / 256.0
    image = tf.expand_dims(image, axis=0)
    return image, image


ds_train = ds_train.map(preprocess_dataset)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(20)

ds_test = ds_test.map(preprocess_dataset2)

# dataset example
print(ds_train.take(1))


# BUILD GENERATOR
# encoder
filters = (8, 16)
output_shape = 16
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
discriminator = gan.get_mnist_discriminator()

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


def generate_images(model, test_input, path=None, show=True):
    prediction = model(test_input)
    display_list = [np.stack([test_input.numpy()[0, :, :, 0],
                              test_input.numpy()[0, :, :, 0],
                              test_input.numpy()[0, :, :, 0]], axis=2),
                    np.stack([prediction.numpy()[0, :, :, 0],
                              prediction.numpy()[0, :, :, 0],
                              prediction.numpy()[0, :, :, 0]], axis=2)]

    title = ['Input Image', 'Reconstructed Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()


def fit(train_ds, epochs, test_ds):

    test_ds = test_ds.make_one_shot_iterator()
    step = 0
    for epoch in range(epochs):

        test_input, test_target = next(test_ds)
        image_name = str(epoch) + '_test.png'
        generate_images(generator,
                        test_input,
                        os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                        show=False)

        for n, (input_image, target_image) in train_ds.enumerate():
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target_image, step)
            # if (n + 1) % 200 == 0:
            #     print('.', end='')
            # if (n + 1) % 10000 == 0:
            #     print()

            step += 1
        print('epoch %d ended' % epoch)
        print("     gen_total_loss {:1.2f}".format(gen_total_loss.numpy()))
        print("     gen_gan_loss {:1.2f}".format(gen_gan_loss.numpy()))
        print("     gen_l1_loss {:1.2f}".format(gen_l1_loss.numpy()))
        print("     disc_loss {:1.2f}".format(disc_loss.numpy()))

        checkpoint.step.assign_add(1)

        if (int(checkpoint.step) + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            print("gen_total_loss {:1.2f}".format(gen_total_loss.numpy()))
            print("disc_loss {:1.2f}".format(disc_loss.numpy()))

            # checkpoint.save(checkpoint_prefix + "_" + str(epoch + 1))


# fit(ds_train.take(10), EPOCHS, ds_test.repeat())
fit(ds_train, EPOCHS, ds_test.repeat())





