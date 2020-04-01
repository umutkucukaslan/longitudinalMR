import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist_dataset import get_mnist_dataset
from model.autoencoder import build_encoder, build_decoder

"""
This is a toy example for GAN training using MNIST dataset.
"""

USE_COLAB = True

BUFFER_SIZE = 400
BATCH_SIZE = 128
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_CHANNEL = 1


if USE_COLAB:
    SUMMARY_FILE_DIR = '/content/drive/My Drive/trained_models/exp_2020_03_30'
else:
    SUMMARY_FILE_DIR = '/Users/umutkucukaslan/Desktop/thesis/testsummaryfile'

if not os.path.isdir(SUMMARY_FILE_DIR):
    os.makedirs(SUMMARY_FILE_DIR)
if not os.path.isdir(os.path.join(SUMMARY_FILE_DIR, 'figures')):
    os.makedirs(os.path.join(SUMMARY_FILE_DIR, 'figures'))

# DATASET
ds_train, ds_test = get_mnist_dataset(use_colab=USE_COLAB)


def preprocess_dataset(example):
    image, label = example['image'], example['label']
    image = tf.dtypes.cast(image, tf.float32)
    image = image / 256.0
    image = tf.expand_dims(image, axis=0)
    return image, image


ds_train = ds_train.map(preprocess_dataset)
ds_test = ds_test.map(preprocess_dataset)

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


loss_object = tf.keras.losses.BinaryCrossentropy()

generator_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)

summary_writer = tf.summary.create_file_writer(SUMMARY_FILE_DIR)


# LOSSES
def generator_loss(gen_output, target):
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
    return l1_loss


# TRAINING
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)

        gen_l1_loss = generator_loss(gen_output, target)

    generator_gradients = gen_tape.gradient(gen_l1_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)


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
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()


def fit(train_ds, epochs, test_ds):

    train_ds = train_ds.batch(BATCH_SIZE)
    test_ds = test_ds.make_one_shot_iterator()
    step = 0
    for epoch in range(epochs):

        test_input, test_target = next(test_ds)
        image_name = str(epoch) + '_test.png'
        generate_images(generator,
                        test_input,
                        os.path.join(SUMMARY_FILE_DIR, 'figures', image_name),
                        show=False)

        for n, (input_image, target_image) in train_ds.enumerate():
            if (n + 1) % 100 == 0:
                print('.', end='')
            if (n + 1) % 1000 == 0:
                print()

            input_image = tf.reshape(input_image, [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])

            train_step(input_image, target_image, step)
            step += 1
        print('epoch %d ended' % epoch)


fit(ds_train.take(40000), 120, ds_test.take(300))

