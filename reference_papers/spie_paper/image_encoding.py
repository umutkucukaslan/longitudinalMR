import tensorflow as tf
import numpy as np


def loss(y, y2):
    return tf.reduce_mean((y - y2) * (y - y2))


def encode_image(generator, image, num_steps=1000, verbose=False):
    """
    Given an input image and generator network, this function finds the optimal input vector of the generator
    that results in the given image at the output of the generator.

    It is used to generate latent vectors corresponding to an image for a given generator network.

    :param generator: Generator model: latent vector -> image
    :param image: 4D image (batch_size=1, height, width, channels)
    :param num_steps: 1000 epochs usually gives good results
    :param verbose: Prints loss at each step
    :return: Latent vector of shape (1, latent_size)
    """
    latent_vector_size = generator.input.shape[1]
    x = tf.Variable(np.random.random((1, latent_vector_size)))

    opt = tf.keras.optimizers.Adam(learning_rate=0.025)
    for i in range(num_steps):
        with tf.GradientTape() as tape:
            y = generator(x, training=False)
            l = loss(y, image)
            if verbose:
                print(f"loss for step {i} is {l.numpy()}")
        grads = tape.gradient(l, x)
        opt.apply_gradients([(grads, x)])
    return x.numpy(), y.numpy()
