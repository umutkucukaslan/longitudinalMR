import tensorflow as tf
import numpy as np


def loss(y, y2):
    return tf.reduce_mean((y - y2) * (y - y2))


def encode_image(generator, image, num_steps=1000, verbose=False):
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
