import tensorflow as tf


def wgan_gp_loss(f, x_real, x_fake, lambda_gp):
    """
    WGAN-GP loss implementation. Returns disc_loss, gen_loss

    :param f: discriminator network
    :param x_real: real example / real image
    :param x_fake: fake example / generated image
    :param lambda_gp: weight of gradient penalty
    :return: disc_loss, gen_loss
    """
    disc_real_output = tf.reduce_mean(f([x_real, x_real], training=True))
    disc_fake_output = tf.reduce_mean(f([x_fake, x_real], training=True))

    x_real_shape = tf.shape(x_real).numpy()
    shape = [1 for i in range(len(x_real_shape))]
    shape[0] = x_real_shape[0]

    alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
    x_hat = x_real + alpha * (x_real - x_fake)
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        x_hat_out = f([x_hat, x_real], training=True)
    grad = tape.gradient(x_hat_out, x_hat)
    grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((grad_norm - 1.) ** 2)
    disc_loss = disc_fake_output - disc_real_output + lambda_gp * gp     # discriminator loss

    gen_loss = -disc_fake_output    # generator loss

    return gen_loss, disc_loss, gp


def wgan_gp_loss_progressive_gan(f, x_real, x_fake, lambda_gp, weight=None):
    """
    WGAN-GP loss implementation. Returns disc_loss, gen_loss

    :param f: discriminator network
    :param x_real: real example / real image
    :param x_fake: fake example / generated image
    :param lambda_gp: weight of gradient penalty
    :return: disc_loss, gen_loss
    """
    if weight is not None:
        disc_real_output = tf.reduce_mean(f([x_real, weight], training=True))
        disc_fake_output = tf.reduce_mean(f([x_fake, weight], training=True))
    else:
        disc_real_output = tf.reduce_mean(f(x_real, training=True))
        disc_fake_output = tf.reduce_mean(f(x_fake, training=True))


    x_real_shape = tf.shape(x_real).numpy()
    shape = [1 for i in range(len(x_real_shape))]
    shape[0] = x_real_shape[0]

    alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
    x_hat = x_real + alpha * (x_real - x_fake)
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        if weight is not None:
            x_hat_out = f([x_hat, weight], training=True)
        else:
            x_hat_out = f(x_hat, training=True)
    grad = tape.gradient(x_hat_out, x_hat)
    grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((grad_norm - 1.) ** 2)
    disc_loss = disc_fake_output - disc_real_output + lambda_gp * gp     # discriminator loss

    gen_loss = -disc_fake_output    # generator loss

    return gen_loss, disc_loss, gp
