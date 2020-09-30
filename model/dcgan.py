import tensorflow as tf
from tensorflow.keras import layers


def make_dcgan_generator_model(input_vector_size=100, seed=None):
    """
    DC GAN implementation for generator. Output shape is (64, 64, 1)
    Output image range is [-1, 1] (tanh activation)

    :param input_vector_size: Eg. 100 or 256
    :return: generator model
    """

    kernel_initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02, seed=seed
    )

    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            4 * 4 * 1024,
            use_bias=False,
            input_shape=(input_vector_size,),
            kernel_initializer=kernel_initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            512,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )
    )
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            256,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )
    )
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            128,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )
    )
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
            kernel_initializer=kernel_initializer,
        )
    )
    assert model.output_shape == (None, 64, 64, 1)

    return model


def make_dcgan_discriminator_model(
    input_shape=(64, 64, 1), kernel_size=(5, 5), leaky_slope=0.2, seed=None
):
    """
    For 64x64 input, output is 4x4x1 within range [0, 1]


    :param input_shape:
    :param kernel_size:
    :param leaky_slope:
    :param seed:
    :return:
    """
    ndf = 64
    kernel_initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02, seed=seed
    )

    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            ndf,
            kernel_size,
            strides=(2, 2),
            padding="same",
            input_shape=input_shape,
            kernel_initializer=kernel_initializer,
        )
    )  # 32x32
    model.add(layers.LeakyReLU(alpha=leaky_slope))

    model.add(
        layers.Conv2D(
            ndf * 2,
            kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )
    )  # 16x16
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_slope))

    model.add(
        layers.Conv2D(
            ndf * 2,
            kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )
    )  # 8x8
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_slope))

    model.add(
        layers.Conv2D(
            ndf * 2,
            kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )
    )  # 4x4
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_slope))

    model.add(
        layers.Conv2D(
            1,
            kernel_size,
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
        )
    )  # 4x4
    model.add(layers.Activation(tf.nn.sigmoid))

    # 4x4x1 output within range [0, 1]

    return model
