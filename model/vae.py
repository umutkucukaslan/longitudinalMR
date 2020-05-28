
import tensorflow as tf


def build_encoder(input_shape=(128, 128, 3), output_shape=128, filters=(32, 64, 128), kernel_size=5,
                  pool_size=(2, 2), batch_normalization=False, activation="relu", name='encoder', alpha=0.2):

    """
    Assumes input is in the range [-1, 1]

    :param input_shape:
    :param output_shape:
    :param filters:
    :param kernel_size:
    :param pool_size:
    :param batch_normalization:
    :param activation: None, "relu", "leakyrelu" or "swish"
    :param name:
    :param alpha:
    :return:
    """

    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = inputs
    for i in range(len(filters)):
        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=kernel_size,
                                   padding='same',
                                   activation=None)(x)
        if activation == "relu":
            x = tf.keras.layers.ReLU()(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(alpha)(x)
        elif activation == "swish":
            x = tf.nn.swish(x)

        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same')(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    output_mean = tf.keras.layers.Dense(output_shape, activation=None)(x)
    output_std = tf.keras.layers.Dense(output_shape, activation=None)(x)

    return tf.keras.Model(inputs=inputs, outputs=[output_mean, output_std], name=name)
