
import tensorflow as tf

from model.custom_layers import ProgressiveGANDownsample, ProgressiveGANUpsample


def progressive_gan(input_shape=None):

    initializer = tf.random_normal_initializer(0, 0.02)

    if input_shape is None:
        input_shape = [192, 160, 1]

    input_0 = tf.keras.Input(input_shape)
    input_1 = ProgressiveGANDownsample([128, 256], name='d_1')(input_0)
    input_2 = ProgressiveGANDownsample([256, 512], name='d_2')(input_1)
    input_3 = ProgressiveGANDownsample([512, 512], name='d_3')(input_2)
    input_4 = ProgressiveGANDownsample([512, 512], name='d_4')(input_3)
    input_5 = ProgressiveGANDownsample([512, 512], name='d_5')(input_4)

    latent_vector = tf.keras.layers.Dense(512)(
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(
            input_5
        )
    )
    output_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(
        tf.keras.layers.Reshape((6, 5, -1))(
            tf.keras.layers.Dense(512 * 6 * 5)(
                latent_vector
            )
        )
    )

    output_4 = ProgressiveGANUpsample([512, 512], name='up_5')(output_5)
    output_3 = ProgressiveGANUpsample([512, 512], name='up_4')(output_4)
    output_2 = ProgressiveGANUpsample([512, 512], name='up_3')(output_3)
    output_1 = ProgressiveGANUpsample([512, 256], name='up_2')(output_2)
    output_0 = ProgressiveGANUpsample([256, 128], name='up_1')(output_1)
    output_image = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(
            output_0
    )

    # generator, encoder and decoder definitions
    generator = tf.keras.Model(inputs=input_0, outputs=output_image)
    encoder = tf.keras.Model(inputs=input_0, outputs=latent_vector)
    decoder = tf.keras.Model(inputs=latent_vector, outputs=output_image)

    # sub model definitions for low resolution training extensions
    down_1 = tf.keras.Model(inputs=input_0, outputs=input_1)
    down_2 = tf.keras.Model(inputs=input_1, outputs=input_2)
    down_3 = tf.keras.Model(inputs=input_2, outputs=input_3)
    down_4 = tf.keras.Model(inputs=input_3, outputs=input_4)
    down_5 = tf.keras.Model(inputs=input_4, outputs=input_5)
    down_6 = tf.keras.Model(inputs=input_5, outputs=latent_vector)
    up_6 = tf.keras.Model(inputs=latent_vector, outputs=output_5)
    up_5 = tf.keras.Model(inputs=output_5, outputs=output_4)
    up_4 = tf.keras.Model(inputs=output_4, outputs=output_3)
    up_3 = tf.keras.Model(inputs=output_3, outputs=output_2)
    up_2 = tf.keras.Model(inputs=output_2, outputs=output_1)
    up_1 = tf.keras.Model(inputs=output_1, outputs=output_0)
    to_image = tf.keras.Model(inputs=output_0, outputs=output_image)





    return generator, encoder, decoder

progressive_gan()

