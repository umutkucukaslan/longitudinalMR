import copy

import tensorflow as tf

from model.custom_layers import ProgressiveGANDownsample, ProgressiveGANUpsample, Conv2D_LeakyReLU_L2normalize


def down_model(input_shape, filters, name=None, **kwargs):
    """
    Downsampling block of progressive GAN network.
    Performs convolutions and maxpools at last. Each convolution block is covn-leakyrelu-l2_normalize

    :param input_shape:
    :param filters:
    :param name:
    :param kwargs:
    :return:
    """
    input = tf.keras.Input(input_shape)
    output = ProgressiveGANDownsample(filters, **kwargs)(input)
    if name:
        return tf.keras.Model(inputs=input, outputs=output, name=name)
    else:
        return tf.keras.Model(inputs=input, outputs=output)


def up_model(input_shape, filters, name=None, **kwargs):
    """
    Upsample block of progressive GAN network.
    Upsamples input by (2, 2) upsample2d block performs convolutions with number of filters specified by filters
    Each convolution block is covn-leakyrelu-l2_normalize

    :param input_shape:
    :param filters: A list of num_filters in convolutional blocks
    :param name:
    :param kwargs:
    :return:
    """
    input = tf.keras.Input(input_shape)
    output = ProgressiveGANUpsample(filters, **kwargs)(input)
    if name:
        return tf.keras.Model(inputs=input, outputs=output, name=name)
    else:
        return tf.keras.Model(inputs=input, outputs=output)


def core_discriminator(input_shape, filters=[512], latent_size=512, kernel_size=3, alpha=0.2, name='core_encoder'):
    """
    Simple discriminator architecture.

    n_filters * [Conv + LeakyReLU + l2_normalize] + Flatten + Dense + Dense

    :param input_shape:
    :param filters:
    :param latent_size:
    :param kernel_size:
    :param initializer:
    :return:
    """
    input = tf.keras.Input(input_shape)
    x = Conv2D_LeakyReLU_L2normalize(filters=filters, kernel_size=kernel_size, alpha=alpha)(input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_size)(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=input, outputs=x, name=name)


def core_encoder(input_shape, filters=[512], latent_size=512, kernel_size=3, alpha=0.2, name='core_encoder'):
    """
    Simple encoder architecture.

    n_filters * [Conv + LeakyReLU + l2_normalize] + Flatten + Dense

    :param input_shape:
    :param filters:
    :param latent_size:
    :param kernel_size:
    :param initializer:
    :return:
    """
    input = tf.keras.Input(input_shape)
    x = Conv2D_LeakyReLU_L2normalize(filters=filters, kernel_size=kernel_size, alpha=alpha)(input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_size)(x)

    return tf.keras.Model(inputs=input, outputs=x, name=name)


def core_decoder(input_size, output_shape, filters, kernel_size=3, alpha=0.2, name='core_decoder'):
    """
    Simple decoder architecture.

    Dense + Reshape + n_filters * [Conv + LeakyReLU + l2_normalize]

    :param output_shape:
    :param filters:
    :param kernel_size:
    :param alpha:
    :return:
    """
    input = tf.keras.Input([input_size])
    x = tf.keras.layers.Dense(output_shape[0]*output_shape[1]*output_shape[2])(input)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
    x = tf.keras.layers.Reshape(output_shape)(x)
    x = Conv2D_LeakyReLU_L2normalize(filters=filters, kernel_size=kernel_size, alpha=alpha)(x)

    return tf.keras.Model(inputs=input, outputs=x, name=name)


def conv_1x1(input_shape, filters, name='from_to_image'):
    input = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, use_bias=False, padding='same')(input)

    return tf.keras.Model(inputs=input, outputs=x, name=name)


def progressive_gan(input_shape, filters, latent_vector_size, verbose=False):
    """
    Build encoder-decoder architecture with discriminator that is suitable for progressive GAN training. It returns
    all levels of generators and discriminators together with fadein networks for training.

    :param input_shape: [image_height, image_width, image_channels]
    :param filters: A list of lists. Ex: [[128, 256], [256, 512], [512, 512], [512, 512], [512, 512]]
                    Each sublist corresponds to the filters in ProgressiveGAN block and reduced input size by half.
                    Ensure input_shape is dividable by 2**len(filters).
    :param latent_vector_size: Encoder latent space vector size
    :param verbose: If True, prints network summaries.

    :return: basic_generators, fadein_generators, basic_discriminators, fadein_discriminators, encoder, decoder
    """

    n_downsampling = len(filters)
    input_shapes = [[input_shape[0] // 2**n, input_shape[1] // 2**n, input_shape[2]] for n in range(n_downsampling + 1)]
    assert input_shapes[0][0] == input_shapes[-1][0] * 2**n_downsampling and input_shapes[0][1] == input_shapes[-1][1] * 2**n_downsampling, "input_shape cannot be dividable by 2**n_downsampling"

    # input shapes of down models
    down_model_input_shapes = copy.deepcopy(input_shapes)
    for i in range(len(filters)):
        down_model_input_shapes[i][2] = filters[i][0]
    down_model_input_shapes[-1][2] = filters[-1][-1]

    # GENERATOR

    # down_models
    down_models = [down_model(down_model_input_shapes[i], filters[i], name='down_model_{}'.format(i)) for i in range(len(filters))]

    # up_models
    up_models = [up_model(down_model_input_shapes[i + 1], list(reversed(filters[i])), name='up_model_{}'.format(i)) for i in range(len(filters))]

    # core generator parts
    core_enc = core_encoder(input_shape=down_model_input_shapes[-1], filters=[filters[-1][-1]], latent_size=latent_vector_size, kernel_size=3, alpha=0.2, name="core_encoder")
    core_dec = core_decoder(input_size=latent_vector_size, output_shape=down_model_input_shapes[-1], filters=[filters[-1][-1]], kernel_size=3, alpha=0.2, name="core_decoder")

    # down and up models
    if verbose:
        for m in down_models:
            m.summary()
        for m in up_models:
            m.summary()
        core_enc.summary()
        core_dec.summary()

    from_image_models = [conv_1x1(input_shapes[i], down_model_input_shapes[i][-1], name='from_image_{}'.format(i)) for i in range(len(input_shapes))]
    to_image_models = [conv_1x1(down_model_input_shapes[i], input_shapes[i][-1], name='to_image_{}'.format(i)) for i in range(len(input_shapes))]

    if verbose:
        for m in from_image_models:
            m.summary()
        for m in to_image_models:
            m.summary()

    input_layers = [tf.keras.Input(x) for x in input_shapes]
    basic_generators = []
    inp = input_layers[-1]
    x = from_image_models[-1](inp)
    x = core_enc(x)
    x = core_dec(x)
    x = to_image_models[-1](x)
    basic_generators.append(tf.keras.Model(inputs=inp, outputs=x, name='basic_generator_0'))

    for i in range(len(down_models)):
        inp = input_layers[-i-2]
        downs = down_models[-i-1:]
        ups = up_models[-i-1:]
        from_image = from_image_models[-i-2]
        to_image = to_image_models[-i-2]

        x = from_image(inp)
        for d in downs:
            x= d(x)
        x = core_enc(x)
        x = core_dec(x)
        for u in reversed(ups):
            x = u(x)
        x = to_image(x)

        basic_generators.append(tf.keras.Model(inputs=inp, outputs=x, name='basic_generator_{}'.format(i + 1)))

    # encoder definitions for inference and testing
    inp = input_layers[0]
    x = from_image_models[0](inp)
    for d in down_models:
        x = d(x)
    latent = core_enc(x)
    encoder = tf.keras.Model(inputs=inp, outputs=latent, name='encoder')

    # decoder definition for inference and testing
    inp = tf.keras.Input([latent_vector_size])
    x = core_dec(inp)
    for u in reversed(up_models):
        x = u(x)
    x = to_image_models[0](x)
    decoder = tf.keras.Model(inputs=inp, outputs=x, name='decoder')

    downsample = tf.keras.layers.AveragePooling2D()
    upsample = tf.keras.layers.UpSampling2D()

    fadein_generators = []
    for i in range(len(down_models)):
        inp = input_layers[-i-2]
        weight = tf.keras.Input([1])

        downs = down_models[-i-1:]
        down_to_merge = downs[0]
        downs = downs[1:]

        ups = up_models[-i-1:]
        up_to_merge = ups[0]
        ups = ups[1:]

        from_image = from_image_models[-i-2]
        from_image_shortcut = from_image_models[-i-1]
        to_image = to_image_models[-i-2]
        to_image_shortcut = to_image_models[-i-1]

        y = downsample(inp)
        y = from_image_shortcut(y)      # short cut connection

        x = from_image(inp)
        x = down_to_merge(x)            # added block for higher resolution

        x = tf.keras.layers.Add()([x * weight, y * (1 - weight)])       # merge of two branches

        # pass through previously trained generator
        for d in downs:
            x= d(x)
        x = core_enc(x)
        x = core_dec(x)
        for u in reversed(ups):
            x = u(x)

        # short cut connection
        y = to_image_shortcut(x)
        y = upsample(y)     # channel dimension is not compatible

        x = up_to_merge(x)
        x = to_image(x)

        x = tf.keras.layers.Add()([x * weight, y * (1 - weight)])       # merge of two branches

        fadein_generators.append(tf.keras.Model(inputs=[inp, weight], outputs=x, name='fadein_generator_{}'.format(i + 1)))


    # DISCRIMINATOR

    # down_models
    disc_down_models = [down_model(down_model_input_shapes[i], filters[i], name='disc_down_model_{}'.format(i)) for i in range(len(filters))]
    for m in disc_down_models:
        m.summary()

    disc_from_image_models = [conv_1x1(input_shapes[i], down_model_input_shapes[i][-1], name='disc_from_image_{}'.format(i)) for i in range(len(input_shapes))]
    for m in disc_from_image_models:
        m.summary()

    core_disc = core_discriminator(input_shape=down_model_input_shapes[-1], filters=[filters[-1][-1]], latent_size=latent_vector_size, kernel_size=3, alpha=0.2, name="core_discriminator")

    # basic discriminator definitions
    basic_discriminators = []
    inp = input_layers[-1]
    x = disc_from_image_models[-1](inp)
    x = core_disc(x)
    basic_discriminators.append(tf.keras.Model(inputs=inp, outputs=x, name='basic_discriminator_0'))

    for i in range(len(disc_down_models)):
        inp = input_layers[-i - 2]
        downs = disc_down_models[-i - 1:]
        from_image = disc_from_image_models[-i - 2]

        x = from_image(inp)
        for d in downs:
            x = d(x)
        x = core_disc(x)

        basic_discriminators.append(tf.keras.Model(inputs=inp, outputs=x, name='basic_discriminator_{}'.format(i + 1)))

    fadein_discriminators = []
    for i in range(len(disc_down_models)):
        inp = input_layers[-i - 2]
        weight = tf.keras.Input([1])

        downs = disc_down_models[-i - 1:]
        down_to_merge = downs[0]
        downs = downs[1:]

        from_image = disc_from_image_models[-i - 2]
        from_image_shortcut = disc_from_image_models[-i - 1]

        y = downsample(inp)
        y = from_image_shortcut(y)  # short cut connection

        x = from_image(inp)
        x = down_to_merge(x)  # added block for higher resolution

        x = tf.keras.layers.Add()([x * weight, y * (1 - weight)])  # merge of two branches

        # pass through previously trained generator
        for d in downs:
            x = d(x)
        x = core_disc(x)

        fadein_discriminators.append(
            tf.keras.Model(inputs=[inp, weight], outputs=x, name='fadein_discriminator_{}'.format(i + 1)))

    return basic_generators, fadein_generators, basic_discriminators, fadein_discriminators, encoder, decoder


# progressive_gan([192, 160, 1], filters=[[128, 256], [256, 512], [512, 512], [512, 512], [512, 512]], latent_vector_size=512, verbose=False)

# for i in range(len(basic_discriminators)):
    #     disc = basic_discriminators[i]
    #     disc.summary()
    #     tf.keras.utils.plot_model(disc, to_file='/Users/umutkucukaslan/Desktop/thesis/fadein/basic_discriminator_{}.jpg'.format(i), show_shapes=True, dpi=150, expand_nested=True)
    #
    # for i in range(len(fadein_discriminators)):
    #     disc = fadein_discriminators[i]
    #     disc.summary()
    #     tf.keras.utils.plot_model(disc, to_file='/Users/umutkucukaslan/Desktop/thesis/fadein/fadein_discriminator_{}.jpg'.format(i), show_shapes=True, dpi=150, expand_nested=True)
    #
    # for i in range(len(basic_generators)):
    #     gen = basic_generators[i]
    #     gen.summary()
    #     tf.keras.utils.plot_model(gen, to_file='/Users/umutkucukaslan/Desktop/thesis/fadein/basic_generator_{}.jpg'.format(i), show_shapes=True, dpi=150, expand_nested=True)
    #
    # for i in range(len(fadein_generators)):
    #     gen = fadein_generators[i]
    #     gen.summary()
    #     tf.keras.utils.plot_model(gen, to_file='/Users/umutkucukaslan/Desktop/thesis/fadein/fadein_generator_{}.jpg'.format(i), show_shapes=True, dpi=150, expand_nested=True)

