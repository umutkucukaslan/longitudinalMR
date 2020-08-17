import sys
import tensorflow as tf


def build_encoder(
    input_shape=(128, 128, 3),
    output_shape=128,
    filters=(32, 64, 128),
    kernel_size=5,
    pool_size=(2, 2),
    batch_normalization=False,
    activation=tf.nn.relu,
    name="encoder",
):

    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = inputs
    for i in range(len(filters)):
        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
        )(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation=None)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_decoder(
    input_shape=128,
    output_shape=(128, 128, 3),
    filters=(128, 64, 32),
    kernel_size=5,
    batch_normalization=False,
    activation=tf.nn.relu,
    name="decoder",
):

    n_upsamplings = len(filters)
    x_init, y_init = (
        output_shape[0] // 2 ** n_upsamplings,
        output_shape[1] // 2 ** n_upsamplings,
    )

    if (
        x_init * 2 ** n_upsamplings != output_shape[0]
        or y_init * 2 ** n_upsamplings != output_shape[1]
    ):
        print(
            "Output image dimensions should be divisible by 2^len(filters). Please set a suitable output_shape"
        )
        sys.exit()

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(x_init * y_init * filters[0], activation=activation)(
        inputs
    )
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((x_init, y_init, filters[0]))(x)
    for i in range(1, len(filters)):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
        )(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(
        filters=output_shape[2],
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_fcn(
    input_shape=128,
    output_shape=128,
    hidden_units=[256],
    batch_normalization=False,
    activation=tf.nn.relu,
    name="fcn",
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units=units, activation=activation)(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation=activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_encoder_with_lrelu_activation(
    input_shape=(128, 128, 3),
    output_shape=128,
    filters=(32, 64, 128),
    kernel_size=5,
    pool_size=(2, 2),
    batch_normalization=False,
    name="encoder",
):

    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = inputs
    for i in range(len(filters)):
        x = tf.keras.layers.Conv2D(
            filters=filters[i], kernel_size=kernel_size, padding="same", activation=None
        )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_shape, activation=None)(x)
    outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_decoder_with_lrelu_activation(
    input_shape=128,
    output_shape=(128, 128, 3),
    filters=(128, 64, 32),
    kernel_size=5,
    batch_normalization=False,
    name="decoder",
):

    n_upsamplings = len(filters)
    x_init, y_init = (
        output_shape[0] // 2 ** n_upsamplings,
        output_shape[1] // 2 ** n_upsamplings,
    )

    if (
        x_init * 2 ** n_upsamplings != output_shape[0]
        or y_init * 2 ** n_upsamplings != output_shape[1]
    ):
        print(
            "Output image dimensions should be divisible by 2^len(filters). Please set a suitable output_sh15ape"
        )
        sys.exit()

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(x_init * y_init * filters[0], activation=None)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((x_init, y_init, filters[0]))(x)
    for i in range(1, len(filters)):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(
            filters=filters[i], kernel_size=kernel_size, padding="same", activation=None
        )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(
        filters=output_shape[2],
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_encoder_2020_04_13(input_shape, latent_space_size, name):

    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3), include_top=False, weights="imagenet"
    )
    mobilenetv2_output = mobilenetv2.get_layer(
        "block_15_add"
    ).output  # selected features
    base_model = tf.keras.Model(
        inputs=mobilenetv2.inputs,
        outputs=mobilenetv2_output,
        name="mobilenetv2_feature_extractor",
    )

    for layer in base_model.layers:
        layer.trainable = False

    # encoder network
    inp = tf.keras.Input(input_shape)
    base_model_inp = tf.concat([inp, inp, inp], axis=-1)
    features = base_model(base_model_inp)
    flattened = tf.keras.layers.Flatten()(features)
    out = tf.keras.layers.Dense(latent_space_size, activation=tf.nn.relu)(flattened)

    return tf.keras.Model(inputs=inp, outputs=out, name=name)


def build_efficientnet_encoder(
    input_shape=(128, 128, 3),
    output_shape=128,
    name="encoder",
    weights="imagenet",
    trainable=False,
):
    efficientnet = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=weights,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    efficientnet_output = efficientnet.get_layer("top_activation").output
    base_model = tf.keras.Model(
        inputs=efficientnet.inputs,
        outputs=efficientnet_output,
        name="efficientnet_feature_extractor",
    )
    for layer in base_model.layers:
        layer.trainable = trainable

    inp = tf.keras.Input(input_shape)
    base_model_inp = tf.concat([inp, inp, inp], axis=-1)
    features = base_model(base_model_inp)
    flattened = tf.keras.layers.Flatten()(features)
    out = tf.keras.layers.Dense(output_shape, activation=None)(flattened)

    return tf.keras.Model(inputs=inp, outputs=out, name=name)


def build_encoder_deeper(
    input_shape=(128, 128, 3),
    output_shape=128,
    filters=(32, 64, 128),
    kernel_size=3,
    batch_normalization=False,
    activation=tf.nn.relu,
    name="encoder",
    num_repeat=2,
):

    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = inputs
    for i in range(len(filters)):
        for repeat_iter in range(num_repeat - 1):
            x = tf.keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_size,
                padding="same",
                activation=None,
            )(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=activation)(x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, padding="same", strides=(2, 2)
        )(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)
        # x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(output_shape, activation=None)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_decoder_deeper(
    input_shape=128,
    output_shape=(128, 128, 3),
    filters=(128, 64, 32),
    kernel_size=3,
    batch_normalization=False,
    activation=tf.nn.relu,
    name="decoder",
    num_repeat=2,
):

    n_upsamplings = len(filters)
    x_init, y_init = (
        output_shape[0] // 2 ** n_upsamplings,
        output_shape[1] // 2 ** n_upsamplings,
    )

    if (
        x_init * 2 ** n_upsamplings != output_shape[0]
        or y_init * 2 ** n_upsamplings != output_shape[1]
    ):
        print(
            "Output image dimensions should be divisible by 2^len(filters). Please set a suitable output_shape"
        )
        sys.exit()

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(x_init * y_init * filters[0], activation=activation)(
        inputs
    )
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((x_init, y_init, filters[0]))(x)
    for i in range(len(filters)):
        for repeat_iter in range(num_repeat - 1):
            x = tf.keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernel_size,
                padding="same",
                activation=None,
            )(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=activation)(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)
        x = tf.keras.layers.UpSampling2D()(x)
    for repeat_iter in range(num_repeat - 1):
        x = tf.keras.layers.Conv2D(
            filters=filters[-1],
            kernel_size=kernel_size,
            padding="same",
            activation=None,
        )(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    outputs = tf.keras.layers.Conv2D(
        filters=output_shape[2],
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
