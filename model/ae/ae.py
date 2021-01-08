import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size,
        activation=tf.nn.silu,
        last_activation=tf.nn.sigmoid,
        structure_vec_size=100,
        longitudinal_vec_size=1,
        **kwargs
    ):
        super(AE, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.structure_vec_size = structure_vec_size
        self.longitudinal_vec_size = longitudinal_vec_size
        self.downsample_conv = [
            tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                strides=2,
                activation=activation,
            )
            for f in filters
        ]
        self.structure_dense = tf.keras.layers.Dense(
            structure_vec_size, activation=tf.nn.tanh, use_bias=False
        )
        self.longitudinal_dense = tf.keras.layers.Dense(
            longitudinal_vec_size, activation=None, use_bias=False
        )
        self.upsample_layers = [
            tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                strides=1,
                activation=activation,
            )
            for f in reversed(filters)[1:]
        ]
        self.upsample_layers.append(
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                strides=1,
                activation=last_activation,
            )
        )

    def build(self, input_shape):
        _, height, width, channels = input_shape
        num_features = (
            height
            // (2 ** len(self.filters))
            * width
            // (2 ** len(self.filters))
            * self.filters[-1]
        )
        self.decode_input_shape = [
            height // (2 ** len(self.filters)),
            width // (2 ** len(self.filters)),
            self.filters[-1],
        ]
        self.num_features = num_features
        self.decode_structure_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )
        self.decode_longitudinal_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )

    def encode(self, image_batch, training=None):
        x = image_batch
        for layer in self.downsample_conv:
            x = layer(x)
        structure = self.structure_dense(x)
        longitudinal_state = self.longitudinal_dense(x)
        return structure, longitudinal_state

    def decode(self, structure, longitudinal_state, training=None):
        x = self.decode_structure_dense(structure)
        y = self.decode_longitudinal_dense(longitudinal_state)
        x = tf.math.add(x, y)
        x = tf.reshape(x, shape=[-1] + self.decode_input_shape)
        for up_conv in self.upsample_layers:
            x = tf.keras.layers.UpSampling2D()(x)
            x = up_conv(x)
        return x

    def call(self, inputs, training=None, mask=None):
        structure, state = self.encode(inputs, training=training)
        out = self.decode(structure, state, training=training)
        return out
