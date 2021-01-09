import tensorflow as tf
import numpy as np


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
            for f in reversed(filters[:-1])
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
        print("num features: ", self.num_features)
        self.decode_structure_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )
        self.decode_longitudinal_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )

    def encode(self, image_batch, training=None):
        print("shape of encode input: ", image_batch.shape)
        x = image_batch
        for layer in self.downsample_conv:
            x = layer(x)
        x = tf.keras.layers.Flatten()(x)
        print("encode shape after flatter: ", tf.shape(x).numpy())
        structure = self.structure_dense(x)
        print("encode shape of structure: ", tf.shape(structure).numpy())
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


if __name__ == "__main__":
    FILTERS = [64, 128, 256, 512]
    KERNEL_SIZE = 3
    ACTIVATION = tf.nn.silu
    LAST_ACTIVATION = tf.nn.sigmoid
    STRUCTURE_VEC_SIZE = 100
    LONGITUDINAL_VEC_SIZE = 1

    model = AE(
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        activation=ACTIVATION,
        last_activation=LAST_ACTIVATION,
        structure_vec_size=STRUCTURE_VEC_SIZE,
        longitudinal_vec_size=LONGITUDINAL_VEC_SIZE,
    )
    print("model defined")
    input_tensor = tf.convert_to_tensor(np.zeros((3, 64, 64, 1)))
    input_tensor = tf.cast(input_tensor, tf.float32)
    _ = model(input_tensor)
    print("model first call")
    structure, state = model.encode(input_tensor)
    print("structure: ", tf.shape(structure))
    print("state: ", tf.shape(state))
    img = model.decode(structure, state)
    print("res: ", tf.shape(img))
    print("done")
