import cv2
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
        **kwargs,
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
        x = tf.keras.layers.Flatten()(x)
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

    def interpolate(
        self,
        inputs1,
        inputs2,
        sample_points,
        structure_mix_type="mean",
        return_as_image=False,
    ):
        structure1, state1 = self.encode(inputs1, training=False)
        structure2, state2 = self.encode(inputs2, training=False)
        if structure_mix_type == "first":
            structure = structure1
        elif structure_mix_type == "second":
            structure = structure2
        elif structure_mix_type == "mean":
            structure = (structure1 + structure2) / 2.0
        else:
            raise ValueError(f"structure mix type {structure_mix_type} is unknown")
        diff = state2 - state1
        state_vecs = [state1 + diff * sample_point for sample_point in sample_points]
        interpolations = [
            self.decode(structure, state, training=False) for state in state_vecs
        ]
        if return_as_image:
            interpolations = [
                np.clip(x.numpy()[0, ...] * 255, 0, 255).astype(np.uint8)
                for x in interpolations
            ]
            interpolations = [
                cv2.putText(
                    x, str(round(p, 2)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255
                )
                for x, p in zip(interpolations, sample_points)
            ]
            interpolations = np.hstack(interpolations)
        return interpolations

    def train_for_patient(self, img1, img2):
        pass

    def reset_model(self):
        # should load default variables
        pass


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
