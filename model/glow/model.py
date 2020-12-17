import os

import tensorflow as tf
import numpy as np
from scipy.linalg import lu as lu_decomposition


class ActNorm(tf.keras.layers.Layer):
    def __init__(self, in_channels, **kwargs):
        super(ActNorm, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.scale = self.add_weight(
            shape=(1, 1, in_channels), trainable=True, initializer="ones", name="scale",
        )
        self.bias = self.add_weight(
            shape=(1, 1, in_channels), trainable=True, initializer="zeros", name="bias",
        )
        self.initialized = False

    def initialize(self, inputs: tf.Tensor) -> None:
        self.bias.assign(
            -tf.reshape(
                tf.reduce_mean(inputs, axis=[0, 1, 2]), shape=(1, 1, inputs.shape[-1])
            )
        )
        self.scale.assign(
            tf.reshape(
                1 / (tf.math.reduce_std(inputs, axis=[0, 1, 2]) + 1e-6),
                shape=(1, 1, inputs.shape[-1]),
            )
        )
        self.initialized = True

    def call(self, inputs, training=False, **kwargs):
        if not self.initialized:
            self.initialize(inputs)
        if training:
            height = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
            width = tf.cast(tf.shape(inputs)[2], dtype=tf.float32)
            logdet = height * width * tf.reduce_sum(tf.math.log(tf.abs(self.scale)))
            self.add_loss(logdet)

        return self.scale * (inputs + self.bias)

    def reverse(self, inputs):
        return inputs / self.scale - self.bias

    def get_config(self):
        config = super(ActNorm, self).get_config()
        config.update({"in_channels": self.in_channels})
        return config


class Invertible1x1Conv(tf.keras.layers.Layer):
    def __init__(self, in_channels, **kwargs):
        """
        1x1 conv layer with random rotation matrix initialization
        :param in_channels:
        :param kwargs:
        """
        super(Invertible1x1Conv, self).__init__(**kwargs)
        self.in_channels = in_channels
        W = tf.random.normal((in_channels, in_channels), dtype=tf.float32)
        q, r = tf.linalg.qr(W)
        self.kernel = tf.Variable(initial_value=q, trainable=True, name="1x1_kernel")

    def call(self, inputs, training=None, **kwargs):
        if training:
            height = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
            width = tf.cast(tf.shape(inputs)[2], dtype=tf.float32)
            logdet = height * width * tf.linalg.slogdet(self.kernel)[1]
            self.add_loss(logdet)
        return tf.nn.conv2d(
            input=inputs,
            filters=tf.reshape(self.kernel, (1, 1, self.in_channels, self.in_channels)),
            strides=1,
            padding="SAME",
        )

    def reverse(self, inputs):
        inv_kernel = tf.linalg.inv(self.kernel)
        return tf.nn.conv2d(
            input=inputs,
            filters=tf.reshape(inv_kernel, (1, 1, self.in_channels, self.in_channels)),
            strides=1,
            padding="SAME",
        )

    def get_config(self):
        config = super(Invertible1x1Conv, self).get_config()
        config.update({"in_channels": self.in_channels})
        return config


class Invertible1x1ConvLU(tf.keras.layers.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        W = np.random.rand(in_channels, in_channels)
        q, r = np.linalg.qr(W)
        p, l, u = lu_decomposition(q)
        u_mask = np.triu(np.ones_like(u)) - np.eye(in_channels)
        l_mask = np.transpose(u_mask)
        s = np.diag(u)
        self.s = tf.Variable(
            initial_value=s, dtype=tf.float32, name="s", trainable=True
        )
        self.u = tf.Variable(
            initial_value=u, dtype=tf.float32, name="u", trainable=True
        )
        self.l = tf.Variable(
            initial_value=l, dtype=tf.float32, name="l", trainable=True
        )
        self.u_mask = tf.Variable(
            initial_value=u_mask, dtype=tf.float32, name="u_mask", trainable=False
        )
        self.l_mask = tf.Variable(
            initial_value=l_mask, dtype=tf.float32, name="l_mask", trainable=False
        )
        self.p = tf.Variable(
            initial_value=p, dtype=tf.float32, name="p", trainable=False
        )
        self.eye = tf.Variable(
            initial_value=np.eye(in_channels),
            dtype=tf.float32,
            name="eye",
            trainable=False,
        )

    def calc_kernel(self):
        return (
            self.p
            @ (self.l * self.l_mask + self.eye)
            @ (self.u * self.u_mask + tf.linalg.diag(self.s))
        )

    def call(self, inputs, training=None, **kwargs):
        kernel = self.calc_kernel()
        if training:
            height = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
            width = tf.cast(tf.shape(inputs)[2], dtype=tf.float32)
            logdet = height * width * tf.reduce_sum(tf.math.log(tf.abs(self.s)))
            self.add_loss(logdet)
        return tf.nn.conv2d(
            input=inputs,
            filters=tf.reshape(kernel, (1, 1, self.in_channels, self.in_channels)),
            strides=1,
            padding="SAME",
        )

    def reverse(self, inputs):
        kernel = self.calc_kernel()
        inv_kernel = tf.linalg.inv(kernel)
        return tf.nn.conv2d(
            input=inputs,
            filters=tf.reshape(inv_kernel, (1, 1, self.in_channels, self.in_channels)),
            strides=1,
            padding="SAME",
        )

    def get_config(self):
        config = super(Invertible1x1ConvLU, self).get_config()
        config.update({"in_channels": self.in_channels})
        return config


class AffineCouling(tf.keras.layers.Layer):
    def __init__(self, num_filters, in_channels, affine=True, **kwargs):
        super(AffineCouling, self).__init__(**kwargs)
        self.affine = affine
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.net = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    use_bias=True,
                    kernel_initializer="random_normal",
                ),
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    activation="relu",
                    use_bias=True,
                    kernel_initializer="random_normal",
                ),
                tf.keras.layers.Conv2D(
                    filters=in_channels // 2,
                    kernel_size=3,
                    padding="same",
                    activation=None,
                    use_bias=True,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
            ]
        )

    def call(self, inputs, training=None):
        x_a, x_b = tf.split(inputs, num_or_size_splits=2, axis=-1)
        if self.affine:
            logs, t = tf.split(
                self.net(x_b, training=training), num_or_size_splits=2, axis=-1
            )
            s = tf.exp(logs)
            y_a = s * x_a + t
            y_b = x_b
            y = tf.concat([y_a, y_b], axis=-1)
            if training:
                self.add_loss(tf.reduce_sum(logs))
            return y
        else:
            logs, t = tf.split(
                self.net(x_b, training=training), num_or_size_splits=2, axis=-1
            )
            y_a = x_a + t
            y_b = x_b
            y = tf.concat([y_a, y_b], axis=-1)
            return y

    def reverse(self, inputs):
        x_a, x_b = tf.split(inputs, num_or_size_splits=2, axis=-1)
        if self.affine:
            logs, t = tf.split(
                self.net(x_b, training=False), num_or_size_splits=2, axis=-1
            )
            s = tf.exp(logs)
            y_a = (x_a - t) / s
            y_b = x_b
            y = tf.concat([y_a, y_b], axis=-1)
            return y
        else:
            logs, t = tf.split(
                self.net(x_b, training=False), num_or_size_splits=2, axis=-1
            )
            y_a = x_a - t
            y_b = x_b
            y = tf.concat([y_a, y_b], axis=-1)
            return y

    def get_config(self):
        config = super(AffineCouling, self).get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "in_channels": self.in_channels,
                "affine": self.affine,
            }
        )
        return config


class Flow(tf.keras.layers.Layer):
    def __init__(
        self, in_channels, num_filters=512, use_lu_decom=True, affine=True, **kwargs
    ):
        super(Flow, self).__init__(**kwargs)
        self.affine = affine
        self.use_lu_decom = use_lu_decom
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.actnorm = ActNorm(in_channels)
        if use_lu_decom:
            self.invertible_conv = Invertible1x1ConvLU(in_channels)
        else:
            self.invertible_conv = Invertible1x1Conv(in_channels)
        self.affine_coupling = AffineCouling(
            num_filters=num_filters, in_channels=in_channels, affine=affine
        )

    def call(self, inputs, training=None, **kwargs):
        x = self.actnorm(inputs, training=training)
        x = self.invertible_conv(x, training=training)
        x = self.affine_coupling(x, training=training)
        return x

    def reverse(self, inputs):
        x = self.affine_coupling.reverse(inputs)
        x = self.invertible_conv.reverse(x)
        x = self.actnorm.reverse(x)
        return x

    def get_config(self):
        config = super(Flow, self).get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "num_filters": self.num_filters,
                "use_lu_decom": self.use_lu_decom,
                "affine": self.affine,
            }
        )


class FlowBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_flows, **kwargs):
        super(FlowBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.num_flows = num_flows


class MyModel(tf.keras.Model):
    def __init__(self, inp_shape, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.inp_shape = inp_shape
        self.actnormlayers = [ActNorm(inp_shape=inp_shape) for x in range(3)]
        self.const = tf.Variable(
            initial_value=5, dtype=tf.float32, trainable=False, name="model_const"
        )

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for l in self.actnormlayers:
            x = l(x, training=training)
        return x

    @tf.function
    def reverse(self, inputs):
        x = inputs
        for l in reversed(self.actnormlayers):
            x = l.reverse(x)
        return x

    def get_config(self):
        return {"inp_shape": self.inp_shape}


if __name__ == "__main__":

    flow = Flow(in_channels=4, num_filters=512, use_lu_decom=True, affine=True)

    input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 4), dtype=tf.float32)
    outputs = flow(input_tensor)
    print("inputs: ", input_tensor)
    print("output: ", outputs)
    back = flow.reverse(outputs)
    print("back: ", back)

    exit()
    l = Invertible1x1ConvLU(in_channels=3)
    print("")

    exit()
    input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 3), dtype=tf.float32)
    outputs = l(input_tensor, training=True)
    print("losses collected: ", l.losses)
    print("input tensor: ", input_tensor)
    print("outputs: ", outputs)
    back = l.reverse(outputs)
    print("back: ", back)

    exit()

    save_dir = "/Users/umutkucukaslan/Desktop/sil"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, "my_model_disco")
    ckpt_path = os.path.join(save_dir, "my_model_disco_ckpt")
    print("instantiating model object...")
    model = MyModel(inp_shape=(2, 2, 3))
    print("(done) instantiating model object...")
    print(f"is model built: {model.built}")
    print("no call yet")
    print("model layer weights:")
    for layer in model.layers:
        print(layer.scale)
        print(layer.bias)

    # model.build(input_shape=(None, 2, 2, 3))
    # model = tf.keras.models.Sequential(
    #     [
    #         ActNorm(return_logdet=False),
    #         ActNorm(return_logdet=False),
    #         ActNorm(return_logdet=False),
    #     ]
    # )
    # inp = tf.keras.Input(shape=(2, 2, 3))
    # x = ActNorm(return_logdet=False)(inp)
    # x = ActNorm(return_logdet=False)(x)
    # x = ActNorm(return_logdet=False)(x)
    # model = tf.keras.models.Model(inputs=inp, outputs=x)
    input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 3), dtype=tf.float32)
    print("calling model with random input")
    outputs = model(input_tensor)
    print("after first call")
    print("model layer weights:")
    for layer in model.layers:
        print(layer.scale)
        print(layer.bias)
    print("calling the same input second time")
    outputs = model(input_tensor)
    # print(model.to_json())
    # model.from_config()

    # input_tensor = tf.convert_to_tensor(np.random.rand(2, 2, 2, 3), dtype=tf.float32)
    # outputs = model(input_tensor)
    print("inputs: ", input_tensor)
    print("outputs: ", outputs)
    np.save(os.path.join(save_dir, "input_tensor"), input_tensor.numpy())
    np.save(os.path.join(save_dir, "output_tensor"), outputs.numpy())
    print("input and output tensors are saved as np arrays")
    model.summary()

    # input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 3), dtype=tf.float32)
    # outputs = model(input_tensor)
    # print("inputs: ", input_tensor)
    # print("outputs: ", outputs)
    # model.summary()

    print("trainable vars:")
    for var in model.trainable_variables:
        print(var)
    print("non trainable vars:")
    for var in model.non_trainable_variables:
        print(var)
    #
    back = model.reverse(outputs)
    print("back: ", back)
    # model.save(model_path + ".h5")
    print(f"saving model to {model_path}")
    model.save(model_path)

    print(f"checkpointint model to {ckpt_path}")
    ckpt = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_path, max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        print(f"restoring from {checkpoint_manager.latest_checkpoint}")
        ckpt.restore(checkpoint_manager.latest_checkpoint)

    checkpoint_manager.save()

    print("NEW INOUT OUTPUT BEFORE SAVING")
    input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 3), dtype=tf.float32)
    outputs = model(input_tensor)
    back = model.reverse(outputs)
    print("inputs: ", input_tensor)
    print("outputs: ", outputs)
    print("back: ", back)

    print("restoring a new model from checkpoint")
    del model, ckpt, checkpoint_manager

    model = MyModel(inp_shape=(2, 2, 3))
    outputs = model(tf.convert_to_tensor(np.random.rand(1, 2, 2, 3), dtype=tf.float32))
    ckpt = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_path, max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        print(f"restoring from {checkpoint_manager.latest_checkpoint}")
        ckpt.restore(checkpoint_manager.latest_checkpoint)

    outputs = model(input_tensor)
    back = model.reverse(outputs)
    print("inputs: ", input_tensor)
    print("outputs: ", outputs)
    print("back: ", back)

    print("")
    print("model.to_json():")
    print(model.to_json())
    print("ckpt:")
    print(ckpt)
    # model.save_weights(model_path)

    # new_model = tf.keras.models.load_model(model_path)
    # print("new model")
    # new_model.summary()

    # layer = ActNorm(name="aysegul")
    # print("--------------")
    # print("layer weights: ", layer.weights)
    # print("trainable vars: ", layer.trainable_variables)
    # print("non-trainable vars: ", layer.non_trainable_variables)
    # print("--------------")
    #
    # input_tensor = tf.convert_to_tensor(np.random.rand(1, 2, 2, 4), dtype=tf.float32)
    # print("input tensor:", input_tensor)
    #
    # outputs, logdet = layer(input_tensor)
    # print("outputs : ", outputs)
    # print("mean output: ", tf.reduce_mean(outputs, axis=[0, 1, 2]))
    # print("std output: ", tf.math.reduce_std(outputs, axis=[0, 1, 2]))
    #
    # back = layer.reverse(outputs)
    # print("reversed output: ", back)
    #
    # print("done")
    #
    # print("--------------")
    # print("layer weights: ", layer.weights)
    # print("trainable vars: ", layer.trainable_variables)
    # print("non-trainable vars: ", layer.non_trainable_variables)
    # print("--------------")
    # print("\n\n\n\n\n\n\n")
    # print("layer weights: ")
    # for w in layer.weights:
    #     print(f"{w.trainable}: {w}")
    #
    # print("training")
    # lr = 0.1
    # for i in range(10000):
    #     with tf.GradientTape() as tape:
    #         outputs, logdet = layer(input_tensor)
    #         loss = tf.reduce_mean(tf.pow(tf.ones_like(outputs) - outputs, 2))
    #
    #     grads = tape.gradient(loss, layer.trainable_variables)
    #     for w, g in zip(layer.trainable_variables, grads):
    #         w.assign_add(-g * lr)
    #
    # print("input tensor:", input_tensor)
    #
    # outputs, logdet = layer(input_tensor)
    # print("outputs : ", outputs)
    #
    # back = layer.reverse(outputs)
    # print("reversed output: ", back)
    #
    # print("trainable variables: ", layer.trainable_variables)
