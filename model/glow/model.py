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
            if tf.math.is_nan(logdet).numpy():
                print(
                    f"NAN in ActNorm layer!  height: {height.numpy()}, width: {width.numpy()}, scale: {self.scale.numpy()}"
                )
                exit()

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
            if any(tf.math.is_nan(logdet).numpy()):
                print(
                    f"NAN in Invertible1x1Conv layer!  height: {height.numpy()}, width: {width.numpy()}, kernel: {self.kernel.numpy()}"
                )
                exit()

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
        self.s_sign = tf.Variable(
            initial_value=np.sign(s),
            dtype=tf.float32,
            trainable=False,
            name="s_sign",
        )
        self.log_s = tf.Variable(
            initial_value=tf.math.log(tf.abs(s)),
            dtype=tf.float32,
            trainable=True,
            name="log_s",
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
            @ (self.u * self.u_mask + tf.linalg.diag(self.s_sign * tf.exp(self.log_s)))
        )

    def call(self, inputs, training=None, **kwargs):
        kernel = self.calc_kernel()
        if training:
            height = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
            width = tf.cast(tf.shape(inputs)[2], dtype=tf.float32)
            logdet = height * width * tf.reduce_sum(tf.math.log(tf.abs(self.s)))
            self.add_loss(logdet)
            if tf.math.is_nan(logdet).numpy():
                print(
                    f"NAN in Invertible1x1ConvLU layer!  height: {height.numpy()}, width: {width.numpy()}, scale: {self.s.numpy()}"
                )
                exit()

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
                    filters=in_channels,
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
                if any(tf.math.is_nan(logs).numpy().flatten()):
                    print(f"NAN in AffineCoupling layer!  logs: {logs.numpy()}")
                    exit()

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
        return config


def gaussian_log_p(z, mean, log_sd):
    # compute prior probability given mean and log std of gaussian with sample
    return (
        -0.5 * tf.math.log(2 * np.pi)
        - log_sd
        - 0.5 * tf.pow(z - mean, 2) / tf.exp(2 * log_sd)
    )


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        num_flows,
        num_filters=512,
        use_lu_decom=True,
        affine=True,
        split=True,
        **kwargs,
    ):
        super(Block, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.num_flows = num_flows
        self.squeeze_dim = in_channels * 4
        self.num_filters = num_filters
        self.use_lu_decom = use_lu_decom
        self.affine = affine
        self.split = split
        self.flows = [
            Flow(
                in_channels=self.squeeze_dim,
                num_filters=num_filters,
                use_lu_decom=use_lu_decom,
                affine=affine,
            )
            for _ in range(num_flows)
        ]
        if split:
            self.prior = tf.keras.layers.Conv2D(
                filters=self.squeeze_dim,
                kernel_size=3,
                padding="same",
                use_bias=True,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
            self.out_channels = self.squeeze_dim // 2
        else:
            self.prior = tf.keras.layers.Conv2D(
                filters=self.squeeze_dim * 2,
                kernel_size=3,
                padding="same",
                use_bias=True,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
            self.out_channels = self.squeeze_dim
            self.out_shape = None

    def build(self, input_shape):
        self.out_shape = [input_shape[1] // 2, input_shape[2] // 2, self.out_channels]

    def call(self, inputs, training=None):
        batch_size, height, width, channels = tf.shape(inputs).numpy()
        x = tf.reshape(inputs, [batch_size, height // 2, 2, width // 2, 2, channels])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [batch_size, height // 2, width // 2, channels * 4])
        for flow in self.flows:
            x = flow(x, training=training)
        if self.split:
            out, new_z = tf.split(x, num_or_size_splits=2, axis=-1)
            if training:
                mean, log_sd = tf.split(
                    self.prior(out, training=training), num_or_size_splits=2, axis=-1
                )
                log_p = gaussian_log_p(new_z, mean, log_sd)
                self.add_loss(tf.reduce_sum(log_p))
                if any(tf.math.is_nan(log_p).numpy().flatten()):
                    print(f"NAN in Block layer!  logp: {log_p.numpy()}")
                    exit()

        else:
            new_z = x
            out = x
            if training:
                mean, log_sd = tf.split(
                    self.prior(tf.zeros_like(new_z), training=training),
                    num_or_size_splits=2,
                    axis=-1,
                )
                log_p = gaussian_log_p(new_z, mean, log_sd)
                self.add_loss(tf.reduce_sum(log_p))
                if any(tf.math.is_nan(log_p).numpy().flatten()):
                    print(f"NAN in Block layer!  height: {log_p.numpy()}")
                    exit()

        return out, new_z

    def reverse(self, inputs, z=None, reconstruct=True):
        if self.split:
            if z is None:
                raise ValueError("z cannot be None when split is True")
            if not reconstruct:
                z = self.sample_z_vector(inputs, z)
            x = tf.concat([inputs, z], axis=-1)
        else:
            x = inputs
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        batch_size, height, width, channels = tf.shape(x).numpy()
        x = tf.reshape(x, [batch_size, height, width, 2, 2, channels // 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [batch_size, height * 2, width * 2, channels // 4])
        return x

    def get_apriori_distribution(self, inputs):
        """
        Computes mean/stf of the learned a priori gaussian distribution for given inputs
        :param inputs: tensor corresponding to the block's output
        :return: mean, log_sd
        """
        if self.split:
            prior_input = inputs
        else:
            prior_input = tf.zeros_like(inputs)
        mean, log_sd = tf.split(self.prior(prior_input), num_or_size_splits=2, axis=-1)
        return mean, log_sd

    def sample_z_vector(self, inputs, random_sample=None):
        """
        Random sample from the learned a priori gaussian distribution for given inputs
        :param inputs: tensor corresponding to the block's output
        :param random_sample: random variable from N(0,1) to generate sample, if None generates itself
        :return: random sample that has the same shape as inputs
        """
        if random_sample is None:
            random_sample = tf.random.normal(
                tf.shape(inputs), mean=0, stddev=1, dtype=tf.float32
            )
        mean, log_sd = self.get_apriori_distribution(inputs)
        return mean + tf.exp(log_sd) * random_sample

    def get_config(self):
        config = super(Block, self).get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "num_flows": self.num_flows,
                "num_filters": self.num_filters,
                "use_lu_decom": self.use_lu_decom,
                "affine": self.affine,
                "split": self.split,
            }
        )
        return config


class Glow(tf.keras.Model):
    def __init__(
        self,
        in_channels,
        num_blocks,
        num_flows,
        num_filters=512,
        use_lu_decom=True,
        affine=True,
        split=True,
        **kwargs,
    ):
        super(Glow, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.num_flows = num_flows
        self.num_filters = num_filters
        self.use_lu_decom = use_lu_decom
        self.affine = affine
        self.split = split
        self.blocks = []
        self.block_in_channels = []
        self.block_output_shapes = []
        for i in range(num_blocks - 1):
            block = Block(
                in_channels=in_channels,
                num_flows=num_flows,
                num_filters=num_filters,
                use_lu_decom=use_lu_decom,
                affine=affine,
                split=split,
            )
            self.blocks.append(block)
            self.block_in_channels.append(in_channels)
            in_channels = in_channels * 2 if split else in_channels * 4
        block = Block(
            in_channels=in_channels,
            num_flows=num_flows,
            num_filters=num_filters,
            use_lu_decom=use_lu_decom,
            affine=affine,
            split=False,
        )
        self.blocks.append(block)
        self.block_in_channels.append(in_channels)

    def call(self, inputs, training=None, mask=None):
        out = inputs
        z_outs = []
        for block in self.blocks:
            out, new_z = block(out, training=training)
            if block.split:
                z_outs.append(new_z)
        z_outs.append(out)
        return z_outs

    def reverse(self, z_list, reconstruct=True):
        out = z_list[-1]
        idx = -2
        for block in reversed(self.blocks):
            z = None
            if block.split:
                z = z_list[idx]
                idx -= 1
            out = block.reverse(inputs=out, z=z, reconstruct=reconstruct)
        return out

    def flatten_z_list(self, z_list):
        flattened = []
        for z in z_list:
            flattened.append(tf.reshape(z, [tf.shape(z)[0], -1]))
        flattened = tf.concat(flattened, axis=-1)
        return flattened

    def unflatten_z_list(self, z):
        z_shapes = [block.out_shape for block in self.blocks if block.split] + [
            self.blocks[-1].out_shape
        ]
        z_sizes = [s[0] * s[1] * s[2] for s in z_shapes]
        batch_size, z_size = tf.shape(z).numpy()
        assert z_size == sum(z_sizes)
        z_list = tf.split(z, num_or_size_splits=z_sizes, axis=-1)
        for i in range(len(z_list)):
            z_list[i] = tf.reshape(z_list[i], [batch_size] + z_shapes[i])
        return z_list

    def get_config(self):
        return {
            "in_channels": self.in_channels,
            "num_blocks": self.num_blocks,
            "num_flows": self.num_flows,
            "num_filters": self.num_filters,
            "use_lu_decom": self.use_lu_decom,
            "affine": self.affine,
            "split": self.split,
        }


if __name__ == "__main__":
    # block = Block(
    #     in_channels=1,
    #     num_flows=32,
    #     num_filters=512,
    #     use_lu_decom=True,
    #     affine=True,
    #     split=True,
    # )
    # input_tensor = tf.convert_to_tensor(np.random.rand(1, 4, 4, 1), dtype=tf.float32)
    # outputs = block(input_tensor, training=True)
    # out, z = block(input_tensor, training=True)
    # back = block.reverse(out, z)
    # N = 25
    # for i in range(N):
    #     input_tensor = tf.convert_to_tensor(
    #         np.random.rand(1, 4, 4, 1), dtype=tf.float32
    #     )
    #     out, z = block(input_tensor, training=True)
    #     back = block.reverse(out, z)
    #     inp = input_tensor.numpy()
    #     b = back.numpy()
    #     d = np.abs(inp - b)
    #     rtol = np.max((d / inp).flatten())
    #     atol = np.max(d.flatten())
    #
    #     print(
    #         f"block reverse equality: {np.allclose(input_tensor.numpy(), back.numpy(), rtol=1e-4, atol=1e-8)}    ->  "
    #         f"atol: {atol}    rtol:{rtol}"
    #     )
    # print("input: ", input_tensor)
    # print("back: ", back)
    # print("output: ", out)
    # print("z: ", z)
    # exit()

    glow = Glow(
        in_channels=1,
        num_blocks=4,
        num_flows=32,
        num_filters=512,
        use_lu_decom=True,
        affine=True,
        split=True,
    )

    input_tensor = tf.convert_to_tensor(
        np.random.rand(1, 64, 64, 1) * 127 + 127, dtype=tf.float32
    )
    outputs = glow(input_tensor, training=True)
    outputs = glow(input_tensor, training=True)
    print("losses collected: ")
    for l in glow.losses:
        print(l)
    exit()
    print("")
    print("inputs: ", input_tensor)
    print("output: ", outputs)
    print("flattened output: ", glow.flatten_z_list(outputs))
    reshaped_outputs = glow.unflatten_z_list(glow.flatten_z_list(outputs))
    print("reshaped outputs: ", reshaped_outputs)
    for o, r in zip(outputs, reshaped_outputs):
        print(
            f"is flattening equal: {np.allclose(o.numpy(), r.numpy())}  shape: {tf.shape(o).numpy()}"
        )
    print("")
    print("input - reverse(output) equality")
    back = glow.reverse(outputs)
    print(
        f"is equal: {np.allclose(input_tensor.numpy(), back.numpy())}  ->  shape of back: {tf.shape(back).numpy()}"
    )
    inp = input_tensor.numpy()
    b = back.numpy()
    d = np.abs(inp - b)
    rtol = np.max((d / inp).flatten())
    atol = np.max(d.flatten())
    print(f"rtol: {rtol} ,    atol: {atol}")
    print("")
    print(f"trainable vars: {len(glow.trainable_variables)}")
    # for var in glow.trainable_variables:
    #     print(var)
    print(f"non-trainable vars: {len(glow.non_trainable_variables)}")
    # for var in glow.non_trainable_variables:
    #     print(var)
    # print("z new: ", z_new)
    # glow.summary()
    # save_dir = "/Users/umutkucukaslan/Desktop/sil"
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # model_path = os.path.join(save_dir, "my_model_disco")
    # model_plot_path = os.path.join(model_path, "plot.jpg")
    # tf.keras.utils.plot_model(
    #     glow, to_file=model_plot_path, show_shapes=True, expand_nested=True, dpi=150
    # )
    # print("model plot done")
    # back = block.reverse(outputs, z_new)
    # print("back: ", back)
    # print("is equal: ", np.allclose(input_tensor.numpy(), back.numpy()))
    # sample = block.sample_z_vector(outputs)
    # print("sample: ", sample)

    exit()
    flow = Flow(in_channels=4, num_filters=512, use_lu_decom=True, affine=True)
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
