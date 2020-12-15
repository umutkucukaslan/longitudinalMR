import os

import tensorflow as tf
import numpy as np


class ActNorm(tf.keras.layers.Layer):
    def __init__(self, inp_shape, **kwargs):
        super(ActNorm, self).__init__(**kwargs)
        self.inp_shape = inp_shape
        self.scale = self.add_weight(
            shape=(1, 1, inp_shape[-1]),
            trainable=True,
            initializer="ones",
            name="scale",
        )
        self.bias = self.add_weight(
            shape=(1, 1, inp_shape[-1]),
            trainable=True,
            initializer="zeros",
            name="bias",
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
        config.update({"inp_shape": self.inp_shape})
        return config


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
