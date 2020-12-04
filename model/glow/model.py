import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def build(self, input_shapes):
        self.w = self.add_weight(input_shapes=(128, 127, 2))

        pass

    def set(self):
        self.add
        self.add_loss()

    def call(self):
        pass


def create_pistons():
    print("ne dedin birader?")


create_pistons()
