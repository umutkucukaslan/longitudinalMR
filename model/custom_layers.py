import tensorflow as tf


class ProgressiveGANDownsample(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, alpha=0.2, **kwargs):
        super(ProgressiveGANDownsample, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.conv_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for f in filters:
            self.conv_layers.append(tf.keras.layers.Conv2D(f,
                                            self.kernel_size,
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
                                    )

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.max_pool = tf.keras.layers.MaxPool2D()

    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)
            # x = tf.math.l2_normalize(x, axis=-1, name='l2_normalize')
        x = self.max_pool(x)
        return x

    def get_config(self):
        config = super(ProgressiveGANDownsample, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'alpha': self.alpha})
        return config


class ProgressiveGANUpsample(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, alpha=0.2, **kwargs):
        super(ProgressiveGANUpsample, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.conv_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for f in filters:
            self.conv_layers.append(tf.keras.layers.Conv2D(f,
                                            self.kernel_size,
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
                                    )

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.upsample = tf.keras.layers.UpSampling2D()

    def call(self, inputs):
        x = self.upsample(inputs)
        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)
            # x = tf.math.l2_normalize(x, axis=-1, name='l2_normalize')
        return x

    def get_config(self):
        config = super(ProgressiveGANUpsample, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'alpha': self.alpha})
        return config


class Conv2D_LeakyReLU_L2normalize(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, alpha=0.2, **kwargs):
        super(Conv2D_LeakyReLU_L2normalize, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.conv_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for f in filters:
            self.conv_layers.append(tf.keras.layers.Conv2D(f,
                                            self.kernel_size,
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
                                    )

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)
            # x = tf.math.l2_normalize(x, axis=-1, name='l2_normalize')
        return x

    def get_config(self):
        config = super(Conv2D_LeakyReLU_L2normalize, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'alpha': self.alpha})
        return config


