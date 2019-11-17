import tensorflow as tf
from dataset import get_train_dataset
from tensorflow.python.client import device_lib

with tf.device('/gpu:0'):
    mnist = tf.keras.datasets.mnist


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)