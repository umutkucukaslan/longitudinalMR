import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging

from dataset import get_autoencoder_dataset


config = configparser.ConfigParser()
config.read("./config.ini")

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(config['Logging'].get('log_file_pth'))
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info('merhaba')


tf.enable_eager_execution()


def show_batch(image_batch):
    plt.figure(figsize=(30, 30))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(np.squeeze(image_batch[n]), cmap='gray')
        plt.axis('off')


train_ds, val_ds, test_ds = get_autoencoder_dataset()
logger.info('Dataset for autoencoder created.')
image_batch, label_batch = next(iter(train_ds))
show_batch(image_batch=image_batch)
show_batch(image_batch=label_batch)



# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print('labels')
# print(y_train)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # model.fit(x_train, y_train, epochs=5)
# # model.evaluate(x_test, y_test, verbose=2)