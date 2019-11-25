import tensorflow as tf
from dataset import Dataset

# dataset = Dataset()

x = tf.random.uniform([3, 3])


print(x.device)





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