import tensorflow as tf
from dataset import get_train_dataset
from tensorflow.python.client import device_lib


def main():
    train_x, train_y = get_train_dataset()
    print("train_x is   ", train_x*10000)
    print("train_y is   ", train_y)
    print("Hello world")
    print("DEVICES")
    print(device_lib.list_local_devices())

main()