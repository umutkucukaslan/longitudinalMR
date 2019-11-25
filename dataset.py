import numpy as np
import configparser
import os
import glob


class Dataset:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("./config.ini")
        self.dataset_path = config['Dataset'].get('dataset_path')
        self.images_dir = os.path.join(self.dataset_path, 'VOC2012/JPEGImages')
        self.annotations_dir = os.path.join(self.dataset_path, 'VOC2012/Annotations')

        self.random_select_images = config['Dataset'].getboolean('random_select_images')
        self.num_train_images = config['Dataset'].getint('num_train_images')
        self.num_val_images = config['Dataset'].getint('num_val_images')

        image_file_paths = glob.glob(self.images_dir + '/*.jpg')
        print(len(image_file_paths))



if __name__ == "__main__":
    d = Dataset()