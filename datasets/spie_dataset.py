import os
import random
from math import exp, log

import cv2
import numpy as np

from datasets.longitudinal_dataset import LongitudinalDataset


class Example:
    def __init__(
        self,
        id: int = -1,
        weight: float = 1.0,
        loss: float = 0.0,
        path=None,
        preprocess_fn=None,
    ):
        self.preprocess_fn = preprocess_fn
        self.path = path
        self.id = id
        self.weight = weight
        self.loss = loss

    def get_weighted_loss(self):
        return self.weight * self.loss

    def update(self, alpha):
        self.weight = self.weight * exp(-alpha * self.loss)

    def get_image(self):
        image = cv2.imread(self.path)
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        return image


class SPIEDataset:
    def __init__(self, train_ims, val_ims, test_ims, preprocess_fn=None):
        self.preprocess_fn = preprocess_fn
        self.train = []
        for i in range(len(train_ims)):
            self.train.append(
                Example(
                    id=i,
                    weight=1 / len(train_ims),
                    loss=0.0,
                    path=train_ims[i],
                    preprocess_fn=preprocess_fn,
                )
            )
        self.val = []
        for i in range(len(val_ims)):
            self.val.append(
                Example(
                    id=i,
                    weight=1 / len(val_ims),
                    loss=0.0,
                    path=val_ims[i],
                    preprocess_fn=preprocess_fn,
                )
            )
        self.test = []
        for i in range(len(test_ims)):
            self.test.append(
                Example(
                    id=i,
                    weight=1 / len(test_ims),
                    loss=0.0,
                    path=test_ims[i],
                    preprocess_fn=preprocess_fn,
                )
            )

    def _get_correct_data(self, split="train"):
        if split == "train":
            data = self.train
        elif split == "val":
            data = self.val
        elif split == "test":
            data = self.test
        else:
            raise ValueError
        return data

    def _get_batch(self, indices, split="train"):
        data = self._get_correct_data(split)
        examples = [data[x] for x in indices]
        example_images = [example.get_image() for example in examples]
        example_images = np.stack(example_images, axis=0)
        weights = [example.weight for example in examples]
        weights = np.stack(weights, axis=0)
        info = {"split": split, "indices": indices}
        return example_images, info, weights

    def _get_images(self, batch_size=1, shuffle=False, split=None):
        data = self._get_correct_data(split)
        indices = list(range(len(data)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices) - batch_size, batch_size):
            try:
                yield self._get_batch(indices[i : i + batch_size], split=split)
            except Exception as e:
                print("EXCEPTION OCCURED DURING GETTING DATA: ", e)
                exit()

    def get_training_images(self, batch_size=1, shuffle=False):
        return self._get_images(batch_size, shuffle, split="train")

    def get_val_images(self, batch_size=1, shuffle=False):
        return self._get_images(batch_size, shuffle, split="val")

    def get_test_images(self, batch_size=1, shuffle=False):
        return self._get_images(batch_size, shuffle, split="test")

    def update_losses(self, info, losses):
        split, indices = info["split"], info["indices"]
        data = self._get_correct_data(split)
        for i in range(len(indices)):
            example = data[indices[i]]
            example.loss = losses[i]

    def _update_weights(self, split="train"):
        data = self._get_correct_data(split)
        mean_loss = 0.0
        for example in data:
            mean_loss += example.get_weighted_loss()
        alpha = 0.5 * log((1 - mean_loss) / (mean_loss + 1e-9))
        for example in data:
            example.update(alpha)

    def update_training_weights(self):
        self._update_weights("train")

    def update_val_weights(self):
        self._update_weights("val")

    def update_test_weights(self):
        self._update_weights("test")


def get_spie_dataset(
    folder_name="training_data_15T_192x160_4slices", machine="none",
):
    if machine == "colab":
        data_dir = os.path.join("/content", folder_name)
    elif machine == "cloud":
        data_dir = os.path.join("/home/umutkucukaslan/data", folder_name)
    else:
        data_dir = os.path.join(
            "/Users/umutkucukaslan/Desktop/thesis/dataset", folder_name
        )
    train_data_dir = os.path.join(data_dir, "train")
    val_data_dir = os.path.join(data_dir, "val")
    test_data_dir = os.path.join(data_dir, "test")

    train_long = LongitudinalDataset(data_dir=train_data_dir)
    val_long = LongitudinalDataset(data_dir=val_data_dir)
    test_long = LongitudinalDataset(data_dir=test_data_dir)

    train_images = (
        train_long.get_ad_images()
        + train_long.get_mci_images()
        + train_long.get_cn_images()
    )
    val_images = (
        val_long.get_ad_images() + val_long.get_mci_images() + val_long.get_cn_images()
    )
    test_images = (
        test_long.get_ad_images()
        + test_long.get_mci_images()
        + test_long.get_cn_images()
    )

    def preprocess_fn(image: np.ndarray):
        """
        preprocess function for SPIE paper model.
        Image size 64x64x1
        Image range: [-1, 1]
        No augmentation
        :param image:
        :return:
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.squeeze(image)
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float)
        image = (image - 127.0) / 128
        image = np.expand_dims(image, axis=-1)
        return image

    spie_dataset = SPIEDataset(
        train_images, val_images, test_images, preprocess_fn=preprocess_fn
    )
    return spie_dataset
