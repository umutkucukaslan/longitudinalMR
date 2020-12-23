import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from datasets.longitudinal_dataset import LongitudinalDataset


class PairDataset(Dataset):
    def __init__(self, pairs, target_shape=None):
        self.target_shape = target_shape
        self.pairs = pairs
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        pair = [cv2.imread(x) for x in pair]
        if self.target_shape:
            height, width, channels = self.target_shape
            pair = [cv2.resize(x, (width, height)) for x in pair]
            if channels == 1 and pair[0].ndim == 3 and pair[0].shape[2] == 3:
                pair = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in pair]
                pair = [np.expand_dims(np.squeeze(x), axis=-1) for x in pair]
            if channels == 3 and pair[0].ndim == 2:
                pair = [cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in pair]
            if channels == 3 and pair[0].ndim == 3 and pair[0].shape[2] == 1:
                pair = [cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in pair]
        pair = [self.to_tensor(x) for x in pair]
        return pair
        # return {"img1": pair[0], "img2": pair[1]}


def get_images_adni_15t_dataset_torch(
    folder_name="training_data_15T_192x160_4slices", machine="none", target_shape=None,
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

    train_pairs = (
        train_long.get_ad_image_pairs()
        + train_long.get_mci_image_pairs()
        + train_long.get_cn_image_pairs()
    )
    train_pairs = [x[0] for x in train_pairs]
    val_pairs = (
        val_long.get_ad_image_pairs()
        + val_long.get_mci_image_pairs()
        + val_long.get_cn_image_pairs()
    )
    val_pairs = [x[0] for x in val_pairs]
    test_pairs = (
        test_long.get_ad_image_pairs()
        + test_long.get_mci_image_pairs()
        + test_long.get_cn_image_pairs()
    )
    test_pairs = [x[0] for x in test_pairs]

    train_ds = PairDataset(train_pairs, target_shape=target_shape)
    val_ds = PairDataset(val_pairs, target_shape=target_shape)
    test_ds = PairDataset(test_pairs, target_shape=target_shape)

    return train_ds, val_ds, test_ds
