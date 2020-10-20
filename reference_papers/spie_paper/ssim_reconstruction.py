import glob
import os
import random
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset
from reference_papers.spie_paper.train_wgan2 import get_generator_discriminator


"""

"""

if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "macbook"
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
else:
    raise ValueError("Unknown machine type")

data_folder = "val"

if MACHINE == "macbook":
    data_dir = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/dataset/training_data_15T_192x160_4slices",
        data_folder,
    )
elif MACHINE == "colab":
    data_dir = os.path.join("/content/training_data_15T_192x160_4slices", data_folder)

longitudinal_dataset = LongitudinalDataset(data_dir=data_dir)

paths = (
    longitudinal_dataset.get_ad_images()
    + longitudinal_dataset.get_mci_images()
    + longitudinal_dataset.get_cn_images()
)
paths = sorted(paths)


def get_matching_res_image_path(image_path, target_dir):
    scan_dir = os.path.dirname(image_path)
    patient_dir = os.path.dirname(scan_dir)
    scan_name = os.path.basename(scan_dir)
    patient_name = os.path.basename(patient_dir)
    image_name = os.path.basename(image_path)
    matching_image_path = os.path.join(
        target_dir, patient_name, scan_name, "res_" + image_name
    )
    return matching_image_path


generator, discriminator, experiment_folder = get_generator_discriminator()
del discriminator, generator

encodings_dir = os.path.join(experiment_folder, data_folder)


def calculate_ssims(source_paths, encodings_dir):
    ssims = []
    for image_path in source_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2GRAY)
        reconst_image = cv2.imread(
            get_matching_res_image_path(image_path, encodings_dir)
        )
        reconst_image = cv2.cvtColor(
            cv2.resize(reconst_image, (64, 64)), cv2.COLOR_BGR2GRAY
        )
        ssims.append(structural_similarity(image, reconst_image, data_range=255))
    return np.mean(ssims), ssims


all_ssims = []
ad_ssim, temp = calculate_ssims(longitudinal_dataset.get_ad_images(), encodings_dir)
all_ssims += temp
mci_ssim, temp = calculate_ssims(longitudinal_dataset.get_mci_images(), encodings_dir)
all_ssims += temp
cn_ssim, temp = calculate_ssims(longitudinal_dataset.get_cn_images(), encodings_dir)
all_ssims += temp

print(f"Mean ssim : {np.mean(all_ssims)}")
print(f"AD ssim   : {ad_ssim}")
print(f"MCI ssim  : {mci_ssim}")
print(f"CN ssim   : {cn_ssim}")
