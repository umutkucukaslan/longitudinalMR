import glob
import os
import random
import shutil
from shutil import copytree

import cv2
import imageio

"""
Splits processed dataset into train, val, test folders
Adds ad/mci/cn prefixes to patient folders accordingly
Set the params below
"""
# --------------------
# set following params
raw_dataset = "/Volumes/SAMSUNG/umut/thesis/processed_data_15T_256x256_4slices"
processed_dataset = (
    "/Volumes/SAMSUNG/umut/thesis/cropped_processed_data_15T_256x256_4slices"
)

source_image_size = (256, 256)
target_image_size = (192, 160)
# --------------------

processed_dataset = (
    processed_dataset
    + "_"
    + str(target_image_size[0])
    + "x"
    + str(target_image_size[1])
)

crop_height = target_image_size[0]
crop_width = target_image_size[1]


def process_disease_folder(source_folder, target_folder):
    patient_folders = sorted(glob.glob(os.path.join(source_folder, "*")))
    patient_type = os.path.basename(source_folder)
    for patient_folder in patient_folders:
        patient_name = os.path.basename(patient_folder)
        print("Processing {}...".format(patient_name))
        target_patient_folder = os.path.join(target_folder, patient_name)
        crop_patient_slices(patient_folder, target_patient_folder)
        print("{} patient {} completed".format(patient_type, patient_name))


process_disease_folder(
    os.path.join(raw_dataset, "AD"), os.path.join(processed_dataset, "AD")
)
process_disease_folder(
    os.path.join(raw_dataset, "MCI"), os.path.join(processed_dataset, "MCI")
)
process_disease_folder(
    os.path.join(raw_dataset, "CN"), os.path.join(processed_dataset, "CN")
)
