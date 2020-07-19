import csv
import os
import glob
import sys

import cv2
import numpy as np
import pandas as pd
import imageio
import nibabel as nib
import logging


# Create a custom logger
from preprocessing.utils import get_axial_cortex_slices, crop_patient_slices

# logging.root.setLevel(logging.DEBUG)
# logger = logging.getLogger(__name__)


# # Create handlers
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler("../logs/log_create_slice_images.log")
# c_handler.setLevel(logging.DEBUG)
# f_handler.setLevel(logging.DEBUG)
#
# # Create formatters and add it to handlers
# c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
# f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)
#
# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)
#
# logger.info("merhaba")


"""
Create slice images from .nii files using a given range for slice index.
Set the following parameters
    - dataset_dir
    - dataset_info_pth: csv file that has Subject and Group columns as, for example, '136_S_1227' and 'MCI'
    - target_dir
    - start_offset: First slice index relative to index of slice that touches top part of the head
    - stop_offset: Last slice index relative to index of slice that touches top part of the head
    - step_size: Sampling interval in terms of index
    - use_registered_image: If true, slices registered image, otherwise original image
    - saved_image_data_type: 'uint8' or 'uint16'
    - show_results: If true, shows each slices after extraction
    - shape_after_padding: If true and slice is smaller than given shape, pads the slice image with zero
    
PS: get_axial_cortex_slices function also performs intensity correction for slices using mean and standard deviation
of intensities in the region of interest.
"""

# Set the following parameters
dataset_dir = "/Volumes/SAMSUNG/umut/thesis/adni_15T"
# dataset_dir = "/Users/umutkucukaslan/Desktop/thesis/dataset/data"
dataset_info_pth = (
    # "/Users/umutkucukaslan/Desktop/thesis/dataset/ADNI1_Complete_2Yr_3T_11_25_2019.csv"
    "/Volumes/SAMSUNG/umut/thesis/ADNI1_Complete_2Yr_1.5T_5_24_2020.csv"
    # "/Volumes/SAMSUNG/umut/thesis/ADNI1_Complete_2Yr_3T_11_25_2019.csv"
)
target_dir = "/Volumes/SAMSUNG/umut/thesis/processed_data_15T_192x160_4slices"
# target_dir = "/Volumes/SAMSUNG/umut/thesis/temp"
start_offset = 75
num_slices = 4
stop_offset = start_offset + num_slices
step_size = 1
saved_image_data_type = "uint8"
show_results = False
shape_after_padding = (300, 300)
shape_after_cropping = (192, 160)
slicing_index_csv_file = os.path.join(
    os.path.dirname(target_dir), "processed_data_15T_192x160_4slices.csv"
)

# Condition of patients
dataset_info = pd.read_csv(dataset_info_pth)
patient_condition = dict()
for item in dataset_info[["Subject", "Group"]].drop_duplicates("Subject").values:
    patient_condition[item[0]] = item[1]

# Patient folder paths
patients = sorted(glob.glob(os.path.join(dataset_dir, "*")))

# If target dir does not exist, create dir
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)


def save_slices(slices, dir_path, dtype="uint8", slicing_pattern_image=None):
    for i in range(len(slices)):
        pth = os.path.join(dir_path, "slice_{:03d}".format(i) + ".png")
        if dtype == "uint16":
            slice = np.asarray(slices[i] * (2 ** 16 - 1), dtype=np.uint16)
        else:
            slice = np.asarray(slices[i] * (2 ** 8 - 1), dtype=np.uint8)
        imageio.imwrite(pth, slice)
    # logger.info("%d slices were written to %s", len(slices), dir_path)

    if slicing_pattern_image is not None:
        pth = os.path.join(dir_path, "summary_slicing_pattern.png")
        imageio.imwrite(pth, slicing_pattern_image)
        # logger.info("Slicing pattern image was written to %s", pth)


def get_numpy_image(img):
    # get numpy correctted dimension order image from nibabel image format Nifti1....
    img_data = img.get_fdata()
    img_data = np.array(img_data)
    img_data = np.flip(img_data, axis=0)
    img_data = np.flip(img_data, axis=1)
    img_data = np.flip(img_data, axis=2)
    img_data = np.transpose(img_data, (2, 1, 0))

    return img_data


def add_row_to_csv_file(
    patient_name, start_index, stop_index, step_size, crop_indexes, crop_shape,
):
    if os.path.isfile(slicing_index_csv_file):
        with open(slicing_index_csv_file, mode="a") as file:
            csv_writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    patient_name,
                    str(start_index),
                    str(stop_index),
                    str(step_size),
                    str(crop_indexes),
                    str(crop_shape),
                ]
            )
    else:
        with open(slicing_index_csv_file, mode="w") as file:
            csv_writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "patient_name",
                    "start_index",
                    "stop_index",
                    "step_size",
                    "crop_indexes",
                    "crop_shape",
                ]
            )
            csv_writer.writerow(
                [
                    patient_name,
                    str(start_index),
                    str(stop_index),
                    str(step_size),
                    str(crop_indexes),
                    str(crop_shape),
                ]
            )


def get_voxel_sizes(mri):
    voxel_sizes = mri.header["pixdim"][1:4]
    return voxel_sizes


def get_resampled_mri_shape(mri_shape, voxel_sizes):
    resampled_shape = [int(float(x) * float(y)) for x, y in zip(mri_shape, voxel_sizes)]
    return resampled_shape


idx = 349
while idx < len(patients):
    # for idx in range(len(patients)):
    patient = patients[idx]
    patient_name = os.path.basename(patient)
    print("{} / {}  {}".format(idx + 1, len(patients), patient_name))
    # logger.info("Starting slicing for new patient (%s)", os.path.basename(patient))

    # Create new patient dir if not exists
    target_patient_dir = os.path.join(
        os.path.join(target_dir, patient_condition[patient_name]), patient_name
    )
    if not os.path.isdir(target_patient_dir):
        os.makedirs(target_patient_dir)

    # Date folders in patient dir
    dates = sorted(glob.glob(os.path.join(patient, "*")))

    # Baseline image folder path
    baseline = dates[0]

    # Target date dir where slices will be saved
    target_date_dir = os.path.join(target_patient_dir, os.path.basename(baseline))
    if not os.path.isdir(target_date_dir):
        os.makedirs(target_date_dir)

    if os.listdir(target_date_dir):
        # logger.info(
        #     "Already processed this patient {}, continuing".format(patient_name)
        # )
        d = input(
            "Already processed this patient {}, continuing? (y/n)".format(patient_name)
        )
        if d in ["y", "Y", "yes", "Yes"]:
            idx += 1
            continue

    # logger.info("Working on image from %s", os.path.basename(baseline))

    # Baseline image path
    baseline_img_pth = glob.glob(os.path.join(baseline, "[!resampled]*.nii"))[0]

    # Baseline image
    baseline_img = nib.load(baseline_img_pth)
    # baseline_img_data = get_numpy_image(baseline_img)
    baseline_img_data = baseline_img.get_fdata()
    voxel_sizes = get_voxel_sizes(baseline_img)
    mri_shape = baseline_img.shape
    resampled_mri_shape = get_resampled_mri_shape(
        mri_shape=mri_shape, voxel_sizes=voxel_sizes
    )
    resampled_slice_shape = resampled_mri_shape[1:3]
    print("resampled shape is {}".format(resampled_slice_shape))
    print("mri shape: {}".format(mri_shape))
    print("voxel sizes: {}".format(voxel_sizes))

    cv2.namedWindow("slices")
    cv2.moveWindow("slices", 20, 20)

    cv2.namedWindow("slicing pattern")
    cv2.moveWindow("slicing pattern", 500, 400)

    pressed_key = 0
    while pressed_key != ord("o"):
        # Extract slices from the baseline image
        (processed_slices, slicing_pattern_image,) = get_axial_cortex_slices(
            baseline_img_data,
            start_offset=start_offset,
            stop_offset=stop_offset,
            step=step_size,
            resampled_slice_shape=resampled_slice_shape,
            shape_after_padding=shape_after_padding,
            show_results=show_results,
        )
        seq = np.hstack(
            [
                # cv2.resize(x, (x.shape[0] // 2, x.shape[1] // 2))
                x
                for x in processed_slices
            ]
        )
        cv2.imshow("slices", seq)
        cv2.imshow("slicing pattern", slicing_pattern_image)
        pressed_key = cv2.waitKey()

        if pressed_key == ord("w"):
            start_offset -= 1
            stop_offset -= 1
        if pressed_key == ord("s"):
            start_offset += 1
            stop_offset += 1
        if pressed_key == ord("q"):
            exit()

    # Save extracted slices to target date dir
    save_slices(
        processed_slices,
        target_date_dir,
        dtype=saved_image_data_type,
        slicing_pattern_image=slicing_pattern_image,
    )

    # Follow-up scan folders (other date folders)
    follow_ups = dates[1:]

    for follow_up in follow_ups:
        # logger.info("Working on image from %s", os.path.basename(follow_up))

        follow_up_img_pth = glob.glob(os.path.join(follow_up, "reg*.nii"))[0]
        follow_up_img = nib.load(follow_up_img_pth)
        # follow_up_img_data = get_numpy_image(follow_up_img)
        follow_up_img_data = follow_up_img.get_fdata()

        (
            follow_up_processed_slices,
            follow_up_slicing_pattern_image,
        ) = get_axial_cortex_slices(
            follow_up_img_data,
            start_offset=start_offset,
            stop_offset=stop_offset,
            step=step_size,
            resampled_slice_shape=resampled_slice_shape,
            shape_after_padding=shape_after_padding,
            show_results=show_results,
        )

        # Target date dir where slices will be saved
        target_date_dir = os.path.join(target_patient_dir, os.path.basename(follow_up))
        if not os.path.isdir(target_date_dir):
            os.makedirs(target_date_dir)

        # Save extracted slices to target date dir
        save_slices(
            follow_up_processed_slices,
            target_date_dir,
            dtype=saved_image_data_type,
            slicing_pattern_image=follow_up_slicing_pattern_image,
        )

    crop_indexes, crop_shape = crop_patient_slices(
        target_patient_dir,
        target_patient_dir,
        crop_height=shape_after_cropping[0],
        crop_width=shape_after_cropping[1],
        source_image_size=shape_after_padding,
    )

    # add the chosen start stop indexes for the patient to the csv file for future use
    add_row_to_csv_file(
        patient_name=patient_name,
        start_index=start_offset,
        stop_index=stop_offset,
        step_size=step_size,
        crop_indexes=crop_indexes,
        crop_shape=crop_shape,
    )

    idx += 1
