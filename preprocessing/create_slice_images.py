import os
import glob
import sys

import numpy as np
import pandas as pd
import imageio
import nibabel as nib
import logging


# Create a custom logger
from preprocessing.utils import get_axial_cortex_slices

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('../logs/log_create_slice_images.log')
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


'''
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
'''

# Set the following parameters
dataset_dir = '/Users/umutkucukaslan/Desktop/pmsd/dataset/data'
dataset_info_pth = '/Users/umutkucukaslan/Desktop/pmsd/dataset/ADNI1_Complete_2Yr_3T_11_25_2019.csv'
target_dir = '/Users/umutkucukaslan/Desktop/pmsd/dataset/processed_data_256x256'
start_offset = 30
stop_offset = 100
step_size = 1
use_registered_image = True
saved_image_data_type = "uint8"
show_results = False
shape_after_padding = (256, 256)

# Condition of patients
dataset_info = pd.read_csv(dataset_info_pth)
patient_condition = dict()
for item in dataset_info[["Subject", "Group"]].drop_duplicates("Subject").values:
    patient_condition[item[0]] = item[1]

# Patient folder paths
patients = sorted(glob.glob(os.path.join(dataset_dir, '*')))

# If target dir does not exist, create dir
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)


def save_slices(slices, dir_path, dtype="uint16", summary_image=None, slicing_pattern_image=None):
    for i in range(len(slices)):
        pth = os.path.join(dir_path, "slice_" + str(i) + ".png")
        if dtype == "uint16":
            slice = np.asarray(slices[i] * (2 ** 16 - 1), dtype=np.uint16)
        else:
            slice = np.asarray(slices[i] * (2 ** 8 - 1), dtype=np.uint8)
        imageio.imwrite(pth, slice)
    logger.info("%d slices were written to %s", len(slices), dir_path)

    if summary_image is not None:
        pth = os.path.join(dir_path, "summary_slicing_range.png")
        imageio.imwrite(pth, summary_image)
        logger.info("Summary image was written to %s", pth)

    if slicing_pattern_image is not None:
        pth = os.path.join(dir_path, "summary_slicing_pattern.png")
        imageio.imwrite(pth, slicing_pattern_image)
        logger.info("Slicing pattern image was written to %s", pth)


for patient in patients:
    logger.info('Starting slicing for new patient (%s)', os.path.basename(patient))

    # Create new patient dir if not exists
    patient_name = os.path.basename(patient)
    target_patient_dir = os.path.join(os.path.join(target_dir, patient_condition[patient_name]), patient_name)
    if not os.path.isdir(target_patient_dir):
        os.makedirs(target_patient_dir)

    # Date folders in patient dir
    dates = sorted(glob.glob(os.path.join(patient, '*')))

    # Baseline image folder path
    baseline = dates[0]

    logger.info("Working on image from %s", os.path.basename(baseline))

    # Baseline image path
    baseline_img_pth = glob.glob(os.path.join(baseline, '*.nii'))[0]

    # Baseline image
    baseline_img = nib.load(baseline_img_pth)

    # Extract slices from the baseline image
    processed_slices, summary_image, slicing_pattern_image = get_axial_cortex_slices(baseline_img,
                                                                                     start_offset=start_offset,
                                                                                     stop_offset=stop_offset,
                                                                                     step=step_size,
                                                                                     shape_after_padding=shape_after_padding,
                                                                                     show_results=show_results)

    # Target date dir where slices will be saved
    target_date_dir = os.path.join(target_patient_dir, os.path.basename(baseline))
    if not os.path.isdir(target_date_dir):
        os.makedirs(target_date_dir)

    # Save extracted slices to target date dir
    save_slices(processed_slices, target_date_dir, dtype=saved_image_data_type, summary_image=summary_image,
                slicing_pattern_image=slicing_pattern_image)

    # Follow-up scan folders (other date folders)
    follow_ups = dates[1:]

    for follow_up in follow_ups:
        logger.info("Working on image from %s", os.path.basename(follow_up))

        # Follow-up image path
        if use_registered_image:
            follow_up_img_pth = glob.glob(os.path.join(follow_up, 'reg_*.nii'))[0]
        else:
            follow_up_img_pth = glob.glob(os.path.join(follow_up, '[!reg_]*.nii'))[0]

        # Follow-up image
        follow_up_img = nib.load(follow_up_img_pth)

        # Extract slices from the follow-up image
        processed_slices, summary_image, slicing_pattern_image = get_axial_cortex_slices(follow_up_img,
                                                                                         start_offset=start_offset,
                                                                                         stop_offset=stop_offset,
                                                                                         step=step_size,
                                                                                         shape_after_padding=shape_after_padding,
                                                                                         show_results=show_results)

        # Target date dir where slices will be saved
        target_date_dir = os.path.join(target_patient_dir, os.path.basename(follow_up))
        if not os.path.isdir(target_date_dir):
            os.makedirs(target_date_dir)

        # Save extracted slices to target date dir
        save_slices(processed_slices, target_date_dir, dtype=saved_image_data_type, summary_image=summary_image,
                    slicing_pattern_image=slicing_pattern_image)

