
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
raw_dataset = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_256x256'
processed_dataset = '/Users/umutkucukaslan/Desktop/thesis/dataset/cropped_processed_data'

source_image_size = (256, 256)
target_image_size = (192, 160)
# --------------------

processed_dataset = processed_dataset + '_' + str(target_image_size[0]) + 'x' + str(target_image_size[1])

crop_height = target_image_size[0]
crop_width = target_image_size[1]


def process_patient(source_patient_folder, target_patient_folder):
    # starts GUI for the patient, receives input for the crop, then crops all the slices and saves them in target patient
    # folder according to the slices
    """
    Key presses:
        w                  up
     a  s  d        left  down  right

     b  n           previous slice      next slice
     v  m           previous scan       next scan

     r              reset crop window locations
     o              okay - process accordingly

    :param source_patient_folder:
    :param target_patient_folder:
    :return:
    """

    scan_folders = sorted(glob.glob(os.path.join(source_patient_folder, '*')))

    scan_index = 0
    slice_index = 0
    image_raw_index = int((source_image_size[0] - crop_height) / 2)
    image_col_index = int((source_image_size[1] - crop_width) / 2)
    pressed_key = 0
    while pressed_key != ord('o'):
        slices = sorted(glob.glob(os.path.join(scan_folders[scan_index], 'slice_*.png')))
        slice = slices[slice_index]
        slice_img = imageio.imread(slice)
        slice_img[image_raw_index, :] = 255
        slice_img[image_raw_index + crop_height, :] = 255
        slice_img[:, image_col_index] = 255
        slice_img[:, image_col_index + crop_width] = 255

        cv2.imshow('cropping tool', slice_img)
        pressed_key = cv2.waitKey()

        if pressed_key == ord('s'):
            # down key pressed
            image_raw_index += 1
            if image_raw_index + crop_height > source_image_size[0]:
                image_raw_index = source_image_size[0] - crop_height

        if pressed_key == ord('w'):
            # up key pressed
            image_raw_index -= 1
            if image_raw_index < 0:
                image_raw_index = 0

        if pressed_key == ord('d'):
            # right key pressed
            image_col_index += 1
            if image_col_index + crop_width > source_image_size[1]:
                image_col_index = source_image_size[1] - crop_width

        if pressed_key == ord('a'):
            # left key pressed
            image_col_index -= 1
            if image_col_index < 0:
                image_col_index = 0

        if pressed_key == ord('r'):
            # reset key
            image_raw_index = int((source_image_size[0] - crop_height) / 2)
            image_col_index = int((source_image_size[1] - crop_width) / 2)

        if pressed_key == ord('n'):
            # next slice
            slice_index = min(len(slices) - 1, slice_index + 1)

        if pressed_key == ord('b'):
            # previous slice
            slice_index = max(0, slice_index - 1)

        if pressed_key == ord('m'):
            # next scan
            scan_index = min(len(scan_folders) - 1, scan_index + 1)

        if pressed_key == ord('v'):
            # previous scan
            scan_index = max(0, scan_index - 1)

        if pressed_key == ord('q'):
            print('Terminated')
            exit()

    for scan_folder in scan_folders:
        scan_folder_name = os.path.basename(scan_folder)
        target_scan_folder = os.path.join(target_patient_folder, scan_folder_name)
        if not os.path.isdir(target_scan_folder):
            os.makedirs(target_scan_folder)

        slices = sorted(glob.glob(os.path.join(scan_folder, 'slice_*.png')))
        for slice in slices:
            slice_name = os.path.basename(slice)
            slice_img = imageio.imread(slice)
            cropped_img = slice_img[image_raw_index: image_raw_index + crop_height, image_col_index: image_col_index + crop_width]
            target_slice_path = os.path.join(target_scan_folder, slice_name)
            imageio.imwrite(target_slice_path, cropped_img)


        other_files = sorted(glob.glob(os.path.join(scan_folder, 'summary*.png')))
        for file in other_files:
            target_file_path = os.path.join(target_scan_folder, os.path.basename(file))
            shutil.copy(file, target_file_path)


def process_disease_folder(source_folder, target_folder):
    patient_folders = sorted(glob.glob(os.path.join(source_folder, '*')))
    for patient_folder in patient_folders:
        patient_name = os.path.basename(patient_folder)
        target_patient_folder = os.path.join(target_folder, patient_name)
        process_patient(patient_folder, target_patient_folder)
        print('patient {} completed'.format(patient_name))


process_disease_folder(os.path.join(raw_dataset, 'AD'), os.path.join(processed_dataset, 'AD'))
process_disease_folder(os.path.join(raw_dataset, 'MCI'), os.path.join(processed_dataset, 'MCI'))
process_disease_folder(os.path.join(raw_dataset, 'CN'), os.path.join(processed_dataset, 'CN'))

