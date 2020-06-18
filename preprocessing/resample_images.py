import glob
import os

import nibabel
from nibabel.processing import resample_to_output


"""
This script resamples MRI images to (1mm, 1mm, 1mm)
It also changes axis directions.
It uses registered MRIs
"""


data_dir = "/Users/umutkucukaslan/Desktop/thesis/dataset/data"

patient_folders = sorted(glob.glob(os.path.join(data_dir, "*")))


for idx in range(len(patient_folders)):
    patient_folder_path = patient_folders[idx]
    patient_name = os.path.basename(patient_folder_path)
    scans = sorted(glob.glob(os.path.join(patient_folder_path, "*")))
    print("processing {}  ({}/{})".format(patient_name, idx + 1, len(patient_folders)))
    for scan_folder in scans:
        reg_paths = glob.glob(os.path.join(scan_folder, "reg_*.nii"))
        original_paths = glob.glob(os.path.join(scan_folder, "[!reg_]*.nii"))
        scan_path = reg_paths[0] if reg_paths else original_paths[0]
        target_path = os.path.join(
            os.path.dirname(scan_path), "resampled_" + os.path.basename(scan_path)
        )
        if os.path.isfile(target_path):
            print("    already existing {}".format(os.path.basename(target_path)))
            continue
        print("    processing {}".format(os.path.basename(scan_path)))
        img = nibabel.load(scan_path)
        out_img = resample_to_output(img, voxel_sizes=1.0)
        nibabel.save(out_img, target_path)
