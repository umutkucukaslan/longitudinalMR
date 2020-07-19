import os
import glob
import time
import numpy as np
import cv2
import nibabel as nib
import logging


# Create a custom logger
from preprocessing.utils import rigid_body_registration

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("../logs/log_registration.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info("merhaba")


"""
Rigid body transformation of images to baseline image for each patient in data folder.
"""


dataset_dir = "/Users/umutkucukaslan/Desktop/thesis/dataset/data_3t"

# Patient folders in data folder. This folder contains folders with
patients = sorted(glob.glob(os.path.join(dataset_dir, "*")))

counter = 1
for patient in patients:
    print("Processing {} / {}".format(counter, len(patients)))
    counter += 1

    logger.info(
        "Starting registrations for new patient (%s)", os.path.basename(patient)
    )
    dates = sorted(glob.glob(os.path.join(patient, "*")))
    baseline = dates[0]
    baseline_img_pth = glob.glob(os.path.join(baseline, "*.nii"))[0]

    follow_ups = dates[1:]

    for follow_up in follow_ups:
        follow_up_img_pth = glob.glob(os.path.join(follow_up, "*.nii"))
        if len(follow_up_img_pth) > 1:
            logger.warning(
                "There are more than one image file, possible registered version is also in the folder. Continuing..."
            )
            continue

        follow_up_img_pth = follow_up_img_pth[0]
        logger.info(
            "Starting registration of (%s, %s)",
            os.path.basename(baseline_img_pth),
            os.path.basename(follow_up_img_pth),
        )

        reg_follow_up_img_pth = os.path.join(
            os.path.dirname(follow_up_img_pth),
            "reg_" + os.path.basename(follow_up_img_pth),
        )

        _, _ = rigid_body_registration(
            baseline_img_pth, follow_up_img_pth, output_path=reg_follow_up_img_pth
        )
        logger.info(
            "Registration done and new image file is written to %s",
            reg_follow_up_img_pth,
        )
