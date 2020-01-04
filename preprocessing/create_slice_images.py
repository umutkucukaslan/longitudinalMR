import os
import glob
import numpy as np
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
'''

dataset_dir = '/Users/umutkucukaslan/Desktop/pmsd/dataset/data'

# Patient folders' paths
patients = sorted(glob.glob(os.path.join(dataset_dir, '*')))


def save_slices(slices, dir_path, dtype="uint16"):
    for i in range(len(slices)):
        pth = os.path.join(dir_path, "slice_" + str(i) + ".png")
        if dtype == "uint16":
            slice = np.asarray(slices[i] * (2 ** 16 - 1), dtype=np.uint16)
        else:
            slice = np.asarray(slices[i] * (2 ** 8 - 1), dtype=np.uint8)
        imageio.imwrite(pth, slice)
    logger.info("%d slices are written to %s", len(slices), dir_path)


for patient in patients:
    logger.info('Starting slicing for new patient (%s)', os.path.basename(patient))
    patient_name = os.path.basename(patient)
    dates = sorted(glob.glob(os.path.join(patient, '*')))
    baseline = dates[0]
    baseline_img_pth = glob.glob(os.path.join(baseline, '*.nii'))[0]

    baseline_img = nib.load(baseline_img_pth)
    processed_slices, summary_image, slicing_pattern_image = get_axial_cortex_slices(baseline_img,
                                                                                     start_offset=30,
                                                                                     stop_offset=100,
                                                                                     step=1,
                                                                                     show_results=False)
    slice_dir = os.path.dirname(baseline_img_pth)
    save_slices(processed_slices, slice_dir, dtype="uint8")
    logger.info("DONE")
    break
    # continue



    follow_ups = dates[1:]

    for follow_up in follow_ups:
        follow_up_img_pth = glob.glob(os.path.join(follow_up, '*.nii'))
        if len(follow_up_img_pth) > 1:
            logger.warning('There are more than one image file, possible registered version is also in the folder. Continuing...')
            continue

        follow_up_img_pth = follow_up_img_pth[0]
        logger.info('Starting registration of (%s, %s)', os.path.basename(baseline_img_pth), os.path.basename(follow_up_img_pth))

        reg_follow_up_img_pth = os.path.join(os.path.dirname(follow_up_img_pth), 'reg_' + os.path.basename(follow_up_img_pth))

        _, _ = rigid_body_registration(baseline_img_pth, follow_up_img_pth, output_path=reg_follow_up_img_pth)
        logger.info('Registration done and new image file is written to %s', reg_follow_up_img_pth)

