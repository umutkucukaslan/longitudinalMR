import copy
import glob
import os
import shutil

import cv2
import imageio
import nibabel as nib
import numpy as np
from dipy.align.imaffine import (
    transform_centers_of_mass,
    AffineMap,
    MutualInformationMetric,
    AffineRegistration,
)
from dipy.align.transforms import (
    TranslationTransform3D,
    RigidTransform3D,
    AffineTransform3D,
)
from skimage.metrics import structural_similarity


def rigid_body_registration(static_image_path, moving_image_path, output_path=None):
    """
    Registers moving image to static image using ridig body transformations. If output_path is given, it saves
    the result. Registration is done in the following steps.
        - Center of mass transform
        - Translation transform
        - Rigid body transform
    :param static_image_path: Path to static image file.
    :param moving_image_path: Path to moving image file.
    :param output_path: If given, resulting image is saved to output_path.
    :return: (transformed_image, associated_transformation_matrix)
    """

    # Reference page for affine registration
    # https://dipy.org/documentation/1.0.0./examples_built/affine_registration_3d/#example-affine-registration-3d

    static_nii = nib.load(static_image_path)
    moving_nii = nib.load(moving_image_path)

    static = static_nii.get_fdata()
    moving = moving_nii.get_fdata()

    static_grid2world = static_nii.affine
    moving_grid2world = moving_nii.affine

    c_of_mass = transform_centers_of_mass(
        static, static_grid2world, moving, moving_grid2world
    )
    print("Center of mass is {}".format(c_of_mass))
    print("Center of mass is {}".format(c_of_mass))

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(
        metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors
    )

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(
        static,
        moving,
        transform,
        params0,
        static_grid2world,
        moving_grid2world,
        starting_affine=starting_affine,
    )

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(
        static,
        moving,
        transform,
        params0,
        static_grid2world,
        moving_grid2world,
        starting_affine=starting_affine,
    )

    transformed = rigid.transform(moving)

    if output_path is not None:
        # new_image = nib.Nifti1Image(transformed, moving_grid2world, moving_nii.header)
        new_image = nib.Nifti1Image(transformed, static_grid2world, moving_nii.header)
        nib.save(new_image, output_path)

    return transformed, rigid


def find_upper_tangent_line_to_head_in_3d_mri(img=None, img_path=None, axis=0):
    """
    This function finds slice index touching head at given axis. You should input either img or img_path.

    :param img: nibabel.Nifti1Image object - what nibabel.load returns for .nii images
    :param img_path: path of image file
    :param axis: Axis you want to find the index of touching slice
    :return: A tuple of (index, image showing the result, dynamic threshold used)
    """

    # Check if axis is correct
    axis = int(axis)
    if axis < 0 or axis > 2:
        print("Unknown axis!")
        raise ValueError

    # Reading image data with error handling
    if img is None:
        if img_path is None:
            print("Provide either nibabel image or .nii file path!")
            raise ValueError
        else:
            try:
                img = nib.load(img_path)
                img_data = img.get_fdata()
            except Exception as e:
                print("Exception occurred while reading data. ", e)
                return None
    else:
        try:
            img_data = img.get_fdata()
        except Exception as e:
            print("Exception occurred while reading data from image object. ", e)
            return None

    # 3D image as numpy array
    img_data = np.asarray(img_data, dtype=np.float32)

    # Threshold for foreground/background
    threshold = img_data.flatten().mean()

    # Image shape
    img_data_shape = img_data.shape

    # Threshold for number of foreground pixels at the edge of head
    head_threshold = 500

    for i in range(img_data_shape[axis]):

        if axis == 0:
            slice = img_data[i, :, :]
        elif axis == 1:
            slice = img_data[:, i, :]
        elif axis == 2:
            slice = img_data[:, :, i]

        # Median filtering for noise removal
        slice = cv2.medianBlur(slice, 5)

        if np.sum(slice.flatten() > threshold) > head_threshold:

            if axis == 0:
                mid_point = int(img_data_shape[2] / 2)
                slice = img_data[:, :, mid_point]
            elif axis == 1:
                mid_point = int(img_data_shape[2] / 2)
                slice = img_data[:, :, mid_point]
            elif axis == 2:
                mid_point = int(img_data_shape[0] / 2)
                slice = img_data[mid_point, :, :]

            image = np.asarray((slice / slice.max()) * 255, dtype=np.uint8)

            if axis == 0:
                image[i, :] = 255
            elif axis == 1:
                image[:, i] = 255
            elif axis == 2:
                image[:, i] = 255

            return i, image, threshold


def get_axial_cortex_slices(
    img_data,
    start_offset=30,
    stop_offset=100,
    step=5,
    resampled_slice_shape=None,
    shape_after_padding=None,
    show_results=False,
):
    """
    Slices the 3D MRI volume in the given range and returns list of slices.

    :param img: MRI image object (nibabel.Nifti1Image -  what nibabel.load() returns)
    :param start_offset: Offset of first slice from the slice touching upper part of the head
    :param stop_offset: Offset of last slice from the slice touching upper part of the head
    :param step: Sampling interval
    :param shape_after_padding: (height, width) If not None and shape of slice is smaller than this tuple,
                                pad zeros to make it this shape
    :param show_results: If true, shows the slices as it extracts.
    :return: processed_slices: list of slices
             summary_image: summary image that shows first and last slice in a saggittal image
             slicing_pattern_image: summary image that shows all slice positions in a sagittal image
    """

    # # Index of slice that touches upper part of the head
    # if head_start is None:
    #     head_start, summary_image, _ = find_upper_tangent_line_to_head_in_3d_mri(
    #         img=img, axis=0
    #     )
    # else:
    #     _, summary_image, _ = find_upper_tangent_line_to_head_in_3d_mri(img=img, axis=0)

    # slicing pattern image
    mid_slice_index = img_data.shape[2] // 2
    slicing_pattern_image = img_data[:, :, mid_slice_index]
    slicing_pattern_image = np.stack(
        (slicing_pattern_image, slicing_pattern_image, slicing_pattern_image), axis=-1
    )

    # Stopping position for slicing
    slice_index_stop = min(stop_offset, img_data.shape[2])

    # Calculate mean and std intensity values in the region of interest (sliced region) to specify brightness range
    roi = img_data[start_offset:slice_index_stop, :, :]

    # Do not include empty areas in mean and std calculation
    threshold_adapted_mean = 20
    nice_points = np.asarray([x for x in roi.flatten() if x > threshold_adapted_mean])
    adapted_mean = nice_points.mean()
    adapted_std = np.std(nice_points)

    # Clip intensity values beyond this
    max_clipping_value = adapted_mean + 1.8 * adapted_std

    slicing_pattern_image = (
        255 * np.clip(slicing_pattern_image, 0, max_clipping_value) / max_clipping_value
    )
    slicing_pattern_image[start_offset, :, 0] = 255
    slicing_pattern_image[start_offset, :, 0] = 255

    processed_slices = []
    for index in range(start_offset, slice_index_stop, step):
        slice = img_data[index, :, :]

        # Clip intensity values beyond the range [0, max_clipping_value]
        processed_slice = np.clip(slice, 0, max_clipping_value)
        processed_slice = cv2.resize(
            processed_slice, (resampled_slice_shape[1], resampled_slice_shape[0])
        )

        # Pad with zeros if the shape of the slice is smaller than desired
        if shape_after_padding is not None:
            desired_height, desired_width = shape_after_padding
            pad_height = max(0, desired_height - processed_slice.shape[0])
            pad_width = max(0, desired_width - processed_slice.shape[1])
            processed_slice = np.pad(
                processed_slice,
                [
                    (int(pad_height / 2), pad_height - int(pad_height / 2)),
                    (int(pad_width / 2), pad_width - int(pad_width / 2)),
                ],
            )

        # Compress intensities in [0, 1] range
        processed_slice = np.asarray(processed_slice / max_clipping_value)
        processed_slices.append(processed_slice)

        # Show this slice position in the sagittal image
        slicing_pattern_image[index, :, 1] += 100

        if show_results:
            cv2.imshow("Slicing pattern", slicing_pattern_image)
            cv2.imshow(
                "Processed slice", np.asarray(processed_slice * 255, dtype=np.uint8)
            )
            saturation_image = np.asarray(processed_slice * 255, dtype=np.uint8)
            saturation_image[saturation_image < 254] = 0
            cv2.imshow("Saturated parts due to clipping", saturation_image)
            cv2.waitKey()

    slicing_pattern_image = np.clip(slicing_pattern_image, 0, 255).astype(np.uint8)

    return processed_slices, slicing_pattern_image


def crop_patient_slices(
    source_patient_folder,
    target_patient_folder,
    crop_height=256,
    crop_width=256,
    source_image_size=(256, 256),
):
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
    :param crop_height:
    :param crop_width:
    :param source_image_size:
    :return:
    """

    scan_folders = sorted(glob.glob(os.path.join(source_patient_folder, "*")))

    scan_index = 0
    slice_index = 0
    image_row_index = int((source_image_size[0] - crop_height) / 2)
    image_col_index = int((source_image_size[1] - crop_width) / 2)
    pressed_key = 0
    while pressed_key != ord("o"):
        slices = sorted(
            glob.glob(os.path.join(scan_folders[scan_index], "slice_*.png"))
        )
        slice = slices[slice_index]
        slice_img = imageio.imread(slice)
        slice_img[image_row_index, :] = 255
        slice_img[image_row_index + crop_height, :] = 255
        slice_img[:, image_col_index] = 255
        slice_img[:, image_col_index + crop_width] = 255

        cv2.imshow("cropping tool", slice_img)
        pressed_key = cv2.waitKey()

        if pressed_key == ord("s"):
            # down key pressed
            image_row_index += 1
            if image_row_index + crop_height >= source_image_size[0]:
                image_row_index = source_image_size[0] - crop_height - 1

        if pressed_key == ord("w"):
            # up key pressed
            image_row_index -= 1
            if image_row_index < 0:
                image_row_index = 0

        if pressed_key == ord("d"):
            # right key pressed
            image_col_index += 1
            if image_col_index + crop_width >= source_image_size[1]:
                image_col_index = source_image_size[1] - crop_width - 1

        if pressed_key == ord("a"):
            # left key pressed
            image_col_index -= 1
            if image_col_index < 0:
                image_col_index = 0

        if pressed_key == ord("r"):
            # reset key
            image_row_index = int((source_image_size[0] - crop_height) / 2)
            image_col_index = int((source_image_size[1] - crop_width) / 2)

        if pressed_key == ord("n"):
            # next slice
            slice_index = min(len(slices) - 1, slice_index + 1)

        if pressed_key == ord("b"):
            # previous slice
            slice_index = max(0, slice_index - 1)

        if pressed_key == ord("m"):
            # next scan
            scan_index = min(len(scan_folders) - 1, scan_index + 1)

        if pressed_key == ord("v"):
            # previous scan
            scan_index = max(0, scan_index - 1)

        if pressed_key == ord("q"):
            print("Terminated")
            exit()

    for scan_folder in scan_folders:
        scan_folder_name = os.path.basename(scan_folder)
        target_scan_folder = os.path.join(target_patient_folder, scan_folder_name)
        if not os.path.isdir(target_scan_folder):
            os.makedirs(target_scan_folder)

        slices = sorted(glob.glob(os.path.join(scan_folder, "slice_*.png")))
        for slice in slices:
            slice_name = os.path.basename(slice)
            slice_img = imageio.imread(slice)
            cropped_img = slice_img[
                image_row_index : image_row_index + crop_height,
                image_col_index : image_col_index + crop_width,
            ]
            target_slice_path = os.path.join(target_scan_folder, slice_name)
            imageio.imwrite(target_slice_path, cropped_img)

        if source_patient_folder != target_patient_folder:
            other_files = sorted(glob.glob(os.path.join(scan_folder, "summary*.png")))
            for file in other_files:
                target_file_path = os.path.join(
                    target_scan_folder, os.path.basename(file)
                )
                shutil.copy(file, target_file_path)

    return (image_row_index, image_col_index), (crop_height, crop_width)
