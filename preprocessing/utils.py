import copy

import cv2
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
        new_image = nib.Nifti1Image(transformed, moving_grid2world, moving_nii.header)
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

    # Show range of slicing in a sagittal image
    # summary_image = np.stack((summary_image, summary_image, summary_image), axis=2)
    # summary_image = np.asarray(summary_image, dtype=np.uint8)
    # summary_image[head_start + start_offset, :, 0] = 255
    # summary_image[head_start + stop_offset, :, 0] = 255

    # Image to show slice positions
    # slicing_pattern_image = copy.copy(summary_image.astype(np.float))

    # Image data array
    # img_data = img.get_fdata()

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
