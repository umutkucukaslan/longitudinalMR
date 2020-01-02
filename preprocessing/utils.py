import time

import cv2
import nibabel as nib
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)



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

    c_of_mass = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

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

    print(threshold)

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
