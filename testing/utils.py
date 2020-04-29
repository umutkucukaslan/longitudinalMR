import numpy as np


def preprocess_image(img):
    """
    Preprocess image for inference on generator/encoder.
    Change image range from [0, 255] to [0, 1] then make it rank 4.

    :param img: Image with range [0, 255]
    :return: Image with range [0, 1] and rank 4.
    """

    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    return img


def postprocess_image(img):
    """
    Post process generator/decoder output image to obtain regular image.
    Change image range from [0, 1] to [0, 255] and squeeze it.

    :param img: Rank 4 image with range [0, 1]
    :return: Regular image with range [0, 255]
    """
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = np.squeeze(img)
    return img


def mse_uint8(im1, im2):
    """
    Mean square error between two images. [0, 255] range images are changed to [0, 1] range then mse is computed.

    :param im1:
    :param im2:
    :return: mse between (0, 1)
    """
    im1 = im1.astype(np.float) / 255.0
    im2 = im2.astype(np.float) / 255.0

    return mse_float(im1, im2)


def mse_float(im1, im2):
    """
    Mean square error between two images. Image range should be [0, 1]

    :param im1:
    :param im2:
    :return: mse between (0, 1)
    """

    return np.mean(np.square(im1.flatten() - im2.flatten()))

