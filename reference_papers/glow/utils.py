import copy

import cv2
import numpy as np
import torch


def read_image(
    image_path,
    size=None,
    channel_first=True,
    as_batch=True,
    as_torch_tensor=True,
    normalize=False,
    return_original=False,
    to_device=None,
):
    image = cv2.imread(image_path)
    if size:
        image = cv2.resize(image, dsize=size)
    image_original = copy.deepcopy(image)
    if normalize:
        image = (image.astype(np.float32) - 127.0) / 128.0
    if channel_first:
        image = np.transpose(image, (2, 0, 1))
    if as_batch:
        image = np.expand_dims(image, axis=0)
    if as_torch_tensor:
        image = torch.from_numpy(image)
    if to_device:
        image = image.to(device=to_device)
    if return_original:
        return image, image_original
    return image


def model_output_to_image(image_tensor):
    x = image_tensor.numpy()[0]
    x = x - np.min(x.flatten())
    x = x / np.max(x.flatten())
    x = x * 255.0
    x = x.transpose((1, 2, 0)).astype(np.uint8)
    return x
