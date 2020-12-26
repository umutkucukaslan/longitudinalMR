import os

import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset
from reference_papers.glow.model import Glow
from reference_papers.glow.train import calc_z_shapes
from reference_papers.glow.utils import read_image


# relative_model_path = "exp_2020_10_30_glow/checkpoint/model_030001.pt"
relative_model_path = "exp_2020_12_23_glow_pair_finetune/checkpoint/model_115001.pt"

# running device
if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "macbook"
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
else:
    raise ValueError("Unknown machine type")

data_folder = "val"

# data set path
if MACHINE == "macbook":
    data_dir = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/dataset/training_data_15T_192x160_4slices",
        data_folder,
    )
    model_path = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/experiments", relative_model_path
    )
elif MACHINE == "colab":
    data_dir = os.path.join("/content/training_data_15T_192x160_4slices", data_folder)
    relative_model_path = "glow_pair_finetune/checkpoint/model_115001.pt"
    model_path = os.path.join(
        "/content/drive/My Drive/experiments", relative_model_path
    )

# create longitudinal data set object
longitudinal_dataset = LongitudinalDataset(data_dir=data_dir)

paths = (
    longitudinal_dataset.get_ad_images()
    + longitudinal_dataset.get_mci_images()
    + longitudinal_dataset.get_cn_images()
)
paths = sorted(paths)  # paths for evaluation images


# model training parameters class
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# define model training params
args = Namespace(
    affine=False,
    batch=16,
    img_size=64,
    iter=200000,
    lr=0.0001,
    n_bits=5,
    n_block=4,
    n_flow=32,
    n_sample=20,
    no_lu=False,
    path="/content/glow_data2",
    save_dir="/content/drive/My Drive/experiments/glow_1",
    temp=0.7,
)


# load trained model
model_single = Glow(
    3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
)
print("initializing model")
rand_img = torch.from_numpy(np.random.rand(1, 3, 64, 64))
with torch.no_grad():
    _ = model_single(rand_img.float())
print("model initialized")
model = torch.nn.DataParallel(model_single)
# loaded_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
if MACHINE == "colab":
    loaded_state_dict = torch.load(model_path, map_location=torch.device("cuda:0"))
model.load_state_dict(loaded_state_dict)
model.eval()  # model in eval mode


def infer(model, image):
    """
    from image to latent vector
    :param model:
    :param image:
    :return:
    """
    with torch.no_grad():
        log_p_sum, logdet, z_outs = model(image)
        gen_im = model.reverse(z_outs, reconstruct=True)
    return z_outs, gen_im


def sample(model, z_list):
    """
    from latent vector to image
    :param model:
    :param z_list:
    :return:
    """
    with torch.no_grad():
        gen_im = model.reverse(z_list, reconstruct=True)
        print("[sample function] gen_im.shape: ", gen_im.shape)
        print("[sample function] gen_im.dtype: ", gen_im.dtype)
        log_p_sum, logdet, z_outs = model(gen_im)
    return z_outs, gen_im


# ssims = []
# c = 0
# for image_path in paths:
#     c += 1
#     image, image_original = read_image(
#         image_path,
#         size=(args.img_size, args.img_size),
#         channel_first=True,
#         as_batch=True,
#         as_torch_tensor=True,
#         normalize=True,
#         return_original=True,
#     )
#     # print("input image shape: ", image.shape)
#     # print("input image dtype: ", image.dtype)
#     z_outs, gen_im = infer(model_single, image)
#     # print("z_outs.shape: ", len(z_outs))
#     # print("genim tye: ", type(gen_im))
#     # print("genim shape: ", gen_im.shape)
#     # print("---------")
#     x = gen_im.numpy()[0]
#     x = x - np.min(x.flatten())
#     x = x / np.max(x.flatten())
#     x = x * 255.0
#     x = x.transpose((1, 2, 0)).astype(np.uint8)
#     ssim = structural_similarity(
#         im1=cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY),
#         im2=cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
#         data_range=255,
#     )
#     print(f"{c}/{len(paths)}: {ssim} <- {image_path}")
#     ssims.append(ssim)
#     cv2.imshow("gen image", x)
#     p = cv2.waitKey(5)
#     if p == ord("q"):
#         break
#
# print(f"ssims: {ssims}")
# print("---")
# print("mean ssim value: ", np.mean(ssims))
# print("std ssim value: ", np.std(ssims))
#
# exit()


def weighted_z_list(w1, w2, z1_list, z2_list):
    return [w1 * z1 + w2 * z2 for z1, z2 in zip(z1_list, z2_list)]


def create_middle_z_sample(days, z1_list, z2_list):
    w1 = (days[2] - days[1]) / (days[2] - days[0])
    w2 = (days[1] - days[0]) / (days[2] - days[0])
    z_new = weighted_z_list(w1, w2, z1_list, z2_list)
    return z_new


def create_future_z_sample(days, z1_list, z2_list):
    z_diff = [z2 - z1 for z1, z2 in zip(z1_list, z2_list)]
    w = (days[2] - days[0]) / (days[1] - days[0])
    z_new = [z1 + w * d for z1, d in zip(z1_list, z_diff)]
    return z_new


def model_output_to_image(image_tensor, logic="experimental"):
    x = image_tensor.numpy()[0]
    # print(f"min: {np.min(x.flatten())}, max: {np.max(x.flatten())}")
    if logic == "experimental":
        x = x - np.min(x.flatten())
        x = x / np.max(x.flatten())
        x = x * 255.0
    else:
        x = np.clip(x * 127 + 128.0, 0, 255)
    x = x.transpose((1, 2, 0)).astype(np.uint8)
    return x


def calculate_ssim_for_triplets(triplet_list, model, type="missing"):
    ssims = []
    counter = 0
    for data in triplet_list:
        counter += 1
        image_paths, days = data
        images = [
            read_image(
                image_path,
                size=(args.img_size, args.img_size),
                channel_first=True,
                as_batch=True,
                as_torch_tensor=True,
                normalize=True,
                return_original=True,
            )
            for image_path in image_paths
        ]
        if type == "missing":
            with torch.no_grad():
                _, _, z0 = model(images[0][0])
                _, _, z2 = model(images[2][0])
                z_missing = create_middle_z_sample(days=days, z1_list=z0, z2_list=z2)
                im_missing = model.reverse(z_missing, reconstruct=True)
            im_missing = model_output_to_image(
                im_missing, logic="new"
            )  # tensor to uint8 image
            im_original = images[1][1]
            ssim = structural_similarity(
                im1=cv2.cvtColor(im_original, cv2.COLOR_RGB2GRAY),
                im2=cv2.cvtColor(im_missing, cv2.COLOR_RGB2GRAY),
                data_range=255,
            )
            ssims.append(ssim)
            # cv2.imshow("original", cv2.cvtColor(im_original, cv2.COLOR_RGB2GRAY))
            # cv2.imshow("missing", cv2.cvtColor(im_missing, cv2.COLOR_RGB2GRAY))
            # cv2.waitKey(5)
            print(f"{counter} / {len(triplet_list)} : {ssim}")
        elif type == "future":
            with torch.no_grad():
                _, _, z0 = model(images[0][0])
                _, _, z1 = model(images[1][0])
                z_missing = create_future_z_sample(days=days, z1_list=z0, z2_list=z1)
                im_missing = model.reverse(z_missing, reconstruct=True)
            im_missing = model_output_to_image(
                im_missing, logic="new"
            )  # tensor to uint8 image
            im_original = images[2][1]
            ssim = structural_similarity(
                im1=cv2.cvtColor(im_original, cv2.COLOR_RGB2GRAY),
                im2=cv2.cvtColor(im_missing, cv2.COLOR_RGB2GRAY),
                data_range=255,
            )
            ssims.append(ssim)
            print(f"({type}) {counter} / {len(triplet_list)} : {ssim}")
    return ssims


def print_ssims(ssims, title=""):
    print(title)
    print("mean ssim value: ", np.mean(ssims))
    print("std ssim value: ", np.std(ssims))
    print(f"ssims: {ssims}")
    print("---")


ad_ssims = calculate_ssim_for_triplets(
    # longitudinal_dataset.get_ad_image_triplets(), model_single, type="missing"
    longitudinal_dataset.get_ad_image_triplets(),
    model,
    type="missing",
)
print_ssims(ad_ssims, "AD")
cn_ssims = calculate_ssim_for_triplets(
    # longitudinal_dataset.get_cn_image_triplets(), model_single, type="missing"
    longitudinal_dataset.get_cn_image_triplets(),
    model,
    type="missing",
)
print_ssims(cn_ssims, "CN")
mci_ssims = calculate_ssim_for_triplets(
    # longitudinal_dataset.get_mci_image_triplets(), model_single, type="missing"
    longitudinal_dataset.get_mci_image_triplets(),
    model,
    type="missing",
)
print_ssims(mci_ssims, "MCI")


# ad_ssims = calculate_ssim_for_triplets(
#     longitudinal_dataset.get_ad_image_triplets(), model_single, type="future"
# )
# print_ssims(ad_ssims, "AD")
# cn_ssims = calculate_ssim_for_triplets(
#     longitudinal_dataset.get_cn_image_triplets(), model_single, type="future"
# )
# print_ssims(cn_ssims, "CN")
# mci_ssims = calculate_ssim_for_triplets(
#     longitudinal_dataset.get_mci_image_triplets(), model_single, type="future"
# )
# print_ssims(mci_ssims, "MCI")


# for data in longitudinal_dataset.get_ad_image_triplets():

# z_sample = []
# z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
# for z in z_shapes:
#     z_new = torch.randn(args.n_sample, *z) * args.temp
#     z_sample.append(z_new)
#
# for z in z_sample:
#     print(z.shape)
#     print(z.dtype)
# print("now, using these z vectors, we generate images using the model")
# z_outs, gen_im = sample(model_single, z_sample)
# diffs = []
# for z_in, z_out in zip(z_sample, z_outs):
#     z_in = np.asarray(z_in)
#     z_out = np.asarray(z_out)
#     diffs.append(z_out - z_in)
# for d in diffs:
#     print(f"Diff value: {np.mean(d)}, {np.min(d)}, {np.max(d)}")
# for d in z_sample:
#     d = np.asarray(d)
#     print(f"Input z value: {np.mean(d)}, {np.min(d)}, {np.max(d)}")
# gen_im = gen_im.numpy()
# print(
#     f"min, max values of output image is ({np.min(gen_im.flatten())},{np.max(gen_im.flatten())})"
# )
# for i in range(gen_im.shape[0]):
#     x = gen_im[i]
#     x = x - np.min(x.flatten())
#     x = x / np.max(x.flatten())
#     x = x * 255.0
#     # x = (gen_im[i] * 127 + 127)
#     x = x.transpose((1, 2, 0)).astype(np.uint8)
#     cv2.imshow(str(i), x)
#
# print("z_outs.shape: ", len(z_outs))
# print("genim type: ", type(gen_im))
# print("genim dtype: ", gen_im.dtype)
# print("genim shape: ", gen_im.shape)
# cv2.waitKey()
# cv2.destroyAllWindows()
# exit()


# print("shapes:")
# z_sample = []
# z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
# for z in z_shapes:
#     z_new = torch.randn(args.n_sample, *z) * args.temp
#     print(z_new.shape)
#     z_sample.append(z_new)
# print("----")
#
# ims = model_single.reverse(z_sample)
# print(f"ims shape: {ims.shape}")

# gen_im = tensor_to_image(gen_im)
# print(f"gen_im   min: {np.min(gen_im.flatten())}, max: {np.max(gen_im.flatten())}")
# # print(z)
# cv2.imshow("image", tensor_to_image(image))
# cv2.imshow("gen", gen_im)
# cv2.waitKey()
