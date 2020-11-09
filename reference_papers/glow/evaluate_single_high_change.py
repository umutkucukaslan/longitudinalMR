import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset

from reference_papers.glow.model import Glow
from reference_papers.glow.utils import read_image, model_output_to_image


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
model_path = "/Users/umutkucukaslan/Desktop/thesis/experiments/exp_2020_10_30_glow/checkpoint/model_030001.pt"
model_single = Glow(
    3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
)
model = torch.nn.DataParallel(model_single)
loaded_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(loaded_state_dict)
model.eval()  # model in eval mode

data_dir = "/Users/umutkucukaslan/Desktop/thesis/dataset/high_change_val_patients"
dataset = LongitudinalDataset(data_dir=data_dir)

print(len(dataset.get_mci_longitudinal_sequences()))
for x in dataset.get_mci_longitudinal_sequences():
    image_paths, days = x
    relative_days = [(x - days[0]) / (days[-1] - days[0]) for x in days]
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
    with torch.no_grad():
        _, _, z_first = model_single(images[0][0])
        _, _, z_last = model_single(images[-1][0])
    diff = [z_l - z_f for z_f, z_l in zip(z_first, z_last)]
    target_z_list_list = []
    relative_days.append(1.25)
    relative_days.append(1.5)
    relative_days.append(1.75)
    relative_days.append(2)
    relative_days.append(2.25)
    relative_days.append(2.5)
    relative_days.append(2.75)
    relative_days.append(3)
    for w in relative_days:
        z_list = [z_f + w * d for z_f, d in zip(z_first, diff)]
        target_z_list_list.append(z_list)
    with torch.no_grad():
        target_ims = []
        for z_list in target_z_list_list:
            im_tensor = model_single.reverse(z_list, reconstruct=True)
            im = model_output_to_image(im_tensor)
            target_ims.append(im)
    original_ims = [im_tuple[1] for im_tuple in images]
    # diff_to_first = [
    #     np.abs(o.astype(np.int) - t.astype(int)).astype(np.uint8)
    #     for o, t in zip(original_ims, target_ims)
    # ]
    # diff_to_first = np.hstack(diff_to_first)
    mean_images = []
    for w in relative_days[: len(original_ims)]:
        new_image = original_ims[0].astype(np.float) + w * (
            original_ims[-1].astype(np.float) - original_ims[0].astype(np.float)
        )
        mean_images.append(new_image.astype(np.uint8))
    ssims = [
        structural_similarity(
            im1=cv2.cvtColor(o, cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(t, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for o, t in zip(original_ims, target_ims[: len(original_ims)])
    ]
    print(f"ssim between original and generated images: {ssims}")
    ssims_mean = [
        structural_similarity(
            im1=cv2.cvtColor(o, cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(m, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for o, m in zip(original_ims, mean_images)
    ]
    print(f"ssim between original and mean images: {ssims_mean}")
    ssims_to_first_original = [
        structural_similarity(
            im1=cv2.cvtColor(original_ims[0], cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(o, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for o in original_ims
    ]
    print(f"ssim of original images wrt first image: {ssims_to_first_original}")
    ssims_to_last_original = [
        structural_similarity(
            im1=cv2.cvtColor(original_ims[-1], cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(o, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for o in original_ims
    ]
    print(f"ssim of original images wrt last image: {ssims_to_last_original}")
    ssim_to_first_generated = [
        structural_similarity(
            im1=cv2.cvtColor(original_ims[0], cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(t, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for t in target_ims
    ]
    print(f"ssim of generated images wrt first image: {ssim_to_first_generated}")
    ssim_to_last_generated = [
        structural_similarity(
            im1=cv2.cvtColor(original_ims[-1], cv2.COLOR_RGB2GRAY),
            im2=cv2.cvtColor(t, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )
        for t in target_ims
    ]
    print(f"ssim of generated images wrt last image: {ssim_to_last_generated}")
    target_ims = np.hstack(target_ims)
    original_ims = np.hstack(original_ims)
    mean_images = np.hstack(mean_images)
    # diff_to_original = np.abs(
    #     target_ims.astype(np.float) - original_ims.astype(np.float)
    # ).astype(np.uint8)
    cv2.imshow("original", original_ims)
    cv2.imshow("generated", target_ims)
    cv2.imshow("mean", mean_images)
    # cv2.imshow("diff to original", diff_to_original)
    # cv2.imshow("diff to first", diff_to_first)
    cv2.waitKey()
