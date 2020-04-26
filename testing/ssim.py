import copy
import os
import random

import imageio
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datasets.longitudinal_dataset import LongitudinalDataset

# data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/test'
data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/train'


longitudinal_dataset = LongitudinalDataset(data_dir=data_dir)

train_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/train')
val_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/val')
test_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/test')



# =====================================================================
# SSIM between mid image and weighted mean image

# For sequence  im1----t1----im2----t2----im3
# im_w = ( t2 * im1 + t1 * t3 ) / (t1 + t2)

# It prints SSIM between im_w and im2

run_this_code = False

if run_this_code:
    # initialize here
    ad_ims = longitudinal_dataset.get_ad_image_triplets()
    title = 'AD patients SSIM of average image'


    random.shuffle(ad_ims)

    ssim_indexes = []
    for triplet_data in ad_ims:
        triplet, days = triplet_data
        im1 = imageio.imread(triplet[0])
        im2 = imageio.imread(triplet[1])
        im3 = imageio.imread(triplet[2])
        average_img = (im1.astype(np.float) * (days[2] - days[1]) + im3.astype(np.float) * (days[1] - days[0]) ) / (days[2] - days[0])
        average_img = average_img.astype(np.uint8)
        ssim_index = ssim(im1=average_img, im2=im2, data_range=255)
        ssim_indexes.append(ssim_index)
        # print('ssim index: {}     days: {}'.format(round(ssim_index, 2), days))
        # img = np.hstack((average_img, im2))
        # cv2.imshow('img', img)
        # pressed_key = cv2.waitKey()
        # if pressed_key == ord('q'):
        #     break

    mean_ssim = np.mean(ssim_indexes)
    std_ssim = np.std(ssim_indexes)
    print('mean: {}'.format(mean_ssim))
    print('std : {}'.format(std_ssim))

    plt.figure()
    figure_path = os.path.join("/Users/umutkucukaslan/Desktop/thesis/testing", title + '.jpg')
    plt.hist(ssim_indexes, bins=30)
    plt.title(title)
    plt.xlabel('SSIM index')
    plt.ylabel('# images')
    plt.savefig(figure_path, dpi=300)
    plt.show()

# =====================================================================
# =====================================================================
# SSIM between first and followup scans

run_this_code = True

if run_this_code:
    # initialize here
    # seqs = train_dataset.get_ad_longitudinal_sequences() + val_dataset.get_ad_longitudinal_sequences() + test_dataset.get_ad_longitudinal_sequences()
    # seqs = train_dataset.get_mci_longitudinal_sequences() + val_dataset.get_mci_longitudinal_sequences() + test_dataset.get_mci_longitudinal_sequences()
    seqs = train_dataset.get_cn_longitudinal_sequences() + val_dataset.get_cn_longitudinal_sequences() + test_dataset.get_cn_longitudinal_sequences()

    mono_dec_counter = 0
    non_mono_dec_counter = 0

    for seq in seqs:
        ims, days = seq
        metrics = []
        im_base = imageio.imread(ims[0])
        im_show = copy.deepcopy(im_base)
        for i in range(1, len(ims)):
            im_future = imageio.imread(ims[i])
            im_show = np.hstack((im_show, im_future))
            metrics.append(ssim(im_base, im_future, data_range=255))

        metrics = [round(x, 3) for x in metrics]
        monotonically_decreasing = np.all(np.diff(metrics) < 0)
        if monotonically_decreasing:
            print("YES    New seq: ", metrics)
            mono_dec_counter += 1
        else:
            print("NO     New seq: ", metrics)
            non_mono_dec_counter += 1

        cv2.imshow('img', im_show)
        pressed_key = cv2.waitKey()
        if pressed_key == ord('q'):
            break

    print("# mon. decreasing: ", mono_dec_counter)
    print("# non mon. decreasing: ", non_mono_dec_counter)
    print("Percentage of monotonically decreasing is {}".format(mono_dec_counter / (mono_dec_counter + non_mono_dec_counter)))



    # title = 'AD patients SSIM of average image'
    #
    # random.shuffle(seqs)
    #
    # ssim_indexes = []
    # for triplet_data in ad_ims:
    #     triplet, days = triplet_data
    #     im1 = imageio.imread(triplet[0])
    #     im2 = imageio.imread(triplet[1])
    #     im3 = imageio.imread(triplet[2])
    #     average_img = (im1.astype(np.float) * (days[2] - days[1]) + im3.astype(np.float) * (days[1] - days[0])) / (
    #                 days[2] - days[0])
    #     average_img = average_img.astype(np.uint8)
    #     ssim_index = ssim(im1=average_img, im2=im2, data_range=255)
    #     ssim_indexes.append(ssim_index)
    #     # print('ssim index: {}     days: {}'.format(round(ssim_index, 2), days))
    #     # img = np.hstack((average_img, im2))
    #     # cv2.imshow('img', img)
    #     # pressed_key = cv2.waitKey()
    #     # if pressed_key == ord('q'):
    #     #     break
    #
    # mean_ssim = np.mean(ssim_indexes)
    # std_ssim = np.std(ssim_indexes)
    # print('mean: {}'.format(mean_ssim))
    # print('std : {}'.format(std_ssim))
    #
    # plt.figure()
    # figure_path = os.path.join("/Users/umutkucukaslan/Desktop/thesis/testing", title + '.jpg')
    # plt.hist(ssim_indexes, bins=30)
    # plt.title(title)
    # plt.xlabel('SSIM index')
    # plt.ylabel('# images')
    # plt.savefig(figure_path, dpi=300)
    # plt.show()


exit()


ad_ims = longitudinal_dataset.get_ad_image_pairs()
random.shuffle(ad_ims)

for pair_data in ad_ims:
    pair, days = pair_data
    im1 = imageio.imread(pair[0])
    im2 = imageio.imread(pair[1])
    ssim_index = ssim(im1=im1, im2=im2, data_range=255)
    print('ssim index: {}     days: {}'.format(round(ssim_index, 2), days))
    img = np.hstack((im1, im2))
    cv2.imshow('img', img)
    pressed_key = cv2.waitKey()
    if pressed_key == ord('q'):
        break

exit()

ad_ims = longitudinal_dataset.get_ad_images()
random.shuffle(ad_ims)

for i in range(0, len(ad_ims) - 1, 2):
    im1 = imageio.imread(ad_ims[i])
    im2 = imageio.imread(ad_ims[i + 1])
    ssim_index = ssim(im1=im1, im2=im2, data_range=255)
    print('ssim index: {}'.format(ssim_index))
    img = np.hstack((im1, im2))
    cv2.imshow('img', img)
    pressed_key = cv2.waitKey()
    if pressed_key == ord('q'):
        break

exit()
for img_path in longitudinal_dataset.get_ad_images():
    img = imageio.imread(img_path)
    ssim_index = ssim(im1=img, im2=img, data_range=255)
    print('ssim index: {}'.format(ssim_index))
    cv2.imshow('img', img)
    pressed_key = cv2.waitKey()
    if pressed_key == ord('q'):
        break

