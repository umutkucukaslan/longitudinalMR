
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from progressbar import Percentage, Bar, ETA, ProgressBar
import imageio
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset
from experiments.exp_2020_05_09_3 import get_encoder_decoder_generator_discriminator
from testing.utils import preprocess_image, postprocess_image, mse_float, mse_uint8


encoder, decoder, generator, discriminator, EXPERIMENT_FOLDER = get_encoder_decoder_generator_discriminator(return_experiment_folder=True)

results_folder = os.path.join(EXPERIMENT_FOLDER, 'testing/longitudinal')
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)

# encoder.save(os.path.join(EXPERIMENT_FOLDER, 'testing', 'encoder'), include_optimizer=False)

data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_192x160'
N_SAMPLES = 250

train_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'train'))
val_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'val'))
test_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'test'))


def blend_vectors(vecs, days):
    diff = vecs[2] - vecs[0]
    blend_vector = vecs[0] + diff * (days[1] - days[0]) / (days[2] - days[0])

    return blend_vector


def blend_images(imgs, days):
    imgs = [x.astype(np.float32) if x is not None else None for x in imgs]
    diff = imgs[2] - imgs[0]
    blend_img = imgs[0] + diff * (days[1] - days[0]) / (days[2] - days[0])

    return np.clip(blend_img, 0, 255).astype(np.uint8)


def generate_statistics(triplets, title):
    ssims_original_generated = []
    mses_original_generated = []
    ssims_original_mean = []
    mses_original_mean = []

    ssim_diff_mean_generated = []
    ssim_diff_truemean_generated = []

    for triplet_idx in range(len(triplets)):
        print(title, '   ', triplet_idx, ' / ', len(triplets))
        triplet = triplets[triplet_idx]
        imgs, days = triplet
        imgs = [imageio.imread(x) for x in imgs]
        encoder_inps = [preprocess_image(x) for x in imgs]
        vecs = [encoder(x) for x in encoder_inps]
        blend_vec = blend_vectors(vecs, days)
        generated_img = postprocess_image(decoder(blend_vec))
        ssim1 = structural_similarity(imgs[1], generated_img)    # ssim between original and generated missing images
        ssims_original_generated.append(ssim1)
        mse1 = mse_uint8(imgs[1], generated_img)     # mse between original and generated missing images
        mses_original_generated.append(mse1)

        # mean_img = blend_images(imgs, days)
        # ssim2 = structural_similarity(imgs[1], mean_img)  # ssim between original and mean missing images
        # ssims_original_mean.append(ssim2)
        # mse2 = mse_uint8(imgs[1], mean_img)  # mse between original and mean missing images
        # mses_original_mean.append(mse2)

        # true_mean_img = blend_images([postprocess_image(decoder(vecs[0])), None, postprocess_image(decoder(vecs[0]))], days)
        # ssim3 = structural_similarity(imgs[1], true_mean_img)  # ssim between original and true mean missing images
        # ssims_original_mean.append(ssim3)
        # mse3 = mse_uint8(imgs[1], true_mean_img)  # mse between original and true mean missing images
        # mses_original_mean.append(mse3)

        # ssim_diff_mean_generated.append(ssim1 - ssim2)
        # ssim_diff_truemean_generated.append(ssim1 - ssim3)


    plt.figure()
    figure_path = os.path.join(results_folder, 'SSIM (actual vs OURS) ' + title + '.jpg')
    plt.hist(ssims_original_generated, bins=100, range=(0, 1))
    mean_ssim = np.mean(ssims_original_generated)
    std_ssim = np.std(ssims_original_generated)
    mean_ssim = round(mean_ssim, 3)
    std_ssim = round(std_ssim, 3)
    plt.title(title + ' SSIM (actual vs OURS)  mean: {}  std: {}'.format(mean_ssim, std_ssim))
    plt.xlabel('SSIM index')
    plt.ylabel('# images')
    plt.xticks(np.arange(0, 1.001, 0.1))
    plt.savefig(figure_path, dpi=300)

    # plt.figure()
    # figure_path = os.path.join(results_folder, 'SSIM (actual vs mean) ' + title + '.jpg')
    # plt.hist(ssims_original_mean, bins=100, range=(0, 1))
    # mean_ssim = np.mean(ssims_original_mean)
    # std_ssim = np.std(ssims_original_mean)
    # mean_ssim = round(mean_ssim, 3)
    # std_ssim = round(std_ssim, 3)
    # plt.title(title + ' SSIM (actual vs mean image)  mean: {}  std: {}'.format(mean_ssim, std_ssim))
    # plt.xlabel('SSIM index')
    # plt.ylabel('# images')
    # plt.xticks(np.arange(0, 1.0001, 0.1))
    # plt.savefig(figure_path, dpi=300)

    # plt.figure()
    # figure_path = os.path.join(results_folder, 'DIFF (OURS - mean of actuals) ' + title + '.jpg')
    # plt.hist(ssim_diff_mean_generated, bins=100)
    # mean_ssim = np.mean(ssim_diff_mean_generated)
    # std_ssim = np.std(ssim_diff_mean_generated)
    # mean_ssim = round(mean_ssim, 3)
    # std_ssim = round(std_ssim, 3)
    # plt.title(title + ' (OURS - mean of actuals)  mean: {}  std: {}'.format(mean_ssim, std_ssim))
    # plt.xlabel('SSIM index')
    # plt.ylabel('# images')
    # plt.savefig(figure_path, dpi=300)

    # plt.figure()
    # figure_path = os.path.join(results_folder, 'DIFF (OURS - mean of generated images) ' + title + '.jpg')
    # plt.hist(ssim_diff_truemean_generated, bins=100)
    # mean_ssim = np.mean(ssim_diff_truemean_generated)
    # std_ssim = np.std(ssim_diff_truemean_generated)
    # mean_ssim = round(mean_ssim, 3)
    # std_ssim = round(std_ssim, 3)
    # plt.title(title + ' (OURS-mean of generateds)  mean: {}  std: {}'.format(mean_ssim, std_ssim))
    # plt.xlabel('SSIM index')
    # plt.ylabel('# images')
    # plt.savefig(figure_path, dpi=300)






# triplets = train_dataset.get_ad_image_triplets()
# random.shuffle(triplets)
# triplets = triplets[:N_SAMPLES]
# generate_statistics(triplets, 'train-ad')
#
# triplets = train_dataset.get_mci_image_triplets()
# random.shuffle(triplets)
# triplets = triplets[:N_SAMPLES]
# generate_statistics(triplets, 'train-mci')

triplets = train_dataset.get_cn_image_triplets()
random.shuffle(triplets)
triplets = triplets[:N_SAMPLES]
generate_statistics(triplets, 'train-cn')


triplets = test_dataset.get_ad_image_triplets()
random.shuffle(triplets)
triplets = triplets[:N_SAMPLES]
generate_statistics(triplets, 'test-ad')

triplets = test_dataset.get_mci_image_triplets()
random.shuffle(triplets)
triplets = triplets[:N_SAMPLES]
generate_statistics(triplets, 'test-mci')

triplets = test_dataset.get_cn_image_triplets()
random.shuffle(triplets)
triplets = triplets[:N_SAMPLES]
generate_statistics(triplets, 'test-cn')