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

if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, 'testing')):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'testing'))

# encoder.save(os.path.join(EXPERIMENT_FOLDER, 'testing', 'encoder'), include_optimizer=False)

data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data'
N_SAMPLES = 500

train_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'train'))
val_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'val'))
test_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'test'))



# ====================================
# define cases for testing
cases = []

img_paths = train_dataset.get_ad_images()
title = 'train_ad'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# -----------------------------------

img_paths = train_dataset.get_mci_images()
title = 'train_mci'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# -----------------------------------

img_paths = train_dataset.get_cn_images()
title = 'train_cn'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# -----------------------------------

img_paths = test_dataset.get_ad_images()
title = 'test_ad'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# -----------------------------------

img_paths = test_dataset.get_mci_images()
title = 'test_mci'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# -----------------------------------

img_paths = test_dataset.get_cn_images()
title = 'test_cn'

random.shuffle(img_paths)
img_paths = img_paths[:N_SAMPLES]
cases.append((img_paths, title))
# ====================================


for img_paths, title in cases:

    widgets = [
                "Running: ",
                Percentage(),
                " ",
                Bar(marker="#", left="[", right="]"),
                " ",
                ETA(),
            ]
    pbar = ProgressBar(widgets=widgets, maxval=len(img_paths))
    pbar.start()
    ssim_indexes = []
    mses = []
    print('starting...')
    for idx in range(len(img_paths)):
        print(title, '   ', idx, ' / ', len(img_paths))
        img_path = img_paths[idx]
        img = imageio.imread(img_path)
        preprocessed_img = preprocess_image(img)
        generated_img = generator(preprocessed_img)
        generated_img = generated_img.numpy()
        generated_img = postprocess_image(generated_img)

        ssim_index = structural_similarity(im1=img, im2=generated_img, data_range=255)
        ssim_indexes.append(ssim_index)

        mse = mse_uint8(im1=img, im2=generated_img)
        mses.append(mse)
        pbar.update(idx)
        # print('ssim index: {}'.format(round(ssim_index, 2)))
        # img = np.hstack((img, generated_img))
        # cv2.imshow('img', img)
        # pressed_key = cv2.waitKey()
        # if pressed_key == ord('q'):
        #     break
    pbar.finish()

    mean_ssim = np.mean(ssim_indexes)
    std_ssim = np.std(ssim_indexes)
    print('mean ssim : {}'.format(mean_ssim))
    print('std ssim  : {}'.format(std_ssim))

    mean_ssim = round(mean_ssim, 4)
    std_ssim = round(std_ssim, 4)

    mean_mse = np.mean(mses)
    std_mse = np.std(mses)
    print('mean mse : {}'.format(mean_mse))
    print('std mse  : {}'.format(std_mse))

    mean_mse = round(mean_mse, 4)
    std_mse = round(std_mse, 4)

    plt.figure()
    figure_path = os.path.join(EXPERIMENT_FOLDER, 'testing', 'ssim_' + title + '.jpg')
    plt.hist(ssim_indexes, bins=30)
    plt.title(title + '_ssim  mean: {}  std: {}'.format(mean_ssim, std_ssim))
    plt.xlabel('SSIM index')
    plt.ylabel('# images')
    plt.savefig(figure_path, dpi=300)

    plt.figure()
    figure_path = os.path.join(EXPERIMENT_FOLDER, 'testing', 'mse_' + title + '.jpg')
    plt.hist(mses, bins=30)
    plt.title(title + '_mse  mean: {}  std: {}'.format(mean_mse, std_mse))
    plt.xlabel('MSE')
    plt.ylabel('# images')
    plt.savefig(figure_path, dpi=300)

    plt.show()



