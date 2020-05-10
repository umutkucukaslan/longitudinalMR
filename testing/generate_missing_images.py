
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from progressbar import Percentage, Bar, ETA, ProgressBar
import imageio
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset
from experiments.exp_2020_05_09 import get_encoder_decoder_generator_discriminator
from testing.utils import preprocess_image, postprocess_image, mse_float, mse_uint8


encoder, decoder, generator, discriminator, EXPERIMENT_FOLDER = get_encoder_decoder_generator_discriminator(return_experiment_folder=True)

if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, 'testing/longitudinal')):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'testing/longitudinal'))

# encoder.save(os.path.join(EXPERIMENT_FOLDER, 'testing', 'encoder'), include_optimizer=False)

data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_192x160'
N_SAMPLES = 500

train_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'train'))
val_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'val'))
test_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'test'))

triplets = train_dataset.get_ad_image_triplets()


def blend_vectors(vecs, days):
    diff = vecs[2] - vecs[0]
    blend_vector = vecs[0] + diff * (days[1] - days[0]) / (days[2] - days[0])

    return blend_vector


def generate_statistics(triplets):
    ssims = []
    for triplet in triplets:
        imgs, days = triplet
        imgs = [imageio.imread(x) for x in imgs]
        encoder_inps = [preprocess_image(x) for x in imgs]
        vecs = [encoder(x) for x in encoder_inps]
        blend_vec = blend_vectors(vecs, days)
        generated_img = postprocess_image(decoder(blend_vec))
        ssim = structural_similarity(imgs[1], generated_img)
        ssims.append(ssim)


        print(triplet)
        break


generate_statistics(triplets)


