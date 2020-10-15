import glob
import os
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from datasets.longitudinal_dataset import LongitudinalDataset, Patient
from reference_papers.spie_paper.train_wgan2 import get_generator_discriminator


def imtoshow(image):
    image = 127 * image + 127
    image = image.astype(np.uint8)
    return image[0, :, :, 0]


generator, discriminator, experiment_folder = get_generator_discriminator()
del discriminator

actual_data_path = (
    "/Users/umutkucukaslan/Desktop/thesis/dataset/training_data_15T_192x160_4slices/val"
)
generated_data_path = (
    "/Users/umutkucukaslan/Desktop/thesis/experiments/ref_spie_wgan_2/val"
)
patient_name = "ad_109_S_1157"

actual = Patient(os.path.join(actual_data_path, patient_name))
generated = Patient(os.path.join(generated_data_path, patient_name))

print(actual.relative_dates)
print(generated.relative_dates)

all_actual_image_triplets = actual.get_all_image_triplets()
all_generated_image_triplets = generated.get_all_image_triplets()

print(all_actual_image_triplets)
print(all_generated_image_triplets)

ssims = []
reconst_ssims = []
for a, g in zip(all_actual_image_triplets, all_generated_image_triplets):
    actual_series, _ = a
    generated_series, d = g
    latents = []
    for p in generated_series:
        name = os.path.basename(p)[4:13]
        name = "encoding_" + name + ".npy"
        latent_path = os.path.join(os.path.dirname(p), name)
        latent = np.load(latent_path)
        latents.append(latent)
    mixed_latent = (latents[0] * (d[2] - d[1]) + latents[2] * (d[1] - d[0])) / (
        d[2] - d[0]
    )

    diff = latents[2] - latents[0]
    seq = [latents[0] + x * diff for x in np.linspace(0, 4, 20)]
    seq = [imtoshow(generator(l, training=False).numpy()) for l in seq]
    seq = np.hstack(seq)
    cv2.imwrite(f"/Users/umutkucukaslan/Desktop/seq/seq_{d[0]}_{d[1]}_{d[2]}.jpg", seq)
    combined_image = generator(mixed_latent, training=False)
    combined_image = imtoshow(combined_image.numpy())
    actual_combined_image = cv2.imread(actual_series[1])
    actual_combined_image = cv2.resize(actual_combined_image, (64, 64))
    actual_combined_image = cv2.cvtColor(actual_combined_image, cv2.COLOR_BGR2GRAY)
    ssim_combined = structural_similarity(
        actual_combined_image, combined_image, data_range=255
    )
    ssims.append(ssim_combined)

    for a_p, g_p in zip(actual_series, generated_series):
        a = cv2.imread(a_p)
        a = cv2.cvtColor(cv2.resize(a, (64, 64)), cv2.COLOR_BGR2GRAY)
        g = cv2.imread(g_p)
        g = cv2.cvtColor(cv2.resize(g, (64, 64)), cv2.COLOR_BGR2GRAY)
        s = structural_similarity(a, g, data_range=255)
        reconst_ssims.append(s)


print(ssims)
print(f"mean ssim: {np.mean(ssims)}")
print(f"mean std: {np.std(ssims)}")

print(reconst_ssims)
print(f"mean ssim: {np.mean(reconst_ssims)}")
print(f"mean std: {np.std(reconst_ssims)}")
