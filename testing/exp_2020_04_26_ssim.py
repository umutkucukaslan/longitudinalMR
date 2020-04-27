import os
import random

import cv2
import imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from progressbar import ProgressBar, Percentage, Bar, ETA


from datasets.longitudinal_dataset import LongitudinalDataset
from model.autoencoder import build_encoder_with_lrelu_activation, build_decoder_with_lrelu_activation

EXPERIMENT_FOLDER = '/Users/umutkucukaslan/Desktop/thesis/experiments/exp_2020_04_26'
RESTORE_FROM_CHECKPOINT = True
# encoder

INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNEL = 1
filters = (128, 128, 128, 256, 256, 256, 512)
output_shape = 1024
kernel_size = 3
batch_norm = True

encoder = build_encoder_with_lrelu_activation(
    input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
    output_shape=output_shape,
    filters=filters,
    kernel_size=kernel_size,
    batch_normalization=batch_norm,
    name='encoder')


decoder = build_decoder_with_lrelu_activation(
    input_shape=output_shape,
    output_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),
    filters=tuple(reversed(filters)),
    kernel_size=kernel_size,
    batch_normalization=batch_norm,
    name='decoder')

generator = tf.keras.Sequential(name='generator')
generator.add(encoder)
generator.add(decoder)

generator.summary()

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                 generator=generator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)




train_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/train')
val_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/val')
test_dataset = LongitudinalDataset(data_dir='/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/test')



img_paths = train_dataset.get_ad_images()
title = 'train_ad_ssim'

random.shuffle(img_paths)
img_paths = img_paths[:200]

def preprocess_image(img):
    # change image range from [0, 255] to [0, 1] then make it rank 4

    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    return img

def post_process_image(img):
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = np.squeeze(img)
    return img

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
print('starting...')
for idx in range(len(img_paths)):
    print(idx, ' / ', len(img_paths))
    img_path = img_paths[idx]
    img = imageio.imread(img_path)
    preprocessed_img = preprocess_image(img)
    generated_img = generator(preprocessed_img)
    generated_img = generated_img.numpy()
    generated_img = post_process_image(generated_img)

    ssim_index = ssim(im1=img, im2=generated_img, data_range=255)
    ssim_indexes.append(ssim_index)
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
print('mean: {}'.format(mean_ssim))
print('std : {}'.format(std_ssim))

mean_ssim = round(mean_ssim, 3)
std_ssim = round(std_ssim, 3)


plt.figure()
figure_path = os.path.join("/Users/umutkucukaslan/Desktop/thesis/", title + '.jpg')
plt.hist(ssim_indexes, bins=30)
plt.title(title + ' mean: {}  std: {}'.format(mean_ssim, std_ssim))
plt.xlabel('SSIM index')
plt.ylabel('# images')
plt.savefig(figure_path, dpi=300)
plt.show()
