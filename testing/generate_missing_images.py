import os
import cv2
import numpy as np
import imageio

from datasets.longitudinal_dataset import LongitudinalDataset
from experiments.exp_2020_05_12 import get_encoder_decoder_generator_discriminator
from testing.utils import preprocess_image, postprocess_image, mse_float, mse_uint8


encoder, decoder, generator, discriminator, EXPERIMENT_FOLDER = get_encoder_decoder_generator_discriminator(return_experiment_folder=True)

results_folder = os.path.join(EXPERIMENT_FOLDER, 'testing/sequences')
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)

# encoder.save(os.path.join(EXPERIMENT_FOLDER, 'testing', 'encoder'), include_optimizer=False)

data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_192x160'

train_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'train'))
val_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'val'))
test_dataset = LongitudinalDataset(data_dir=os.path.join(data_dir, 'test'))


def generate_sequence(seq, timepoints, save_folder):
    imgs, days = seq
    slice_name = os.path.splitext(os.path.basename(imgs[0]))[0]
    patient_name = os.path.basename(os.path.dirname(os.path.dirname(imgs[0])))
    patient_type = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(imgs[0]))))
    norm_days = [x / days[2] for x in days]
    imgs = [imageio.imread(x) for x in imgs]
    vecs = [encoder(preprocess_image(imgs[0])), encoder(preprocess_image(imgs[2]))]
    diff_vec = vecs[1] - vecs[0]
    target_vecs = [vecs[0] + diff_vec * x for x in timepoints]
    target_imgs = [postprocess_image(decoder(x)) for x in target_vecs]
    target_imgs = [cv2.putText(target_imgs[i], 't=' + str(round(timepoints[i], 2)), (1, 10), cv2.FONT_HERSHEY_SIMPLEX, color=255, fontScale=0.3) for i in range(len(target_imgs))]
    target_img = np.zeros((192, 160 * len(target_imgs)), dtype=np.uint8)
    real_img = np.zeros((192, 160 * len(target_imgs)), dtype=np.uint8)
    for i in range(len(target_imgs)):
        idx = 160 * i
        target_img[:, idx: idx+160] = target_imgs[i]
        for d in range(len(norm_days)):
            if norm_days[d] == timepoints[i]:
                temp = imgs[d]
                if norm_days[d] == 0 or norm_days[d] == 1:
                    temp[5:20, 5:20] = 255
                real_img[:, idx:idx+160] = temp
    out_img = np.vstack((real_img, target_img))

    cv2.imshow('out img - ' + patient_name, out_img)
    pressed_key = cv2.waitKey()
    if pressed_key == ord('s'):
        path = os.path.join(save_folder, patient_type + '_' + patient_name + '_' + slice_name + '.png')
        imageio.imwrite(path, out_img)
        cv2.destroyWindow('out img - ' + patient_name)
    if pressed_key == ord('d'):
        cv2.destroyWindow('out img - ' + patient_name)


def populate_vec(vec, step):
    check = False
    for i in range(len(vec) - 1):
        if vec[i+1] - vec[i] > step:
            check = True
            vec.insert(i + 1, (vec[i+1] + vec[i]) / 2)
    if check:
        vec = populate_vec(vec, step)

    return vec


def ui(sequences):
    idx = 0
    pressed_key = ord('0')
    while pressed_key != ord('q'):
        seq = sequences[idx]
        imgs, days = seq
        bl_img = imageio.imread(imgs[0])
        cv2.imshow('baseline slice', bl_img)
        pressed_key = cv2.waitKey()

        if pressed_key == ord('n'):
            idx = min(len(sequences) - 1, idx + 1)
        if pressed_key == ord('b'):
            idx = max(0, idx - 1)
        if pressed_key == ord('m'):
            idx = min(len(sequences) - 1, idx + 70)
        if pressed_key == ord('v'):
            idx = max(0, idx - 70)
        if pressed_key == ord('g'):
            timepoints = [x / days[2] for x in days]
            timepoints = populate_vec(timepoints, 0.25)
            generate_sequence(seq, timepoints, results_folder)


sequences = train_dataset.get_ad_longitudinal_sequences()
ui(sequences)
sequences = train_dataset.get_mci_longitudinal_sequences()
ui(sequences)
sequences = train_dataset.get_cn_longitudinal_sequences()
ui(sequences)


sequences = test_dataset.get_ad_longitudinal_sequences()
ui(sequences)
sequences = test_dataset.get_mci_longitudinal_sequences()
ui(sequences)
sequences = test_dataset.get_cn_longitudinal_sequences()
ui(sequences)