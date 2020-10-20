import glob
import os
import random
import time

import cv2
import numpy as np

from datasets.longitudinal_dataset import LongitudinalDataset
from reference_papers.spie_paper.image_encoding import encode_image
from reference_papers.spie_paper.train_wgan_rw import get_generator_discriminator


"""

"""

if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "macbook"
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
else:
    raise ValueError("Unknown machine type")

data_folder = "val"

if MACHINE == "macbook":
    data_dir = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/dataset/training_data_15T_192x160_4slices",
        data_folder,
    )
elif MACHINE == "colab":
    data_dir = os.path.join("/content/training_data_15T_192x160_4slices", data_folder)

# longitudinal_dataset = LongitudinalDataset(data_dir=data_dir)
#
# paths = (
#     longitudinal_dataset.get_ad_images()
#     + longitudinal_dataset.get_mci_images()
#     + longitudinal_dataset.get_cn_images()
# )
# paths = sorted(paths)

generator, discriminator, experiment_folder = get_generator_discriminator()
del discriminator

encodings_dir = os.path.join(experiment_folder, data_folder)
ad_features = np.load(os.path.join(encodings_dir, "ad_features.npy"))

all_ad_encodings = glob.glob(os.path.join(encodings_dir, "ad_*", "*", "*.npy"))
all_cn_encodings = glob.glob(os.path.join(encodings_dir, "cn_*", "*", "*.npy"))
all_mci_encodings = glob.glob(os.path.join(encodings_dir, "mci_*", "*", "*.npy"))


def imtoshow(image):
    image = 127 * image + 127
    image = image.astype(np.uint8)
    return image[0, :, :, 0]


print("mean ad features value: ", np.mean(ad_features))
c = 0
c_save = 0
save_dir = "/Users/umutkucukaslan/Desktop/seq"
random.shuffle(all_cn_encodings)
for latent in all_cn_encodings:
    print(f"{c}: latent: ", latent)
    latent = np.load(latent)
    print("mean latent value: ", np.mean(latent))
    weights = [weight for weight in np.linspace(-4, 4, 17)]
    targets = [latent + ad_features * weight for weight in weights]
    target_imgs = [
        imtoshow(generator(target_latent, training=False).numpy())
        for target_latent in targets
    ]
    for i in range(len(weights)):
        w = weights[i]
        cv2.putText(
            target_imgs[i], str(round(w, 2)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255
        )

    collage = np.hstack(target_imgs)
    cv2.imshow("collage", collage)
    p = cv2.waitKey()
    if p == ord("q"):
        exit()
    if p == ord("s"):
        c_save += 1
        save_path = os.path.join(save_dir, f"seq_{c_save}.jpg")
        cv2.imwrite(save_path, collage)


exit()

print("TARGET DIR: ", target_dir)
print("Experiment folder: ", experiment_folder)


c = 0
for p in paths:
    print(f"Processing scan {c} at {p} ...")
    start_time = time.time()
    scan_name = os.path.basename(os.path.dirname(p))
    patient_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    write_to = os.path.join(target_dir, patient_name, scan_name)
    image_name = os.path.basename(p)

    if not os.path.exists(write_to):
        os.makedirs(write_to)

    res_image_path = os.path.join(write_to, "res_" + image_name)
    if os.path.exists(res_image_path):
        print("Already exists")
        c += 1
        continue

    image = cv2.imread(p)
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.squeeze(image)
    image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
    image = np.asarray(image, dtype=np.float)
    image = (image - 127) / 128

    encoding, res_image = encode_image(generator, image, num_steps=1000, verbose=False)
    res_image = imtoshow(res_image)
    cv2.imwrite(res_image_path, res_image)

    encoding_path = os.path.join(
        write_to, "encoding_" + os.path.splitext(image_name)[0]
    )
    np.save(encoding_path, encoding)
    end_time = time.time()
    print(f"Processed scan {c} at {p} in {end_time - start_time} seconds")
    # print((patient_name, scan_name), "  ", p)

    c += 1
