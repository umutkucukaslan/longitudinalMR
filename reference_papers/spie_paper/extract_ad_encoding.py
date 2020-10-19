import glob
import os
import numpy as np

from reference_papers.spie_paper.train_wgan2 import get_generator_discriminator


generator, discriminator, experiment_folder = get_generator_discriminator()
del discriminator, generator

encodings_dir = os.path.join(experiment_folder, "val")
# print('encodings dir: ', encodings_dir)

all_ad_encodings = glob.glob(os.path.join(encodings_dir, "ad_*", "*", "*.npy"))
all_cn_encodings = glob.glob(os.path.join(encodings_dir, "cn_*", "*", "*.npy"))

ad_encodings = [np.load(p) for p in all_ad_encodings]
cn_encodings = [np.load(p) for p in all_cn_encodings]

mean_ad = np.mean(np.asarray(ad_encodings), axis=0)
mean_cn = np.mean(np.asarray(cn_encodings), axis=0)

ad_features = mean_ad - mean_cn
ad_features_path = os.path.join(encodings_dir, "ad_features")
print("ad_features_path: ", ad_features_path)
np.save(ad_features_path, ad_features)
