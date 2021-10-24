
# Longitudinal Brain MR Image Modeling using Personalized Memory for Alzheimer's Disease

This repository contains the code for model definition, training and testing for reproducing the results in the article.

https://ieeexplore.ieee.org/document/9580859

## Install Requirements
```shell
pip3 install tensorflow opencv-python nibabel imageio pandas tqdm
```

## Training and Inference Scripts

The training scripts can be found in experiments folder.

* `proposed_method` script builds the proposed autoencoder architecture, trained the model using a longitudinal slice dataset, each example of which contains images from three different time points, where the two is used to generate two points in the latent space and the point in latent space that should correspond to the third one is interpolated using the time info, from which the output image is generated.

* `proposed_method_wihtout_sequence_learning` script builds the same proposed autoencoder model, but trains it with plane images without any time info, meaning that the input image is first encoded into latent vector, from which the output image is directly generated and compared with the input. (no disentanglement in latent space)

* `simple_autoencoder` script does the same as `proposed_method_wihtout_sequence_learning` but uses a simple autoencoder architecture. (no disentanglement in latent space)

* `simple_autoencoder_sequence_learning` script does the same as `propsed_method` but uses a simple autoencoder architecture.

The inference scripts can be found in testing folder.

* `interpolate_using_proposed_method` script interpolated images using the proposed model

* `interpolate_using_simple_autoencoder` script interpolated images using the simple autoencoder model

* `visualize_statistics` visualizes some of the results saved in csv files during inference/testing


## Model files

* `ae.ae` contains the definition of the proposed architecture

* `ae.ae_basic` contains the detinition of the simple autoencoder used for ablation study

## Running on Google Colab

Upload the training data to Google Drive under My Drive folder. Open a colab instance with GPU runtime.

Mount Google Drive, unzip the training data to local disk and clone the repository.
```
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/My\ Drive/training_data_15T_192x160_4slices.zip -d /content
! cd /content && git clone https://github.com/umutkucukaslan/longitudinalMR.git
```
Pull recent changes, and run the module.
```
! cd /content/longitudinalMR && git pull -q
print('Pulled recent changes from the repository.')
print('Running module...')
# run module for training
! cd /content/longitudinalMR && python3 -m experiments.exp_2020_12_22_glow 32
# run module for testing
! cd /content/longitudinalMR && python3 -m testing.interpolate_2021_01_31_custom_ae 1
```
The experiment results and checkpoints will be saved to Google Drive under `My Drive/experiments` folder.