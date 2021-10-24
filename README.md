
# Longitudinal Brain MR Image Modeling using Personalized Memory for Alzheimer's Disease

This repository contains the code for model definition, training and testing for reproducing the results in the article.

https://ieeexplore.ieee.org/document/9580859

## Install Requirements
```shell
pip3 install tensorflow opencv-python nibabel imageio pandas tqdm
```

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