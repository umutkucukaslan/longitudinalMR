
import glob
import os
import random
from shutil import copytree

"""
Splits processed dataset into train, val, test folders
Adds ad/mci/cn prefixes to patient folders accordingly
Set the params below
"""
# set following params
raw_dataset = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_256x256_new'
processed_dataset = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_new'
split_ratios = [70, 10, 20]     # train, val, test


train_folder = os.path.join(processed_dataset, 'train')
val_folder = os.path.join(processed_dataset, 'val')
test_folder = os.path.join(processed_dataset, 'test')

if not os.path.isdir(train_folder):
    os.makedirs(train_folder)
if not os.path.isdir(val_folder):
    os.makedirs(val_folder)
if not os.path.isdir(test_folder):
    os.makedirs(test_folder)

ad_folders = [x for x in glob.glob(os.path.join(raw_dataset, 'AD', '*')) if os.path.isdir(x)]
mci_folders = [x for x in glob.glob(os.path.join(raw_dataset, 'MCI', '*')) if os.path.isdir(x)]
cn_folders = [x for x in glob.glob(os.path.join(raw_dataset, 'CN', '*')) if os.path.isdir(x)]

random.shuffle(ad_folders)
random.shuffle(mci_folders)
random.shuffle(cn_folders)


def copy_to_folder(folder, dirs, prefix):
    for dir_path in dirs:
        dir_basename = prefix + "_" + os.path.basename(dir_path)
        target_path = os.path.join(folder, dir_basename)
        copytree(dir_path, target_path)


def split_list(filelist, split_ratios):
    i1 = int(len(filelist) * split_ratios[0] / sum(split_ratios))
    i2 = int(len(filelist) * (split_ratios[0] + split_ratios[1]) / sum(split_ratios))
    return filelist[:i1], filelist[i1:i2], filelist[i2:]

copy_to_folder(train_folder, split_list(ad_folders, split_ratios)[0], 'ad')
copy_to_folder(train_folder, split_list(mci_folders, split_ratios)[0], 'mci')
copy_to_folder(train_folder, split_list(cn_folders, split_ratios)[0], 'cn')

copy_to_folder(val_folder, split_list(ad_folders, split_ratios)[1], 'ad')
copy_to_folder(val_folder, split_list(mci_folders, split_ratios)[1], 'mci')
copy_to_folder(val_folder, split_list(cn_folders, split_ratios)[1], 'cn')

copy_to_folder(test_folder, split_list(ad_folders, split_ratios)[2], 'ad')
copy_to_folder(test_folder, split_list(mci_folders, split_ratios)[2], 'mci')
copy_to_folder(test_folder, split_list(cn_folders, split_ratios)[2], 'cn')