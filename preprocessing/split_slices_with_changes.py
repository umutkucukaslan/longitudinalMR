import os
import glob
import csv
import numpy as np

import random
from shutil import copytree

raw_dataset = (
    "/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data_15T_192x160_4slices"
)
processed_dataset = (
    "/Users/umutkucukaslan/Desktop/thesis/dataset/training_data_15T_192x160_4slices"
)
split_ratios = [70, 10, 20]  # train, val, test


train_folder = os.path.join(processed_dataset, "train")
val_folder = os.path.join(processed_dataset, "val")
test_folder = os.path.join(processed_dataset, "test")

if not os.path.isdir(train_folder):
    os.makedirs(train_folder)
if not os.path.isdir(val_folder):
    os.makedirs(val_folder)
if not os.path.isdir(test_folder):
    os.makedirs(test_folder)

ad_folders = [
    x for x in glob.glob(os.path.join(raw_dataset, "AD", "*")) if os.path.isdir(x)
]
mci_folders = [
    x for x in glob.glob(os.path.join(raw_dataset, "MCI", "*")) if os.path.isdir(x)
]
cn_folders = [
    x for x in glob.glob(os.path.join(raw_dataset, "CN", "*")) if os.path.isdir(x)
]


faulty_15T_path = "/Users/umutkucukaslan/Desktop/thesis/dataset/asıl veriler/dataset_info/faulty_15T.csv"
high_changes_15T_path = "/Users/umutkucukaslan/Desktop/thesis/dataset/asıl veriler/dataset_info/high_change_15T.csv"
middle_changes_15T_path = "/Users/umutkucukaslan/Desktop/thesis/dataset/asıl veriler/dataset_info/middle_change_15T.csv"
small_changes_15T_path = "/Users/umutkucukaslan/Desktop/thesis/dataset/asıl veriler/dataset_info/small_change_15T.csv"
very_small_changes_15T_path = "/Users/umutkucukaslan/Desktop/thesis/dataset/asıl veriler/dataset_info/very_small_change_15T.csv"


def read_patients_from_csv(csv_file_path):
    patients = []
    with open(csv_file_path) as f:
        csv_reader = csv.reader(f, delimiter=" ")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                patients.append(row[0])

    return patients


faulty_15T = read_patients_from_csv(faulty_15T_path)
high_change_15T = read_patients_from_csv(high_changes_15T_path)
middle_changes_15T = read_patients_from_csv(middle_changes_15T_path)
small_change_15T = read_patients_from_csv(small_changes_15T_path)
very_small_change_15T = read_patients_from_csv(very_small_changes_15T_path)

ad_folders = [x for x in ad_folders if os.path.basename(x) not in faulty_15T]
mci_folders = [x for x in mci_folders if os.path.basename(x) not in faulty_15T]
cn_folders = [x for x in cn_folders if os.path.basename(x) not in faulty_15T]


def print_patient_type_stats(patient_folders, all_lists, title=""):
    " print stats for patient type indicating how many subjects there are in each change category"
    s = ["high", "middle", "small", "very small", "none"]
    splitted = [[] for x in range(len(all_lists) + 1)]
    for patient_folder in patient_folders:
        matched = False
        patient_name = os.path.basename(patient_folder)
        for i, l in enumerate(all_lists):
            if patient_name in l:
                splitted[i].append(patient_folder)
                matched = True
                break
        if not matched:
            splitted[-1].append(patient_folder)
    print(title)
    for i, l in enumerate(splitted):
        print("{}: {}".format(s[i], len(l)))


all_change_patientname_lists = [
    high_change_15T,
    middle_changes_15T,
    small_change_15T,
    very_small_change_15T,
]

print_patient_type_stats(ad_folders, all_change_patientname_lists, "AD")
print_patient_type_stats(mci_folders, all_change_patientname_lists, "MCI")
print_patient_type_stats(cn_folders, all_change_patientname_lists, "CN")

splitted_full_paths = [[] for x in range(len(all_change_patientname_lists) + 1)]

for patient_folder_path in ad_folders + mci_folders + cn_folders:
    matched = False
    patient_name = os.path.basename(patient_folder_path)
    for i, l in enumerate(all_change_patientname_lists):
        if patient_name in l:
            splitted_full_paths[i].append(patient_folder_path)
            matched = True
            break
    if not matched:
        splitted_full_paths[-1].append(patient_folder_path)


training_splits = [[], [], []]
s = ["high", "middle", "small", "very small", "no"]
print("")
print("Training dataset stats:")
for i, l in enumerate(splitted_full_paths):
    random.shuffle(l)
    n_samples = len(l)
    n_val = int(np.ceil(split_ratios[1] / np.sum(split_ratios) * n_samples))
    n_test = int(np.ceil(split_ratios[2] / np.sum(split_ratios) * n_samples))
    training_splits[1] = training_splits[1] + l[:n_val]
    training_splits[2] = training_splits[2] + l[n_val : n_val + n_test]
    training_splits[0] = training_splits[0] + l[n_val + n_test :]
    print(
        "For {} change, we have {} train, {} val and {} test samples".format(
            s[i],
            len(l[n_val + n_test :]),
            len(l[:n_val]),
            len(l[n_val : n_val + n_test]),
        )
    )


s = ["Train", "Val", "Test"]
for i, split in enumerate(training_splits):
    print("{}: {}".format(s[i], len(split)))


def copy_to_folder(destination_folder, source_dir_list, use_prefix=True):
    """
    Copy list of folders to destination folder.

    :param destination_folder:
    :param source_dir_list:
    :param use_prefix:
    :return:
    """
    prefix_dict = {"AD": "ad", "MCI": "mci", "CN": "cn"}
    for dir_path in source_dir_list:
        patient_type = os.path.basename(os.path.dirname(dir_path))
        prefix = prefix_dict[patient_type]
        dir_basename = prefix + "_" + os.path.basename(dir_path)
        target_path = os.path.join(destination_folder, dir_basename)
        copytree(dir_path, target_path)


copy_to_folder(train_folder, training_splits[0], use_prefix=True)
copy_to_folder(val_folder, training_splits[1], use_prefix=True)
copy_to_folder(test_folder, training_splits[2], use_prefix=True)
