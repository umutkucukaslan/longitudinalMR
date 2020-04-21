import random
from datetime import datetime
import glob
import os
from random import randint


class Patient:
    def __init__(self, patient_folder_path, patient_type=None):
        self.patient_type = patient_type
        self.folder_path = patient_folder_path
        self.patient_name = os.path.basename(patient_folder_path)
        self.scan_folders = sorted(glob.glob(os.path.join(patient_folder_path, '*')))

        self.dates_str = [os.path.basename(x) for x in self.scan_folders]  # dates of scans in str format
        dates = [datetime.strptime(x, '%Y-%m-%d_%H_%M_%S') for x in self.dates_str]
        relative_dates = [x - dates[0] for x in dates]
        self.relative_dates = [x.days for x in relative_dates]  # scan ages in days relative to first scan

        images = []
        for scan_folder in self.scan_folders:
            slice_paths = sorted(glob.glob(os.path.join(scan_folder, 'slice_*.png')))
            images.append(slice_paths)

        self.images = list(zip(*images))
        self.n_slices = len(self.images)
        self.n_scans = len(dates)
        self.scan_pairs = [(i, j) for i in range(self.n_scans) for j in range(i + 1, self.n_scans)]
        self.scan_triplets = [(i, j, k) for i in range(self.n_scans) for j in range(i + 1, self.n_scans) for k in
                              range(j + 1, self.n_scans)]

    def get_image(self, slice_index=None, scan_index=None):
        """
        Returns image path with specified parameters, if parameters not specified, returns a random slice image
        for this patient

        :param slice_index:
        :param scan_index:
        :return: slice_image_path
        """
        if slice_index is None:
            slice_index = randint(0, self.n_slices - 1)
        if scan_index is None:
            scan_index = randint(0, self.n_scans - 1)

        assert slice_index < self.n_slices
        assert scan_index < self.n_scans

        return self.images[slice_index][scan_index]

    def get_all_images(self):
        """
        Returns list of all image paths from this patient's scans.

        :return:
        """
        all_images = [self.images[index_slice][index_scan] for index_slice in range(self.n_slices) for index_scan in
                      range(self.n_scans)]

        return all_images

    def get_image_generator(self, repeat=False):
        """
        Yields image paths.

        :param repeat: If true, yields image paths indefinitely.
        :return: slice_image_path
        """
        while True:
            for index_slice in range(self.n_slices):
                for index_scan in range(self.n_scans):
                    yield self.images[index_slice][index_scan]
            if not repeat:
                break

    def get_image_pair(self, slice_index=None, scan_pair=None):
        """
        Returns image pair paths with time period in between

        :param slice_index: index of slice to be extracted
        :param scan_pair: indices of scans to be extracted
        :return: (scan1_slice, scan2_slice), days
        """
        if slice_index is None:
            slice_index = randint(0, self.n_slices - 1)
        if scan_pair is None:
            scan_pair = self.scan_pairs[randint(0, len(self.scan_pairs))]

        assert slice_index < self.n_slices
        assert scan_pair in self.scan_pairs

        return (self.images[slice_index][scan_pair[0]], self.images[slice_index][scan_pair[1]]), self.relative_dates[
            scan_pair[1]] - self.relative_dates[scan_pair[0]]

    def get_all_image_pairs(self):
        """
        Returns all image pairs as list

        :return:
        """
        all_image_pairs = []
        for slice_index in range(self.n_slices):
            for scan_pair in self.scan_pairs:
                all_image_pairs.append(self.get_image_pair(slice_index, scan_pair))

        return all_image_pairs

    def get_image_pair_generator(self, repeat=False):
        """
        Yields image pair. If repeat true, yields indefinitely, otherwise yields until last pair

        :param repeat: True or False
        :return:
        """
        while True:
            for slice_index in range(self.n_slices):
                for scan_pair in self.scan_pairs:
                    yield self.get_image_pair(slice_index, scan_pair)

            if not repeat:
                break

    def get_image_triplet(self, slice_index=None, scan_triplet=None):
        """
        Returns image triplet paths with time periods in between. Days are relative to first scan

        :param slice_index: index of slice to be extracted
        :param scan_pair: indices of scans to be extracted
        :return: (scan1_slice, scan2_slice, scan3_slice), (days1, days2)
        """
        if slice_index is None:
            slice_index = randint(0, self.n_slices - 1)
        if scan_triplet is None:
            scan_triplet = self.scan_triplets[randint(0, len(self.scan_triplets))]

        assert slice_index < self.n_slices
        assert scan_triplet in self.scan_triplets

        return (
                   self.images[slice_index][scan_triplet[0]],
                   self.images[slice_index][scan_triplet[1]],
                   self.images[slice_index][scan_triplet[2]]
               ), (self.relative_dates[scan_triplet[0]],
                   self.relative_dates[scan_triplet[1]],
                   self.relative_dates[scan_triplet[2]]
               )

    def get_all_image_triplets(self, slice_index=None):
        """
        Returns all image triplets as list

        :return:
        """
        all_image_triplets = []
        if slice_index is None:
            for slice_index in range(self.n_slices):
                for scan_triplet in self.scan_triplets:
                    all_image_triplets.append(self.get_image_triplet(slice_index, scan_triplet))
        else:
            for scan_triplet in self.scan_triplets:
                all_image_triplets.append(self.get_image_triplet(slice_index, scan_triplet))
        return all_image_triplets

    def get_image_triplet_generator(self, repeat=False):
        """
        Yields image triplet. If repeat true, yields indefinitely, otherwise yields until last triplet

        :param repeat: True or False
        :return:
        """
        while True:
            for slice_index in range(self.n_slices):
                for scan_triplet in self.scan_triplets:
                    yield self.get_image_triplet(slice_index, scan_triplet)

            if not repeat:
                break

    def get_longitudinal_sequence(self, slice_index=None):
        """
        Returns image triplet paths with time periods in between. Days are relative to first scan

        :param slice_index: index of slice to be extracted
        :param scan_pair: indices of scans to be extracted
        :return: (scan1_slice, scan2_slice, scan3_slice), (days1, days2)
        """
        if slice_index is None:
            slice_index = randint(0, self.n_slices - 1)

        assert slice_index < self.n_slices

        return self.images[slice_index], self.relative_dates

    def get_all_longitudinal_sequences(self):

        all_longitudinals = []
        for slice_index in range(self.n_slices):
            all_longitudinals.append(self.get_longitudinal_sequence(slice_index))
        return all_longitudinals


class LongitudinalDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        ad_patient_folder_paths = glob.glob(os.path.join(self.data_dir, 'ad_*'))
        mci_patient_folder_paths = glob.glob(os.path.join(self.data_dir, 'mci_*'))
        cn_patient_folder_paths = glob.glob(os.path.join(self.data_dir, 'cn_*'))

        self.ad_patients = [Patient(patient_folder_path=x, patient_type='ad') for x in ad_patient_folder_paths]
        self.mci_patients = [Patient(patient_folder_path=x, patient_type='ad') for x in mci_patient_folder_paths]
        self.cn_patients = [Patient(patient_folder_path=x, patient_type='ad') for x in cn_patient_folder_paths]

    def get_image(self, patient_type=None, slice_index=None, scan_index=None):
        if patient_type is None:
            patient_type = random.choice(['ad', 'mci', 'cn'])

        if patient_type == 'ad':
            return random.choice(self.ad_patients).get_image(slice_index=slice_index, scan_index=scan_index)
        elif patient_type == 'mci':
            return random.choice(self.mci_patients).get_image(slice_index=slice_index, scan_index=scan_index)
        elif patient_type == 'cn':
            return random.choice(self.cn_patients).get_image(slice_index=slice_index, scan_index=scan_index)

    def get_ad_images(self):
        all_images = []
        for patient in self.ad_patients:
            all_images += patient.get_all_images()
        return all_images

    def get_mci_images(self):
        all_images = []
        for patient in self.mci_patients:
            all_images += patient.get_all_images()
        return all_images

    def get_cn_images(self):
        all_images = []
        for patient in self.cn_patients:
            all_images += patient.get_all_images()
        return all_images

    def get_ad_image_pairs(self):
        all_image_pairs = []
        for patient in self.ad_patients:
            all_image_pairs = all_image_pairs + patient.get_all_image_pairs()
        return all_image_pairs

    def get_mci_image_pairs(self):
        all_image_pairs = []
        for patient in self.mci_patients:
            all_image_pairs += patient.get_all_image_pairs()
        return all_image_pairs

    def get_cn_image_pairs(self):
        all_image_pairs = []
        for patient in self.cn_patients:
            all_image_pairs += patient.get_all_image_pairs()
        return all_image_pairs

    def get_ad_image_triplets(self, slice_index=None):
        all_image_triplets = []
        for patient in self.ad_patients:
            all_image_triplets += patient.get_all_image_triplets(slice_index=slice_index)
        return all_image_triplets

    def get_mci_image_triplets(self, slice_index=None):
        all_image_triplets = []
        for patient in self.mci_patients:
            all_image_triplets += patient.get_all_image_triplets(slice_index=slice_index)
        return all_image_triplets

    def get_cn_image_triplets(self, slice_index=None):
        all_image_triplets = []
        for patient in self.cn_patients:
            all_image_triplets += patient.get_all_image_triplets(slice_index=slice_index)
        return all_image_triplets

    def get_ad_longitudinal_sequences(self):
        all_sequences = []
        for patient in self.ad_patients:
            all_sequences = all_sequences + patient.get_all_longitudinal_sequences()
        return all_sequences

    def get_mci_longitudinal_sequences(self):
        all_sequences = []
        for patient in self.mci_patients:
            all_sequences = all_sequences + patient.get_all_longitudinal_sequences()
        return all_sequences

    def get_cn_longitudinal_sequences(self):
        all_sequences = []
        for patient in self.cn_patients:
            all_sequences = all_sequences + patient.get_all_longitudinal_sequences()
        return all_sequences

    def get_all_longitudinal_sequences(self):
        return self.get_ad_longitudinal_sequences() + self.get_mci_longitudinal_sequences() + self.get_cn_longitudinal_sequences()


if __name__ == "__main__":
    data_dir = '/Users/umutkucukaslan/Desktop/thesis/dataset/processed_data/test'
    longitudinal_dataset = LongitudinalDataset(data_dir=data_dir)

    # print(longitudinal_dataset.get_ad_image_pairs())
    # ad_image_pairs = longitudinal_dataset.get_ad_image_triplets()
    # for pair in ad_image_pairs:
    #     print(pair)
    # print(len(ad_image_pairs))

    long_seq = longitudinal_dataset.get_mci_image_triplets(20)
    for seq in long_seq:
        print(seq)
    print(len(long_seq))