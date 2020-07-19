import os, glob, csv


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


all_lists = [
    high_change_15T,
    middle_changes_15T,
    small_change_15T,
    very_small_change_15T,
]
for i in range(len(all_lists)):
    list1 = all_lists[i]
    for j in range(i + 1, len(all_lists)):
        list2 = all_lists[j]
        for p in list1:
            if p in list2:
                print("Patient {} from list {} is also in list {}".format(p, i, j))
            if p in faulty_15T:
                print("Patient {} from list {} has a faulty registration.".format(p, i))

count = 0
for l in all_lists:
    count += len(l)
print("Number of patients {}".format(count))

# count = 0
# for p in very_small_15T:
#     print(p)
#     count += 1
# print('n = {}'.format(count))
