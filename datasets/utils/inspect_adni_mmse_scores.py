import math

import pandas as pd
import numpy as np

csv_file = '/Users/umutkucukaslan/Desktop/thesis/dataset/ADNI_cognitive_scores/ADNIMERGE.csv'


df = pd.read_csv(csv_file)

patients = {'AD': {}, 'CN': {}, 'LMCI': {}, 'SMC': {} }
for index, row in df.iterrows():
    patient_name = row['PTID']
    viscode = row['VISCODE']
    mmse = row['MMSE']
    DX_bl = row['DX_bl']

    if mmse is None or math.isnan(mmse):
        continue

    if not(viscode in ['bl', 'm06', 'm12', 'm18', 'm24', 'm36']):
        continue

    if not DX_bl in patients.keys():
        continue

    if patient_name in patients[DX_bl]:
        patients[DX_bl][patient_name][viscode] = mmse
    else:
        patients[DX_bl][patient_name] = {viscode: mmse}

print(patients)
print(len(patients))

def is_monotonic(arr):
    return np.all(np.diff(arr) <= 0), np.all(np.diff(arr) >= 0)

for group in patients:
    patients[group]['monotonically_decreasing'] = 0
    patients[group]['monotonically_increasing'] = 0
    patients[group]['non_monotonic'] = 0


for group in patients:
    for key in patients[group].keys():
        if key == 'monotonically_decreasing' or key == 'monotonically_increasing' or key == 'non_monotonic':
            continue
        data = patients[group][key]
        keys = sorted(data.keys())
        arr = [data[k] for k in keys]
        m_d, m_i = is_monotonic(arr)
        patients[group][key] = arr
        if m_d:
            patients[group]['monotonically_decreasing'] = patients[group]['monotonically_decreasing'] + 1
        elif m_i:
            patients[group]['monotonically_increasing'] = patients[group]['monotonically_increasing'] + 1
        else:
            patients[group]['non_monotonic'] = patients[group]['non_monotonic'] + 1

print(patients)




for group in patients:
    print(group)
    print('    monotonically increasing: {}'.format(patients[group]['monotonically_increasing']))
    print('    monotonically decreasing: {}'.format(patients[group]['monotonically_decreasing']))
    print('    non monotonic           : {}'.format(patients[group]['non_monotonic']))


