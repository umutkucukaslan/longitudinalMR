
# Preprocessing Steps
Preprocess MRI images using steps below in this order.
- Reorganize files: This removes unwanted midway folders and
puts them in an hierarchy such as 
    ```
    data
        patient1
            scan1
                mri_file
            scan2
                mri_file
            scan3
                mri_file
        patient2
            ...
    ```
    Use `reorganize_files.py`

- Register MRIs to baseline image: This step registers each 
future scan to the baseline scan of a patient for each patient.
Use `register_mris_to_baseline_image.py`

- Resample MRI images to 1mm x 1mm x 1mm voxel size.
Use `resample_images.py`


- Create slice images: This extracts axial slices from MRIs and also 
groups them as AD (Alzheimer), MCI (mild congitive impairment) and 
CN (cognitively normal)
Use `create_slice_images.py`
