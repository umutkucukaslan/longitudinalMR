import os
import cv2
import numpy as np

from datasets.adni_dataset import get_triplets_adni_15t_dataset
from experiments.exp_2021_01_07_ae import get_model


model, EXPERIMENT_FOLDER = get_model(return_experiment_folder=True)

results_folder = os.path.join(EXPERIMENT_FOLDER, "testing/sequences/val")
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)


INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = 64, 64, 1
# choose machine type
if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "none"
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
else:
    raise ValueError("Unknown machine type, no machine MACHINE")

train_ds, val_ds, test_ds = get_triplets_adni_15t_dataset(
    folder_name="training_data_15T_192x160_4slices",
    machine=MACHINE,
    target_shape=[INPUT_HEIGHT, INPUT_WIDTH],
    channels=INPUT_CHANNEL,
)

test_ds = test_ds.batch(1).prefetch(2)
train_ds = train_ds.batch(1).prefetch(2)
val_ds = val_ds.batch(1).prefetch(2)

# for n, sample in enumerate(train_ds):
# for n, sample in enumerate(test_ds):
for n, sample in enumerate(val_ds):
    imgs = sample["imgs"]
    days = sample["days"]

    days = [x.numpy()[0] for x in days]

    # true seq
    true_sequence = [
        np.clip(x.numpy()[0, ...] * 255, 0, 255).astype(np.uint8) for x in imgs
    ]
    true_sequence = [
        cv2.putText(x, str(int(d)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        for x, d in zip(true_sequence, days)
    ]
    true_sequence = np.hstack(true_sequence)

    # extrapolate days
    days += [days[-1] + i * (days[-1] - days[-2]) for i in range(1, 13)]
    # missing prediction
    sample_points = [(x - days[0]) / days[2] for x in days]
    missing_interpolation_image = model.interpolate(
        inputs1=imgs[0],
        inputs2=imgs[2],
        sample_points=sample_points,
        structure_mix_type="mean",
        return_as_image=True,
    )
    # future prediction
    sample_points = [(x - days[0]) / days[1] for x in days]
    future_interpolation_image = model.interpolate(
        inputs1=imgs[0],
        inputs2=imgs[1],
        sample_points=sample_points,
        structure_mix_type="mean",
        return_as_image=True,
    )

    cv2.putText(true_sequence, "T", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255)
    true_sequence = np.hstack(
        [
            true_sequence,
            true_sequence * 0,
            true_sequence * 0,
            true_sequence * 0,
            true_sequence * 0,
        ]
    )
    cv2.putText(
        missing_interpolation_image, "M", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255
    )
    cv2.putText(
        future_interpolation_image, "F", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255
    )
    image = np.vstack(
        [true_sequence, missing_interpolation_image, future_interpolation_image]
    )
    image_path = os.path.join(results_folder, f"sample_{n:05d}.jpg")
    cv2.imwrite(image_path, image)

    print(days)
