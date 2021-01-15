import os
import time

import cv2
import numpy as np
import tensorflow as tf
import csv
import sys

from datasets.adni_dataset import get_triplets_adni_15t_dataset

from model.ae.ae import AE

EXPERIMENT_NAME = "exp_2021_01_07_ae"
CHECKPOINT_DIR_NAME = "checkpoints"

# choose machine type
if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "none"
    EXPERIMENT_FOLDER = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/experiments", EXPERIMENT_NAME
    )
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
    EXPERIMENT_FOLDER = os.path.join(
        "/content/drive/My Drive/experiments", EXPERIMENT_NAME
    )
else:
    raise ValueError("Unknown machine type, no machine MACHINE")
CHECKPOINT_DIR = os.path.join(EXPERIMENT_FOLDER, CHECKPOINT_DIR_NAME)


FILTERS = [64, 128, 256, 512]
KERNEL_SIZE = 3
ACTIVATION = tf.nn.silu
LAST_ACTIVATION = tf.nn.sigmoid
STRUCTURE_VEC_SIZE = 100
LONGITUDINAL_VEC_SIZE = 1

BATCH_SIZE = 32
EPOCHS = 5000
CHECKPOINT_SAVE_INTERVAL = 5
MAX_TO_KEEP = 5
LR = 1e-4

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNEL = 1
STRUCTURE_VEC_SIMILARITY_LOSS_MULT = 100

# _, EXPERIMENT_FOLDER = get_model(return_experiment_folder=True)


model = AE(
    filters=FILTERS,
    kernel_size=KERNEL_SIZE,
    activation=ACTIVATION,
    last_activation=LAST_ACTIVATION,
    structure_vec_size=STRUCTURE_VEC_SIZE,
    longitudinal_vec_size=LONGITUDINAL_VEC_SIZE,
)
# model first call to initialize layers
input_tensor = tf.convert_to_tensor(
    np.zeros((BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL))
)
input_tensor = tf.cast(input_tensor, tf.float32)
_ = model(input_tensor)


print("restoring model")
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, "checkpoints")  # latest checkpoint
# checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, "best_checkpoint")
model.restore_model(checkpoint_dir)
print("model restored")
print("args:")
print(sys.argv)


check_interval = None
interval = None
if len(sys.argv) > 1:
    interval = sys.argv[1]
    if interval == "0":
        check_interval = [0, 450]
    elif interval == "1":
        check_interval = [450, 900]
    elif interval == "2":
        check_interval = [900, 1350]
    elif interval == "3":
        check_interval = [1350, 2000]

print("check interval is ", check_interval)

results_folder = os.path.join(
    EXPERIMENT_FOLDER, "testing/sequences/test_train_for_patient2"
)
if interval:
    results_folder = results_folder + f"_{interval}"

if not os.path.isdir(results_folder):
    os.makedirs(results_folder)


INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL = 64, 64, 1


train_ds, val_ds, test_ds = get_triplets_adni_15t_dataset(
    folder_name="training_data_15T_192x160_4slices",
    machine=MACHINE,
    target_shape=[INPUT_HEIGHT, INPUT_WIDTH],
    channels=INPUT_CHANNEL,
)

test_ds = test_ds.batch(1).prefetch(2)
train_ds = train_ds.shuffle(1000).batch(32).prefetch(2)
val_ds = val_ds.batch(32).prefetch(2)


class CSVHandler:
    def __init__(self, csv_path, columns, read_existing=False):
        self.csv_path = csv_path
        self.columns = columns
        self.rows = None
        if os.path.isfile(csv_path):
            if read_existing:
                self.rows = []
                with open(self.csv_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            line_count += 1
                        else:
                            self.rows.append(row)
        else:
            with open(self.csv_path, mode="w") as file:
                csv_writer = csv.writer(
                    file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(columns)

    def add_data(self, d):
        """
        add data dict to csv file
        :param d: {sample_id, identifier, train_step, ssim}
        :return:
        """
        row = [d[c] for c in self.columns]
        # row = [d["sample_id"], d["identifier"], d["train_step"], d["ssim"]]
        with open(self.csv_path, mode="a") as file:
            csv_writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(row)


class Saver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.progress_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_progress.csv"),
            ["sample_id"],
            read_existing=True,
        )
        self.interpolation_ssim_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_interpolation_ssims.csv"),
            ["sample_id", "identifier", "train_step", "ssim"],
        )
        self.pair_loss_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_pair_loss.csv"),
            [
                "sample_id",
                "identifier",
                "train_step",
                "total_loss",
                "image_similarity_loss",
                "structure_vec_sim_loss",
                "ssim",
            ],
        )
        self.val_loss_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_val_loss.csv"),
            [
                "sample_id",
                "identifier",
                "train_step",
                "total_loss",
                "image_similarity_loss",
                "structure_vec_sim_loss",
                "ssim",
            ],
        )
        self.train_loss_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_train_loss.csv"),
            [
                "sample_id",
                "identifier",
                "train_step",
                "total_loss",
                "image_similarity_loss",
                "structure_vec_sim_loss",
                "ssim",
            ],
        )
        self.train_and_pair_loss_csv_handler = CSVHandler(
            os.path.join(save_dir, "csv_train_and_pair_loss.csv"),
            [
                "sample_id",
                "identifier",
                "train_step",
                "total_loss",
                "image_similarity_loss",
                "structure_vec_sim_loss",
                "ssim",
                "image_similarity_loss_pair",
                "structure_vec_sim_loss_pair",
                "ssim_pair",
            ],
        )

    def save_interpolations_and_ssim(
        self,
        sample_id: int,
        identifier: str,
        train_step: int,
        interpolations: np.ndarray,
        interpolation_ssim: float,
    ) -> None:
        image_name = f"sample_{int(sample_id):05d}_{str(identifier)}_step_{int(train_step):05d}.jpg"
        image_path = os.path.join(self.save_dir, image_name)
        cv2.imwrite(image_path, interpolations)
        self.interpolation_ssim_csv_handler.add_data(
            {
                "sample_id": sample_id,
                "identifier": identifier,
                "train_step": train_step,
                "ssim": interpolation_ssim,
            }
        )

    def save_true_sequence(self, sample_id: int, image_sequence: np.ndarray) -> None:
        image_name = f"sample_{int(sample_id):05d}_true.jpg"
        image_path = os.path.join(self.save_dir, image_name)
        cv2.imwrite(image_path, image_sequence)

    def save_pair_losses(self, sample_id, identifier, train_step, pair_losses):
        self.pair_loss_csv_handler.add_data(
            {
                "sample_id": sample_id,
                "identifier": identifier,
                "train_step": train_step,
                "total_loss": pair_losses["total_loss"],
                "image_similarity_loss": pair_losses["image_similarity_loss"],
                "structure_vec_sim_loss": pair_losses["structure_vec_sim_loss"],
                "ssim": pair_losses["ssim"],
            }
        )

    def save_val_losses(self, sample_id, identifier, train_step, val_losses):
        self.val_loss_csv_handler.add_data(
            {
                "sample_id": sample_id,
                "identifier": identifier,
                "train_step": train_step,
                "total_loss": val_losses["total_loss"],
                "image_similarity_loss": val_losses["image_similarity_loss"],
                "structure_vec_sim_loss": val_losses["structure_vec_sim_loss"],
                "ssim": val_losses["ssim"],
            }
        )

    def save_train_losses(self, sample_id, identifier, train_step, train_losses):
        self.train_loss_csv_handler.add_data(
            {
                "sample_id": sample_id,
                "identifier": identifier,
                "train_step": train_step,
                "total_loss": train_losses["total_loss"],
                "image_similarity_loss": train_losses["image_similarity_loss"],
                "structure_vec_sim_loss": train_losses["structure_vec_sim_loss"],
                "ssim": train_losses["ssim"],
            }
        )

    def save_train_and_pair_losses(
        self, sample_id, identifier, train_step, train_and_pair_losses
    ):
        self.train_and_pair_loss_csv_handler.add_data(
            {
                "sample_id": sample_id,
                "identifier": identifier,
                "train_step": train_step,
                "total_loss": train_and_pair_losses["total_loss"],
                "image_similarity_loss": train_and_pair_losses["image_similarity_loss"],
                "structure_vec_sim_loss": train_and_pair_losses[
                    "structure_vec_sim_loss"
                ],
                "ssim": train_and_pair_losses["ssim"],
                "image_similarity_loss_pair": train_and_pair_losses[
                    "image_similarity_loss_pair"
                ],
                "structure_vec_sim_loss_pair": train_and_pair_losses[
                    "structure_vec_sim_loss_pair"
                ],
                "ssim_pair": train_and_pair_losses["ssim_pair"],
            }
        )


saver = Saver(save_dir=results_folder)


for sample_id, sample in enumerate(test_ds):
    if check_interval:
        if sample_id < check_interval[0] or sample_id >= check_interval[1]:
            continue
    # check if tried this case before
    if saver.progress_csv_handler.rows:
        matches = [sample_id == int(x[0]) for x in saver.progress_csv_handler.rows]
        if any(matches):
            print(f"sample id {sample_id} tested before")
            continue
    start_time = time.time()
    # if sample_id == 1:
    #     exit()
    imgs = sample["imgs"]
    days = sample["days"]

    days = [x.numpy()[0] for x in days]

    # extrapolate days
    days += [days[-1] + i * 90 for i in range(1, 13)]
    months = [int(round(d / 30.0)) for d in days]
    print(f"sample id: {sample_id}")
    print("months: ", months)

    # true seq
    true_sequence = [
        np.clip(x.numpy()[0, ...] * 255, 0, 255).astype(np.uint8) for x in imgs
    ]
    true_sequence = [
        cv2.putText(x, str(int(d)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        for x, d in zip(true_sequence, months[:3])
    ]
    true_sequence = np.hstack(true_sequence)
    saver.save_true_sequence(sample_id=sample_id, image_sequence=true_sequence)

    # =======================================================================================
    # future slice extrapolation
    sample_points = [(x - days[0]) / (days[1] - days[0]) for x in days]

    def extrapolation_future_callback_fn(step):
        future_interpolation_image, ssim = model.interpolate_and_calculate_ssim(
            imgs[0],
            imgs[1],
            sample_points,
            imgs[2],
            ground_truth_index=2,
            dates=months,
            structure_mix_type="mean",
        )
        saver.save_interpolations_and_ssim(
            sample_id=sample_id,
            identifier="f",
            train_step=step,
            interpolations=future_interpolation_image,
            interpolation_ssim=ssim,
        )

    def callback_fn_save_pair_losses(step, pair_losses):
        saver.save_pair_losses(
            sample_id, identifier="f", train_step=step, pair_losses=pair_losses
        )

    def callback_fn_save_val_losses(step, val_losses):
        saver.save_val_losses(
            sample_id, identifier="f", train_step=step, val_losses=val_losses
        )

    def callback_fn_save_train_losses(step, train_losses):
        saver.save_train_losses(
            sample_id, identifier="f", train_step=step, train_losses=train_losses
        )

    def callback_fn_save_train_and_pair_losses(step, train_and_pair_losses):
        saver.save_train_and_pair_losses(
            sample_id,
            identifier="f",
            train_step=step,
            train_and_pair_losses=train_and_pair_losses,
        )

    model.restore_model(checkpoint_dir)
    print(f"sample id: {sample_id} - {1}/3 - f")
    # model.train_for_patient(
    #     imgs[0],
    #     imgs[1],
    #     train_ds,
    #     val_ds,
    #     num_steps=1000,
    #     period=10,
    #     lr=1e-4,
    #     callback_fn_generate_seq=extrapolation_future_callback_fn,
    #     callback_fn_save_pair_losses=callback_fn_save_pair_losses,
    #     callback_fn_save_val_losses=callback_fn_save_val_losses,
    #     callback_fn_save_train_losses=callback_fn_save_train_losses,
    # )
    model.train_for_patient2(
        imgs[0],
        imgs[1],
        train_ds,
        val_ds,
        num_steps=120,
        period=10,
        lr=1e-4,
        callback_fn_generate_seq=extrapolation_future_callback_fn,
        callback_fn_save_val_losses=callback_fn_save_val_losses,
        callback_fn_save_train_and_pair_losses=callback_fn_save_train_and_pair_losses,
    )

    # =======================================================================================
    # missing slice interpolation
    sample_points = [(x - days[0]) / (days[2] - days[0]) for x in days]

    def interpolation_missing_callback_fn(step):
        missing_interpolation_image, ssim = model.interpolate_and_calculate_ssim(
            imgs[0],
            imgs[2],
            sample_points,
            imgs[1],
            ground_truth_index=1,
            dates=months,
            structure_mix_type="mean",
        )
        saver.save_interpolations_and_ssim(
            sample_id=sample_id,
            identifier="m",
            train_step=step,
            interpolations=missing_interpolation_image,
            interpolation_ssim=ssim,
        )

    def callback_fn_save_pair_losses(step, pair_losses):
        saver.save_pair_losses(
            sample_id, identifier="m", train_step=step, pair_losses=pair_losses
        )

    def callback_fn_save_val_losses(step, val_losses):
        saver.save_val_losses(
            sample_id, identifier="m", train_step=step, val_losses=val_losses
        )

    def callback_fn_save_train_losses(step, train_losses):
        saver.save_train_losses(
            sample_id, identifier="m", train_step=step, train_losses=train_losses
        )

    def callback_fn_save_train_and_pair_losses(step, train_and_pair_losses):
        saver.save_train_and_pair_losses(
            sample_id,
            identifier="m",
            train_step=step,
            train_and_pair_losses=train_and_pair_losses,
        )

    model.restore_model(checkpoint_dir)
    print(f"sample id: {sample_id} - {2}/3 - m")
    # model.train_for_patient(
    #     imgs[0],
    #     imgs[2],
    #     train_ds,
    #     val_ds,
    #     num_steps=1000,
    #     period=10,
    #     lr=1e-4,
    #     callback_fn_generate_seq=interpolation_missing_callback_fn,
    #     callback_fn_save_pair_losses=callback_fn_save_pair_losses,
    #     callback_fn_save_val_losses=callback_fn_save_val_losses,
    #     callback_fn_save_train_losses=callback_fn_save_train_losses,
    # )
    model.train_for_patient2(
        imgs[0],
        imgs[2],
        train_ds,
        val_ds,
        num_steps=120,
        period=10,
        lr=1e-4,
        callback_fn_generate_seq=interpolation_missing_callback_fn,
        callback_fn_save_val_losses=callback_fn_save_val_losses,
        callback_fn_save_train_and_pair_losses=callback_fn_save_train_and_pair_losses,
    )

    # =======================================================================================
    # previous slice extrapolation
    sample_points = [(x - days[1]) / (days[2] - days[1]) for x in days]

    def extrapolation_previous_callback_fn(step):
        future_interpolation_image, ssim = model.interpolate_and_calculate_ssim(
            imgs[1],
            imgs[2],
            sample_points,
            imgs[0],
            ground_truth_index=0,
            dates=months,
            structure_mix_type="mean",
        )
        saver.save_interpolations_and_ssim(
            sample_id=sample_id,
            identifier="p",
            train_step=step,
            interpolations=future_interpolation_image,
            interpolation_ssim=ssim,
        )

    def callback_fn_save_pair_losses(step, pair_losses):
        saver.save_pair_losses(
            sample_id, identifier="p", train_step=step, pair_losses=pair_losses
        )

    def callback_fn_save_val_losses(step, val_losses):
        saver.save_val_losses(
            sample_id, identifier="p", train_step=step, val_losses=val_losses
        )

    def callback_fn_save_train_losses(step, train_losses):
        saver.save_train_losses(
            sample_id, identifier="p", train_step=step, train_losses=train_losses
        )

    def callback_fn_save_train_and_pair_losses(step, train_and_pair_losses):
        saver.save_train_and_pair_losses(
            sample_id,
            identifier="p",
            train_step=step,
            train_and_pair_losses=train_and_pair_losses,
        )

    model.restore_model(checkpoint_dir)
    print(f"sample id: {sample_id} - {3}/3 - p")
    # model.train_for_patient(
    #     imgs[1],
    #     imgs[2],
    #     train_ds,
    #     val_ds,
    #     num_steps=1000,
    #     period=10,
    #     lr=1e-4,
    #     callback_fn_generate_seq=extrapolation_previous_callback_fn,
    #     callback_fn_save_pair_losses=callback_fn_save_pair_losses,
    #     callback_fn_save_val_losses=callback_fn_save_val_losses,
    #     callback_fn_save_train_losses=callback_fn_save_train_losses,
    # )
    model.train_for_patient2(
        imgs[1],
        imgs[2],
        train_ds,
        val_ds,
        num_steps=120,
        period=10,
        lr=1e-4,
        callback_fn_generate_seq=extrapolation_previous_callback_fn,
        callback_fn_save_val_losses=callback_fn_save_val_losses,
        callback_fn_save_train_and_pair_losses=callback_fn_save_train_and_pair_losses,
    )

    end_time = time.time()
    print(f"sample id: {sample_id} took {round(end_time - start_time)} seconds")
    saver.progress_csv_handler.add_data({"sample_id": sample_id})
