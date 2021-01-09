import copy
import datetime
import os
import statistics
import sys
import time
from tqdm import tqdm

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasets.adni_dataset import (
    get_triplets_adni_15t_dataset,
    get_images_adni_15t_dataset,
)
from model.ae.ae import AE
from model.glow.model import Glow

"""
autoencoder

encodes images to structure vector and state vector
structure vector should be the same for all images in the seqs
state vector is used for interpolation in time

filters: 64, 128, 256, 512
image shape: (64, 64, 1)

"""
FILTERS = [64, 128, 256, 512]
KERNEL_SIZE = 3
ACTIVATION = tf.nn.silu
LAST_ACTIVATION = tf.nn.sigmoid
STRUCTURE_VEC_SIZE = 100
LONGITUDINAL_VEC_SIZE = 1

TEMPERATURE = 0.7
NUM_SAMPLE_IMAGES = 20

RESTORE_FROM_CHECKPOINT = True

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNEL = 1
STRUCTURE_VEC_SIMILARITY_LOSS_MULT = 100


BATCH_SIZE = 32
EPOCHS = 5000
CHECKPOINT_SAVE_INTERVAL = 5
MAX_TO_KEEP = 5
LR = 1e-4


EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# choose machine type
if __file__.startswith("/Users/umutkucukaslan/Desktop/thesis"):
    MACHINE = "none"
elif __file__.startswith("/content/thesis"):
    MACHINE = "colab"
else:
    raise ValueError("Unknown machine type, no machine MACHINE")

# set batch size easily
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])

# set memory growth to true
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if MACHINE == "colab":
    EXPERIMENT_FOLDER = os.path.join(
        "/content/drive/My Drive/experiments", EXPERIMENT_NAME
    )
elif MACHINE == "cloud":
    EXPERIMENT_FOLDER = os.path.join(
        "/home/umutkucukaslan/experiments", EXPERIMENT_NAME
    )
else:
    EXPERIMENT_FOLDER = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/experiments", EXPERIMENT_NAME
    )

# create experiment folder
if __name__ == "__main__":
    if not os.path.isdir(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)

    # folder to save generated test images during training
    if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, "figures")):
        os.makedirs(os.path.join(EXPERIMENT_FOLDER, "figures"))


def log_print(msg, add_timestamp=False):
    if not isinstance(msg, str):
        msg = str(msg)
    if add_timestamp:
        msg += " (logged at {})".format(
            datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    with open(os.path.join(EXPERIMENT_FOLDER, "logs.txt"), "a+") as log_file:
        log_file.write(msg + "\n")


# model
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

# optimizers
optimizer = tf.optimizers.Adam(LR, beta_1=0.5)
# optimizer = tf.optimizers.RMSprop(learning_rate=LR)

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, "checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    epoch=tf.Variable(0), model=model, optimizer=optimizer,
)
manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP
)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    input_tensor = tf.convert_to_tensor(
        np.random.rand(1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL), dtype=tf.float32
    )
    _ = model(input_tensor)
    log_print("Restored from {}".format(manager.latest_checkpoint))
    initialized_from_scratch = False
else:
    log_print("Initializing from scratch.")
    initialized_from_scratch = True

initial_epoch = checkpoint.epoch.numpy() + 1


def get_model(return_experiment_folder=True):
    """
    Returns the restored model
    :return: model, experiment_folder (optional)
    """
    if return_experiment_folder:
        return model, EXPERIMENT_FOLDER
    return model


if __name__ == "__main__":
    # summary file writer for tensorboard
    log_dir = os.path.join(EXPERIMENT_FOLDER, "logs")
    summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    # data set
    train_ds, val_ds, _ = get_triplets_adni_15t_dataset(
        folder_name="training_data_15T_192x160_4slices",
        machine=MACHINE,
        target_shape=[INPUT_HEIGHT, INPUT_WIDTH],
        channels=INPUT_CHANNEL,
    )
    train_ds = (
        train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER_SIZE)
    )
    val_ds = (
        val_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER_SIZE)
    )

    def calculate_ssim(imgs, generated_imgs):
        ssims = [
            tf.image.ssim(img1, img2, max_val=1.0)
            for img1, img2 in zip(imgs, generated_imgs)
        ]
        return tf.reduce_mean(ssims)

    def get_predicted_states(states, days):
        while tf.rank(days[0]) < tf.rank(states[0]):
            days = [tf.expand_dims(x, axis=-1) for x in days]
        past = states[1] + (days[0] - days[2]) / (days[1] - days[2]) * (
            states[1] - states[2]
        )
        missing = states[0] + (days[1] - days[0]) / (days[2] - days[0]) * (
            states[2] - states[0]
        )
        future = states[0] + (days[2] - days[0]) / (days[1] - days[0]) * (
            states[1] - states[0]
        )
        return [past, missing, future]

    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(imgs, days):
        predicted_imgs_for_vis = None
        with tf.GradientTape() as tape:
            structures = []
            states = []
            for image_batch in imgs:
                structure, state = model.encode(image_batch, training=True)
                structures.append(structure)
                states.append(state)
            structure_sim_mse = [
                mse_loss_fn(structures[0], structures[1]),
                mse_loss_fn(structures[0], structures[2]),
                mse_loss_fn(structures[1], structures[2]),
            ]
            structure_vec_sim_loss = tf.reduce_mean(
                structure_sim_mse
            )  # sum(structure_sim_mse) / 3.0

            predicted_states = get_predicted_states(states, days)
            predicted_imgs = [
                model.decode(structure, state)
                for structure, state in zip(structures, predicted_states)
            ]
            if predicted_imgs_for_vis is None:
                predicted_imgs_for_vis = predicted_imgs
            image_similarity_mse = [
                mse_loss_fn(real, pred) for real, pred in zip(imgs, predicted_imgs)
            ]
            image_similarity_loss = tf.reduce_mean(
                image_similarity_mse
            )  # sum(image_similarity_mse) / 3.0
            total_loss = (
                STRUCTURE_VEC_SIMILARITY_LOSS_MULT * structure_vec_sim_loss
                + image_similarity_loss
            )

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        ssims = calculate_ssim(imgs, predicted_imgs)

        return (
            total_loss,
            image_similarity_loss,
            structure_vec_sim_loss,
            ssims,
            predicted_imgs_for_vis,
        )

    def eval_step(imgs, days):
        predicted_imgs_for_vis = None
        structures = []
        states = []
        for image_batch in imgs:
            structure, state = model.encode(image_batch, training=False)
            structures.append(structure)
            states.append(state)
        structure_sim_mse = [
            mse_loss_fn(structures[0], structures[1]),
            mse_loss_fn(structures[0], structures[2]),
            mse_loss_fn(structures[1], structures[2]),
        ]
        structure_vec_sim_loss = tf.reduce_mean(
            structure_sim_mse
        )  # sum(structure_sim_mse) / 3.0

        predicted_states = get_predicted_states(states, days)
        predicted_imgs = [
            model.decode(structure, state)
            for structure, state in zip(structures, predicted_states)
        ]
        if predicted_imgs_for_vis is None:
            predicted_imgs_for_vis = predicted_imgs
        image_similarity_mse = [
            mse_loss_fn(real, pred) for real, pred in zip(imgs, predicted_imgs)
        ]
        image_similarity_loss = tf.reduce_mean(
            image_similarity_mse
        )  # sum(image_similarity_mse) / 3.0
        total_loss = (
            STRUCTURE_VEC_SIMILARITY_LOSS_MULT * structure_vec_sim_loss
            + image_similarity_loss
        )

        ssims = calculate_ssim(imgs, predicted_imgs)

        return (
            total_loss,
            image_similarity_loss,
            structure_vec_sim_loss,
            ssims,
            predicted_imgs_for_vis,
        )

    def generate_images(predicted_imgs, image_name):
        path = os.path.join(EXPERIMENT_FOLDER, "figures", image_name)
        batch_size = tf.shape(predicted_imgs[0])[0]
        hseq = []
        for imgs in predicted_imgs:
            imgs = imgs.numpy()
            vseq = [imgs[i, ...] for i in range(batch_size)]
            vseq = np.vstack(vseq)
            vseq = np.clip(vseq * 255, 0, 255).astype(np.uint8)
            hseq.append(vseq)
        hseq = np.hstack(hseq)
        cv2.imwrite(path, hseq)

    def fit(train_ds, val_ds, num_epochs, initial_epoch=0):
        assert initial_epoch < num_epochs
        for epoch in range(initial_epoch, num_epochs):
            print("Epoch: {}".format(epoch))
            start_time = time.time()
            image_name_train = str(epoch) + "_train.png"
            image_name_val = str(epoch) + "_val.png"

            # training
            log_print("Training epoch {}".format(epoch), add_timestamp=True)
            losses = [[], [], [], []]
            # with tqdm() as pbar:
            pbar = tqdm()
            for n, inputs in train_ds.enumerate():
                imgs = inputs["imgs"]
                days = inputs["days"]
                (
                    total_loss,
                    image_similarity_loss,
                    structure_vec_sim_loss,
                    ssims,
                    predicted_imgs,
                ) = train_step(imgs, days)

                losses[0].append(total_loss.numpy())
                losses[1].append(image_similarity_loss.numpy())
                losses[2].append(structure_vec_sim_loss.numpy())
                losses[3].append(ssims.numpy())
                pbar.update(1)
                pbar.set_description(
                    f"training..... Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                    + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
                )
            generate_images(predicted_imgs, image_name_train)
            losses = [statistics.mean(x) for x in losses]
            pbar.set_description(
                f"training..... Total loss: {losses[0]:.5f}; image_sim_mse: {losses[1]:.5f}; "
                + f"structure_vec_mse: {losses[2]:.5f}; ssim: {losses[3]:.5f}"
            )
            pbar.close()
            with summary_writer.as_default():
                tf.summary.scalar("total_loss", losses[0], step=epoch)
                tf.summary.scalar("image_similarity_loss", losses[1], step=epoch)
                tf.summary.scalar("structure_vec_sim_loss", losses[2], step=epoch)
                tf.summary.scalar("ssims", losses[3], step=epoch)
            summary_writer.flush()

            # testing
            log_print("Calculating validation losses...")
            val_losses = [[], [], [], []]
            # with tqdm() as pbar:
            pbar = tqdm()
            for n, inputs in val_ds.enumerate():
                imgs = inputs["imgs"]
                days = inputs["days"]
                (
                    total_loss,
                    image_similarity_loss,
                    structure_vec_sim_loss,
                    ssims,
                    predicted_imgs,
                ) = eval_step(imgs, days)
                val_losses[0].append(total_loss.numpy())
                val_losses[1].append(image_similarity_loss.numpy())
                val_losses[2].append(structure_vec_sim_loss.numpy())
                val_losses[3].append(ssims.numpy())
                pbar.update(1)
                pbar.set_description(
                    f"validations.. Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                    + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
                )
            generate_images(predicted_imgs, image_name_val)
            val_losses = [statistics.mean(x) for x in val_losses]
            pbar.set_description(
                f"validations.. Total loss: {val_losses[0]:.5f}; image_sim_mse: {val_losses[1]:.5f}; "
                + f"structure_vec_mse: {val_losses[2]:.5f}; ssim: {val_losses[3]:.5f}"
            )
            pbar.close()
            with summary_writer.as_default():
                tf.summary.scalar("val_total_loss", val_losses[0], step=epoch)
                tf.summary.scalar(
                    "val_image_similarity_loss", val_losses[1], step=epoch
                )
                tf.summary.scalar(
                    "val_structure_vec_sim_loss", val_losses[2], step=epoch
                )
                tf.summary.scalar("val_ssims", val_losses[3], step=epoch)
            summary_writer.flush()
            # print(
            #     f"[TRAIN] Total loss: {losses[0]:.5f}; image_sim_mse: {losses[1]:.5f}; "
            #     + f"structure_vec_mse: {losses[2]:.5f}; ssim: {losses[3]:.5f}"
            # )
            # print(
            #     f"[VAL] Total loss: {val_losses[0]:.5f}; image_sim_mse: {val_losses[1]:.5f}; "
            #     + f"structure_vec_mse: {val_losses[2]:.5f}; ssim: {val_losses[3]:.5f}"
            # )

            end_time = time.time()
            log_print(
                f"Epoch {epoch} completed in {round(end_time - start_time)} seconds."
                + f"[TRAIN] Total loss: {losses[0]:.5f}; image_sim_mse: {losses[1]:.2f}; "
                + f"structure_vec_mse: {losses[2]:.2f}; ssim: {losses[3]:.3f}"
                + f"[VAL] Total loss: {val_losses[0]:.5f}; image_sim_mse: {val_losses[1]:.2f}; "
                + f"structure_vec_mse: {val_losses[2]:.2f}; ssim: {val_losses[3]:.3f}"
            )

            checkpoint.epoch.assign(epoch)
            if int(checkpoint.epoch) % CHECKPOINT_SAVE_INTERVAL == 0:
                save_path = manager.save()
                log_print(
                    "Saved checkpoint for epoch {}: {}".format(
                        int(checkpoint.epoch), save_path
                    )
                )

    try:
        log_print("Fitting to the data set", add_timestamp=True)
        log_print(" ")
        log_print("Parameters:")
        log_print("Experiment name: " + str(EXPERIMENT_NAME))
        log_print("Batch size: " + str(BATCH_SIZE))
        log_print("Epochs: " + str(EPOCHS))
        log_print("Restore from checkpoint: " + str(RESTORE_FROM_CHECKPOINT))
        log_print("Chechpoint save interval: " + str(CHECKPOINT_SAVE_INTERVAL))
        log_print("Max number of checkpoints kept: " + str(MAX_TO_KEEP))
        log_print("MACHINE: " + str(MACHINE))
        log_print("Prefetch buffer size: " + str(PREFETCH_BUFFER_SIZE))
        log_print("Shuffle buffer size: " + str(SHUFFLE_BUFFER_SIZE))
        log_print(
            "Input shape: ( "
            + str(INPUT_HEIGHT)
            + ", "
            + str(INPUT_WIDTH)
            + ", "
            + str(INPUT_CHANNEL)
            + " )"
        )
        log_print(" ")
        log_print("Initial epoch: {}".format(initial_epoch))

        fit(
            train_ds, val_ds, num_epochs=EPOCHS, initial_epoch=initial_epoch,
        )
        # fit(
        #     train_ds.take(5),
        #     val_ds.take(2),
        #     num_epochs=EPOCHS,
        #     initial_epoch=initial_epoch,
        # )

        # save last checkpoint
        save_path = manager.save()
        log_print(
            "Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path)
        )
        summary_writer.close()
    except KeyboardInterrupt:
        log_print("Keyboard Interrupt", add_timestamp=True)
        # save latest checkpoint and close log file
        save_path = manager.save()
        log_print(
            "Saved checkpoint for epoch {}: {} due to KeyboardInterrupt".format(
                int(checkpoint.epoch), save_path
            )
        )
        summary_writer.close()
    except:
        summary_writer.close()
