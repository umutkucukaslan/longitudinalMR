import copy
import datetime
import os
import statistics
import sys
import time

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasets.adni_dataset import (
    get_triplets_adni_15t_dataset,
    get_images_adni_15t_dataset,
)
from model.glow.model import Glow

"""
GLOW model implemented in TF2.
input range: -0.5 to 0.5

num_flows = 32
num_blocks = 4
split = True
use_lu_decom = True
affine = True

n_bits = 5
lr = 1e-4
temp = 0.7

image shape: (64, 64, 1)
hidden 1x1 conv filters = 512
other conv kernel size = 3


"""
NUM_FLOWS = 32
NUM_BLOCKS = 4
SPLIT = True
USE_LU_DECOM = True
AFFINE = True

NUM_FILTERS = 512  # affine coupling layer's NN's 1x1 conv filter size

N_BITS = 5
TEMPERATURE = 0.7
NUM_SAMPLE_IMAGES = 20

RESTORE_FROM_CHECKPOINT = True

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNEL = 1

BATCH_SIZE = 5
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
model = Glow(
    in_channels=INPUT_CHANNEL,
    num_blocks=NUM_BLOCKS,
    num_flows=NUM_FLOWS,
    num_filters=NUM_FILTERS,
    use_lu_decom=USE_LU_DECOM,
    affine=AFFINE,
    split=SPLIT,
)

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
    train_ds, val_ds, test_ds = get_images_adni_15t_dataset(
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
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

    def calculate_ssim(imgs, generated_imgs):
        ssims = [
            tf.image.ssim(img1, img2, max_val=1.0)
            for img1, img2 in zip(imgs, generated_imgs)
        ]
        return tf.reduce_mean([ssims[0], ssims[2]]), tf.reduce_mean(ssims[1])

    def train_step(image_batch):
        image_batch = image_batch * 255

        if N_BITS < 8:
            image_batch = tf.floor(image_batch / 2 ** (8 - N_BITS))

        image_batch = image_batch / (2 ** N_BITS) - 0.5  # todo: think about 0.5

        with tf.GradientTape() as gen_tape:
            z_list = model(image_batch, training=True)
            likelihood = sum(model.losses) / BATCH_SIZE
            print("likelihood: ", likelihood.numpy())
            loss = -likelihood

        grads = gen_tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss, likelihood

    def eval_step(image_batch):
        z_list = model(image_batch, training=True)
        likelihood = sum(model.losses) / BATCH_SIZE
        return likelihood

    def generate_images(z_list, path=None, show=False):
        images = model.reverse(z_list, reconstruct=False)
        num_images = tf.shape(images).numpy()[0]
        images = tf.split(images, num_images, axis=0)
        images = [x.numpy().squeeze() for x in images]
        images = [np.clip(x + 0.5, 0, 1) for x in images]
        images = [images[: num_images // 2], images[num_images // 2 :]]
        images = [np.hstack(x) for x in images]
        images = np.vstack(images)
        images = np.clip(images * 255, 0, 255).astype(np.uint8)
        if path is not None:
            cv2.imwrite(path, images)
            # plt.savefig(path)
        if show:
            plt.show()

    def fit(train_ds, val_ds, num_epochs, initial_epoch=0):

        assert initial_epoch < num_epochs

        # z_list sample for training progress images
        z_shapes = []
        in_shape = [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL]
        for block in model.blocks:
            if block.split:
                in_shape = [in_shape[0] // 2, in_shape[1] // 2, in_shape[2] * 2]
            else:
                in_shape = [in_shape[0] // 2, in_shape[1] // 2, in_shape[2] * 4]
            z_shapes.append(in_shape)
        z_sample_list = [
            tf.random.normal(
                [NUM_SAMPLE_IMAGES] + s, mean=0, stddev=TEMPERATURE, dtype=tf.float32
            )
            for s in z_shapes
        ]

        # initialize act norm layer if model is not restored
        if initialized_from_scratch:
            for image_batch in train_ds:
                outputs = model(image_batch)
                break

        for epoch in range(initial_epoch, num_epochs):
            print("Epoch: {}".format(epoch))
            start_time = time.time()
            image_name = str(epoch) + "_train.png"
            generate_images(
                z_list=z_sample_list,
                path=os.path.join(EXPERIMENT_FOLDER, "figures", image_name),
                show=False,
            )

            # training
            log_print("Training epoch {}".format(epoch), add_timestamp=True)
            losses = [[], []]
            for n, image_batch in train_ds.enumerate():
                loss, likelihood = train_step(image_batch)
                losses[0].append(loss.numpy())
                losses[1].append(likelihood.numpy())
            losses = [statistics.mean(x) for x in losses]
            with summary_writer.as_default():
                tf.summary.scalar("loss", losses[0], step=epoch)
                tf.summary.scalar("likelihood", losses[1], step=epoch)
            summary_writer.flush()

            # testing
            log_print("Calculating validation losses...")
            val_losses = [[]]
            for image_batch in val_ds:
                likelihood = eval_step(image_batch)
                val_losses[0].append(likelihood.numpy())
            val_losses = [statistics.mean(x) for x in val_losses]
            with summary_writer.as_default():
                tf.summary.scalar("val_likelihood", val_losses[0], step=epoch)
            summary_writer.flush()

            end_time = time.time()
            log_print(
                "Epoch {} completed in {} seconds. Loss: {}, likelihood: {}, val_likelihood: {}".format(
                    epoch,
                    round(end_time - start_time),
                    losses[0],
                    losses[1],
                    val_losses[0],
                )
            )

            log_print("     loss               {:1.4f}".format(losses[0]))
            log_print("     likelihood         {:1.4f}".format(losses[1]))
            log_print("     val_likelihood     {:1.4f}".format(val_losses[0]))

            checkpoint.epoch.assign(epoch)
            if int(checkpoint.epoch) % CHECKPOINT_SAVE_INTERVAL == 0:
                save_path = manager.save()
                log_print(
                    "Saved checkpoint for epoch {}: {}".format(
                        int(checkpoint.epoch), save_path
                    )
                )
            print(
                "Epoch {} completed in {} seconds. Loss: {}, likelihood: {}, val_likelihood: {}".format(
                    epoch,
                    round(end_time - start_time),
                    losses[0],
                    losses[1],
                    val_losses[0],
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
