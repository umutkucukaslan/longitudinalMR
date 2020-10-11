import datetime
import os
import statistics
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from datasets.spie_dataset import get_spie_dataset, SPIEDataset
from model.losses import binary_cross_entropy_with_logits
from model.dcgan import make_dcgan_discriminator_model, make_dcgan_generator_model

"""
SPIE paper implementation using W-GAN (without example re-weighting)
Latent vector size: 256
Output image shape: 64 x 64 x 1

PS: 192x160 images are resized to 64x64

n_critic: 5
"""

# Input latent vector size for generator
latent_vector_size = 256

# Kernel size for generator and discriminator conv layers
kernel_size = 5

# shuffle training images
shuffle_training_data = True

# num_critic_train_steps / num_generator_train_steps
n_critic = 5

# use example re-weighting when computing total loss
example_reweighting = True

RUNTIME = "colab"  # cloud, colab or none
RESTORE_FROM_CHECKPOINT = True
EXPERIMENT_NAME = "ref_spie_wgan_rw"

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000

BATCH_SIZE = 128
CLIP_DISC_WEIGHT = 0.01  # clip disc weight

EPOCHS = 2000
CHECKPOINT_SAVE_INTERVAL = 50
MAX_TO_KEEP = 5
LEARNING_RATE = 0.00005

USE_TPU = False

# set batch size easily
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])


if USE_TPU:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    except ValueError:
        raise BaseException(
            "ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!"
        )

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if RUNTIME == "colab":
    if USE_TPU:
        EXPERIMENT_FOLDER = os.path.join("/content/experiments", EXPERIMENT_NAME)
    else:
        EXPERIMENT_FOLDER = os.path.join(
            "/content/drive/My Drive/experiments", EXPERIMENT_NAME
        )
elif RUNTIME == "cloud":
    EXPERIMENT_FOLDER = os.path.join(
        "/home/umutkucukaslan/experiments", EXPERIMENT_NAME
    )
else:
    EXPERIMENT_FOLDER = os.path.join(
        "/Users/umutkucukaslan/Desktop/thesis/experiments", EXPERIMENT_NAME
    )

if __name__ == "__main__":
    if not os.path.isdir(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)


def log_print(msg, add_timestamp=False):
    if not isinstance(msg, str):
        msg = str(msg)
    if add_timestamp:
        msg += " (logged at {})".format(
            datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    with open(os.path.join(EXPERIMENT_FOLDER, "logs.txt"), "a+") as log_file:
        log_file.write(msg + "\n")


# generator model plot path
GEN_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, "gen_model_plot.jpg")
DIS_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, "dis_model_plot.jpg")

# folder to save generated test images during training
if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, "figures")):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "figures"))

# generator and discriminator
generator = make_dcgan_generator_model(input_vector_size=latent_vector_size)
discriminator = make_dcgan_discriminator_model(kernel_size=(kernel_size, kernel_size))

if __name__ == "__main__":
    generator.summary()
    generator.summary(print_fn=log_print)
    tf.keras.utils.plot_model(
        generator,
        to_file=GEN_MODEL_PLOT_PATH,
        show_shapes=True,
        dpi=150,
        expand_nested=True,
    )
    discriminator.summary()
    discriminator.summary(print_fn=log_print)
    tf.keras.utils.plot_model(
        discriminator,
        to_file=DIS_MODEL_PLOT_PATH,
        show_shapes=True,
        dpi=150,
        expand_nested=False,
    )

# optimizers
generator_optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=LEARNING_RATE)

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, "checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    epoch=tf.Variable(0),
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)
manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP
)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    log_print("Restored from {}".format(manager.latest_checkpoint))
else:
    log_print("Initializing from scratch.")

initial_epoch = checkpoint.epoch.numpy() + 1


def get_generator_discriminator(return_experiment_folder=True):
    """
    This function returns the constructed and restored (if possible) sub-models that are constructed in this experiment

    :return: generator, discriminator, experiment_folder (optional)
    """
    if return_experiment_folder:
        return generator, discriminator, EXPERIMENT_FOLDER
    return generator, discriminator


if __name__ == "__main__":

    # summary file writer for tensorboard
    log_dir = os.path.join(EXPERIMENT_FOLDER, "logs")
    summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    spie_dataset = get_spie_dataset(
        folder_name="training_data_15T_192x160_4slices", machine=RUNTIME
    )

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def generator_loss(fake_output, weights):
        ce = binary_cross_entropy_with_logits(
            tf.ones_like(fake_output), fake_output, from_logits=False
        )
        weights = tf.convert_to_tensor(weights, dtype=ce.dtype)
        gen_loss = tf.reduce_mean(ce * weights)

        return gen_loss

    def discriminator_loss(real_output, fake_output, weights):
        real_loss = binary_cross_entropy_with_logits(
            tf.ones_like(real_output), real_output, from_logits=False
        )
        fake_loss = binary_cross_entropy_with_logits(
            tf.zeros_like(fake_output), fake_output, from_logits=False
        )
        loss_per_sample = real_loss + fake_loss
        weights = tf.convert_to_tensor(weights, dtype=loss_per_sample.dtype)
        total_loss = tf.reduce_mean(loss_per_sample * weights)

        return total_loss, loss_per_sample

    def train_step(images, weights, train_generator=False, train_discriminator=False):
        noise = tf.random.normal([BATCH_SIZE, latent_vector_size])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output, weights)
            disc_loss, loss_per_sample = discriminator_loss(
                real_output, fake_output, weights
            )

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        if train_generator:
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables)
            )
        if train_discriminator:
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables)
            )
            # enforces K-Lipschitzness for WGAN
            for variable in discriminator.trainable_variables:
                variable.assign(
                    tf.clip_by_value(variable, -CLIP_DISC_WEIGHT, CLIP_DISC_WEIGHT)
                )

        return gen_loss, disc_loss, loss_per_sample

    def eval_step(images):
        noise = tf.random.normal([BATCH_SIZE, latent_vector_size])
        weights = tf.ones([images.shape[0]])
        generated_images = generator(noise, training=False)
        real_output = discriminator(images, training=False)
        fake_output = discriminator(generated_images, training=False)
        gen_loss = generator_loss(fake_output, weights)
        disc_loss, loss_per_sample = discriminator_loss(
            real_output, fake_output, weights
        )

        return gen_loss, disc_loss, loss_per_sample

    def generate_images(
        noise: tf.Tensor, generator: tf.keras.Model, path=None, show=False
    ):
        """
        Generates images from given noise and saves/plots them as specified.
        Noise batch should have at least 6 samples.

        :param noise:
        :param generator:
        :param path:
        :param show:
        :return:
        """
        generated_images = generator(noise, training=False)
        images = [
            generated_images[i, :, :, 0].numpy()
            for i in range(generated_images.shape[0])
        ]
        upper_display_list = images[0:3]
        lower_display_list = images[3:6]
        for i in range(3):
            plt.subplot(2, 3, i + 1)
            plt.imshow(upper_display_list[i], cmap=plt.get_cmap("gray"))
            plt.axis("off")
            plt.subplot(2, 3, i + 4)
            plt.imshow(lower_display_list[i], cmap=plt.get_cmap("gray"))
            plt.axis("off")
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()

    def fit(dataset: SPIEDataset, num_epochs, initial_epoch=0):
        sample_noise = tf.random.normal(
            [6, latent_vector_size]
        )  # used for images plotted
        assert initial_epoch < num_epochs
        for epoch in range(initial_epoch, num_epochs):
            print("Epoch: {}".format(epoch))
            start_time = time.time()

            image_name = str(epoch) + "_images.png"
            generate_images(
                sample_noise,
                generator,
                path=os.path.join(EXPERIMENT_FOLDER, "figures", image_name),
                show=False,
            )

            log_print("Training epoch {}".format(epoch), add_timestamp=True)
            losses = [[], []]
            counter = 0
            for images, info, weights in dataset.get_training_images(
                batch_size=BATCH_SIZE, shuffle=shuffle_training_data
            ):
                if counter == n_critic:
                    counter = 0
                    gen_loss, disc_loss, loss_per_sample = train_step(
                        images, weights, train_generator=True, train_discriminator=False
                    )
                else:
                    gen_loss, disc_loss, loss_per_sample = train_step(
                        images, weights, train_generator=False, train_discriminator=True
                    )

                    counter += 1

                if example_reweighting:
                    dataset.update_losses(info, loss_per_sample.numpy())

                losses[0].append(gen_loss.numpy())
                losses[1].append(disc_loss.numpy())

            if example_reweighting:
                print("UPDATING WEIGHTS")
                dataset.update_training_weights(logic="simple")
                print("UPDATED WEIGHTS")
            losses = [statistics.mean(x) for x in losses]
            with summary_writer.as_default():
                tf.summary.scalar("generator_loss", losses[0], step=epoch)
                tf.summary.scalar("critic_loss", losses[1], step=epoch)
            summary_writer.flush()

            # testing
            log_print("Calculating validation losses...")
            val_losses = [[], []]
            for images, info, weights in dataset.get_val_images(
                batch_size=BATCH_SIZE, shuffle=False
            ):
                gen_loss, disc_loss, loss_per_sample = eval_step(images)
                val_losses[0].append(gen_loss.numpy())
                val_losses[1].append(disc_loss.numpy())

            val_losses = [statistics.mean(x) for x in val_losses]
            with summary_writer.as_default():
                tf.summary.scalar(
                    "validation_generator_loss", val_losses[0], step=epoch
                )
                tf.summary.scalar("validation_critic_loss", val_losses[1], step=epoch)
            summary_writer.flush()

            end_time = time.time()
            log_print(
                "Epoch {} completed in {} seconds".format(
                    epoch, round(end_time - start_time)
                )
            )
            log_print("     generator loss       {:1.4f}".format(losses[0]))
            log_print("     discriminator loss   {:1.4f}".format(losses[1]))
            log_print("     val generator loss   {:1.4f}".format(val_losses[0]))
            log_print("     val critic loss      {:1.4f}".format(val_losses[1]))

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
        log_print("Checkpoint save interval: " + str(CHECKPOINT_SAVE_INTERVAL))
        log_print("Max number of checkpoints kept: " + str(MAX_TO_KEEP))
        log_print("Runtime: " + str(RUNTIME))
        log_print("Use TPU: " + str(USE_TPU))
        log_print("Input shape: ( " + str(64) + ", " + str(64) + ", " + str(1) + " )")
        log_print(" ")
        log_print("Initial epoch: {}".format(initial_epoch))
        log_print("----------------")
        log_print("n_critic: {}".format(n_critic))
        log_print("Example re-weighting: {}".format(example_reweighting))
        log_print("Conv kernel size: {}".format(kernel_size))
        log_print("Critic variable clip value: {}".format(CLIP_DISC_WEIGHT))
        log_print("Learning rate: {}".format(LEARNING_RATE))

        fit(
            spie_dataset, EPOCHS, initial_epoch=initial_epoch,
        )

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
