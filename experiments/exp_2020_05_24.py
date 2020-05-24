import datetime
import os
import statistics
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datasets.adni_dataset import get_adni_dataset
from model.losses import wgan_gp_loss, wgan_gp_loss_progressive_gan
from model.progressive_gan import progressive_gan

"""
Progressive GAN implementation

It trains in steps starting from low resolution model to high resolution one using input shapes
(6x5), (12x10), (24x20), (48x40), (96x80), (192x160).

Activation: LeakyReLU of leakiness 0.2
Normalization: L2_normalize for feature vectors

Loss: WGAN-GP loss with GP weight of 10
n_critic = 1
lr = 0.001

"""


RUNTIME = 'colab'   # cloud, colab or none
USE_TPU = False
RESTORE_FROM_CHECKPOINT = True
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]

PREFETCH_BUFFER_SIZE = 3
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
INPUT_WIDTH = 160
INPUT_HEIGHT = 192
INPUT_CHANNEL = 1
LATENT_VECTOR_SIZE = 1024

DISC_TRAIN_STEPS = 5
LAMBDA_GP = 10
CLIP_DISC_WEIGHT = None    # clip disc weight
CLIP_BY_NORM = None    # clip gradients to this norm or None
CLIP_BY_VALUE = None   # clip gradient to this value or None

EPOCHS = 5000
EPOCHS_PER_SUB_MODEL = 40
CHECKPOINT_SAVE_INTERVAL = 10
MAX_TO_KEEP = 5
LR = 1e-4


# set batch size easily
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])


if USE_TPU:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException(
            'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if RUNTIME == 'colab':
    if USE_TPU:
        EXPERIMENT_FOLDER = os.path.join('/content/experiments', EXPERIMENT_NAME)
    else:
        EXPERIMENT_FOLDER = os.path.join('/content/drive/My Drive/experiments', EXPERIMENT_NAME)
elif RUNTIME == 'cloud':
    EXPERIMENT_FOLDER = os.path.join('/home/umutkucukaslan/experiments', EXPERIMENT_NAME)
else:
    EXPERIMENT_FOLDER = os.path.join('/Users/umutkucukaslan/Desktop/thesis/experiments', EXPERIMENT_NAME)

if __name__ == "__main__":
    if not os.path.isdir(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)


def log_print(msg, add_timestamp=False):
    if not isinstance(msg, str):
        msg = str(msg)
    if add_timestamp:
        msg += ' (logged at {})'.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    with open(os.path.join(EXPERIMENT_FOLDER, 'logs.txt'), 'a+') as log_file:
        log_file.write(msg + '\n')

# generator model plot path
GEN_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'gen_model_plot.jpg')
DIS_MODEL_PLOT_PATH = os.path.join(EXPERIMENT_FOLDER, 'dis_model_plot.jpg')

# folder to save generated test images during training
if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, 'figures')):
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'figures'))

# generator and discriminator
filters = [[128, 256], [256, 512], [512, 512], [512, 512]]

basic_generators, fadein_generators, basic_discriminators, fadein_discriminators, encoder, decoder = progressive_gan(
    input_shape=[INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL],
    filters=filters,
    latent_vector_size=LATENT_VECTOR_SIZE,
    verbose=False
)

generator = basic_generators[-1]
discriminator = basic_discriminators[-1]

generator_fadein = fadein_generators[-1]
discriminator_fadein = fadein_discriminators[-1]

if __name__ == "__main__":
    encoder.summary()
    encoder.summary(print_fn=log_print)
    decoder.summary()
    decoder.summary(print_fn=log_print)
    generator.summary()
    generator.summary(print_fn=log_print)
    tf.keras.utils.plot_model(generator, to_file=GEN_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=True)
    discriminator.summary()
    discriminator.summary(print_fn=log_print)
    tf.keras.utils.plot_model(discriminator, to_file=DIS_MODEL_PLOT_PATH, show_shapes=True, dpi=150, expand_nested=False)


# optimizers
generator_optimizer = tf.optimizers.Adam(LR, beta_1=0, beta_2=0.99)
# generator_optimizer = tf.optimizers.RMSprop(learning_rate=LR)
discriminator_optimizer = tf.optimizers.Adam(LR, beta_1=0, beta_2=0.99)
# discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=LR)

# checkpoint writer
checkpoint_dir = os.path.join(EXPERIMENT_FOLDER, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 generator_fadein=generator_fadein,
                                 discriminator_fadein=discriminator_fadein)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=MAX_TO_KEEP)

if RESTORE_FROM_CHECKPOINT:
    checkpoint.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    log_print("Restored from {}".format(manager.latest_checkpoint))
else:
    log_print("Initializing from scratch.")

initial_epoch = checkpoint.epoch.numpy() + 1


def get_encoder_decoder_generator_discriminator(return_experiment_folder=True):
    """
    This function returns the constructed and restored (if possible) sub-models that are constructed in this experiment

    :return: encoder, decoder, generator, discriminator
    """
    if return_experiment_folder:
        return encoder, decoder, generator, discriminator, EXPERIMENT_FOLDER
    return encoder, decoder, generator, discriminator


if __name__ == "__main__":

    # summary file writer for tensorboard
    log_dir = os.path.join(EXPERIMENT_FOLDER, 'logs')
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))


    # DATASET
    # train_ds, train_ds2, val_ds, test_ds = get_adni_dataset(folder_name='processed_data_192x160', machine=RUNTIME, return_two_trains=True)

    # train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
    # val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)


    # for example in train_ds.take(5):
    #     plt.imshow(np.squeeze(example.numpy()[0]), cmap=plt.get_cmap('gray'))
    #     # plt.show()
    #     img = example.numpy()
    #     print('mean value: ', img.mean())
    #     print('max value : ', img.max())
    #     print('min value : ', img.min())
    # exit()


    # training

    def train_step(generator, discriminator, weight, input_image, target, train_generator=True, train_discriminator=True):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            if weight is not None:
                generated_image = generator([input_image, weight], training=True)
                gen_loss, disc_loss, gp_loss = wgan_gp_loss_progressive_gan(discriminator, target, generated_image, LAMBDA_GP, weight)
            else:
                generated_image = generator(input_image, training=True)
                gen_loss, disc_loss, gp_loss = wgan_gp_loss_progressive_gan(discriminator, target, generated_image, LAMBDA_GP)

        if train_generator:
            generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            if CLIP_BY_NORM is not None:
                generator_gradients = [tf.clip_by_norm(t, CLIP_BY_NORM) for t in generator_gradients]
            if CLIP_BY_VALUE is not None:
                generator_gradients = [tf.clip_by_value(t, -CLIP_BY_VALUE, CLIP_BY_VALUE) for t in generator_gradients]
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        if train_discriminator:
            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            if CLIP_BY_NORM is not None:
                discriminator_gradients = [tf.clip_by_norm(t, CLIP_BY_NORM) for t in discriminator_gradients]
            if CLIP_BY_VALUE is not None:
                discriminator_gradients = [tf.clip_by_value(t, -CLIP_BY_VALUE, CLIP_BY_VALUE) for t in discriminator_gradients]
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        if CLIP_DISC_WEIGHT:
            for disc_var in discriminator.trainable_variables:
                disc_var.assign(tf.clip_by_value(disc_var, -CLIP_DISC_WEIGHT, CLIP_DISC_WEIGHT))

        return gen_loss, disc_loss, gp_loss


    def eval_step(generator, discriminator, weight, input_image, target):
        if weight is not None:
            # weight = np.asarray(input_image.shape[0] * [[weight]])
            generated_image = generator([input_image, weight], training=False)
            gen_loss, disc_loss, gp_loss = wgan_gp_loss_progressive_gan(discriminator, target, generated_image, LAMBDA_GP, weight)
        else:
            generated_image = generator(input_image, training=False)
            gen_loss, disc_loss, gp_loss = wgan_gp_loss_progressive_gan(discriminator, target, generated_image, LAMBDA_GP)

        return gen_loss, disc_loss, gp_loss


    def generate_images(model, test_input, path=None, show=True, weight=None):
        if test_input.ndim < 4:
            test_input = np.expand_dims(test_input, axis=0)
        if weight is not None:
            prediction = model([test_input, weight])
        else:
            prediction = model(test_input)
        if isinstance(test_input, tf.Tensor):
            display_list = [np.squeeze(test_input.numpy()[0, :, :, 0]), np.squeeze(prediction.numpy()[0, :, :, 0])]
        else:
            display_list = [np.squeeze(test_input[0, :, :, 0]), np.squeeze(prediction[0, :, :, 0])]
        title = ['Input Image', 'Reconstructed Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap=plt.get_cmap('gray'))
            plt.axis('off')
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()

    def get_weight(epoch, train_interval):
        if epoch < train_interval[0]:
            return 0.
        if epoch > train_interval[1]:
            return 1.
        return (epoch - train_interval[0]) / (train_interval[1] - train_interval[0])

    def fit_to_given_models(generator, discriminator, train_ds, val_ds, test_ds, train_ds_images, num_epochs, initial_epoch=0, use_weight=False, train_interval=None, name=None):

        # assert initial_epoch < num_epochs
        test_ds = iter(test_ds)
        train_ds_images = iter(train_ds_images)
        for epoch in range(initial_epoch, num_epochs):
            weight = None
            if use_weight and train_interval:
                weight = get_weight(epoch, train_interval)

            print('Processing for epoch {}, weight {}, name {}'.format(epoch, str(weight), name))
            start_time = time.time()
            test_input = next(test_ds)
            image_name = str(epoch) + '_test.png'
            generate_images(generator,
                            test_input.numpy(),
                            os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                            show=False,
                            weight=weight)
            train_input = next(train_ds_images)
            image_name = str(epoch) + '_train.png'
            generate_images(generator,
                            train_input.numpy(),
                            os.path.join(EXPERIMENT_FOLDER, 'figures', image_name),
                            show=False,
                            weight=weight)

            # training
            log_print('Training epoch {}'.format(epoch), add_timestamp=True)
            losses = [[], [], []]
            for n, input_image in train_ds.enumerate():
                if n.numpy() % (DISC_TRAIN_STEPS + 1) == 0:
                    gen_loss, disc_loss, gp_loss = train_step(generator, discriminator, weight, input_image=input_image, target=input_image, train_generator=True, train_discriminator=False)
                else:
                    gen_loss, disc_loss, gp_loss = train_step(generator, discriminator, weight, input_image=input_image, target=input_image, train_generator=False, train_discriminator=True)

                losses[0].append(gen_loss.numpy())
                losses[1].append(disc_loss.numpy())
                losses[2].append(gp_loss.numpy())
            losses = [statistics.mean(x) for x in losses]
            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', losses[0], step=epoch)
                tf.summary.scalar('disc_loss', losses[1], step=epoch)
                tf.summary.scalar('gp_loss', losses[2], step=epoch)
            summary_writer.flush()

            # testing
            log_print('Calculating validation losses...')
            val_losses = [[], [], []]
            for input_image in val_ds:
                gen_loss, disc_loss, gp_loss = eval_step(generator, discriminator, weight, input_image, input_image)
                val_losses[0].append(gen_loss.numpy())
                val_losses[1].append(disc_loss.numpy())
                val_losses[2].append(gp_loss.numpy())

            val_losses = [statistics.mean(x) for x in val_losses]
            with summary_writer.as_default():
                tf.summary.scalar('val_gen_loss', val_losses[0], step=epoch)
                tf.summary.scalar('val_disc_loss', val_losses[1], step=epoch)
                tf.summary.scalar('val_gp_loss', val_losses[2], step=epoch)
            summary_writer.flush()

            end_time = time.time()
            log_print('Epoch {} completed in {} seconds'.format(epoch, round(end_time - start_time)))
            log_print("     gen_loss       {:1.4f}".format(losses[0]))
            log_print("     disc_loss      {:1.4f}".format(losses[1]))
            log_print("     gp_loss        {:1.4f}".format(losses[2]))

            log_print("     val_gen_loss       {:1.4f}".format(val_losses[0]))
            log_print("     val_disc_loss      {:1.4f}".format(val_losses[1]))
            log_print("     val_gp_loss        {:1.4f}".format(val_losses[2]))

            checkpoint.epoch.assign(epoch)

            if int(checkpoint.epoch) % CHECKPOINT_SAVE_INTERVAL == 0:
                save_path = manager.save()
                log_print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))
                # print("gen_total_loss {:1.2f}".format(gen_total_loss.numpy()))
                # print("disc_loss {:1.2f}".format(disc_loss.numpy()))

    def get_train_models():
        """
        Returns a list of generator and discriminator pair with their input shape in the order of training.
        :return:
        """
        input_shape = [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL]
        n_downsampling = len(filters)
        input_shapes = [[input_shape[0] // 2 ** n, input_shape[1] // 2 ** n, input_shape[2]] for n in
                        range(n_downsampling + 1)]

        # generator, discriminator, input_shape to be trained
        train_models = []
        train_models.append((basic_generators[0], basic_discriminators[0], input_shapes[-1], False, 'basic'))

        for i in range(len(fadein_generators)):
            fadein_gen = fadein_generators[i]
            fadein_disc = fadein_discriminators[i]
            basic_gen = basic_generators[i + 1]
            basic_disc = basic_discriminators[i + 1]
            in_shape = input_shapes[-i - 2]

            train_models.append((fadein_gen, fadein_disc, in_shape, True, 'fadein'))
            train_models.append((basic_gen, basic_disc, in_shape, False, 'basic'))

        return train_models


    def process_datasets(raw_datasets, input_shape):
        train_list_ds, train_list_ds2, val_list_ds, test_list_ds = raw_datasets

        image_shape = [input_shape[0], input_shape[1]]
        num_channels = input_shape[2]

        def decode_img(img, image_shape, num_channel=1):
            img = tf.io.decode_png(img, channels=num_channel)
            img = tf.image.resize(img, image_shape)
            img = tf.cast(img, tf.float32)
            img = img / 256.0
            return img

        def process_path(file_path):
            img = tf.io.read_file(file_path)
            img = decode_img(img, image_shape, num_channels)
            return img

        train_ds = train_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds2 = train_list_ds2.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

        return train_ds, train_ds2, val_ds, test_ds

    def fit(epochs_per_model=40, check_with_small_dataset=False):
        list_datasets = get_adni_dataset(folder_name='processed_data_192x160', machine=RUNTIME, return_two_trains=True, return_raw_dataset=True)
        train_models = get_train_models()

        train_models = train_models[4:]

        for i in range(len(train_models)):
            gen, disc, in_shape, use_weight, name = train_models[i]
            train_ds, train_ds2, val_ds, test_ds = process_datasets(list_datasets, in_shape)

            print('MODELS BEING TRAINED WITH INPUT SHAPE   {}  of type {}'.format(str(in_shape), name))
            gen.summary()
            disc.summary()

            if check_with_small_dataset:
                train_ds = train_ds.take(10)
                train_ds2 = train_ds2.take(10)
                val_ds = val_ds.take(3)
                test_ds = test_ds.take(5)

            initial_epoch = checkpoint.epoch.numpy() + 1
            train_interval = [i * epochs_per_model, (i + 1) * epochs_per_model]
            fit_to_given_models(gen, disc, train_ds, val_ds, test_ds.repeat(), train_ds2.repeat(), (i + 1) * epochs_per_model, initial_epoch, use_weight, train_interval, name)

    try:
        log_print('Fitting to the data set', add_timestamp=True)
        log_print(' ')
        log_print('Parameters:')
        log_print('Experiment name: ' + str(EXPERIMENT_NAME))
        log_print('Batch size: ' + str(BATCH_SIZE))
        # log_print('Epochs: ' + str(EPOCHS))
        log_print('Epochs per sub model: ' + str(EPOCHS_PER_SUB_MODEL))
        log_print('Restore from checkpoint: ' + str(RESTORE_FROM_CHECKPOINT))
        log_print('Chechpoint save interval: ' + str(CHECKPOINT_SAVE_INTERVAL))
        log_print('Max number of checkpoints kept: ' + str(MAX_TO_KEEP))
        log_print('Runtime: ' + str(RUNTIME))
        log_print('Use TPU: ' + str(USE_TPU))
        log_print('Prefetch buffer size: ' + str(PREFETCH_BUFFER_SIZE))
        log_print('Shuffle buffer size: ' + str(SHUFFLE_BUFFER_SIZE))
        log_print('Input shape: ( ' + str(INPUT_HEIGHT) + ', ' + str(INPUT_WIDTH) + ', ' + str(INPUT_CHANNEL) + ' )')
        log_print('LAMBDA_GP: ' + str(LAMBDA_GP))
        log_print('Clip by norm: ' + str(CLIP_BY_NORM))
        log_print('Clip by value: ' + str(CLIP_BY_VALUE))
        log_print('Discriminator train steps/epoch: ' + str(DISC_TRAIN_STEPS))

        log_print(' ')

        log_print('Initial epoch: {}'.format(initial_epoch))
        # fit(train_ds.take(10), EPOCHS, val_ds.take(2), test_ds.repeat(), train_ds2.repeat(), initial_epoch=initial_epoch)
        fit(epochs_per_model=EPOCHS_PER_SUB_MODEL, check_with_small_dataset=False)

        # save last checkpoint
        save_path = manager.save()
        log_print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))
        summary_writer.close()

    except KeyboardInterrupt:
        log_print('Keyboard Interrupt', add_timestamp=True)

        # save latest checkpoint and close log file
        save_path = manager.save()
        log_print("Saved checkpoint for epoch {}: {} due to KeyboardInterrupt".format(int(checkpoint.epoch), save_path))
        summary_writer.close()

    except:
        summary_writer.close()