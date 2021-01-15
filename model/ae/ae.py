import statistics

import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class AE(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size,
        activation=tf.nn.silu,
        last_activation=tf.nn.sigmoid,
        structure_vec_size=100,
        longitudinal_vec_size=1,
        **kwargs,
    ):
        super(AE, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.structure_vec_size = structure_vec_size
        self.longitudinal_vec_size = longitudinal_vec_size
        self.downsample_conv = [
            tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                strides=2,
                activation=activation,
            )
            for f in filters
        ]
        self.structure_dense = tf.keras.layers.Dense(
            structure_vec_size, activation=tf.nn.tanh, use_bias=False
        )
        self.longitudinal_dense = tf.keras.layers.Dense(
            longitudinal_vec_size, activation=None, use_bias=False
        )
        self.upsample_layers = [
            tf.keras.layers.Conv2D(
                filters=f,
                kernel_size=kernel_size,
                padding="same",
                strides=1,
                activation=activation,
            )
            for f in reversed(filters[:-1])
        ]
        self.upsample_layers.append(
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                strides=1,
                activation=last_activation,
            )
        )

    def build(self, input_shape):
        _, height, width, channels = input_shape
        num_features = (
            height
            // (2 ** len(self.filters))
            * width
            // (2 ** len(self.filters))
            * self.filters[-1]
        )
        self.decode_input_shape = [
            height // (2 ** len(self.filters)),
            width // (2 ** len(self.filters)),
            self.filters[-1],
        ]
        self.num_features = num_features
        self.decode_structure_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )
        self.decode_longitudinal_dense = tf.keras.layers.Dense(
            num_features, activation=None, use_bias=False
        )

    def encode(self, image_batch, training=None):
        x = image_batch
        for layer in self.downsample_conv:
            x = layer(x)
        x = tf.keras.layers.Flatten()(x)
        structure = self.structure_dense(x)
        longitudinal_state = self.longitudinal_dense(x)
        return structure, longitudinal_state

    def decode(self, structure, longitudinal_state, training=None):
        x = self.decode_structure_dense(structure)
        y = self.decode_longitudinal_dense(longitudinal_state)
        x = tf.math.add(x, y)
        x = tf.reshape(x, shape=[-1] + self.decode_input_shape)
        for up_conv in self.upsample_layers:
            x = tf.keras.layers.UpSampling2D()(x)
            x = up_conv(x)
        return x

    def call(self, inputs, training=None, mask=None):
        structure, state = self.encode(inputs, training=training)
        out = self.decode(structure, state, training=training)
        return out

    def interpolate_and_calculate_ssim(
        self,
        inputs1,
        inputs2,
        sample_points,
        ground_truth,
        ground_truth_index,
        dates=None,
        structure_mix_type="mean",
    ):
        if dates is None:
            dates = [round(p, 2) for p in sample_points]
        dates = [str(x) for x in dates]

        interpolations = self.interpolate(
            inputs1,
            inputs2,
            sample_points,
            dates=dates,
            structure_mix_type=structure_mix_type,
            return_as_image=False,
        )
        ssim = self.calculate_ssim([ground_truth], [interpolations[ground_truth_index]])
        interpolations = [
            np.clip(x.numpy()[0, ...] * 255, 0, 255).astype(np.uint8)
            for x in interpolations
        ]
        interpolations = [
            cv2.putText(x, str(p), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            for x, p in zip(interpolations, dates)
        ]
        for idx in {0, 1, 2} - {ground_truth_index}:
            cv2.putText(
                interpolations[idx], "*", (0, 63), cv2.FONT_HERSHEY_PLAIN, 1, 255
            )
        interpolations = np.hstack(interpolations)
        return interpolations, ssim.numpy()

    def interpolate(
        self,
        inputs1,
        inputs2,
        sample_points,
        dates=None,
        structure_mix_type="mean",
        return_as_image=False,
    ):
        if dates is None:
            dates = [round(p, 2) for p in sample_points]
        structure1, state1 = self.encode(inputs1, training=False)
        structure2, state2 = self.encode(inputs2, training=False)
        if structure_mix_type == "first":
            structure = structure1
        elif structure_mix_type == "second":
            structure = structure2
        elif structure_mix_type == "mean":
            structure = (structure1 + structure2) / 2.0
        else:
            raise ValueError(f"structure mix type {structure_mix_type} is unknown")
        diff = state2 - state1
        state_vecs = [state1 + diff * sample_point for sample_point in sample_points]
        interpolations = [
            self.decode(structure, state, training=False) for state in state_vecs
        ]
        if return_as_image:
            interpolations = [
                np.clip(x.numpy()[0, ...] * 255, 0, 255).astype(np.uint8)
                for x in interpolations
            ]
            interpolations = [
                cv2.putText(x, str(p), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, 255)
                for x, p in zip(interpolations, dates)
            ]
            interpolations = np.hstack(interpolations)
        return interpolations

    def interpolate_and_ssim(
        self,
        inputs1,
        inputs2,
        sample_point,
        ground_truth,
        structure_mix_type="mean",
        return_as_image=False,
    ):
        interpolations = self.interpolate(
            inputs1,
            inputs2,
            [sample_point],
            structure_mix_type=structure_mix_type,
            return_as_image=False,
        )
        ssim = self.calculate_ssim([ground_truth], interpolations)
        if return_as_image:
            grth = np.clip(ground_truth.numpy()[0, ...] * 255, 0, 255).astype(np.uint8)
            pred = np.clip(interpolations[0].numpy()[0, ...] * 255, 0, 255).astype(
                np.uint8
            )
            img = np.hstack([grth, pred])
            return ssim, img
        return ssim

    def train_for_patient(
        self,
        img1,
        img2,
        train_ds,
        val_ds,
        num_steps=1,
        period=10,
        val_period=100,
        train_loss_period=100,
        lr=1e-4,
        callback_fn_generate_seq=None,
        callback_fn_save_pair_losses=None,
        callback_fn_save_val_losses=None,
        callback_fn_save_train_losses=None,
    ):
        optimizer = tf.optimizers.Adam(lr, beta_1=0.5)
        sim_loss_fn = tf.keras.losses.MeanSquaredError()
        structure_vec_similarity_loss_mult = 100

        steps_counter = 0
        train_losses = [[], [], [], []]
        while steps_counter < num_steps:
            for n, inputs in train_ds.enumerate():
                if steps_counter % val_period == 0 and callback_fn_save_val_losses:
                    losses = self.validate_on_dataset(
                        val_ds,
                        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
                        structure_vec_similarity_loss_mult=100,
                        verbose=False,
                    )
                    val_ds_losses = {
                        "total_loss": losses[0],
                        "image_similarity_loss": losses[1],
                        "structure_vec_sim_loss": losses[2],
                        "ssim": losses[3],
                    }
                    callback_fn_save_val_losses(steps_counter, val_ds_losses)

                if steps_counter % period == 0:
                    if callback_fn_generate_seq:
                        callback_fn_generate_seq(steps_counter)
                    (
                        total_loss,
                        image_similarity_loss,
                        structure_vec_sim_loss,
                        ssims,
                        predicted_imgs_for_vis,
                    ) = self.train_using_pair([img1, img2], optimizer)
                    pair_losses = {
                        "total_loss": total_loss.numpy(),
                        "image_similarity_loss": image_similarity_loss.numpy(),
                        "structure_vec_sim_loss": structure_vec_sim_loss.numpy(),
                        "ssim": ssims.numpy(),
                    }
                    if callback_fn_save_pair_losses:
                        callback_fn_save_pair_losses(steps_counter, pair_losses)

                imgs = inputs["imgs"]
                days = inputs["days"]
                (
                    total_loss,
                    image_similarity_loss,
                    structure_vec_sim_loss,
                    ssims,
                    predicted_imgs,
                ) = self.train_using_triplet(
                    imgs,
                    days,
                    optimizer,
                    sim_loss_fn=sim_loss_fn,
                    structure_vec_similarity_loss_mult=structure_vec_similarity_loss_mult,
                )
                train_losses[0].append(total_loss.numpy())
                train_losses[1].append(image_similarity_loss.numpy())
                train_losses[2].append(structure_vec_sim_loss.numpy())
                train_losses[3].append(ssims.numpy())
                if steps_counter % train_loss_period == 0:
                    train_losses = [statistics.mean(x) for x in train_losses]
                    train_ds_losses = {
                        "total_loss": train_losses[0],
                        "image_similarity_loss": train_losses[1],
                        "structure_vec_sim_loss": train_losses[2],
                        "ssim": train_losses[3],
                    }
                    train_losses = [[], [], [], []]
                    if callback_fn_save_train_losses:
                        callback_fn_save_train_losses(steps_counter, train_ds_losses)

                steps_counter += 1
                if steps_counter == num_steps:
                    break

    def train_for_patient2(
        self,
        img1,
        img2,
        train_ds,
        val_ds,
        num_repeat=1,
        period=10,
        lr=1e-4,
        callback_fn=None,
    ):
        optimizer = tf.optimizers.Adam(lr, beta_1=0.5)
        sim_loss_fn = tf.keras.losses.MeanSquaredError()
        structure_vec_similarity_loss_mult = 100

        for repeat_count in range(num_repeat):
            self.validate_on_dataset(
                val_ds, sim_loss_fn, structure_vec_similarity_loss_mult
            )
            self.train_on_dataset_and_pair(
                train_ds,
                [img1, img2],
                optimizer,
                period=period,
                sim_loss_fn=sim_loss_fn,
                structure_vec_similarity_loss_mult=structure_vec_similarity_loss_mult,
                callback_fn=callback_fn,
            )
            # pbar = tqdm()
            # for repeat_pair_count in range(num_repeat_pair):
            #     (
            #         total_loss,
            #         image_similarity_loss,
            #         structure_vec_sim_loss,
            #         ssims,
            #         predicted_imgs_for_vis,
            #     ) = self.train_using_pair([img1, img2], optimizer)
            #     pbar.update(1)
            #     pbar.set_description(
            #         f"training using pair.. Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
            #         + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
            #     )
            # pbar.close()
            # self.validate_on_dataset(
            #     val_ds, sim_loss_fn, structure_vec_similarity_loss_mult
            # )
            # self.train_on_dataset(
            #     train_ds, optimizer, sim_loss_fn, structure_vec_similarity_loss_mult,
            # )
        self.validate_on_dataset(
            val_ds, sim_loss_fn, structure_vec_similarity_loss_mult
        )

    def train_on_dataset_and_pair(
        self,
        ds,
        pair,
        optimizer,
        period=10,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
        callback_fn=None,
    ):
        losses = [[], [], [], []]
        pbar = tqdm()
        for n, inputs in ds.enumerate():
            if n % period == 0:
                if callback_fn:
                    callback_fn()
                (
                    total_loss,
                    image_similarity_loss,
                    structure_vec_sim_loss,
                    ssims,
                    predicted_imgs_for_vis,
                ) = self.train_using_pair(pair, optimizer)
                print(
                    f"\ntraining using pair.. Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                    + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
                )
            imgs = inputs["imgs"]
            days = inputs["days"]
            (
                total_loss,
                image_similarity_loss,
                structure_vec_sim_loss,
                ssims,
                predicted_imgs,
            ) = self.train_using_triplet(
                imgs,
                days,
                optimizer,
                sim_loss_fn=sim_loss_fn,
                structure_vec_similarity_loss_mult=structure_vec_similarity_loss_mult,
            )

            losses[0].append(total_loss.numpy())
            losses[1].append(image_similarity_loss.numpy())
            losses[2].append(structure_vec_sim_loss.numpy())
            losses[3].append(ssims.numpy())
            pbar.update(1)
            pbar.set_description(
                f"training..... Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
            )
        losses = [statistics.mean(x) for x in losses]
        pbar.set_description(
            f"training..... Total loss: {losses[0]:.5f}; image_sim_mse: {losses[1]:.5f}; "
            + f"structure_vec_mse: {losses[2]:.5f}; ssim: {losses[3]:.5f}"
        )
        pbar.close()

    def validate_on_dataset(
        self,
        ds,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
        verbose=False,
    ):
        val_losses = [[], [], [], []]
        if verbose:
            pbar = tqdm()
        for n, inputs in ds.enumerate():
            imgs = inputs["imgs"]
            days = inputs["days"]
            (
                total_loss,
                image_similarity_loss,
                structure_vec_sim_loss,
                ssims,
                predicted_imgs,
            ) = self.eval_using_triplet(
                imgs,
                days,
                sim_loss_fn=sim_loss_fn,
                structure_vec_similarity_loss_mult=structure_vec_similarity_loss_mult,
            )
            val_losses[0].append(total_loss.numpy())
            val_losses[1].append(image_similarity_loss.numpy())
            val_losses[2].append(structure_vec_sim_loss.numpy())
            val_losses[3].append(ssims.numpy())
            if verbose:
                pbar.update(1)
                pbar.set_description(
                    f"validating.. Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                    + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
                )
        val_losses = [statistics.mean(x) for x in val_losses]
        if verbose:
            pbar.set_description(
                f"validating.. Total loss: {val_losses[0]:.5f}; image_sim_mse: {val_losses[1]:.5f}; "
                + f"structure_vec_mse: {val_losses[2]:.5f}; ssim: {val_losses[3]:.5f}"
            )
            pbar.close()
        return val_losses

    def train_on_dataset(
        self,
        ds,
        optimizer,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
    ):
        losses = [[], [], [], []]
        pbar = tqdm()
        for n, inputs in ds.enumerate():
            imgs = inputs["imgs"]
            days = inputs["days"]
            (
                total_loss,
                image_similarity_loss,
                structure_vec_sim_loss,
                ssims,
                predicted_imgs,
            ) = self.train_using_triplet(
                imgs,
                days,
                optimizer,
                sim_loss_fn=sim_loss_fn,
                structure_vec_similarity_loss_mult=structure_vec_similarity_loss_mult,
            )

            losses[0].append(total_loss.numpy())
            losses[1].append(image_similarity_loss.numpy())
            losses[2].append(structure_vec_sim_loss.numpy())
            losses[3].append(ssims.numpy())
            pbar.update(1)
            pbar.set_description(
                f"training..... Total loss: {total_loss.numpy():.5f}; image_sim_mse: {image_similarity_loss.numpy():.5f}; "
                + f"structure_vec_mse: {structure_vec_sim_loss.numpy():.5f}; ssim: {ssims.numpy():.5f}"
            )
        losses = [statistics.mean(x) for x in losses]
        pbar.set_description(
            f"training..... Total loss: {losses[0]:.5f}; image_sim_mse: {losses[1]:.5f}; "
            + f"structure_vec_mse: {losses[2]:.5f}; ssim: {losses[3]:.5f}"
        )
        pbar.close()

    def eval_using_triplet(
        self,
        imgs,
        days,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
    ):
        predicted_imgs_for_vis = None
        structures = []
        states = []
        for image_batch in imgs:
            structure, state = self.encode(image_batch, training=False)
            structures.append(structure)
            states.append(state)
        structure_sim_mse = [
            sim_loss_fn(structures[0], structures[1]),
            sim_loss_fn(structures[0], structures[2]),
            sim_loss_fn(structures[1], structures[2]),
        ]
        structure_vec_sim_loss = tf.reduce_mean(
            structure_sim_mse
        )  # sum(structure_sim_mse) / 3.0
        predicted_states = self.get_predicted_states(states, days)
        predicted_imgs = [
            self.decode(structure, state)
            for structure, state in zip(structures, predicted_states)
        ]
        if predicted_imgs_for_vis is None:
            predicted_imgs_for_vis = predicted_imgs
        image_similarity_mse = [
            sim_loss_fn(real, pred) for real, pred in zip(imgs, predicted_imgs)
        ]
        image_similarity_loss = tf.reduce_mean(
            image_similarity_mse
        )  # sum(image_similarity_mse) / 3.0
        total_loss = (
            structure_vec_similarity_loss_mult * structure_vec_sim_loss
            + image_similarity_loss
        )
        ssims = self.calculate_ssim(imgs, predicted_imgs)
        return (
            total_loss,
            image_similarity_loss,
            structure_vec_sim_loss,
            ssims,
            predicted_imgs_for_vis,
        )

    def train_using_pair(
        self,
        imgs,
        optimizer,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
    ):
        predicted_imgs_for_vis = None
        with tf.GradientTape() as tape:
            structures = []
            states = []
            for image_batch in imgs:
                structure, state = self.encode(image_batch, training=True)
                structures.append(structure)
                states.append(state)
            structure_sim_mse = [sim_loss_fn(structures[0], structures[1])]
            structure_vec_sim_loss = tf.reduce_mean(structure_sim_mse)
            predicted_imgs = [
                self.decode(structure, state)
                for structure, state in zip(structures, states)
            ]
            if predicted_imgs_for_vis is None:
                predicted_imgs_for_vis = predicted_imgs
            image_similarity_mse = [
                sim_loss_fn(real, pred) for real, pred in zip(imgs, predicted_imgs)
            ]
            image_similarity_loss = tf.reduce_mean(image_similarity_mse)
            total_loss = (
                structure_vec_similarity_loss_mult * structure_vec_sim_loss
                + image_similarity_loss
            )
        grads = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        ssims = self.calculate_ssim(imgs, predicted_imgs)
        return (
            total_loss,
            image_similarity_loss,
            structure_vec_sim_loss,
            ssims,
            predicted_imgs_for_vis,
        )

    @staticmethod
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

    @staticmethod
    def calculate_ssim(imgs, generated_imgs):
        ssims = [
            tf.image.ssim(img1, img2, max_val=1.0)
            for img1, img2 in zip(imgs, generated_imgs)
        ]
        return tf.reduce_mean(ssims)

    def train_using_triplet(
        self,
        imgs,
        days,
        optimizer,
        sim_loss_fn=tf.keras.losses.MeanSquaredError(),
        structure_vec_similarity_loss_mult=100,
    ):
        predicted_imgs_for_vis = None
        with tf.GradientTape() as tape:
            structures = []
            states = []
            for image_batch in imgs:
                structure, state = self.encode(image_batch, training=True)
                structures.append(structure)
                states.append(state)
            structure_sim_mse = [
                sim_loss_fn(structures[0], structures[1]),
                sim_loss_fn(structures[0], structures[2]),
                sim_loss_fn(structures[1], structures[2]),
            ]
            structure_vec_sim_loss = tf.reduce_mean(structure_sim_mse)

            predicted_states = self.get_predicted_states(states, days)
            predicted_imgs = [
                self.decode(structure, state)
                for structure, state in zip(structures, predicted_states)
            ]
            if predicted_imgs_for_vis is None:
                predicted_imgs_for_vis = predicted_imgs
            image_similarity_mse = [
                sim_loss_fn(real, pred) for real, pred in zip(imgs, predicted_imgs)
            ]
            image_similarity_loss = tf.reduce_mean(
                image_similarity_mse
            )  # sum(image_similarity_mse) / 3.0
            total_loss = (
                structure_vec_similarity_loss_mult * structure_vec_sim_loss
                + image_similarity_loss
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        ssims = self.calculate_ssim(imgs, predicted_imgs)

        return (
            total_loss,
            image_similarity_loss,
            structure_vec_sim_loss,
            ssims,
            predicted_imgs_for_vis,
        )

    def restore_model(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
        checkpoint.restore(manager.latest_checkpoint)


if __name__ == "__main__":
    FILTERS = [64, 128, 256, 512]
    KERNEL_SIZE = 3
    ACTIVATION = tf.nn.silu
    LAST_ACTIVATION = tf.nn.sigmoid
    STRUCTURE_VEC_SIZE = 100
    LONGITUDINAL_VEC_SIZE = 1

    model = AE(
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        activation=ACTIVATION,
        last_activation=LAST_ACTIVATION,
        structure_vec_size=STRUCTURE_VEC_SIZE,
        longitudinal_vec_size=LONGITUDINAL_VEC_SIZE,
    )
    print("model defined")
    input_tensor = tf.convert_to_tensor(np.zeros((3, 64, 64, 1)))
    input_tensor = tf.cast(input_tensor, tf.float32)
    _ = model(input_tensor)
    print("model first call")
    structure, state = model.encode(input_tensor)
    print("structure: ", tf.shape(structure))
    print("state: ", tf.shape(state))
    img = model.decode(structure, state)
    print("res: ", tf.shape(img))
    print("done")
