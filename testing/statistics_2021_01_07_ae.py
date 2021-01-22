import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class CSVHandler:
    def __init__(self, csv_path, columns, read_existing=True):
        self.csv_path = csv_path
        self.columns = columns
        self.rows = None
        self.queue = []
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

                self.rows_dict_list = [
                    {c: el for el, c in zip(row, columns)} for row in self.rows
                ]


class Statistics:
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
            read_existing=True,
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
            read_existing=True,
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
            read_existing=True,
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
            read_existing=True,
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
            read_existing=True,
        )


def x():
    # example data
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    # example error bar values that vary with x-position
    error = 0.1 + 0.2 * x
    # error bar values w/ different -/+ errors
    lower_error = 0.4 * error
    upper_error = error
    asymmetric_error = [lower_error, upper_error]

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.errorbar(x, y, yerr=error, fmt="-o")
    ax0.set_title("variable, symmetric error")

    ax1.errorbar(x, y, xerr=asymmetric_error, fmt="o")
    ax1.set_title("variable, asymmetric error")
    ax1.set_yscale("log")
    plt.show()


def plot(dicts, title=""):
    fig, (ax0) = plt.subplots(nrows=1, sharex=True)
    for d in dicts:
        x = d.keys()
        y = [d[key][0] for key in d]
        y_err = [d[key][1] for key in d]
        ax0.errorbar(x, y, yerr=y_err, fmt="-o", capthick=10)
        ax0.set_title(title)
        ax0.set_xlabel("Train steps")
        ax0.set_ylabel("SSIM")
        ax0.grid = "on"
    plt.show()


if __name__ == "__main__":

    EXPERIMENT_NAME = "exp_2021_01_07_ae"
    CHECKPOINT_DIR_NAME = "checkpoints"
    NUM_SUBFOLDERS = 4

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

    results_folders = [
        os.path.join(
            EXPERIMENT_FOLDER,
            f"testing/sequences/reduced_trainset_25/test_train_for_patient2_{i}",
        )
        for i in range(NUM_SUBFOLDERS)
    ]
    results_folder = os.path.dirname(results_folders[0])

    statistics = [Statistics(save_dir=dir_path) for dir_path in results_folders]
    interpolation_statistics = []
    for statistic in statistics:
        interpolation_statistics += (
            statistic.interpolation_ssim_csv_handler.rows_dict_list
        )
    val_loss_statistics = []
    for statistic in statistics:
        val_loss_statistics += statistic.val_loss_csv_handler.rows_dict_list
    training_statistics = []
    for statistic in statistics:
        training_statistics += statistic.train_and_pair_loss_csv_handler.rows_dict_list

    def find_image(image_name, folders):
        image_path = None
        for folder in folders:
            if os.path.isfile(os.path.join(folder, image_name)):
                image_path = os.path.join(folder, image_name)
        return image_path

    for d in interpolation_statistics:
        image_name = f"sample_{int(d['sample_id']):05d}_{str(d['identifier'])}_step_{int(d['train_step']):05d}.jpg"
        image_path = find_image(image_name, results_folders)
        d["image"] = image_path

    print(interpolation_statistics[0])
    print(interpolation_statistics[1])
    print(interpolation_statistics[42033])
    print("")
    print(val_loss_statistics[0])
    print(val_loss_statistics[1])
    print(val_loss_statistics[1233])

    print("")
    print(training_statistics[0])
    print(training_statistics[1])
    print(training_statistics[1233])
    print("")

    # print interpolation ssin vs train steps
    interpolation_ssims_vs_train_steps = {}
    for d in interpolation_statistics:
        if int(d["train_step"]) in interpolation_ssims_vs_train_steps:
            interpolation_ssims_vs_train_steps[int(d["train_step"])].append(
                float(d["ssim"])
            )
        else:
            interpolation_ssims_vs_train_steps[int(d["train_step"])] = [
                float(d["ssim"])
            ]
    print(interpolation_ssims_vs_train_steps)
    for k in interpolation_ssims_vs_train_steps:
        ssims = interpolation_ssims_vs_train_steps[k]
        m, std, lolim, uplim = (
            np.mean(ssims),
            np.std(ssims),
            np.min(ssims),
            np.max(ssims),
        )
        interpolation_ssims_vs_train_steps[k] = [m, std, lolim, uplim]
    print(interpolation_ssims_vs_train_steps)
    x = interpolation_ssims_vs_train_steps.keys()
    y = [
        interpolation_ssims_vs_train_steps[key][0]
        for key in interpolation_ssims_vs_train_steps
    ]
    y_err = [
        [
            interpolation_ssims_vs_train_steps[key][2],
            interpolation_ssims_vs_train_steps[key][3],
        ]
        for key in interpolation_ssims_vs_train_steps
    ]
    y_err = [[el - err[0], err[1] - el] for err, el in zip(y_err, y)]
    y_err = np.asarray(y_err).transpose()
    std = [
        interpolation_ssims_vs_train_steps[key][1]
        for key in interpolation_ssims_vs_train_steps
    ]
    y_std_lower = [y_el - s for y_el, s in zip(y, std)]
    y_std_upper = [y_el + s for y_el, s in zip(y, std)]

    fig, (ax0) = plt.subplots(nrows=1)
    ax0.errorbar(x, y, yerr=y_err, fmt="ob", linestyle="dotted")
    ax0.scatter(x, y_std_lower, c="b", marker="_")
    ax0.scatter(x, y_std_upper, c="b", marker="_")
    ax0.set_title("Interpolation SSIM vs Finetune Steps")
    ax0.set_xlabel("Train steps")
    ax0.set_ylabel("SSIM")
    ax0.grid()
    plt.savefig(os.path.join(results_folder, "ssim_vs_finetuning.png"), dpi=300)
    print("plot saved")
    # plt.show()

    sample_id = None
    score = 1.0
    for d in interpolation_statistics:
        if int(d["train_step"]) == 40:
            if float(d["ssim"]) < score:
                score = float(d["ssim"])
                sample_id = d["sample_id"]
    print("sample with least ssim: ", sample_id)

    true_sequences = {}
    for d in interpolation_statistics:
        sample_id = d["sample_id"]
        image_name = f"sample_{int(sample_id):05d}_true.jpg"
        image_path = find_image(image_name, results_folders)
        true_sequences[sample_id] = image_path

    def make_collage(sample_id, identifier, interpolation_statistics, true_image_path):
        true_image = cv2.imread(true_image_path)
        predictions = {}
        for d in interpolation_statistics:
            if d["sample_id"] == sample_id and d["identifier"] == identifier:
                predictions[int(d["train_step"])] = d["image"]
        sorted_train_steps = sorted(predictions.keys())
        predictions = [
            cv2.imread(predictions[key]) for key in sorted(predictions.keys())
        ]
        true_padded = np.zeros_like(predictions[0])
        if true_image.ndim == 3:
            true_padded[:, : true_image.shape[1], :] = true_image
        else:
            true_padded[:, : true_image.shape[1]] = true_image
        predictions = np.vstack(predictions)
        collage = np.vstack([true_padded, predictions])
        edge = 5
        prefix = collage[:, : 100 + edge] * 0
        cv2.putText(
            prefix, "Ground", (10, 32), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)
        )
        cv2.putText(
            prefix, "truth", (10, 46), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)
        )
        for n, step in enumerate(sorted_train_steps):
            if n == 0:
                cv2.putText(
                    prefix,
                    "Finetuning",
                    (10, (n + 1) * 64 + 18),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                )
            cv2.putText(
                prefix,
                f"Step {step}",
                (10, (n + 1) * 64 + 32),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
            )
        suffix = collage[:, :edge] * 0
        collage = np.hstack([prefix, collage, suffix])
        suffix = collage[:edge, :] * 0
        collage = np.vstack([suffix, collage, suffix])
        return collage

    class ImageSaver:
        def __init__(self, dir_path="/Users/umutkucukaslan/Desktop/imgseq_reduced_25"):
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            self.dir_path = dir_path
            self.counter = 0

        def save_image(self, image):
            filename = f"image_{self.counter}.jpg"
            self.counter += 1
            path = os.path.join(self.dir_path, filename)
            cv2.imwrite(path, image)

    image_saver = ImageSaver()
    sample_id = 0
    identifier = "f"
    pressed_key = 0
    while pressed_key != ord("q"):
        img = make_collage(
            sample_id=str(sample_id),
            identifier=identifier,
            interpolation_statistics=interpolation_statistics,
            true_image_path=true_sequences[str(sample_id)],
        )

        cv2.imshow("collage", img)
        pressed_key = cv2.waitKey()
        if pressed_key == ord("n"):
            sample_id = min(1600, sample_id + 1)
        elif pressed_key == ord("b"):
            sample_id = max(0, sample_id - 1)
        elif pressed_key == ord("f"):
            identifier = "f"
        elif pressed_key == ord("m"):
            identifier = "m"
        elif pressed_key == ord("p"):
            identifier = "p"
        elif pressed_key == ord("s"):
            image_saver.save_image(img)
