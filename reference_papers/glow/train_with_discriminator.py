import glob
import os

from torch.nn import BCELoss
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from datasets.torch_dataset import get_triplets_adni_15t_dataset_torch
from reference_papers.glow.discriminator import Discriminator
from reference_papers.glow.model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument(
    "--save_interval", default=1000, type=int, help="Save interval for checkpoints"
)
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument(
    "save_dir", metavar="PATH", type=str, help="Path to model save directory"
)
parser.add_argument(
    "--loss_weight", default=10.0, type=float, help="weight of the pair loss"
)


def sample_data(path, batch_size, image_size):
    dataset, _, _ = get_triplets_adni_15t_dataset_torch(
        folder_name="training_data_15T_192x160_4slices",
        machine="colab",
        target_shape=[64, 64, 3],
    )

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


mse_loss = torch.nn.MSELoss()


def calc_loss_pair(z1_list, z2_list):
    loss = 0
    for z1, z2 in zip(z1_list[:-1], z2_list[:-1]):
        loss += mse_loss(z1, z2)
    return loss


def weighted_z(z1_list, z2_list, w1, w2):
    z_w = []
    w1 = w1.view(-1, 1, 1, 1)
    w2 = w2.view(-1, 1, 1, 1)
    for z1, z2 in zip(z1_list, z2_list):
        print("z1: ", z1.shape)
        print("z2: ", z2.shape)
        print("w1: ", w1)
        print("w2: ", w2)
        z_w.append(z1 * w1 + z2 * w2)
    return z_w


def blend_vectors(z_vecs, t):

    z0 = weighted_z(
        z_vecs[1],
        z_vecs[2],
        (t[2] - t[0]) / (t[2] - t[1]),
        (t[0] - t[1]) / (t[2] - t[1]),
    )
    z1 = weighted_z(
        z_vecs[0],
        z_vecs[2],
        (t[2] - t[1]) / (t[2] - t[0]),
        (t[1] - t[0]) / (t[2] - t[0]),
    )
    z2 = weighted_z(
        z_vecs[0],
        z_vecs[1],
        (t[1] - t[2]) / (t[1] - t[0]),
        (t[2] - t[0]) / (t[1] - t[0]),
    )

    return [z0, z1, z2]


def generate_predictions(model, imgs, days):

    z_vectors = []
    for img in imgs:
        log_p, logdet, z = model(img)
        z_vectors.append(z)
    blended_z_vectors = blend_vectors(z_vectors, days)
    predictions = [model.module.reverse(x, reconstruct=True) for x in blended_z_vectors]

    return predictions


def train(
    args,
    model,
    optimizer,
    discriminator,
    discriminator_optimizer,
    initial_iter=0,
    restore=None,
    loaded_state_dict_model=None,
):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    real_label = 1.0
    fake_label = 0.0
    criterion = BCELoss()

    with tqdm(range(initial_iter, args.iter)) as pbar:
        for i in pbar:

            batch = next(dataset)
            imgs = [batch["img1"], batch["img2"], batch["img3"]]
            days = batch["days"]
            # print("days : ", days)
            days = [x.to(device) for x in days]
            imgs = [x.to(device) for x in imgs]

            imgs = [x * 255 for x in imgs]

            if args.n_bits < 8:
                imgs = [torch.floor(image / 2 ** (8 - args.n_bits)) for image in imgs]

            imgs = [image / n_bins - 0.5 for image in imgs]

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        imgs[0] + torch.rand_like(imgs[0]) / n_bins
                    )
                if restore:
                    print(
                        "RESTORE is true, loading state dict of model again due to the first initialization probably broke things"
                    )
                    model.load_state_dict(loaded_state_dict_model)
                continue

            predicted_imgs = generate_predictions(model, imgs, days)

            # train discriminator
            discriminator.zero_grad()
            b_size = imgs[0].shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            D_x = []
            D_G = []
            errs = []
            for img in imgs:
                output = discriminator(img).view(-1)
                err_real = criterion(output, label)
                err_real.backward()
                D_x.append(output.mean().item())
                errs.append(err_real)
            label.fill_(fake_label)
            for pred in predicted_imgs:
                output = discriminator(pred.detach()).view(-1)
                err_fake = criterion(output, label)
                err_fake.backward()
                D_G.append(output.mean().item())
                errs.append(err_fake)
            err = sum(errs) / 6.0
            discriminator_optimizer.step()

            # logdet = (logdet1.mean() + logdet2.mean()) / 2
            # log_p = torch.cat([log_p1, log_p2])
            # loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            # pair_loss = calc_loss_pair(z1, z2)
            # pair_loss *= args.loss_weight
            # loss += pair_loss
            # model.zero_grad()
            # loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            # warmup_lr = args.lr
            # optimizer.param_groups[0]["lr"] = warmup_lr
            # optimizer.step()

            # pbar.set_description(
            #     f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; pair_loss: {pair_loss.item()}; lr: {warmup_lr:.7f}"
            # )
            pbar.set_description(f"Disc loss: {err.item()}")

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        os.path.join(
                            args.save_dir, f"sample/{str(i + 1).zfill(6)}.png"
                        ),
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % args.save_interval == 0:
                # if i % 2 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir, f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                    ),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        args.save_dir, f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                    ),
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"checkpoint/discriminator_{str(i + 1).zfill(6)}.pt",
                    ),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        args.save_dir, f"checkpoint/disc_optim_{str(i + 1).zfill(6)}.pt"
                    ),
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        print("making dirs")
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, "checkpoint"))
        os.makedirs(os.path.join(args.save_dir, "sample"))

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # load trained model to resume training
    model_paths = sorted(
        glob.glob(os.path.join(args.save_dir, "checkpoint/model_*.pt"))
    )
    restore = None
    loaded_state_dict_model = None
    if model_paths:
        restore = True
        model_path = model_paths[-1]
        optim_path = sorted(
            glob.glob(os.path.join(args.save_dir, "checkpoint/optim_*.pt"))
        )[-1]
        loaded_state_dict_model = torch.load(model_path)
        model.load_state_dict(loaded_state_dict_model)
        loaded_state_dict = torch.load(optim_path)
        optimizer.load_state_dict(loaded_state_dict)
        initial_iter = int(model_path[-9:-3])
        print(
            f"loaded trained model from {model_path} to resume training from iter {initial_iter}"
        )
    else:
        initial_iter = 0

    discriminator = Discriminator(nc=3, ndf=64)
    discriminator.to(device)

    discriminator_model_paths = sorted(
        glob.glob(os.path.join(args.save_dir, "checkpoint/discriminator_*.pt"))
    )
    d_lr = 0.0002
    beta1 = 0.5
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=d_lr, betas=(beta1, 0.999)
    )

    if discriminator_model_paths:
        discriminator_model_path = discriminator_model_paths[-1]
        optim_path = sorted(
            glob.glob(os.path.join(args.save_dir, "checkpoint/disc_optim_*.pt"))
        )[-1]
        loaded_state_dict_disc = torch.load(discriminator_model_path)
        discriminator.load_state_dict(loaded_state_dict_disc)
        loaded_state_dict_discoptim = torch.load(optim_path)
        discriminator_optimizer.load_state_dict(loaded_state_dict_discoptim)
        print(
            f"Resroted discriminator and its optimizer from {discriminator_model_path} and {optim_path}"
        )

    train(
        args,
        model,
        optimizer,
        discriminator,
        discriminator_optimizer,
        initial_iter=initial_iter,
        restore=restore,
        loaded_state_dict_model=loaded_state_dict_model,
    )
    model()
