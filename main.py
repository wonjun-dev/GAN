import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms as transforms
from torchvision.utils import save_image

from models.nets import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of sepochs")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mnist", "cifar10"],
    default="mnist",
    help="dataset for training",
)
parser.add_argument("--latent_dim", type=int, default=64, help="dimension of noise z")
opt = parser.parse_args()
print("Running Options: \n", opt)


def train():

    for epoch in tqdm(range(opt.n_epochs)):
        print(epoch)


if __name__ == "__main__":
    ## Define dataset & dataloader
    dataset_name = opt.dataset
    data_root = os.path.join("./data", dataset_name)
    os.makedirs(data_root, exist_ok=True)

    if dataset_name == "mnist":
        img_dims = [1, 28, 28]
        mean = 0.5
        std = 0.5
    else:
        img_dims = [3, 32, 32]
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(img_dims[-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset_name == "mnist":
        train_dataset = MNIST(root=data_root, train=True, download=True, transform=data_transforms)
        valid_dataset = MNIST(root=data_root, train=False, download=True, transform=data_transforms)
    else:
        train_dataset = CIFAR10(
            root=data_root, train=True, download=True, transform=data_transforms
        )
        valid_dataset = CIFAR10(
            root=data_root, train=False, download=True, transform=data_transforms
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
    )

    # Define model
    generator = Generator(opt.latent_dim, img_dims)
    discriminator = Discriminator(img_dims)

    # Define loss & optimizer
    loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    train()
