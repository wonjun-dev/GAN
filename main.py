import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as transforms
from torchvision.utils import save_image
from models.nets import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--img_size", type=int, default=28, help="dimension of image row & column")
opt = parser.parse_args()
print("Runnint Options: \n", opt)


def train():
    pass


if __name__ == "__main__":
    ## Define model & loss function

    ## Define dataloader
    data_root = "./data/mnist"
    os.makedirs(data_root, exist_ok=True)
    data_transforms = transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_loader = DataLoader(
        dataset=MNIST(root=data_root, train=True, download=True, transform=data_transforms),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=MNIST(root=data_root, train=False, download=True, transform=data_transforms),
        batch_size=opt.batch_size,
        shuffle=False,
    )

    # Define optimizer

    train()
