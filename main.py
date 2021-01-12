import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from models.nets import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("n_epochs", default=100, help="number of epochs")
parser.add_argument("batch_size", default=64)
opt = parser.parse_args()


def train():
    pass


if __name__ == "__main__":
    ## Define model & loss function

    ## Define dataloader

    # Define optimizer

    train()
