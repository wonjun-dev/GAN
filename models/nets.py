import numpy as np
import torch.nn as nn
from main import img_dims
from main import opt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define layers
        def _block(in_dim, out_dim, normalize=True):
            block = [nn.Linear(in_dim, out_dim)]
            if normalize:
                block.append(nn.BatchNorm1d(out_dim))
            block.append(nn.LeakyReLU())
            return block

        self.model = nn.Sequential(
            *_block(opt.latent_dim, 128, False),
            *_block(128, 256),
            *_block(256, 512),
            *_block(512, 1024),
            nn.Linear(1024, np.prod(img_dims)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.model(z)
        img = x.view(x.size(0), *img_dims)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define layers
        pass

    def forward(self, x):
        # Define forward pass
        pass
