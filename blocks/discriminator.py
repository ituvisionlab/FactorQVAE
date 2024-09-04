import torch
from torch import nn

class Discriminator(nn.Module):
    """
    A discriminator network for total correlation calculation.
    
    Args:
        latent_dim (int): The dimension of the latent representation.
        hid_channel (int): The number of hidden channels.
        flg_bn (bool): Whether to use batch normalization.
        layers (int): The number of layers.
    """
    def __init__(self, latent_dim, hid_channel=1000, flg_bn=False, layers=4):
        super().__init__()

        self.latent_dim = latent_dim
        self.hid_channel = hid_channel
        self.layers = layers

        modules = []
        modules.append(nn.Linear(latent_dim, hid_channel))
        if flg_bn:
            modules.append(nn.BatchNorm1d(hid_channel))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(self.layers):
            modules.append(nn.Linear(hid_channel, hid_channel))
            if flg_bn:
                modules.append(nn.BatchNorm1d(hid_channel))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        
        modules.append(nn.Linear(hid_channel, 2))

        self.discriminator = nn.Sequential(*modules)

    def forward(self, x):
        return self.discriminator(x)



