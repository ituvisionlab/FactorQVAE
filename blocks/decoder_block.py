import torch.nn as nn

from collections import OrderedDict

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers):
        """
        Decoder block used in original dVAE of Dall-E.

        Args:
            in_channel (int): The number of input channels.
            out_channel (int): The number of output channels.
            n_layers (int): The number of layers.
        """
        super(DecoderBlock, self).__init__()
        self.n_in = in_channel
        self.n_hid = out_channel // 4
        self.n_out = out_channel
        self.n_layers = n_layers
        self.post_gain = 1 / (self.n_layers ** 2)

        self.id_path = nn.Conv2d(self.n_in, self.n_out, kernel_size=1) if self.n_in!= self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', nn.Conv2d(self.n_in,  self.n_hid, 1)),
                ('relu_2', nn.ReLU()),
                ('conv_2', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
                ('relu_3', nn.ReLU()),
                ('conv_3', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
                ('relu_4', nn.ReLU()),
                ('conv_4', nn.Conv2d(self.n_hid, self.n_out, kernel_size=3, padding=1)),]))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)