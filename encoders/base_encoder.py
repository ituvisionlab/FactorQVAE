import torch.nn as nn

from collections import OrderedDict

from blocks.encoder_block import EncoderBlock

class BaseEncoder(nn.Module):
    def __init__(self, config):
        """
        Encoder architecture used in dVAE of Dall-E.
        """
        super(BaseEncoder, self).__init__()

        self.group_count = config.model.group_count
        self.n_blk_per_group = config.model.n_blk_per_group
        self.n_layers = self.group_count * self.n_blk_per_group
        self.channel = config.model.channel
        self.output_size = ((config.data.img_size//(2**(self.group_count-1)))**2)*(2**(self.group_count-1)*self.channel) 
        self.num_latents = config.data.num_latents
        self.hidden_dim = config.model.hidden_dim # equals to num_embeddings for dvae/factordvae, specified otherwise 

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=3, padding=1)),            
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(self.channel, self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),

            *[(f'group_{j + 2}', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(2**(j)*self.channel if i == 0 else 2**(j+1)*self.channel, 2**(j+1)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))) for j in range(self.group_count - 2)],

            ('group_last', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EncoderBlock(2**(self.group_count-2)*self.channel if i == 0 else 2**(self.group_count-1)*self.channel, 2**(self.group_count-1)*self.channel, self.n_layers)) for i in range(self.n_blk_per_group)],
            ]))),     
        ]))

        self.dense = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.output_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU())

    def forward(self, x):
        raise NotImplementedError()
