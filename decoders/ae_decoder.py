import torch.nn as nn

from decoders.base_decoder import BaseDecoder

class AEDecoder(BaseDecoder):
    def __init__(self, config):
        super(AEDecoder, self).__init__(config)

        self.input_size = (self.width**2)*self.input_channel

        self.dense = nn.Linear(self.num_latents, self.input_size)

    def forward(self, x):
        x = self.dense(x).view(-1, self.input_channel, self.width, self.width)        
        return self.decoder(x)
