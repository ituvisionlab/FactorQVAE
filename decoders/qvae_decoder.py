import torch.nn as nn

from decoders.base_decoder import BaseDecoder

class QVAEDecoder(BaseDecoder):
    def __init__(self, config):
        super(QVAEDecoder, self).__init__(config)

        self.embedding_dim = config.model.codebook.embedding_dim
        self.input_size = (self.width**2)*self.input_channel

        self.dense = nn.Linear(self.embedding_dim*self.num_latents, self.input_size)

    def forward(self, x):
        x = self.dense(x).view(-1, self.input_channel, self.width, self.width)        
        return self.decoder(x)