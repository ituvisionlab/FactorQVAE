from torch import nn

from encoders.base_encoder import BaseEncoder

class DVAEEncoder(BaseEncoder):
    def __init__(self, config):
        super(DVAEEncoder, self).__init__(config)
        
        self.num_embeddings = config.model.codebook.num_embeddings
        self.proj = nn.Linear(self.hidden_dim, self.num_embeddings*self.num_latents)

    def forward(self, x):
        x = self.encoder(x).reshape(-1, self.output_size)
        x = self.dense(x)
        return self.proj(x)
