from torch import nn

from encoders.base_encoder import BaseEncoder

class QVAEEncoder(BaseEncoder):
    def __init__(self, config):
        super(QVAEEncoder, self).__init__(config)
        
        self.embedding_dim = config.model.codebook.embedding_dim
        self.proj = nn.Linear(self.hidden_dim, self.embedding_dim*self.num_latents)

    def forward(self, x):
        x = self.encoder(x).reshape(-1, self.output_size)
        x = self.dense(x)
        return self.proj(x)