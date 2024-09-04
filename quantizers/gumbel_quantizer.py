import torch
from torch import nn, einsum
from torch.nn import functional as F

from quantizers.base_quantizer import BaseQuantizer
from quantizers.utils import measure_perplexity

class GumbelQuantizer(BaseQuantizer):
    def __init__(self, config):
        super(GumbelQuantizer, self).__init__(config)

        self.temp = config.model.quantizer.temp
    
    def forward(self, logits, iter):
        logits = logits.reshape(-1, self.num_latents, self.num_embeddings).contiguous()
        if self.training:
            # soft one-hot conversion from encoder's logits
            vector_weights = F.gumbel_softmax(logits, tau=self.temp, hard=False, dim=-1)
            indices = vector_weights.argmax(dim=-1)
        else:
            indices = logits.argmax(dim=-1)
            vector_weights = F.one_hot(indices, self.num_embeddings).float()
        
        # use these soft one-hot representation as weights of each vector
        z_q = einsum('b m n, n d -> b m d', vector_weights, self.codebook.weight)

        # convert encoder's logits to probabilities
        posterior = F.softmax(logits, dim=-1)

        # measure kl distance between these probabilities and uniform dist
        kl_loss = torch.sum(posterior * torch.log(posterior * self.num_embeddings + 1e-10), dim=-1).mean()
        
        # measure perplexity
        encodings = F.one_hot(indices, self.num_embeddings).float().reshape(-1, self.num_embeddings)
        perplexity = measure_perplexity(encodings)

        z_q = z_q.reshape(-1, self.num_latents*self.embedding_dim).contiguous()
        indices = indices.reshape(-1, self.num_latents)

        return z_q, kl_loss, vector_weights, perplexity
