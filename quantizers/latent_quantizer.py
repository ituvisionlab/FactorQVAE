import torch
from torch import nn
import torch.nn.functional as F

class LatentQuantizer(nn.Module):
    def __init__(self, config):
        super(LatentQuantizer, self).__init__()

        self.num_embeddings = config.model.codebook.num_embeddings
        self.embedding_dim = config.model.codebook.embedding_dim

        self.num_latents = config.data.num_latents
        self.codebook = nn.Parameter(torch.randn(self.num_latents, self.num_embeddings, requires_grad=True))

        self.commit_coeff = config.model.quantizer.commit_coeff
        self.quantize_coeff = config.model.quantizer.quantize_coeff

    def quantize(self, z_dim, codebook):
        """
        z is the latent's variable, codebook is the variable's corresponding codebook
        """
        distances = torch.abs(z_dim - codebook)
        index = distances.argmin()

        return codebook[index], index

    def forward(self, z_batch, iter):
        z_q = []
        min_encoding_indices = []
        for z in z_batch:
            quantized_and_indices = [self.quantize(z_dim, codebook_dim) for z_dim, codebook_dim in zip(z, self.codebook)]
            z_q.append(torch.stack([x[0] for x in quantized_and_indices], dim=0))
            min_encoding_indices.append(torch.stack([x[1] for x in quantized_and_indices], dim=0))
        z_q = torch.stack(z_q, dim=0)
        min_encoding_indices = torch.stack(min_encoding_indices, dim=0)

        # loss for codebook
        commitment_loss = self.commit_coeff * torch.mean((z_q.detach()-z_batch)**2) + \
            self.quantize_coeff * torch.mean((z_q - z_batch.detach()) ** 2)

        z_q = z_batch + (z_q - z_batch).detach()

        return z_q, commitment_loss, min_encoding_indices
