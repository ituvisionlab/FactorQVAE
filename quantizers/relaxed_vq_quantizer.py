import torch
import torch.nn.functional as F

from quantizers.base_quantizer import BaseQuantizer
from quantizers.utils import calculate_distance, measure_perplexity

class RelaxedVQQuantizer(BaseQuantizer):

    def __init__(self, config):
        super(RelaxedVQQuantizer, self).__init__(config)

        self.temp = config.model.quantizer.temp
        self.kl_weight = config.model.quantizer.kl_weight
    
    def forward(self, z, iter):
        z = z.reshape(-1, self.num_latents, self.embedding_dim).contiguous()

        # find min distances and indices
        distances = calculate_distance(z, self.codebook.weight)

        if self.training:
            vector_weights = F.gumbel_softmax(-distances, tau=self.temp, hard=False, dim=-1) # soft-one-hot in training
            indices = vector_weights.argmax(dim=-1)
        else:
            indices = distances.argmin(dim=-1).unsqueeze(1)
            vector_weights = torch.zeros(indices.shape[0], self.num_embeddings, device="cuda")
            vector_weights.scatter_(1, indices, 1)

        z_q = torch.mm(vector_weights, self.codebook.weight).reshape(-1, self.num_latents*self.embedding_dim).contiguous()

        posterior = F.softmax(-distances, dim=-1)

        # measure kl distance between these probabilities and uniform dist
        kl_loss = self.kl_weight * torch.sum(posterior * torch.log(posterior * self.num_embeddings + 1e-10), dim=-1).mean()

        encodings = F.one_hot(indices, self.num_embeddings).float().reshape(-1, self.num_embeddings)
        perplexity = measure_perplexity(encodings)

        vector_weights = vector_weights.reshape(-1, self.num_latents, self.num_embeddings).contiguous()

        return z_q, kl_loss, vector_weights, perplexity