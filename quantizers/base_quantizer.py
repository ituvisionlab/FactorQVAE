from torch import nn

class BaseQuantizer(nn.Module):
    def __init__(self, config):
        super(BaseQuantizer, self).__init__()

        self.num_embeddings = config.model.codebook.num_embeddings
        self.embedding_dim = config.model.codebook.embedding_dim

        self.num_latents = config.data.num_latents
        
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if config.model.codebook.init_type == "uniform": # otherwise, it is initialized with normal distribution
            self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 
                                                1.0 / self.num_embeddings)
    def forward(self, z, iter):
        raise NotImplementedError()
