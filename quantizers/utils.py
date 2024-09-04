import torch
import torch.nn.functional as F

def calculate_distance(z, codebook):
    bs, num_latents, embedding_dim = z.shape
    z_flattened = z.view(-1, embedding_dim)

    # find distances between the embeddings and the codebook
    d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
        torch.sum(codebook**2, dim=1) - 2 * \
        torch.matmul(z_flattened, codebook.t())

    return d

def find_min_indices(d):
    # find closest indices
    min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

    return min_encoding_indices

def measure_perplexity(encodings):
    # measure perplexity
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()

    return perplexity
