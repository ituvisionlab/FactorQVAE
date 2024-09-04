import numpy as np
import torch
from omegaconf import OmegaConf

def generate_batch_factor_code(data, model, representation_function, num_points, random_state, batch_size):
    """Generate batched observation and factor paired data."""
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = sample(data, num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(model, current_observations.cuda()).detach().cpu()
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                        representation_function(
                                            model,
                                            current_observations.cuda()).detach().cpu()))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)

def sample(data, num, random_state):
    """Sample a batch of factors and observations."""
    factors = sample_latent_factors(data, num, random_state)
    return factors, sample_observations_from_factors(data, factors, random_state)

def sample_latent_factors(data, num, random_state):
    """Sample a batch of the latent factors."""
    factors = np.zeros(
        shape=(num, len(data.latent_factor_indices)), dtype=np.int64)
    for pos, i in enumerate(data.latent_factor_indices):
        factors[:, pos] = _sample_factor(data, i, num, random_state)
    return factors

def _sample_factor(data, i, num, random_state):
    return random_state.randint(data.factor_sizes[i], size=num)

def sample_observations_from_factors(data, factors, random_state):
    """Sample observations based the given factors."""
    all_factors = sample_all_factors(data, factors, random_state)
    indices = np.array(np.dot(all_factors, data.factor_bases), dtype=np.int64)
    images = np.zeros(shape=(indices.shape[0], 3, data.img_size, data.img_size), dtype=np.float32)
    for b, index in enumerate(indices):
        image = data.__getitem__(index)
        images[b] = image
    batch = torch.from_numpy(images)

    return batch

def sample_all_factors(data, latent_factors, random_state):
    """Samples the remaining factors based on the latent factors."""
    num_samples = latent_factors.shape[0]
    all_factors = np.zeros(
        shape=(num_samples, data.num_total_factors), dtype=np.int64)
    all_factors[:, data.latent_factor_indices] = latent_factors
    # Complete all the other factors
    for i in data.observation_factor_indices:
        all_factors[:, i] = _sample_factor(data, i, num_samples, random_state)
    return all_factors

def quantizer_representation_function(model, x):
    """Obtain latent representation."""
    if model.config.model.name in ["AE", "BioAE"]:
        z = model.encoder(x)
    elif model.config.model.name in ["VAE", "FactorVAE"]:
        logits = model.encoder(x)
        mu = logits[:, :model.config.data.num_latents]
        logvar = logits[:, model.config.data.num_latents:]
        z = model.reparameterize(mu, logvar)
    elif model.config.model.name == "QLAE":
        logits = model.encoder(x)
        _, _, min_encoding_indices = model.quantizer(logits, 0)
        z = min_encoding_indices
    elif model.config.model.name in ["DVAE", "FactorDVAE", "QVAE", "FactorQVAE"]:
        logits = model.encoder(x)
        _, _, indices, _ = model.quantizer(logits, 0)
        z = indices.argmax(dim=-1)   
         
    return z

def load_config(config_path):
    """Load the config by the given path."""
    config = OmegaConf.load(config_path)

    return config