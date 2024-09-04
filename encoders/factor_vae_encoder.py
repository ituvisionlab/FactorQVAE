from encoders.vae_encoder import VAEEncoder

class FactorVAEEncoder(VAEEncoder):
    def __init__(self, config):
        super(FactorVAEEncoder, self).__init__(config)