from decoders.vae_decoder import VAEDecoder

class FactorVAEDecoder(VAEDecoder):
    def __init__(self, config):
        super(FactorVAEDecoder, self).__init__(config)