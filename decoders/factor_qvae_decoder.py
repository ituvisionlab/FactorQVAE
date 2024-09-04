from decoders.qvae_decoder import QVAEDecoder

class FactorQVAEDecoder(QVAEDecoder):
    def __init__(self, config):
        super(FactorQVAEDecoder, self).__init__(config)