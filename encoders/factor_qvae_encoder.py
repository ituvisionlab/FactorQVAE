from encoders.qvae_encoder import QVAEEncoder

class FactorQVAEEncoder(QVAEEncoder):
    def __init__(self, config):
        super(FactorQVAEEncoder, self).__init__(config)