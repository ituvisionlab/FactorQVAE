from encoders.dvae_encoder import DVAEEncoder

class FactorDVAEEncoder(DVAEEncoder):
    def __init__(self, config):
        super(FactorDVAEEncoder, self).__init__(config)